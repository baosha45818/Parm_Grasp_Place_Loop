#!/usr/bin/env python3
"""Directly grasp a configured workpiece at a semantic-frame XYZ point.

The arm starts from the taught grasp-ready posture.  With the gripper open,
it plans one trajectory that moves to the negative-X extension of the
workpiece line and finishes with its X/Y axes parallel to the workpiece
semantic X/Y axes.  It then advances through the grasp point,
closes the gripper, and lifts the confirmed grasp vertically.
The workpiece-specific geometry, TCP calibration and gripper confirmation
range are loaded from ``--config``.  The configuration is the sole source of
those parameters; this file contains the common motion implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

PUSH_T_DIR = Path(__file__).resolve().parent
if str(PUSH_T_DIR) not in sys.path:
    sys.path.insert(0, str(PUSH_T_DIR))

from move_by_eef_pose import compose_pose, inverse_pose, pose_from_lookup_transform, rotate_vector
from move_by_joints import (
    ensure_ready, finite, import_sdk, move_l as servo_move_l,
    rotvec_to_quat, rotvec_to_rpy, rpy_to_rotvec,
)
from planning_log import planner_start_pose, planning_event


DEFAULT_LINE_ENTRY_CLEARANCE_M = 0.020
DEFAULT_LINE_MAX_REAR_DISTANCE_M = 0.060
RIGHT_EEF_FROM_TRANSFER_BASE = [0.274, 0.0, -0.05075, 0.0, 0.0, 0.0]
DEFAULT_Q1_XY_TOLERANCE_M = 0.0005
DEFAULT_Q1_Z_TOLERANCE_M = 0.0005
DEFAULT_Q1_PITCH_TOLERANCE_RAD = math.radians(0.1)
DEFAULT_Q1_POST_GRASP_LIFT_M = (0.050, 0.040, 0.030, 0.020, 0.010)
RRTCONNECT_TIMEOUT_S = 30.0
Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE = 20


class TrajectoryPlanningError(RuntimeError):
    """A trajectory cannot be planned."""


class GraspNotConfirmedError(RuntimeError):
    """The gripper closed, but its terminal opening did not confirm a grasp."""


class Q1DirectPathUnavailableError(RuntimeError):
    """No direct Q1 approach candidate can be planned."""


CONFIG_SCHEMA = "direct-grasp-workpiece-config/v2"
CONFIGURABLE_OPTIONS = frozenset({
    "--grasp-point-semantic-offset-m",
    "--tcp-local-offset-at-grasp-m",
    "--grasp-retry-z-step-m",
    "--tcp-yaw-toward-workpiece-y-deg",
    "--planner",
    "--final-approach-mode",
    "--servo-linear-speed-mps",
    "--servo-angular-speed-radps",
    "--servo-max-linear-accel-mps2",
    "--servo-max-angular-accel-radps2",
    "--servo-hz",
    "--servo-position-tolerance-m",
    "--servo-orientation-tolerance-rad",
    "--servo-settle-timeout-s",
    "--line-entry-clearance-m",
    "--line-max-rear-distance-m",
    "--line-scan-count",
    "--max-vel",
    "--max-acc",
    "--start-sync-retries",
    "--gripper-timeout-s",
    "--gripper-position-tolerance",
    "--grasp-position-min",
    "--grasp-position-max",
    "--q1-post-grasp-lift-m",
    "--q1-xy-tolerance-m",
    "--q1-z-tolerance-m",
    "--q1-pitch-tolerance-rad",
    "--z-pitch-correction-attempts",
})


def load_grasp_config(path: Path) -> dict[str, object]:
    """Read a workpiece calibration JSON used to populate parser options."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取抓取配置 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("抓取配置的顶层必须是 JSON 对象")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"配置 schema 必须是 {CONFIG_SCHEMA!r}")
    if not isinstance(payload.get("direct_grasp_arguments"), dict):
        raise ValueError("配置缺少对象 direct_grasp_arguments")
    return payload


def apply_grasp_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Set recognised argparse options from the selected JSON calibration."""
    assert isinstance(args.config, Path)
    payload = load_grasp_config(args.config)
    configured = payload["direct_grasp_arguments"]
    assert isinstance(configured, dict)
    by_option = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }
    missing = sorted(CONFIGURABLE_OPTIONS - configured.keys())
    if missing:
        raise ValueError(f"配置缺少抓取参数: {', '.join(missing)}")
    for option, raw_value in configured.items():
        if not isinstance(option, str) or option not in CONFIGURABLE_OPTIONS or option not in by_option:
            raise ValueError(f"配置包含未知抓取参数: {option!r}")
        action = by_option[option]
        try:
            if action.nargs in ("+", "*") or isinstance(action.nargs, int):
                if not isinstance(raw_value, list) or not raw_value:
                    raise ValueError("必须是非空数组")
                if isinstance(action.nargs, int) and len(raw_value) != action.nargs:
                    raise ValueError(f"必须包含 {action.nargs} 项")
                value = [action.type(str(item)) if action.type else item for item in raw_value]
            else:
                value = action.type(str(raw_value)) if action.type else raw_value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"配置参数 {option} 的值无效: {raw_value!r} ({exc})") from exc
        if action.choices is not None:
            values = value if isinstance(value, list) else [value]
            if any(item not in action.choices for item in values):
                raise ValueError(f"配置参数 {option} 不在允许范围内: {value!r}")
        setattr(args, action.dest, value)
    workpiece = payload.get("workpiece", args.config.stem)
    if not isinstance(workpiece, str) or not workpiece.strip():
        raise ValueError("配置参数 workpiece 必须是非空字符串")
    args.workpiece_name = workpiece
    print(f"加载抓取配置: {workpiece} ({args.config})")


def matrix_to_rpy_zyx(rotation: list[float]) -> list[float]:
    """Convert a row-major 3x3 rotation matrix to intrinsic ZYX RPY."""
    pitch = math.asin(max(-1.0, min(1.0, -rotation[6])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[7], rotation[8])
        yaw = math.atan2(rotation[3], rotation[0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[1], rotation[4])
    return [roll, pitch, yaw]


def load_workpiece_grasp_point(path: Path, index: int, semantic_offset_m: list[float]) -> tuple[list[float], list[float]]:
    """Transform a configured semantic-frame XYZ offset into body_base.

    The pose estimator writes ``T_body_base_workpiece_semantic``.  Its first
    three columns are the workpiece semantic axes and its translation is the
    semantic origin.  All three configured offsets therefore rotate with the
    workpiece when it is tilted.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"工件位姿 JSON 无法读取: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"工件位姿 JSON 无法解析: {path}: {exc}") from exc
    if payload.get("coordinate_frame") not in (None, "body_base"):
        raise ValueError(f"工件 JSON 坐标系不是 body_base: {payload.get('coordinate_frame')!r}")
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list) or not 0 <= index < len(workpieces):
        count = len(workpieces) if isinstance(workpieces, list) else 0
        raise ValueError(f"--workpiece-index={index} 超出范围；文件中共有 {count} 个工件")
    entry = workpieces[index]
    if not isinstance(entry, dict):
        raise ValueError(f"workpieces[{index}] 不是对象")
    transform = entry.get("T_body_base_workpiece_semantic")
    if not isinstance(transform, list) or len(transform) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in transform):
        raise ValueError("JSON 缺少有效的 T_body_base_workpiece_semantic；请使用 tabletop_pointcloud_workpiece_pose.py 生成")
    values = finite([value for row in transform for value in row], 16, "T_body_base_workpiece_semantic")
    if any(abs(values[12 + axis] - expected) > 1e-6 for axis, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("T_body_base_workpiece_semantic 不是有效齐次变换")
    origin_xyz = [values[3], values[7], values[11]]
    x_axis = [values[0], values[4], values[8]]
    x_axis_norm_3d = math.sqrt(sum(component * component for component in x_axis))
    if abs(x_axis_norm_3d - 1.0) > 1e-3:
        raise ValueError(f"工件 +X 轴不是单位向量（长度={x_axis_norm_3d:.6f}）")
    x_axis_xy = x_axis[:2]
    x_axis_norm = math.hypot(*x_axis_xy)
    if x_axis_norm < 1e-6:
        raise ValueError("工件 +X 轴在 body_base XY 平面的投影过小，无法规划桌面抓取")
    workpiece_rpy = matrix_to_rpy_zyx([
        values[0], values[1], values[2],
        values[4], values[5], values[6],
        values[8], values[9], values[10],
    ])
    rotation = [
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]],
    ]
    grasp_xyz = [
        origin_xyz[row] + sum(rotation[row][column] * semantic_offset_m[column] for column in range(3))
        for row in range(3)
    ]
    return grasp_xyz, workpiece_rpy


def set_gripper_state(gripper: object, *, arm_id: int, target_open: bool, timeout_s: float, tolerance: float) -> None:
    target_position = 100.0 if target_open else 0.0
    action = "打开" if target_open else "闭合"
    gripper.move(arm_id, target_position, mode="async", preempt=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = gripper.get_state().get(arm_id)
        if isinstance(state, dict) and state.get("position") is not None and abs(float(state["position"]) - target_position) <= tolerance:
            print(f"右夹爪已{action}到位（position={float(state['position']):.1f}）。")
            return
        time.sleep(0.05)
    raise RuntimeError(f"右夹爪{action}等待超时；机械臂不会继续运动。")


def close_gripper_for_configured_workpiece(
    gripper: object,
    *,
    arm_id: int,
    timeout_s: float,
    grasp_position_min: float,
    grasp_position_max: float,
) -> None:
    """Close the gripper and classify this workpiece as grasped or empty.

    The valid terminal opening interval comes from the selected workpiece JSON.
    There is no separate narrow-grasp recovery mode: anything outside the
    calibrated interval is treated as an unconfirmed grasp.
    """
    gripper.move(arm_id, 0.0, mode="async", preempt=True)
    print("已下发右夹爪 Q1 闭合抓取命令，等待夹爪接触工件并停止...")
    deadline = time.monotonic() + timeout_s
    observed_closing = False
    while time.monotonic() < deadline:
        state = gripper.get_state().get(arm_id)
        if not isinstance(state, dict) or state.get("position") is None:
            time.sleep(0.05)
            continue
        position = float(state["position"])
        motion_state = str(state.get("motion_state", "")).lower()
        observed_closing |= motion_state == "moving" or position < 99.0
        if motion_state in {"error", "fault", "failed"}:
            raise RuntimeError(f"Q1 闭爪失败：gripper motion_state={motion_state!r}")
        if motion_state not in {"arrived", "gripping"} or not observed_closing:
            time.sleep(0.05)
            continue
        if not grasp_position_min <= position <= grasp_position_max:
            raise GraspNotConfirmedError(
                "抓取未确认：夹爪终止开度不在校准范围内"
                f"（position={position:.1f}，有效范围 "
                f"{grasp_position_min:.1f}--{grasp_position_max:.1f}）。"
            )
        print(
            "抓取成功：夹爪终止开度在校准范围内"
            f"（position={position:.1f}，范围 "
            f"{grasp_position_min:.1f}--{grasp_position_max:.1f}）。"
        )
        return
    raise GraspNotConfirmedError("闭爪抓取等待超时；未确认夹住工件。")


def planner_target(tcp_target: list[float], tcp_offset: list[float]) -> list[float]:
    return compose_pose(tcp_target, inverse_pose(tcp_offset))


def preplan_tcp_trajectory(planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str, tcp_offset: list[float], target: list[float], planner_kind: str, max_vel: float, max_acc: float, stage: str = "direct_grasp_plan") -> tuple[list, float]:
    """Plan via the table-aware Cartesian RRTConnect profile (cart+ompl)."""
    target_for_ik = planner_target(target, tcp_offset)
    start = planner_start_pose(planner, frame_id, ik_frame)
    started = time.perf_counter()
    target_record = {
        "frame_id": frame_id, "ik_frame": ik_frame,
        "tcp_pose_xyz_rotvec": target,
        "ik_pose_xyz_rotvec": target_for_ik,
    }
    try:
        transport = getattr(planner, "_trajectory", None)
        if transport is None:
            raise TrajectoryPlanningError("当前 parm_sdk 不支持 RRTConnect 所需的 _trajectory 接口")
        payload = {
            "waypoints": [{
                "name": "configured_workpiece_rrtconnect_target", "type": "cart", "group": group,
                "planner": "ompl", "max_velocity_scaling_factor": max_vel,
                "max_acceleration_scaling_factor": max_acc, "frame_id": frame_id,
                "ik_frame": ik_frame, "position": target_for_ik[:3],
                "orientation": rotvec_to_quat(target_for_ik[3:]),
            }],
            "topp": {"max_vel_factor": max_vel, "max_acc_factor": max_acc,
                     "final_phase_ratio": 0.05, "final_phase_oversample": 2.0},
        }
        result = transport.send_payload(payload, timeout=RRTCONNECT_TIMEOUT_S)
        data = result.get("data") or {}
        trajectory, planned_arm_id, frequency = data.get("trajectory"), data.get("arm_id"), data.get("sample_frequency", 10.0)
        if planned_arm_id != arm_id or not isinstance(trajectory, list) or not trajectory or not isinstance(frequency, (int, float)) or frequency <= 0:
            raise TrajectoryPlanningError("RRTConnect 未返回当前右臂的有效轨迹")
    except Exception as exc:
        planning_event(stage=stage, method=f"{planner_kind}/RRTConnect", start=start,
                       target=target_record, started_at=started, success=False,
                       failure_reason=f"{type(exc).__name__}: {exc}", arm_id=arm_id)
        if isinstance(exc, TrajectoryPlanningError):
            raise
        raise TrajectoryPlanningError(f"RRTConnect 轨迹规划失败: {exc}") from exc
    planning_event(stage=stage, method=f"{planner_kind}/RRTConnect", start=start,
                   target=target_record, started_at=started, success=True,
                   arm_id=arm_id, trajectory_waypoints=len(trajectory),
                   sample_frequency_hz=float(frequency))
    return trajectory, float(frequency)


def can_plan_target(
    planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str,
    tcp_offset: list[float], target: list[float], target_is_tcp: bool,
    planner_kind: str, max_vel: float, max_acc: float, timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """Check one non-RRTConnect Cartesian target without executing it."""
    target_for_ik = planner_target(target, tcp_offset) if target_is_tcp else target
    start = planner_start_pose(planner, frame_id, ik_frame)
    started = time.perf_counter()
    target_record = {
        "frame_id": frame_id, "ik_frame": ik_frame,
        "tcp_pose_xyz_rotvec": target if target_is_tcp else None,
        "ik_pose_xyz_rotvec": target_for_ik,
    }
    try:
        result = planner.plan_cart_trajectory(
            group=group, planner=planner_kind, frame_id=frame_id, ik_frame=ik_frame,
            position=target_for_ik[:3], orientation=rotvec_to_quat(target_for_ik[3:]),
            max_vel=max_vel, max_acc=max_acc, timeout=timeout_s,
        )
        data = result.get("data") or {}
        trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
        if planned_arm_id != arm_id:
            raise TrajectoryPlanningError(f"规划结果 arm_id={planned_arm_id!r}，与请求不一致")
        if not isinstance(trajectory, list) or not trajectory:
            raise TrajectoryPlanningError("规划器未返回有效 trajectory")
    except Exception as exc:
        planning_event(stage="feasibility_check", method=planner_kind, start=start,
                       target=target_record, started_at=started, success=False,
                       failure_reason=f"{type(exc).__name__}: {exc}", arm_id=arm_id)
        return False, str(exc)
    planning_event(stage="feasibility_check", method=planner_kind, start=start,
                   target=target_record, started_at=started, success=True,
                   arm_id=arm_id, trajectory_waypoints=len(trajectory))
    return True, f"{len(trajectory)} 个关节路点"


def plan_and_follow(arm: object, planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str, tcp_offset: list[float], target: list[float], planner_kind: str, max_vel: float, max_acc: float, retries: int, stage: str) -> None:
    print(f"{stage} RRTConnect TCP target: {[round(value, 6) for value in target]}")
    for attempt in range(retries + 1):
        trajectory, frequency = preplan_tcp_trajectory(planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=target, planner_kind=planner_kind, max_vel=max_vel, max_acc=max_acc, stage=f"{stage} planning attempt {attempt + 1}/{retries + 1}")
        try:
            arm.follow_trajectory(arm_id=arm_id, trajectory=trajectory, sample_frequency=frequency)
            return
        except Exception as exc:
            if "DiscreteTask::validate: first point mismatch" not in str(exc) or attempt >= retries:
                raise RuntimeError(f"{stage} 执行失败: {exc}") from exc
            time.sleep(0.2)


def servo_follow_tcp(arm: object, *, arm_id: int, tcp_target: list[float], servo_eef_offset: list[float], args: argparse.Namespace, stage: str) -> None:
    """Move a native right_eef target through ServoPose's calibrated TCP."""
    print(f"{stage}：使用控制器 servo 直线段，不经过 RRTConnect 碰撞预检。")
    servo_move_l(
        arm, arm_id, tcp_target,
        linear_speed_mps=args.servo_linear_speed_mps,
        angular_speed_radps=args.servo_angular_speed_radps,
        max_linear_accel_mps2=args.servo_max_linear_accel_mps2,
        max_angular_accel_radps2=args.servo_max_angular_accel_radps2,
        hz=args.servo_hz, position_tolerance_m=args.servo_position_tolerance_m,
        orientation_tolerance_rad=args.servo_orientation_tolerance_rad,
        settle_timeout_s=args.servo_settle_timeout_s, tcp=servo_eef_offset,
    )


def recover_target_tcp_pose(arm: object, planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str, tcp_offset: list[float], target: list[float], planner_kind: str, max_vel: float, max_acc: float, retries: int, xy_tolerance_m: float, z_tolerance_m: float, orientation_tolerance_rad: float, correction_attempts: int, stage: str) -> None:
    """Verify the complete TCP pose and correct it with RRTConnect if needed."""
    target_rpy = rotvec_to_rpy(target[3:])
    for attempt in range(correction_attempts + 1):
        time.sleep(0.2)
        current_tcp = compose_pose(
            pose_from_lookup_transform(planner.lookup_transform(frame_id, ik_frame).get("data")), tcp_offset,
        )
        position_errors = [current_tcp[axis] - target[axis] for axis in range(3)]
        current_rpy = rotvec_to_rpy(current_tcp[3:])
        orientation_errors = [
            (current_rpy[axis] - target_rpy[axis] + math.pi) % (2.0 * math.pi) - math.pi
            for axis in range(3)
        ]
        if (abs(position_errors[0]) <= xy_tolerance_m
                and abs(position_errors[1]) <= xy_tolerance_m
                and abs(position_errors[2]) <= z_tolerance_m
                and all(abs(error) <= orientation_tolerance_rad for error in orientation_errors)):
            return
        if attempt == correction_attempts:
            raise RuntimeError(
                f"{stage} TCP 位姿恢复失败："
                f"xyz=({position_errors[0] * 1000:.3f}, {position_errors[1] * 1000:.3f}, "
                f"{position_errors[2] * 1000:.3f}) mm, "
                f"roll/pitch/yaw={math.degrees(orientation_errors[0]):.3f}°/"
                f"{math.degrees(orientation_errors[1]):.3f}°/"
                f"{math.degrees(orientation_errors[2]):.3f}°"
            )
        plan_and_follow(arm, planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=target, planner_kind=planner_kind, max_vel=max_vel, max_acc=max_acc, retries=retries, stage=f"{stage}（{attempt + 1}/{correction_attempts}）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="工件直接抓取参数 JSON（必填）")
    parser.add_argument("--arm-id", type=int, choices=(2,), required=True, help="目前仅支持右臂 2")
    parser.add_argument("--workpiece-pose-file", type=Path, required=True)
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument(
        "--grasp-point-semantic-offset-m", nargs=3, type=float,
        default=(-0.030, 0.0, 0.0), metavar=("X", "Y", "Z"),
        help="语义原点沿工件自身 X/Y/Z 三轴到几何抓取点的偏置（m）",
    )
    parser.add_argument(
        "--tcp-local-offset-at-grasp-m", nargs=3, type=float,
        default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
        help="TCP 到几何抓取点后沿 TCP 自身 X/Y/Z 三轴施加的目标补偿（m）",
    )
    parser.add_argument("--grasp-retry-z-step-m", type=float, default=0.005,
                        help="未抓住后仅原地 +Z 复抓的高度（m；默认 0.005）")
    parser.add_argument(
        "--tcp-yaw-toward-workpiece-y-deg", type=float, default=-3.0,
        help="TCP 相对工件的本地 +Z 偏转（度；默认 -3）",
    )
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    parser.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "localhost"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
    parser.add_argument("--frame-id", default=os.environ.get("PARM_PLANNER_FRAME", "body_base"))
    parser.add_argument("--planner", choices=("ompl",), default="ompl", help="固定使用 table-aware RRTConnect")
    parser.add_argument("--final-approach-mode", choices=("servo", "rrt"), default="servo",
                        help="最后后方点到抓取点：servo 控制器直线段（默认）或 rrt")
    parser.add_argument("--servo-linear-speed-mps", type=float, default=0.05)
    parser.add_argument("--servo-angular-speed-radps", type=float, default=0.30)
    parser.add_argument("--servo-max-linear-accel-mps2", type=float, default=0.10)
    parser.add_argument("--servo-max-angular-accel-radps2", type=float, default=0.60)
    parser.add_argument("--servo-hz", type=float, choices=(10.0, 20.0, 25.0, 50.0, 100.0), default=50.0)
    parser.add_argument("--servo-position-tolerance-m", type=float, default=0.0005)
    parser.add_argument("--servo-orientation-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--servo-settle-timeout-s", type=float, default=8.0)
    parser.add_argument("--line-entry-clearance-m", type=float, default=DEFAULT_LINE_ENTRY_CLEARANCE_M)
    parser.add_argument("--line-max-rear-distance-m", type=float, default=DEFAULT_LINE_MAX_REAR_DISTANCE_M)
    parser.add_argument("--line-scan-count", type=int, default=15)
    parser.add_argument("--max-vel", type=float, default=0.1)
    parser.add_argument("--max-acc", type=float, default=0.1)
    parser.add_argument("--start-sync-retries", type=int, default=3)
    parser.add_argument("--gripper-timeout-s", type=float, default=5.0)
    parser.add_argument("--gripper-position-tolerance", type=float, default=5.0)
    parser.add_argument(
        "--grasp-position-min", type=float, default=6.0,
        help="闭爪后确认已抓住的最小 position（由配置提供）",
    )
    parser.add_argument(
        "--grasp-position-max", type=float, default=20.0,
        help="闭爪后确认已抓住的最大 position（由配置提供）",
    )
    parser.add_argument("--q1-post-grasp-lift-m", nargs="+", type=float, default=DEFAULT_Q1_POST_GRASP_LIFT_M)
    parser.add_argument("--q1-xy-tolerance-m", type=float, default=DEFAULT_Q1_XY_TOLERANCE_M)
    parser.add_argument("--q1-z-tolerance-m", type=float, default=DEFAULT_Q1_Z_TOLERANCE_M)
    parser.add_argument("--q1-pitch-tolerance-rad", type=float, default=DEFAULT_Q1_PITCH_TOLERANCE_RAD)
    parser.add_argument("--z-pitch-correction-attempts", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="实际下发运动；省略时仅预览")
    args = parser.parse_args()
    try:
        apply_grasp_config(parser, args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    if not args.workpiece_pose_file.is_file():
        print(f"错误: 工件位姿文件不存在: {args.workpiece_pose_file}", file=sys.stderr)
        return 1
    if (not 0.0 <= args.line_entry_clearance_m <= args.line_max_rear_distance_m
            or args.line_scan_count < 2 or args.grasp_retry_z_step_m <= 0
            or any(abs(value) > 0.200 for value in args.grasp_point_semantic_offset_m)
            or any(abs(value) > 0.200 for value in args.tcp_local_offset_at_grasp_m)):
        print("错误: 抓取线参数无效", file=sys.stderr)
        return 1
    if not (0 < args.max_vel <= 1 and 0 < args.max_acc <= 1):
        print("错误: --max-vel 和 --max-acc 必须在 (0, 1] 内", file=sys.stderr)
        return 1
    if (args.q1_xy_tolerance_m < 0 or args.q1_z_tolerance_m < 0
            or args.q1_pitch_tolerance_rad < 0 or args.z_pitch_correction_attempts < 0):
        print("错误: Q1 TCP 位姿校验参数无效", file=sys.stderr)
        return 1
    if (args.servo_linear_speed_mps <= 0 or args.servo_angular_speed_radps <= 0
            or args.servo_max_linear_accel_mps2 <= 0 or args.servo_max_angular_accel_radps2 <= 0
            or args.servo_position_tolerance_m < 0 or args.servo_orientation_tolerance_rad < 0
            or args.servo_settle_timeout_s <= 0):
        print("错误: servo 最终接近参数无效", file=sys.stderr)
        return 1
    if not 0 <= args.grasp_position_min <= args.grasp_position_max <= 100 or not args.q1_post_grasp_lift_m or any(value <= 0 for value in args.q1_post_grasp_lift_m):
        print("错误: Q1 夹爪参数无效", file=sys.stderr)
        return 1
    try:
        grasp_xyz, workpiece_rpy = load_workpiece_grasp_point(
            args.workpiece_pose_file, args.workpiece_index,
            list(args.grasp_point_semantic_offset_m),
        )
        # right_eef is a native planner frame; its target pose needs no
        # application-level virtual-TCP conversion.
        tcp_offset = [0.0] * 6
        servo_eef_offset = RIGHT_EEF_FROM_TRANSFER_BASE
        correction = [0.0] * 3
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    workpiece_orientation = rpy_to_rotvec(workpiece_rpy)
    local_yaw_correction = rpy_to_rotvec([
        0.0, 0.0, math.radians(args.tcp_yaw_toward_workpiece_y_deg),
    ])
    # T_body_tcp = T_body_workpiece ∘ Rz_local(correction).  Positive local
    # yaw maps TCP +X toward workpiece +Y, as requested.
    orientation = compose_pose(
        [0.0, 0.0, 0.0, *workpiece_orientation],
        [0.0, 0.0, 0.0, *local_yaw_correction],
    )[3:]
    tcp_local_offset_base = rotate_vector(
        rotvec_to_quat(orientation), list(args.tcp_local_offset_at_grasp_m)
    )
    grasp_xyz = [
        grasp_xyz[axis] + tcp_local_offset_base[axis] + correction[axis]
        for axis in range(3)
    ]
    # Generate every rear alignment candidate from the fully corrected final
    # TCP target, so the final advance remains exactly along TCP +X.
    approach_anchor_xyz = grasp_xyz.copy()
    target_rpy = rotvec_to_rpy(orientation)
    # With the default zero local yaw correction, TCP +X/+Y are respectively
    # parallel and same-direction to workpiece +X/+Y.  Therefore the rear
    # approach and servo advance follow the full 3-D workpiece X axis.
    tcp_x_axis = rotate_vector(rotvec_to_quat(orientation), [1.0, 0.0, 0.0])
    tcp_x_xy_norm_sq = sum(component * component for component in tcp_x_axis[:2])
    if tcp_x_xy_norm_sq < 1e-9:
        print("错误: TCP +X 在 body_base XY 平面的投影过小，无法生成接近线", file=sys.stderr)
        return 1
    print(
        "工件语义坐标直接抓取："
        f"几何抓取点语义 XYZ 偏置(m)={list(args.grasp_point_semantic_offset_m)}，"
        f"TCP 局部 XYZ 补偿(m)={list(args.tcp_local_offset_at_grasp_m)}，"
        f"换算到 body_base 的 TCP 局部补偿(m)={tcp_local_offset_base}，"
        f"目标 TCP roll/pitch/yaw={math.degrees(target_rpy[0]):.2f}°/"
        f"{math.degrees(target_rpy[1]):.2f}°/{math.degrees(target_rpy[2]) % 360.0:.2f}°，"
        f"TCP 相对工件局部 Z 偏转 {args.tcp_yaw_toward_workpiece_y_deg:.1f}°，"
        f"抓取点 xyz(m)={grasp_xyz}，"
        f"接近/推进方向 TCP +X={tcp_x_axis}"
    )
    if not args.execute:
        print("这是预览；添加 --execute 后将执行配置的语义 XYZ 抓取。")
        return 0

    arm = planner = gripper = None
    try:
        sdk = import_sdk()
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=args.planner_host)
        gripper = sdk.Gripper(host=args.gripper_host)
        group, ik_frame = "right_arm", "right_eef"
        set_gripper_state(gripper, arm_id=args.arm_id, target_open=True, timeout_s=args.gripper_timeout_s, tolerance=args.gripper_position_tolerance)

        start_tcp = compose_pose(pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data")), tcp_offset)
        projection = sum(
            (start_tcp[axis] - approach_anchor_xyz[axis]) * tcp_x_axis[axis]
            for axis in range(2)
        ) / tcp_x_xy_norm_sq
        preferred = min(max(projection, -args.line_max_rear_distance_m), -args.line_entry_clearance_m)
        parameters = [preferred] + [-args.line_max_rear_distance_m + (args.line_max_rear_distance_m - args.line_entry_clearance_m) * index / (args.line_scan_count - 1) for index in range(args.line_scan_count)]
        parameters = list(dict.fromkeys(round(value, 12) for value in parameters))
        alignment_trajectory = alignment_frequency = alignment_xy = alignment_target = None
        failures: list[str] = []
        # ``orientation`` was formed above from the complete ICP semantic
        # rotation and the calibrated local +Z yaw correction.
        for candidate_index, parameter in enumerate(parameters, start=1):
            candidate_xyz = [approach_anchor_xyz[axis] + parameter * tcp_x_axis[axis] for axis in range(3)]
            candidate = candidate_xyz + orientation
            try:
                trajectory, frequency = preplan_tcp_trajectory(planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=candidate, planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc, stage=f"Q1 rear alignment candidate {candidate_index}/{len(parameters)}")
            except TrajectoryPlanningError as exc:
                failures.append(f"{parameter * 1000:.1f} mm: {exc}")
                continue
            alignment_trajectory, alignment_frequency, alignment_xy, alignment_target = (
                trajectory, frequency, candidate_xyz[:2], candidate,
            )
            break
        if alignment_trajectory is None or alignment_frequency is None or alignment_xy is None or alignment_target is None:
            raise Q1DirectPathUnavailableError("工件 −X 抓取线后方安全窗口内没有直接轨迹；" + "；".join(failures))
        print(f"执行沿 TCP −X 后移 + X/Y 轴对齐的 RRTConnect 轨迹，终点 xy(m)={alignment_xy}。")
        arm.follow_trajectory(arm_id=args.arm_id, trajectory=alignment_trajectory, sample_frequency=alignment_frequency)
        recover_target_tcp_pose(arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=alignment_target, planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries, xy_tolerance_m=args.q1_xy_tolerance_m, z_tolerance_m=args.q1_z_tolerance_m, orientation_tolerance_rad=args.q1_pitch_tolerance_rad, correction_attempts=args.z_pitch_correction_attempts, stage="−X 对齐后 TCP 位姿校验")

        # The final contact point is the configured semantic -X grasp point,
        # plus the optional final-only body-base Z calibration.
        grasp_target = grasp_xyz + orientation
        if args.final_approach_mode == "servo":
            servo_follow_tcp(
                arm, arm_id=args.arm_id, tcp_target=grasp_target,
                servo_eef_offset=servo_eef_offset, args=args,
                stage="沿 TCP +X 从后方延长线推进至抓取点",
            )
        else:
            plan_and_follow(arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=grasp_target, planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries, stage="沿 TCP +X 从后方延长线推进至抓取点")
            recover_target_tcp_pose(arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=grasp_target, planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries, xy_tolerance_m=args.q1_xy_tolerance_m, z_tolerance_m=args.q1_z_tolerance_m, orientation_tolerance_rad=args.q1_pitch_tolerance_rad, correction_attempts=args.z_pitch_correction_attempts, stage="抓取点 TCP 位姿校验")

        grasp_attempt_targets = [
            ("初始抓取点", grasp_target),
            (
                f"初始点 +Z {args.grasp_retry_z_step_m * 1000:.1f} mm 复抓",
                grasp_target[:2] + [grasp_target[2] + args.grasp_retry_z_step_m] + grasp_target[3:],
            ),
        ]
        for attempt_index, (label, attempt_target) in enumerate(grasp_attempt_targets):
            if attempt_index:
                # A failed close leaves the jaws closed.  Reopen before
                # changing height; otherwise the next close cannot test the
                # workpiece at its new Z position.
                set_gripper_state(
                    gripper, arm_id=args.arm_id, target_open=True,
                    timeout_s=args.gripper_timeout_s,
                    tolerance=args.gripper_position_tolerance,
                )
                servo_follow_tcp(
                    arm, arm_id=args.arm_id, tcp_target=attempt_target,
                    servo_eef_offset=servo_eef_offset, args=args, stage=label,
                )
            try:
                close_gripper_for_configured_workpiece(
                    gripper, arm_id=args.arm_id, timeout_s=args.gripper_timeout_s,
                    grasp_position_min=args.grasp_position_min,
                    grasp_position_max=args.grasp_position_max,
                )
                break
            except GraspNotConfirmedError as exc:
                if attempt_index == len(grasp_attempt_targets) - 1:
                    raise RuntimeError(
                        f"{args.workpiece_name} 已在初始高度和 +Z "
                        f"{args.grasp_retry_z_step_m * 1000:.1f} mm 两个高度尝试闭爪，"
                        "仍未确认抓住；闭环管线应重新拍照计算。"
                    ) from exc
                print(f"{label} 未确认抓住（{exc}）；准备下一高度复抓。")

        tcp = compose_pose(pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data")), tcp_offset)
        lift_failures: list[str] = []
        for lift_m in args.q1_post_grasp_lift_m:
            try:
                plan_and_follow(arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, target=tcp[:2] + [tcp[2] + lift_m] + tcp[3:], planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries, stage=f"Q1 抓后 +Z 上抬 {lift_m * 1000:.0f} mm")
                break
            except TrajectoryPlanningError as exc:
                lift_failures.append(str(exc))
        else:
            raise RuntimeError("Q1 已夹住工件，但所有抓后 +Z 上抬高度均不可规划；" + "；".join(lift_failures))
        print("Q1 直接抓取完成。")
    except Q1DirectPathUnavailableError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        for client in (planner, gripper, arm):
            if client is not None:
                client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
