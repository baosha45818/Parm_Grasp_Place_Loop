#!/usr/bin/env python3
"""反面分支工件2：执行已确认凸耳点的左臂反抓。

只有显式传入 ``--execute`` 才会发送运动。Planner 直接使用新夹爪张开中心
``left_eef``，不叠加已废弃的法兰 TCP 或 bridge 偏置。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
QUALITY_GATE = HERE / "execution_quality_gate.json"
sys.path.insert(0, str(PROJECT))

from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.grasp.weight_verification import (
    POST_CLOSE_CHECK_LIFT_M,
    capture_open_gripper_baseline,
    verify_object_after_lift,
)
from modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.grasp.profiles import load_reverse_ready_profile
from tools.execution_timing import ExecutionTiming


NO_GRASP_EXIT_CODE = 20
REVERSE_GRASP_CONFIG = load_reverse_ready_profile("2")


@dataclass(frozen=True)
class PreparedGrasp:
    """工件2预测、标定和夹爪坐标系的预处理结果。"""

    prediction: dict
    pose_source: dict
    quality: dict
    camera_tcp: np.ndarray
    body_head: np.ndarray
    calibration_path: Path
    calibration: dict
    jaw_alignment: float
    object_rotation_error: float
    point_camera: np.ndarray


def within_here(path: Path) -> Path:
    result = path.expanduser().resolve()
    try:
        result.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须在 {HERE}: {result}") from exc
    return result


def pose(T: np.ndarray):
    """把 body_base 齐次矩阵转换为 Planner 位姿。"""
    from loop_types import ObjectPose
    return ObjectPose(frame_id="body_base", position=T[:3, 3].tolist(),
                      orientation=Rotation.from_matrix(T[:3, :3]).as_quat().tolist())


def debug_pose(T: np.ndarray) -> dict:
    """生成供抓取报告记录的紧凑位姿。"""
    return {"translation_m": T[:3, 3].round(8).tolist(),
            "rotation_matrix": T[:3, :3].round(8).tolist(),
            "quaternion_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().round(8).tolist()}


def normalize(v: np.ndarray) -> np.ndarray:
    """归一化三维方向向量。"""
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def nearest_rotation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """把 ICP 的近似旋转块投影到最近的正规旋转矩阵。"""
    raw = np.asarray(matrix, dtype=float)
    u, _, vt = np.linalg.svd(raw)
    result = u @ vt
    if np.linalg.det(result) < 0.:
        u[:, -1] *= -1.
        result = u @ vt
    error = float(np.max(np.abs(raw.T @ raw - np.eye(3))))
    return result, error


def make_required_grasp_frame(T_camera_object: np.ndarray, table_normal_camera: np.ndarray) -> np.ndarray:
    """构造 camera <- TCP，并固定左右手指相对工件 +X 的方向。

    TCP +X 是下探方向，TCP +Y 是夹爪开合方向；+Y 对应左指，-Y 对应右指。
    """
    object_rotation, _ = nearest_rotation(T_camera_object[:3, :3])
    approach = -normalize(table_normal_camera)           # physical down -> table
    # Keep the requested +X finger side, while enforcing that jaws lie in the
    # table plane and remain perpendicular to the real down direction.
    left_finger = object_rotation[:, 0] - approach * np.dot(object_rotation[:, 0], approach)
    left_finger = normalize(left_finger)
    if abs(float(np.dot(approach, left_finger))) > 1e-4:
        raise RuntimeError("工件 +X 与 +Z 不正交，拒绝构造夹爪姿态")
    side = normalize(np.cross(approach, left_finger))
    left_finger = normalize(np.cross(side, approach))
    T = np.eye(4)
    T[:3, :3] = np.column_stack((approach, left_finger, side))
    if np.linalg.det(T[:3, :3]) < 0.0:
        raise RuntimeError("夹爪姿态不是右手坐标系")
    return T


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pose_quality(pose_source: dict) -> dict:
    gate = load_json(QUALITY_GATE)
    icp = pose_source.get("icp") or {}
    # 正式链路只接受 Module 4 当前 full-6DoF ICP 结果。
    match = icp.get("match_fraction")
    fitness = float(icp.get("fitness", float("nan")))
    rmse = (
        float(icp["rmse_m"])
        if icp.get("rmse_m") is not None
        else float(icp.get("rmse_mm", float("nan"))) * 1e-3
    )
    depth_score_mm = icp.get("depth_score_mm")
    result = {"source_schema": "full_6d_icp", "fitness": fitness, "icp_rmse_m": rmse, "depth_match_fraction": match,
              "depth_score_mm": depth_score_mm,
              "minimum_fitness": float(gate["minimum_icp_fitness"]),
              "minimum_depth_match_fraction": float(gate["minimum_depth_match_fraction"]),
              "maximum_depth_score_mm": float(gate["maximum_depth_score_mm"]),
              "maximum_icp_rmse_m": float(gate["maximum_icp_rmse_m"]),
              "gate_file": str(QUALITY_GATE)}
    depth_ok = (
        float(match) >= result["minimum_depth_match_fraction"]
        if match is not None
        else depth_score_mm is not None
        and float(depth_score_mm) <= result["maximum_depth_score_mm"]
    )
    result["accepted"] = bool(
        np.isfinite(fitness)
        and fitness >= result["minimum_fitness"]
        and depth_ok
        and np.isfinite(rmse)
        and rmse <= result["maximum_icp_rmse_m"]
    )
    return result


def build_targets(T_camera_grasp_center: np.ndarray, body_head: np.ndarray,
                  pregrasp_m: float) -> tuple[np.ndarray, np.ndarray]:
    """生成直接使用 ``left_eef`` 的抓取与预抓取目标。

    视觉抓取中心就是 IK 目标原点，不能再叠加旧 EEF->TCP 或
    EEF->left_arm_end_link 变换，否则会引入厘米级系统误差。
    """
    target = body_head @ T_camera_grasp_center
    above_center = T_camera_grasp_center.copy()
    above_center[:3, 3] -= T_camera_grasp_center[:3, 0] * pregrasp_m
    return target, body_head @ above_center


def wait_left(arm_host: str, joints: list[float], label: str, timeout: float = 90.) -> None:
    """等待左臂实时反馈到达指定七关节终点。"""
    from parm_sdk import Arm
    target = np.asarray(joints, dtype=float)
    deadline = time.monotonic() + timeout
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = np.asarray(((state.get("state") or {}).get("left") or {}).get("position") or [], dtype=float)
            if actual.shape == (7,):
                error = (actual - target + 180.) % 360. - 180.
                if float(np.max(np.abs(error))) <= .5:
                    return
            time.sleep(.05)
    raise RuntimeError(f"{label} 未在 {timeout:.0f}s 内到位")


def wait_gripper_terminal(gripper: object, hand: int, label: str, timeout: float = 15.) -> dict:
    """等待夹爪命令进入 arrived/gripping 物理终态。"""
    deadline = time.monotonic() + timeout
    last_state: dict | None = None
    while time.monotonic() < deadline:
        states = gripper.get_state()
        state = states.get(hand) if isinstance(states, dict) else None
        if isinstance(state, dict):
            last_state = state
            motion_state = str(state.get("motion_state", "")).strip().lower()
            if motion_state in {"error", "fault", "failed"}:
                raise RuntimeError(f"{label}失败: {state}")
            if motion_state in {"arrived", "gripping"}:
                return state
        time.sleep(.05)
    raise RuntimeError(f"{label}在 {timeout:.1f}s 内未到终态，最后状态: {last_state}")


def check_head(head_host: str, expected: dict) -> None:
    """按当前 mode 的反馈符号确认云台仍在标定观察角。"""
    from parm_sdk import Head
    with Head(host=head_host) as head:
        state = head.get_state() or {}
    pitch_sign = int(os.environ.get("PARM_HEAD_FEEDBACK_PITCH_SIGN", "1"))
    pitch = pitch_sign * float(state["pitch"]["position"])
    yaw = float(state["yaw"]["position"])
    if abs(pitch - float(expected["pitch"])) > .5 or abs(yaw - float(expected["yaw"])) > .5:
        raise RuntimeError(f"云台不在标定观察位：实际 pitch={pitch:.2f}, yaw={yaw:.2f}；期望 {expected}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="显式允许机器人、夹爪与 Planner 运动")
    p.add_argument("--preflight-only", action="store_true", help="仅做 Planner RRTStar+LIN 可达性检查，不运动")
    p.add_argument("--pregrasp-m", type=float, default=.060)
    p.add_argument(
        "--lift-m",
        type=float,
        default=float(REVERSE_GRASP_CONFIG["post_close_lift_m"]),
        help="固定为 0.03m：闭夹爪后沿 body_base +Z 上提 3cm 再做力传感器判断",
    )
    p.add_argument("--approach-angles-deg", type=float, nargs="+", default=[90., 15., -15.],
                   help="相对桌面的接近角；90 为竖直，15/-15 为绕两指轴侧向接近")
    p.add_argument(
        "--speed", type=float,
        default=float(os.environ.get("PARM_LEFT_ARM_SPEED", "0.1")),
    )
    p.add_argument(
        "--planning-inflight",
        type=int,
        default=3,
        help="并发离线规划请求数；每个请求使用独立 Planner 客户端",
    )
    p.add_argument("--close-force", type=float, default=100.)
    p.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "127.0.0.1"))
    p.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"))
    p.add_argument("--head-host", default=os.environ.get("PARM_HEAD_HOST", "192.168.3.22"))
    p.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在访问云台、Planner 和机器人前校验所有运动参数。"""
    if args.execute and args.preflight_only:
        raise ValueError("--execute 与 --preflight-only 不能同时使用")
    invalid = (
        args.pregrasp_m <= 0.0
        or abs(args.lift_m - POST_CLOSE_CHECK_LIFT_M) > 1e-9
        or not 0.0 < args.speed <= 1.0
        or not args.approach_angles_deg
        or any(not 0.0 < abs(angle) <= 90.0 for angle in args.approach_angles_deg)
    )
    if invalid:
        raise ValueError(
            f"pregrasp 必须为正，lift 必须固定为 {POST_CLOSE_CHECK_LIFT_M:.2f}m，"
            "speed 必须在 (0,1]，接近角必须在 [-90,90] 内且不能为 0"
        )
    if args.planning_inflight < 1:
        raise ValueError("--planning-inflight 必须不小于 1")


def load_body_head_calibration() -> tuple[Path, dict, np.ndarray]:
    """加载由顶层 mode 原子绑定的 body_base <- head_camera 标定。"""
    raw_path = os.environ.get("PARM_BODY_HEAD_CALIBRATION")
    if not raw_path:
        raise RuntimeError("必须由 Module 5-1 --mode 提供 PARM_BODY_HEAD_CALIBRATION")
    path = Path(raw_path).expanduser().resolve()
    payload = load_json(path)
    transform_key = next(
        (key for key in ("T_body_base_head_camera", "T_base_head_camera", "T") if key in payload),
        None,
    )
    if transform_key is None:
        raise RuntimeError(f"头部标定缺少 body_base <- head_camera 变换: {path}")
    transform = np.asarray(payload[transform_key], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError(f"头部标定矩阵无效: {path}")
    return path, payload, transform


def prepare_grasp(args: argparse.Namespace) -> PreparedGrasp:
    """把 Module 4 位姿和固定凸耳点转换成 head_camera 系夹爪中心。"""
    prediction_path = args.prediction_json.expanduser().resolve()
    prediction = load_json(prediction_path)
    if prediction.get("event") != "workpiece_2_convex_tab_grasp_prediction":
        raise RuntimeError("prediction-json 不是本项目的凸起抓取点预测")

    pose_source = load_json(Path(prediction["source_pose"]))
    quality = validate_pose_quality(pose_source)
    camera_object = np.asarray(pose_source["T_camera_object"], dtype=float)
    object_rotation, rotation_error = nearest_rotation(camera_object[:3, :3])
    camera_object[:3, :3] = object_rotation

    raw_table_normal = np.asarray(
        pose_source.get("table_normal_camera", []), dtype=float
    )
    if raw_table_normal.shape != (3,):
        raise RuntimeError("Module 4 位姿缺少 table_normal_camera")
    table_normal = normalize(raw_table_normal)
    camera_tcp = make_required_grasp_frame(camera_object, table_normal)
    point_camera = np.asarray(
        prediction["grasp_contact_point_camera_m"], dtype=float
    )
    camera_tcp[:3, 3] = point_camera

    # TCP +Y 必须始终指向工件 +X，保证左右手指不会在重构后互换。
    approach = -table_normal
    required_jaw = normalize(camera_object[:3, 0])
    required_jaw = normalize(
        required_jaw - approach * np.dot(required_jaw, approach)
    )
    jaw_alignment = float(np.dot(camera_tcp[:3, 1], required_jaw))
    if jaw_alignment < 0.999:
        raise RuntimeError(
            f"左指 +Y 未对齐工件 +X 的桌面投影: dot={jaw_alignment:.6f}"
        )

    calibration_path, calibration, body_head = load_body_head_calibration()
    return PreparedGrasp(
        prediction=prediction,
        pose_source=pose_source,
        quality=quality,
        camera_tcp=camera_tcp,
        body_head=body_head,
        calibration_path=calibration_path,
        calibration=calibration,
        jaw_alignment=jaw_alignment,
        object_rotation_error=rotation_error,
        point_camera=point_camera,
    )


def make_report(args: argparse.Namespace, prepared: PreparedGrasp) -> dict:
    """建立贯穿预检和真实执行阶段的抓取报告。"""
    return {
        "event": "workpiece_2_convex_tab_left_grasp_plan",
        "mode": (
            "execute"
            if args.execute
            else ("preflight_only" if args.preflight_only else "dry_run")
        ),
        "prediction_json": str(args.prediction_json.expanduser().resolve()),
        "body_head_calibration": str(prepared.calibration_path),
        "source_pose": prepared.prediction["source_pose"],
        "grasp_frame_convention": (
            "Module-1-1 direct grasp centre: Planner IK frame left_eef; "
            "+X approach-into-table, +Y jaw-opening; +Y side is left finger"
        ),
        "planner_ik_frame": "left_eef",
        "planner_target_mode": "direct_grasp_center_no_legacy_tcp_or_bridge",
        "finger_side_constraint": {
            "left_finger": "grasp point object +X",
            "right_finger": "grasp point object -X",
        },
        "left_finger_axis_alignment_to_object_x_table_projection": (
            prepared.jaw_alignment
        ),
        "grasp_tcp_head_camera": debug_pose(prepared.camera_tcp),
        "source_icp_rotation_orthogonality_error": prepared.object_rotation_error,
        "pose_quality_gate": prepared.quality,
        "grasp_contact_camera_m": prepared.point_camera.round(8).tolist(),
        "local_table_normal_down_offset_m": float(
            prepared.prediction.get("grasp_down_offset_m", 0.0)
        ),
        "pregrasp_m": args.pregrasp_m,
        "lift_m": args.lift_m,
        "approach_angles_deg": args.approach_angles_deg,
        "created_at": time.time(),
    }


def make_planner_config(args: argparse.Namespace):
    """创建工件2使用的直接 ``left_eef`` Planner 配置。"""
    from loop_config import LoopConfig

    return LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_eef",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.speed,
        max_acceleration=args.speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )


def move_to_ready_and_open(
    args: argparse.Namespace,
    runtime: object,
    ready_joints: list[float],
    timing: ExecutionTiming,
) -> None:
    """真实执行前回到统一反抓预备位并张开左夹爪。"""
    from parm_sdk import Gripper

    timing.start("move_to_reverse_ready")
    runtime.move_left_joints(ready_joints, planner="rrtstar")
    wait_left(args.arm_host, ready_joints, "左臂预备位")
    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        state = gripper.get_state().get(1)
        if not is_gripper_usable(state):
            raise RuntimeError("左夹爪不可用或未初始化")
        gripper.move(
            1,
            100.0,
            force=args.close_force,
            speed=80.0,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
    timing.finish("move_to_reverse_ready")


def build_approach_candidates(
    camera_tcp: np.ndarray,
    body_head: np.ndarray,
    approach_angles_deg: list[float],
    pregrasp_m: float,
) -> list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    """生成保持 TCP +Y 指向工件 +X 的接近候选。"""
    candidates = []
    for angle in sorted(dict.fromkeys(approach_angles_deg), key=lambda x: -abs(x)):
        tilt_deg = float(np.copysign(90.0 - abs(angle), angle))
        candidate_tcp = camera_tcp.copy()
        candidate_tcp[:3, :3] = camera_tcp[:3, :3] @ Rotation.from_rotvec(
            [0.0, np.deg2rad(tilt_deg), 0.0]
        ).as_matrix()
        target, above = build_targets(candidate_tcp, body_head, pregrasp_m)
        candidates.append((angle, tilt_deg, candidate_tcp, target, above))
    return candidates


def plan_offline_candidate(
    candidate: tuple[float, float, np.ndarray, np.ndarray, np.ndarray],
    planner_config: object,
    ready_joints: list[float],
) -> dict:
    """使用独立 Planner 客户端验证一组 RRTStar 预抓取和 LIN 下探。"""
    from tools.planner_runtime import PlannerRuntime

    angle, tilt_deg, candidate_tcp, target, above = candidate
    item = {
        "table_approach_angle_deg": angle,
        "side_tilt_about_tcp_y_deg": tilt_deg,
        "left_finger_axis_alignment": "TCP +Y unchanged = object +X",
        "pregrasp_body_base": debug_pose(above),
        "grasp_body_base": debug_pose(target),
        "planning_start_joints_deg": ready_joints,
    }
    planner = PlannerRuntime(planner_config)
    try:
        planner.open(required_frames={"body_base", "left_eef"})
        pregrasp_plan = planner.plan_cartesian_pose(
            "left",
            pose(above),
            planner="rrtstar",
            start_joints_deg=ready_joints,
            rrtstar_goal_tolerance=True,
        )
        trajectory = ((pregrasp_plan or {}).get("data") or {}).get("trajectory") or []
        if not trajectory or len(trajectory[-1]) != 7:
            raise RuntimeError("预抓取规划没有返回七关节终点")
        pregrasp_endpoint = [float(value) for value in trajectory[-1]]
        descent_plan = planner.plan_cartesian_pose(
            "left",
            pose(target),
            planner="lin",
            start_joints_deg=pregrasp_endpoint,
        )
        item["reachable"] = True
        item["pregrasp_joint_target_deg"] = pregrasp_endpoint
        return {
            "item": item,
            "candidate_tcp": candidate_tcp,
            "target": target,
            "above": above,
            "pre": pregrasp_plan,
            "descent": descent_plan,
            "pre_endpoint": pregrasp_endpoint,
        }
    except Exception as exc:
        item["reachable"] = False
        item["reason"] = str(exc)
        return {"item": item}
    finally:
        planner.close()


def plan_candidates_concurrently(
    candidates: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]],
    planner_config: object,
    ready_joints: list[float],
    planning_inflight: int,
) -> tuple[list[dict], list[dict]]:
    """并发预检候选；保持原逻辑，选择最先完成的可达结果。"""
    attempts: list[dict] = []
    reachable: list[dict] = []
    print(
        f"[反抓离线规划并发] 候选={len(candidates)}，最多{planning_inflight}条在飞；"
        "每条使用独立 Planner 客户端。",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=min(planning_inflight, len(candidates)),
        thread_name_prefix="module5-reverse-grasp-plan",
    ) as executor:
        futures = {
            executor.submit(
                plan_offline_candidate, candidate, planner_config, ready_joints
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            result = future.result()
            attempts.append(result["item"])
            if result["item"]["reachable"]:
                reachable.append(result)
    return attempts, reachable


def execute_selected_grasp(
    args: argparse.Namespace,
    runtime: object,
    selected: dict,
    timing: ExecutionTiming,
    report: dict,
) -> None:
    """执行预抓取、LIN 下探、闭夹爪、3cm 上提和力传感器判断。"""
    from parm_sdk import Gripper

    timing.start("move_to_pregrasp")
    runtime.execute_planned_result(selected["pre"])
    wait_left(args.arm_host, selected["pre_endpoint"], "左臂预抓取位")
    timing.finish("move_to_pregrasp")

    timing.start("linear_grasp_descent")
    runtime.execute_planned_result(selected["descent"])
    descent_trajectory = (
        ((selected["descent"] or {}).get("data") or {}).get("trajectory") or []
    )
    if not descent_trajectory:
        raise RuntimeError("LIN 下探没有返回终点")
    wait_left(args.arm_host, descent_trajectory[-1], "左臂抓取位")
    timing.finish("linear_grasp_descent")

    force_host = os.environ["PARM_FORCE_SENSOR_HOST"]
    weight_baseline = capture_open_gripper_baseline(force_host)
    timing.start("gripper_close")
    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        response = gripper.move(
            1,
            0.0,
            force=args.close_force,
            speed=80.0,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
        report["gripper_close_state"] = wait_gripper_terminal(
            gripper, 1, "左夹爪闭合"
        )
    timing.finish("gripper_close")

    # 先脱离桌面，再用同姿态下的力差判断是否抓到工件。
    timing.start("post_grasp_lift_and_force_check")
    check_lift = selected["target"].copy()
    check_lift[2, 3] += POST_CLOSE_CHECK_LIFT_M
    lift_plan = runtime.plan_cartesian_pose(
        "left", pose(check_lift), planner="rrtstar"
    )
    runtime.execute_planned_result(lift_plan)
    lift_trajectory = ((lift_plan or {}).get("data") or {}).get("trajectory") or []
    if not lift_trajectory:
        raise RuntimeError("抬升规划没有返回终点")
    wait_left(args.arm_host, lift_trajectory[-1], "左臂上提3cm检查位")
    force_result = verify_object_after_lift(weight_baseline, force_host)
    report["post_lift_force_check"] = force_result
    timing.finish("post_grasp_lift_and_force_check")
    if not force_result["accepted"]:
        with Gripper(host=args.gripper_host, status_port=7702) as gripper:
            gripper.move(
                1,
                100.0,
                force=100.0,
                speed=80.0,
                mode="sync",
                preempt=False,
                timeout=60.0,
            )
        print(
            "NO_GRASP: 反抓上提3cm后力传感器未确认抓到工件；"
            "已打开左夹爪，返回预备位后重拍。",
            flush=True,
        )
        raise SystemExit(NO_GRASP_EXIT_CODE)
    report["gripper_close_response"] = response


def main() -> int:
    args = parse_args()
    validate_args(args)
    output = within_here(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")

    prepared = prepare_grasp(args)
    output.mkdir(parents=True)
    report = make_report(args, prepared)
    report_path = output / "grasp_plan.json"
    timing = ExecutionTiming()
    timing_status = "FAILURE"

    try:
        if not prepared.quality["accepted"]:
            raise RuntimeError(
                "execution blocked: Module-5-1 ICP quality gate rejected pose "
                + json.dumps(prepared.quality, ensure_ascii=False)
            )
        # 纯 dry-run 在这里结束，保证不访问云台、Planner 或机器人。
        if not args.execute and not args.preflight_only:
            report["status"] = "dry_run_ok_no_hardware_access"
            timing_status = "SUCCESS_DRY_RUN"
            return 0

        check_head(args.head_host, prepared.calibration["head_pose_deg"])
        from tools.planner_runtime import PlannerRuntime

        ready_joints = list(REVERSE_GRASP_CONFIG["ready_joints_deg"])
        report["preflight_start_joint_angles_deg"] = ready_joints
        planner_config = make_planner_config(args)
        runtime = PlannerRuntime(planner_config)
        try:
            runtime.open(required_frames={"body_base", "left_eef"})
            if args.execute:
                move_to_ready_and_open(args, runtime, ready_joints, timing)

            candidates = build_approach_candidates(
                prepared.camera_tcp,
                prepared.body_head,
                args.approach_angles_deg,
                args.pregrasp_m,
            )
            timing.start("parallel_approach_preflight")
            attempts, reachable = plan_candidates_concurrently(
                candidates,
                planner_config,
                ready_joints,
                args.planning_inflight,
            )
            timing.finish("parallel_approach_preflight")
            report["approach_preflight"] = attempts
            if not reachable:
                raise RuntimeError(
                    "所有保持左右指方向的接近角均不可达；"
                    "检查 grasp_plan.json 的 approach_preflight"
                )

            # 保留原行为：选择最先完成规划的可达候选。
            selected = reachable[0]
            report["selected_approach"] = selected["item"]
            report["preflight_status"] = "ptp_pregrasp_and_lin_descent_reachable"
            if args.preflight_only:
                report["status"] = "preflight_ok_no_motion"
                timing_status = "SUCCESS_PREFLIGHT"
                return 0

            execute_selected_grasp(args, runtime, selected, timing, report)
            report["status"] = "grasp_and_lift_complete"
            timing_status = "SUCCESS"
        finally:
            runtime.close()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["timing"] = timing.complete(timing_status)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"GRASP_REPORT={report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
