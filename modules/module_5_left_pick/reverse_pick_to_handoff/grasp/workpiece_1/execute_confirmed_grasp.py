#!/usr/bin/env python3
"""反面分支工件1：执行已确认的反抓位姿。

本文件只负责 Planner 预检和真实抓取事务。只有 ``--execute`` 才会发送运动；
``--preflight-only`` 只连接 Planner 做可达性检查，不执行轨迹。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[5]
MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(MODULE_ROOT), str(ROOT)]

from loop_config import LoopConfig
from loop_types import ObjectPose
from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.profiles import (
    load_reverse_ready_profile,
)
from parm_sdk.api.exceptions import ParmAPIError
from tools.execution_timing import ExecutionTiming
from tools.planner_runtime import PlannerRuntime


NO_GRASP_EXIT_CODE = 20
REVERSE_GRASP_CONFIG = load_reverse_ready_profile("1")
POST_CLOSE_LIFT_M = 0.03
LEFT_GRIPPER_HELD_POSITION_MIN = 35.0


@dataclass(frozen=True)
class GraspCandidate:
    """一个接近角对应的相机系和 body_base 系目标。"""

    approach_deg: float
    tilt_deg: float
    camera_grasp: np.ndarray
    body_grasp: np.ndarray
    body_pregrasp: np.ndarray
    camera_pregrasp: np.ndarray


@dataclass
class PlannedCandidate:
    """候选的离线规划结果；不可达时 ``pregrasp_plan`` 为 None。"""

    candidate: GraspCandidate
    pregrasp_plan: dict | None
    report: dict


def parse_args() -> argparse.Namespace:
    """解析工件 1 执行器参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-json", type=Path, required=True)
    parser.add_argument(
        "--body-head",
        type=Path,
        required=True,
        help="body_base <- head camera 标定",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="规划全部接近候选，但不执行任何轨迹",
    )
    parser.add_argument("--speed", type=float, default=0.20)
    parser.add_argument("--above-m", type=float, default=0.03)
    parser.add_argument(
        "--approach-angles-deg",
        nargs="+",
        type=float,
        default=[90.0, 15.0, -15.0],
        help="相对桌面的接近角；90 为垂直下探",
    )
    parser.add_argument("--body-z-down-m", type=float, default=0.005)
    parser.add_argument(
        "--lift-m",
        type=float,
        default=float(REVERSE_GRASP_CONFIG["post_close_lift_m"]),
        help="固定为 0.03m：闭夹爪后上提 3cm",
    )
    parser.add_argument(
        "--gripper-host",
        default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"),
    )
    parser.add_argument(
        "--planner-host",
        default=os.environ.get("PARM_PLANNER_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--arm-host",
        default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"),
    )
    parser.add_argument(
        "--execute-url",
        default="http://127.0.0.1:5667/middle/waypoint/execute",
    )
    parser.add_argument("--timing-file", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在连接 Planner 和机器人之前拒绝不安全参数。"""
    if args.execute and args.preflight_only:
        raise ValueError("--execute 与 --preflight-only 不能同时使用")
    invalid = (
        args.above_m <= 0.0
        or abs(args.lift_m - POST_CLOSE_LIFT_M) > 1e-9
        or args.body_z_down_m < 0.0
        or not args.approach_angles_deg
        or any(not 0.0 < abs(angle) <= 90.0 for angle in args.approach_angles_deg)
    )
    if invalid:
        raise ValueError(
            f"--above-m 必须为正，--lift-m 必须固定为 {POST_CLOSE_LIFT_M:.2f}m；"
            "--body-z-down-m 必须非负；接近角必须在 [-90,90] 内且不能为 0"
        )


def load_body_head_transform(path: Path) -> np.ndarray:
    """读取 body_base <- head_camera 齐次变换。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = next(
        (name for name in ("T_base_head_camera", "T_body_base_head_camera", "T") if name in payload),
        None,
    )
    if key is None:
        raise RuntimeError(f"头部相机标定缺少 4x4 外参: {path}")
    transform = np.asarray(payload[key], dtype=float)
    valid = (
        transform.shape == (4, 4)
        and np.all(np.isfinite(transform))
        and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
    )
    if not valid:
        raise RuntimeError(f"头部相机标定不是有效 4x4 矩阵: {path}")
    return transform


def load_unique_grasp(path: Path) -> tuple[dict, np.ndarray]:
    """加载唯一抓取候选，并转成 head_camera <- left_eef 矩阵。"""
    records = json.loads(path.read_text(encoding="utf-8"))
    if len(records) != 1:
        raise RuntimeError(
            f"执行被阻止：需要唯一反抓候选，当前为 {len(records)} 个；"
            "请检查 grasp_overlay.png"
        )
    record = records[0]
    grasp = np.eye(4)
    grasp[:3, :3] = record["grasp_pose"]["rotation_matrix"]
    grasp[:3, 3] = record["grasp_pose"]["translation_m"]
    return record, grasp


def object_pose(transform: np.ndarray) -> ObjectPose:
    """把 4x4 目标矩阵转换为 Planner 使用的 body_base 位姿。"""
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return ObjectPose(
        frame_id="body_base",
        position=transform[:3, 3].tolist(),
        orientation=quaternion.tolist(),
    )


def pose_debug(transform: np.ndarray) -> dict:
    """生成便于排查坐标系问题的位姿记录。"""
    rotation = Rotation.from_matrix(transform[:3, :3])
    return {
        "position_m": transform[:3, 3].round(7).tolist(),
        "rpy_xyz_deg": rotation.as_euler("xyz", degrees=True).round(3).tolist(),
        "quaternion_xyzw": rotation.as_quat().round(8).tolist(),
        "rotation_matrix": transform[:3, :3].round(8).tolist(),
    }


def tilt_from_table_angle_deg(angle_deg: float) -> float:
    """把桌面接近角转换成绕夹爪局部 +Y 的侧倾角。"""
    angle = float(angle_deg)
    return float(np.copysign(90.0 - abs(angle), angle))


def make_planner_config(args: argparse.Namespace, *, live: bool) -> LoopConfig:
    """创建只使用 ``left_eef`` 直接抓取中心的 Planner 配置。"""
    return LoopConfig(
        planner_host=args.planner_host,
        execute_url=args.execute_url,
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_eef",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.speed,
        max_acceleration=args.speed,
        idle_poll_seconds=0.2,
        live=live,
        poses={},
    )


def build_candidates(
    grasp_camera: np.ndarray,
    body_head: np.ndarray,
    approach_angles_deg: list[float],
    above_m: float,
    body_z_down_m: float,
) -> list[GraspCandidate]:
    """生成按优先级排列的接近姿态和预抓取姿态。"""
    candidates: list[GraspCandidate] = []
    angles = sorted(dict.fromkeys(approach_angles_deg), key=lambda value: -abs(value))
    for approach_deg in angles:
        tilt_deg = tilt_from_table_angle_deg(approach_deg)
        tilt = np.eye(4)
        tilt[:3, :3] = Rotation.from_rotvec(
            [0.0, np.deg2rad(tilt_deg), 0.0]
        ).as_matrix()
        camera_grasp = grasp_camera @ tilt
        camera_pregrasp = camera_grasp.copy()
        camera_pregrasp[:3, 3] -= camera_grasp[:3, 0] * float(above_m)
        body_grasp = body_head @ camera_grasp
        body_pregrasp = body_head @ camera_pregrasp
        # 两个目标应用相同的现场 Z 补偿，保持预抓取间距不变。
        body_grasp[2, 3] -= body_z_down_m
        body_pregrasp[2, 3] -= body_z_down_m
        candidates.append(
            GraspCandidate(
                approach_deg=float(approach_deg),
                tilt_deg=tilt_deg,
                camera_grasp=camera_grasp,
                body_grasp=body_grasp,
                body_pregrasp=body_pregrasp,
                camera_pregrasp=camera_pregrasp,
            )
        )
    return candidates


def plan_candidate(
    candidate: GraspCandidate,
    planner_config: LoopConfig,
    *,
    live_planner: bool,
) -> PlannedCandidate:
    """使用独立 Planner 客户端验证一个接近候选。"""
    report = {
        "approach_angle_from_table_deg": candidate.approach_deg,
        "side_approach_tilt_deg": candidate.tilt_deg,
        "left_eef_head_camera": pose_debug(candidate.camera_grasp),
        "pregrasp_left_eef_head_camera": pose_debug(candidate.camera_pregrasp),
        "left_eef_target_body_base": pose_debug(candidate.body_grasp),
        "above_target_body_base": pose_debug(candidate.body_pregrasp),
    }
    started = time.perf_counter()
    if not live_planner:
        report["reachable"] = None
        report["planning_duration_seconds"] = time.perf_counter() - started
        return PlannedCandidate(candidate, None, report)

    planner = PlannerRuntime(planner_config)
    try:
        planner.open(required_frames={"body_base", "left_eef"})
        try:
            pregrasp_plan = planner.plan_cartesian_pose(
                "left", object_pose(candidate.body_pregrasp), planner="rrtstar"
            )
        except ParmAPIError as exc:
            report.update(reachable=False, reason=str(exc))
            return PlannedCandidate(candidate, None, report)

        trajectory = (pregrasp_plan.get("data") or {}).get("trajectory") or []
        if not trajectory or len(trajectory[-1]) != 7:
            report.update(
                reachable=False,
                reason="预抓取规划没有返回用于下探验证的七关节终点",
            )
            return PlannedCandidate(candidate, None, report)
        endpoint = [float(value) for value in trajectory[-1]]
        report["pregrasp_joint_target_deg"] = endpoint

        # 必须从预抓取规划的精确终点继续验证下探，不能只验证两个孤立目标。
        try:
            planner.plan_cartesian_pose(
                "left",
                object_pose(candidate.body_grasp),
                planner="rrtstar",
                start_joints_deg=endpoint,
            )
        except (ParmAPIError, RuntimeError) as exc:
            report.update(
                reachable=False,
                descent_reachable=False,
                reason=f"预抓取可达，但从其终点下探不可达: {exc}",
            )
            return PlannedCandidate(candidate, None, report)

        report.update(reachable=True, descent_reachable=True)
        return PlannedCandidate(candidate, pregrasp_plan, report)
    finally:
        report["planning_duration_seconds"] = time.perf_counter() - started
        planner.close()


def plan_all_candidates(
    candidates: list[GraspCandidate],
    planner_config: LoopConfig,
    *,
    live_planner: bool,
) -> tuple[list[dict], list[PlannedCandidate]]:
    """并发预检所有候选，并保持输入优先级顺序汇总结果。"""
    if live_planner:
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            planned = list(
                pool.map(
                    lambda candidate: plan_candidate(
                        candidate, planner_config, live_planner=True
                    ),
                    candidates,
                )
            )
    else:
        planned = [
            plan_candidate(candidate, planner_config, live_planner=False)
            for candidate in candidates
        ]

    reports: list[dict] = []
    reachable: list[PlannedCandidate] = []
    for result in planned:
        reports.append(result.report)
        if result.report["reachable"]:
            reachable.append(result)
            print(
                f"Preflight approach={result.candidate.approach_deg:.0f}deg "
                f"tilt={result.candidate.tilt_deg:.0f}deg: reachable (not executed)"
            )
        elif live_planner:
            print(
                f"Preflight approach={result.candidate.approach_deg:.0f}deg "
                f"tilt={result.candidate.tilt_deg:.0f}deg: unreachable: "
                f"{result.report['reason']}"
            )
    return reports, reachable


def wait_for_left_joint_target(
    arm_host: str,
    target_deg: list[float],
    label: str,
    tolerance_deg: float = 0.5,
    timeout_s: float = 90.0,
) -> None:
    """等待 Arm 实时反馈到达规划终点，防止异步执行尚未完成就进入下一步。"""
    from parm_sdk import Arm

    target = np.asarray(target_deg, dtype=float)
    deadline = time.monotonic() + timeout_s
    last_error = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = np.asarray(
                ((state.get("state") or {}).get("left") or {}).get("position") or [],
                dtype=float,
            )
            if actual.shape == (7,):
                delta = (actual - target + 180.0) % 360.0 - 180.0
                last_error = float(np.max(np.abs(delta)))
                if last_error <= tolerance_deg:
                    print(f"{label} reached; max joint error={last_error:.3f}deg")
                    return
            time.sleep(0.05)
    raise RuntimeError(
        f"{label} did not arrive within {timeout_s:.1f}s; "
        f"max joint error={last_error:.3f}deg"
    )


def trajectory_endpoint(plan: dict, label: str) -> list[float]:
    """从 Planner 返回中提取并校验七关节终点。"""
    trajectory = (plan.get("data") or {}).get("trajectory") or []
    if not trajectory or len(trajectory[-1]) != 7:
        raise RuntimeError(f"{label}没有返回七关节终点")
    return [float(value) for value in trajectory[-1]]


def close_left_gripper(gripper_host: str) -> None:
    """同步闭合左夹爪，并验证本次命令的权威终态。"""
    from parm_sdk import Gripper

    with Gripper(host=gripper_host, status_port=7702) as gripper:
        state = gripper.get_state().get(1)
        if not is_gripper_usable(state):
            raise RuntimeError("左夹爪不可用或未初始化")
        response = gripper.move(
            1,
            0.0,
            force=100.0,
            speed=80.0,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
    payload = response.get("data") or response
    device = (payload.get("devices") or {}).get("left_gripper") or {}
    device_state = str(device.get("device_state", "")).strip().lower()
    if response.get("status") != "success" or device_state not in {"arrived", "gripping"}:
        raise RuntimeError(f"左夹爪同步闭合响应无效: {response}")


def read_left_hold_state(gripper_host: str) -> tuple[bool, str, float | None]:
    """上提后读取夹爪接触状态和开度，返回是否可靠持件。"""
    from parm_sdk import Gripper

    with Gripper(host=gripper_host, status_port=7702) as gripper:
        state = gripper.get_state().get(1) or {}
    motion_state = str(state.get("motion_state", "")).strip().lower()
    try:
        position = float(state.get("position"))
    except (TypeError, ValueError):
        position = None
    held = (
        motion_state == "gripping"
        and position is not None
        and position > LEFT_GRIPPER_HELD_POSITION_MIN
    )
    return held, motion_state, position


def open_left_gripper(gripper_host: str) -> None:
    """抓空后同步张开左夹爪，为上层回预备位重试做准备。"""
    from parm_sdk import Gripper

    with Gripper(host=gripper_host, status_port=7702) as gripper:
        gripper.move(
            1,
            100.0,
            force=100.0,
            speed=80.0,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )


def execute_grasp_transaction(
    args: argparse.Namespace,
    planner: PlannerRuntime,
    selected: PlannedCandidate,
    timing: ExecutionTiming,
) -> None:
    """执行预抓取、下探、闭夹爪、上提和持件确认。"""
    candidate = selected.candidate
    if selected.pregrasp_plan is None:
        raise RuntimeError("选中的候选缺少预抓取规划")

    print(
        f"Selected table approach={candidate.approach_deg:.0f}deg "
        f"(side tilt={candidate.tilt_deg:.0f}deg); above target (body_base): "
        f"{candidate.body_pregrasp[:3, 3].round(4).tolist()}"
    )
    timing.start("pregrasp_motion")
    planner.execute_planned_result(selected.pregrasp_plan)
    wait_for_left_joint_target(
        args.arm_host,
        trajectory_endpoint(selected.pregrasp_plan, "预抓取规划"),
        "left pregrasp",
    )
    timing.finish("pregrasp_motion")

    print("At pregrasp; automatically descending with RRTstar.")
    timing.start("grasp_descent_plan_and_motion")
    grasp_plan = planner.plan_cartesian_pose(
        "left", object_pose(candidate.body_grasp), planner="rrtstar"
    )
    if grasp_plan is None:
        raise RuntimeError("抓取下探规划没有返回轨迹")
    planner.execute_planned_result(grasp_plan)
    wait_for_left_joint_target(
        args.arm_host,
        trajectory_endpoint(grasp_plan, "抓取下探规划"),
        "left grasp pose",
    )
    timing.finish("grasp_descent_plan_and_motion")

    print("At grasp pose; closing left gripper synchronously...")
    timing.start("gripper_close")
    close_left_gripper(args.gripper_host)
    timing.finish("gripper_close")

    # 必须先离开桌面再判断持件，避免桌面接触造成夹爪状态误判。
    timing.start("post_grasp_lift_and_hold_check")
    check_lift = candidate.body_grasp.copy()
    check_lift[2, 3] += POST_CLOSE_LIFT_M
    print("Check lift target (body_base):", check_lift[:3, 3].round(4).tolist())
    lift_plan = planner.plan_cartesian_pose(
        "left", object_pose(check_lift), planner="rrtstar"
    )
    if lift_plan is None:
        raise RuntimeError("抓取后上提规划没有返回轨迹")
    planner.execute_planned_result(lift_plan)
    wait_for_left_joint_target(
        args.arm_host,
        trajectory_endpoint(lift_plan, "抓取后上提规划"),
        "left 3cm post-grasp lift",
    )
    held, motion_state, position = read_left_hold_state(args.gripper_host)
    print(
        "[反抓夹爪] 上提3cm后状态: "
        f"motion_state={motion_state!r} position={position}; held={held} "
        f"(要求 gripping 且 position > {LEFT_GRIPPER_HELD_POSITION_MIN:.1f}%)",
        flush=True,
    )
    timing.finish("post_grasp_lift_and_hold_check")
    if not held:
        open_left_gripper(args.gripper_host)
        print(
            "NO_GRASP: 上提3cm后左夹爪未确认持件；已打开左夹爪，"
            "返回预备位后重拍。",
            flush=True,
        )
        raise SystemExit(NO_GRASP_EXIT_CODE)


def write_preflight_debug(
    grasp_json: Path,
    grasp_record: dict,
    reports: list[dict],
    body_z_down_m: float,
) -> None:
    """写出所有候选的坐标与规划结果，供现场排查。"""
    payload = {
        "approach_preflight": reports,
        "body_z_down_m": body_z_down_m,
        "planner_frame": "left_eef",
        "grasp_target_mode": "direct_urdf_left_eef",
        "grasp_frame_convention": grasp_record["grasp_frame_convention"],
        "pregrasp_planner_mode": "rrtstar",
        "grasp_descent_planner_mode": "rrtstar",
        "post_grasp_lift_planner_mode": "rrtstar",
    }
    path = grasp_json.parent / "planner_pose_debug.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Pose debug:", path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    grasp_record, grasp_camera = load_unique_grasp(args.grasp_json)
    body_head = load_body_head_transform(args.body_head)
    live_planner = args.execute or args.preflight_only
    planner_config = make_planner_config(args, live=live_planner)
    timing = ExecutionTiming()
    timing_path = args.timing_file or args.grasp_json.parent / "reverse_grasp_execution_timing.json"
    outcome = "FAILURE"
    planner = PlannerRuntime(planner_config)
    try:
        planner.open(required_frames={"body_base", "left_eef"})
        candidates = build_candidates(
            grasp_camera,
            body_head,
            args.approach_angles_deg,
            args.above_m,
            args.body_z_down_m,
        )
        timing.start("planner_preflight_parallel")
        reports, reachable = plan_all_candidates(
            candidates, planner_config, live_planner=live_planner
        )
        timing.finish("planner_preflight_parallel")
        write_preflight_debug(
            args.grasp_json, grasp_record, reports, args.body_z_down_m
        )
        if not reachable:
            raise RuntimeError(
                "执行被阻止：没有可达的接近角；请检查 planner_pose_debug.json"
            )

        selected = reachable[0]
        if args.preflight_only:
            outcome = "SUCCESS_PREFLIGHT"
            print("preflight complete; no trajectory was executed")
            return 0
        if not args.execute:
            outcome = "SUCCESS_DRY_RUN"
            print("dry-run complete; live Planner preflight was not requested")
            return 0

        execute_grasp_transaction(args, planner, selected, timing)
        outcome = "SUCCESS"
        print("grasp-and-lift complete (held confirmed by post-lift left gripper state)")
        return 0
    finally:
        planner.close()
        timing.write(timing_path, outcome)
        print(f"REVERSE_GRASP_EXECUTION_TIMING={timing_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
