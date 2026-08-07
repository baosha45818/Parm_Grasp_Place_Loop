#!/usr/bin/env python3
"""独立执行左臂正反两面放置，只消费 Module 2 Inspect 的 JSON 结果。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from loop_config import LoopConfig
from loop_types import ObjectPose
from modules.module_3_left_place.module_3_1_front_place.implementation import (
    PROFILE as FRONT_PROFILE,
)
from modules.module_3_left_place.module_3_3_reversed_place.implementation import (
    PROFILE as REVERSED_PROFILE,
)
from modules.module_3_left_place.placement_parameters import (
    DISCARD_BELOW_GRASPUP_M,
    GRASPUP_ABOVE_JOINTS_DEG,
    GRASPUP_ABOVE_PLANNING_TIMEOUT_S,
    GRASPUP_LIFT_M,
    GRASPUP_JOINTS_DEG,
)
from modules.module_3_left_place.placement_config import load_workpiece_parameters
from modules.stage_result import (
    read_stage_result,
    validate_inspection_success,
    write_stage_result,
)
from tool.planner_runtime import PlannerRuntime


WORKPIECE_PARAMETERS = {
    "1": load_workpiece_parameters("1"),
    "2": load_workpiece_parameters("2"),
}
WORKPIECE_1 = WORKPIECE_PARAMETERS["1"]

PLACE_PROFILES = {
    FRONT_PROFILE.classification: FRONT_PROFILE,
    REVERSED_PROFILE.classification: REVERSED_PROFILE,
}
GRIPPER_HOLD_STABLE_S = 0.5
GRIPPER_HOLD_POLL_S = 0.1
GRIPPER_HOLD_TIMEOUT_S = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workpiece",
        choices=("1", "2"),
        default="1",
        help="工件类型；默认1保持既有放置参数",
    )
    parser.add_argument("--inspection-result", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--gripper-hand", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--execute-url",
        default="http://127.0.0.1:5667/middle/waypoint/execute",
    )
    left_scale = float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0"))
    parser.add_argument("--preset-speed", type=float, default=0.2 * left_scale)
    parser.add_argument("--motion-speed", type=float, default=0.1 * left_scale)
    parser.add_argument(
        "--post-graspup-planner",
        choices=("rrtconnect", "rrtstar"),
        default="rrtstar",
        help="graspup 上方到工件目标的离线笛卡尔规划算法；默认 RRTstar。",
    )
    parser.add_argument("--offline-start-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--placement-planning-inflight",
        "--placement-planning-workers",
        dest="placement_planning_inflight",
        type=int,
        default=8,
        help=(
            "最终放置候选的最大在飞离线规划数；默认8（5604同时规划2条、"
            "最多预取6条），使用滑动窗口补充候选"
        ),
    )
    parser.add_argument("--graspup-lift-m", type=float, default=GRASPUP_LIFT_M)
    parser.add_argument(
        "--target-xy-radius-m",
        type=float,
        default=WORKPIECE_1.TARGET_XY_RADIUS_M,
    )
    parser.add_argument(
        "--branch1-front-target-object-xyz",
        "--front-target-object-xyz",
        dest="front_target_object_xyz",
        nargs=3,
        type=float,
        default=WORKPIECE_1.FRONT_TARGET_OBJECT_XYZ_M,
    )
    parser.add_argument(
        "--branch1-front-target-z-candidates-m",
        "--front-target-z-candidates-m",
        dest="front_target_z_candidates_m",
        nargs="+",
        type=float,
        default=WORKPIECE_1.FRONT_TARGET_Z_CANDIDATES_M,
    )
    parser.add_argument(
        "--branch1-front-release-object-z-candidates-m",
        "--front-release-object-z-candidates-m",
        dest="front_release_object_z_candidates_m",
        nargs="*",
        type=float,
        default=WORKPIECE_1.FRONT_RELEASE_Z_CANDIDATES_M,
    )
    parser.add_argument(
        "--branch1-reversed-target-object-xyz",
        "--reversed-target-object-xyz",
        dest="reversed_target_object_xyz",
        nargs=3,
        type=float,
        default=WORKPIECE_1.REVERSED_TARGET_OBJECT_XYZ_M,
    )
    parser.add_argument(
        "--branch1-reversed-target-z-candidates-m",
        "--reversed-target-z-candidates-m",
        dest="reversed_target_z_candidates_m",
        nargs="+",
        type=float,
        default=WORKPIECE_1.REVERSED_TARGET_Z_CANDIDATES_M,
    )
    parser.add_argument(
        "--branch1-reversed-release-object-z-candidates-m",
        "--reversed-release-object-z-candidates-m",
        dest="reversed_release_object_z_candidates_m",
        nargs="*",
        type=float,
        default=WORKPIECE_1.REVERSED_RELEASE_Z_CANDIDATES_M,
    )
    parser.add_argument("--weight-handoff-file", type=Path)
    parser.add_argument("--placement-ready-file", type=Path)
    parser.add_argument("--placement-permit-file", type=Path)
    parser.add_argument("--placement-permit-poll-s", type=float, default=0.2)
    parser.add_argument("--auto-confirm-motion", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    args = parser.parse_args()
    # 两个工件使用同一套参数覆盖规则；显式命令行值始终优先于参数文件。
    workpiece_parameters = WORKPIECE_PARAMETERS[args.workpiece]
    supplied = set(sys.argv[1:])

    def was_supplied(*options: str) -> bool:
        return any(
            token == option or token.startswith(f"{option}=")
            for token in supplied
            for option in options
        )

    defaults = (
        ("front_target_object_xyz", ("--branch1-front-target-object-xyz", "--front-target-object-xyz"), workpiece_parameters.FRONT_TARGET_OBJECT_XYZ_M),
        ("front_target_z_candidates_m", ("--branch1-front-target-z-candidates-m", "--front-target-z-candidates-m"), workpiece_parameters.FRONT_TARGET_Z_CANDIDATES_M),
        ("front_release_object_z_candidates_m", ("--branch1-front-release-object-z-candidates-m", "--front-release-object-z-candidates-m"), workpiece_parameters.FRONT_RELEASE_Z_CANDIDATES_M),
        ("reversed_target_object_xyz", ("--branch1-reversed-target-object-xyz", "--reversed-target-object-xyz"), workpiece_parameters.REVERSED_TARGET_OBJECT_XYZ_M),
        ("reversed_target_z_candidates_m", ("--branch1-reversed-target-z-candidates-m", "--reversed-target-z-candidates-m"), workpiece_parameters.REVERSED_TARGET_Z_CANDIDATES_M),
        ("reversed_release_object_z_candidates_m", ("--branch1-reversed-release-object-z-candidates-m", "--reversed-release-object-z-candidates-m"), workpiece_parameters.REVERSED_RELEASE_Z_CANDIDATES_M),
    )
    for attribute, options, value in defaults:
        if not was_supplied(*options):
            setattr(args, attribute, list(value))
    return args


def confirm_motion(args: argparse.Namespace, prompt: str) -> None:
    if args.auto_confirm_motion:
        print(f"{prompt}[自动确认]", flush=True)
        return
    input(prompt)


def matrix_pose(matrix: np.ndarray) -> ObjectPose:
    return ObjectPose(
        frame_id="body_base",
        position=np.asarray(matrix[:3, 3], dtype=float).tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def planner_matrix(response: dict) -> np.ndarray:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [translation.get(axis) for axis in "xyz"]
    xyzw = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in xyz + xyzw):
        raise RuntimeError(f"规划器 TF 格式无效: {response}")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    result[:3, 3] = xyz
    return result


def read_left_joints_deg(arm_host: str) -> np.ndarray:
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
    joints = np.asarray(
        ((state.get("state") or {}).get("left") or {}).get("position") or [],
        dtype=float,
    )
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"无法读取有效的左臂七关节状态: {state}")
    return joints


def wait_for_left_joint_target(
    arm_host: str,
    target_joints_deg: list[float],
    *,
    label: str,
    tolerance_deg: float = 0.5,
    timeout_s: float = 90.0,
) -> None:
    from parm_sdk import Arm

    target = np.asarray(target_joints_deg, dtype=float)
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = np.asarray(
                ((state.get("state") or {}).get("left") or {}).get("position") or [],
                dtype=float,
            )
            if actual.shape == (7,):
                delta = (actual - target + 180.0) % 360.0 - 180.0
                last_error_deg = float(np.max(np.abs(delta)))
                if last_error_deg <= tolerance_deg:
                    print(
                        f"[{label}] 左臂实际到位，最大关节误差="
                        f"{last_error_deg:.3f}deg",
                        flush=True,
                    )
                    return
            time.sleep(0.05)
    raise RuntimeError(
        f"左臂未在 {timeout_s:.1f}s 内到达{label}；"
        f"最后最大关节误差={last_error_deg:.3f}deg"
    )


def wait_for_left_graspup(arm_host: str) -> None:
    wait_for_left_joint_target(
        arm_host,
        GRASPUP_JOINTS_DEG,
        label="graspup",
    )


def validate_increasing_candidates(name: str, values: list[float]) -> list[float]:
    candidates = np.asarray(values, dtype=float)
    if (
        candidates.size == 0
        or not np.all(np.isfinite(candidates))
        or np.any(np.diff(candidates) <= 0.0)
    ):
        raise ValueError(f"{name} 必须是严格递增的有限数值")
    return candidates.tolist()


def validate_optional_increasing_candidates(name: str, values: list[float]) -> list[float]:
    """空 release 列表表示在 target 位直接松爪，不存在第二段 LIN。"""
    if not values:
        return []
    return validate_increasing_candidates(name, values)


def workpiece_object_target(
    workpiece_parameters,
    xyz: np.ndarray,
    classification: str,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """按工件参数文件定义，生成正面或反面的目标工件位姿。"""
    base_rpy_deg = np.asarray(
        workpiece_parameters.REVERSED_BASE_RPY_DEG
        if classification == "REVERSED"
        else workpiece_parameters.FRONT_BASE_RPY_DEG,
        dtype=float,
    )
    if base_rpy_deg.shape != (3,) or not np.all(np.isfinite(base_rpy_deg)):
        raise ValueError(f"{classification} 基准 RPY 必须是三个有限角度: {base_rpy_deg}")
    if workpiece_parameters.ROTATION_CONVENTION == "body_base_yaw_then_local_x":
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler(
            "XYZ", base_rpy_deg, degrees=True
        )
    elif workpiece_parameters.ROTATION_CONVENTION == "body_base_yaw_then_local_rp":
        # Keep the original body_base yaw convention, then add roll/pitch in
        # the already oriented workpiece frame.  Zero roll/pitch is exactly
        # the legacy ``body_base_yaw_then_local_x`` pose.
        rotation = (
            Rotation.from_euler("z", yaw_deg, degrees=True)
            * Rotation.from_euler("XYZ", base_rpy_deg, degrees=True)
            * Rotation.from_euler("XY", [roll_deg, pitch_deg], degrees=True)
        )
    elif workpiece_parameters.ROTATION_CONVENTION == "local_rpy":
        rotation = Rotation.from_euler("XYZ", base_rpy_deg, degrees=True) * Rotation.from_euler(
            "XYZ", [roll_deg, pitch_deg, yaw_deg], degrees=True
        )
    else:
        raise RuntimeError(
            "未知工件放置旋转定义: "
            f"{workpiece_parameters.ROTATION_CONVENTION!r}"
        )
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = xyz
    return result


def tool_positive_z_points_toward_body_base_negative_z(
    T_body_tool: np.ndarray,
) -> bool:
    """仅接受末端局部 +Z 在 body_base Z 方向上朝下的目标。"""
    return float(T_body_tool[2, 2]) < 0.0


def resolve_place_profile(classification: str, branch: str):
    profile = PLACE_PROFILES.get(classification)
    if profile is None:
        raise RuntimeError(f"没有对应的左放子模块: {classification!r}")
    if branch != profile.placement_branch:
        raise RuntimeError(
            f"Inspect 分类与放置分支不一致: classification={classification}, "
            f"branch={branch}, expected={profile.placement_branch}"
        )
    return profile


def open_left_gripper(args: argparse.Namespace) -> float | None:
    from parm_sdk import Gripper
    from modules.module_0_prepare.gripper_state_compat import is_gripper_usable

    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        hand_state = gripper.get_state().get(args.gripper_hand)
        if hand_state is None:
            raise RuntimeError(
                f"左夹爪 hand={args.gripper_hand} 没有同步状态或未连接"
            )
        if not is_gripper_usable(hand_state):
            raise RuntimeError("左夹爪尚未初始化，禁止发送打开指令")
        response = gripper.move(
            args.gripper_hand,
            100.0,
            force=100.0,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
    # 新夹爪驱动把同步终态放在 envelope.devices[device_id]；保留旧 data
    # 位置读取作为兼容回退，保证放置结果记录的最终开度不是 None。
    payload = response.get("data") or response
    devices = payload.get("devices") or {}
    device_id = "left_gripper" if args.gripper_hand == 1 else "right_gripper"
    interfaces = (devices.get(device_id) or {}).get("state_interfaces") or {}
    position = interfaces.get("gripper/position")
    if position is None:
        position = payload.get("position")
    return None if position is None else float(position)


def publish_ready(
    args: argparse.Namespace,
    payload: dict,
) -> None:
    if args.placement_ready_file is not None:
        write_stage_result(args.placement_ready_file, payload)


def wait_permit(args: argparse.Namespace) -> None:
    if args.placement_permit_file is None:
        return
    print(f"[left-place] 等待 Buffer 许可: {args.placement_permit_file}", flush=True)
    while not args.placement_permit_file.is_file():
        time.sleep(float(args.placement_permit_poll_s))
    content = args.placement_permit_file.read_text(encoding="utf-8").strip().lower()
    if content not in {"allow", "y", "yes", "1"}:
        raise RuntimeError(f"放置许可内容无效: {content!r}")


def main() -> int:
    args = parse_args()
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    # Keep the two placement motion records separate from inspection and all
    # generic planner diagnostics, even when this module is launched by a wrapper.
    os.environ["PARM_PLANNING_RECORD_DIR"] = str(
        ROOT / "output" / "left_place_trajectories" / time.strftime("%Y%m%d_%H%M%S")
    )
    placement_trace_path = args.result_file.parent / "placement_planning_attempts.jsonl"

    def append_placement_trace(record: dict) -> None:
        """Keep object placement targets beside the generic PlannerRuntime joint traces."""
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "planning_scheduler": planning_scheduler_log,
            **record,
        }
        payload.setdefault("from", {
            "source": "offline_graspup_above_start_state",
            "joint_angles_deg": payload.get("planning_start_joints_deg"),
        })
        payload.setdefault("to", {
            "kind": "object_placement_point",
            **(payload.get("placement_point") or {}),
        })
        payload.setdefault("logical_motion", {
            "from": "left_graspup_top",
            "to": (
                "left_release_position"
                if payload.get("record_type") == "release_descent"
                else "left_place_position"
            ),
        })
        with placement_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if (args.placement_ready_file is None) != (args.placement_permit_file is None):
        raise ValueError("placement ready/permit 文件必须同时提供")
    if not np.isfinite(args.graspup_lift_m) or not np.isclose(
        args.graspup_lift_m, GRASPUP_LIFT_M, atol=1e-9
    ):
        raise ValueError("--graspup-lift-m 当前必须为原版固定值 0.10m")
    if not np.isfinite(args.target_xy_radius_m) or args.target_xy_radius_m < 0.0:
        raise ValueError("--target-xy-radius-m 必须是非负的有限值")
    if args.offline_start_tolerance_deg <= 0.0:
        raise ValueError("--offline-start-tolerance-deg 必须大于0")
    if not 1 <= args.placement_planning_inflight <= 16:
        raise ValueError(
            "--placement-planning-inflight 必须在[1, 16]内；"
            "默认8条候选在飞请求，由 RRTstar 队列依次处理"
        )

    inspection = read_stage_result(
        args.inspection_result,
        expected_stage="left_inspect",
    )
    if inspection["outcome"] != "success":
        forwarded = {
            "stage": "left_place",
            "outcome": inspection["outcome"],
            "placement_branch": None,
            "source_inspection": str(args.inspection_result),
        }
        write_stage_result(args.result_file, forwarded)
        publish_ready(args, {
            "stage": "left_place_ready",
            **{key: value for key, value in forwarded.items() if key != "stage"},
            "result_path": str(args.result_file),
        })
        return 0

    validate_inspection_success(inspection)
    # 两个工件均
    # 使用工件 +Z 相对 body_base 正负Z的夹角来决定正反候选的优先顺序。
    classification = "FRONT"
    branch = FRONT_PROFILE.placement_branch
    profile = FRONT_PROFILE
    T_tool_object = np.asarray(inspection["T_tool_object"], dtype=float)
    post_graspup_planner = {
        "rrtconnect": "ompl",
        "rrtstar": "rrtstar",
    }[args.post_graspup_planner]
    planning_scheduler_log = {
        "mode": "sliding_window",
        "max_inflight": int(args.placement_planning_inflight),
        "planner": post_graspup_planner,
        "offline_start_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
        "face_priority_policy": "body_base_z_angle_selects_front_reversed_or_alternating",
    }
    tool_z_rejected_count = 0

    planner_cfg = LoopConfig(
        planner_host=args.planner_host,
        execute_url=args.execute_url,
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.motion_speed,
        max_acceleration=args.motion_speed,
        idle_poll_seconds=0.2,
        live=not args.skip_motion,
        poses={},
    )
    planner = PlannerRuntime(planner_cfg)
    weight_handoff = None
    holding_checks: list[dict] = []
    object_dropped = threading.Event()
    drop_details: list[dict] = []

    if args.weight_handoff_file is not None:
        weight_payload = json.loads(
            args.weight_handoff_file.read_text(encoding="utf-8")
        )
    else:
        weight_payload = inspection.get("graspnet_weight_handoff")
    if weight_payload is not None:
        if weight_payload.get("weight_check_enabled") is not True:
            raise RuntimeError("GraspNet handoff 未启用重量检查")
        weight_handoff = weight_payload.get("weight_check")
        if not isinstance(weight_handoff, dict):
            raise RuntimeError("GraspNet handoff 缺少首次重量检查数据")
        workpiece_parameters = WORKPIECE_PARAMETERS[args.workpiece]
        allowed_handoff_classes = set(workpiece_parameters.ALLOWED_HANDOFF_WEIGHT_CLASSES)
        if weight_handoff.get("classification") not in allowed_handoff_classes:
            raise RuntimeError(
                "首次重量判断不在允许集合 "
                f"{sorted(allowed_handoff_classes)}，禁止执行 Left Place"
            )
    def gripper_holds_object(label: str) -> bool:
        if args.skip_motion:
            return True
        if object_dropped.is_set():
            return False
        try:
            from parm_sdk import Gripper

            with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                deadline = time.monotonic() + GRIPPER_HOLD_TIMEOUT_S
                stable_since: float | None = None
                stable_samples = 0
                state = None
                while time.monotonic() < deadline:
                    candidate = gripper.get_state().get(args.gripper_hand)
                    if not isinstance(candidate, dict):
                        stable_since = None
                        stable_samples = 0
                    else:
                        state = candidate
                        motion_state = str(candidate.get("motion_state", "")).lower()
                        if motion_state == "gripping":
                            stable_since = stable_since or time.monotonic()
                            stable_samples += 1
                            if time.monotonic() - stable_since >= GRIPPER_HOLD_STABLE_S:
                                record = {
                                    "before_motion": label,
                                    "source": "gripper_state",
                                    "motion_state": motion_state,
                                    "position_percent": candidate.get("position"),
                                    "initialized": candidate.get("initialized"),
                                    "stable_duration_s": GRIPPER_HOLD_STABLE_S,
                                    "stable_samples": stable_samples,
                                    "passed": True,
                                }
                                break
                        else:
                            stable_since = None
                            stable_samples = 0
                    time.sleep(GRIPPER_HOLD_POLL_S)
                else:
                    record = None
            if record is not None:
                holding_checks.append(record)
                print(
                    f"[运动前持件复检] {label}: "
                    f"motion_state={record.get('motion_state')} passed=True",
                    flush=True,
                )
                return True
            if not isinstance(state, dict):
                raise RuntimeError(f"未收到 hand={args.gripper_hand} 夹爪状态: {state}")
            motion_state = state.get("motion_state")
            record = {
                "before_motion": label,
                "source": "gripper_state",
                "motion_state": motion_state,
                "position_percent": state.get("position"),
                "initialized": state.get("initialized"),
                "stable_duration_s": 0.0,
                "stable_samples": stable_samples,
                "passed": False,
            }
        except Exception as exc:
            record = {
                "before_motion": label,
                "source": "gripper_state",
                "motion_state": None,
                "reason": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        holding_checks.append(record)
        print(
            f"[运动前持件复检] {label}: "
            f"motion_state={record.get('motion_state')} passed={record['passed']}",
            flush=True,
        )
        if not record["passed"]:
            drop_details.append(record)
            object_dropped.set()
            print("[工件掉落] 左夹爪未报告 gripping，禁止当前及后续持件运动", flush=True)
            return False
        return True

    def write_terminal_outcome(outcome: str, **extra) -> None:
        payload = {
            "stage": "left_place",
            "outcome": outcome,
            "place_module": profile.module_id,
            "classification": classification,
            "placement_branch": branch,
            "holding_checks": holding_checks,
            "planning_scheduler": planning_scheduler_log,
            **extra,
        }
        write_stage_result(args.result_file, payload)
        publish_ready(args, {
            "stage": "left_place_ready",
            "outcome": outcome,
            "place_module": profile.module_id,
            "classification": classification,
            "placement_branch": branch,
            "result_path": str(args.result_file),
        })

    def finish_dropped_cycle() -> None:
        print("[工件掉落恢复] 正在回 graspup 并打开左夹爪", flush=True)
        planner.config = replace(
            planner.config,
            max_velocity=float(args.preset_speed),
            max_acceleration=float(args.preset_speed),
        )
        planner.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
        wait_for_left_graspup(args.arm_host)
        final_position = open_left_gripper(args)
        write_terminal_outcome(
            "held_object_dropped",
            completed_without_error=True,
            drop_detected_before_motion=(
                drop_details[-1]["before_motion"] if drop_details else None
            ),
            returned_to_graspup=True,
            gripper_final_position_percent=final_position,
        )
        print("[工件掉落恢复完成] 本轮正常结束", flush=True)

    approach_thread = None
    timeout_thread = None
    approach_errors: list[BaseException] = []
    timeout_errors: list[BaseException] = []

    try:
        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        if not args.skip_motion:
            # Module 1 已完成同姿态重量分类。Module 3 的任意姿态只判断
            # 新夹爪驱动是否仍报告 gripping，不做跨姿态力传感器估重。
            if not gripper_holds_object("Module 3开始（graspup）"):
                finish_dropped_cycle()
                return 0

        workpiece_parameters = WORKPIECE_PARAMETERS[args.workpiece]
        T_body_object = np.asarray(inspection["T_body_base_object"], dtype=float)
        object_z_body = T_body_object[:3, 2]
        object_z_body /= np.linalg.norm(object_z_body)
        positive_angle = float(np.degrees(np.arccos(np.clip(object_z_body[2], -1.0, 1.0))))
        negative_angle = float(np.degrees(np.arccos(np.clip(-object_z_body[2], -1.0, 1.0))))
        threshold = float(workpiece_parameters.BODY_BASE_Z_FACE_SELECTION_MAX_ANGLE_DEG)
        if positive_angle < threshold:
            face_branch_mode = "front_then_reversed"
        elif negative_angle < threshold:
            face_branch_mode = "reversed_then_front"
        else:
            face_branch_mode = "alternating"
        face_specs = [
            {
                "classification": "FRONT",
                "branch": "BRANCH_1_FRONT",
                "profile": FRONT_PROFILE,
                "center": np.asarray(args.front_target_object_xyz, dtype=float),
                "target_z": validate_increasing_candidates(
                    "--front-target-z-candidates-m", args.front_target_z_candidates_m,
                ),
                "release_z": validate_optional_increasing_candidates(
                    "--front-release-object-z-candidates-m", args.front_release_object_z_candidates_m,
                ),
                "yaw": list(workpiece_parameters.FRONT_YAW_CANDIDATES_DEG),
                "roll": list(workpiece_parameters.FRONT_ROLL_CANDIDATES_DEG),
                "pitch": list(workpiece_parameters.FRONT_PITCH_CANDIDATES_DEG),
            },
            {
                "classification": "REVERSED",
                "branch": "BRANCH_1_REVERSED",
                "profile": REVERSED_PROFILE,
                "center": np.asarray(args.reversed_target_object_xyz, dtype=float),
                "target_z": validate_increasing_candidates(
                    "--reversed-target-z-candidates-m", args.reversed_target_z_candidates_m,
                ),
                "release_z": validate_optional_increasing_candidates(
                    "--reversed-release-object-z-candidates-m", args.reversed_release_object_z_candidates_m,
                ),
                "yaw": list(workpiece_parameters.REVERSED_YAW_CANDIDATES_DEG),
                "roll": list(workpiece_parameters.REVERSED_ROLL_CANDIDATES_DEG),
                "pitch": list(workpiece_parameters.REVERSED_PITCH_CANDIDATES_DEG),
            },
        ]
        if any(spec["center"].shape != (3,) or not np.all(np.isfinite(spec["center"])) for spec in face_specs):
            raise ValueError("目标工件 XYZ 必须包含三个有限米制数值")
        print(
            f"[工件{args.workpiece}候选] +Z->body_base+Z={positive_angle:.1f}deg, "
            f"+Z->body_base-Z={negative_angle:.1f}deg, mode={face_branch_mode}; "
            "正反面均会纳入规划。",
            flush=True,
        )
        planning_attempts: list[dict] = []
        planning_submission_count = 0
        planning_completion_count = 0
        selected = None
        selected_plan = None

        approach_reached = threading.Event()
        approach_finished = threading.Event()
        plan_found = threading.Event()
        planning_finished = threading.Event()
        planning_timed_out = threading.Event()

        if not args.skip_motion:
            confirm_motion(
                args,
                "确认回 graspup 并沿 body_base +Z 上移0.100m路径安全，"
                "按回车开始边运动边离线规划...",
            )

            def move_to_graspup_above() -> None:
                motion_planner = PlannerRuntime(planner_cfg)
                try:
                    motion_planner.open(
                        required_frames={"body_base", "left_hand_transfer_end"}
                    )
                    motion_planner.config = replace(
                        motion_planner.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    if not gripper_holds_object("从观察位回graspup"):
                        return
                    motion_planner.set_next_plan_context(
                        "left_inspection_pose", "left_graspup"
                    )
                    motion_planner.move_left_joints(GRASPUP_JOINTS_DEG)
                    wait_for_left_graspup(args.arm_host)
                    response = motion_planner.lookup_transform(
                        "body_base",
                        "left_hand_transfer_end",
                    )
                    if response is None:
                        raise RuntimeError("规划器没有返回 graspup 末端位姿")
                    above = planner_matrix(response)
                    above[2, 3] += float(args.graspup_lift_m)
                    motion_planner.config = replace(
                        motion_planner.config,
                        max_velocity=float(args.motion_speed),
                        max_acceleration=float(args.motion_speed),
                    )
                    if not gripper_holds_object("从graspup移动到graspup_above"):
                        return
                    motion_planner.set_next_plan_context(
                        "left_graspup", "left_graspup_top"
                    )
                    print(
                        "[运动线程] 正在规划 graspup -> graspup_above 的 LIN 轨迹",
                        flush=True,
                    )
                    above_plan = motion_planner.plan_cartesian_pose(
                        "left",
                        matrix_pose(above),
                        planner="lin",
                    )
                    if above_plan is None:
                        raise RuntimeError("graspup_above LIN 规划返回空结果")
                    above_trajectory = (
                        (above_plan.get("data") or {}).get("trajectory") or []
                    )
                    if not above_trajectory or len(above_trajectory[-1]) != 7:
                        raise RuntimeError(
                            "graspup_above LIN 规划没有返回七关节终点"
                        )
                    above_endpoint = [
                        float(value) for value in above_trajectory[-1]
                    ]
                    print(
                        "[运动线程] 开始执行 graspup -> graspup_above LIN；"
                        f"终点关节={np.round(above_endpoint, 4).tolist()}",
                        flush=True,
                    )
                    motion_planner.execute_planned_result(above_plan)
                    approach_reached.set()
                    print(
                        "[运动线程] graspup -> graspup_above LIN 已同步执行完成；"
                        f"规划剩余等待上限={GRASPUP_ABOVE_PLANNING_TIMEOUT_S:.1f}s",
                        flush=True,
                    )
                except BaseException as exc:
                    approach_errors.append(exc)
                finally:
                    approach_finished.set()
                    motion_planner.close()

            approach_thread = threading.Thread(
                target=move_to_graspup_above,
                name="module3-graspup-above-motion",
                daemon=False,
            )
            approach_thread.start()

            def recover_after_timeout() -> None:
                while not approach_finished.is_set():
                    if approach_reached.wait(timeout=0.05):
                        break
                if not approach_reached.is_set():
                    return
                deadline = (
                    time.monotonic() + GRASPUP_ABOVE_PLANNING_TIMEOUT_S
                )
                while not plan_found.is_set():
                    if planning_finished.is_set():
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    plan_found.wait(timeout=min(0.05, remaining))
                if plan_found.is_set():
                    return
                planning_timed_out.set()
                print(
                    "[规划超时] 到达 graspup 上方后10.0s仍无可行解；"
                    "正在执行原版丢弃恢复",
                    flush=True,
                )
                recovery = PlannerRuntime(planner_cfg)
                try:
                    recovery.open(
                        required_frames={"body_base", "left_hand_transfer_end"}
                    )
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    recovery.set_next_plan_context(
                        "left_graspup_top", "left_graspup"
                    )
                    recovery.move_left_joints(
                        GRASPUP_JOINTS_DEG,
                        planner="rrtstar",
                    )
                    wait_for_left_graspup(args.arm_host)
                    response = recovery.lookup_transform(
                        "body_base",
                        "left_hand_transfer_end",
                    )
                    if response is None:
                        raise RuntimeError("规划器没有返回 graspup 末端位姿")
                    discard = planner_matrix(response)
                    discard[2, 3] -= DISCARD_BELOW_GRASPUP_M
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.motion_speed),
                        max_acceleration=float(args.motion_speed),
                    )
                    recovery.set_next_plan_context(
                        "left_graspup", "left_timeout_discard_position"
                    )
                    recovery.move_cartesian_pose(
                        "left",
                        matrix_pose(discard),
                        planner="lin",
                    )
                    open_left_gripper(args)
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    recovery.set_next_plan_context(
                        "left_timeout_discard_position", "left_graspup"
                    )
                    recovery.move_left_joints(
                        GRASPUP_JOINTS_DEG,
                        planner="rrtstar",
                    )
                    wait_for_left_graspup(args.arm_host)
                except BaseException as exc:
                    timeout_errors.append(exc)
                finally:
                    recovery.close()

            timeout_thread = threading.Thread(
                target=recover_after_timeout,
                name="module3-planning-timeout-recovery",
                daemon=False,
            )
            timeout_thread.start()

        print(
            "[离线规划] start_state=GRASPUP_ABOVE_JOINTS_DEG="
            f"{GRASPUP_ABOVE_JOINTS_DEG}",
            flush=True,
        )
        print(
            "[离线规划并发] 最多"
            f"{args.placement_planning_inflight}条在飞请求；"
            "完成一条失败候选就立刻补发下一条",
            flush=True,
        )

        def plan_offline_candidate(target_tool: np.ndarray) -> dict:
            # Separate SDK connections are required: one PlannerRuntime would
            # serialize requests behind the SDK client's internal lock.
            candidate_planner = PlannerRuntime(planner_cfg)
            try:
                candidate_planner.open(
                    required_frames={"body_base", "left_hand_transfer_end"}
                )
                response = candidate_planner.plan_cartesian_pose(
                    "left",
                    matrix_pose(target_tool),
                    planner=post_graspup_planner,
                    start_joints_deg=GRASPUP_ABOVE_JOINTS_DEG,
                    rrtstar_goal_tolerance=(post_graspup_planner == "rrtstar"),
                )
                if response is None:
                    raise RuntimeError("离线规划返回空结果")
                return response
            finally:
                candidate_planner.close()

        try:
            # T_tool_object is already final when Module 2 succeeds, while
            # the offline start is the fixed GRASPUP_ABOVE_JOINTS_DEG. Submit
            # candidate plans immediately and overlap them with PC -> graspup
            # -> graspup_top motion. Execution still waits for the motion and
            # validates the planned trajectory's first joint state.
            radius_layers = tuple(
                float(args.target_xy_radius_m) * float(factor)
                for factor in workpiece_parameters.XY_RADIUS_FACTORS
            )
            for radius_index, radius in enumerate(radius_layers):
                candidates = []
                ordered_by_branch = []
                for spec in face_specs:
                    branch_candidates = []
                    if radius == 0.0:
                        face_samples = [{
                            "name": f"{spec['classification'].lower()}_center",
                            "radius_m": 0.0,
                            "direction_deg": None,
                            "xy": spec["center"][:2].copy(),
                        }]
                    else:
                        face_samples = []
                        for direction in range(
                            0, 360, int(workpiece_parameters.XY_SAMPLE_DIRECTION_STEP_DEG)
                        ):
                            angle = math.radians(direction)
                            face_samples.append({
                                "name": f"{spec['classification'].lower()}_r{radius:.3f}_{direction}deg",
                                "radius_m": radius,
                                "direction_deg": direction,
                                "xy": spec["center"][:2] + radius * np.array(
                                    [math.cos(angle), math.sin(angle)]
                                ),
                            })
                    for sample in face_samples:
                        for z_index, target_z in enumerate(spec["target_z"]):
                            for yaw_index, yaw in enumerate(spec["yaw"]):
                                for roll_index, roll in enumerate(spec["roll"]):
                                    for pitch_index, pitch in enumerate(spec["pitch"]):
                                        branch_candidates.append((
                                            spec, sample, z_index, float(target_z),
                                            yaw_index, roll_index, pitch_index,
                                            float(roll), float(pitch), float(yaw),
                                        ))
                    branch_candidates.sort(
                        key=lambda item: (
                            item[5] + item[6], item[5], item[6],
                            item[4], item[2], str(item[1]["name"]),
                        )
                    )
                    ordered_by_branch.append(branch_candidates)
                front_candidates, reversed_candidates = ordered_by_branch
                if face_branch_mode == "front_then_reversed":
                    candidates = front_candidates + reversed_candidates
                elif face_branch_mode == "reversed_then_front":
                    candidates = reversed_candidates + front_candidates
                else:
                    for index in range(max(len(front_candidates), len(reversed_candidates))):
                        if index < len(front_candidates):
                            candidates.append(front_candidates[index])
                        if index < len(reversed_candidates):
                            candidates.append(reversed_candidates[index])
                print(
                    f"[候选层] radius={radius:.3f}m，共{len(candidates)}项",
                    flush=True,
                )
                next_candidate_index = 0
                in_flight = {}
                layer_successes = []
                with ThreadPoolExecutor(
                    max_workers=int(args.placement_planning_inflight),
                    thread_name_prefix="module3-placement-plan",
                ) as planning_executor:
                    while in_flight or next_candidate_index < len(candidates):
                        while (
                            not layer_successes
                            and len(in_flight) < int(args.placement_planning_inflight)
                            and next_candidate_index < len(candidates)
                            and not planning_timed_out.is_set()
                            and not object_dropped.is_set()
                        ):
                            candidate = candidates[next_candidate_index]
                            (spec, sample, z_index, target_z, yaw_index,
                             roll_index, pitch_index, roll, pitch, yaw) = candidate
                            rotation_index = roll_index
                            base_rpy_deg = list(
                                workpiece_parameters.REVERSED_BASE_RPY_DEG
                                if spec["classification"] == "REVERSED"
                                else workpiece_parameters.FRONT_BASE_RPY_DEG
                            )
                            candidate_classification = str(spec["classification"])
                            candidate_branch = str(spec["branch"])
                            candidate_profile = spec["profile"]
                            candidate_release_z = list(spec["release_z"])
                            xyz = np.array(
                                [sample["xy"][0], sample["xy"][1], target_z],
                                dtype=float,
                            )
                            target_object = workpiece_object_target(
                                workpiece_parameters,
                                xyz,
                                candidate_classification,
                                roll,
                                pitch,
                                yaw,
                            )
                            target_tool = target_object @ np.linalg.inv(T_tool_object)
                            attempt = {
                                "place_module": candidate_profile.module_id,
                                "classification": candidate_classification,
                                "placement_branch": candidate_branch,
                                "release_z_candidates_m": candidate_release_z,
                                "priority_index": next_candidate_index,
                                "xy_layer": sample["name"],
                                "xy_radius_m": radius,
                                "xy_direction_deg": sample["direction_deg"],
                                "z_index": z_index,
                                "object_z_m": target_z,
                                "yaw_deg": yaw,
                                "rotation_index": rotation_index,
                                "object_base_rpy_deg": base_rpy_deg,
                                "object_local_roll_deg": roll,
                                "object_local_roll_index": roll_index,
                                "object_local_pitch_deg": pitch,
                                "object_local_pitch_index": pitch_index,
                                "target_object_xyz": xyz.tolist(),
                                "target_tool_xyz": target_tool[:3, 3].tolist(),
                                "placement_point": {
                                    "frame_id": "body_base",
                                    "object_xyz_m": xyz.tolist(),
                                    "tool_xyz_m": target_tool[:3, 3].tolist(),
                                },
                                "planning_start_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
                                "planning_submission_order": planning_submission_count + 1,
                                "planning_inflight_after_submit": len(in_flight) + 1,
                                "target_object": target_object,
                                "target_tool": target_tool,
                            }
                            if not tool_positive_z_points_toward_body_base_negative_z(
                                target_tool
                            ):
                                tool_z_rejected_count += 1
                                rejected_attempt = {
                                    key: value for key, value in attempt.items()
                                    if key not in {"target_object", "target_tool"}
                                }
                                rejected_attempt.pop("planning_submission_order")
                                rejected_attempt.pop("planning_inflight_after_submit")
                                rejected_attempt.update({
                                    "planned": False,
                                    "rejected_before_planning": True,
                                    "planning_inflight_at_rejection": len(in_flight),
                                    "error": (
                                        "目标末端约束不满足："
                                        "left_hand_transfer_end +Z 在 body_base 的 "
                                        "Z 投影必须为负"
                                    ),
                                    "target_tool_positive_z_in_body_base": (
                                        target_tool[:3, 2].tolist()
                                    ),
                                    "target_tool_positive_z_body_base_z_component": (
                                        float(target_tool[2, 2])
                                    ),
                                })
                                planning_attempts.append(rejected_attempt)
                                append_placement_trace(rejected_attempt)
                                print(
                                    "[跳过放置候选] "
                                    "left_hand_transfer_end +Z="
                                    f"{np.round(target_tool[:3, 2], 6).tolist()}, "
                                    f"body_base Z投影={target_tool[2, 2]:+.6f}（必须<0）",
                                    flush=True,
                                )
                                next_candidate_index += 1
                                continue
                            future = planning_executor.submit(
                                plan_offline_candidate, target_tool
                            )
                            in_flight[future] = attempt
                            planning_submission_count += 1
                            print(
                                f"[提交 {next_candidate_index + 1}/{len(candidates)}] "
                                f"radius={radius:.3f} sample={sample['name']} "
                                f"z={target_z:.3f} yaw={yaw} base_rpy={base_rpy_deg} "
                                f"roll={roll:+.1f} pitch={pitch:+.1f}; "
                                f"当前在飞={len(in_flight)}",
                                flush=True,
                            )
                            next_candidate_index += 1

                        if not in_flight:
                            break
                        future = next(as_completed(in_flight))
                        attempt = in_flight.pop(future)
                        planning_completion_count += 1
                        try:
                            response = future.result()
                        except Exception as exc:
                            failed_attempt = {
                                key: value for key, value in attempt.items()
                                if key not in {"target_object", "target_tool"}
                            }
                            failed_attempt.update({
                                "planned": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "planning_completion_order": planning_completion_count,
                                "planning_inflight_after_complete": len(in_flight),
                            })
                            planning_attempts.append(failed_attempt)
                            append_placement_trace(failed_attempt)
                            continue

                        successful = {
                            key: value for key, value in attempt.items()
                            if key not in {"target_object", "target_tool"}
                        }
                        target_object = attempt["target_object"]
                        target_tool = attempt["target_tool"]
                        successful.update({
                            "planned": True,
                            "planning_completion_order": planning_completion_count,
                            "planning_inflight_after_complete": len(in_flight),
                            "radius_layer_index": radius_index,
                            "target_object_pose": {
                                "position": target_object[:3, 3].tolist(),
                                "orientation_xyzw": Rotation.from_matrix(
                                    target_object[:3, :3]
                                ).as_quat().tolist(),
                            },
                            "target_tool_pose": {
                                "position": target_tool[:3, 3].tolist(),
                                "orientation_xyzw": Rotation.from_matrix(
                                    target_tool[:3, :3]
                                ).as_quat().tolist(),
                            },
                        })
                        data = response.get("data") or {}
                        trajectory = data.get("trajectory") or []
                        successful["planned_trajectory_joint_deg"] = trajectory
                        successful["planned_start_joints_deg"] = (
                            trajectory[0] if trajectory else GRASPUP_ABOVE_JOINTS_DEG
                        )
                        successful["planned_end_joints_deg"] = trajectory[-1] if trajectory else None
                        successful["sample_frequency_hz"] = data.get("sample_frequency")
                        planning_attempts.append(successful)
                        append_placement_trace(successful)
                        layer_successes.append((successful, response))
                        plan_found.set()
                        print(
                            "[找到可行候选] "
                            f"radius={radius:.3f} sample={successful['xy_layer']} "
                            f"z={successful['object_z_m']:.3f} "
                            f"yaw={successful['yaw_deg']} "
                            f"base_rpy={successful['object_base_rpy_deg']}; "
                            "停止补发更低优先级候选",
                            flush=True,
                        )

                if layer_successes:
                    # 正反两面共用滑动窗口：最先返回的可行解立即被选中，
                    # 后续不再提交新的候选；已在飞的请求不影响已选结果。
                    selected, selected_plan = layer_successes[0]
                    classification = str(selected["classification"])
                    branch = str(selected["placement_branch"])
                    profile = resolve_place_profile(classification, branch)
                    release_z_candidates = list(selected["release_z_candidates_m"])
                if selected is not None or planning_timed_out.is_set():
                    break
        finally:
            planning_finished.set()

        if approach_thread is not None:
            approach_thread.join()
        if timeout_thread is not None:
            timeout_thread.join()
        if approach_errors:
            raise RuntimeError(
                f"运动到 graspup 上方失败: {approach_errors[0]}"
            ) from approach_errors[0]
        if timeout_errors:
            raise RuntimeError(
                f"规划超时恢复失败: {timeout_errors[0]}"
            ) from timeout_errors[0]
        if object_dropped.is_set():
            finish_dropped_cycle()
            return 0
        if planning_timed_out.is_set():
            write_terminal_outcome(
                "planning_timeout_recovered_at_graspup",
                completed_without_error=True,
                planning_attempts=planning_attempts,
                returned_to_graspup=True,
            )
            return 0
        if selected is None:
            if tool_z_rejected_count and planning_submission_count == 0:
                raise RuntimeError(
                    "全部放置候选均不满足末端朝下约束："
                    "left_hand_transfer_end +Z 在 body_base 的 Z 投影必须<0"
                )
            raise RuntimeError("全部 XY/Z/yaw/局部X候选均规划失败")

        planning_mode = "dry_run"
        if not args.skip_motion:
            if selected_plan is None:
                raise RuntimeError("离线规划结果为空")
            trajectory = ((selected_plan.get("data") or {}).get("trajectory") or [])
            if not trajectory:
                raise RuntimeError("离线规划结果缺少可执行轨迹")
            offline_start = np.asarray(trajectory[0], dtype=float)
            if offline_start.shape != (7,) or not np.all(np.isfinite(offline_start)):
                raise RuntimeError("离线轨迹首点不是有效七关节角")
            actual_start = read_left_joints_deg(args.arm_host)
            delta = (actual_start - offline_start + 180.0) % 360.0 - 180.0
            start_error = float(np.max(np.abs(delta)))
            selected["offline_start_state_deg"] = GRASPUP_ABOVE_JOINTS_DEG
            selected["offline_trajectory_first_point_deg"] = offline_start.tolist()
            selected["actual_execution_start_deg"] = actual_start.tolist()
            selected["offline_start_max_error_deg"] = start_error
            if start_error <= args.offline_start_tolerance_deg:
                planning_mode = "offline"
                print(
                    "[离线轨迹起点校验通过] 最大关节误差="
                    f"{start_error:.4f}deg",
                    flush=True,
                )
            else:
                print(
                    "[离线轨迹起点误差超限] 丢弃离线轨迹并在线重规划："
                    f"{start_error:.4f}deg",
                    flush=True,
                )
                target_tool = np.eye(4)
                target_tool[:3, :3] = Rotation.from_quat(
                    selected["target_tool_pose"]["orientation_xyzw"]
                ).as_matrix()
                target_tool[:3, 3] = selected["target_tool_pose"]["position"]
                planner.set_next_plan_context(
                    "left_graspup_top", "left_place_position"
                )
                selected_plan = planner.plan_cartesian_pose(
                    "left",
                    matrix_pose(target_tool),
                    planner=post_graspup_planner,
                    rrtstar_goal_tolerance=(post_graspup_planner == "rrtstar"),
                )
                if selected_plan is None:
                    raise RuntimeError("在线重规划结果为空")
                planning_mode = "online_replan_after_start_mismatch"

        publish_ready(args, {
            "stage": "left_place_ready",
            "outcome": "success",
            "place_module": profile.module_id,
            "classification": classification,
            "placement_branch": branch,
            "result_path": str(args.result_file),
            "selected_candidate": selected,
            "at": "graspup_above_final_placement_planned",
        })
        wait_permit(args)

        if not args.skip_motion:
            confirm_motion(
                args,
                f"确认最终 {args.post_graspup_planner.upper()} 轨迹安全，"
                "按回车执行...",
            )
            if not gripper_holds_object("移动到最终目标工件位姿"):
                finish_dropped_cycle()
                return 0
            planner.execute_planned_result(selected_plan)

        selected_object = np.eye(4)
        selected_object[:3, :3] = Rotation.from_quat(
            selected["target_object_pose"]["orientation_xyzw"]
        ).as_matrix()
        selected_object[:3, 3] = selected["target_object_pose"]["position"]
        selected_object_z = float(selected_object[2, 3])
        active_release_z = [
            float(value)
            for value in release_z_candidates
            if float(value) <= selected_object_z + 1e-9
        ]
        release_motion_required = bool(release_z_candidates)
        if release_motion_required and not active_release_z:
            raise RuntimeError(
                "松爪 Z 候选中没有不高于已选放置 Z 的候选"
            )
        release_attempts = []
        release_succeeded = not release_motion_required
        target_tool = np.eye(4)
        target_tool[:3, :3] = Rotation.from_quat(
            selected["target_tool_pose"]["orientation_xyzw"]
        ).as_matrix()
        target_tool[:3, 3] = selected["target_tool_pose"]["position"]
        selected_release_tool = target_tool if not release_motion_required else None
        if not release_motion_required:
            record = {
                "record_type": "release_descent",
                "skipped_by_empty_release_z_candidates": True,
                "succeeded": True,
                "error": None,
            }
            release_attempts.append(record)
            append_placement_trace(record)
            print("[松爪位] release_z_candidates_m=[]；跳过第二段 LIN，直接松爪。", flush=True)
        elif not args.skip_motion:
            confirm_motion(
                args,
                f"确认依次尝试松爪Z={active_release_z}的 LIN 路径安全，"
                "按回车开始...",
            )
        for release_z in active_release_z:
            release_object = selected_object.copy()
            release_object[2, 3] = release_z
            release_tool = release_object @ np.linalg.inv(T_tool_object)
            record = {
                "record_type": "release_descent",
                "placement_point": {
                    "frame_id": "body_base",
                    "object_xyz_m": release_object[:3, 3].tolist(),
                    "tool_xyz_m": release_tool[:3, 3].tolist(),
                },
                "object_z_m": release_z,
                "target_tool_xyz": release_tool[:3, 3].tolist(),
                "succeeded": False,
                "error": None,
            }
            try:
                if not args.skip_motion:
                    planner.set_next_plan_context(
                        "left_place_position", "left_release_position"
                    )
                    planner.move_cartesian_pose(
                        "left",
                        matrix_pose(release_tool),
                        planner="lin",
                    )
                record["succeeded"] = True
                release_succeeded = True
                selected_release_tool = release_tool
                release_attempts.append(record)
                append_placement_trace(record)
                break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                release_attempts.append(record)
                append_placement_trace(record)
                print(
                    f"[松爪候选失败] z={release_z}: {record['error']}",
                    flush=True,
                )

        if not release_succeeded and not args.skip_motion:
            print(
                "[下降失败恢复] 所有松爪候选均失败；"
                "按原版在机械臂当前位置打开左夹爪",
                flush=True,
            )
        final_position = None
        if not args.skip_motion:
            final_position = open_left_gripper(args)

        if release_succeeded and release_motion_required and not args.skip_motion:
            confirm_motion(
                args,
                "确认松爪后回到下降前末端位姿路径安全，按回车执行 LIN...",
            )
            planner.set_next_plan_context(
                "left_release_position", "left_place_position"
            )
            planner.move_cartesian_pose(
                "left",
                matrix_pose(target_tool),
                planner="lin",
            )

        if not args.skip_motion:
            confirm_motion(
                args,
                "确认回 graspup 路径安全，按回车规划并执行...",
            )
            planner.config = replace(
                planner.config,
                max_velocity=float(args.preset_speed),
                max_acceleration=float(args.preset_speed),
            )
            planner.set_next_plan_context(
                "left_place_position", "left_graspup"
            )
            planner.move_left_joints(
                GRASPUP_JOINTS_DEG,
                planner=post_graspup_planner,
            )
            wait_for_left_graspup(args.arm_host)

        if not release_succeeded:
            # 工件虽已按原版打开，但没有到达任何经规划验证的松爪位。
            # 禁止总控把它当作反抓区内的工件继续 Module 5，按安全丢弃恢复处理。
            write_terminal_outcome(
                "held_object_dropped",
                completed_without_error=True,
                release_motion_succeeded=False,
                release_motion_attempts=release_attempts,
                released_at_verified_target=False,
                gripper_final_position_percent=final_position,
                returned_to_graspup=not args.skip_motion,
            )
            print(
                "[下降失败恢复完成] 未到达有效松爪位；"
                "禁止进入反抓，buffer保持为空并开始下一轮。",
                flush=True,
            )
            return 0

        write_stage_result(args.result_file, {
            "stage": "left_place",
            "outcome": "success",
            "place_module": profile.module_id,
            "classification": classification,
            "placement_branch": branch,
            "selected_candidate": selected,
            "planning_mode": planning_mode,
            "planning_scheduler": planning_scheduler_log,
            "planning_attempts": planning_attempts,
            "release_motion_succeeded": release_succeeded,
            "release_motion_attempts": release_attempts,
            # 松爪阶段只改变工件 Z，高度之外的姿态与已选放置候选完全相同。
            "release_object_base_rpy_deg": selected["object_base_rpy_deg"],
            "release_tool_pose": (
                {
                    "frame_id": "body_base",
                    "position": selected_release_tool[:3, 3].tolist(),
                    "orientation_xyzw": Rotation.from_matrix(
                        selected_release_tool[:3, :3]
                    ).as_quat().tolist(),
                }
                if selected_release_tool is not None else None
            ),
            "gripper_final_position_percent": final_position,
            "holding_checks": holding_checks,
            "released": not args.skip_motion,
            "returned_to_graspup": not args.skip_motion,
        })
        print(
            f"[left-place] {branch} 放置结束，已验证回到 graspup",
            flush=True,
        )
        return 0
    finally:
        if approach_thread is not None and approach_thread.is_alive():
            approach_thread.join()
        if timeout_thread is not None and timeout_thread.is_alive():
            timeout_thread.join()
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
