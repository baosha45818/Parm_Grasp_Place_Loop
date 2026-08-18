#!/usr/bin/env python3
"""Module 3：把 Module 1 已验证的单个工件放到固定左臂 Buffer。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
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
from modules.module_3_left_place.placement_config import load_workpiece_parameters
from modules.module_3_left_place.placement_parameters import (
    DISCARD_BELOW_GRASPUP_M,
    GRASPUP_ABOVE_JOINTS_DEG,
    GRASPUP_ABOVE_PLANNING_TIMEOUT_S,
    GRASPUP_JOINTS_DEG,
    GRASPUP_LIFT_M,
)
from modules.stage_result import write_stage_result
from tools.planner_runtime import PlannerRuntime


MODULE_ID = "3-buffer-place"
CLASSIFICATION = "BUFFER"
PLACEMENT_BRANCH = "BUFFER_DIRECT"
LEFT_GRIPPER_HAND = 1
GRIPPER_HOLD_STABLE_S = 0.5
GRIPPER_HOLD_POLL_S = 0.1
GRIPPER_HOLD_TIMEOUT_S = 2.0


def parse_args() -> argparse.Namespace:
    """解析独立运行参数，并落实真实运动的双参数授权门禁。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", choices=("1", "2"), default="1")
    parser.add_argument(
        "--weight-handoff-file",
        type=Path,
        required=True,
        help="Module 1 verified SINGLE handoff JSON",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=(
            ROOT
            / "output/module_3_left_place"
            / f"buffer_place_{time.strftime('%Y%m%d_%H%M%S')}.json"
        ),
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument(
        "--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11")
    )
    parser.add_argument(
        "--gripper-host",
        default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"),
    )
    parser.add_argument(
        "--gripper-hand",
        type=int,
        default=LEFT_GRIPPER_HAND,
        choices=(LEFT_GRIPPER_HAND,),
        help="compatibility option; Module 3 uses the left gripper (hand=1)",
    )
    parser.add_argument(
        "--execute-url", default="http://127.0.0.1:5667/middle/waypoint/execute"
    )
    left_scale = float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0"))
    parser.add_argument("--preset-speed", type=float, default=0.2 * left_scale)
    parser.add_argument("--motion-speed", type=float, default=0.1 * left_scale)
    parser.add_argument(
        "--post-graspup-planner",
        choices=("rrtconnect", "rrtstar"),
        default="rrtstar",
    )
    parser.add_argument("--offline-start-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--placement-planning-inflight",
        "--placement-planning-workers",
        dest="placement_planning_inflight",
        type=int,
        default=8,
    )
    parser.add_argument("--graspup-lift-m", type=float, default=GRASPUP_LIFT_M)
    parser.add_argument("--pre-place-middle-joints-deg", nargs=7, type=float)
    parser.add_argument("--placement-ready-file", type=Path)
    parser.add_argument("--placement-permit-file", type=Path)
    parser.add_argument("--placement-permit-poll-s", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-real-place", action="store_true")
    args = parser.parse_args()
    if args.execute != args.confirm_real_place:
        parser.error("physical motion requires both --execute and --confirm-real-place")
    args.skip_motion = not args.execute
    return args


def matrix_pose(matrix: np.ndarray) -> ObjectPose:
    """把 4×4 齐次矩阵转换为 Planner 使用的 ``ObjectPose``。"""
    return ObjectPose(
        frame_id="body_base",
        position=np.asarray(matrix[:3, 3], dtype=float).tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def planner_matrix(response: dict) -> np.ndarray:
    """把 Planner TF 响应转换为 4×4 齐次矩阵并校验字段完整性。"""
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
    """从 Arm SDK 读取左臂七关节角，单位为度。"""
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
    """轮询实际关节角，直到左臂到达指定目标或超时。"""
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
    """等待左臂到达固定 ``graspup`` 安全关节位。"""
    wait_for_left_joint_target(arm_host, GRASPUP_JOINTS_DEG, label="graspup")


def validate_z_candidates(
    name: str, values: tuple[float, ...], *, allow_empty: bool = False
) -> list[float]:
    """校验严格递增的 Z 候选；松爪候选可显式为空。"""
    candidates = np.asarray(values, dtype=float)
    if allow_empty and candidates.size == 0:
        return []
    if (
        candidates.size == 0
        or not np.all(np.isfinite(candidates))
        or np.any(np.diff(candidates) <= 0.0)
    ):
        raise ValueError(f"{name} 必须是严格递增的有限数值")
    return candidates.tolist()


def open_left_gripper(args: argparse.Namespace) -> float | None:
    """确认左夹爪可用后完全打开，并返回驱动报告的最终开度。"""
    from parm_sdk import Gripper
    from modules.module_0_prepare.gripper_state_compat import is_gripper_usable

    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        hand_state = gripper.get_state().get(LEFT_GRIPPER_HAND)
        if hand_state is None:
            raise RuntimeError("左夹爪 hand=1 没有同步状态或未连接")
        if not is_gripper_usable(hand_state):
            raise RuntimeError("左夹爪尚未初始化，禁止发送打开指令")
        response = gripper.move(
            LEFT_GRIPPER_HAND,
            100.0,
            force=100.0,
            speed=80.0,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
    payload = response.get("data") or response
    devices = payload.get("devices") or {}
    interfaces = (devices.get("left_gripper") or {}).get("state_interfaces") or {}
    position = interfaces.get("gripper/position")
    if position is None:
        position = payload.get("position")
    return None if position is None else float(position)


def publish_ready(args: argparse.Namespace, payload: dict) -> None:
    """如配置了外部协调文件，则原子写出当前阶段状态。"""
    if args.placement_ready_file is not None:
        write_stage_result(args.placement_ready_file, payload)


def wait_permit(args: argparse.Namespace) -> None:
    """仅在显式配置时等待外部 Buffer 执行许可。"""
    if args.skip_motion or args.placement_permit_file is None:
        return
    print(f"[module3] 等待 Buffer 许可: {args.placement_permit_file}", flush=True)
    while not args.placement_permit_file.is_file():
        time.sleep(float(args.placement_permit_poll_s))
    content = args.placement_permit_file.read_text(encoding="utf-8").strip().lower()
    if content not in {"allow", "y", "yes", "1"}:
        raise RuntimeError(f"放置许可内容无效: {content!r}")


def validate_runtime_args(args: argparse.Namespace) -> None:
    """校验会影响运动安全、并发数和外部门禁的运行参数。"""
    if (args.placement_ready_file is None) != (args.placement_permit_file is None):
        raise ValueError("placement ready/permit 文件必须同时提供")
    if not np.isfinite(args.graspup_lift_m) or not np.isclose(
        args.graspup_lift_m, GRASPUP_LIFT_M, atol=1e-9
    ):
        raise ValueError("--graspup-lift-m 当前必须为固定值 0.10m")
    if args.pre_place_middle_joints_deg is not None and not np.all(
        np.isfinite(args.pre_place_middle_joints_deg)
    ):
        raise ValueError("--pre-place-middle-joints-deg 必须包含七个有限角度")
    for name, value in (
        ("--preset-speed", args.preset_speed),
        ("--motion-speed", args.motion_speed),
    ):
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} 必须是 (0, 1] 内的有限数值")
    if (
        not np.isfinite(args.offline_start_tolerance_deg)
        or args.offline_start_tolerance_deg <= 0.0
    ):
        raise ValueError("--offline-start-tolerance-deg 必须是大于0的有限数值")
    if not 1 <= args.placement_planning_inflight <= 16:
        raise ValueError("--placement-planning-inflight 必须在[1, 16]内")
    if (
        not np.isfinite(args.placement_permit_poll_s)
        or args.placement_permit_poll_s <= 0.0
    ):
        raise ValueError("--placement-permit-poll-s 必须是大于0的有限数值")


def load_verified_handoff(path: Path) -> dict:
    """读取 Module 1 handoff，并只接受经过重量验证的单个工件。"""
    if not path.is_file():
        raise FileNotFoundError(f"Module 1 handoff 不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("weight_check_enabled") is not True:
        raise RuntimeError("Module 1 handoff 未启用重量检查")
    weight_check = payload.get("weight_check")
    if not isinstance(weight_check, dict):
        raise RuntimeError("Module 1 handoff 缺少 weight_check")
    if weight_check.get("classification") != "SINGLE":
        raise RuntimeError(
            "Module 3 只接受 Module 1 验证过的 SINGLE 工件，实际为 "
            f"{weight_check.get('classification')!r}"
        )
    return payload


def build_planner_config(args: argparse.Namespace) -> LoopConfig:
    """集中构造 Module 3 的 Planner/执行配置，避免主流程散落参数。"""
    return LoopConfig(
        planner_host=args.planner_host,
        execute_url=args.execute_url,
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_eef",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.motion_speed,
        max_acceleration=args.motion_speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )


def main() -> int:
    """按“校验→到位→规划→放置→松爪→回程”的顺序执行 Module 3。"""
    # 阶段 1：解析参数、同步 SDK 主机并完成所有静态校验。
    args = parse_args()
    os.environ["PARM_ARM_HOST"] = str(args.arm_host)
    os.environ["PARM_GRIPPER_HOST"] = str(args.gripper_host)
    if args.skip_motion:
        print(
            "[module3] plan-only: robot/gripper motion is disabled; "
            "only Planner feasibility is evaluated",
            flush=True,
        )
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    validate_runtime_args(args)

    # 阶段 2：验证 Module 1 交接文件，读取唯一的固定 Buffer 目标。
    load_verified_handoff(args.weight_handoff_file)

    parameters = load_workpiece_parameters(args.workpiece)
    target_xyz = np.asarray(parameters.target_left_eef_xyz_m, dtype=float)
    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        raise ValueError("Buffer target_left_eef_xyz_m 必须包含三个有限数值")
    target_z_candidates = validate_z_candidates(
        "[buffer].target_z_candidates_m", parameters.target_z_candidates_m
    )
    release_z_candidates = validate_z_candidates(
        "[buffer].release_z_candidates_m",
        parameters.release_z_candidates_m,
        allow_empty=True,
    )
    post_graspup_planner = {
        "rrtconnect": "ompl",
        "rrtstar": "rrtstar",
    }[args.post_graspup_planner]
    scheduler = {
        "mode": "sliding_window",
        "max_inflight": int(args.placement_planning_inflight),
        "planner": post_graspup_planner,
        "offline_start_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
        "policy": "fixed_left_eef_buffer_target",
        "left_placement_frame": "left_eef",
    }
    planning_attempts: list[dict] = []
    holding_checks: list[dict] = []
    drop_details: list[dict] = []
    placement_trace_path = args.result_file.parent / "placement_planning_attempts.jsonl"
    os.environ["PARM_PLANNING_RECORD_DIR"] = str(
        ROOT / "output/left_place_trajectories" / time.strftime("%Y%m%d_%H%M%S")
    )

    def append_trace(record: dict) -> None:
        """追加一条带调度器上下文的规划诊断记录。"""
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "planning_scheduler": scheduler,
            **record,
        }
        with placement_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    planner_cfg = build_planner_config(args)
    planner = PlannerRuntime(planner_cfg)
    object_dropped = threading.Event()

    def gripper_holds_object(label: str) -> bool:
        """确认左夹爪连续稳定报告 ``gripping``，失败时锁死后续持件运动。"""
        if args.skip_motion:
            return True
        if object_dropped.is_set():
            return False
        state = None
        stable_samples = 0
        record = None
        try:
            from parm_sdk import Gripper

            with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                deadline = time.monotonic() + GRIPPER_HOLD_TIMEOUT_S
                stable_since = None
                while time.monotonic() < deadline:
                    candidate = gripper.get_state().get(LEFT_GRIPPER_HAND)
                    if isinstance(candidate, dict):
                        state = candidate
                        if str(candidate.get("motion_state", "")).lower() == "gripping":
                            stable_since = stable_since or time.monotonic()
                            stable_samples += 1
                            if time.monotonic() - stable_since >= GRIPPER_HOLD_STABLE_S:
                                record = {
                                    "before_motion": label,
                                    "source": "gripper_state",
                                    "motion_state": "gripping",
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
                    else:
                        stable_since = None
                        stable_samples = 0
                    time.sleep(GRIPPER_HOLD_POLL_S)
            if record is None:
                if not isinstance(state, dict):
                    raise RuntimeError("未收到 hand=1 夹爪状态")
                record = {
                    "before_motion": label,
                    "source": "gripper_state",
                    "motion_state": state.get("motion_state"),
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
            f"[运动前持件复检] {label}: motion_state="
            f"{record.get('motion_state')} passed={record['passed']}",
            flush=True,
        )
        if not record["passed"]:
            drop_details.append(record)
            object_dropped.set()
            print("[工件掉落] 左夹爪未报告 gripping，禁止后续持件运动", flush=True)
            return False
        return True

    def write_terminal_outcome(outcome: str, **extra) -> None:
        """统一写出提前结束或恢复结束的终态结果。"""
        write_stage_result(
            args.result_file,
            {
                "stage": "buffer_place",
                "outcome": outcome,
                "module": "module_3_left_place",
                "place_module": MODULE_ID,
                "classification": CLASSIFICATION,
                "placement_branch": PLACEMENT_BRANCH,
                "holding_checks": holding_checks,
                "planning_scheduler": scheduler,
                **extra,
            },
        )
        publish_ready(
            args,
            {
                "stage": "buffer_place_ready",
                "outcome": outcome,
                "result_path": str(args.result_file),
            },
        )

    def finish_dropped_cycle() -> None:
        """掉件后回到安全位、打开夹爪并把本轮记录为已恢复。"""
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

    def return_to_graspup_after_buffer() -> None:
        """从 Buffer 区域依次返回 ``graspup_top`` 和 ``graspup``。"""
        planner.config = replace(
            planner.config,
            max_velocity=float(args.preset_speed),
            max_acceleration=float(args.preset_speed),
        )
        planner.move_left_joints(
            GRASPUP_ABOVE_JOINTS_DEG, planner=post_graspup_planner
        )
        wait_for_left_joint_target(
            args.arm_host, GRASPUP_ABOVE_JOINTS_DEG, label="graspup_top"
        )
        planner.move_left_joints(GRASPUP_JOINTS_DEG, planner=post_graspup_planner)
        wait_for_left_graspup(args.arm_host)

    approach_reached = threading.Event()
    approach_finished = threading.Event()
    plan_found = threading.Event()
    planning_finished = threading.Event()
    planning_timed_out = threading.Event()
    approach_errors: list[BaseException] = []
    timeout_errors: list[BaseException] = []
    approach_thread = None
    timeout_thread = None

    try:
        # 阶段 3：真实模式先复检持件状态，并把机械臂送到 graspup_top。
        # 纯规划模式只连接 Planner，不创建 Arm/Gripper 执行动作。
        planner.open(required_frames={"body_base", "left_eef"})
        if not args.skip_motion:
            if not gripper_holds_object("Module 3开始（graspup）"):
                finish_dropped_cycle()
                return 0
            def move_to_graspup_top() -> None:
                """真实模式下把持件左臂送到离线规划起点 ``graspup_top``。"""
                motion = PlannerRuntime(planner_cfg)
                try:
                    motion.open(required_frames={"body_base", "left_eef"})
                    motion.config = replace(
                        motion.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    if args.pre_place_middle_joints_deg is not None:
                        middle = [float(value) for value in args.pre_place_middle_joints_deg]
                        if not gripper_holds_object("移动到放置前中间位"):
                            return
                        motion.set_next_plan_context("module1_handoff", "pre_place_middle")
                        motion.move_left_joints(middle, planner="rrtstar")
                        wait_for_left_joint_target(args.arm_host, middle, label="放置前中间位")
                        if not gripper_holds_object("从中间位移动到graspup_top"):
                            return
                        motion.set_next_plan_context("pre_place_middle", "left_graspup_top")
                        motion.move_left_joints(GRASPUP_ABOVE_JOINTS_DEG, planner="rrtstar")
                        wait_for_left_joint_target(
                            args.arm_host,
                            GRASPUP_ABOVE_JOINTS_DEG,
                            label="graspup_top",
                        )
                    else:
                        if not gripper_holds_object("从当前位置回graspup"):
                            return
                        motion.set_next_plan_context("module1_handoff", "left_graspup")
                        motion.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
                        wait_for_left_graspup(args.arm_host)
                        response = motion.lookup_transform("body_base", "left_eef")
                        if response is None:
                            raise RuntimeError("规划器没有返回 graspup 末端位姿")
                        above = planner_matrix(response)
                        above[2, 3] += float(args.graspup_lift_m)
                        motion.config = replace(
                            motion.config,
                            max_velocity=float(args.motion_speed),
                            max_acceleration=float(args.motion_speed),
                        )
                        if not gripper_holds_object("从graspup移动到graspup_top"):
                            return
                        motion.set_next_plan_context("left_graspup", "left_graspup_top")
                        above_plan = motion.plan_cartesian_pose(
                            "left", matrix_pose(above), planner="lin"
                        )
                        if above_plan is None:
                            raise RuntimeError("graspup_top LIN 规划返回空结果")
                        motion.execute_planned_result(above_plan)
                    approach_reached.set()
                    print("[运动线程] 已到达 graspup_top", flush=True)
                except BaseException as exc:
                    approach_errors.append(exc)
                finally:
                    approach_finished.set()
                    motion.close()

            approach_thread = threading.Thread(
                target=move_to_graspup_top,
                name="module3-graspup-top-motion",
                daemon=False,
            )
            approach_thread.start()

            def recover_after_planning_failure() -> None:
                """到位后长期无可行解时执行安全丢弃并返回 ``graspup``。"""
                while not approach_finished.is_set():
                    if approach_reached.wait(timeout=0.05):
                        break
                if not approach_reached.is_set():
                    return
                deadline = time.monotonic() + GRASPUP_ABOVE_PLANNING_TIMEOUT_S
                while not plan_found.is_set() and not planning_finished.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    plan_found.wait(timeout=min(0.05, remaining))
                if plan_found.is_set():
                    return
                planning_timed_out.set()
                print("[规划失败恢复] 无可行 Buffer 轨迹，执行安全丢弃", flush=True)
                recovery = PlannerRuntime(planner_cfg)
                try:
                    recovery.open(required_frames={"body_base", "left_eef"})
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    recovery.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
                    wait_for_left_graspup(args.arm_host)
                    response = recovery.lookup_transform("body_base", "left_eef")
                    if response is None:
                        raise RuntimeError("规划器没有返回 graspup 末端位姿")
                    discard = planner_matrix(response)
                    discard[2, 3] -= DISCARD_BELOW_GRASPUP_M
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.motion_speed),
                        max_acceleration=float(args.motion_speed),
                    )
                    recovery.move_cartesian_pose(
                        "left", matrix_pose(discard), planner="lin"
                    )
                    open_left_gripper(args)
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    recovery.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
                    wait_for_left_graspup(args.arm_host)
                except BaseException as exc:
                    timeout_errors.append(exc)
                finally:
                    recovery.close()

            timeout_thread = threading.Thread(
                target=recover_after_planning_failure,
                name="module3-planning-failure-recovery",
                daemon=False,
            )
            timeout_thread.start()

        if not args.skip_motion:
            while not approach_reached.wait(timeout=0.05):
                if approach_finished.is_set():
                    if approach_errors:
                        raise RuntimeError(
                            f"到达 graspup_top 失败: {approach_errors[0]}"
                        ) from approach_errors[0]
                    if object_dropped.is_set():
                        finish_dropped_cycle()
                        return 0
                    raise RuntimeError("到达 graspup_top 前运动线程已结束")

        # 阶段 4：读取 graspup_top 末端旋转。Buffer 只改变 TCP 平移，
        # 不猜测工件相对夹爪的姿态。
        response = planner.lookup_transform("body_base", "left_eef")
        if response is None:
            raise RuntimeError("规划器没有返回 graspup_top 末端位姿")
        target_rotation = planner_matrix(response)[:3, :3].copy()
        scheduler["orientation_source"] = "body_base_T_left_eef_at_graspup_top"
        scheduler["tool_rpy_deg"] = Rotation.from_matrix(target_rotation).as_euler(
            "xyz", degrees=True
        ).tolist()
        print(
            f"[Buffer] 保持 graspup_top left_eef RPY="
            f"{np.round(scheduler['tool_rpy_deg'], 3).tolist()}",
            flush=True,
        )

        def plan_candidate(target_tool: np.ndarray) -> dict:
            """使用独立 Planner 连接，从固定关节起点规划一个 Buffer 候选。"""
            candidate_planner = PlannerRuntime(planner_cfg)
            try:
                candidate_planner.open(required_frames={"body_base", "left_eef"})
                result = candidate_planner.plan_cartesian_pose(
                    "left",
                    matrix_pose(target_tool),
                    planner=post_graspup_planner,
                    start_joints_deg=GRASPUP_ABOVE_JOINTS_DEG,
                    rrtstar_goal_tolerance=(post_graspup_planner == "rrtstar"),
                    use_left_eef_collision_box=False,
                )
                if result is None:
                    raise RuntimeError("离线规划返回空结果")
                return result
            finally:
                candidate_planner.close()

        print(
            f"[离线规划] start_state={GRASPUP_ABOVE_JOINTS_DEG}; "
            f"最多{args.placement_planning_inflight}条在飞请求",
            flush=True,
        )
        # 阶段 5：按配置顺序并发预取 Z 候选；首个成功结果停止补发。
        selected = None
        selected_plan = None
        submitted = 0
        completed = 0
        next_index = 0
        in_flight = {}
        try:
            with ThreadPoolExecutor(
                max_workers=int(args.placement_planning_inflight),
                thread_name_prefix="module3-buffer-plan",
            ) as executor:
                while in_flight or next_index < len(target_z_candidates):
                    while (
                        selected is None
                        and len(in_flight) < int(args.placement_planning_inflight)
                        and next_index < len(target_z_candidates)
                        and not planning_timed_out.is_set()
                        and not object_dropped.is_set()
                    ):
                        z_index = next_index
                        target_z = float(target_z_candidates[z_index])
                        target_tool = np.eye(4)
                        target_tool[:3, :3] = target_rotation
                        target_tool[:3, 3] = [target_xyz[0], target_xyz[1], target_z]
                        attempt = {
                            "place_module": MODULE_ID,
                            "classification": CLASSIFICATION,
                            "placement_branch": PLACEMENT_BRANCH,
                            "priority_index": z_index,
                            "z_index": z_index,
                            "target_left_eef_z_m": target_z,
                            "target_tool_xyz": target_tool[:3, 3].tolist(),
                            "target_kind": "fixed_left_eef",
                            "release_z_candidates_m": release_z_candidates,
                            "planning_start_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
                            "planning_submission_order": submitted + 1,
                            "planning_inflight_after_submit": len(in_flight) + 1,
                            "use_left_eef_collision_box": False,
                            "target_tool": target_tool,
                        }
                        if float(target_tool[2, 2]) >= 0.0:
                            rejected = {
                                key: value
                                for key, value in attempt.items()
                                if key != "target_tool"
                            }
                            rejected.update(
                                {
                                    "planned": False,
                                    "rejected_before_planning": True,
                                    "error": "left_eef +Z在body_base Z上的投影必须为负",
                                    "target_tool_positive_z_in_body_base": target_tool[
                                        :3, 2
                                    ].tolist(),
                                }
                            )
                            planning_attempts.append(rejected)
                            append_trace(rejected)
                            next_index += 1
                            continue
                        future = executor.submit(plan_candidate, target_tool)
                        in_flight[future] = attempt
                        submitted += 1
                        next_index += 1
                        print(
                            f"[提交 {z_index + 1}/{len(target_z_candidates)}] "
                            f"Buffer left_eef z={target_z:.3f}",
                            flush=True,
                        )
                    if not in_flight:
                        break
                    future = next(as_completed(in_flight))
                    attempt = in_flight.pop(future)
                    completed += 1
                    try:
                        plan = future.result()
                    except Exception as exc:
                        failed = {
                            key: value for key, value in attempt.items() if key != "target_tool"
                        }
                        failed.update(
                            {
                                "planned": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "planning_completion_order": completed,
                                "planning_inflight_after_complete": len(in_flight),
                            }
                        )
                        planning_attempts.append(failed)
                        append_trace(failed)
                        continue
                    successful = {
                        key: value for key, value in attempt.items() if key != "target_tool"
                    }
                    data = plan.get("data") or {}
                    trajectory = data.get("trajectory") or []
                    target_tool = attempt["target_tool"]
                    successful.update(
                        {
                            "planned": True,
                            "planning_completion_order": completed,
                            "planning_inflight_after_complete": len(in_flight),
                            "target_tool_pose": {
                                "position": target_tool[:3, 3].tolist(),
                                "orientation_xyzw": Rotation.from_matrix(
                                    target_tool[:3, :3]
                                ).as_quat().tolist(),
                            },
                            "planned_trajectory_joint_deg": trajectory,
                            "planned_start_joints_deg": (
                                trajectory[0] if trajectory else GRASPUP_ABOVE_JOINTS_DEG
                            ),
                            "planned_end_joints_deg": trajectory[-1] if trajectory else None,
                            "sample_frequency_hz": data.get("sample_frequency"),
                        }
                    )
                    planning_attempts.append(successful)
                    append_trace(successful)
                    if selected is None and not planning_timed_out.is_set():
                        selected = successful
                        selected_plan = plan
                        plan_found.set()
                        print(
                            f"[找到可行Buffer候选] left_eef z="
                            f"{successful['target_left_eef_z_m']:.3f}",
                            flush=True,
                        )
        finally:
            planning_finished.set()

        if approach_thread is not None:
            approach_thread.join()
        if timeout_thread is not None:
            timeout_thread.join()
        if approach_errors:
            raise RuntimeError(f"到达 graspup_top 失败: {approach_errors[0]}") from approach_errors[0]
        if timeout_errors:
            raise RuntimeError(f"规划失败恢复异常: {timeout_errors[0]}") from timeout_errors[0]
        if object_dropped.is_set():
            finish_dropped_cycle()
            return 0
        if planning_timed_out.is_set():
            write_terminal_outcome(
                "planning_failed_recovered_at_graspup",
                completed_without_error=True,
                planning_attempts=planning_attempts,
                returned_to_graspup=True,
            )
            return 0
        if selected is None or selected_plan is None:
            raise RuntimeError("全部 Buffer Z 候选均规划失败")

        # 阶段 6：真实执行前核对离线轨迹起点；不一致时在线重规划。
        planning_mode = "dry_run"
        if not args.skip_motion:
            trajectory = ((selected_plan.get("data") or {}).get("trajectory") or [])
            if not trajectory:
                raise RuntimeError("离线规划结果缺少可执行轨迹")
            offline_start = np.asarray(trajectory[0], dtype=float)
            if offline_start.shape != (7,) or not np.all(np.isfinite(offline_start)):
                raise RuntimeError("离线轨迹首点不是有效七关节角")
            actual_start = read_left_joints_deg(args.arm_host)
            delta = (actual_start - offline_start + 180.0) % 360.0 - 180.0
            start_error = float(np.max(np.abs(delta)))
            selected["offline_trajectory_first_point_deg"] = offline_start.tolist()
            selected["actual_execution_start_deg"] = actual_start.tolist()
            selected["offline_start_max_error_deg"] = start_error
            if start_error <= args.offline_start_tolerance_deg:
                planning_mode = "offline"
            else:
                print(
                    f"[离线起点误差{start_error:.4f}deg] 在线重规划",
                    flush=True,
                )
                target_tool = np.eye(4)
                target_tool[:3, :3] = Rotation.from_quat(
                    selected["target_tool_pose"]["orientation_xyzw"]
                ).as_matrix()
                target_tool[:3, 3] = selected["target_tool_pose"]["position"]
                selected_plan = planner.plan_cartesian_pose(
                    "left",
                    matrix_pose(target_tool),
                    planner=post_graspup_planner,
                    rrtstar_goal_tolerance=(post_graspup_planner == "rrtstar"),
                    use_left_eef_collision_box=False,
                )
                if selected_plan is None:
                    raise RuntimeError("在线重规划结果为空")
                planning_mode = "online_replan_after_start_mismatch"

        # 阶段 7：发布“轨迹已准备”状态，等待可选外部门禁，然后执行放置。
        publish_ready(
            args,
            {
                "stage": "buffer_place_ready",
                "outcome": "success",
                "result_path": str(args.result_file),
                "selected_candidate": selected,
            },
        )
        wait_permit(args)
        if not args.skip_motion:
            if not gripper_holds_object("从graspup_top移动到Buffer"):
                finish_dropped_cycle()
                return 0
            planner.execute_planned_result(selected_plan)

        # 阶段 8：到达 Buffer 后处理可选松爪下降，并打开左夹爪。
        target_tool = np.eye(4)
        target_tool[:3, :3] = Rotation.from_quat(
            selected["target_tool_pose"]["orientation_xyzw"]
        ).as_matrix()
        target_tool[:3, 3] = selected["target_tool_pose"]["position"]
        target_tool_z = float(target_tool[2, 3])
        active_release_z = [
            float(value) for value in release_z_candidates if float(value) <= target_tool_z + 1e-9
        ]
        release_required = bool(release_z_candidates)
        if release_required and not active_release_z:
            raise RuntimeError("松爪 Z 候选中没有不高于已选 Buffer Z 的候选")
        release_attempts: list[dict] = []
        release_succeeded = not release_required
        selected_release_tool = target_tool if not release_required else None
        if not release_required:
            record = {
                "record_type": "release_descent",
                "skipped_by_empty_release_z_candidates": True,
                "succeeded": True,
                "error": None,
            }
            release_attempts.append(record)
            append_trace(record)
            print("[松爪位] release_z_candidates_m=[]；在Buffer目标直接松爪", flush=True)
        for release_z in active_release_z:
            release_tool = target_tool.copy()
            release_tool[2, 3] = release_z
            record = {
                "record_type": "release_descent",
                "release_z_m": release_z,
                "release_z_frame": "left_eef",
                "target_tool_xyz": release_tool[:3, 3].tolist(),
                "succeeded": False,
                "error": None,
            }
            try:
                if not args.skip_motion:
                    planner.move_cartesian_pose(
                        "left", matrix_pose(release_tool), planner="lin"
                    )
                record["succeeded"] = True
                release_succeeded = True
                selected_release_tool = release_tool
                release_attempts.append(record)
                append_trace(record)
                break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                release_attempts.append(record)
                append_trace(record)
                print(f"[松爪候选失败] z={release_z}: {record['error']}", flush=True)

        # 所有必需下降候选都失败时，工件仍由左夹爪持有。禁止在未经验证的
        # 高度松爪；先安全返回 graspup，再用非零状态结束本轮。
        if release_required and not release_succeeded:
            if not args.skip_motion:
                if not gripper_holds_object("松爪下降失败后返回graspup"):
                    finish_dropped_cycle()
                    return 1
                return_to_graspup_after_buffer()
            write_terminal_outcome(
                "release_motion_failed_object_retained",
                completed_without_error=False,
                release_motion_succeeded=False,
                release_motion_attempts=release_attempts,
                released=False,
                released_at_verified_target=False,
                object_retained_in_left_gripper=not args.skip_motion,
                gripper_final_position_percent=None,
                returned_to_graspup=not args.skip_motion,
            )
            return 1

        final_position = None
        # 阶段 9：确认松爪位有效后打开夹爪，再按 Buffer → graspup_top →
        # graspup 安全回程。
        if not args.skip_motion:
            final_position = open_left_gripper(args)
        if release_succeeded and release_required and not args.skip_motion:
            planner.move_cartesian_pose("left", matrix_pose(target_tool), planner="lin")
        if not args.skip_motion:
            return_to_graspup_after_buffer()

        # 阶段 10：写出可供新总控消费的最终结果。
        write_stage_result(
            args.result_file,
            {
                "schema_version": 1,
                "stage": "buffer_place",
                "outcome": "success" if args.execute else "planned",
                "module": "module_3_left_place",
                "source_module1_handoff": str(args.weight_handoff_file.resolve()),
                "workpiece": args.workpiece,
                "place_module": MODULE_ID,
                "classification": CLASSIFICATION,
                "placement_branch": PLACEMENT_BRANCH,
                "selected_candidate": selected,
                "planning_mode": planning_mode,
                "planning_scheduler": scheduler,
                "planning_attempts": planning_attempts,
                "release_motion_succeeded": release_succeeded,
                "release_motion_attempts": release_attempts,
                "release_tool_pose": (
                    {
                        "frame_id": "body_base",
                        "position": selected_release_tool[:3, 3].tolist(),
                        "orientation_xyzw": Rotation.from_matrix(
                            selected_release_tool[:3, :3]
                        ).as_quat().tolist(),
                    }
                    if selected_release_tool is not None
                    else None
                ),
                "gripper_final_position_percent": final_position,
                "holding_checks": holding_checks,
                "released": not args.skip_motion,
                "returned_to_graspup": not args.skip_motion,
            },
        )
        print(
            (
                "[module3] Buffer放置完成，已返回 graspup"
                if args.execute
                else "[module3] Buffer纯规划完成；未发送机械臂或夹爪命令"
            ),
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
