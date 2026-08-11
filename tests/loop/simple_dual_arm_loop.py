#!/usr/bin/env python3
"""最小双臂串联：左臂抓取/inspect，右臂按缓冲区状态消费工件。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


LOOP_ROOT = Path(__file__).resolve().parents[2]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))

from loop_config import LoopConfig
from loop_types import ResultStatus
from modules.module_status import report as report_module_status
from tool.workpiece_timer import (
    WORKPIECE_ENV,
    begin as begin_workpiece_timer,
    complete as complete_workpiece_timer,
    module_timer,
)
from tool.planner_runtime import PlannerRuntime
PYTHON = Path("/home/dainanjun/Public/data_collector/.venv/bin/python")
GRASPNET_FLOW_WORKDIR = Path("/home/dainanjun/qiuguolin/graspnet")
GRASPNET_FLOW = GRASPNET_FLOW_WORKDIR / "deploy/version7/run_grasp_to_module2.sh"
APART_INSPECT_PLACE_FLOW = (
    LOOP_ROOT
    / "modules/module_2_left_inspect_apart/run_inspect_then_place.py"
)
LEFT_FLOW_SCRIPTS = {
    "apart": APART_INSPECT_PLACE_FLOW,
}
LEFT_PICK = LOOP_ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/main.py"
FRONT_BUFFER_REGRASP = LOOP_ROOT / "modules/module_5_left_pick/front_buffer_regrasp.py"
LEFT_PLACE = LOOP_ROOT / "modules/module_3_left_place/standalone_left_place.py"
HEAD_MOTION = LOOP_ROOT / "tool/move_head_gimbal.py"
HEAD_CALIBRATION = (
    LOOP_ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/"
    "config/body_head_calibration.json"
)
BUFFER_FACE_CHECK = Path("/home/dainanjun/qiuguolin/graspnet/buffer_check/check_buffer_face.py")
LEFT_HANDOFF_PREP = (
    LOOP_ROOT
    / "modules/module_5_left_pick/module_5_2_handoff_prepare/main.py"
)
RIGHT_LEFT_HANDOFF = (
    LOOP_ROOT
    / "modules/module_8_right_grasp/module_8_2_left_handoff/main.py"
)
RIGHT_FINISH = LOOP_ROOT / "tests/loop/right_place_and_idle.py"
RIGHT_PUSH_TO_Q1 = Path("/home/dainanjun/zhuyiming/push_T_ALL/push_to_q1_pipeline.py")
WORKPIECE_2_RIGHT_PUSH_TO_Q1 = Path(
    "/home/dainanjun/zhuyiming/push_T_ALL/workpiece_2/"
    "q1_closed_loop_pipeline.py"
)
MODULAR_RIGHT_BUFFER_PIPELINE = (
    LOOP_ROOT
    / "modules/module_8_right_grasp/module_8_1_buffer_grasp/"
    "workpiece_pipeline.py"
)
RIGHT_PLACE_WIGGLE = (
    LOOP_ROOT / "modules/module_9_right_place/place_hole_a_wiggle.py"
)
LEFT_GRASPUP_JOINTS_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]
LEFT_GRASPUP_APPROACH_JOINTS_DEG = [
    14.281,
    -29.886,
    -85.633,
    102.504,
    37.378,
    56.016,
    9.241,
]
# 反抓头部观察链的 ``left_handoff.prepare_joints_deg``。它是桌面区域
# 撤离后的中间观察位；Buffer Check 重拍只需要到这里，不需要继续回 graspup。
LEFT_MIDDLE_OBSERVATION_JOINTS_DEG = LEFT_GRASPUP_APPROACH_JOINTS_DEG
# 工件1正面/侧面经缓冲区后的既有放置准备位；工件2从自己的
# workpiece_2_handoff_profile读取目标，两个配置源保持独立。
RIGHT_PLACE_READY_JOINTS_DEG = [
    30.4353,
    23.8637,
    0.5905,
    66.3409,
    -62.0439,
    -64.3304,
    25.7149,
]
RIGHT_REVERSE_HANDOFF_PREP_JOINTS_DEG = [
    161.36, 19.07, -107.979, 71.451, -55.833, -14.896, 88.2,
]


def reverse_handoff_prepare_joints_deg() -> list[float]:
    """Select each workpiece's own safe right-hand handoff entry pose."""
    if (
        os.environ.get("PARM_MODULAR_LOOP_ENTRIES") == "1"
        and os.environ.get("PARM_LOOP_WORKPIECE") == "2"
    ):
        from modules.workpiece_2_handoff_profile import (
            workpiece_2_right_prepare_joints_deg,
        )

        return workpiece_2_right_prepare_joints_deg()
    return list(RIGHT_REVERSE_HANDOFF_PREP_JOINTS_DEG)

# 0: 缓冲区空/右手没有待处理工件；1: 正面或侧面工件在缓冲区；
# 2: 左手已完成反抓，工件在右手交接位。反抓进行中保持 0，避免右手提前进入。
BUFFER_EMPTY = 0
BUFFER_TABLE_READY = 1
BUFFER_HANDOFF_READY = 2
NO_GRASP_EXIT_CODE = 20
MAX_REVERSE_TABLE_GRASP_ATTEMPTS = 5
BUFFER_WORKPIECE_FILE = "active_buffer_workpiece.json"
MODULE2_INSUFFICIENT_POINTS_DISCARDED = "INSUFFICIENT_POINT_CLOUD_DISCARDED"
MODULE2_REGISTRATION_FAILED_RECOVERED = "PC1_PC2_REGISTRATION_FAILED_RECOVERED"
MODULE2_HELD_OBJECT_DROPPED = "HELD_OBJECT_DROPPED"
MODULE2_PLANNING_TIMEOUT_RECOVERED = "PLANNING_TIMEOUT_RECOVERED"
BUFFER_FACE_BACK_EXIT_CODE = 20
BUFFER_FACE_UNCERTAIN_EXIT_CODE = 21


def _apply_modular_entries_for_spawned_worker() -> None:
    """Restore module entrypoints when multiprocessing ``spawn`` reimports us.

    ``modular_dual_arm_loop.py`` changes these globals in its parent process.
    Spawned workers start from a fresh import, so the environment flag is the
    durable handoff that prevents them from silently returning to legacy paths.
    """
    if os.environ.get("PARM_MODULAR_LOOP_ENTRIES") != "1":
        return
    module_root = LOOP_ROOT / "modules"
    right_place_wiggle = (
        module_root / "module_9_right_place/place_hole_a_wiggle_new_config.py"
    )
    globals().update({
        "PYTHON": LOOP_ROOT / ".venv/bin/python",
        "GRASPNET_FLOW_WORKDIR": (
            module_root / "module_1_left_grasp/module_1_1_graspnet"
        ),
        "GRASPNET_FLOW": (
            module_root
            / "module_1_left_grasp/module_1_1_graspnet/"
            "run_grasp_to_inspect_place.sh"
        ),
        "LEFT_FLOW_SCRIPTS": {
            "apart": module_root / "module_2_left_inspect_apart/run_inspect_then_place.py",
        },
        "LEFT_PICK": module_root / "module_5_left_pick/run_head_judge_then_pick.py",
        "LEFT_HANDOFF_PREP": (
            module_root / "module_5_left_pick/module_5_2_handoff_prepare/main.py"
        ),
        "RIGHT_PUSH_TO_Q1": (
            MODULAR_RIGHT_BUFFER_PIPELINE
        ),
        "RIGHT_LEFT_HANDOFF": (
            module_root / "module_8_right_grasp/module_8_2_left_handoff/main.py"
        ),
        "RIGHT_PLACE_WIGGLE": right_place_wiggle,
        "RIGHT_FINISH": module_root / "module_10_right_idle/main.py",
    })


_apply_modular_entries_for_spawned_worker()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--head-host", default="127.0.0.1",
                        help="兼容参数；云台动作由对应业务模块自行管理")
    parser.add_argument("--right-idle-pose", type=Path,
                        default=Path("/home/dainanjun/zhuyiming/push_T/saved_poses/3.json"))
    parser.add_argument("--left-cycle-limit", type=int, default=0,
                        help="左臂循环次数；0 表示持续运行")
    parser.add_argument(
        "--left-only", action="store_true",
        help=(
            "只运行左臂一个成功工件周期：不打开/移动/取消右臂或右夹爪，"
            "正面/侧面放到buffer后结束，反面到左手交接准备位后结束"
        ),
    )
    parser.add_argument(
        "--left-flow",
        choices=tuple(LEFT_FLOW_SCRIPTS),
        default="apart",
        help="左臂 GraspNet 后处理流程：apart=拆分后的 Inspect + Left Place",
    )
    parser.add_argument("--right-speed", type=float, default=0.3)
    parser.add_argument("--prepare-speed", type=float, default=0.3,
                        help="启动前左 graspup、右 Idle 的规划速度比例")
    parser.add_argument(
        "--front-regrasp-pregrasp-height-m", type=float, default=0.05,
        help="正面桌面重抓预抓位相对 ICP 的 body_base +Z 高度，默认0.05m",
    )
    parser.add_argument(
        "--front-regrasp-descent-distance-m", type=float, default=0.022,
        help="正面桌面重抓从预抓位 LIN 下移距离，默认0.02m",
    )
    parser.add_argument("--require-confirmation", action="store_true",
                        help="右臂交接及最终放置也逐步要求输入 y；默认沿用现有自动交接")
    parser.add_argument("--runtime-dir", type=Path,
                        help="本次跨进程握手文件目录；默认写入 loop/output")
    return parser.parse_args()


def status_label(value: int) -> str:
    return {
        BUFFER_EMPTY: "0=EMPTY",
        BUFFER_TABLE_READY: "1=TABLE_READY",
        BUFFER_HANDOFF_READY: "2=HANDOFF_READY",
    }.get(value, f"UNKNOWN({value})")


def get_status(buffer_state: mp.Value) -> int:
    with buffer_state.get_lock():
        return int(buffer_state.value)


def set_status(buffer_state: mp.Value, value: int, reason: str) -> None:
    with buffer_state.get_lock():
        previous = int(buffer_state.value)
        buffer_state.value = int(value)
    print(f"[buffer] {status_label(previous)} -> {status_label(value)}: {reason}", flush=True)


def publish_buffer_workpiece(runtime_dir: Path, workpiece_id: str, branch: str) -> None:
    """Publish the one Buffer item's identity for the independent right worker."""
    path = runtime_dir / BUFFER_WORKPIECE_FILE
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"workpiece_id": workpiece_id, "branch": branch}),
        encoding="utf-8",
    )
    temporary.replace(path)


def current_buffer_workpiece(runtime_dir: Path) -> str:
    path = runtime_dir / BUFFER_WORKPIECE_FILE
    if not path.is_file():
        raise RuntimeError(f"Buffer 已非空但缺少工件计时标记: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    workpiece_id = payload.get("workpiece_id")
    if not isinstance(workpiece_id, str) or not workpiece_id:
        raise RuntimeError(f"Buffer 工件计时标记无效: {payload}")
    return workpiece_id


def clear_buffer_workpiece(runtime_dir: Path, workpiece_id: str) -> None:
    path = runtime_dir / BUFFER_WORKPIECE_FILE
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("workpiece_id") != workpiece_id:
        raise RuntimeError(f"拒绝清除其他工件的 Buffer 标记: {payload}")
    path.unlink()


def right_servo_is_active(right_servo_active: mp.Value) -> bool:
    """左臂只读右臂 ServoPose 会话状态，右臂 worker 是唯一写入方。"""
    with right_servo_active.get_lock():
        return bool(right_servo_active.value)


def set_right_servo_active(right_servo_active: mp.Value, active: bool) -> None:
    """记录右臂 ServoPose 状态，仅用于日志，不再作为运动互斥锁。"""
    with right_servo_active.get_lock():
        right_servo_active.value = bool(active)
    print(
        f"[right-servo] {'ACTIVE：左臂 Planner 运动允许并行' if active else 'IDLE'}",
        flush=True,
    )


def right_is_waiting_for_observation(right_waiting_for_observation: mp.Value) -> bool:
    """左臂只读：右臂是否已完成前置动作、正在等下一轮观察位通知。"""
    with right_waiting_for_observation.get_lock():
        return bool(right_waiting_for_observation.value)


def set_right_waiting_for_observation(
    right_waiting_for_observation: mp.Value, waiting: bool,
) -> None:
    """仅由右臂 worker 写入，避免首轮右臂 Idle 时出现观察事件死锁。"""
    with right_waiting_for_observation.get_lock():
        right_waiting_for_observation.value = bool(waiting)


def wait_for_right_servo_idle(
    right_servo_active: mp.Value, stop_event: mp.Event, label: str,
) -> None:
    """兼容旧调用：左臂动作不再等待右臂 ServoPose。"""
    del right_servo_active, label
    if stop_event.is_set():
        raise InterruptedError("左臂动作前收到停止请求")


def publish_allow_file(path: Path) -> None:
    """Atomically publish a cross-process motion permit."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("allow\n", encoding="utf-8")
    temporary.replace(path)


def load_right_idle_joints(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    joints = (payload.get("arms") or {}).get("right", {}).get("joint_positions_deg")
    if not isinstance(joints, list) or len(joints) != 7:
        raise RuntimeError(f"右臂 Idle 文件缺少7个关节角: {path}")
    return [float(value) for value in joints]


def wait_for_joint_target(
    arm_host: str,
    arm_id: int,
    target_deg: list[float],
    label: str,
    *,
    tolerance_deg: float = 0.5,
    timeout_s: float = 90.0,
) -> None:
    """Do not treat the asynchronous planner execute acknowledgement as physical arrival."""
    from parm_sdk import Arm
    import numpy as np

    arm_key = "left" if arm_id == 1 else "right"
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            per_arm = (state.get("state") or {}).get(arm_key) or {}
            actual = np.asarray(per_arm.get("position") or [], dtype=float)
            if actual.shape == (7,):
                delta = (actual - np.asarray(target_deg, dtype=float) + 180.0) % 360.0 - 180.0
                last_error_deg = float(np.max(np.abs(delta)))
                if last_error_deg <= tolerance_deg:
                    print(f"[prepare] {label} 已到位，最大关节误差 {last_error_deg:.3f}deg", flush=True)
                    return
            time.sleep(0.05)
    raise RuntimeError(
        f"{label} 未在 {timeout_s:.1f}s 内到位；"
        f"最后最大关节误差={last_error_deg:.3f}deg"
    )


def current_joint_target_error_deg(
    arm_host: str, arm_id: int, target_deg: list[float]
) -> float:
    """Read one live joint state so a zero-length RRTStar request is never sent."""
    from parm_sdk import Arm
    import numpy as np

    arm_key = "left" if arm_id == 1 else "right"
    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
    actual = np.asarray(
        ((state.get("state") or {}).get(arm_key) or {}).get("position") or [],
        dtype=float,
    )
    if actual.shape != (7,):
        raise RuntimeError(f"无法读取{arm_key}臂的7维关节状态: {state}")
    delta = (actual - np.asarray(target_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(delta)))


def open_both_grippers_for_prepare(args: argparse.Namespace) -> None:
    """启动准备的第一步：确保两侧夹爪张开，避免带着旧夹持状态开始循环。"""
    from parm_sdk import Gripper
    from modules.module_0_prepare.gripper_state_compat import is_gripper_usable

    left_only = bool(getattr(args, "left_only", False))
    print(
        "[prepare] 仅打开左夹爪。" if left_only else "[prepare] 同步打开左、右夹爪。",
        flush=True,
    )
    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        def wait_for_usable_state(hand: int, label: str, phase: str) -> dict:
            """等待新驱动 PUB 从 initializing/executing 切换到稳定终态。"""
            deadline = time.monotonic() + 8.0
            latest = gripper.get_state().get(hand)
            while not is_gripper_usable(latest):
                motion_state = str((latest or {}).get("motion_state", "")).lower()
                if motion_state in {"error", "dropped"}:
                    raise RuntimeError(
                        f"{label}夹爪 {phase} 时进入异常状态: {latest}"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{label}夹爪 {phase} 后等待稳定状态超时: {latest}"
                    )
                time.sleep(0.1)
                latest = gripper.get_state().get(hand)
            return latest

        hands = ((1, "左"),) if left_only else ((1, "左"), (2, "右"))
        for hand, label in hands:
            wait_for_usable_state(hand, label, "启动预检")
            response = gripper.move(
                hand,
                100.0,
                force=100.0,
                speed=80.0 if hand == 1 else None,
                mode="sync",
                preempt=False,
                timeout=60.0,
            )
            final_state = gripper.get_state().get(hand)
            device_id = "left_gripper" if hand == 1 else "right_gripper"
            response_devices = (
                (response.get("data") or response).get("devices") or {}
                if isinstance(response, dict)
                else {}
            )
            response_state = response_devices.get(device_id) or {}
            response_interfaces = response_state.get("state_interfaces") or {}
            response_position = response_interfaces.get("gripper/position", -1.0)
            response_arrived = response_state.get("device_state") == "arrived"
            pub_arrived = (
                isinstance(final_state, dict)
                and final_state.get("motion_state") == "arrived"
            )
            final_position = float(
                (final_state or {}).get("position", response_position)
            )
            if (
                not (pub_arrived or response_arrived)
                or final_position < 95.0
            ):
                raise RuntimeError(
                    f"{label}夹爪打开后未确认到位: response={response}, "
                    f"state={final_state}"
                )
            if not pub_arrived and response_arrived:
                print(
                    f"[prepare] {label}夹爪同步响应已到位，"
                    "等待下一帧 gripper.state 更新。",
                    flush=True,
                )
            print(
                f"[prepare] {label}夹爪已打开：position={final_position:.1f}%",
                flush=True,
            )


def prepare_both_arms(args: argparse.Namespace) -> None:
    """先打开两侧夹爪，再完成右 Idle 与左 graspup 的启动准备。"""
    open_both_grippers_for_prepare(args)
    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.prepare_speed),
        max_acceleration=float(args.prepare_speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={
            "body_base", "left_hand_transfer_end", "right_hand_transfer_end",
        })
        right_idle = load_right_idle_joints(args.right_idle_pose)
        right_error_deg = current_joint_target_error_deg(args.arm_host, 2, right_idle)
        if right_error_deg <= 0.5:
            print(
                f"[prepare] 右臂已在 Idle，跳过所有右臂启动运动"
                f"（误差 {right_error_deg:.3f}deg）。",
                flush=True,
            )
        else:
            print("[prepare] 尝试用关节 RRTConnect(OMPL) 规划右臂到 Idle。", flush=True)
            try:
                direct_idle_plan = planner.plan_joint_trajectory(
                    "right", right_idle, planner="ompl"
                )
            except Exception as direct_plan_error:
                print(
                    "[prepare] 右臂直接到 Idle 规划失败，改走启动过渡位："
                    f"{direct_plan_error}",
                    flush=True,
                )
                transition = reverse_handoff_prepare_joints_deg()
                transition_error_deg = current_joint_target_error_deg(
                    args.arm_host, 2, transition
                )
                if transition_error_deg <= 0.5:
                    print(
                        "[prepare] 右臂已在启动过渡位，跳过第一段零长度 RRTStar"
                        f"（误差 {transition_error_deg:.3f}deg）。",
                        flush=True,
                    )
                else:
                    planner.move_right_joints(transition)
                    wait_for_joint_target(
                        args.arm_host,
                        2,
                        transition,
                        "右臂启动过渡位",
                    )
                print(
                    "[prepare] 右臂启动过渡位已到达；重新用关节 RRTConnect(OMPL) "
                    "规划到 Idle。",
                    flush=True,
                )
                planner.move_right_joints(right_idle, planner="ompl")
            else:
                if direct_idle_plan is None:
                    raise RuntimeError("实时模式下右臂直接到 Idle 的规划结果为空")
                print(
                    "[prepare] 右臂可直接规划到 Idle；跳过启动过渡位并执行。",
                    flush=True,
                )
                planner.execute_planned_result(direct_idle_plan)
            wait_for_joint_target(args.arm_host, 2, right_idle, "右臂 Idle")
        print("[prepare] 右臂已到 Idle；规划左臂 RRTStar 到 graspup。", flush=True)
        left_error_deg = current_joint_target_error_deg(
            args.arm_host, 1, LEFT_GRASPUP_JOINTS_DEG
        )
        if left_error_deg <= 0.5:
            print(
                f"[prepare] 左臂已在 graspup，跳过零长度 RRTStar（误差 {left_error_deg:.3f}deg）。",
                flush=True,
            )
        else:
            planner.move_left_joints(LEFT_GRASPUP_JOINTS_DEG)
            wait_for_joint_target(args.arm_host, 1, LEFT_GRASPUP_JOINTS_DEG, "左臂 graspup")
        print("[prepare] 左右臂准备完成，开始 loop。", flush=True)
    finally:
        planner.close()


def cancel_both_arms(arm_host: str, reason: str) -> None:
    """Use independent command-only SDK clients, mirroring data_collector's stop path."""
    left_only = os.environ.get("PARM_LOOP_LEFT_ONLY") == "1"
    targets = ((1, "left"),) if left_only else ((1, "left"), (2, "right"))
    target_label = "left cancel_task()" if left_only else "left/right cancel_task()"
    print(f"[stop] {reason}: 直接发送 {target_label}", flush=True)
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(f"[stop] 无法导入 parm_sdk，未能发送 cancel_task(): {exc}", flush=True)
        return
    for arm_id, label in targets:
        arm = None
        try:
            arm = Arm(host=arm_host, status_port=None)
            arm.cancel_task(arm_id)
            print(f"[stop] {label} arm cancel_task({arm_id}) 已发送", flush=True)
        except Exception as exc:
            print(f"[stop] {label} arm cancel_task 失败: {exc}", flush=True)
        finally:
            if arm is not None:
                try:
                    arm.close()
                except Exception:
                    pass


def terminate_process_group(process: subprocess.Popen[object], label: str) -> None:
    if process.poll() is not None:
        return
    print(f"[stop] 终止 {label} 子流程 pid={process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_command(
    command: list[str], *, cwd: Path, stop_event: mp.Event, arm_host: str,
    label: str, env: dict[str, str] | None = None,
    recoverable_exit_codes: frozenset[int] = frozenset(),
) -> int:
    """运行一个业务模块，并把可恢复失败显式返回给状态机。"""
    module = Path(command[1] if len(command) > 1 else command[0]).stem
    report_module_status(module, ResultStatus.WAIT, f"开始：{label}")
    print("\n[run] " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, start_new_session=True,
    )
    try:
        while process.poll() is None:
            if stop_event.wait(0.2):
                cancel_both_arms(arm_host, f"停止 {label}")
                terminate_process_group(process, label)
                raise InterruptedError(f"{label} 被停止请求中断")
    except KeyboardInterrupt:
        cancel_both_arms(arm_host, f"Ctrl+C 停止 {label}")
        terminate_process_group(process, label)
        raise
    if process.returncode != 0:
        if process.returncode in recoverable_exit_codes:
            report_module_status(
                module,
                ResultStatus.RETRY,
                f"{label}未取得工件；保持当前安全位并回退到本模块重试",
                reason_code=f"EXIT_{process.returncode}",
            )
            return process.returncode
        report_module_status(
            module,
            ResultStatus.FAULT,
            f"{label}失败，exit={process.returncode}",
            reason_code=f"EXIT_{process.returncode}",
        )
        raise RuntimeError(f"{label} 失败，exit={process.returncode}")
    report_module_status(module, ResultStatus.SUCCESS, f"完成：{label}")
    return 0


def check_reversed_workpiece_on_table(
    args: argparse.Namespace, stop_event: mp.Event, cycle_dir: Path,
) -> tuple[str, Path]:
    """在反面放桌后复用 Module 5 的 ICP，返回可靠 FRONT/BACK 判定。"""
    calibration = json.loads(HEAD_CALIBRATION.read_text(encoding="utf-8"))
    pose = calibration["head_pose_deg"]
    run_command(
        [
            str(PYTHON), str(HEAD_MOTION), "--execute", "--host", "127.0.0.1",
            "--pitch", str(pose["pitch"]), "--yaw", str(pose["yaw"]),
        ],
        cwd=LOOP_ROOT, stop_event=stop_event, arm_host=args.arm_host,
        label="Buffer Check 头部观察位",
    )
    result_file = cycle_dir / "buffer_face_check.json"
    exit_code = run_command(
        [
            str(PYTHON), str(BUFFER_FACE_CHECK), "--result-file", str(result_file),
        ],
        cwd=BUFFER_FACE_CHECK.parent, stop_event=stop_event, arm_host=args.arm_host,
        label="Buffer Check 正反面 ICP 复检",
        recoverable_exit_codes=frozenset({
            BUFFER_FACE_BACK_EXIT_CODE,
            BUFFER_FACE_UNCERTAIN_EXIT_CODE,
        }),
    )
    if not result_file.is_file():
        raise RuntimeError("Buffer Check 结束但未发布本轮结果文件")
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    decision = str(payload.get("decision") or "")
    expected_exit = {
        "FRONT": 0,
        "BACK": BUFFER_FACE_BACK_EXIT_CODE,
        "UNCERTAIN": BUFFER_FACE_UNCERTAIN_EXIT_CODE,
    }.get(decision)
    if expected_exit is None or exit_code != expected_exit:
        raise RuntimeError(
            "Buffer Check 退出码与结果不一致："
            f"exit={exit_code}, decision={decision}, payload={payload}"
        )
    print(f"[left] Buffer Check 判定={decision}；结果={result_file}", flush=True)
    return decision, result_file


def regrasp_front_and_place_to_left_buffer(
    args: argparse.Namespace, stop_event: mp.Event, cycle_dir: Path,
) -> bool:
    """正面拨正后重抓，并让 Module 3 回到既有正面左侧放置位。"""
    face_result = cycle_dir / "buffer_face_check.json"
    regrasp_result = cycle_dir / "front_buffer_regrasp_result.json"
    place_result = cycle_dir / "front_buffer_replacement_place_result.json"
    workpiece = os.environ.get("PARM_LOOP_WORKPIECE", "1")
    exit_code = run_command(
        [
            str(PYTHON), str(FRONT_BUFFER_REGRASP), "--execute",
            "--buffer-check-result", str(face_result), "--result-file", str(regrasp_result),
            "--planner-host", args.planner_host, "--arm-host", args.arm_host,
            "--gripper-host", args.gripper_host, "--speed", str(args.prepare_speed),
            "--pregrasp-height-m", str(args.front_regrasp_pregrasp_height_m),
            "--descent-distance-m", str(args.front_regrasp_descent_distance_m),
        ],
        cwd=LOOP_ROOT, stop_event=stop_event, arm_host=args.arm_host,
        label="正面工件 ICP 参考重抓",
        recoverable_exit_codes=frozenset({NO_GRASP_EXIT_CODE}),
    )
    if exit_code == NO_GRASP_EXIT_CODE:
        return False
    payload = json.loads(regrasp_result.read_text(encoding="utf-8"))
    if payload.get("outcome") != "success" or payload.get("classification") != "FRONT":
        raise RuntimeError(f"正面重抓结果无效: {payload}")
    run_command(
        [
            str(PYTHON), str(LEFT_PLACE), "--inspection-result", str(regrasp_result),
            "--result-file", str(place_result), "--workpiece", workpiece,
            "--planner-host", args.planner_host, "--arm-host", args.arm_host,
            "--gripper-host", args.gripper_host, "--preset-speed", str(args.prepare_speed),
            "--motion-speed", str(args.prepare_speed), "--auto-confirm-motion",
        ],
        cwd=LOOP_ROOT, stop_event=stop_event, arm_host=args.arm_host,
        label="Module 3 回原左手正面放置位",
    )
    placement = json.loads(place_result.read_text(encoding="utf-8"))
    if placement.get("placement_branch") != "BRANCH_1_FRONT":
        raise RuntimeError(
            "正面恢复后 Module 3 未放到原正面左侧位置："
            f"{placement}"
        )
    return True


def wait_for_module2_ready(
    ready_file: Path, permit_file: Path, grasp_permit_file: Path,
    observation_ready_file: Path, left_observation_ready: mp.Event,
    buffer_state: mp.Value, right_servo_active: mp.Value,
    right_waiting_for_observation: mp.Value,
    stop_event: mp.Event, arm_host: str, process: subprocess.Popen[object],
) -> str:
    """右臂非 ServoPose 时放行真实抓取；最终放置仍须确认 buffer=0。"""
    branch: str | None = None
    grasp_permitted = False
    observation_published = False
    while process.poll() is None:
        if stop_event.wait(0.2):
            cancel_both_arms(arm_host, "停止左臂 inspect")
            terminate_process_group(process, "左臂 inspect")
            raise InterruptedError("左臂 inspect 被停止请求中断")
        if not observation_published and observation_ready_file.is_file():
            content = observation_ready_file.read_text(encoding="utf-8").strip().lower()
            if content != "ready":
                raise RuntimeError(
                    f"左臂 GraspNet 观察位标记内容无效: {content!r}"
                )
            left_observation_ready.set()
            observation_published = True
            print(
                "[left] 已确认左臂到达 GraspNet 观察位；右臂 ServoPose 可以启动。",
                flush=True,
            )
        # The observation pose is the last left-arm action allowed before
        # right ServoPose starts.  After it is published, only the right
        # worker's ServoPose flag gates the next real GraspNet motion.
        if not grasp_permitted and observation_published:
            # Only later cycles have a right worker waiting for this marker.
            # In the first cycle right is idle, so no consumer exists and the
            # grasp permit must not wait for the marker to be cleared.
            observation_consumed = not left_observation_ready.is_set()
            right_waiting = right_is_waiting_for_observation(right_waiting_for_observation)
            if not right_waiting or observation_consumed:
                publish_allow_file(grasp_permit_file)
                grasp_permitted = True
                print(
                    "[left] 右臂未处于 ServoPose，已允许 GraspNet 执行预测出的真实抓取。",
                    flush=True,
                )
        if branch is None and ready_file.is_file():
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
            if payload.get("outcome") == "insufficient_point_cloud_discarded":
                print(
                    "[left] inspect 点云不足，已回graspup并松开夹爪；"
                    f"有效点数={payload.get('valid_point_count')}，"
                    f"阈值={payload.get('point_threshold')}。"
                    "本轮正常结束，不等待buffer许可。",
                    flush=True,
                )
                return MODULE2_INSUFFICIENT_POINTS_DISCARDED
            if payload.get("outcome") == "pc1_pc2_registration_failed_recovered":
                print(
                    "[left] inspect 在PC1、PC2均未得到可信位姿，已回graspup并松开夹爪；"
                    "本轮正常结束，不等待buffer许可。",
                    flush=True,
                )
                return MODULE2_REGISTRATION_FAILED_RECOVERED
            if payload.get("outcome") == "held_object_dropped":
                print(
                    "[left] inspect运动前重量复检发现工件已掉落；"
                    "左臂已回graspup，本轮正常结束。",
                    flush=True,
                )
                return MODULE2_HELD_OBJECT_DROPPED
            if payload.get("outcome") == "planning_timeout_recovered_at_graspup":
                print(
                    "[left] inspect 最终放置规划超时，已回graspup并松开夹爪；"
                    "本轮正常结束，不等待buffer许可。",
                    flush=True,
                )
                return MODULE2_PLANNING_TIMEOUT_RECOVERED
            branch = str(payload.get("placement_branch") or "")
            if branch not in {"BRANCH_1_FRONT", "BRANCH_1_REVERSED", "BRANCH_1_DIRECT"}:
                raise RuntimeError(f"inspect 返回未知放置分支: {payload}")
            print(
                f"[left] Left Place 已在 graspup 上方完成最终规划；分支={branch}，"
                f"当前 {status_label(get_status(buffer_state))}",
                flush=True,
            )
        if branch is not None and get_status(buffer_state) == BUFFER_EMPTY:
            publish_allow_file(permit_file)
            print(
                "[left] 缓冲区状态=0；已允许 Left Place 执行最终放置。",
                flush=True,
            )
            return branch
    exit_code = process.returncode
    if exit_code == 0:
        if ready_file.is_file():
            # The child can exit between polling intervals.  Consume its
            # already-written result rather than treating a successful exit
            # as a controller failure.
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
            if payload.get("outcome") == "insufficient_point_cloud_discarded":
                print(
                    "[left] Inspect 已正常完成低点云丢弃；"
                    "子进程先于轮询退出，已从结果文件确认。",
                    flush=True,
                )
                return MODULE2_INSUFFICIENT_POINTS_DISCARDED
            if payload.get("outcome") == "pc1_pc2_registration_failed_recovered":
                print(
                    "[left] Inspect 已完成PC1/PC2配准失败恢复；"
                    "子进程先于轮询退出，已从结果文件确认。",
                    flush=True,
                )
                return MODULE2_REGISTRATION_FAILED_RECOVERED
            if payload.get("outcome") == "held_object_dropped":
                print(
                    "[left] inspect已正常完成工件掉落恢复；"
                    "子进程先于轮询退出，已从结果文件确认。",
                    flush=True,
                )
                return MODULE2_HELD_OBJECT_DROPPED
            if payload.get("outcome") == "planning_timeout_recovered_at_graspup":
                print(
                    "[left] Left Place 已正常完成规划超时恢复；"
                    "子进程先于轮询退出，已从结果文件确认。",
                    flush=True,
                )
                return MODULE2_PLANNING_TIMEOUT_RECOVERED
            branch = str(payload.get("placement_branch") or "")
            if branch in {"BRANCH_1_FRONT", "BRANCH_1_REVERSED", "BRANCH_1_DIRECT"}:
                print(
                    "[left] inspect 子进程已正常结束；"
                    f"已从结果文件恢复分支={branch}。",
                    flush=True,
                )
                return branch
            raise RuntimeError(f"inspect 结果文件缺少有效分支: {payload}")
        raise RuntimeError(
            "左臂 inspect 以exit=0结束但未发布结果文件；"
            "拒绝把未确认的退出误判为规划超时恢复。"
        )
    if exit_code == NO_GRASP_EXIT_CODE:
        raise NoGraspPredicted(
            "GraspNet 没有有效抓取候选；没有发送抓取运动命令。"
        )
    raise RuntimeError(
        "左臂 graspnet->Module 2 子流程在 Module 2 发布 graspup 分支前退出；"
        f"exit={exit_code}。请先检查上方 GraspNet 的真实抓取确认、"
        "夹爪和轨迹执行日志，而不是把它当作 inspect 故障。"
    )


class NoGraspPredicted(RuntimeError):
    """GraspNet 已耗尽候选，但机械臂和硬件没有发生故障。"""


def run_left_grasp_inspect_place_cycle(
    args: argparse.Namespace, buffer_state: mp.Value, stop_event: mp.Event,
    cycle_dir: Path, left_observation_ready: mp.Event, right_servo_active: mp.Value,
    right_waiting_for_observation: mp.Value, workpiece_id: str,
) -> str | None:
    report_module_status(
        "module_1_to_3_left_pipeline",
        ResultStatus.WAIT,
        "开始 Module 1 抓取、Module 2 检测和 Module 3 放置链",
    )
    ready_file = cycle_dir / "module2_ready.json"
    permit_file = cycle_dir / "module2_permit"
    grasp_permit_file = cycle_dir / "grasp_execution_permit"
    observation_ready_file = cycle_dir / "grasp_observation_ready"
    environment = os.environ.copy()
    environment["LOOP_AUTO_CONFIRM_MOTION"] = "1"
    environment["GRASPNET_V7_ARM_HOST"] = args.arm_host
    environment["PARM_PLANNER_HOST"] = args.planner_host
    environment["PARM_GRIPPER_HOST"] = args.gripper_host
    environment["LOOP_GRASPNET_EXECUTION_PERMIT_FILE"] = str(grasp_permit_file)
    environment["PARM_LOOP_ROOT"] = str(LOOP_ROOT)
    environment["LOOP_GRASPNET_OBSERVATION_READY_FILE"] = str(
        observation_ready_file
    )
    environment["MODULE2_PLACEMENT_READY_FILE"] = str(ready_file)
    environment["MODULE2_PLACEMENT_PERMIT_FILE"] = str(permit_file)
    environment["MODULE2_SCRIPT"] = str(LEFT_FLOW_SCRIPTS[args.left_flow])
    environment[WORKPIECE_ENV] = workpiece_id
    wait_for_right_servo_idle(right_servo_active, stop_event, "左臂下一轮 GraspNet 观察动作")
    command = ["bash", str(GRASPNET_FLOW)]
    print("\n[run] " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=GRASPNET_FLOW_WORKDIR, env=environment,
                               start_new_session=True)
    try:
        branch = wait_for_module2_ready(
            ready_file, permit_file, grasp_permit_file, observation_ready_file,
            left_observation_ready, buffer_state, right_servo_active,
            right_waiting_for_observation,
            stop_event, args.arm_host, process,
        )
        while process.poll() is None:
            if stop_event.wait(0.2):
                cancel_both_arms(args.arm_host, "停止左臂抓取/inspect/放置")
                terminate_process_group(process, "左臂抓取/inspect/放置")
                raise InterruptedError("左臂流程被停止请求中断")
    except KeyboardInterrupt:
        cancel_both_arms(args.arm_host, "Ctrl+C 停止左臂流程")
        terminate_process_group(process, "左臂抓取/inspect/放置")
        raise
    except NoGraspPredicted:
        # The child has exited already.  It never reached Module 2 and has
        # not sent any grasp command, so the caller can retry later without
        # sending an additional left-arm motion.
        report_module_status(
            "module_1_to_3_left_pipeline",
            ResultStatus.RETRY,
            "Module 1 未预测到可抓取姿态；保持当前位置，重新预测",
            reason_code="NO_GRASP_PREDICTED",
        )
        return None
    if process.returncode != 0:
        report_module_status(
            "module_1_to_3_left_pipeline",
            ResultStatus.FAULT,
            f"左侧抓取/检测/放置链失败，exit={process.returncode}",
            reason_code=f"EXIT_{process.returncode}",
        )
        raise RuntimeError(
            f"左臂 GraspNet->Inspect->Left Place 失败，exit={process.returncode}"
        )
    report_module_status(
        "module_1_to_3_left_pipeline",
        ResultStatus.SUCCESS,
        f"左侧链完成，返回分支={branch}",
    )
    return branch


def move_left_to_middle_observation(
    args: argparse.Namespace, stop_event: mp.Event, right_servo_active: mp.Value,
) -> None:
    """撤离桌面并到反抓链已标定的中间观察位，确认后才允许头部采集。"""
    if stop_event.is_set():
        raise InterruptedError("停止请求已到达，跳过左臂回中间观察位")
    wait_for_right_servo_idle(right_servo_active, stop_event, "左臂回中间观察位")

    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.prepare_speed),
        max_acceleration=float(args.prepare_speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        error_deg = current_joint_target_error_deg(
            args.arm_host, 1, LEFT_MIDDLE_OBSERVATION_JOINTS_DEG
        )
        if error_deg <= 0.5:
            print(
                f"[left] 已在中间观察位，跳过零长度 RRTStar（误差 {error_deg:.3f}deg）。",
                flush=True,
            )
            return
        print("[left] 规划左臂 RRTStar 撤离到中间观察位。", flush=True)
        planner.move_left_joints(LEFT_MIDDLE_OBSERVATION_JOINTS_DEG, planner="rrtstar")
        wait_for_joint_target(
            args.arm_host, 1, LEFT_MIDDLE_OBSERVATION_JOINTS_DEG,
            "左臂中间观察位",
        )
    finally:
        planner.close()


def move_left_to_graspup(
    args: argparse.Namespace, stop_event: mp.Event, right_servo_active: mp.Value,
) -> None:
    """交接结束后先经中间观察位，再送左臂回 graspup。"""
    error_deg = current_joint_target_error_deg(
        args.arm_host, 1, LEFT_GRASPUP_JOINTS_DEG
    )
    if error_deg <= 0.5:
        print(
            f"[left] 已在 graspup，跳过零长度 RRTStar（误差 {error_deg:.3f}deg）。",
            flush=True,
        )
        return
    move_left_to_middle_observation(args, stop_event, right_servo_active)
    if stop_event.is_set():
        raise InterruptedError("左臂到达中间观察位后收到停止请求")
    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base", left_group="left_arm", right_group="right_arm",
        left_tool_frame="left_hand_transfer_end", right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.prepare_speed), max_acceleration=float(args.prepare_speed),
        idle_poll_seconds=0.2, live=True, poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        print("[left] 中间观察位已到位：规划左臂 RRTStar 到 graspup。", flush=True)
        planner.move_left_joints(LEFT_GRASPUP_JOINTS_DEG, planner="rrtstar")
        wait_for_joint_target(args.arm_host, 1, LEFT_GRASPUP_JOINTS_DEG, "左臂 graspup")
    finally:
        planner.close()


def move_right_to_reverse_handoff_prep(
    args: argparse.Namespace, stop_event: mp.Event,
) -> None:
    """反面分支从 Idle 出发时，先用规划器 RRTStar 到右臂预交接关节位。"""
    if stop_event.is_set():
        raise InterruptedError("停止请求已到达，跳过右臂反面交接预备位")
    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.right_speed),
        max_acceleration=float(args.right_speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "right_hand_transfer_end"})
        target = reverse_handoff_prepare_joints_deg()
        error_deg = current_joint_target_error_deg(
            args.arm_host, 2, target
        )
        if error_deg <= 0.5:
            print(
                "[right] 已在反面交接预备关节位，跳过零长度 RRTStar"
                f"（误差 {error_deg:.3f}deg）。",
                flush=True,
            )
            return
        print(
            "[right] 主分支2：从 Idle 规划右臂关节 RRTStar 到反面交接预备位，"
            f"target(deg)={target}",
            flush=True,
        )
        planner.move_right_joints(target, planner="rrtstar")
        wait_for_joint_target(
            args.arm_host,
            2,
            target,
            "右臂反面交接预备位",
        )
        if stop_event.is_set():
            raise InterruptedError("右臂反面交接预备位到达后收到停止请求")
    finally:
        planner.close()


def left_worker(
    args: argparse.Namespace, buffer_state: mp.Value, stop_event: mp.Event,
    fault_queue: mp.Queue, runtime_dir: Path, left_done: mp.Event,
    handoff_left_clear: mp.Event, left_needs_graspup: mp.Event,
    left_observation_ready: mp.Event, right_servo_active: mp.Value,
    right_waiting_for_observation: mp.Value,
) -> None:
    try:
        completed = 0
        active_workpiece_id: str | None = None
        while not stop_event.is_set():
            # 状态1/2允许左臂回观察位并完成预测/规划，但会在真实抓取前拦住。
            # 反向交接例外：必须先由交接脚本把左臂安全回到统一较高安全位。
            while not handoff_left_clear.is_set():
                if stop_event.wait(0.2):
                    return
            if left_needs_graspup.is_set():
                move_left_to_graspup(args, stop_event, right_servo_active)
                left_needs_graspup.clear()
            completed += 1
            if args.left_cycle_limit and completed > args.left_cycle_limit:
                print("[left] 已达到 left-cycle-limit，停止启动新循环。", flush=True)
                left_done.set()
                return
            cycle_dir = runtime_dir / f"left_cycle_{completed:04d}"
            cycle_dir.mkdir(parents=True, exist_ok=False)
            active_workpiece_id = f"{runtime_dir.name}_{cycle_dir.name}"
            begin_workpiece_timer(
                active_workpiece_id,
                metadata={"left_cycle": completed, "cycle_dir": str(cycle_dir)},
            )
            os.environ[WORKPIECE_ENV] = active_workpiece_id
            left_observation_ready.clear()
            print(f"\n[left] 开始循环 {completed}", flush=True)
            branch = run_left_grasp_inspect_place_cycle(
                args, buffer_state, stop_event, cycle_dir, left_observation_ready,
                right_servo_active, right_waiting_for_observation, active_workpiece_id,
            )

            if branch is None:
                complete_workpiece_timer(active_workpiece_id, outcome="NO_GRASP")
                print(
                    "[left] GraspNet 没有有效抓取候选；不发送左臂回程指令，"
                    "左臂保持当前位置，立即重启下一轮预测。",
                    flush=True,
                )
                continue

            if branch == MODULE2_INSUFFICIENT_POINTS_DISCARDED:
                complete_workpiece_timer(active_workpiece_id, outcome="INSUFFICIENT_POINT_CLOUD")
                print(
                    "[left] 本轮因inspect有效点云不足已正常丢弃；"
                    "保持现有缓冲区状态，直接开始下一轮。",
                    flush=True,
                )
                continue

            if branch == MODULE2_REGISTRATION_FAILED_RECOVERED:
                complete_workpiece_timer(
                    active_workpiece_id, outcome="PC1_PC2_REGISTRATION_FAILED"
                )
                # 这是本轮左臂手中工件的局部恢复。若右臂正在处理上一件
                # Buffer 工件，状态会是 TABLE_READY/HANDOFF_READY，绝不能
                # 因本轮 Inspect 失败而将那一件错误标成 EMPTY。
                report_module_status(
                    "module_2_left_inspect",
                    ResultStatus.RETRY,
                    "PC1/PC2配准失败已完成左臂安全恢复；保持现有缓冲区状态并从 Module 1 重新抓取",
                    reason_code="PC1_PC2_REGISTRATION_FAILED_RECOVERED",
                )
                print(
                    "[left] 本轮PC1/PC2均未得到可信位姿；"
                    "保持现有缓冲区状态，直接从 Module 1 开始下一轮。",
                    flush=True,
                )
                continue

            if branch == MODULE2_HELD_OBJECT_DROPPED:
                complete_workpiece_timer(active_workpiece_id, outcome="HELD_OBJECT_DROPPED")
                # Module 3 has already performed the physical recovery
                # (return to graspup and open the left gripper). 此处仅重置
                # 本轮左臂流程，不能清空右臂正在处理的旧 Buffer 项。
                report_module_status(
                    "force_single_workpiece_check",
                    ResultStatus.RETRY,
                    "力传感器未确认单件或检测到掉件；保持现有缓冲区状态，从 Module 1 重新开始",
                    reason_code="HELD_OBJECT_DROPPED",
                )
                print(
                    "[left] 本轮工件途中掉落或不再是单件；左臂状态已重置，"
                    "直接从 Module 1 开始下一轮 GraspNet。",
                    flush=True,
                )
                continue

            if branch == MODULE2_PLANNING_TIMEOUT_RECOVERED:
                complete_workpiece_timer(active_workpiece_id, outcome="PLANNING_TIMEOUT_RECOVERED")
                print(
                    "[left] 本轮最终放置规划超时后已完成回graspup/松爪恢复；"
                    "保持现有缓冲区状态，直接开始下一轮。",
                    flush=True,
                )
                continue

            if branch == "BRANCH_1_FRONT":
                left_observation_ready.clear()
                publish_buffer_workpiece(runtime_dir, active_workpiece_id, branch)
                set_status(buffer_state, BUFFER_TABLE_READY, f"左臂 {branch} 已放置并回 graspup")
                if bool(getattr(args, "left_only", False)):
                    complete_workpiece_timer(
                        active_workpiece_id,
                        outcome="LEFT_ONLY_BUFFER_READY",
                        detail=f"{branch}: 左臂已放到buffer；右臂未启动",
                    )
                    print(
                        "[left-only] 左臂已将工件放到buffer并回安全位；"
                        "右臂从未启动，本次单左臂流程结束。",
                        flush=True,
                    )
                    left_done.set()
                    return
                continue

            # 反面或 direct 放置都不能预先确认桌面朝向。先在与 Module 5-1
            # 相同的头部观察位复用 SAM2/CAD-ICP 复检：正面则重新抓取并回
            # 既有左侧正面 Buffer 位；反面才进入原反抓/交接链。
            decision, _ = check_reversed_workpiece_on_table(
                args, stop_event, cycle_dir
            )
            if decision == "UNCERTAIN":
                complete_workpiece_timer(
                    active_workpiece_id,
                    outcome="TABLE_OBJECT_LOST",
                    detail="Buffer Check 重试耗尽仍未识别到桌面工件",
                )
                report_module_status(
                    "module_4_head_judge",
                    ResultStatus.RETRY,
                    "Buffer Check 多次观察仍不确定；视为工件已掉落，"
                    "保持 Buffer 空并从 Module 1 重新抓取",
                    reason_code="BUFFER_OBJECT_LOST",
                )
                print(
                    "[left] Buffer Check 重试耗尽仍未看到工件；判定工件已掉落，"
                    "直接开始下一轮 Module 1 抓取。",
                    flush=True,
                )
                continue
            if decision == "FRONT":
                wait_for_right_servo_idle(
                    right_servo_active, stop_event, "左臂正面 ICP 重抓动作"
                )
                front_regrasp_attempt = 0
                front_recovered = False
                while not stop_event.is_set() and decision == "FRONT":
                    front_regrasp_attempt += 1
                    print(
                        "[left] 工件已被拨正为正面：按 ICP 参考位重抓，"
                        f"尝试 {front_regrasp_attempt}/{MAX_REVERSE_TABLE_GRASP_ATTEMPTS}。",
                        flush=True,
                    )
                    front_recovered = regrasp_front_and_place_to_left_buffer(
                        args, stop_event, cycle_dir
                    )
                    if front_recovered:
                        break
                    if front_regrasp_attempt >= MAX_REVERSE_TABLE_GRASP_ATTEMPTS:
                        raise RuntimeError(
                            "左臂桌面正面重抓已连续 5 次未确认夹到工件；"
                            "已停止重试并保持现场。"
                        )
                    # A failed ICP regrasp exits after its 3 cm verification
                    # lift.  It is not the head-observation posture: capturing
                    # there would let the arm/gripper occlude the tabletop.
                    # Return to the established middle observation pose and
                    # verify it before commanding the head or acquiring RGB-D.
                    move_left_to_middle_observation(
                        args, stop_event, right_servo_active
                    )
                    print(
                        "[left][rollback] 正面 ICP 重抓未持件；已开爪，"
                        "左臂已回中间观察位；"
                        "重新执行 Buffer Check/ICP 后再决定正面重抓或反抓。",
                        flush=True,
                    )
                    decision, _ = check_reversed_workpiece_on_table(
                        args, stop_event, cycle_dir
                    )
                if stop_event.is_set():
                    return
                if decision == "UNCERTAIN":
                    complete_workpiece_timer(
                        active_workpiece_id,
                        outcome="TABLE_OBJECT_LOST",
                        detail="正面重抓后 Buffer Check 重试耗尽仍未识别到桌面工件",
                    )
                    report_module_status(
                        "module_4_head_judge",
                        ResultStatus.RETRY,
                        "Buffer Check 多次观察仍不确定；视为工件已掉落，"
                        "保持 Buffer 空并从 Module 1 重新抓取",
                        reason_code="BUFFER_OBJECT_LOST",
                    )
                    print(
                        "[left] Buffer Check 重试耗尽仍未看到工件；判定工件已掉落，"
                        "直接开始下一轮 Module 1 抓取。",
                        flush=True,
                    )
                    continue
                if front_recovered:
                    left_observation_ready.clear()
                    publish_buffer_workpiece(runtime_dir, active_workpiece_id, "BRANCH_1_FRONT")
                    set_status(
                        buffer_state,
                        BUFFER_TABLE_READY,
                        "反面落桌后经 Buffer Check 确认已拨正；"
                        "左臂重抓并回原正面放置位",
                    )
                    if bool(getattr(args, "left_only", False)):
                        complete_workpiece_timer(
                            active_workpiece_id,
                            outcome="LEFT_ONLY_BUFFER_READY",
                            detail="反面落桌后已拨正、重抓并放回正面 Buffer 位",
                        )
                        left_done.set()
                        return
                    continue
            # 可靠 BACK 才由左臂在桌上重新反抓。此时不通知右臂，直到左臂持件
            # 已到交接准备状态，才把共享变量设为2。
            print("[left] Buffer Check 确认反面：开始左臂桌面反抓，右臂暂不进入任何分支。", flush=True)
            wait_for_right_servo_idle(right_servo_active, stop_event, "左臂桌面反抓动作")
            # A reverse-grasp NO_GRASP exit occurs after its 3 cm verification
            # lift, not at a camera-safe pose.  Before retrying its first step
            # (head RGB-D capture), withdraw to the same middle observation
            # pose used by the front regrasp recovery.
            reverse_attempt = 0
            handoff_result_file = runtime_dir / "reverse_handoff_left_ready.json"
            while not stop_event.is_set():
                reverse_attempt += 1
                exit_code = run_command(
                    [str(PYTHON), str(LEFT_PICK), "--execute",
                     "--planner-host", args.planner_host,
                     "--arm-host", args.arm_host,
                     "--gripper-host", args.gripper_host],
                    cwd=LEFT_PICK.parent, stop_event=stop_event,
                    arm_host=args.arm_host, label="左臂反抓",
                    recoverable_exit_codes=frozenset({NO_GRASP_EXIT_CODE}),
                )
                if exit_code != 0:
                    print(
                        f"[left][rollback] 反抓第 {reverse_attempt} 次未抓到工件；"
                        "状态回退到 Module 5-1，重新反抓（不重跑前序模块）。",
                        flush=True,
                    )
                    if reverse_attempt >= MAX_REVERSE_TABLE_GRASP_ATTEMPTS:
                        raise RuntimeError(
                            "左臂桌面反抓已连续 5 次未确认夹到工件；"
                            "已停止重试并保持现场。"
                        )
                    move_left_to_middle_observation(
                        args, stop_event, right_servo_active
                    )
                    if stop_event.wait(0.2):
                        return
                    continue

                print("[left] 外部桌面反抓流程成功结束，开始观察位持件复检。", flush=True)
                if handoff_result_file.exists():
                    raise RuntimeError(
                        f"Module 5-2交接握手文件已存在，拒绝复用旧结果："
                        f"{handoff_result_file}"
                    )
                with module_timer("module_5_2_handoff_prepare", workpiece_id=active_workpiece_id):
                    handoff_exit = run_command(
                        [
                            str(PYTHON),
                            str(LEFT_HANDOFF_PREP),
                            "--planner-host",
                            args.planner_host,
                            "--arm-host",
                            args.arm_host,
                            "--gripper-host",
                            args.gripper_host,
                            "--workpiece",
                            os.environ.get("PARM_LOOP_WORKPIECE", "1"),
                            "--handoff-result-file",
                            str(handoff_result_file),
                            *(
                                ["--require-confirmation"]
                                if args.require_confirmation
                                else []
                            ),
                        ],
                        cwd=LOOP_ROOT,
                        stop_event=stop_event,
                        arm_host=args.arm_host,
                        label="Module 5-2左手交接准备",
                        recoverable_exit_codes=frozenset({NO_GRASP_EXIT_CODE}),
                    )
                if handoff_exit == 0:
                    break
                print(
                    f"[left][rollback] 反抓第 {reverse_attempt} 次在观察位复检时"
                    "未确认持件；机械臂已原路返回，重新执行 Module 5-1。",
                    flush=True,
                )
                if reverse_attempt >= MAX_REVERSE_TABLE_GRASP_ATTEMPTS:
                    raise RuntimeError(
                        "左臂桌面反抓已连续 5 次未通过观察位持件复检；"
                        "已停止重试并保持现场。"
                    )
                move_left_to_middle_observation(
                    args, stop_event, right_servo_active
                )
                if stop_event.wait(0.2):
                    return
            if stop_event.is_set():
                return
            print(
                "[left] Module 5-2完成：本轮ICP结果已发布，"
                "左手持件位于交接位。",
                flush=True,
            )
            if bool(getattr(args, "left_only", False)):
                complete_workpiece_timer(
                    active_workpiece_id,
                    outcome="LEFT_ONLY_HANDOFF_READY",
                    detail="反面分支：左臂持件到交接准备位；右臂未启动",
                )
                print(
                    "[left-only] 反面分支已到左手交接准备位；"
                    "左夹爪保持夹持，右臂从未启动，请人工接管。",
                    flush=True,
                )
                left_done.set()
                return
            # 右手将接取并到 Module 9 上方准备位，再由 Module 8-2
            # 让左手回中间位。
            # buffer 保持状态2，直到右手完成孔位放置并确认打开夹爪。
            handoff_left_clear.clear()
            left_observation_ready.clear()
            publish_buffer_workpiece(runtime_dir, active_workpiece_id, branch)
            set_status(
                buffer_state,
                BUFFER_HANDOFF_READY,
                "Module 5-2已完成左手交接准备，等待Module 8-2接取",
            )
    except (InterruptedError, KeyboardInterrupt):
        if active_workpiece_id is not None:
            complete_workpiece_timer(active_workpiece_id, outcome="ABORTED")
        return
    except Exception as exc:
        if active_workpiece_id is not None:
            complete_workpiece_timer(active_workpiece_id, outcome="LEFT_WORKER_FAILURE", detail=repr(exc))
        fault_queue.put(("left", repr(exc)))
        stop_event.set()
        cancel_both_arms(args.arm_host, "左臂流程异常")


def run_right_idle(
    args: argparse.Namespace, stop_event: mp.Event, runtime_dir: Path,
    *, close_first: bool = False,
) -> None:
    report_module_status("module_10_right_idle", ResultStatus.WAIT, "开始右臂放置收尾并回 Idle")
    opened_file = runtime_dir / f"right_gripper_opened_{time.time_ns()}.json"
    command = [
        str(PYTHON), str(RIGHT_FINISH), "--execute",
        "--planner-host", args.planner_host,
        "--gripper-host", args.gripper_host,
        "--arm-host", args.arm_host,
        "--idle-pose", str(args.right_idle_pose),
        "--speed", str(args.right_speed),
        "--gripper-opened-file", str(opened_file),
    ]
    if close_first:
        command.extend([
            "--close-first",
            "--place-ready-joints-deg",
            *[str(value) for value in RIGHT_PLACE_READY_JOINTS_DEG],
        ])
    if args.require_confirmation:
        command.append("--require-confirmation")
    print("\n[run] " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=LOOP_ROOT, start_new_session=True)
    opened_published = False
    try:
        while process.poll() is None:
            if stop_event.wait(0.1):
                cancel_both_arms(args.arm_host, "停止右臂放置收尾")
                terminate_process_group(process, "右臂放置收尾")
                raise InterruptedError("右臂放置收尾被停止请求中断")
            if not opened_published and opened_file.is_file():
                payload = json.loads(opened_file.read_text(encoding="utf-8"))
                if payload.get("event") != "right_gripper_opened_after_place":
                    raise RuntimeError(f"右夹爪打开同步标记无效: {payload}")
                opened_published = True
    except KeyboardInterrupt:
        cancel_both_arms(args.arm_host, "Ctrl+C 停止右臂放置收尾")
        terminate_process_group(process, "右臂放置收尾")
        raise
    except Exception:
        cancel_both_arms(args.arm_host, "右臂放置收尾同步异常")
        terminate_process_group(process, "右臂放置收尾")
        raise

    if not opened_published and opened_file.is_file():
        payload = json.loads(opened_file.read_text(encoding="utf-8"))
        if payload.get("event") != "right_gripper_opened_after_place":
            raise RuntimeError(f"右夹爪打开同步标记无效: {payload}")
        opened_published = True
    if process.returncode != 0:
        report_module_status(
            "module_10_right_idle", ResultStatus.FAULT,
            f"右臂放置收尾失败，exit={process.returncode}",
            reason_code=f"EXIT_{process.returncode}",
        )
        raise RuntimeError(f"右臂放置收尾失败，exit={process.returncode}")
    if not opened_published:
        raise RuntimeError("右臂放置收尾结束但没有确认右夹爪已打开")
    report_module_status("module_10_right_idle", ResultStatus.SUCCESS, "右臂已放置、开爪并回 Idle")


def release_buffer_at_right_place_ready(
    buffer_state: mp.Value, runtime_dir: Path, workpiece_id: str, reason: str,
) -> None:
    """右臂已抓走当前 Buffer 工件并到达 place_ready，释放单容量 Buffer。"""
    current_state = get_status(buffer_state)
    if current_state not in {BUFFER_TABLE_READY, BUFFER_HANDOFF_READY}:
        raise RuntimeError(
            "右臂到 place_ready 时缓冲区状态异常："
            f"{status_label(current_state)}"
        )
    clear_buffer_workpiece(runtime_dir, workpiece_id)
    set_status(buffer_state, BUFFER_EMPTY, reason)


def move_right_to_place_ready_after_pusht(
    args: argparse.Namespace, stop_event: mp.Event,
) -> None:
    """右臂缓冲区抓取完成后，用关节 RRTConnect 到模块9放置准备位。"""
    use_workpiece_2_place_ready = (
        RIGHT_PUSH_TO_Q1 == WORKPIECE_2_RIGHT_PUSH_TO_Q1
        or (
            RIGHT_PUSH_TO_Q1 == MODULAR_RIGHT_BUFFER_PIPELINE
            and os.environ.get("PARM_LOOP_WORKPIECE") == "2"
        )
    )
    if use_workpiece_2_place_ready:
        from modules.workpiece_2_handoff_profile import (
            load_workpiece_2_handoff_profile,
        )

        target_joints_deg = load_workpiece_2_handoff_profile()[
            "right_handoff"
        ]["place_ready_joints_deg"]
        target_label = "工件2 right_place_ready"
    else:
        target_joints_deg = RIGHT_PLACE_READY_JOINTS_DEG
        target_label = "right_place_ready"
    if stop_event.is_set():
        raise InterruptedError("停止请求已到达，跳过右臂去放置准备位")
    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.right_speed),
        max_acceleration=float(args.right_speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "right_hand_transfer_end"})
        print(
            "[right] 缓冲区抓取完成：用关节 RRTConnect(OMPL) 规划到 "
            f"{target_label}，target(deg)={target_joints_deg}",
            flush=True,
        )
        planner.move_right_joints(target_joints_deg, planner="ompl")
        wait_for_joint_target(
            args.arm_host, 2, target_joints_deg, target_label
        )
    finally:
        planner.close()


def run_right_place_wiggle(
    args: argparse.Namespace, stop_event: mp.Event, right_servo_active: mp.Value,
) -> None:
    """两个右臂分支共用的持续 ServoPose 正式插孔/摆动放置模块。"""
    command = [str(PYTHON), str(RIGHT_PLACE_WIGGLE), "--live", "--confirm-live"]
    if os.environ.get("PARM_MODULAR_LOOP_ENTRIES") == "1":
        workpiece = os.environ.get("PARM_LOOP_WORKPIECE")
        if workpiece not in {"1", "2"}:
            raise RuntimeError(f"模块化总控缺少有效工件号: {workpiece!r}")
        command.extend(["--workpiece", workpiece])
    command.extend([
        "--arm-host", args.arm_host, "--reset-right-task-before-servo",
    ])
    try:
        with module_timer("module_9_right_place"):
            run_command(
                command, cwd=LOOP_ROOT, stop_event=stop_event,
                arm_host=args.arm_host, label="右臂孔位摆动放置",
            )
    finally:
        set_right_servo_active(right_servo_active, False)


def wait_for_left_grasp_observation(
    left_observation_ready: mp.Event, stop_event: mp.Event,
    right_servo_active: mp.Value, right_waiting_for_observation: mp.Value,
) -> None:
    """右臂仅在下一轮左臂已到 GraspNet 观察位后启动 ServoPose。"""
    if not left_observation_ready.is_set():
        print(
            "[right] 已完成 ServoPose 前的准备动作；等待左臂到达下一轮 "
            "GraspNet 观察位。",
            flush=True,
        )
    set_right_waiting_for_observation(right_waiting_for_observation, True)
    try:
        while not left_observation_ready.wait(0.2):
            if stop_event.is_set():
                raise InterruptedError("等待左臂 GraspNet 观察位时收到停止请求")
        left_observation_ready.clear()
        # Clear the observation handoff only after the ServoPose guard is active,
        # so the left worker cannot race ahead into its next real motion.
        set_right_servo_active(right_servo_active, True)
        print(
            "[right] 左臂已在 GraspNet 观察位；立即启动右臂 ServoPose 放置。",
            flush=True,
        )
    finally:
        set_right_waiting_for_observation(right_waiting_for_observation, False)


def validate_handoff_right_place_ready(
    marker_file: Path, buffer_state: mp.Value,
) -> None:
    """确认右手到达 right_place_ready；调用方随后释放已被带走的 Buffer 项。"""
    payload = json.loads(marker_file.read_text(encoding="utf-8"))
    if payload.get("event") != "right_place_ready":
        raise RuntimeError(f"交接同步标记内容无效: {payload}")
    expected_workpiece = os.environ.get("PARM_LOOP_WORKPIECE", "1")
    marker_workpiece = str(payload.get("workpiece", "1"))
    if marker_workpiece != expected_workpiece:
        raise RuntimeError(
            "交接同步标记工件类型不匹配："
            f"expected={expected_workpiece}, marker={marker_workpiece}"
        )
    if expected_workpiece == "2":
        from modules.workpiece_2_handoff_profile import (
            load_workpiece_2_handoff_profile,
        )

        profile = load_workpiece_2_handoff_profile()
        expected_joints = np.asarray(
            profile["right_handoff"]["place_ready_joints_deg"], dtype=float
        )
        marker_joints = np.asarray(payload.get("right_joints_deg", []), dtype=float)
        if payload.get("profile_id") != profile["profile_id"]:
            raise RuntimeError(
                "右手到位标记profile不匹配："
                f"expected={profile['profile_id']}, "
                f"marker={payload.get('profile_id')}"
            )
        if marker_joints.shape != (7,) or not np.allclose(
            marker_joints, expected_joints, atol=1e-9
        ):
            raise RuntimeError(
                "右手到位标记关节目标与工件2profile不匹配"
            )
    if get_status(buffer_state) != BUFFER_HANDOFF_READY:
        raise RuntimeError(
            "右臂到 right_place_ready 时缓冲区状态异常："
            f"{status_label(get_status(buffer_state))}"
        )
    print(
        "[buffer] 右臂已到 right_place_ready 并接走工件；"
        "状态保持 2=HANDOFF_READY，直到孔位放置后夹爪打开。",
        flush=True,
    )


def validate_reverse_handoff_contract(
    handoff_result_file: Path,
) -> tuple[Path, str]:
    """Validate Module 5-2's typed profile contract before any right motion."""
    if not handoff_result_file.is_file():
        raise FileNotFoundError(
            f"Module 5-2交接握手不存在：{handoff_result_file}"
        )
    handoff_payload = json.loads(
        handoff_result_file.read_text(encoding="utf-8")
    )
    expected_workpiece = os.environ.get("PARM_LOOP_WORKPIECE", "1")
    if (
        handoff_payload.get("event")
        != "module_5_2_left_handoff_ready"
        or handoff_payload.get("left_gripper_holds_object") is not True
    ):
        raise RuntimeError(
            f"Module 5-2交接握手无效：{handoff_payload}"
        )
    payload_workpiece = str(handoff_payload.get("workpiece", "1"))
    if payload_workpiece != expected_workpiece:
        raise RuntimeError(
            "Module 5-2交接握手工件类型不匹配："
            f"expected={expected_workpiece}, payload={payload_workpiece}"
        )
    if expected_workpiece in {"1", "2"}:
        from modules.workpiece_2_handoff_profile import (
            load_workpiece_2_handoff_profile,
        )

        profile = load_workpiece_2_handoff_profile()
        if (
            expected_workpiece == "2"
            and handoff_payload.get("profile_id") != profile["profile_id"]
        ):
            raise RuntimeError(
                "Module 5-2交接握手profile不匹配："
                f"expected={profile['profile_id']}, "
                f"payload={handoff_payload.get('profile_id')}"
            )
        actual_center = np.asarray(
            handoff_payload.get("handoff_center_body_m", []), dtype=float
        )
        expected_center = np.asarray(
            profile["handoff"]["center_body_m"], dtype=float
        )
        if actual_center.shape != (3,) or not np.allclose(
            actual_center, expected_center, atol=1e-9
        ):
            raise RuntimeError(
                "Module 5-2交接握手中心与统一高位profile不匹配："
                f"payload={actual_center.tolist()}, "
                f"expected={expected_center.tolist()}"
            )
    result_value = handoff_payload.get("result")
    if not isinstance(result_value, str) or not result_value:
        raise RuntimeError(
            f"Module 5-2交接握手缺少result路径：{handoff_payload}"
        )
    result = Path(result_value).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Module 5-2 ICP结果不存在：{result}")
    return result, expected_workpiece


def run_reverse_handoff_until_right_place_ready(
    args: argparse.Namespace, buffer_state: mp.Value, stop_event: mp.Event,
    handoff_result_file: Path, marker_file: Path,
) -> None:
    """运行Module 8-2，确认右臂实际到达对应工件的 place_ready。"""
    report_module_status("module_8_2_left_handoff", ResultStatus.WAIT, "开始右手接取反抓工件")
    result, expected_workpiece = validate_reverse_handoff_contract(
        handoff_result_file
    )

    marker_file.parent.mkdir(parents=True, exist_ok=True)
    if marker_file.exists():
        raise RuntimeError(f"反向交接标记文件已存在，拒绝复用旧结果: {marker_file}")

    command = [
        str(PYTHON), str(RIGHT_LEFT_HANDOFF),
        "--result", str(result),
        "--planner-host", args.planner_host,
        "--gripper-host", args.gripper_host,
        "--arm-host", args.arm_host,
        "--workpiece", expected_workpiece,
        "--right-place-ready-file", str(marker_file),
    ]
    if args.require_confirmation:
        command.append("--require-confirmation")
    print("\n[run] " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=LOOP_ROOT, start_new_session=True)
    marker_published = False
    try:
        while process.poll() is None:
            if stop_event.wait(0.2):
                cancel_both_arms(args.arm_host, "停止右臂反抓交接")
                terminate_process_group(process, "右臂反抓交接")
                raise InterruptedError("右臂反抓交接被停止请求中断")
            if not marker_published and marker_file.is_file():
                validate_handoff_right_place_ready(marker_file, buffer_state)
                marker_published = True
    except KeyboardInterrupt:
        cancel_both_arms(args.arm_host, "Ctrl+C 停止右臂反抓交接")
        terminate_process_group(process, "右臂反抓交接")
        raise

    # 进程刚好在一次轮询间隔内退出时，仍须读取已经落盘的 right_place_ready 标记。
    if not marker_published and marker_file.is_file():
        validate_handoff_right_place_ready(marker_file, buffer_state)
        marker_published = True
    if process.returncode != 0:
        report_module_status(
            "module_8_2_left_handoff", ResultStatus.FAULT,
            f"右臂反抓交接失败，exit={process.returncode}",
            reason_code=f"EXIT_{process.returncode}",
        )
        raise RuntimeError(f"右臂反抓交接失败，exit={process.returncode}")
    if not marker_published:
        raise RuntimeError("交接脚本结束但没有确认右臂实际到达 right_place_ready")
    handoff_result_file.unlink()
    report_module_status(
        "module_8_2_left_handoff", ResultStatus.SUCCESS,
        "右手已接取工件并到达 right_place_ready",
    )


def right_worker(
    args: argparse.Namespace, buffer_state: mp.Value, stop_event: mp.Event,
    fault_queue: mp.Queue, runtime_dir: Path,
    handoff_left_clear: mp.Event, left_needs_graspup: mp.Event,
    right_idle_event: mp.Event, left_observation_ready: mp.Event,
    right_servo_active: mp.Value, right_waiting_for_observation: mp.Value,
) -> None:
    active_workpiece_id: str | None = None
    try:
        while not stop_event.wait(0.2):
            state = get_status(buffer_state)
            if state == BUFFER_EMPTY:
                continue
            if state == BUFFER_TABLE_READY:
                active_workpiece_id = current_buffer_workpiece(runtime_dir)
                os.environ[WORKPIECE_ENV] = active_workpiece_id
                right_idle_event.clear()
                print(
                    "\n[right] 主分支1：启动右臂观察/摆正/抓取；完成后关节 RRTConnect(OMPL) "
                    "到 right_place_ready，随后立即执行 ServoPose 孔位摆动放置；"
                    "到 place_ready 后释放缓冲区，随后继续放置并回 Idle。",
                    flush=True,
                )
                environment = os.environ.copy()
                environment["PARM_ARM_HOST"] = args.arm_host
                environment["PARM_PLANNER_HOST"] = args.planner_host
                environment["PARM_GRIPPER_HOST"] = args.gripper_host
                if RIGHT_PUSH_TO_Q1 == MODULAR_RIGHT_BUFFER_PIPELINE:
                    # ``active_workpiece_id`` 是本轮的日志/计时实例 ID，不能传给
                    # 仅接受工件类型 1 或 2 的右臂模块。
                    pipeline_workpiece = os.environ.get("PARM_LOOP_WORKPIECE")
                    if pipeline_workpiece not in {"1", "2"}:
                        raise RuntimeError(
                            "右臂 buffer 抓取缺少有效工件类型 "
                            "PARM_LOOP_WORKPIECE（必须为 1 或 2）"
                        )
                    right_push_command = [
                        str(PYTHON), str(RIGHT_PUSH_TO_Q1),
                        "--workpiece", pipeline_workpiece,
                        "--joint-speed", str(args.right_speed),
                        "--execute",
                    ]
                elif RIGHT_PUSH_TO_Q1 == WORKPIECE_2_RIGHT_PUSH_TO_Q1:
                    right_push_command = [
                        str(PYTHON), str(RIGHT_PUSH_TO_Q1),
                        "--arm-id", "2", "--execute",
                    ]
                else:
                    right_push_command = [
                        str(PYTHON), str(RIGHT_PUSH_TO_Q1), "--execute",
                    ]
                run_command(
                    right_push_command,
                    cwd=RIGHT_PUSH_TO_Q1.parent, stop_event=stop_event,
                    arm_host=args.arm_host, label="右臂观察/摆正/抓取", env=environment,
                )
                move_right_to_place_ready_after_pusht(args, stop_event)
                release_buffer_at_right_place_ready(
                    buffer_state,
                    runtime_dir,
                    active_workpiece_id,
                    "右臂已抓走缓冲区工件并到达 place_ready",
                )
                run_right_place_wiggle(args, stop_event, right_servo_active)
                with module_timer("module_10_right_idle"):
                    run_right_idle(args, stop_event, runtime_dir)
                complete_workpiece_timer(active_workpiece_id, outcome="SUCCESS")
                active_workpiece_id = None
                right_idle_event.set()
                print("[right] 主分支1正式放置完成并回 Idle。", flush=True)
                continue
            if state == BUFFER_HANDOFF_READY:
                active_workpiece_id = current_buffer_workpiece(runtime_dir)
                os.environ[WORKPIECE_ENV] = active_workpiece_id
                right_idle_event.clear()
                handoff_result_file = (
                    runtime_dir / "reverse_handoff_left_ready.json"
                )
                # The typed Module 5-2 contract must be valid before the right
                # arm leaves Idle for the handoff region.
                validate_reverse_handoff_contract(handoff_result_file)
                move_right_to_reverse_handoff_prep(args, stop_event)
                marker_file = runtime_dir / (
                    "reverse_handoff_right_place_ready_"
                    f"{time.time_ns()}.json"
                )
                with module_timer("module_8_2_left_handoff"):
                    run_reverse_handoff_until_right_place_ready(
                        args,
                        buffer_state,
                        stop_event,
                        handoff_result_file,
                        marker_file,
                    )
                # 右臂已实际到达此工件对应的 place_ready 并抓走工件；此后右臂
                # 的放置路径不再占用缓冲区，允许左臂向单容量缓冲区放入下一件。
                release_buffer_at_right_place_ready(
                    buffer_state,
                    runtime_dir,
                    active_workpiece_id,
                    "右臂已完成交接并到达 place_ready",
                )
                left_needs_graspup.set()
                handoff_left_clear.set()
                print(
                    "[right] 左臂已回安全回程位，已放行其回 graspup 进入下一轮；"
                    "右臂继续独立执行放置。",
                    flush=True,
                )
                run_right_place_wiggle(args, stop_event, right_servo_active)
                with module_timer("module_10_right_idle"):
                    run_right_idle(args, stop_event, runtime_dir)
                complete_workpiece_timer(active_workpiece_id, outcome="SUCCESS")
                active_workpiece_id = None
                right_idle_event.set()
                print("[right] 主分支2正式放置完成并回 Idle。", flush=True)
                continue
            raise RuntimeError(f"右臂收到未知缓冲区状态: {state}")
    except (InterruptedError, KeyboardInterrupt):
        if active_workpiece_id is not None:
            complete_workpiece_timer(active_workpiece_id, outcome="ABORTED")
        return
    except EOFError:
        fault_queue.put(("right", "终端输入流关闭"))
        stop_event.set()
        cancel_both_arms(args.arm_host, "右臂终端输入中断")
    except Exception as exc:
        if active_workpiece_id is not None:
            complete_workpiece_timer(active_workpiece_id, outcome="RIGHT_WORKER_FAILURE", detail=repr(exc))
        fault_queue.put(("right", repr(exc)))
        stop_event.set()
        cancel_both_arms(args.arm_host, "右臂流程异常")


def unexpected_worker_exit(
    left: mp.Process, right: mp.Process, left_done: mp.Event,
) -> str | None:
    """Return a fault message when a worker disappears without reporting it.

    A bounded left worker may exit normally after publishing ``left_done``.
    The right worker has no independent normal-exit path: the parent stops it
    only after the final buffer item is consumed and the right arm is idle.
    """
    if not left.is_alive():
        exit_code = left.exitcode
        if exit_code not in (None, 0):
            return f"left worker 非正常退出且未上报故障: exitcode={exit_code}"
        if not left_done.is_set():
            return "left worker 提前退出且未上报故障: exitcode=0, left_done=False"
    if not right.is_alive():
        return (
            "right worker 提前退出且未上报故障: "
            f"exitcode={right.exitcode}"
        )
    return None


def main() -> None:
    args = parse_args()
    # PlannerRuntime executes planned joint trajectories through the direct Arm SDK.
    os.environ["PARM_ARM_HOST"] = args.arm_host
    if args.left_only:
        os.environ["PARM_LOOP_LEFT_ONLY"] = "1"
        # ``left_cycle_limit`` counts attempts, including NO_GRASP and other
        # recoverable perception/placement outcomes.  Left-only completion is
        # instead bounded by the explicit successful BUFFER/HANDOFF returns in
        # left_worker, so failed attempts must remain retryable.
        args.left_cycle_limit = 0
    else:
        os.environ.pop("PARM_LOOP_LEFT_ONLY", None)
    selected_left_flow = LEFT_FLOW_SCRIPTS[args.left_flow]
    os.environ["PARM_LOOP_ROOT"] = str(LOOP_ROOT)
    for path in (
                 PYTHON, GRASPNET_FLOW, selected_left_flow, LEFT_PICK,
                 FRONT_BUFFER_REGRASP, LEFT_PLACE, HEAD_MOTION,
                 HEAD_CALIBRATION, BUFFER_FACE_CHECK,
                 LEFT_HANDOFF_PREP, RIGHT_LEFT_HANDOFF,
                 RIGHT_FINISH, RIGHT_PUSH_TO_Q1, RIGHT_PLACE_WIGGLE,
                 args.right_idle_pose):
        if not path.is_file():
            raise FileNotFoundError(f"串联所需文件不存在: {path}")
    if (
        args.left_cycle_limit < 0
        or not (0.0 < args.right_speed <= 1.0)
        or not (0.0 < args.prepare_speed <= 1.0)
    ):
        raise ValueError("--left-cycle-limit 必须非负，所有速度必须为正")
    print(
        f"[prepare] 左臂后处理模式={args.left_flow}: {selected_left_flow}",
        flush=True,
    )

    runtime_dir = args.runtime_dir or (
        LOOP_ROOT / "output/simple_dual_arm_loop" /
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    runtime_dir.mkdir(parents=True, exist_ok=False)
    try:
        prepare_both_arms(args)
    except KeyboardInterrupt:
        cancel_both_arms(args.arm_host, "准备阶段 Ctrl+C")
        raise
    except Exception:
        # Planner execution may fail after the server has accepted a task.
        # Stop both arms before surfacing the preparation failure.
        cancel_both_arms(args.arm_host, "准备阶段异常")
        raise

    context = mp.get_context("spawn")
    buffer_state = context.Value("i", BUFFER_EMPTY)
    stop_event = context.Event()
    fault_queue = context.Queue()
    left_done = context.Event()
    handoff_left_clear = context.Event()
    handoff_left_clear.set()
    left_needs_graspup = context.Event()
    left_observation_ready = context.Event()
    # 右臂 worker 是唯一写入方；左臂只在发起下一次动作前读取/等待。
    right_servo_active = context.Value("b", False)
    right_waiting_for_observation = context.Value("b", False)
    # 准备阶段已经验证右臂位于 Idle；仅在 --left-cycle-limit 收尾时使用，
    # 防止 buffer 已提前释放但右臂仍在放置时主进程提前退出。
    right_idle_event = context.Event()
    right_idle_event.set()
    left = context.Process(
        target=left_worker, name="left-loop",
        args=(
            args, buffer_state, stop_event, fault_queue, runtime_dir, left_done,
            handoff_left_clear, left_needs_graspup, left_observation_ready,
            right_servo_active, right_waiting_for_observation,
        ),
    )
    right = None
    if not args.left_only:
        right = context.Process(
            target=right_worker, name="right-loop",
            args=(
                args, buffer_state, stop_event, fault_queue, runtime_dir,
                handoff_left_clear, left_needs_graspup, right_idle_event,
                left_observation_ready, right_servo_active,
                right_waiting_for_observation,
            ),
        )
    print(f"[loop] runtime={runtime_dir}")
    if args.left_only:
        print(
            "[loop] LEFT-ONLY：右 worker 未创建；右臂/右夹爪不会收到任何命令；"
            "Ctrl+C 仅发送左臂 cancel_task(1)。"
        )
    else:
        print("[loop] 缓冲区初始状态=0；Ctrl+C 会向两臂直接发送 SDK cancel_task()。")
    left.start()
    if right is not None:
        right.start()
    try:
        while left.is_alive() or (right is not None and right.is_alive()):
            if args.left_only and left_done.is_set():
                print("[loop] 单左臂成功周期完成；右臂未启动。", flush=True)
                break
            if (
                left_done.is_set()
                and get_status(buffer_state) == BUFFER_EMPTY
                and right_idle_event.is_set()
            ):
                print("[loop] 限定左臂循环已全部完成，右臂已放置并回 Idle。", flush=True)
                break
            try:
                worker, error = fault_queue.get(timeout=0.2)
            except queue.Empty:
                if args.left_only:
                    worker_exit = None
                    if not left.is_alive():
                        if left.exitcode not in (None, 0):
                            worker_exit = (
                                "left worker 非正常退出且未上报故障: "
                                f"exitcode={left.exitcode}"
                            )
                        elif not left_done.is_set():
                            worker_exit = (
                                "left worker 提前退出且未上报故障: "
                                "exitcode=0, left_done=False"
                            )
                else:
                    assert right is not None
                    worker_exit = unexpected_worker_exit(left, right, left_done)
                if worker_exit is not None:
                    raise RuntimeError(worker_exit)
                continue
            raise RuntimeError(f"{worker} worker 异常: {error}")
    except KeyboardInterrupt:
        print("\n[loop] 收到 Ctrl+C，停止两个 worker。", flush=True)
        stop_event.set()
        cancel_both_arms(args.arm_host, "总控 Ctrl+C")
    finally:
        stop_event.set()
        processes = (left,) if right is None else (left, right)
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        cancel_both_arms(args.arm_host, "总控退出")
    if not fault_queue.empty():
        worker, error = fault_queue.get_nowait()
        raise RuntimeError(f"{worker} worker 异常: {error}")


if __name__ == "__main__":
    main()
