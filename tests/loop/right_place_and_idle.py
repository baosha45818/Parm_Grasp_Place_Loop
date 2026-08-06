#!/usr/bin/env python3
"""右臂到放置初始位后松爪、沿 body_base +Z 抬升并关节规划回 Idle。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_config import LoopConfig
from loop_types import ObjectPose
from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from tool.planner_runtime import PlannerRuntime


DEFAULT_IDLE_POSE = Path("/home/dainanjun/zhuyiming/push_T/saved_poses/3.json")
DEFAULT_PLACE_READY_JOINTS_DEG = [
    -5.6937, 43.1502, -112.7074, -53.3867, 77.7186, -39.9765, 28.4257,
]
IDLE_FALLBACK_JOINTS_DEG = [
    161.36, 19.07, -107.979, 71.451, -55.833, -14.896, 88.2,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--idle-pose", type=Path, default=DEFAULT_IDLE_POSE)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--lift-m", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=100.0)
    parser.add_argument("--gripper-settle-s", type=float, default=1.0)
    parser.add_argument("--joint-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--joint-timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--gripper-opened-file",
        type=Path,
        help="右夹爪确认打开后原子写入的总控同步标记；单独运行时可省略",
    )
    parser.add_argument(
        "--close-first", action="store_true",
        help="先同步闭合右夹爪；用于缓冲区主分支1的人工推正/就位后",
    )
    parser.add_argument(
        "--place-ready-joints-deg", type=float, nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="闭爪后先用规划器 RRTStar 到右臂放置初始位；仅与 --close-first 一起使用",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--require-confirmation", action="store_true")
    return parser.parse_args()


def transform_from_planner(response: dict) -> np.ndarray:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [translation.get(axis) for axis in "xyz"]
    xyzw = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in xyz + xyzw):
        raise RuntimeError(f"规划器右臂末端 TF 格式无效: {response}")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    result[:3, 3] = xyz
    return result


def load_idle_joints(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    joints = (payload.get("arms") or {}).get("right", {}).get("joint_positions_deg")
    if not isinstance(joints, list) or len(joints) != 7:
        raise RuntimeError(f"右臂 Idle 文件没有 seven joint_positions_deg: {path}")
    result = [float(value) for value in joints]
    if not all(np.isfinite(result)):
        raise RuntimeError(f"右臂 Idle 关节角存在非有限数: {path}")
    return result


def confirm(args: argparse.Namespace, message: str) -> None:
    if not args.require_confirmation:
        print(f"{message} [自动确认]")
        return
    if input(f"{message} 输入 y 并回车继续: ").strip().lower() != "y":
        raise RuntimeError("操作者取消右臂放置收尾")


def sync_gripper_arrived(response: object, hand: int) -> bool:
    """Read the new driver's synchronous completion result for one gripper."""
    if not isinstance(response, dict):
        return False
    payload = response.get("data") or response
    devices = payload.get("devices") if isinstance(payload, dict) else None
    device_id = "left_gripper" if hand == 1 else "right_gripper"
    device = devices.get(device_id) if isinstance(devices, dict) else None
    return isinstance(device, dict) and device.get("device_state") == "arrived"


def wait_for_gripper_terminal_state(
    gripper: object,
    hand: int,
    response: object,
    *,
    accepted_states: set[str],
    timeout_s: float = 3.0,
) -> tuple[dict | None, bool]:
    """Handle the lag between a synchronous command reply and the PUB state."""
    response_arrived = sync_gripper_arrived(response, hand)
    deadline = time.monotonic() + timeout_s
    latest = gripper.get_state().get(hand)
    while time.monotonic() < deadline:
        motion_state = str((latest or {}).get("motion_state", "")).lower()
        if motion_state in accepted_states:
            return latest, True
        if motion_state in {"error", "dropped"}:
            raise RuntimeError(f"右夹爪动作异常: {latest}")
        # The command reply is authoritative for a sync operation.  The PUB
        # state can legitimately lag by one frame on the new gripper driver.
        if response_arrived:
            return latest, True
        time.sleep(0.05)
        latest = gripper.get_state().get(hand)
    return latest, response_arrived


def wait_for_right_joint_target(
    arm_host: str,
    target_deg: list[float],
    tolerance_deg: float,
    timeout_s: float,
    label: str,
) -> None:
    """The planner execute endpoint only accepts a task; poll robot feedback to gate the next action."""
    from parm_sdk import Arm

    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            right = (state.get("state") or {}).get("right") or {}
            actual = np.asarray(right.get("position") or [], dtype=float)
            if actual.shape == (7,):
                # Robot status can retain equivalent multi-turn joint values.
                delta = (actual - np.asarray(target_deg, dtype=float) + 180.0) % 360.0 - 180.0
                last_error_deg = float(np.max(np.abs(delta)))
                if last_error_deg <= tolerance_deg:
                    print(f"[右臂到位] {label}: 最大关节误差 {last_error_deg:.3f}deg")
                    return
            time.sleep(0.05)
    raise RuntimeError(
        f"右臂未在 {timeout_s:.1f}s 内到达 {label}；"
        f"最后最大关节误差={last_error_deg:.3f}deg"
    )


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("右臂放置收尾会实际开夹爪和运动，必须传 --execute")
    if not (0.0 < args.speed <= 1.0 and args.lift_m > 0.0):
        raise ValueError("--speed 必须在 (0, 1]，--lift-m 必须大于0")
    if args.joint_tolerance_deg <= 0.0 or args.joint_timeout_s <= 0.0:
        raise ValueError("关节到位容差和等待时间必须大于0")

    idle_joints = load_idle_joints(args.idle_pose)
    if args.place_ready_joints_deg is not None and not args.close_first:
        raise ValueError("--place-ready-joints-deg 只能与 --close-first 一起使用")
    if args.gripper_opened_file is not None and args.gripper_opened_file.exists():
        raise RuntimeError(f"夹爪打开标记已存在，拒绝复用旧结果: {args.gripper_opened_file}")
    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.speed),
        max_acceleration=float(args.speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "right_hand_transfer_end"})
        from parm_sdk import Gripper

        if args.close_first:
            confirm(args, "确认右臂已完成推正并位于缓冲区抓取位。")
            with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                state = gripper.get_state().get(2)
                if not is_gripper_usable(state):
                    raise RuntimeError("右夹爪 hand=2 不可用或尚未初始化")
                response = gripper.move(
                    2, 0.0, force=float(args.gripper_force), mode="sync",
                    preempt=False, timeout=60.0,
                )
                final_state, closed = wait_for_gripper_terminal_state(
                    gripper, 2, response, accepted_states={"arrived", "gripping"}
                )
            if not closed:
                raise RuntimeError(f"右夹爪闭合后状态异常: {final_state}")
            print(f"[右臂抓取] 右夹爪已闭合: {(response.get('data') or {}).get('position')}%")
            if args.place_ready_joints_deg is not None:
                print("[右臂抓取] 闭爪后规划 RRTStar 到右臂放置初始位。")
                place_ready = [float(value) for value in args.place_ready_joints_deg]
                planner.move_right_joints(place_ready)
                wait_for_right_joint_target(
                    args.arm_host,
                    place_ready,
                    args.joint_tolerance_deg,
                    args.joint_timeout_s,
                    "右臂放置初始位",
                )

        response = planner.lookup_transform("body_base", "right_hand_transfer_end")
        if response is None:
            raise RuntimeError("规划器没有返回右臂当前末端位姿")
        current = transform_from_planner(response)
        lift = current.copy()
        lift[2, 3] += float(args.lift_m)
        print(
            "[右臂放置] 当前放置位 body_base xyz="
            f"{np.round(current[:3, 3], 6).tolist()}；"
            f"松爪后目标上升 {args.lift_m:.3f}m"
        )
        confirm(args, "确认右夹爪下方已完成放置且上方无障碍。")

        with Gripper(host=args.gripper_host, status_port=7702) as gripper:
            state = gripper.get_state().get(2)
            if not is_gripper_usable(state):
                raise RuntimeError("右夹爪 hand=2 不可用或尚未初始化")
            response = gripper.move(
                2, 100.0, force=float(args.gripper_force), mode="sync",
                preempt=False, timeout=60.0,
            )
            final_state, opened = wait_for_gripper_terminal_state(
                gripper, 2, response, accepted_states={"arrived"}
            )
        if not opened or float((final_state or {}).get("position", 100.0)) < 95.0:
            raise RuntimeError(f"右夹爪打开后未到 arrived 状态: {final_state}")
        print(f"[右臂放置] 右夹爪已打开: {(response.get('data') or {}).get('position')}%")
        if args.gripper_opened_file is not None:
            opened_file = args.gripper_opened_file.expanduser().resolve()
            opened_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = opened_file.with_name(opened_file.name + ".tmp")
            temporary_file.write_text(
                json.dumps({
                    "event": "right_gripper_opened_after_place",
                    "timestamp": time.time(),
                    "motion_state": "arrived",
                    "position": (response.get("data") or {}).get("position"),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_file.replace(opened_file)
            print(f"[右臂放置] 已写入夹爪打开同步标记: {opened_file}")
        time.sleep(float(args.gripper_settle_s))

        lift_pose = ObjectPose(
            frame_id="body_base",
            position=lift[:3, 3].tolist(),
            orientation=Rotation.from_matrix(lift[:3, :3]).as_quat().tolist(),
        )
        lift_plan = planner.plan_cartesian_pose("right", lift_pose, planner="lin")
        if lift_plan is None:
            raise RuntimeError("右臂放置后抬升规划没有返回轨迹")
        planner.execute_planned_result(lift_plan)
        lift_target = ((lift_plan.get("data") or {}).get("trajectory") or [])[-1]
        wait_for_right_joint_target(
            args.arm_host,
            [float(value) for value in lift_target],
            args.joint_tolerance_deg,
            args.joint_timeout_s,
            "右臂放置后抬升位",
        )
        print("[右臂放置] 已沿 body_base +Z 抬升，开始 RRTStar 回右臂 Idle。")
        try:
            planner.move_right_joints(idle_joints)
        except Exception as direct_idle_error:
            print(
                "[右臂放置] 直接规划回 Idle 失败，改经固定过渡关节位回退："
                f"{direct_idle_error}"
            )
            planner.move_right_joints(IDLE_FALLBACK_JOINTS_DEG)
            wait_for_right_joint_target(
                args.arm_host,
                IDLE_FALLBACK_JOINTS_DEG,
                args.joint_tolerance_deg,
                args.joint_timeout_s,
                "右臂 Idle 回退过渡位",
            )
            print("[右臂放置] 回退过渡位已到达，重新规划 RRTStar 回右臂 Idle。")
            planner.move_right_joints(idle_joints)
        wait_for_right_joint_target(
            args.arm_host,
            idle_joints,
            args.joint_tolerance_deg,
            args.joint_timeout_s,
            "右臂 Idle",
        )
        print("[右臂放置] 已到 Idle。")
    finally:
        planner.close()


if __name__ == "__main__":
    main()
