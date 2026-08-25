#!/usr/bin/env python3
"""Module 0：预检共享服务，使双臂与夹爪进入安全初始状态。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_0_prepare.ensure_sam2_service import ensure_sam2_service
from modules.module_0_prepare.preflight_robot_infrastructure import preflight_robot_infrastructure


LEFT_GRASPUP_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]


def _joint_error(arm_host: str, arm_id: int, target: list[float]) -> float:
    import numpy as np
    from parm_sdk import Arm

    key = "left" if arm_id == 1 else "right"
    with Arm(host=arm_host) as arm:
        position = ((((arm.get_state() or {}).get("state") or {}).get(key) or {}).get("position") or [])
        actual = np.asarray(position, dtype=float)
    if actual.shape != (7,):
        raise RuntimeError(f"无法读取 {key} 臂 7 维关节状态")
    delta = (actual - np.asarray(target, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(delta)))


def _wait_joint(arm_host: str, arm_id: int, target: list[float], label: str) -> None:
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        error = _joint_error(arm_host, arm_id, target)
        if error <= 0.5:
            print(f"[module-0] {label} 到位，最大关节误差 {error:.3f}deg", flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError(f"{label} 90 秒内未到位")


def _move_joint(*, arm_host: str, planner_host: str, side: str, target: list[float], speed: float, label: str) -> None:
    from loop_config import LoopConfig
    from tools.planner_runtime import PlannerRuntime

    config = LoopConfig(
        planner_host=planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=speed,
        max_acceleration=speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    arm_id = 1 if side == "left" else 2
    frame = "left_hand_transfer_end" if side == "left" else "right_hand_transfer_end"
    try:
        planner.open(required_frames={"body_base", frame})
        if _joint_error(arm_host, arm_id, target) <= 0.5:
            print(f"[module-0] {label} 已到位，跳过零长度规划", flush=True)
            return
        if side == "left":
            planner.move_left_joints(target, planner="rrtstar")
        elif side == "right":
            planner.move_right_joints(target, planner="rrtstar")
        else:
            raise ValueError(f"未知机械臂侧别: {side}")
        _wait_joint(arm_host, arm_id, target, label)
    finally:
        planner.close()


def _right_idle() -> list[float]:
    path = ROOT / "modules/module_10_right_idle/right_idle_pose.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = ((payload.get("arms") or {}).get("right") or {}).get("joint_positions_deg")
    if not isinstance(values, list) or len(values) != 7:
        raise RuntimeError(f"右臂 Idle 配置无效: {path}")
    return [float(value) for value in values]


def _open_grippers(host: str) -> None:
    from parm_sdk import Gripper

    with Gripper(host=host, status_port=7702) as gripper:
        for hand, label in ((1, "左"), (2, "右")):
            response = gripper.move(
                hand, 100.0, force=100.0,
                speed=80.0 if hand == 1 else None,
                mode="sync", preempt=False, timeout=60.0,
            )
            state = gripper.get_state().get(hand) or {}
            devices = ((response.get("data") or response).get("devices") or {}) if isinstance(response, dict) else {}
            device = devices.get("left_gripper" if hand == 1 else "right_gripper") or {}
            response_position = float((device.get("state_interfaces") or {}).get("gripper/position", 0.0))
            arrived = state.get("motion_state") == "arrived" or device.get("device_state") == "arrived"
            position = float(state.get("position", response_position))
            if not arrived or position < 95.0:
                raise RuntimeError(f"{label}夹爪打开未确认到位: {state}")
            print(f"[module-0] {label}夹爪已打开：position={position:.1f}%", flush=True)


def _write(path: Path, outcome: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"outcome": outcome}, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_robot(
    *,
    planner_host: str,
    arm_host: str,
    gripper_host: str,
    force_host: str,
    speed: float = 0.3,
    open_grippers_before_motion: bool = True,
) -> None:
    """完成 Module 0 准备；可按调用场景选择夹爪打开时机。"""
    if not 0.0 < speed <= 1.0:
        raise ValueError("speed 必须在 (0, 1] 内")

    ensure_sam2_service()
    preflight_robot_infrastructure(
        planner_host=planner_host,
        arm_host=arm_host,
        gripper_host=gripper_host,
        force_sensor_host=force_host,
    )
    if open_grippers_before_motion:
        _open_grippers(gripper_host)

    errors: list[BaseException] = []

    def move(side: str, target: list[float], label: str) -> None:
        try:
            _move_joint(
                arm_host=arm_host, planner_host=planner_host,
                side=side, target=target, speed=speed, label=label,
            )
        except BaseException as exc:
            errors.append(exc)

    left = threading.Thread(target=move, args=("left", LEFT_GRASPUP_DEG, "左臂 graspup"))
    right = threading.Thread(target=move, args=("right", _right_idle(), "右臂 Idle"))
    left.start(); right.start(); left.join(); right.join()
    if errors:
        raise RuntimeError(f"启动归位失败: {errors[0]}") from errors[0]

    if not open_grippers_before_motion:
        _open_grippers(gripper_host)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", required=True)
    parser.add_argument("--gripper-host", required=True)
    parser.add_argument("--force-host", required=True)
    parser.add_argument("--speed", type=float, default=0.3)
    args = parser.parse_args()
    prepare_robot(
        planner_host=args.planner_host,
        arm_host=args.arm_host,
        gripper_host=args.gripper_host,
        force_host=args.force_host,
        speed=args.speed,
    )
    _write(Path(args.result_file), "success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
