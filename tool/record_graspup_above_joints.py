#!/usr/bin/env python3
"""把左臂移动到 graspup 上方，并输出该位置的实际七关节角。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_config import LoopConfig
from loop_types import ObjectPose
from tool.planner_runtime import PlannerRuntime


GRASPUP_JOINTS_DEG = [
    -4.081,
    -52.722,
    3.637,
    100.478,
    -10.051,
    53.969,
    8.843,
]


def read_left_joints_deg(arm: object, joint_unit: str) -> np.ndarray:
    state = arm.get_state() or {}
    left = ((state.get("state") or {}).get("left") or {})
    joints = np.asarray(left.get("position") or [], dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"无法读取有效的左臂七关节状态: {left}")
    return np.rad2deg(joints) if joint_unit == "rad" else joints


def wait_for_joint_target(
    arm: object,
    target_deg: np.ndarray,
    *,
    joint_unit: str,
    tolerance_deg: float,
    timeout_s: float,
) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    last_actual = None
    last_error = float("inf")
    while time.monotonic() < deadline:
        actual = read_left_joints_deg(arm, joint_unit)
        delta = (actual - target_deg + 180.0) % 360.0 - 180.0
        last_actual = actual
        last_error = float(np.max(np.abs(delta)))
        if last_error <= tolerance_deg:
            print(f"[到位] graspup最大关节误差={last_error:.4f}deg")
            return actual
        time.sleep(0.05)
    raise RuntimeError(
        f"左臂未在{timeout_s:.1f}s内到达graspup；"
        f"最后误差={last_error:.4f}deg，actual={last_actual}"
    )


def wait_for_stable_joints(
    arm: object,
    joint_unit: str,
    *,
    stable_delta_deg: float,
    timeout_s: float,
) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    previous = None
    stable_samples = 0
    while time.monotonic() < deadline:
        actual = read_left_joints_deg(arm, joint_unit)
        if previous is not None:
            delta = (actual - previous + 180.0) % 360.0 - 180.0
            if float(np.max(np.abs(delta))) <= stable_delta_deg:
                stable_samples += 1
                if stable_samples >= 5:
                    return actual
            else:
                stable_samples = 0
        previous = actual
        time.sleep(0.1)
    raise RuntimeError(f"上移后关节状态在{timeout_s:.1f}s内没有稳定")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="左臂运动到graspup上方并打印实际关节角（deg）"
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument(
        "--execute-url",
        default="http://127.0.0.1:5667/middle/waypoint/execute",
    )
    parser.add_argument("--graspup-speed", type=float, default=0.2)
    parser.add_argument("--lift-speed", type=float, default=0.1)
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument("--joint-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--graspup-tolerance-deg", type=float, default=0.5)
    args = parser.parse_args()

    for name, value in (
        ("--graspup-speed", args.graspup_speed),
        ("--lift-speed", args.lift_speed),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            parser.error(f"{name}必须在(0, 1]内")
    if not math.isfinite(args.lift_m) or not 0.0 < args.lift_m <= 0.20:
        parser.error("--lift-m必须在(0, 0.20]m内")
    if (
        not math.isfinite(args.graspup_tolerance_deg)
        or args.graspup_tolerance_deg <= 0.0
    ):
        parser.error("--graspup-tolerance-deg必须大于0")

    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url=args.execute_url,
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.graspup_speed,
        max_acceleration=args.graspup_speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)

    from parm_sdk import Arm

    target_graspup = np.asarray(GRASPUP_JOINTS_DEG, dtype=float)
    try:
        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        with Arm(host=args.arm_host) as arm:
            input(
                "确认左臂到graspup的路径安全，按回车规划RRTStar并执行..."
            )
            planner.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
            wait_for_joint_target(
                arm,
                target_graspup,
                joint_unit=args.joint_unit,
                tolerance_deg=args.graspup_tolerance_deg,
                timeout_s=90.0,
            )

            transform_response = planner.lookup_transform(
                "body_base", "left_hand_transfer_end"
            )
            transform = (transform_response or {}).get("data") or transform_response or {}
            translation = transform.get("translation") or {}
            rotation = transform.get("rotation") or {}
            current_xyz = [float(translation[axis]) for axis in "xyz"]
            current_xyzw = [float(rotation[axis]) for axis in "xyzw"]
            target_xyz = [
                current_xyz[0],
                current_xyz[1],
                current_xyz[2] + args.lift_m,
            ]
            target_pose = ObjectPose(
                frame_id="body_base",
                position=target_xyz,
                orientation=current_xyzw,
            )

            print("\n=== graspup上方目标 ===")
            print("frame_id = body_base")
            print("ik_frame = left_hand_transfer_end")
            print(f"graspup xyz (m) = {current_xyz}")
            print(f"target xyz (m) = {target_xyz}")
            print(f"target quaternion xyzw = {current_xyzw}")
            input(
                f"确认沿body_base +Z上移{args.lift_m:.3f}m路径安全，"
                "按回车规划LIN并执行..."
            )
            planner.config = replace(
                planner.config,
                max_velocity=args.lift_speed,
                max_acceleration=args.lift_speed,
            )
            planner.move_cartesian_pose("left", target_pose, planner="lin")
            actual_deg = wait_for_stable_joints(
                arm,
                args.joint_unit,
                stable_delta_deg=0.02,
                timeout_s=10.0,
            )

            formatted = ", ".join(f"{value:.6f}" for value in actual_deg)
            print("\n=== graspup_above实际关节角 ===")
            print("joint unit = deg")
            print(f"GRASPUP_ABOVE_JOINTS_DEG = [{formatted}]")
    finally:
        planner.close()


if __name__ == "__main__":
    main()
