#!/usr/bin/env python3
"""Place right-held workpiece 2 through three explicitly taught joint poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from loop_config import LoopConfig
from tool.planner_runtime import PlannerRuntime
from tests.reversed_head_observation.check_plan import MODEL_JOINT_LIMITS_DEG
from tests.reversed_head_observation.execute_handoff import move_gripper
from tests.reversed_head_observation.handoff_attachment import request_attachment_owner


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "right_place_config.json"
DEFAULT_STATE = HERE / "handoff_attachment_state.json"


def load_place_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    names = (
        "right_preplace_joints_deg",
        "right_place_joints_deg",
        "right_retreat_joints_deg",
    )
    for name in names:
        values = config.get(name)
        if not isinstance(values, list) or len(values) != 7:
            raise RuntimeError(f"{path}: {name} 尚未完成7轴示教，拒绝执行完整放置")
        for index, (value, limits) in enumerate(zip(values, MODEL_JOINT_LIMITS_DEG), 1):
            value = float(value)
            if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
                raise ValueError(f"{name} J{index}={value} 超出模型范围 {limits}")
    if config.get("calibration_required") is True:
        raise RuntimeError(
            f"{path}: calibration_required 仍为 true；现场示教并核验三个位姿后才能置 false"
        )
    return config


def wait_right_joints(arm_host: str, target: list[float], tolerance: float) -> None:
    from parm_sdk import Arm

    deadline = time.monotonic() + 90.0
    last_error = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = ((state.get("state") or {}).get("right") or {}).get("position") or []
            if len(actual) == 7:
                error = (np.asarray(actual, dtype=float) - target + 180.0) % 360.0 - 180.0
                last_error = float(np.max(np.abs(error)))
                if last_error <= tolerance:
                    return
            time.sleep(0.05)
    raise RuntimeError(f"右臂放置关节位未到位，最大误差={last_error:.3f}°")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--attachment-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-workpiece-2-place-calibration", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_place_config(args.config)
    if args.execute and not args.confirm_workpiece_2_place_calibration:
        raise RuntimeError("实体放置必须增加 --confirm-workpiece-2-place-calibration")
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed 必须在 (0,1] 内")
    runtime = PlannerRuntime(LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.speed,
        max_acceleration=args.speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    ))
    poses = [
        ("预放置", list(map(float, config["right_preplace_joints_deg"]))),
        ("支撑释放", list(map(float, config["right_place_joints_deg"]))),
        ("释放后撤离", list(map(float, config["right_retreat_joints_deg"]))),
    ]
    try:
        runtime.open(required_frames={"body_base", "right_hand_transfer_end"})
        first_plan = runtime.plan_joint_trajectory("right", poses[0][1], planner="rrtstar")
        if first_plan is None:
            raise RuntimeError("当前位到右手预放置位规划失败")
        print(json.dumps({
            "mode": "EXECUTE" if args.execute else "PLAN_ONLY",
            "first_stage": poses[0][0],
            "trajectory_points": len((first_plan.get("data") or {}).get("trajectory") or []),
            "later_stages": "执行到前一阶段后从实时状态重新规划",
        }, ensure_ascii=False, indent=2))
        if not args.execute:
            print("[完成] 当前位到预放置位只规划成功；未执行机械臂或夹爪动作。")
            return 0

        tolerance = float(config.get("arrival_tolerance_deg", 0.5))
        plans = [(poses[0][0], poses[0][1], first_plan)]
        for index, (label, target) in enumerate(poses):
            if index:
                plan = runtime.plan_joint_trajectory("right", target, planner="rrtstar")
                if plan is None:
                    raise RuntimeError(f"右手{label}规划失败")
            else:
                plan = plans[0][2]
            if input(f"确认右手{label}轨迹、工件净空及急停均安全，输入 y 执行：").strip().lower() != "y":
                print(f"已取消；未执行右手{label}。")
                return 0
            runtime.execute_planned_result(plan)
            wait_right_joints(args.arm_host, target, tolerance)
            if index == 1:
                if input("确认工件已由放置面稳定支撑，输入 y 解除碰撞附着并打开右夹爪：").strip().lower() != "y":
                    print("已停止；右手继续夹持工件。")
                    return 0
                print(request_attachment_owner(args.attachment_state, "none", timeout_s=5.0))
                print(move_gripper(
                    args.gripper_host, 2,
                    float(config.get("right_open_position", 100.0)),
                    float(config.get("gripper_force", 60.0)), "right",
                ))
        print("[完成] 第二类工件已由右手放置、松夹并撤离。")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
