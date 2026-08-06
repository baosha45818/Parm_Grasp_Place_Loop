#!/usr/bin/env python3
"""Move arm1 to the fixed D405 observation pose after terminal confirmation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))
from planner_motion import DEFAULT_EXECUTE_URL, DEFAULT_PLANNER_HOST, plan_joint_rrtstar


# OBSERVATION_JOINTS_DEG = [
#     # 358.441 deg is equivalent to -1.559 deg, but J1 accepts only [-227, 227].
#     -1.559,
#     -8.031,
#     -153.808,
#     -57.798,
#     -162.852,
#     59.599,
#     -42.667,
# ]

OBSERVATION_JOINTS_DEG =[-184.04, -14.589, 175.401, 90.419, -101.435, 32.519, 91.542]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move arm1 to the fixed D405 observation joint pose."
    )
    parser.add_argument("--arm-host", default="192.168.88.1", help=argparse.SUPPRESS)
    parser.add_argument("--arm-id", type=int, default=1, choices=(1,))
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--planner-host", default=DEFAULT_PLANNER_HOST)
    parser.add_argument("--execute-url", default=DEFAULT_EXECUTE_URL)
    return parser


def confirm_target(joints_deg: Sequence[float]) -> bool:
    print("目标：arm1 move_j，关节单位为度（deg）")
    print("  ", [f"{value:.3f}" for value in joints_deg])
    print("请确认机械臂周围无人、无障碍物，并确认目标位姿安全。")
    answer = input("确认执行请输入 y/yes，其他输入取消：").strip().lower()
    return answer in {"y", "yes"}


def choose_nearest_equivalent(
    target: float, current: float, *, lower: float | None = None, upper: float | None = None
) -> float:
    """Choose the nearest equivalent target, optionally constrained to command limits."""
    candidates = [target + 360.0 * turns for turns in range(-2, 3)]
    if lower is not None or upper is not None:
        candidates = [
            value
            for value in candidates
            if (lower is None or value >= lower) and (upper is None or value <= upper)
        ]
    if not candidates:
        raise RuntimeError("目标不存在控制器范围内的 360 度等效角")
    return min(candidates, key=lambda value: abs(value - current))


def move_arm(args: argparse.Namespace, requested_joints_deg: Sequence[float]) -> None:
    if len(requested_joints_deg) != 7 or not all(
        math.isfinite(value) for value in requested_joints_deg
    ):
        raise RuntimeError("目标关节角必须是 7 个有限数值")
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise RuntimeError("--speed 必须是正数")

    target_joints = [float(value) for value in requested_joints_deg]
    print("请求目标(deg):", [f"{value:.3f}" for value in target_joints])
    if not confirm_target(target_joints):
        print("已取消，未发送任何机器人指令。")
        return
    plan_joint_rrtstar(target_joints, planner_host=args.planner_host, speed=args.speed,
                   execute_url=args.execute_url, execute=True)


def main() -> None:
    args = build_parser().parse_args()
    move_arm(args, OBSERVATION_JOINTS_DEG)


if __name__ == "__main__":
    main()
