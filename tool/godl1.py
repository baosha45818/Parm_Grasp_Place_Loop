#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_pc.py 对应的 D405 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-112.208, 51.35, -3.083, 71.536, -0.401, 30.53, 84.765]


def main() -> None:
    args = build_parser("通过规划器移动左臂到 PC 观察关节位").parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG)


if __name__ == "__main__":
    main()
