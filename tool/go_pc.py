#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_pc.py 对应的 D405 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-184.04, -14.589, 175.401, 90.419, -101.435, 32.519, 89.542]


def main() -> None:
    args = build_parser("通过规划器移动左臂到 PC 观察关节位").parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG)


if __name__ == "__main__":
    main()
