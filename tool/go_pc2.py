#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_pc2.py 对应的 D405 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-10.496, -31.557, -4.878, 135.978, -99.147, 5.249, 45.166]


def main() -> None:
    args = build_parser("通过规划器移动左臂到 PC2 观察关节位").parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG)


if __name__ == "__main__":
    main()
