#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_graspnet.py 对应的关节观察位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-6.25, -21.348, 12.599, 78.268, 4.039, 56.382, -11.374]


def main() -> None:
    args = build_parser("通过规划器移动左臂到 GraspNet 观察关节位") .parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG)


if __name__ == "__main__":
    main()
