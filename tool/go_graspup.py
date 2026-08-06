#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_graspup.py 对应的抓取后上提关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


GRASPUP_JOINTS_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]


def main() -> None:
    args = build_parser("通过规划器移动左臂到抓取后上提关节位").parse_args()
    plan_and_execute(args, GRASPUP_JOINTS_DEG)


if __name__ == "__main__":
    main()
