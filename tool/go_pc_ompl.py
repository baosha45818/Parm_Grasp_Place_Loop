#!/usr/bin/env python3
"""通过 OMPL/RRTConnect 将左臂移动到 D405 的 PC 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-184.04, -14.589, 175.401, 90.419, -101.435, 32.519, 89.542]


def main() -> None:
    parser = build_parser("通过 OMPL/RRTConnect 移动左臂到 PC 观察关节位")
    parser.set_defaults(arm_host="192.168.3.11")
    args = parser.parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG, planner_name="ompl")


if __name__ == "__main__":
    main()
