#!/usr/bin/env python3
"""通过 OMPL/RRTConnect 将左臂移动到 PC2 的 D405 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-10.496, -31.557, -4.878, 135.978, -99.147, 5.249, 45.166]


def main() -> None:
    parser = build_parser("通过 OMPL/RRTConnect 移动左臂到 PC2 观察关节位")
    parser.set_defaults(arm_host="192.168.3.11")
    args = parser.parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG, planner_name="ompl")


if __name__ == "__main__":
    main()
