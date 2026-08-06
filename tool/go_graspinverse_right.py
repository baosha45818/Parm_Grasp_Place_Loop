#!/usr/bin/env python3
"""通过规划器把右臂移动到反向交接尾部抓取示教关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


GRASP_INVERSE_RIGHT_JOINTS_DEG = [ 161.36, 19.07, -107.979, 71.451, -55.833, -14.896, 88.2 ]


def main() -> None:
    args = build_parser(
        "通过规划器移动右臂到反向交接尾部抓取示教关节位"
    ).parse_args()
    plan_and_execute(args, GRASP_INVERSE_RIGHT_JOINTS_DEG, arm="right")


if __name__ == "__main__":
    main()
