#!/usr/bin/env python3
"""通过规划器把右臂移动到反向交接尾部抓取示教关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


GRASP_INVERSE_RIGHT_JOINTS_DEG = [-5.6937, 43.1502, -112.7074, -53.3867, 77.7186, -39.9765, 28.4257]


def main() -> None:
    args = build_parser(
        "通过规划器移动右臂到反向交接尾部抓取示教关节位"
    ).parse_args()
    plan_and_execute(args, GRASP_INVERSE_RIGHT_JOINTS_DEG, arm="right")


if __name__ == "__main__":
    main()
