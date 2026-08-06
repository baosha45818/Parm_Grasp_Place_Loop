#!/usr/bin/env python3
"""通过规划器把右臂移动到原 go_pc.py 对应的 D405 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [85.2016, 28.0536, -77.7928, 75.8290, -144.0116, -34.8774, -71.2299]


def main() -> None:
    args = build_parser("通过规划器移动右臂到 PC 观察关节位").parse_args()
    plan_and_execute(
        args,
        OBSERVATION_JOINTS_DEG,
        arm="right",
        planner_name="rrtstar",
    )


if __name__ == "__main__":
    main()
