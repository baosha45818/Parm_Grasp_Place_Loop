#!/usr/bin/env python3
"""通过规划器把右臂移动到原 go_pc.py 对应的 D405 观察关节位。"""

from planner_joint_motion import build_parser, plan_and_execute

# # 1 old
# OBSERVATION_JOINTS_DEG =[
#         171.9469,
#         51.0013,
#         -71.8698,
#         90.78,
#         -26.4565,
#         -8.5611,
#         70.5734
# ]

# # 1 new
# OBSERVATION_JOINTS_DEG =[
#         164.5464,
#         56.8637,
#         -68.9227,
#         78.2515,
#         -24.3923,
#         3.9098,
#         78.3202
# ]

# # 2 old
# OBSERVATION_JOINTS_DEG =[
#         162.5673,
#         58.7975,
#         -71.0898,
#         78.7033,
#         -21.5511,
#         -1.0972,
#         79.6811
#       ]

# 2 new
OBSERVATION_JOINTS_DEG = [
    163.136,
    59.4977,
    -71.1186,
    80.2579,
    -21.8408,
    -0.7594,
    79.0549,
]
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
