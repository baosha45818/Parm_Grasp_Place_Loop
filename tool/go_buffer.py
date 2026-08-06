#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_buffer.py 对应的关节 Buffer 位。"""

from planner_joint_motion import build_parser, plan_and_execute


BUFFER_JOINTS_DEG = [49.671, -56.037, -115.398, 72.201, 9.075, 53.369, 11.563]


def main() -> None:
    args = build_parser("通过规划器移动左臂到 Buffer 关节位").parse_args()
    plan_and_execute(args, BUFFER_JOINTS_DEG)


if __name__ == "__main__":
    main()
