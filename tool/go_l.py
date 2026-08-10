#!/usr/bin/env python3
"""通过规划器把左臂移动到原 go_graspnet.py 对应的关节观察位。"""

from planner_joint_motion import build_parser, plan_and_execute


OBSERVATION_JOINTS_DEG = [-14.704, 36.499, -75.262, 92.138, -91.44, 50.882, 88]


def print_reached_cartesian_endpoint(planner_host: str) -> None:
    """打印执行后由 Planner TF 报告的左臂末端位姿。"""
    from parm_sdk import Planner

    with Planner(host=planner_host) as planner:
        response = planner.lookup_transform(
            "body_base", "left_hand_transfer_end"
        )
    payload = (response or {}).get("data") or response or {}
    translation = payload.get("translation") or {}
    rotation = payload.get("rotation") or {}
    try:
        xyz = [float(translation[axis]) for axis in "xyz"]
        xyzw = [float(rotation[axis]) for axis in "xyzw"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"规划器未返回有效的左臂末端 TF: {response!r}") from exc
    print(
        "笛卡尔路径末点（执行后实时 TF）："
        f"frame=body_base, tool=left_hand_transfer_end, "
        f"xyz(m)={[round(value, 6) for value in xyz]}, "
        f"xyzw={[round(value, 6) for value in xyzw]}",
        flush=True,
    )


def main() -> None:
    args = build_parser("通过规划器移动左臂到 GraspNet 观察关节位").parse_args()
    plan_and_execute(args, OBSERVATION_JOINTS_DEG)
    if args.plan_only:
        print("--plan-only：关节轨迹没有执行，无法读取实际笛卡尔末点。")
        return
    print_reached_cartesian_endpoint(args.planner_host)


if __name__ == "__main__":
    main()
