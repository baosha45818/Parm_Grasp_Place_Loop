#!/usr/bin/env python3
"""只规划对比同一笛卡尔目标的 Pilz PTP 与 OMPL RRTConnect，不下发机械臂运动。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


SDK_WHEEL = next(iter(sorted((Path(__file__).resolve().parents[1] / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True)), None)


def plan_once(planner: object, *, planner_kind: str, args: argparse.Namespace) -> None:
    waypoint = {
        "name": f"compare_{planner_kind}_cartesian_target",
        "type": "cart",
        "group": args.group,
        "planner": planner_kind,
        "max_velocity_scaling_factor": args.max_vel,
        "max_acceleration_scaling_factor": args.max_acc,
        "frame_id": args.frame_id,
        "ik_frame": args.ik_frame,
        "position": args.target_xyz,
        "orientation": args.target_quaternion,
    }
    payload = {
        "waypoints": [waypoint],
        "topp": {
            "max_vel_factor": args.max_vel,
            "max_acc_factor": args.max_acc,
            "final_phase_ratio": 0.05,
            "final_phase_oversample": 2.0,
        },
    }
    print(f"\n=== {planner_kind.upper()}：提交完全相同的笛卡尔目标 ===")
    print(json.dumps(waypoint, ensure_ascii=False, indent=2))
    started = time.monotonic()
    try:
        result = planner._trajectory.send_payload(payload, timeout=args.timeout_s)
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f"{planner_kind.upper()} 规划 RPC 异常，耗时 {elapsed:.3f}s：{exc}")
        return

    elapsed = time.monotonic() - started
    data = result.get("data") or {}
    trajectory = data.get("trajectory") or []
    if trajectory:
        print(
            f"{planner_kind.upper()} 规划成功：{len(trajectory)} 个关节路点，"
            f"sample_frequency={data.get('sample_frequency')} Hz，耗时 {elapsed:.3f}s"
        )
        return
    print(f"{planner_kind.upper()} 未返回轨迹，耗时 {elapsed:.3f}s：")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="同一 body_base 笛卡尔目标分别用 PTP 与 RRTConnect 只规划对比。"
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--group", choices=("left_arm", "right_arm"), required=True)
    parser.add_argument("--arm-id", choices=(1, 2), type=int, required=True)
    parser.add_argument("--frame-id", default="body_base")
    parser.add_argument("--ik-frame", required=True)
    parser.add_argument("--target-xyz", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--target-quaternion",
        nargs=4,
        type=float,
        required=True,
        metavar=("QX", "QY", "QZ", "QW"),
    )
    parser.add_argument("--max-vel", type=float, default=0.1)
    parser.add_argument("--max-acc", type=float, default=0.1)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    args = parser.parse_args()

    expected_arm_id = {"left_arm": 1, "right_arm": 2}[args.group]
    if args.arm_id != expected_arm_id:
        parser.error(f"--group {args.group} 必须配合 --arm-id {expected_arm_id}")
    if args.frame_id != "body_base":
        parser.error("--frame-id 必须为 body_base")
    if not 0.0 < args.max_vel <= 1.0 or not 0.0 < args.max_acc <= 1.0:
        parser.error("--max-vel 与 --max-acc 必须在 (0, 1] 内")
    if args.timeout_s <= 0.0:
        parser.error("--timeout-s 必须大于 0")
    values = args.target_xyz + args.target_quaternion
    if not all(math.isfinite(value) for value in values):
        parser.error("目标位姿必须全为有限数")
    norm = math.sqrt(sum(value * value for value in args.target_quaternion))
    if norm < 1e-9:
        parser.error("--target-quaternion 不能为零")
    args.target_quaternion = [value / norm for value in args.target_quaternion]
    if not SDK_WHEEL.is_file():
        raise FileNotFoundError(f"找不到新版 SDK wheel: {SDK_WHEEL}")

    sys.path.insert(0, str(SDK_WHEEL))
    from parm_sdk import Planner

    print("只进行规划比较，不会调用执行接口，也不会向机械臂发送运动指令。")
    print(f"group={args.group}, arm_id={args.arm_id}, frame_id={args.frame_id}, ik_frame={args.ik_frame}")
    with Planner(host=args.planner_host) as planner:
        frames = set(planner.list_frame_ids())
        missing = sorted({args.frame_id, args.ik_frame}.difference(frames))
        if missing:
            raise RuntimeError(f"规划器 TF 缺少坐标系: {missing}")
        plan_once(planner, planner_kind="ptp", args=args)
        plan_once(planner, planner_kind="ompl", args=args)


if __name__ == "__main__":
    main()
