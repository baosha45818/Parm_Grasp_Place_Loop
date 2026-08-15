#!/usr/bin/env python3
"""只测试 PARM 离线关节规划，不向机械臂发送任何运动指令。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


SDK_WHEEL = next(iter(sorted((Path(__file__).resolve().parents[1] / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True)), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取当前关节角作为离线 start_state，只请求规划，不执行。"
    )
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-id", choices=(1, 2), type=int, default=1)
    parser.add_argument("--joint-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--planner", choices=("ptp", "lin", "ompl"), default="ompl")
    parser.add_argument("--delta-joint-index", choices=range(1, 8), type=int, default=7)
    parser.add_argument("--delta-deg", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=0.1)
    args = parser.parse_args()

    if not SDK_WHEEL.is_file():
        raise FileNotFoundError(f"找不到新版 SDK wheel: {SDK_WHEEL}")
    if not math.isfinite(args.delta_deg) or abs(args.delta_deg) > 5.0:
        parser.error("--delta-deg 必须为 [-5, 5] 内的有限数值")
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed 必须在 (0, 1] 内")

    sys.path.insert(0, str(SDK_WHEEL))
    from parm_sdk import Arm, Planner

    arm_name = "left" if args.arm_id == 1 else "right"
    group = f"{arm_name}_arm"
    joint_names = [f"{arm_name}_arm_joint{index}" for index in range(1, 8)]

    with Arm(host=args.arm_host) as arm:
        state = arm.get_state()
    arm_state = (state or {}).get("state", {}).get(arm_name)
    if arm_state is None:
        raise RuntimeError(f"未读取到 {arm_name} arm 状态: {state}")
    raw_q = [float(value) for value in arm_state.get("position", [])]
    if len(raw_q) != 7 or not all(math.isfinite(value) for value in raw_q):
        raise RuntimeError(f"当前关节角无效: {raw_q}")

    start_q_deg = raw_q if args.joint_unit == "deg" else [math.degrees(value) for value in raw_q]
    target_q_deg = list(start_q_deg)
    target_q_deg[args.delta_joint_index - 1] += args.delta_deg

    payload = {
        "start_state": {
            "joint_names": joint_names,
            "joint_values": start_q_deg,
        },
        "waypoints": [{
            "name": "offline_joint_plan_smoke_test",
            "type": "joint",
            "group": group,
            "planner": args.planner,
            "max_velocity_scaling_factor": args.speed,
            "max_acceleration_scaling_factor": args.speed,
            "joint_names": joint_names,
            "joint_values": target_q_deg,
        }],
        "topp": {
            "max_vel_factor": args.speed,
            "max_acc_factor": args.speed,
            "final_phase_ratio": 0.05,
            "final_phase_oversample": 2.0,
        },
    }

    print("=== PARM 离线关节规划只规划测试 ===")
    print("不会调用 /middle/waypoint/execute，不会发送任何机械臂运动指令。")
    print(f"group = {group}, planner = {args.planner}")
    print(f"start_state joint_values (deg) = {start_q_deg}")
    print(f"target joint_values (deg) = {target_q_deg}")
    print("payload =")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    with Planner(host=args.planner_host) as planner:
        result = planner._trajectory.send_payload(payload, timeout=30.0)

    data = result.get("data") or {}
    trajectory = data.get("trajectory") or []
    if not trajectory:
        raise RuntimeError(f"离线规划未返回轨迹: {result}")
    if int(data.get("arm_id", -1)) != args.arm_id:
        raise RuntimeError(f"规划器返回 arm_id={data.get('arm_id')}，预期 {args.arm_id}")
    if any(not isinstance(point, list) or len(point) != 7 for point in trajectory):
        raise RuntimeError(f"规划器返回的关节轨迹格式无效: {trajectory}")

    planned_start = [float(value) for value in trajectory[0]]
    planned_end = [float(value) for value in trajectory[-1]]
    start_error_deg = max(abs(actual - expected) for actual, expected in zip(planned_start, start_q_deg))
    goal_error_deg = max(abs(actual - expected) for actual, expected in zip(planned_end, target_q_deg))
    if start_error_deg > 0.05 or goal_error_deg > 0.05:
        raise RuntimeError(
            "离线关节轨迹首尾校验失败: "
            f"起点最大误差={start_error_deg:.4f}deg, 终点最大误差={goal_error_deg:.4f}deg"
        )

    print(
        f"\n离线规划成功：{len(trajectory)} 个关节路点，"
        f"sample_frequency={data.get('sample_frequency')} Hz"
    )
    print(f"returned first q (deg) = {planned_start}")
    print(f"returned last  q (deg) = {planned_end}")
    print(f"起点最大误差={start_error_deg:.4f}deg，终点最大误差={goal_error_deg:.4f}deg")
    print("测试结束：轨迹未执行，机械臂不会运动。")


if __name__ == "__main__":
    main()
