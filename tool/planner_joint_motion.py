#!/usr/bin/env python3
"""提供独立脚本共用的左右臂关节规划、确认和轨迹执行功能。"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

from planner_task_retry import retry_task_creation


LEFT_JOINT_NAMES = [f"left_arm_joint{index}" for index in range(1, 8)]
RIGHT_JOINT_NAMES = [f"right_arm_joint{index}" for index in range(1, 8)]
LEFT_GROUP = "left_arm"
RIGHT_GROUP = "right_arm"
LEFT_ARM_ID = 1
RIGHT_ARM_ID = 2
def build_parser(description: str) -> argparse.ArgumentParser:
    """创建固定关节位姿脚本共用的命令行参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--planner-host",
        default="127.0.0.1",
        help="plucky 规划器服务所在主机，默认本机",
    )
    parser.add_argument(
        "--arm-host",
        default="192.168.3.11",
        help="PARM Arm SDK 主机，默认 192.168.3.11",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="规划器 max_vel/max_acc 比例，范围 (0, 1]，默认 0.1",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只规划并打印轨迹，不请求执行",
    )
    return parser


def plan_and_execute(
    args: argparse.Namespace,
    target_joints_deg: Sequence[float],
    *,
    arm: str = "left",
    planner_name: str = "rrtstar",
) -> None:
    """经 Planner 规划关节轨迹，再直接由 Arm SDK 执行该规划结果。"""
    _validate_target(target_joints_deg, args.speed)
    if arm == "left":
        group = LEFT_GROUP
        joint_names = LEFT_JOINT_NAMES
        expected_arm_id = LEFT_ARM_ID
        arm_label = "左臂"
    elif arm == "right":
        group = RIGHT_GROUP
        joint_names = RIGHT_JOINT_NAMES
        expected_arm_id = RIGHT_ARM_ID
        arm_label = "右臂"
    else:
        raise ValueError("arm 必须是 'left' 或 'right'")
    from parm_sdk import Planner

    if planner_name not in {"ompl", "rrtstar"}:
        raise ValueError("planner_name 必须是 'ompl' 或 'rrtstar'")
    print(
        f"目标：{group} 关节空间 {planner_name.upper()} 规划，"
        "关节单位为度（deg）"
    )
    print("  target:", [f"{value:.3f}" for value in target_joints_deg])
    print("规划器：", args.planner_host)
    try:
        with Planner(host=args.planner_host) as planner:
            # The deployed planner supports RRTStar, while this SDK's public
            # helper still locally restricts the planner names.
            payload = {
                "waypoints": [{
                    "name": f"{arm}_joint_target",
                    "type": "joint",
                    "group": group,
                    "planner": planner_name,
                    "max_velocity_scaling_factor": float(args.speed),
                    "max_acceleration_scaling_factor": float(args.speed),
                    "joint_names": joint_names,
                    "joint_values": [float(value) for value in target_joints_deg],
                }],
                "topp": {
                    "max_vel_factor": float(args.speed),
                    "max_acc_factor": float(args.speed),
                },
            }
            result = retry_task_creation(
                lambda: planner._trajectory.send_payload(payload, timeout=60.0),
                label=f"{arm_label} {planner_name.upper()} 关节规划",
            )
    except Exception as exc:
        raise RuntimeError(
            f"规划器不可用或规划失败（{args.planner_host}:5604）。"
            "请先启动 /home/dainanjun/Public/plucky_hjt_parm 的 docker compose，"
            "并确认 trajectory planner 服务已监听 5604。"
        ) from exc

    data = result.get("data") or {}
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError(f"规划器没有返回有效轨迹: {result}")
    arm_id = int(data.get("arm_id", -1))
    if arm_id != expected_arm_id:
        raise RuntimeError(
            f"规划器返回 arm_id={arm_id}，"
            f"预期{arm_label} arm_id={expected_arm_id}"
        )
    sample_frequency = float(data.get("sample_frequency", 10.0))
    if not math.isfinite(sample_frequency) or sample_frequency <= 0:
        raise RuntimeError(f"规划器返回无效 sample_frequency: {sample_frequency}")

    print(f"规划成功：{len(trajectory)} 个关节路点，{sample_frequency:.2f} Hz")
    print("  first :", [f"{value:.3f}" for value in trajectory[0]])
    print("  final :", [f"{value:.3f}" for value in trajectory[-1]])
    if args.plan_only:
        print("--plan-only：未发送任何执行请求。")
        return

    answer = input("确认规划轨迹和机械臂周围安全后，输入 y/yes 执行：").strip().lower()
    if answer not in {"y", "yes"}:
        print("已取消，未发送执行请求。")
        return

    from parm_sdk import Arm

    with Arm(host=args.arm_host) as sdk_arm:
        sdk_arm.follow_trajectory(
            arm_id=arm_id,
            trajectory=[[float(value) for value in point] for point in trajectory],
            sample_frequency=sample_frequency,
        )
    print(f"{arm_label}已完成规划轨迹执行。")


def _validate_target(target_joints_deg: Sequence[float], speed: float) -> None:
    if len(target_joints_deg) != 7 or not all(math.isfinite(value) for value in target_joints_deg):
        raise ValueError("目标关节角必须是 7 个有限数值")
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("--speed 必须在 (0, 1] 范围内")
