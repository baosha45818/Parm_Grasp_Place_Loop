#!/usr/bin/env python3
"""只规划、不执行：检查左臂从反向抓取中间位到头部观察位的路径。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


LEFT_GROUP = "left_arm"
LEFT_ARM_ID = 1
LEFT_JOINT_NAMES = [f"left_arm_joint{i}" for i in range(1, 8)]

# SDK / Planner 的关节输入和返回轨迹均为度。
START_JOINTS_DEG = [-112.208, 51.35, -3.083, 71.536, -0.401, 30.53, 84.765]
GOAL_JOINTS_DEG = [-14.704, 36.499, -75.262, 92.138, -91.44, 50.882, 88]
HEAD_TARGET_DEG = {"pitch": 2.0, "yaw": 26.5}
MODEL_JOINT_LIMITS_DEG = [
    (-224.943, 224.943),
    (-99.981, 99.981),
    (-224.943, 224.943),
    (-99.981, 149.943),
    (-224.943, 224.943),
    (-89.954, 89.954),
    (-89.954, 89.954),
]


class JointLimitError(ValueError):
    def __init__(self, violations: list[dict[str, Any]]):
        self.violations = violations
        details = ", ".join(
            f"{item['pose']} J{item['joint']}={item['value_deg']:.3f}° "
            f"不在 [{item['lower_deg']:.3f}, {item['upper_deg']:.3f}]°"
            for item in violations
        )
        super().__init__(details)


def joint_waypoint(name: str, values_deg: Sequence[float]) -> dict[str, Any]:
    """构造 Planner 接受的左臂 RRTStar 关节路点。"""
    validate_joint_vector(values_deg, name)
    return {
        "name": name,
        "type": "joint",
        "group": LEFT_GROUP,
        "planner": "rrtstar",
        "max_velocity_scaling_factor": 0.1,
        "max_acceleration_scaling_factor": 0.1,
        "joint_names": LEFT_JOINT_NAMES,
        "joint_values": [float(value) for value in values_deg],
    }


def validate_joint_vector(values: Sequence[float], label: str) -> None:
    if len(values) != 7 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} 必须包含 7 个有限关节角")


def max_joint_error_deg(actual: Sequence[float], expected: Sequence[float]) -> float:
    validate_joint_vector(actual, "actual")
    validate_joint_vector(expected, "expected")
    return max(abs(float(a) - float(b)) for a, b in zip(actual, expected))


def find_joint_limit_violations() -> list[dict[str, Any]]:
    """按当前 left_arm.urdf.xacro 检查两个记录位姿。"""
    violations: list[dict[str, Any]] = []
    for pose, values in (("start", START_JOINTS_DEG), ("goal", GOAL_JOINTS_DEG)):
        for index, (value, limits) in enumerate(zip(values, MODEL_JOINT_LIMITS_DEG), 1):
            lower, upper = limits
            if not lower <= value <= upper:
                violations.append(
                    {
                        "pose": pose,
                        "joint": index,
                        "value_deg": value,
                        "lower_deg": lower,
                        "upper_deg": upper,
                    }
                )
    return violations


def analyze_response(
    response: dict[str, Any],
    waypoint_tolerance_deg: float = 0.5,
    *,
    start_joints_deg: Sequence[float] = START_JOINTS_DEG,
    goal_joints_deg: Sequence[float] = GOAL_JOINTS_DEG,
) -> dict[str, Any]:
    """严格检查规划响应，并确认轨迹经过指定起点且到达终点。"""
    if response.get("success") is not True:
        raise RuntimeError(f"规划器报告失败: {response.get('message', response)!r}")

    data = response.get("data") or {}
    if int(data.get("arm_id", -1)) != LEFT_ARM_ID:
        raise RuntimeError(f"规划器返回 arm_id={data.get('arm_id')}，预期为 1（左臂）")

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError("规划器未返回轨迹")
    for index, point in enumerate(trajectory):
        validate_joint_vector(point, f"trajectory[{index}]")

    validate_joint_vector(start_joints_deg, "recorded start")
    validate_joint_vector(goal_joints_deg, "recorded goal")
    start_errors = [
        max_joint_error_deg(point, start_joints_deg) for point in trajectory
    ]
    start_index = min(range(len(start_errors)), key=start_errors.__getitem__)
    start_error = start_errors[start_index]
    goal_error = max_joint_error_deg(trajectory[-1], goal_joints_deg)
    if start_error > waypoint_tolerance_deg:
        raise RuntimeError(
            f"返回轨迹没有经过记录起点：最近误差 {start_error:.3f}°，"
            f"容差 {waypoint_tolerance_deg:.3f}°"
        )
    if goal_error > waypoint_tolerance_deg:
        raise RuntimeError(
            f"返回轨迹未到达观察位：终点误差 {goal_error:.3f}°，"
            f"容差 {waypoint_tolerance_deg:.3f}°"
        )

    segment = trajectory[start_index:]
    max_step = 0.0
    for previous, current in zip(segment, segment[1:]):
        max_step = max(max_step, max_joint_error_deg(previous, current))

    frequency = float(data.get("sample_frequency", 0.0))
    if not math.isfinite(frequency) or frequency <= 0:
        raise RuntimeError(f"无效 sample_frequency: {frequency}")

    return {
        "result": "PLANNED_COLLISION_FREE_IN_CONFIGURED_MOVEIT_MODEL",
        "arm_id": LEFT_ARM_ID,
        "trajectory_points_total": len(trajectory),
        "tested_segment_start_index": start_index,
        "tested_segment_points": len(segment),
        "sample_frequency_hz": frequency,
        "nearest_start_error_deg": start_error,
        "goal_error_deg": goal_error,
        "max_adjacent_joint_step_deg": max_step,
    }


def run_check(host: str, speed: float, tolerance_deg: float) -> dict[str, Any]:
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("--speed 必须在 (0, 1] 范围内")
    violations = find_joint_limit_violations()
    if violations:
        raise JointLimitError(violations)
    from parm_sdk import Planner

    # 预检通过后，从实时 current state 直接规划到观察位，再严格检查返回轨迹
    # 确实从记录起点附近开始；若现场不在记录起点则拒绝结论。
    waypoints = [joint_waypoint("reversed_head_observation", GOAL_JOINTS_DEG)]
    with Planner(host=host) as planner:
        payload = {
            "waypoints": waypoints,
            "topp": {
                "max_vel_factor": speed,
                "max_acc_factor": speed,
            },
        }
        # The SDK helper validates only ptp/lin/ompl locally.  The 5604
        # service accepts rrtstar joint waypoints, so do not reject this
        # reverse-grasp preflight before it reaches the planner.
        if waypoints[0]["planner"] == "rrtstar":
            response = planner._trajectory.send_payload(payload, timeout=60.0)
        else:
            response = planner.plan_trajectory(
                waypoints=waypoints,
                max_vel=speed,
                max_acc=speed,
                timeout=60.0,
            )
    result = analyze_response(response, tolerance_deg)
    result.update(
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "planner_host": host,
            "group": LEFT_GROUP,
            "start_joints_deg": START_JOINTS_DEG,
            "goal_joints_deg": GOAL_JOINTS_DEG,
            "requested_head_target_deg": HEAD_TARGET_DEG,
            "head_in_collision_model": False,
            "limitation": (
                "当前 plucky joints.yaml 将 gimbal 设为 simulation=true 且默认 0°；"
                f"本次 MoveIt 结果未按 pitch={HEAD_TARGET_DEG['pitch']}°, "
                f"yaw={HEAD_TARGET_DEG['yaw']}° 建模，且未包含被夹持工件几何体。"
            ),
            "executed": False,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只规划并检查反向抓取中间位到头部观察位；不会运动机械臂或云台"
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).with_name("last_plan_report.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_check(args.planner_host, args.speed, args.tolerance_deg)
    except JointLimitError as exc:
        report = {
            "result": "INVALID_JOINT_LIMITS_COLLISION_NOT_TESTED",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "violations": exc.violations,
            "model_limit_source": (
                "/home/dainanjun/Public/plucky_hjt_parm/welding_config/description/"
                "urdf/arm/left/left_arm.urdf.xacro"
            ),
            "requested_head_target_deg": HEAD_TARGET_DEG,
            "executed": False,
        }
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    except Exception as exc:
        report = {
            "result": "PLAN_FAILED_OR_COLLISION",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "executed": False,
        }
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("注意：这里只做规划，未向机械臂、夹爪或云台发送运动指令。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
