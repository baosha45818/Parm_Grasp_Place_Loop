#!/usr/bin/env python3
"""规划成功并人工确认后，执行左臂轨迹并把云台转到观察角。"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

from .check_plan import (
    GOAL_JOINTS_DEG,
    HEAD_TARGET_DEG,
    LEFT_ARM_ID,
    START_JOINTS_DEG,
    analyze_response,
    joint_waypoint,
    max_joint_error_deg,
    validate_joint_vector,
)


class TaskBusyError(RuntimeError):
    """目标机械臂已有 SDK 任务占用。"""


def analyze_target_response(
    response: dict[str, Any], target_deg: list[float], tolerance_deg: float
) -> dict[str, Any]:
    """校验一段轨迹有效且准确到达指定关节目标。"""
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
    goal_error = max_joint_error_deg(trajectory[-1], target_deg)
    if goal_error > tolerance_deg:
        raise RuntimeError(
            f"返回轨迹未到达阶段目标：终点误差 {goal_error:.3f}°，"
            f"容差 {tolerance_deg:.3f}°"
        )
    frequency = float(data.get("sample_frequency", 0.0))
    if not math.isfinite(frequency) or frequency <= 0:
        raise RuntimeError(f"无效 sample_frequency: {frequency}")
    max_step = 0.0
    for previous, current in zip(trajectory, trajectory[1:]):
        max_step = max(max_step, max_joint_error_deg(previous, current))
    return {
        "result": "PLANNED_COLLISION_FREE_IN_CONFIGURED_MOVEIT_MODEL",
        "arm_id": LEFT_ARM_ID,
        "trajectory_points": len(trajectory),
        "sample_frequency_hz": frequency,
        "goal_error_deg": goal_error,
        "max_adjacent_joint_step_deg": max_step,
    }


def plan_to_target(
    host: str,
    speed: float,
    tolerance_deg: float,
    waypoint_name: str,
    target_deg: list[float],
    *,
    require_recorded_start: bool,
    recorded_start_deg: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """从 MoveIt 实时状态规划一段轨迹。"""
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("--speed 必须在 (0, 1] 范围内")
    from parm_sdk import Planner

    with Planner(host=host) as planner:
        waypoint = joint_waypoint(waypoint_name, target_deg)
        payload = {
            "waypoints": [waypoint],
            "topp": {
                "max_vel_factor": speed,
                "max_acc_factor": speed,
            },
        }
        # SDK 20260803's plan_trajectory() validates the historical
        # ptp/lin/ompl set locally.  The 5604 planner server additionally
        # supports rrtstar, so send this standard waypoint payload directly.
        if waypoint["planner"] == "rrtstar":
            response = planner._trajectory.send_payload(payload, timeout=60.0)
        else:
            response = planner.plan_trajectory(
                waypoints=[waypoint],
                max_vel=speed,
                max_acc=speed,
                timeout=60.0,
            )
    if require_recorded_start:
        summary = analyze_response(
            response,
            tolerance_deg,
            start_joints_deg=(recorded_start_deg or START_JOINTS_DEG),
            goal_joints_deg=target_deg,
        )
    else:
        summary = analyze_target_response(response, target_deg, tolerance_deg)
    return response, summary


def plan_to_middle(
    host: str,
    speed: float,
    tolerance_deg: float,
    target_deg: list[float] = START_JOINTS_DEG,
):
    return plan_to_target(
        host,
        speed,
        tolerance_deg,
        "reversed_grasp_middle",
        target_deg,
        require_recorded_start=False,
    )


def plan_motion(
    host: str,
    speed: float,
    tolerance_deg: float,
    start_deg: list[float] = START_JOINTS_DEG,
    goal_deg: list[float] = GOAL_JOINTS_DEG,
):
    """从已到达的记录中间位规划到观察位，并验证第二段起点。"""
    return plan_to_target(
        host,
        speed,
        tolerance_deg,
        "reversed_head_observation",
        goal_deg,
        require_recorded_start=True,
        recorded_start_deg=start_deg,
    )


def build_execute_payload(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data") or {}
    trajectory = data.get("trajectory")
    frequency = float(data.get("sample_frequency", 0.0))
    arm_id = int(data.get("arm_id", -1))
    if arm_id != LEFT_ARM_ID or not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError("规划响应缺少有效左臂轨迹")
    if not math.isfinite(frequency) or frequency <= 0:
        raise RuntimeError(f"无效 sample_frequency: {frequency}")
    return {
        "arm_id": arm_id,
        "trajectory": trajectory,
        "sample_frequency": frequency,
    }


def execute_trajectory(arm_host: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send the Planner's degree-valued joint trajectory directly to Arm SDK."""
    arm_id = int(payload.get("arm_id", -1))
    trajectory = payload.get("trajectory")
    sample_frequency = float(payload.get("sample_frequency", 0.0))
    if arm_id != LEFT_ARM_ID or not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError("直接 SDK 执行需要有效的左臂规划轨迹")
    if not math.isfinite(sample_frequency) or sample_frequency <= 0.0:
        raise RuntimeError(f"无效 sample_frequency: {sample_frequency}")
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
        task_state = str(
            (((state.get("state") or {}).get("left") or {}).get("task_state") or "")
        ).strip().lower()
        if task_state == "running":
            raise TaskBusyError("左臂已有 SDK 任务正在运行")
        arm.follow_trajectory(
            arm_id=LEFT_ARM_ID,
            trajectory=[[float(value) for value in point] for point in trajectory],
            sample_frequency=sample_frequency,
        )
    return {"success": True, "arm_id": LEFT_ARM_ID}


def wait_for_left_joint_target(
    arm_host: str,
    target_deg: list[float],
    *,
    tolerance_deg: float,
    timeout_s: float = 90.0,
) -> None:
    """Wait for direct arm feedback; the HTTP trajectory executor returns on acceptance."""
    from parm_sdk import Arm

    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = ((state.get("state") or {}).get("left") or {}).get("position") or []
            if len(actual) == 7:
                errors = [
                    abs((float(value) - float(target) + 180.0) % 360.0 - 180.0)
                    for value, target in zip(actual, target_deg)
                ]
                last_error_deg = max(errors)
                if last_error_deg <= tolerance_deg:
                    print(f"左臂已实际到位，最大关节误差={last_error_deg:.3f}deg", flush=True)
                    return
            time.sleep(0.05)
    raise RuntimeError(
        f"左臂未在 {timeout_s:.1f}s 内到达规划终点；"
        f"最后最大关节误差={last_error_deg:.3f}deg"
    )


def cancel_existing_task(arm_host: str) -> None:
    """直接通过 SDK 取消左臂当前任务；调用前必须由操作者明确确认。"""
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        arm.cancel_task(LEFT_ARM_ID)


def confirmed(answer: str) -> bool:
    return answer.strip().lower() in {"y", "yes"}


def ask_confirm(args: argparse.Namespace, prompt: str) -> bool:
    if args.auto_confirm:
        print(f"{prompt}[自动确认]")
        return True
    return confirmed(input(prompt))


def run_arm_phase(args: argparse.Namespace, label: str, planner_fn) -> dict[str, Any]:
    """规划、确认并执行一个阶段；任务占用时可确认软停止后重新规划。"""
    response, summary = planner_fn(
        args.planner_host, args.speed, args.tolerance_deg
    )
    payload = build_execute_payload(response)
    print(f"{label}规划通过：")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not ask_confirm(args, f"确认现场安全，输入 y 执行{label}："):
        raise InterruptedError(f"已取消{label}，未发送运动指令")

    try:
        result = execute_trajectory(args.arm_host, payload)
        wait_for_left_joint_target(
            args.arm_host,
            payload["trajectory"][-1],
            tolerance_deg=args.tolerance_deg,
        )
        return result
    except TaskBusyError:
        print("检测到左臂已有任务，可能是录制/示教任务没有退出或其他程序正在控制。")
        if not ask_confirm(args, "确认要软停止左臂现有任务并重新规划，输入 y："):
            raise InterruptedError("已取消；没有停止现有任务")
        cancel_existing_task(args.arm_host)
        time.sleep(0.5)
        response, summary = planner_fn(
            args.planner_host, args.speed, args.tolerance_deg
        )
        payload = build_execute_payload(response)
        print(f"现有任务已停止，{label}重新规划通过：")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not ask_confirm(args, f"再次确认现场安全，输入 y 执行{label}新轨迹："):
            raise InterruptedError(f"已取消{label}新轨迹")
        result = execute_trajectory(args.arm_host, payload)
        wait_for_left_joint_target(
            args.arm_host,
            payload["trajectory"][-1],
            tolerance_deg=args.tolerance_deg,
        )
        return result


def head_target_reached(
    state: dict[str, Any] | None, tolerance_deg: float = 1.0
) -> bool:
    if not isinstance(state, dict):
        return False
    pitch = state.get("pitch") or {}
    yaw = state.get("yaw") or {}
    try:
        # 当前 SDK 下发 pitch 前会取反，但状态读取没有把符号转回来。
        expected_pitch_feedback = -float(HEAD_TARGET_DEG["pitch"])
        return (
            abs(float(pitch["position"]) - expected_pitch_feedback) <= tolerance_deg
            and abs(float(yaw["position"]) - float(HEAD_TARGET_DEG["yaw"]))
            <= tolerance_deg
        )
    except (KeyError, TypeError, ValueError):
        return False


def wait_head_state(head: Any, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_state = None
    while time.monotonic() < deadline:
        state = head.get_state()
        if isinstance(state, dict):
            last_state = state
            if head_target_reached(state):
                return state
        time.sleep(0.1)
    raise RuntimeError(
        f"云台在 {timeout_s:.1f}s 内未到达目标；最后状态: {last_state}"
    )


def move_head(host: str, timeout_s: float) -> dict[str, Any]:
    from parm_sdk import Head

    with Head(host=host) as head:
        head.ping()
        # 等待 SUB 收到首帧；已使能时不要重复 init，以免重新归中/锁定覆盖动作。
        deadline = time.monotonic() + 2.0
        state = None
        while state is None and time.monotonic() < deadline:
            state = head.get_state()
            time.sleep(0.05)
        pitch = (state or {}).get("pitch") or {}
        yaw = (state or {}).get("yaw") or {}
        if pitch.get("alarm_message") or yaw.get("alarm_message"):
            raise RuntimeError(f"云台存在报警，拒绝运动: {state}")
        if not pitch.get("enabled") or not yaw.get("enabled"):
            head.init()
            time.sleep(0.3)
        head.move_to(
            pitch=HEAD_TARGET_DEG["pitch"],
            yaw=HEAD_TARGET_DEG["yaw"],
        )
        return wait_head_state(head, timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="规划并执行反向抓取中间位到头部观察位，然后转动云台"
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--tolerance-deg", type=float, default=0.5)
    parser.add_argument(
        "--middle-joints-deg",
        type=float,
        nargs=7,
        default=START_JOINTS_DEG,
        help="观察动作第一阶段的七轴准备位；缺省保持原工件1记录值",
    )
    parser.add_argument(
        "--observation-joints-deg",
        type=float,
        nargs=7,
        default=GOAL_JOINTS_DEG,
        help="第二阶段的七轴观察位；缺省保持原工件1记录值",
    )
    parser.add_argument(
        "--head-timeout",
        type=float,
        default=15.0,
        help="等待云台到达目标的秒数，默认 15",
    )
    parser.add_argument(
        "--skip-head",
        action="store_true",
        help="只执行左臂轨迹，不转动云台",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="手臂已到位时只转动并校验云台，不规划或执行手臂",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_joint_vector(args.middle_joints_deg, "middle joints")
    validate_joint_vector(args.observation_joints_deg, "observation joints")
    if args.head_only:
        print(
            f"仅云台模式：目标 pitch={HEAD_TARGET_DEG['pitch']}°, "
            f"yaw={HEAD_TARGET_DEG['yaw']}°."
        )
        if not ask_confirm(args, "确认云台周围安全，输入 y 执行："):
            print("已取消，云台未运动。")
            return 2
        try:
            state = move_head(args.driver_host, args.head_timeout)
        except Exception as exc:
            print(f"云台未到达目标：{type(exc).__name__}: {exc}")
            return 4
        print(f"云台已到达目标，状态反馈={state}")
        return 0

    print("阶段 1/2：规划当前状态到反向抓取中间起始点。")
    try:
        result = run_arm_phase(
            args,
            "阶段 1（到中间起始点）",
            lambda host, speed, tolerance: plan_to_middle(
                host, speed, tolerance, list(args.middle_joints_deg)
            ),
        )
    except InterruptedError as exc:
        print(exc)
        return 2
    except Exception as exc:
        print(f"阶段 1 失败：{type(exc).__name__}: {exc}")
        return 1
    print(f"阶段 1 执行完成：{result}")

    print("阶段 2/2：从实时中间状态规划到目标观察位。")
    print(f"目标关节角(°): {args.observation_joints_deg}")
    if not args.skip_head:
        print(
            "警告：当前 MoveIt 未按目标云台角和被夹持工件几何体做碰撞检查；"
            f"手臂完成后将转到 pitch={HEAD_TARGET_DEG['pitch']}°, "
            f"yaw={HEAD_TARGET_DEG['yaw']}°."
        )
    try:
        result = run_arm_phase(
            args,
            "阶段 2（到目标观察位）",
            lambda host, speed, tolerance: plan_motion(
                host,
                speed,
                tolerance,
                list(args.middle_joints_deg),
                list(args.observation_joints_deg),
            ),
        )
    except InterruptedError as exc:
        print(exc)
        return 2
    except Exception as exc:
        print(f"阶段 2 失败；未转动云台：{type(exc).__name__}: {exc}")
        return 3
    print(f"阶段 2 执行完成：{result}")

    if args.skip_head:
        print("--skip-head：云台未运动。")
        return 0
    try:
        head_state = move_head(args.driver_host, args.head_timeout)
    except Exception as exc:
        print(f"左臂已到观察位，但云台运动失败：{type(exc).__name__}: {exc}")
        return 4
    print(
        f"云台已到达目标：pitch={HEAD_TARGET_DEG['pitch']}°, "
        f"yaw={HEAD_TARGET_DEG['yaw']}°；状态反馈={head_state}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
