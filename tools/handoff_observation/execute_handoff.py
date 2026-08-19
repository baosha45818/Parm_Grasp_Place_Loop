#!/usr/bin/env python3
"""Staged direct handoff with replanning and explicit operator confirmations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(__file__).resolve().parents[2] / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable) != PYTHON:
    os.execv(
        str(PYTHON),
        [
            str(PYTHON),
            "-m",
            "tools.handoff_observation.execute_handoff",
            *sys.argv[1:],
        ],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.handoff_observation.handoff_attachment import (
    request_attachment_owner,
)
from tools.handoff_observation.handoff_geometry import (
    compute_handoff_targets,
    planner_tf_target_source,
    transform_debug,
    transform_error,
    transform_to_object_pose,
    validate_transform,
)
from tools.handoff_observation.preflight_handoff import (
    build_runtime,
    load_icp_object_transform,
    required_transform,
    trajectory_summary,
    validate_icp_for_motion,
)
from tools.handoff_observation.check_plan import (
    START_JOINTS_DEG,
    max_joint_error_deg,
    validate_joint_vector,
)


HERE = Path(__file__).resolve().parent
DEFAULT_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="左手送到胸前、右手固定接取的分阶段执行器；默认只规划左手"
    )
    parser.add_argument("--icp-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--attachment-state",
        type=Path,
        default=HERE / "handoff_attachment_state.json",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--execute-url", default=DEFAULT_EXECUTE_URL)
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--right-close-position", type=float, default=0.0)
    parser.add_argument("--right-open-position", type=float, default=100.0)
    parser.add_argument("--left-open-position", type=float, default=100.0)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    parser.add_argument(
        "--left-return-joints-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=START_JOINTS_DEG,
        help="右手接稳、左手释放后回到的左臂中间初始关节位（度）",
    )
    parser.add_argument("--left-return-tolerance-deg", type=float, default=0.5)
    return parser.parse_args()


def confirmed(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def read_tcp(planner, tcp_frame: str) -> np.ndarray:
    response = planner.lookup_transform(tcp_frame, "body_base")
    if response is None:
        raise RuntimeError(f"Planner returned no TF for {tcp_frame}")
    return planner_tf_target_source(response, tcp_frame, "body_base")


def require_pose_close(
    actual: np.ndarray,
    target: np.ndarray,
    translation_tolerance_mm: float,
    rotation_tolerance_deg: float,
    label: str,
) -> dict[str, float]:
    translation_m, rotation_deg = transform_error(actual, target)
    error = {
        "translation_mm": translation_m * 1000.0,
        "rotation_deg": rotation_deg,
    }
    if (
        error["translation_mm"] > translation_tolerance_mm
        or error["rotation_deg"] > rotation_tolerance_deg
    ):
        raise RuntimeError(
            f"{label} did not reach target: {error['translation_mm']:.3f} mm/"
            f"{error['rotation_deg']:.3f}°"
        )
    return error


def move_gripper(
    host: str, hand: int, position: float, force: float, label: str
) -> dict:
    from parm_sdk import Gripper

    with Gripper(host=host) as gripper:
        state = gripper.get_state().get(hand)
        if state is None or state.get("initialized") is not True:
            raise RuntimeError(f"{label} gripper is unavailable or not initialized")
        response = gripper.move(
            hand,
            position,
            force=force,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
        final_state = gripper.get_state().get(hand)
    if response.get("status") != "success":
        raise RuntimeError(f"{label} gripper command failed: {response}")
    expected_states = {"arrived", "gripping"} if position <= 5.0 else {"arrived"}
    if final_state is None or final_state.get("motion_state") not in expected_states:
        raise RuntimeError(
            f"{label} gripper command returned but has unexpected final state: {final_state}"
        )
    return response


def read_gripper_state(host: str, hand: int, label: str) -> dict:
    from parm_sdk import Gripper

    with Gripper(host=host) as gripper:
        state = gripper.get_state().get(hand)
    if state is None or state.get("initialized") is not True:
        raise RuntimeError(f"{label} gripper is unavailable or not initialized")
    return state


def validate_joint_return_plan(
    response: dict, target_deg: list[float], tolerance_deg: float
) -> dict:
    """Verify that a left-arm joint plan reaches the requested safe return pose."""
    if tolerance_deg <= 0.0:
        raise ValueError("left return tolerance must be positive")
    data = response.get("data") or {}
    if int(data.get("arm_id", -1)) != 1:
        raise RuntimeError(f"left return plan has unexpected arm_id: {data.get('arm_id')}")
    trajectory = data.get("trajectory") or []
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError("left return plan has no trajectory")
    for index, point in enumerate(trajectory):
        validate_joint_vector(point, f"left_return_trajectory[{index}]")
    endpoint_error = max_joint_error_deg(trajectory[-1], target_deg)
    if endpoint_error > tolerance_deg:
        raise RuntimeError(
            f"left return plan endpoint error {endpoint_error:.3f}° exceeds "
            f"{tolerance_deg:.3f}°"
        )
    return {
        "arm_id": 1,
        "trajectory_points": len(trajectory),
        "sample_frequency_hz": float(data.get("sample_frequency", 10.0)),
        "target_joints_deg": target_deg,
        "endpoint_error_deg": endpoint_error,
    }


def main() -> int:
    args = parse_args()
    if not (
        0.0 <= args.right_close_position <= 100.0
        and 0.0 <= args.right_open_position <= 100.0
        and 0.0 <= args.left_open_position <= 100.0
        and 20.0 <= args.gripper_force <= 100.0
    ):
        raise ValueError("gripper positions must be 0..100 and force must be 20..100")
    if args.right_open_position - args.right_close_position < 10.0:
        raise ValueError("right open and close positions must differ by at least 10%")
    validate_joint_vector(args.left_return_joints_deg, "left return joints")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("transform_convention") != "p_A = T_A_B @ p_B":
        raise RuntimeError("handoff config transform convention is missing or invalid")
    if not config.get("attached_collision"):
        raise RuntimeError("handoff config has no attached collision model")
    T_base_object_current, icp_result = load_icp_object_transform(args.icp_result)
    T_base_left_tcp_captured = validate_icp_for_motion(icp_result, config)
    T_base_object_handoff = required_transform(config, "T_base_object_handoff")
    T_right_tcp_object = required_transform(config, "T_right_tcp_object")
    T_left_tcp_object_recorded = required_transform(config, "T_left_tcp_object")
    pregrasp_offset = config.get("right_pregrasp_offset_tcp_m")
    if pregrasp_offset is None:
        raise RuntimeError("right_pregrasp_offset_tcp_m has not been recorded")

    relation_from_icp = np.linalg.inv(T_base_left_tcp_captured) @ T_base_object_current
    relation_error = require_pose_close(
        relation_from_icp,
        T_left_tcp_object_recorded,
        0.5,
        0.2,
        "configured left TCP/object relation",
    )
    attachment_ack_left = request_attachment_owner(
        args.attachment_state, "left", timeout_s=5.0
    )

    planner = build_runtime(
        config, args.planner_host, execute_url=args.execute_url
    )
    try:
        left_frame = planner.config.left_tool_frame
        right_frame = planner.config.right_tool_frame
        planner.open(required_frames={"body_base", left_frame, right_frame})
        T_base_left_tcp_current = read_tcp(planner, left_frame)
        require_pose_close(
            T_base_left_tcp_current,
            T_base_left_tcp_captured,
            float(config["current_vs_capture_tcp_max_translation_mm"]),
            float(config["current_vs_capture_tcp_max_rotation_deg"]),
            "left TCP versus ICP capture pose",
        )
        targets = compute_handoff_targets(
            T_base_object_current,
            T_base_left_tcp_captured,
            T_base_object_handoff,
            T_right_tcp_object,
            pregrasp_offset,
        )
        left_plan = planner.plan_cartesian_pose(
            "left",
            transform_to_object_pose(targets["T_base_left_tcp_handoff"]),
            planner="rrtstar",
        )
        if left_plan is None:
            raise RuntimeError("Planner returned no left handoff plan")
        summary = {
            "mode": "EXECUTE" if args.execute else "PLAN_ONLY",
            "left_tcp_object_relation_error": relation_error,
            "attachment_ack_left": attachment_ack_left,
            "left_target": transform_debug(targets["T_base_left_tcp_handoff"]),
            "left_plan": trajectory_summary(left_plan, 1),
            "right_stage": "deferred_until_left_is_physically_at_handoff",
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not args.execute:
            print("[完成] 只规划了左手到胸前；没有执行运动或夹爪动作。")
            return 0

        if not confirmed("确认现场安全，输入 y 执行左手到胸前："):
            print("已取消；未执行运动。")
            return 0
        planner.execute_planned_result(left_plan)
        left_arrival = read_tcp(planner, left_frame)
        left_arrival_error = require_pose_close(
            left_arrival,
            targets["T_base_left_tcp_handoff"],
            float(config["left_at_handoff_translation_tolerance_mm"]),
            float(config["left_at_handoff_rotation_tolerance_deg"]),
            "left handoff pose",
        )
        print(f"左手已到胸前交接位：{left_arrival_error}")

        right_state = read_gripper_state(args.gripper_host, 2, "right")
        right_opening = float(right_state.get("position", -1.0))
        if right_opening < args.right_open_position - 5.0:
            if not confirmed(
                f"右夹爪当前开度={right_opening:.1f}%，输入 y 张开到"
                f"{args.right_open_position:.1f}%："
            ):
                print("已取消；左手保持交接位并继续持有工件。")
                return 0
            opened = move_gripper(
                args.gripper_host,
                2,
                args.right_open_position,
                args.gripper_force,
                "right",
            )
            print(f"右夹爪已张开：{opened}")

        right_plan = planner.plan_cartesian_sequence(
            "right",
            [
                    ("rrtstar", transform_to_object_pose(
                    targets["T_base_right_tcp_pregrasp"]
                )),
                ("lin", transform_to_object_pose(
                    targets["T_base_right_tcp_grasp"]
                )),
            ],
        )
        if right_plan is None:
            raise RuntimeError("Planner returned no right handoff approach plan")
        print(json.dumps({
            "right_target": transform_debug(targets["T_base_right_tcp_grasp"]),
            "right_plan": trajectory_summary(right_plan, 2),
        }, indent=2, ensure_ascii=False))
        if not confirmed("确认双臂间隙安全，输入 y 执行右手预抓取和直线接近："):
            print("已取消；左手保持交接位并继续持有工件。")
            return 0
        planner.execute_planned_result(right_plan)
        right_arrival = read_tcp(planner, right_frame)
        require_pose_close(
            right_arrival,
            targets["T_base_right_tcp_grasp"],
            float(config["left_at_handoff_translation_tolerance_mm"]),
            float(config["left_at_handoff_rotation_tolerance_deg"]),
            "right grasp pose",
        )

        if not confirmed("右手已到抓取位，输入 y 闭合右夹爪："):
            print("已取消；双臂保持当前位置，左手仍持有工件。")
            return 0
        right_grip = move_gripper(
            args.gripper_host, 2, args.right_close_position, args.gripper_force, "right"
        )
        print(f"右夹爪闭合完成：{right_grip}")
        if not confirmed("人工确认右手已稳定夹住工件，输入 y 切换碰撞体并释放左手："):
            print("已取消释放；当前双手共同夹持工件。")
            return 0

        attachment_ack_right = request_attachment_owner(
            args.attachment_state, "right", timeout_s=5.0
        )
        print(f"MoveIt碰撞体已切换到右TCP：{attachment_ack_right}")
        left_release = move_gripper(
            args.gripper_host, 1, args.left_open_position, args.gripper_force, "left"
        )
        print(f"左夹爪已释放：{left_release}")

        retreat_plan = planner.plan_cartesian_pose(
            "right",
            transform_to_object_pose(targets["T_base_right_tcp_pregrasp"]),
            planner="lin",
        )
        if retreat_plan is None:
            raise RuntimeError("Planner returned no right retreat plan")
        trajectory_summary(retreat_plan, 2)
        if not confirmed("右手已持有工件，输入 y 直线退回预抓取位："):
            print("已停止；右手持件停在交接位，左手保持张开。")
            return 0
        planner.execute_planned_result(retreat_plan)
        left_return_plan = planner.plan_joint_trajectory(
            "left", list(args.left_return_joints_deg), planner="rrtstar"
        )
        if left_return_plan is None:
            raise RuntimeError("Planner returned no left return-to-middle plan")
        left_return_summary = validate_joint_return_plan(
            left_return_plan,
            list(args.left_return_joints_deg),
            args.left_return_tolerance_deg,
        )
        print(json.dumps({"left_return_to_middle": left_return_summary}, indent=2, ensure_ascii=False))
        if not confirmed("右手已退至安全位，输入 y 让左臂回到中间初始位："):
            print("已停止；右手持件在预抓取位，左手保持张开。")
            return 0
        planner.execute_planned_result(left_return_plan)
        print("[完成] 左→右直接交接完成；右手持件在预抓取位，左臂已回到中间初始位。")
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
