#!/usr/bin/env python3
"""Build and collision-plan direct-handoff targets without executing motion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

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
            "tools.handoff_observation.preflight_handoff",
            *sys.argv[1:],
        ],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from loop_config import LoopConfig
from tools.planner_runtime import PlannerRuntime
from tools.handoff_observation.handoff_geometry import (
    compute_handoff_targets,
    object_pose_to_transform,
    planner_tf_target_source,
    transform_debug,
    transform_error,
    transform_to_object_pose,
    validate_transform,
)
from tools.handoff_observation.handoff_attachment import (
    request_attachment_owner,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "handoff_config.json"
DEFAULT_ATTACHMENT_STATE = HERE / "handoff_attachment_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 body_base 系 ICP 结果计算并只规划双臂固定交接目标"
    )
    parser.add_argument("--icp-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument(
        "--confirm-attached-collision-active",
        action="store_true",
        help="only after verifying /attached_collision_object in the running planner",
    )
    parser.add_argument(
        "--attachment-state", type=Path, default=DEFAULT_ATTACHMENT_STATE
    )
    parser.add_argument(
        "--report", type=Path, default=HERE / "last_handoff_preflight.json"
    )
    return parser.parse_args()


def load_icp_object_transform(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    pose = result.get("object_pose") or {}
    frame_id = pose.get("frame_id")
    if frame_id != "body_base":
        raise RuntimeError(
            f"ICP result frame is {frame_id!r}, expected 'body_base'. "
            "Recalibrate the head D435 at pitch=2°, yaw=26.5° and rerun pose "
            "estimation with --output-frame body_base. Camera-frame output must "
            "not be sent to the arm planner."
        )
    if result.get("T_output_object") is not None:
        matrix = validate_transform(result["T_output_object"], "T_base_object_current")
    else:
        matrix = object_pose_to_transform(pose, "object_pose")
    return matrix, result


def required_transform(config: dict[str, Any], key: str) -> np.ndarray:
    value = config.get(key)
    if value is None:
        raise RuntimeError(
            f"handoff config field {key!r} is not calibrated; fill it before preflight"
        )
    return validate_transform(value, key)


def trajectory_summary(
    response: dict[str, Any], expected_arm_id: int
) -> dict[str, Any]:
    data = response.get("data") or {}
    trajectory = data.get("trajectory") or []
    if not trajectory:
        raise RuntimeError(f"planner returned no trajectory: {response}")
    arm_id = int(data.get("arm_id", -1))
    if arm_id != expected_arm_id:
        raise RuntimeError(
            f"planner returned arm_id={arm_id}, expected {expected_arm_id}"
        )
    return {
        "arm_id": arm_id,
        "trajectory_points": len(trajectory),
        "sample_frequency_hz": float(data.get("sample_frequency", 10.0)),
    }


def validate_icp_for_motion(
    result: dict[str, Any], config: dict[str, Any]
) -> np.ndarray:
    transform_info = result.get("transform_info") or {}
    transform_status = transform_info.get("status")
    if transform_status not in {
        "calibrated_fixed_head_pose",
        "kinematically_propagated_head_extrinsic",
    }:
        raise RuntimeError(
            "ICP result is not backed by an accepted body_base head extrinsic"
        )
    # A propagated extrinsic intentionally observes from a different gimbal
    # angle than the fixed-pose calibration. Its body_base transform is already
    # rebuilt by the pose-estimation module, so do not reject that source for
    # not matching the calibration angle again here.
    if transform_status == "kinematically_propagated_head_extrinsic":
        calibrated_head: dict[str, Any] | None = None
    else:
        calibrated_head = transform_info.get("calibration_head_pose_deg") or {}
    if transform_status == "calibrated_fixed_head_pose":
        expected_head = config.get("expected_head_pose_deg") or {}
        actual_head = result.get("head_pose_command_deg") or {}
        for axis in ("pitch", "yaw"):
            if (
                axis not in expected_head
                or axis not in actual_head
                or calibrated_head is None
                or axis not in calibrated_head
            ):
                raise RuntimeError(f"ICP result is missing head {axis} metadata")
            if abs(float(expected_head[axis]) - float(actual_head[axis])) > 1.0:
                raise RuntimeError(
                    f"ICP head {axis}={actual_head[axis]}° does not match configured "
                    f"handoff observation angle {expected_head[axis]}°"
                )
            if abs(float(expected_head[axis]) - float(calibrated_head[axis])) > 1.0:
                raise RuntimeError(
                    f"D435 calibration head {axis}={calibrated_head[axis]}° does not "
                    f"match handoff observation angle {expected_head[axis]}°"
                )

    registration = result.get("registration") or {}
    try:
        fitness = float(registration["fitness"])
        rmse_m = float(registration["rmse_m"])
        valid_points = int((result.get("point_report") or {})["valid_point_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ICP result is missing registration quality fields") from exc
    if fitness < float(config["motion_quality_min_fitness"]):
        raise RuntimeError(f"ICP fitness {fitness:.4f} is below the motion threshold")
    if rmse_m > float(config["motion_quality_max_rmse_m"]):
        raise RuntimeError(f"ICP RMSE {rmse_m:.6f} m exceeds the motion threshold")
    if valid_points < int(config["motion_min_registration_points"]):
        raise RuntimeError(
            f"ICP valid points {valid_points} are below the motion threshold"
        )

    snapshot = result.get("left_tcp_snapshot") or {}
    if snapshot.get("frame_id") != "body_base":
        raise RuntimeError("ICP result has no synchronized body_base left TCP snapshot")
    if snapshot.get("tcp_frame") != str(config["left_tool_frame"]):
        raise RuntimeError(
            f"ICP TCP frame {snapshot.get('tcp_frame')!r} does not match "
            f"handoff frame {config['left_tool_frame']!r}"
        )
    return validate_transform(snapshot.get("T_base_left_tcp"), "captured left TCP")


def build_runtime(
    config: dict[str, Any],
    planner_host: str,
    *,
    execute_url: str = "disabled://preflight-never-executes",
) -> PlannerRuntime:
    return PlannerRuntime(LoopConfig(
        planner_host=planner_host,
        execute_url=execute_url,
        base_frame="body_base",
        left_group=str(config.get("left_group", "left_arm")),
        right_group=str(config.get("right_group", "right_arm")),
        left_tool_frame=str(config["left_tool_frame"]),
        right_tool_frame=str(config["right_tool_frame"]),
        max_velocity=float(config.get("max_velocity", 0.05)),
        max_acceleration=float(config.get("max_acceleration", 0.05)),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    ))


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("transform_convention") != "p_A = T_A_B @ p_B":
        raise RuntimeError("handoff config transform_convention is missing or invalid")
    if args.confirm_attached_collision_active and not config.get("attached_collision"):
        raise RuntimeError(
            "--confirm-attached-collision-active requires attached_collision in config"
        )
    T_base_object_current, icp_result = load_icp_object_transform(args.icp_result)
    T_base_left_tcp_captured = validate_icp_for_motion(icp_result, config)
    T_base_object_handoff = required_transform(config, "T_base_object_handoff")
    T_right_tcp_object = required_transform(config, "T_right_tcp_object")
    pregrasp_offset = config.get("right_pregrasp_offset_tcp_m")
    if pregrasp_offset is None:
        raise RuntimeError("right_pregrasp_offset_tcp_m has not been taught")
    attachment_ack = None
    if args.confirm_attached_collision_active:
        attachment_ack = request_attachment_owner(
            args.attachment_state, "left", timeout_s=5.0
        )

    planner = build_runtime(config, args.planner_host)
    try:
        left_frame = planner.config.left_tool_frame
        right_frame = planner.config.right_tool_frame
        planner.open(required_frames={"body_base", left_frame, right_frame})
        response = planner.lookup_transform(left_frame, "body_base")
        if response is None:
            raise RuntimeError("live Planner did not return the current left TCP transform")
        T_base_left_tcp_current = planner_tf_target_source(
            response, left_frame, "body_base"
        )
        moved_translation_m, moved_rotation_deg = transform_error(
            T_base_left_tcp_captured, T_base_left_tcp_current
        )
        if (
            moved_translation_m * 1000.0
            > float(config["current_vs_capture_tcp_max_translation_mm"])
            or moved_rotation_deg
            > float(config["current_vs_capture_tcp_max_rotation_deg"])
        ):
            raise RuntimeError(
                "left TCP moved after ICP capture; recapture before handoff preflight: "
                f"{moved_translation_m * 1000.0:.3f} mm/{moved_rotation_deg:.3f}°"
            )
        targets = compute_handoff_targets(
            T_base_object_current,
            T_base_left_tcp_captured,
            T_base_object_handoff,
            T_right_tcp_object,
            pregrasp_offset,
        )

        left_response = planner.plan_cartesian_pose(
            "left",
            transform_to_object_pose(targets["T_base_left_tcp_handoff"]),
            planner="rrtstar",
        )
        if left_response is None:
            raise RuntimeError("live Planner returned no result for left_handoff")
        plans = {
            "left_handoff": {
                "waypoints": ["left_handoff"],
                "planners": ["rrtstar"],
                **trajectory_summary(left_response, 1),
            },
        }
        left_target_translation_m, left_target_rotation_deg = transform_error(
            T_base_left_tcp_current, targets["T_base_left_tcp_handoff"]
        )
        left_is_at_handoff = (
            left_target_translation_m * 1000.0
            <= float(config["left_at_handoff_translation_tolerance_mm"])
            and left_target_rotation_deg
            <= float(config["left_at_handoff_rotation_tolerance_deg"])
        )
        if left_is_at_handoff and args.confirm_attached_collision_active:
            right_response = planner.plan_cartesian_sequence(
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
            if right_response is None:
                raise RuntimeError(
                    "live Planner returned no result for right_handoff_sequence"
                )
            plans["right_handoff_sequence"] = {
                "status": "PLANNED_WITH_LEFT_PHYSICALLY_AT_HANDOFF",
                "waypoints": ["right_pregrasp", "right_grasp"],
                "planners": ["rrtstar", "lin"],
                **trajectory_summary(right_response, 2),
            }
        else:
            plans["right_handoff_sequence"] = {
                "status": (
                    "DEFERRED_UNTIL_ATTACHED_COLLISION_IS_CONFIRMED"
                    if left_is_at_handoff
                    else "DEFERRED_UNTIL_LEFT_IS_PHYSICALLY_AT_HANDOFF"
                ),
                "left_target_error_mm": left_target_translation_m * 1000.0,
                "left_target_error_deg": left_target_rotation_deg,
            }

        registration = icp_result.get("registration") or {}
        report = {
            "result": (
                "LEFT_AND_RIGHT_PREFLIGHT_PLANNED_NOT_EXECUTED"
                if left_is_at_handoff and args.confirm_attached_collision_active
                else "LEFT_PREFLIGHT_PLANNED_RIGHT_DEFERRED_NOT_EXECUTED"
            ),
            "icp_result": str(args.icp_result),
            "registration": {
                "fitness": registration.get("fitness"),
                "rmse_m": registration.get("rmse_m"),
                "quality_gate_passed": registration.get("quality_gate_passed"),
            },
            "frames": {
                "base": "body_base",
                "left_tcp": left_frame,
                "right_tcp": right_frame,
            },
            "transforms": {
                "T_base_object_current": transform_debug(T_base_object_current),
                "T_base_left_tcp_current": transform_debug(T_base_left_tcp_current),
                "T_base_left_tcp_captured": transform_debug(T_base_left_tcp_captured),
                "T_left_tcp_object": transform_debug(targets["T_left_tcp_object"]),
                "T_base_object_handoff": transform_debug(T_base_object_handoff),
                "T_base_left_tcp_handoff": transform_debug(
                    targets["T_base_left_tcp_handoff"]
                ),
                "T_right_tcp_object": transform_debug(T_right_tcp_object),
                "T_base_right_tcp_pregrasp": transform_debug(
                    targets["T_base_right_tcp_pregrasp"]
                ),
                "T_base_right_tcp_grasp": transform_debug(
                    targets["T_base_right_tcp_grasp"]
                ),
            },
            "plans": plans,
            "executed": False,
            "attached_collision": {
                "configured": bool(config.get("attached_collision")),
                "runtime_presence_confirmed_by_handshake": bool(
                    args.confirm_attached_collision_active
                ),
                "publisher_ack": attachment_ack,
                "note": (
                    "start MoveIt with docker-compose.handoff.yml and verify "
                    "/attached_collision_object before treating plans as collision-aware"
                ),
            },
            "collision_model_limitations": [
                "this client cannot query MoveIt's PlanningScene to prove the attached object was received",
                "left and right are staged plans and must be replanned after each physical phase",
                "intentional gripper proximity at handoff needs staged real-world validation",
            ],
        }
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[完成] 仅规划预检报告: {args.report}")
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
