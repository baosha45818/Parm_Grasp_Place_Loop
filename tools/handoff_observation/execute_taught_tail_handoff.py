#!/usr/bin/env python3
"""Use a taught right-arm joint pose to take the object at the fixed handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


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
            "tools.handoff_observation.execute_taught_tail_handoff",
            *sys.argv[1:],
        ],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tools.handoff_observation.check_plan import (
    MODEL_JOINT_LIMITS_DEG,
    START_JOINTS_DEG,
    max_joint_error_deg,
    validate_joint_vector,
)
from tools.handoff_observation.execute_handoff import (
    move_gripper,
    read_gripper_state,
    read_tcp,
    require_pose_close,
    validate_joint_return_plan,
)
from tools.handoff_observation.handoff_attachment import (
    request_attachment_owner,
)
from tools.handoff_observation.handoff_geometry import (
    transform_debug,
    transform_error,
    transform_to_object_pose,
    validate_transform,
)
from tools.handoff_observation.move_object_center import handoff_axis_errors
from tools.handoff_observation.preflight_handoff import (
    build_runtime,
    load_icp_object_transform,
    trajectory_summary,
    validate_icp_for_motion,
)
from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "handoff_config.json"
DEFAULT_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"
_WORKPIECE_1_HANDOFF = load_workpiece_handoff_profile("1")
DEFAULT_RIGHT_GRASP_JOINTS_DEG = list(
    _WORKPIECE_1_HANDOFF["right_handoff"]["grasp_joints_deg"]
)
DEFAULT_RIGHT_PLACE_READY_JOINTS_DEG = list(
    _WORKPIECE_1_HANDOFF["right_handoff"]["place_ready_joints_deg"]
)
REQUIRE_CONFIRMATION = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", "--icp-result", dest="result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--execute-url", default=DEFAULT_EXECUTE_URL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=4.0,
        help="相对handoff_config速度/加速度倍率，默认4.0",
    )
    parser.add_argument(
        "--require-confirmation",
        action="store_true",
        help="恢复每一步人工输入y确认；默认自动确认",
    )
    parser.add_argument(
        "--resume-after-left-release",
        action="store_true",
        help="恢复上次中断：假设右手已夹稳且左夹爪已打开，直接执行撤离流程",
    )
    parser.add_argument(
        "--resume-right-at-place-ready",
        action="store_true",
        help="恢复上次中断：右手已持件到放置准备位，仅让左手回中间",
    )
    parser.add_argument(
        "--right-pregrasp-ready-joints-deg",
        type=float,
        nargs=7,
        help="已由并发阶段到达的右臂安全预抓取关节角；跳过重复RRTstar，只执行最终抓取",
    )
    parser.add_argument(
        "--confirm-resume-from-current-held-pose",
        action="store_true",
        help="与--resume-after-left-release联用：确认从当前非抓取位持件姿态撤离",
    )
    parser.add_argument(
        "--right-grasp-joints-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=DEFAULT_RIGHT_GRASP_JOINTS_DEG,
        help="手工拖动示教的右臂尾部抓取关节角，单位deg",
    )
    parser.add_argument(
        "--final-grasp-motion",
        choices=("lin", "taught-joints"),
        default="lin",
        help=(
            "最终交接运动；lin按实时工件位姿做笛卡尔修正，"
            "taught-joints直接RRTStar到示教关节位"
        ),
    )
    parser.add_argument(
        "--right-grasp-tolerance-deg",
        type=float,
        default=0.5,
        help="最终示教关节位的最大单关节到位误差，默认0.5deg",
    )
    parser.add_argument(
        "--expected-center-m",
        type=float,
        nargs=3,
        default=list(_WORKPIECE_1_HANDOFF["handoff"]["center_body_m"]),
    )
    parser.add_argument(
        "--taught-object-tilt-deg",
        type=float,
        default=0.0,
        help=(
            "建立示教工件->右TCP关系时的工件绕局部X倾角；"
            "默认0°，即固定朝向高位交接；旧5°流程需显式传5°"
        ),
    )
    parser.add_argument(
        "--center-tolerance-mm",
        type=float,
        default=25.0,
        help="允许左臂冗余交接候选相对名义中心的偏移，默认25mm",
    )
    parser.add_argument("--axis-tolerance-deg", type=float, default=10.0)
    parser.add_argument(
        "--object-orientation-tolerance-deg",
        type=float,
        default=7.5,
        help="允许相对示教工件姿态进行TCP刚性补偿的最大角度，默认7.5°",
    )
    parser.add_argument(
        "--diagnostic-use-nominal-handoff-pose",
        action="store_true",
        help="仅用于只规划：不使用实时左臂重建值，假设工件位于示教交接姿态",
    )
    parser.add_argument("--right-close-position", type=float, default=0.0)
    parser.add_argument("--right-open-position", type=float, default=100.0)
    parser.add_argument("--left-open-position", type=float, default=100.0)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    parser.add_argument(
        "--pregrasp-distance-m",
        type=float,
        default=0.05,
        help="预抓取点沿--pregrasp-axis指定负轴后退的距离，默认0.05m",
    )
    parser.add_argument(
        "--pregrasp-axis",
        choices=("tool-minus-z", "object-minus-x"),
        default="tool-minus-z",
        help=(
            "预抓取偏移轴；默认保持夹爪局部-Z，"
            "object-minus-x表示沿本轮实时工件坐标系-X"
        ),
    )
    parser.add_argument("--arrival-translation-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--arrival-rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--right-lift-distance-m",
        type=float,
        default=0.05,
        help="左夹爪打开后右手沿body_base +Z上升距离，默认0.03m",
    )
    parser.add_argument(
        "--right-retreat-distance-m",
        type=float,
        default=0.07,
        help="右手上升后沿body_base -Y后退距离，默认0.07m",
    )
    parser.add_argument(
        "--right-place-ready-joints-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=DEFAULT_RIGHT_PLACE_READY_JOINTS_DEG,
    )
    parser.add_argument("--right-place-ready-tolerance-deg", type=float, default=0.5)
    parser.add_argument(
        "--right-place-ready-file",
        type=Path,
        help="可选：右臂实际到达 right_place_ready 后原子写入此标记文件",
    )
    parser.add_argument(
        "--left-return-joints-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=START_JOINTS_DEG,
    )
    parser.add_argument("--left-return-tolerance-deg", type=float, default=0.5)
    parser.add_argument(
        "--attachment-state",
        type=Path,
        default=HERE / "handoff_attachment_state.json",
    )
    parser.add_argument("--switch-attachment", action="store_true")
    return parser.parse_args()


def confirmed(prompt: str) -> bool:
    if not REQUIRE_CONFIRMATION:
        print(f"{prompt}[自动确认]")
        return True
    return input(prompt).strip().lower() == "y"


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.speed_multiplier) or args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    if args.resume_after_left_release and args.resume_right_at_place_ready:
        raise ValueError("two resume modes cannot be enabled together")
    if (
        args.confirm_resume_from_current_held_pose
        and not args.resume_after_left_release
    ):
        raise ValueError(
            "--confirm-resume-from-current-held-pose requires "
            "--resume-after-left-release"
        )
    if args.execute and args.diagnostic_use_nominal_handoff_pose:
        raise ValueError("diagnostic nominal pose cannot be used with --execute")
    validate_joint_vector(args.right_grasp_joints_deg, "right grasp joints")
    validate_joint_vector(args.right_place_ready_joints_deg, "right place ready joints")
    validate_joint_vector(args.left_return_joints_deg, "left return joints")
    if (
        args.center_tolerance_mm <= 0.0
        or args.axis_tolerance_deg <= 0.0
        or args.object_orientation_tolerance_deg <= 0.0
        or args.right_grasp_tolerance_deg <= 0.0
        or args.arrival_translation_tolerance_mm <= 0.0
        or args.arrival_rotation_tolerance_deg <= 0.0
        or args.pregrasp_distance_m <= 0.0
        or args.right_lift_distance_m <= 0.0
        or args.right_retreat_distance_m <= 0.0
    ):
        raise ValueError("handoff pose tolerances must be positive")
    if not (
        0.0 <= args.right_close_position <= 100.0
        and 0.0 <= args.right_open_position <= 100.0
        and 0.0 <= args.left_open_position <= 100.0
        and 20.0 <= args.gripper_force <= 100.0
    ):
        raise ValueError("gripper positions must be 0..100 and force must be 20..100")


def origin_transform(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def pregrasp_offset_body(
    T_base_right_tcp_grasp: np.ndarray,
    T_base_object: np.ndarray,
    distance_m: float,
    axis: str,
) -> tuple[np.ndarray, dict[str, str | float]]:
    """Return the requested pregrasp offset expressed in body_base."""
    grasp = validate_transform(T_base_right_tcp_grasp, "pregrasp grasp TCP")
    object_pose = validate_transform(T_base_object, "pregrasp object pose")
    distance = float(distance_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("pregrasp distance must be positive")
    if axis == "tool-minus-z":
        direction_body = -grasp[:3, 2]
        metadata: dict[str, str | float] = {
            "frame": "right_hand_transfer_end",
            "axis": "-Z",
            "distance_m": distance,
        }
    elif axis == "object-minus-x":
        direction_body = -object_pose[:3, 0]
        metadata = {
            "frame": "live workpiece",
            "axis": "-X",
            "distance_m": distance,
        }
    else:
        raise ValueError(f"unsupported pregrasp axis: {axis!r}")
    return direction_body * distance, metadata


def taught_handoff_object_transform(
    center_body_m: list[float], tilt_deg: float
) -> np.ndarray:
    """Build the nominal taught workpiece pose used to calibrate right TCP."""
    center = np.asarray(center_body_m, dtype=float)
    tilt = float(tilt_deg)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("expected handoff center must contain three finite values")
    if not np.isfinite(tilt) or abs(tilt) > 20.0:
        raise ValueError("taught object tilt must be finite and within +/-20deg")
    tilt_rad = np.deg2rad(tilt)
    transform = np.eye(4)
    transform[:3, 3] = center
    transform[:3, :3] = np.asarray([
        [0.0, -np.cos(tilt_rad), np.sin(tilt_rad)],
        [1.0, 0.0, 0.0],
        [0.0, np.sin(tilt_rad), np.cos(tilt_rad)],
    ])
    return validate_transform(transform, "T_base_object_taught_handoff")


def taught_right_tcp_from_joints(joints_deg: list[float]) -> np.ndarray:
    """FK from the deployed URDF for body_base -> right_hand_transfer_end."""
    joint_origins = (
        ((0.0002, 0.00006, 0.1565), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0)),
        ((0.0, 0.274, 0.0), (-1.5708, 0.0, 0.0)),
        ((-0.025, 0.0, 0.0), (1.5708, 0.0, 0.0)),
        ((0.025, 0.265, 0.0), (-1.5708, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.5708, -1.5708, 0.0)),
        ((0.071, 0.0, 0.0), (1.5708, 0.0, 0.0)),
    )
    transform = origin_transform(
        (-6.5522e-05, -0.10253, -0.0014374),
        (0.0, -1.3614, 1.5708),
    )
    for value_deg, (xyz, rpy) in zip(joints_deg, joint_origins):
        transform = transform @ origin_transform(xyz, rpy)
        rotation_z = np.eye(4)
        rotation_z[:3, :3] = Rotation.from_euler(
            "z", value_deg, degrees=True
        ).as_matrix()
        transform = transform @ rotation_z
    transform = transform @ origin_transform((0.084, 0.0, 0.0))
    transform = transform @ origin_transform(
        (0.0495, 0.0, 0.0002853),
        (-1.5708, 0.0, -1.5708),
    )
    return validate_transform(transform, "T_base_right_tcp_taught_grasp")


def solve_nearby_right_joints(
    T_base_right_tcp_target: np.ndarray, seed_joints_deg: list[float]
) -> tuple[list[float], dict[str, float]]:
    """Solve a nearby 7-DOF IK branch using the locally verified URDF FK."""
    target = validate_transform(T_base_right_tcp_target, "right IK target")
    seed_rad = np.deg2rad(np.asarray(seed_joints_deg, dtype=float))
    lower_rad = np.deg2rad(
        np.asarray([-224.8, -99.8, -224.8, -99.8, -224.8, -89.8, -89.8])
    )
    upper_rad = np.deg2rad(
        np.asarray([224.8, 99.8, 224.8, 149.8, 224.8, 89.8, 89.8])
    )

    def residual(joints_rad: np.ndarray) -> np.ndarray:
        actual = taught_right_tcp_from_joints(np.rad2deg(joints_rad).tolist())
        translation = (actual[:3, 3] - target[:3, 3]) * 20.0
        rotation = Rotation.from_matrix(
            actual[:3, :3].T @ target[:3, :3]
        ).as_rotvec()
        regularization = (joints_rad - seed_rad) * 1e-3
        return np.r_[translation, rotation, regularization]

    solution = least_squares(
        residual,
        seed_rad,
        bounds=(lower_rad, upper_rad),
        max_nfev=1000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    joints_deg = np.rad2deg(solution.x).tolist()
    achieved = taught_right_tcp_from_joints(joints_deg)
    translation_m, rotation_deg = transform_error(target, achieved)
    if (
        not solution.success
        or translation_m * 1000.0 > 1.0
        or rotation_deg > 0.2
    ):
        raise RuntimeError(
            "无法在示教关节支路附近求得5cm预抓取IK："
            f"success={solution.success}, "
            f"位置误差={translation_m * 1000.0:.3f} mm, "
            f"姿态误差={rotation_deg:.3f}°"
        )
    return joints_deg, {
        "translation_error_mm": translation_m * 1000.0,
        "rotation_error_deg": rotation_deg,
        "max_joint_delta_from_taught_deg": max_joint_error_deg(
            joints_deg, seed_joints_deg
        ),
    }


def validate_arm_joint_plan(
    response: dict,
    target_deg: list[float],
    *,
    arm_id: int,
    label: str,
    tolerance_deg: float,
) -> dict:
    data = response.get("data") or {}
    if int(data.get("arm_id", -1)) != arm_id:
        raise RuntimeError(f"{label} has unexpected arm_id: {data.get('arm_id')}")
    trajectory = data.get("trajectory") or []
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError(f"{label} has no trajectory")
    for index, point in enumerate(trajectory):
        validate_joint_vector(point, f"{label}[{index}]")
    endpoint_error = max_joint_error_deg(trajectory[-1], target_deg)
    if endpoint_error > tolerance_deg:
        raise RuntimeError(
            f"{label} endpoint error {endpoint_error:.3f}° exceeds "
            f"{tolerance_deg:.3f}°"
        )
    return {
        "arm_id": arm_id,
        "trajectory_points": len(trajectory),
        "sample_frequency_hz": float(data.get("sample_frequency", 10.0)),
        "target_joints_deg": target_deg,
        "endpoint_error_deg": endpoint_error,
    }


def validate_right_cartesian_plan(
    response: dict,
    seed_joints_deg: list[float],
    target_tcp: np.ndarray,
    translation_tolerance_mm: float,
    rotation_tolerance_deg: float,
) -> dict:
    """Validate a right-arm Cartesian plan before it can be executed.

    Cartesian plans are produced by the live Planner, while this module's
    local FK is only used to recover taught relations.  Do *not* reject a
    valid live plan by comparing its final joints with that local FK.  The
    executor instead verifies the actual Planner TF after execution through
    :func:`wait_for_pose_close`.  Before execution we can, and must, verify
    the response shape, arm identity, finite seven-axis points, and that the
    plan starts near the expected current right-arm seed.

    ``target_tcp`` and its tolerances are validated here as part of the call
    contract; they are consumed by the post-execution real-TF gate.
    """
    validate_joint_vector(seed_joints_deg, "right Cartesian seed joints")
    validate_transform(target_tcp, "right Cartesian target TCP")
    if translation_tolerance_mm <= 0.0 or rotation_tolerance_deg <= 0.0:
        raise ValueError("right Cartesian arrival tolerances must be positive")

    data = response.get("data") or {}
    if int(data.get("arm_id", -1)) != 2:
        raise RuntimeError(
            "right Cartesian plan has unexpected arm_id: "
            f"{data.get('arm_id')}"
        )
    trajectory = data.get("trajectory") or []
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise RuntimeError("right Cartesian plan has no usable trajectory")
    for index, point in enumerate(trajectory):
        validate_joint_vector(point, f"right_cartesian[{index}]")

    start_error_deg = max_joint_error_deg(trajectory[0], seed_joints_deg)
    # The real start-state gate below is stricter (2 deg) immediately before
    # execution.  This wider offline guard allows a Planner to normalize an
    # equivalent wrapped joint representation without confusing it with an
    # unrelated trajectory.
    if start_error_deg > 10.0:
        raise RuntimeError(
            "right Cartesian plan starts too far from its expected seed: "
            f"{start_error_deg:.3f}deg > 10.000deg"
        )
    return {
        "arm_id": 2,
        "trajectory_points": len(trajectory),
        "sample_frequency_hz": float(data.get("sample_frequency", 10.0)),
        "start_error_deg": start_error_deg,
        "target_translation_tolerance_mm": float(translation_tolerance_mm),
        "target_rotation_tolerance_deg": float(rotation_tolerance_deg),
    }


def validate_trajectory_start_matches_current_state(
    response: dict,
    *,
    arm_host: str,
    arm_id: int,
    label: str,
    tolerance_deg: float,
) -> dict:
    """仅确认规划轨迹首点与机器人当前关节反馈可以安全衔接。"""
    from parm_sdk import Arm

    data = response.get("data") or {}
    trajectory = data.get("trajectory") or []
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError(f"{label} has no trajectory")
    for index, point in enumerate(trajectory):
        validate_joint_vector(point, f"{label}[{index}]")

    arm_key = "left" if arm_id == 1 else "right"
    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
    current = ((state.get("state") or {}).get(arm_key) or {}).get("position") or []
    validate_joint_vector(current, f"{label} current {arm_key} joints")
    start_error_deg = max_joint_error_deg(trajectory[0], current)
    if start_error_deg > tolerance_deg:
        raise RuntimeError(
            f"{label} trajectory start error {start_error_deg:.3f}deg exceeds "
            f"{tolerance_deg:.3f}deg"
        )
    return {
        "current_joints_deg": [float(value) for value in current],
        "trajectory_first_joints_deg": [float(value) for value in trajectory[0]],
        "start_error_deg": start_error_deg,
        "tolerance_deg": tolerance_deg,
    }


def validate_nonzero_trajectory(response: dict, label: str) -> dict:
    data = response.get("data") or {}
    trajectory = data.get("trajectory") or []
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise RuntimeError(f"{label} has fewer than two trajectory points")
    for index, point in enumerate(trajectory):
        validate_joint_vector(point, f"{label}[{index}]")
    endpoint_delta_deg = max_joint_error_deg(trajectory[-1], trajectory[0])
    if endpoint_delta_deg < 0.01:
        raise RuntimeError(
            f"{label} is a no-op trajectory: endpoint delta "
            f"{endpoint_delta_deg:.6f}°"
        )
    return {
        "first_joints_deg": trajectory[0],
        "final_joints_deg": trajectory[-1],
        "endpoint_delta_deg": endpoint_delta_deg,
    }


def validate_trajectory_joint_limit_margin(
    response: dict, label: str, margin_deg: float,
) -> dict:
    """Hard gate every returned waypoint against request-local safe bounds."""
    from tools.planner_runtime import ARM_JOINT_POSITION_LIMITS_DEG

    trajectory = ((response.get("data") or {}).get("trajectory") or [])
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError(f"{label} has no trajectory for joint-margin validation")
    minimum_margin = float("inf")
    minimum_location = None
    for point_index, point in enumerate(trajectory):
        validate_joint_vector(point, f"{label}[{point_index}]")
        for joint_index, (value, limits) in enumerate(
            zip(point, ARM_JOINT_POSITION_LIMITS_DEG), start=1
        ):
            lower, upper = limits
            actual_margin = min(float(value) - lower, upper - float(value))
            if actual_margin < minimum_margin:
                minimum_margin = actual_margin
                minimum_location = {
                    "trajectory_point": point_index,
                    "joint": f"right_arm_joint{joint_index}",
                    "value_deg": float(value),
                    "lower_deg": lower,
                    "upper_deg": upper,
                }
            if actual_margin + 1e-9 < margin_deg:
                raise RuntimeError(
                    f"{label} violates {margin_deg:.3f}deg joint-limit margin at "
                    f"point {point_index}, J{joint_index}={float(value):.6f}deg; "
                    f"bounds=[{lower:.6f}, {upper:.6f}]deg, "
                    f"actual_margin={actual_margin:.6f}deg"
                )
    return {
        "required_margin_deg": float(margin_deg),
        "minimum_actual_margin_deg": minimum_margin,
        "minimum_margin_location": minimum_location,
    }


def save_right_joint_limit_diagnostic(
    args: argparse.Namespace,
    retreat_motion: dict,
    right_ready_target: list[float],
    *,
    planning_status: str = "not_started",
    planning_error: BaseException | None = None,
) -> tuple[Path, dict]:
    """Persist the post-retreat start state before Module 9 planning."""
    from parm_sdk import Arm

    with Arm(host=args.arm_host) as arm:
        state = arm.get_state() or {}
    current = ((state.get("state") or {}).get("right") or {}).get("position") or []
    validate_joint_vector(current, "post-retreat current right joints")

    joints = []
    for index, (raw, limits) in enumerate(
        zip(current, MODEL_JOINT_LIMITS_DEG), start=1
    ):
        raw = float(raw)
        lower, upper = (float(limits[0]), float(limits[1]))
        equivalent = [raw + 360.0 * turns for turns in range(-2, 3)]
        in_bounds_equivalents = [
            value for value in equivalent if lower <= value <= upper
        ]
        normalized = (
            min(in_bounds_equivalents, key=lambda value: abs(value - raw))
            if in_bounds_equivalents
            else None
        )
        joints.append({
            "joint": f"right_arm_joint{index}",
            "actual_deg": raw,
            "planner_lower_deg": lower,
            "planner_upper_deg": upper,
            "raw_in_planner_bounds": lower <= raw <= upper,
            "underflow_deg": max(0.0, lower - raw),
            "overflow_deg": max(0.0, raw - upper),
            "equivalent_in_bounds_deg": normalized,
            "would_require_angle_wrapping": (
                normalized is not None and abs(normalized - raw) > 1e-9
            ),
        })

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "after_right_body_base_minus_y_retreat_before_module9_rrtstar",
        "source_result": str(args.result.resolve()),
        "arm_host": args.arm_host,
        "actual_right_joints_deg": [float(value) for value in current],
        "retreat_planned_first_joints_deg": retreat_motion.get("first_joints_deg"),
        "retreat_planned_final_joints_deg": retreat_motion.get("final_joints_deg"),
        "module9_place_ready_target_joints_deg": [
            float(value) for value in right_ready_target
        ],
        "planner_position_limits_source": (
            "welding_config right_arm.urdf.xacro; mirrored by "
            "tools.handoff_observation.check_plan.MODEL_JOINT_LIMITS_DEG"
        ),
        "joint_limit_checks": joints,
        "raw_start_state_within_planner_bounds": all(
            item["raw_in_planner_bounds"] for item in joints
        ),
        "planning_status": planning_status,
        "planning_error": (
            f"{type(planning_error).__name__}: {planning_error}"
            if planning_error is not None else None
        ),
    }
    path = args.result.resolve().parent / "right_post_handoff_joint_limit_diagnostic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path, payload


def densify_short_trajectory(response: dict, minimum_points: int = 21) -> dict:
    """Densify short joint paths that the deployed TOPPRA returns at zero speed."""
    data = response.get("data") or {}
    trajectory = data.get("trajectory") or []
    if len(trajectory) >= minimum_points:
        return response
    source = np.asarray(trajectory, dtype=float)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] != 7:
        raise RuntimeError("cannot densify malformed trajectory")
    source_t = np.linspace(0.0, 1.0, source.shape[0])
    target_t = np.linspace(0.0, 1.0, minimum_points)
    dense = np.column_stack([
        np.interp(target_t, source_t, source[:, joint_index])
        for joint_index in range(7)
    ])
    updated = dict(response)
    updated_data = dict(data)
    updated_data["trajectory"] = dense.tolist()
    updated["data"] = updated_data
    return updated


def wait_for_pose_close(
    planner,
    frame: str,
    target: np.ndarray,
    translation_tolerance_mm: float,
    rotation_tolerance_deg: float,
    label: str,
    *,
    timeout_s: float = 30.0,
) -> dict:
    """Poll live TF because the trajectory execute HTTP endpoint is asynchronous."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        live = read_tcp(planner, frame)
        try:
            result = require_pose_close(
                live,
                target,
                translation_tolerance_mm,
                rotation_tolerance_deg,
                label,
            )
            return result
        except RuntimeError as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(
        f"{label} did not arrive within {timeout_s:.1f}s; last={last_error}"
    )


def wait_for_joint_target(
    arm_host: str,
    arm_id: int,
    target_deg: list[float],
    tolerance_deg: float,
    label: str,
    *,
    timeout_s: float,
) -> None:
    """Poll direct arm feedback because planner trajectory execution is asynchronous."""
    from parm_sdk import Arm

    arm_key = "left" if arm_id == 1 else "right"
    target = np.asarray(target_deg, dtype=float)
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = np.asarray(
                ((state.get("state") or {}).get(arm_key) or {}).get("position") or [],
                dtype=float,
            )
            if actual.shape == (7,):
                # State feedback may keep equivalent unwrapped revolutions.
                delta = (actual - target + 180.0) % 360.0 - 180.0
                last_error_deg = float(np.max(np.abs(delta)))
                if last_error_deg <= tolerance_deg:
                    print(
                        f"{label} reached: max joint error={last_error_deg:.3f}deg",
                        flush=True,
                    )
                    return
            time.sleep(0.05)
    raise RuntimeError(
        f"{label} did not arrive within {timeout_s:.1f}s; "
        f"last max joint error={last_error_deg:.3f}deg"
    )


def finish_after_left_release(planner, args: argparse.Namespace, config: dict) -> int:
    """Lift the right-held object, move right away, then return the left arm."""
    right_frame = config["right_tool_frame"]
    T_base_right_tcp_live = read_tcp(planner, right_frame)
    T_base_right_tcp_lift = T_base_right_tcp_live.copy()
    T_base_right_tcp_lift[2, 3] += float(args.right_lift_distance_m)
    right_lift_plan = planner.plan_cartesian_pose(
        "right",
        transform_to_object_pose(T_base_right_tcp_lift),
        planner="lin",
    )
    if right_lift_plan is None:
        raise RuntimeError("Planner returned no right post-handoff lift plan")
    lift_motion = validate_nonzero_trajectory(
        right_lift_plan, "right_lift_trajectory"
    )
    original_lift_points = len(
        (right_lift_plan.get("data") or {}).get("trajectory") or []
    )
    right_lift_plan = densify_short_trajectory(right_lift_plan)
    executed_lift_points = len(
        (right_lift_plan.get("data") or {}).get("trajectory") or []
    )
    print(json.dumps({
        "right_lift": {
            "distance_m": float(args.right_lift_distance_m),
            "direction": "body_base +Z",
            "target": transform_debug(T_base_right_tcp_lift),
            "plan": trajectory_summary(right_lift_plan, 2),
            "joint_motion": lift_motion,
            "short_trajectory_workaround": {
                "original_points": original_lift_points,
                "executed_points": executed_lift_points,
            },
        }
    }, indent=2, ensure_ascii=False))
    if not args.execute:
        print("[完成] 只规划右手上升；未执行运动。后续步骤须在上升实际到位后规划。")
        return 0
    if not confirmed(
        f"确认右手已夹稳、左夹爪已完全打开，且上方"
        f"{args.right_lift_distance_m * 100.0:.1f}cm无障碍，"
        "输入 y 让右手直线上升："
    ):
        print("已停止；右手持件保持当前位置。")
        return 0
    planner.execute_planned_result(right_lift_plan)
    lift_error = wait_for_pose_close(
        planner,
        right_frame,
        T_base_right_tcp_lift,
        args.arrival_translation_tolerance_mm,
        args.arrival_rotation_tolerance_deg,
        "right post-handoff lift pose",
        timeout_s=15.0,
    )
    print(f"右手已上升 {args.right_lift_distance_m * 100.0:.1f} cm：{lift_error}")

    T_base_right_tcp_after_lift = read_tcp(planner, right_frame)
    T_base_right_tcp_retreat = T_base_right_tcp_after_lift.copy()
    T_base_right_tcp_retreat[1, 3] -= float(args.right_retreat_distance_m)
    right_retreat_plan = planner.plan_cartesian_pose(
        "right",
        transform_to_object_pose(T_base_right_tcp_retreat),
        planner="lin",
        joint_limit_margin_deg=1.0,
    )
    if right_retreat_plan is None:
        raise RuntimeError("Planner returned no right post-handoff retreat plan")
    retreat_joint_margin = validate_trajectory_joint_limit_margin(
        right_retreat_plan, "right_retreat_trajectory", 1.0
    )
    retreat_motion = validate_nonzero_trajectory(
        right_retreat_plan, "right_retreat_trajectory"
    )
    original_retreat_points = len(
        (right_retreat_plan.get("data") or {}).get("trajectory") or []
    )
    right_retreat_plan = densify_short_trajectory(right_retreat_plan)
    executed_retreat_points = len(
        (right_retreat_plan.get("data") or {}).get("trajectory") or []
    )
    print(json.dumps({
        "right_retreat": {
            "distance_m": float(args.right_retreat_distance_m),
            "direction": "body_base -Y",
            "target": transform_debug(T_base_right_tcp_retreat),
            "plan": trajectory_summary(right_retreat_plan, 2),
            "joint_motion": retreat_motion,
            "joint_limit_margin": retreat_joint_margin,
            "short_trajectory_workaround": {
                "original_points": original_retreat_points,
                "executed_points": executed_retreat_points,
            },
        }
    }, indent=2, ensure_ascii=False))
    if not confirmed(
        f"确认右手已上升、body_base -Y后方"
        f"{args.right_retreat_distance_m * 100.0:.1f}cm无障碍，"
        "输入 y 让右手保持姿态直线后退："
    ):
        print("已停止；右手持件停在上升后的安全位。")
        return 0
    planner.execute_planned_result(right_retreat_plan)
    retreat_error = wait_for_pose_close(
        planner,
        right_frame,
        T_base_right_tcp_retreat,
        args.arrival_translation_tolerance_mm,
        args.arrival_rotation_tolerance_deg,
        "right post-handoff retreat pose",
        timeout_s=30.0,
    )
    print(
        f"右手已沿body_base -Y后退 "
        f"{args.right_retreat_distance_m * 100.0:.1f} cm：{retreat_error}"
    )

    right_ready_target = list(args.right_place_ready_joints_deg)
    diagnostic_path, diagnostic = save_right_joint_limit_diagnostic(
        args, retreat_motion, right_ready_target
    )
    print(json.dumps({
        "right_post_retreat_joint_limit_diagnostic": {
            "path": str(diagnostic_path),
            "raw_start_state_within_planner_bounds": diagnostic[
                "raw_start_state_within_planner_bounds"
            ],
            "joint_limit_checks": diagnostic["joint_limit_checks"],
        }
    }, indent=2, ensure_ascii=False), flush=True)
    try:
        right_ready_plan = planner.plan_joint_trajectory(
            "right", right_ready_target, planner="rrtstar"
        )
    except BaseException as exc:
        save_right_joint_limit_diagnostic(
            args,
            retreat_motion,
            right_ready_target,
            planning_status="failure",
            planning_error=exc,
        )
        raise
    if right_ready_plan is None:
        error = RuntimeError("Planner returned no right place-ready plan")
        save_right_joint_limit_diagnostic(
            args,
            retreat_motion,
            right_ready_target,
            planning_status="failure",
            planning_error=error,
        )
        raise error
    save_right_joint_limit_diagnostic(
        args, retreat_motion, right_ready_target, planning_status="success"
    )
    right_ready_summary = validate_arm_joint_plan(
        right_ready_plan,
        right_ready_target,
        arm_id=2,
        label="right_place_ready_trajectory",
        tolerance_deg=args.right_place_ready_tolerance_deg,
    )
    print(json.dumps(
        {"right_move_to_place_ready": right_ready_summary},
        indent=2,
        ensure_ascii=False,
    ))
    if not confirmed(
        "确认右手持件到放置准备位的RRTStar路径安全，输入 y 执行："
    ):
        print("已停止；右手持件停在上升后的安全位。")
        return 0
    planner.execute_planned_result(right_ready_plan)
    right_ready_tcp = taught_right_tcp_from_joints(right_ready_target)
    right_ready_error = wait_for_pose_close(
        planner,
        right_frame,
        right_ready_tcp,
        args.arrival_translation_tolerance_mm,
        args.arrival_rotation_tolerance_deg,
        "right place-ready pose",
        timeout_s=60.0,
    )
    print(f"右手已到放置准备位：{right_ready_error}")
    if args.right_place_ready_file is not None:
        args.right_place_ready_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_marker = args.right_place_ready_file.with_suffix(
            args.right_place_ready_file.suffix + ".tmp"
        )
        temporary_marker.write_text(
            json.dumps({
                "event": "right_place_ready",
                "right_joints_deg": right_ready_target,
                "timestamp": time.time(),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_marker, args.right_place_ready_file)
        print(
            f"[handoff] 已发布右臂 right_place_ready 标记："
            f"{args.right_place_ready_file}",
            flush=True,
        )

    return return_left_to_middle(planner, args)


def return_left_to_middle(planner, args: argparse.Namespace) -> int:
    left_target = list(args.left_return_joints_deg)
    left_return_plan = planner.plan_joint_trajectory(
        "left", left_target, planner="rrtstar"
    )
    if left_return_plan is None:
        raise RuntimeError("Planner returned no left return-to-middle plan")
    left_summary = validate_joint_return_plan(
        left_return_plan, left_target, args.left_return_tolerance_deg
    )
    print(json.dumps(
        {"left_return_to_middle": left_summary}, indent=2, ensure_ascii=False
    ))
    if not args.execute:
        print("[完成] 左臂回中间初始位规划通过；未执行运动。")
        return 0
    if not confirmed("右手已移出交接区域，输入 y 让左臂回中间初始位："):
        print("已停止；右手位于放置准备位，左手保持张开。")
        return 0
    planner.execute_planned_result(left_return_plan)
    wait_for_joint_target(
        args.arm_host,
        1,
        left_target,
        args.left_return_tolerance_deg,
        "left return-to-middle",
        timeout_s=90.0,
    )
    print(
        "[完成] 右手已上升、后退并到放置准备位，"
        "左手已回到中间初始位。"
    )
    return 0


def main() -> int:
    global REQUIRE_CONFIRMATION
    args = parse_args()
    validate_args(args)
    REQUIRE_CONFIRMATION = bool(args.require_confirmation)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["max_velocity"] = min(
        1.0, float(config.get("max_velocity", 0.05)) * args.speed_multiplier
    )
    config["max_acceleration"] = min(
        1.0, float(config.get("max_acceleration", 0.05)) * args.speed_multiplier
    )
    if args.resume_right_at_place_ready:
        planner = build_runtime(
            config, args.planner_host, execute_url=args.execute_url
        )
        try:
            planner.open(required_frames={
                "body_base",
                config["left_tool_frame"],
                config["right_tool_frame"],
            })
            current_right = read_tcp(planner, config["right_tool_frame"])
            expected_right = taught_right_tcp_from_joints(
                list(args.right_place_ready_joints_deg)
            )
            translation_m, rotation_deg = transform_error(
                expected_right, current_right
            )
            right_state = read_gripper_state(args.gripper_host, 2, "right")
            if (
                translation_m * 1000.0 > 10.0
                or rotation_deg > 3.0
                or float(right_state.get("position", 100.0)) > 95.0
            ):
                raise RuntimeError(
                    "放置准备位恢复模式拒绝执行：右手不在准备位附近，"
                    "或右夹爪未保持闭合。"
                    f"TCP误差={translation_m * 1000.0:.3f} mm/"
                    f"{rotation_deg:.3f}°，右夹爪状态={right_state}"
                )
            print(json.dumps({
                "resume_state": "RIGHT_AT_PLACE_READY",
                "right_tcp_error_mm": translation_m * 1000.0,
                "right_tcp_error_deg": rotation_deg,
                "right_gripper_state": right_state,
            }, indent=2, ensure_ascii=False))
            return return_left_to_middle(planner, args)
        finally:
            planner.close()

    if args.resume_after_left_release:
        planner = build_runtime(
            config, args.planner_host, execute_url=args.execute_url
        )
        try:
            planner.open(required_frames={
                "body_base",
                config["left_tool_frame"],
                config["right_tool_frame"],
            })
            current_right = read_tcp(planner, config["right_tool_frame"])
            taught_object = np.eye(4)
            taught_object[:3, 3] = np.asarray(args.expected_center_m, dtype=float)
            taught_tilt_rad = np.deg2rad(5.0)
            taught_object[:3, :3] = np.asarray([
                [0.0, -np.cos(taught_tilt_rad), np.sin(taught_tilt_rad)],
                [1.0, 0.0, 0.0],
                [0.0, np.sin(taught_tilt_rad), np.cos(taught_tilt_rad)],
            ])
            taught_tcp = taught_right_tcp_from_joints(
                list(args.right_grasp_joints_deg)
            )
            T_taught_object_right_tcp = (
                np.linalg.inv(taught_object) @ taught_tcp
            )
            resume_errors = []
            for tilt_deg in (0.0, 5.0):
                tilt_rad = np.deg2rad(tilt_deg)
                candidate_object = np.eye(4)
                candidate_object[:3, 3] = np.asarray(
                    args.expected_center_m, dtype=float
                )
                candidate_object[:3, :3] = np.asarray([
                    [0.0, -np.cos(tilt_rad), np.sin(tilt_rad)],
                    [1.0, 0.0, 0.0],
                    [0.0, np.sin(tilt_rad), np.cos(tilt_rad)],
                ])
                candidate_tcp = candidate_object @ T_taught_object_right_tcp
                translation_m, rotation_deg = transform_error(
                    candidate_tcp, current_right
                )
                resume_errors.append({
                    "object_tilt_deg": tilt_deg,
                    "translation_error_mm": translation_m * 1000.0,
                    "rotation_error_deg": rotation_deg,
                })
            best_resume_error = min(
                resume_errors,
                key=lambda item: (
                    item["translation_error_mm"], item["rotation_error_deg"]
                ),
            )
            right_state = read_gripper_state(args.gripper_host, 2, "right")
            invalid_resume_pose = (
                best_resume_error["translation_error_mm"] > 20.0
                or best_resume_error["rotation_error_deg"] > 5.0
            )
            invalid_resume_gripper = (
                float(right_state.get("position", 100.0)) > 95.0
                or right_state.get("motion_state") != "gripping"
            )
            invalid_resume_state = (
                invalid_resume_gripper
                or (
                    invalid_resume_pose
                    and not args.confirm_resume_from_current_held_pose
                )
            )
            if invalid_resume_state and args.execute:
                raise RuntimeError(
                    "恢复模式拒绝执行：当前右手不在合法交接抓取位附近，"
                    "或右夹爪未保持夹持。若已人工确认当前姿态持件安全，"
                    "需显式增加--confirm-resume-from-current-held-pose。"
                    f"候选误差={resume_errors}，右夹爪状态={right_state}"
                )
            if invalid_resume_state:
                print(
                    "[恢复模式只规划警告] 当前状态不满足恢复执行门禁；"
                    f"仅对当前位置规划上升轨迹。候选误差={resume_errors}，"
                    f"右夹爪状态={right_state}"
                )
            print(
                "[恢复模式] 假设右手已夹稳工件且左夹爪已完全打开；"
                "跳过ICP、抓取和夹爪交接。"
            )
            return finish_after_left_release(planner, args, config)
        finally:
            planner.close()

    T_base_object_observed, result = load_icp_object_transform(args.result)
    T_base_left_tcp_captured = validate_icp_for_motion(result, config)
    T_left_tcp_object = validate_transform(
        np.linalg.inv(T_base_left_tcp_captured) @ T_base_object_observed,
        "T_left_tcp_object_from_observation",
    )

    planner = build_runtime(config, args.planner_host, execute_url=args.execute_url)
    try:
        planner.open(required_frames={
            "body_base", config["left_tool_frame"], config["right_tool_frame"]
        })
        T_base_left_tcp_live = read_tcp(planner, config["left_tool_frame"])
        T_base_object_live = validate_transform(
            T_base_left_tcp_live @ T_left_tcp_object,
            "T_base_object_reconstructed_from_live_left_tcp",
        )
        expected_object = taught_handoff_object_transform(
            args.expected_center_m,
            args.taught_object_tilt_deg,
        )
        object_pose_source = "live_left_tcp @ observed_T_left_tcp_object"
        if args.diagnostic_use_nominal_handoff_pose:
            T_base_object_live = expected_object.copy()
            object_pose_source = "diagnostic_nominal_handoff_pose_not_live"
        center_error_mm = float(
            np.linalg.norm(
                T_base_object_live[:3, 3]
                - np.asarray(args.expected_center_m, dtype=float)
            )
            * 1000.0
        )
        front_error_deg, tail_error_deg = handoff_axis_errors(T_base_object_live)
        _, orientation_error_deg = transform_error(
            expected_object, T_base_object_live
        )
        if center_error_mm > args.center_tolerance_mm:
            raise RuntimeError(
                f"工件中心偏离示教交接位 {center_error_mm:.3f} mm，"
                f"超过 {args.center_tolerance_mm:.3f} mm；"
                "超出右臂示教邻域的笛卡尔补偿范围"
            )
        if (
            front_error_deg > args.axis_tolerance_deg
            or tail_error_deg > args.axis_tolerance_deg
            or orientation_error_deg > args.object_orientation_tolerance_deg
        ):
            raise RuntimeError(
                "工件姿态偏离示教交接姿态："
                f"front={front_error_deg:.3f}°, tail={tail_error_deg:.3f}°, "
                f"完整姿态误差={orientation_error_deg:.3f}°；"
                "超出右臂示教邻域的笛卡尔补偿范围"
            )

        right_target = list(args.right_grasp_joints_deg)
        T_base_right_tcp_taught = taught_right_tcp_from_joints(right_target)
        T_taught_object_right_tcp = validate_transform(
            np.linalg.inv(expected_object) @ T_base_right_tcp_taught,
            "T_taught_object_right_tcp",
        )
        T_base_right_tcp_grasp = validate_transform(
            T_base_object_live @ T_taught_object_right_tcp,
            "T_base_right_tcp_adapted_grasp",
        )
        T_base_right_tcp_pregrasp = T_base_right_tcp_grasp.copy()
        pregrasp_offset, pregrasp_offset_metadata = pregrasp_offset_body(
            T_base_right_tcp_grasp,
            T_base_object_live,
            args.pregrasp_distance_m,
            args.pregrasp_axis,
        )
        T_base_right_tcp_pregrasp[:3, 3] += pregrasp_offset
        pregrasp_already_ready = args.right_pregrasp_ready_joints_deg is not None
        if pregrasp_already_ready:
            right_pregrasp_joints = list(args.right_pregrasp_ready_joints_deg)
            pregrasp_ik = {"source": "parallel_pregrasp_ready_marker"}
            right_plan = None
            right_summary = {"planner": "PARALLEL_RRTSTAR_ALREADY_EXECUTED", "endpoint_joints_deg": right_pregrasp_joints}
            trajectory: list[list[float]] = []
            trajectory_start_check = {"gate": "parallel_pregrasp_joint_feedback"}
        else:
            right_pregrasp_joints, pregrasp_ik = solve_nearby_right_joints(
                T_base_right_tcp_pregrasp, right_target
            )
            right_plan = planner.plan_joint_trajectory(
                "right", right_pregrasp_joints, planner="rrtstar"
            )
            if right_plan is None:
                raise RuntimeError("Planner returned no right pregrasp joint plan")
            right_summary = validate_arm_joint_plan(
                right_plan,
                right_pregrasp_joints,
                arm_id=2,
                label="right_pregrasp_trajectory",
                tolerance_deg=2.0,
            )
            trajectory = (right_plan.get("data") or {}).get("trajectory") or []
            trajectory_start_check = validate_trajectory_start_matches_current_state(
                right_plan,
                arm_host=args.arm_host,
                arm_id=2,
                label="right_pregrasp_trajectory",
                tolerance_deg=2.0,
            )
        report = {
            "mode": "EXECUTE" if args.execute else "PLAN_ONLY",
            "strategy": (
                "RRTSTAR_PREGRASP_THEN_EXACT_TAUGHT_JOINTS"
                if args.final_grasp_motion == "taught-joints"
                else "RRTSTAR_PREGRASP_THEN_LIVE_ICP_REBUILD_AND_LIN"
            ),
            "source_result": str(args.result.resolve()),
            "object_pose_source": object_pose_source,
            "reconstructed_object": transform_debug(T_base_object_live),
            "center_error_mm": center_error_mm,
            "front_up_error_deg": front_error_deg,
            "tail_to_right_error_deg": tail_error_deg,
            "orientation_error_from_taught_handoff_deg": orientation_error_deg,
            "right_taught_grasp_tcp": transform_debug(T_base_right_tcp_taught),
            "right_adapted_grasp_tcp": transform_debug(T_base_right_tcp_grasp),
            "grasp_adaptation": {
                "method": "live_object @ taught_object_to_right_tcp",
                "object_orientation_delta_deg": orientation_error_deg,
            },
            "right_pregrasp_tcp": transform_debug(T_base_right_tcp_pregrasp),
            "right_pregrasp_joints_deg": right_pregrasp_joints,
            "right_pregrasp_ik": pregrasp_ik,
            "pregrasp_offset": {
                **pregrasp_offset_metadata,
                "body_base_vector_m": pregrasp_offset.tolist(),
            },
            "trajectory_start_check": trajectory_start_check,
            "right_pregrasp_plan": right_summary,
            "trajectory": trajectory_summary(right_plan, 2) if right_plan is not None else {"source": "parallel_pregrasp_ready_marker"},
            "safety_note": (
                "先RRTStar到工件尾端外侧预抓取位；真实到位后重新读取左TCP，"
                + (
                    "确认工件仍在示教邻域，再RRTStar直达示教关节位。"
                    if args.final_grasp_motion == "taught-joints"
                    else
                    "由观察ICP的TCP-工件刚性关系重建工件位姿，"
                    "再LIN到补偿抓取位。"
                )
                + "MoveIt若未附着工件模型，不能检查左手所持工件本体。"
            ),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not args.execute:
            print(
                "[完成] 右臂关节RRTStar预抓取规划通过；最终运动将在预抓取"
                "真实到位后重新规划。机械臂和夹爪均未动作。"
            )
            return 0

        right_state = read_gripper_state(args.gripper_host, 2, "right")
        right_opening = float(right_state.get("position", -1.0))
        if right_opening < args.right_open_position - 5.0:
            if not confirmed(
                f"右夹爪当前开度={right_opening:.1f}%，输入 y 张开到"
                f"{args.right_open_position:.1f}%："
            ):
                print("已取消；右臂未运动。")
                return 0
            print(move_gripper(
                args.gripper_host, 2, args.right_open_position,
                args.gripper_force, "right",
            ))

        if not pregrasp_already_ready and not confirmed(
            "确认右夹爪已全开，关节RRTStar到工件尾端外侧预抓取位的"
            "扫掠路径不会碰左臂/工件，输入 y 执行："
        ):
            print("已取消；右臂未运动。")
            return 0
        if pregrasp_already_ready:
            wait_for_joint_target(
                args.arm_host, 2, right_pregrasp_joints, 0.5,
                "parallel right pregrasp", timeout_s=30.0,
            )
            pregrasp_arrival = {"gate": "parallel_pregrasp_joint_feedback", "target_joints_deg": right_pregrasp_joints}
        else:
            planner.execute_planned_result(right_plan)
            pregrasp_arrival = wait_for_pose_close(
                planner,
                config["right_tool_frame"],
                T_base_right_tcp_pregrasp,
                args.arrival_translation_tolerance_mm,
                args.arrival_rotation_tolerance_deg,
                "right taught tail pregrasp pose",
                timeout_s=90.0,
            )
        print(f"右臂已到示教区域预抓取位：{pregrasp_arrival}")

        # The left hand still rigidly holds the workpiece. Rebuild the workpiece
        # pose after the coarse right-arm move, then plan the configured final move.
        T_base_left_tcp_adjust = read_tcp(planner, config["left_tool_frame"])
        T_base_object_adjust = validate_transform(
            T_base_left_tcp_adjust @ T_left_tcp_object,
            "T_base_object_rebuilt_before_right_lin",
        )
        adjusted_center_error_mm = float(
            np.linalg.norm(
                T_base_object_adjust[:3, 3]
                - np.asarray(args.expected_center_m, dtype=float)
            )
            * 1000.0
        )
        adjusted_front_error_deg, adjusted_tail_error_deg = handoff_axis_errors(
            T_base_object_adjust
        )
        _, adjusted_orientation_error_deg = transform_error(
            expected_object, T_base_object_adjust
        )
        if adjusted_center_error_mm > args.center_tolerance_mm:
            raise RuntimeError(
                "预抓取到位后工件中心超出补偿范围："
                f"{adjusted_center_error_mm:.3f} mm > "
                f"{args.center_tolerance_mm:.3f} mm；左手保持夹持"
            )
        if (
            adjusted_front_error_deg > args.axis_tolerance_deg
            or adjusted_tail_error_deg > args.axis_tolerance_deg
            or adjusted_orientation_error_deg
            > args.object_orientation_tolerance_deg
        ):
            raise RuntimeError(
                "预抓取到位后工件姿态超出补偿范围："
                f"front={adjusted_front_error_deg:.3f}°, "
                f"tail={adjusted_tail_error_deg:.3f}°, "
                f"orientation={adjusted_orientation_error_deg:.3f}°；"
                "左手保持夹持"
            )
        T_base_right_tcp_grasp_adjust = validate_transform(
            T_base_object_adjust @ T_taught_object_right_tcp,
            "T_base_right_tcp_live_adjusted_grasp",
        )
        correction_translation_m, correction_rotation_deg = transform_error(
            T_base_right_tcp_grasp, T_base_right_tcp_grasp_adjust
        )
        if args.final_grasp_motion == "taught-joints":
            right_final_plan = planner.plan_joint_trajectory(
                "right", right_target, planner="rrtstar"
            )
            if right_final_plan is None:
                raise RuntimeError(
                    "Planner returned no exact taught-joint grasp plan"
                )
            right_final_validation = validate_arm_joint_plan(
                right_final_plan,
                right_target,
                arm_id=2,
                label="right_taught_grasp_trajectory",
                tolerance_deg=args.right_grasp_tolerance_deg,
            )
            right_final_summary = validate_trajectory_start_matches_current_state(
                right_final_plan,
                arm_host=args.arm_host,
                arm_id=2,
                label="right_taught_grasp_trajectory",
                tolerance_deg=2.0,
            )
            right_final_summary.update({
                "planner": "RRTSTAR",
                "trajectory_points": len(
                    (right_final_plan.get("data") or {}).get("trajectory") or []
                ),
                "target_joints_deg": right_target,
                "endpoint_validation": right_final_validation,
            })
            final_motion_note = (
                "实时ICP仅用于确认左手所持工件仍在示教邻域；"
                "最终位置不应用未标定FK的笛卡尔修正，"
                "直接到godr1验证的示教关节位"
            )
        else:
            right_final_plan = planner.plan_cartesian_pose(
                "right",
                transform_to_object_pose(T_base_right_tcp_grasp_adjust),
                planner="lin",
            )
            if right_final_plan is None:
                raise RuntimeError(
                    "Planner returned no live-adjusted right LIN grasp plan"
                )
            right_final_summary = validate_trajectory_start_matches_current_state(
                right_final_plan,
                arm_host=args.arm_host,
                arm_id=2,
                label="right_live_adjusted_lin_trajectory",
                tolerance_deg=2.0,
            )
            right_final_summary["planner"] = "LIN"
            right_final_summary["trajectory_points"] = len(
                (right_final_plan.get("data") or {}).get("trajectory") or []
            )
            final_motion_note = "最终LIN按实时左TCP×ICP关系修正抓取TCP"
        print(json.dumps({
            "final_grasp_motion": {
                "mode": args.final_grasp_motion,
                "object_pose": transform_debug(T_base_object_adjust),
                "object_center_error_mm": adjusted_center_error_mm,
                "computed_tcp_change_since_preplan_mm": (
                    correction_translation_m * 1000.0
                ),
                "computed_tcp_change_since_preplan_deg": correction_rotation_deg,
                "computed_live_adjusted_tcp": transform_debug(
                    T_base_right_tcp_grasp_adjust
                ),
                "plan": right_final_summary,
                "note": final_motion_note,
            }
        }, indent=2, ensure_ascii=False))
        final_prompt = (
            "右臂已在示教区域，确认从预抓取位RRTStar到godr1验证的"
            "示教关节位路径安全，输入 y 执行最终交接："
            if args.final_grasp_motion == "taught-joints"
            else
            "右臂已在示教区域，确认按实时ICP关系修正后的LIN路径安全，"
            "输入 y 执行最终抓取调整："
        )
        if not confirmed(final_prompt):
            print("已停止；右手停在预抓取位，左手仍持有工件。")
            return 0
        planner.execute_planned_result(right_final_plan)
        if args.final_grasp_motion == "taught-joints":
            wait_for_joint_target(
                args.arm_host,
                2,
                right_target,
                args.right_grasp_tolerance_deg,
                "right exact taught grasp",
                timeout_s=90.0,
            )
            print(
                "右臂已到godr1验证的最终示教关节位；"
                "未使用未标定FK生成最终笛卡尔目标。",
                flush=True,
            )
        else:
            arrival_error = wait_for_pose_close(
                planner,
                config["right_tool_frame"],
                T_base_right_tcp_grasp_adjust,
                args.arrival_translation_tolerance_mm,
                args.arrival_rotation_tolerance_deg,
                "right live-adjusted tail grasp pose",
                timeout_s=30.0,
            )
            print(f"右臂已完成ICP关系笛卡尔微调：{arrival_error}")

        if not confirmed("确认右夹爪已对准工件尾部，输入 y 闭合右夹爪："):
            print("已停止；右臂停在示教抓取位，左手仍持有工件。")
            return 0
        print(move_gripper(
            args.gripper_host, 2, args.right_close_position,
            args.gripper_force, "right",
        ))
        if not confirmed("人工确认右手已夹稳尾部，输入 y 释放左夹爪："):
            print("已停止；双手共同夹持工件。")
            return 0

        if args.switch_attachment:
            attachment_ack = request_attachment_owner(
                args.attachment_state, "right", timeout_s=5.0
            )
            print(f"MoveIt附着体已切换到右手：{attachment_ack}")
        else:
            print("[警告] 未切换附着碰撞体；后续规划不包含右手持件几何。")
        print(move_gripper(
            args.gripper_host, 1, args.left_open_position,
            args.gripper_force, "left",
        ))
        return finish_after_left_release(planner, args, config)
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
