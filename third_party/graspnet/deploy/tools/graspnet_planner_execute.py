#!/usr/bin/env python3
"""Run the unchanged Version5 GraspNet detector and execute with Planner.

The detector, camera acquisition, candidate filtering, calibration checks and
``grasp_execution_plan.json`` are owned by ``deploy/version5``.  This entry
point only replaces Version5's Cartesian servo execution with Planner RRTStar/LIN
trajectory planning.  The resulting joint trajectory is submitted to the
plucky planned-trajectory execution endpoint; no ``start_servo_pose`` or
``push_servo_pose`` command is used here.

The bridge calibration maps a deployment plan (``base`` -> ``eef_pose``) into
its captured arm-flange frames.  At runtime the script then uses Planner TF to
extend that result to the actual Cartesian IK TCP (normally
``body_base`` -> ``left_hand_transfer_end``):

    T_planner_base_planner_tool = T_planner_base_base
                                  @ T_base_eef
                                  @ T_eef_planner_tool

Generate and validate this calibration first with
``deploy/tools/calibrate_planner_tf_bridge.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _allow_right_hand_tall_contact() -> bool:
    """Return whether this loop is inside Module 9's expected contact window."""
    marker = os.environ.get("PARM_RIGHT_SERVO_TALL_CONTACT_FILE")
    return bool(marker) and Path(marker).is_file()


def _with_right_hand_tall_contact(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the narrow request-local ACM exception only while Module 9 is active."""
    if _allow_right_hand_tall_contact():
        payload["allow_right_hand_tall_contact"] = True
    return payload
LOOP_ROOT = Path(os.environ.get("PARM_LOOP_ROOT", str(HERE.parents[4])))
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))
from tool.planner_task_retry import retry_task_creation
# Use the SDK installed in the GraspNet virtualenv.  In particular, do not
# prepend runtime/parm_sdk's archived 20260717 wheel: it speaks the legacy
# gripper protocol and maps schema-3 left_gripper/right_gripper states to None.
# The deployment environment is provisioned with the 20260803 SDK, which
# supplies Arm, Gripper and Planner together.
try:
    from parm_sdk import Gripper as _GripperSDKProbe
    from parm_sdk import Planner as _PlannerSDKProbe
except ImportError as exc:
    raise RuntimeError(
        "GraspNet virtualenv requires the current Planner + schema-3 Gripper parm-sdk"
    ) from exc

from deploy.version5 import run_dual_camera_fused_graspnet as version5  # noqa: E402
from deploy.version4.run_d405_graspnet import (  # noqa: E402
    collector_gripper_session,
    close_collector_gripper_session,
    close_gripper,
    move_gripper,
)


DEFAULT_BRIDGE = ROOT / "deploy" / "calibration" / "output" / "planner_tf_bridge.json"
DEFAULT_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"
DEFAULT_WEIGHT_CHECK_ROOT = LOOP_ROOT
LEFT_ARM_ID = 1
LEFT_GROUP = "left_arm"
LEFT_JOINT_NAMES = [f"left_arm_joint{index}" for index in range(1, 8)]
GRIPPER_HOLD_STABLE_S = 0.5
GRIPPER_HOLD_POLL_S = 0.1
GRIPPER_HOLD_TIMEOUT_S = 2.0


class RetryGrasp(RuntimeError):
    """Request a fresh observation/inference/grasp attempt after rejection."""

    def __init__(self, grasp_class: str, weight_result: Any | None = None):
        super().__init__(grasp_class)
        self.grasp_class = grasp_class
        self.weight_result = weight_result


class PlannerNoTrajectoryError(RuntimeError):
    """Planner responded normally but could not produce a usable trajectory."""


class CandidatePlanRejected(RuntimeError):
    """A specific grasp candidate is unreachable before the gripper closes."""

    def __init__(self, phase: str, cause: Exception):
        super().__init__(f"{phase}: {cause}")
        self.phase = phase
        self.cause = cause


class PregraspDistanceRejected(RuntimeError):
    """A candidate-specific pregrasp distance safety gate rejection."""


class ForceSensorDataError(RuntimeError):
    """Force stream/sample data was unavailable at a named grasp phase."""

    def __init__(self, phase: str, cause: Exception):
        super().__init__(f"{phase}: {cause}")
        self.phase = phase
        self.cause = cause


def is_candidate_unreachable_error(exc: Exception) -> bool:
    """Recognize the Planner SDK's explicit no-path response.

    The deployed SDK raises ``ParmAPIError`` rather than returning an empty
    trajectory when MoveIt cannot solve a request.  Restrict this conversion
    to explicit trajectory-planning failures so transport, TF and API-format
    failures remain hard faults.
    """
    if type(exc).__name__ != "ParmAPIError":
        return False
    text = str(exc).lower()
    return any(marker in text for marker in (
        "轨迹规划失败", "规划失败", "无可行轨迹", "无解", "ik failed", "no solution",
    ))


def is_planner_connection_error(exc: Exception) -> bool:
    """Recognize a transient failure while opening the three Planner sockets."""
    if isinstance(exc, OSError):
        return True
    if type(exc).__name__ != "ParmAPIError":
        return False
    text = str(exc).lower()
    return any(marker in text for marker in (
        "no server listening",
        "connection refused",
        "连接被拒绝",
        "连接超时",
        "timed out",
        "timeout",
    ))


def connect_planner(args: argparse.Namespace, label: str):
    """Wait for a transient Planner restart without sending a robot command.

    Constructing ``Planner`` only establishes its trajectory/TF/waypoint
    clients.  No plan or execution request is sent here.  Once all sockets are
    back, callers still create a new trajectory from the live robot state;
    trajectories produced before a service restart are never executed.
    """
    timeout_s = float(args.planner_connect_timeout_s)
    retry_s = float(args.planner_connect_retry_s)
    if not math.isfinite(timeout_s) or timeout_s < 0.0:
        raise RuntimeError("--planner-connect-timeout-s must be finite and non-negative")
    if not math.isfinite(retry_s) or retry_s <= 0.0:
        raise RuntimeError("--planner-connect-retry-s must be finite and positive")
    started = time.monotonic()
    announced = False
    while True:
        try:
            planner = _PlannerSDKProbe(
                host=args.planner_host,
                trajectory_port=args.planner_trajectory_port,
                tf_port=args.planner_tf_port,
                waypoint_port=args.planner_waypoint_port,
            )
            if announced:
                print(
                    f"-> Planner reconnected for {label} after "
                    f"{time.monotonic() - started:.1f}s; replanning from live state",
                    flush=True,
                )
            return planner
        except Exception as exc:
            if not is_planner_connection_error(exc):
                raise
            elapsed = time.monotonic() - started
            if elapsed >= timeout_s:
                raise RuntimeError(
                    f"Planner unavailable for {label} after {elapsed:.1f}s; "
                    "no robot or gripper command was sent during reconnect"
                ) from exc
            if not announced:
                print(
                    f"-> Planner temporarily unavailable during {label}; "
                    f"waiting up to {timeout_s:.1f}s without robot motion",
                    flush=True,
                )
                announced = True
            time.sleep(min(retry_s, max(0.0, timeout_s - elapsed)))


def build_parser() -> argparse.ArgumentParser:
    """Keep every Version5 detection option and add Planner-only execution options."""
    parser = _VERSION5_BUILD_PARSER()
    parser.description = (
        "Run unchanged Version5 GraspNet detection, then execute its plan with Planner "
        "instead of direct Cartesian servo control."
    )
    group = parser.add_argument_group("Planner execution")
    # Some local Version5 revisions already expose --planner-host for their
    # own diagnostics.  Reuse that argument instead of creating a conflict.
    if "--planner-host" not in parser._option_string_actions:
        group.add_argument("--planner-host", default="127.0.0.1")
    group.add_argument(
        "--planner-trajectory-port", type=int, default=5605,
        help="GraspNet dedicated Planner trajectory port (default: 5605).",
    )
    group.add_argument(
        "--planner-tf-port", type=int, default=5614,
        help="GraspNet dedicated Planner TF port (default: 5614).",
    )
    group.add_argument(
        "--planner-waypoint-port", type=int, default=5615,
        help="GraspNet dedicated Planner waypoint port (default: 5615).",
    )
    group.add_argument(
        "--planner-connect-timeout-s", type=float, default=30.0,
        help=(
            "wait this long for a transient Planner restart before failing; "
            "waiting never sends a robot/gripper command (default: 30)"
        ),
    )
    group.add_argument(
        "--planner-connect-retry-s", type=float, default=0.5,
        help="Planner reconnect polling interval in seconds (default: 0.5)",
    )
    group.add_argument("--planner-execute-url", default=DEFAULT_EXECUTE_URL)
    group.add_argument("--planner-bridge-calibration", default=str(DEFAULT_BRIDGE))
    group.add_argument(
        "--planner-direct-pose", action="store_true",
        help=(
            "Plan poses already expressed in --planner-base-frame directly. "
            "No planner_tf_bridge calibration is loaded or applied."
        ),
    )
    group.add_argument(
        "--planner-base-frame", default="body_base",
        help="Planner Cartesian reference frame; must match the Planner motion configuration.",
    )
    group.add_argument(
        "--planner-tool-frame", default="left_hand_transfer_end",
        help="Planner IK TCP; must be an available left-hand tool frame, not left_arm_end_link.",
    )
    group.add_argument(
        "--planner-direct-grasp-center", action="store_true",
        help=(
            "Treat Version7 pregrasp/grasp poses as the desired Planner IK-frame "
            "center itself. This bypasses the legacy SDK-EEF/TCP tool offset and "
            "requires --planner-tool-frame left_eef."
        ),
    )
    group.add_argument("--planner-group", default=LEFT_GROUP)
    group.add_argument("--planner-max-vel", type=float, default=0.1)
    group.add_argument("--planner-max-acc", type=float, default=0.1)
    group.add_argument(
        "--planner-pregrasp-mode", choices=("lin", "rrtstar"), default="rrtstar",
        help=(
            "Planner mode for the collision-free approach to the pregrasp pose. "
            "rrtstar is sent as a raw Cartesian waypoint because the SDK public "
            "helper only accepts ptp/lin/ompl."
        ),
    )
    group.add_argument(
        "--planner-descend-mode", choices=("lin",), default="lin",
        help="Planner mode for pregrasp-to-grasp descent; keep lin for a straight descent.",
    )
    group.add_argument(
        "--planner-plan-only", action="store_true",
        help="Plan and print each trajectory, but never submit it to the execution endpoint.",
    )
    group.add_argument(
        "--auto-confirm-real-grasp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "bypass the terminal y/yes prompt before real grasp execution; "
            "omit or pass --no-auto-confirm-real-grasp to require manual confirmation"
        ),
    )
    weight = parser.add_argument_group("Left grasp weight verification")
    weight.add_argument(
        "--grasp-weight-check",
        action="store_true",
        help="verify the lifted left-hand grasp and retry until exactly one object is held",
    )
    weight.add_argument("--force-sensor-host", default="127.0.0.1")
    weight.add_argument("--force-sensor-hand", type=int, choices=(1,), default=1)
    weight.add_argument("--weight-check-samples", type=int, default=31)
    weight.add_argument("--weight-check-timeout-s", type=float, default=3.0)
    weight.add_argument("--weight-check-settle-s", type=float, default=0.3)
    weight.add_argument("--weight-empty-max-g", type=float, default=30.0)
    weight.add_argument("--weight-single-min-g", type=float, default=40.0)
    weight.add_argument("--weight-single-max-g", type=float, default=100.0)
    weight.add_argument("--weight-multiple-min-g", type=float, default=180.0)
    weight.add_argument(
        "--weight-contiguous-single-band",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("classify every stable weight strictly between EMPTY and MULTIPLE as SINGLE; "
              "disabled by default to preserve the legacy UNKNOWN hysteresis bands"),
    )
    weight.add_argument("--weight-max-axis-mad-n", type=float, default=0.15)
    weight.add_argument(
        "--weight-check-root",
        type=Path,
        default=DEFAULT_WEIGHT_CHECK_ROOT,
        help="project root containing tests.grasp_weight_check",
    )
    return parser


def load_weight_check_api(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    """Load the already-tested weight classifier without duplicating it here."""
    root = Path(args.weight_check_root).resolve()
    module_file = root / "tests" / "grasp_weight_check" / "grasp_weight_check.py"
    if not module_file.exists():
        raise RuntimeError(f"grasp weight check module is missing: {module_file}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests.grasp_weight_check import (  # type: ignore[import-not-found]
        GraspClass,
        WeightThresholds,
        collect_force_window,
        verify_grasp_after_lift,
    )

    return GraspClass, WeightThresholds, collect_force_window, verify_grasp_after_lift


def build_weight_thresholds(args: argparse.Namespace, threshold_type: Any) -> Any:
    return threshold_type(
        empty_max_g=float(args.weight_empty_max_g),
        single_min_g=float(args.weight_single_min_g),
        single_max_g=float(args.weight_single_max_g),
        multiple_min_g=float(args.weight_multiple_min_g),
        max_axis_mad_n=float(args.weight_max_axis_mad_n),
    )


def print_weight_result(result: Any) -> None:
    print("-> grasp weight result")
    print(f"   class: {result.grasp_class.value}")
    print(f"   mass: {result.mass_g:.1f} g")
    print(f"   delta force: {[round(float(value), 4) for value in result.delta_force_n]} N")
    print(
        "   baseline median/MAD: "
        f"{[round(float(value), 4) for value in result.baseline.median_force_n]} / "
        f"{[round(float(value), 4) for value in result.baseline.axis_mad_n]} N"
    )
    print(
        "   loaded median/MAD: "
        f"{[round(float(value), 4) for value in result.loaded.median_force_n]} / "
        f"{[round(float(value), 4) for value in result.loaded.axis_mad_n]} N"
    )
    print(f"   reason: {result.reason}")


def require_left_gripper_holding(args: argparse.Namespace, *, phase: str) -> None:
    """Require the schema-3 left gripper to report a stable held state.

    This check intentionally runs only after the arm has reached ``graspup``.
    A closing gripper can briefly report a non-holding state while the workpiece
    is still supported by the table, so it must never prevent the configured
    post-grasp lift from being attempted.
    """

    gripper = collector_gripper_session(args)
    deadline = time.monotonic() + GRIPPER_HOLD_TIMEOUT_S
    last_state: dict[str, Any] | None = None
    stable_since: float | None = None
    stable_samples = 0
    while time.monotonic() < deadline:
        state = gripper.get_state().get(args.gripper_hand)
        if isinstance(state, dict):
            last_state = state
            motion_state = str(state.get("motion_state", "")).strip().lower()
            if motion_state == "gripping":
                stable_since = stable_since or time.monotonic()
                stable_samples += 1
                if time.monotonic() - stable_since >= GRIPPER_HOLD_STABLE_S:
                    print(
                        "-> gripper hold check passed: "
                        f"gripping stable for {GRIPPER_HOLD_STABLE_S:.1f}s "
                        f"({stable_samples} samples), position={state.get('position')}",
                        flush=True,
                    )
                    return
            else:
                stable_since = None
                stable_samples = 0
            if motion_state in {"dropped", "error"} or state.get("error"):
                break
        time.sleep(GRIPPER_HOLD_POLL_S)

    print(
        f"-> gripper hold check failed at {phase}: "
        f"expected motion_state='gripping', got {last_state}; "
        "opening at the current graspup pose and starting fresh inference",
        flush=True,
    )
    move_gripper(args, 100.0, "release non-holding grasp at graspup")
    raise RetryGrasp("EMPTY")


def apply_contiguous_single_band(
    result: Any,
    thresholds: Any,
    grasp_class: Any,
    enabled: bool,
) -> Any:
    """Remove legacy hysteresis bands without masking unstable force data.

    The reusable force-check package intentionally enforces strict threshold
    gaps.  Version7 needs a continuous operational policy instead: stable
    weights above EMPTY and below MULTIPLE represent exactly one workpiece.
    Keep its MAD-based UNKNOWN outcome unchanged.
    """
    if not enabled or result.grasp_class is not grasp_class.UNKNOWN:
        return result
    if max(result.baseline.max_axis_mad_n, result.loaded.max_axis_mad_n) > thresholds.max_axis_mad_n:
        return result

    mass_g = float(result.mass_g)
    # The force-to-mass conversion can differ from an exact decimal boundary
    # by ~1e-14 g.  This is only numerical tolerance, not a hysteresis band.
    boundary_tolerance_g = 1e-6
    if mass_g <= float(thresholds.empty_max_g) + boundary_tolerance_g:
        normalized = grasp_class.EMPTY
    elif mass_g >= float(thresholds.multiple_min_g) - boundary_tolerance_g:
        normalized = grasp_class.MULTIPLE
    else:
        normalized = grasp_class.SINGLE
    return replace(
        result,
        grasp_class=normalized,
        reason=(
            f"连续单件区间策略：稳定估算重量 {mass_g:.1f} g "
            f"归类为 {normalized.value}"
        ),
    )


def _finite_transform(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"{label} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise RuntimeError(f"{label} does not contain a proper rotation")
    return matrix


def load_bridge(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str, str]:
    if bool(getattr(args, "planner_direct_pose", False)):
        # Direct Planner calibration path: the caller has already expressed
        # the desired IK-frame pose in planner_base_frame.  Do not read a
        # legacy SDK-to-Planner bridge file.
        return np.eye(4), np.eye(4), args.planner_base_frame, args.planner_tool_frame
    path = Path(args.planner_bridge_calibration)
    if not path.exists():
        raise RuntimeError(
            f"Planner bridge calibration is missing: {path}. Run calibrate_planner_tf_bridge.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ACCEPTED":
        raise RuntimeError(f"Planner bridge calibration is not ACCEPTED: {path}")
    if payload.get("source_base_frame") != "base" or payload.get("source_eef_frame") != "eef_pose":
        raise RuntimeError(f"Planner bridge source frames do not match the GraspNet plan: {path}")
    bridge_base_frame = payload.get("planner_base_frame")
    bridge_tool_frame = payload.get("planner_tool_frame")
    if not isinstance(bridge_base_frame, str) or not isinstance(bridge_tool_frame, str):
        raise RuntimeError(f"Planner bridge is missing its captured Planner frame names: {path}")
    return (
        _finite_transform(payload.get("T_planner_base_base"), "T_planner_base_base"),
        _finite_transform(payload.get("T_eef_planner_tool"), "T_eef_planner_tool"),
        bridge_base_frame,
        bridge_tool_frame,
    )


def deployment_pose_to_bridge_pose(
    pose: Sequence[float], bridge_base: np.ndarray, bridge_tool: np.ndarray
) -> tuple[list[float], list[float]]:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise RuntimeError("GraspNet execution pose must contain six finite xyz/rotation-vector values")
    base_eef = np.eye(4, dtype=np.float64)
    base_eef[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    base_eef[:3, 3] = values[:3]
    bridge_base_tool = bridge_base @ base_eef @ bridge_tool
    return bridge_base_tool[:3, 3].tolist(), Rotation.from_matrix(bridge_base_tool[:3, :3]).as_quat().tolist()


def _planner_lookup_target_source(planner: Any, source: str, target: str) -> np.ndarray:
    """Return T_target_source from this deployed Planner's inverse TF response.

    The deployed service labels a lookup as source -> target but serializes the
    inverse transform.  This convention is documented by the normalized bridge
    samples used to create ``planner_tf_bridge.json``; normalize it here before
    composing frames rather than relying on the SDK's nominal convention.
    """
    response = planner.lookup_transform(source, target)
    data = (response or {}).get("data") or {}
    if data.get("source_frame") != source or data.get("target_frame") != target:
        raise RuntimeError(f"unexpected Planner TF response for {source} -> {target}: {data}")
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = np.asarray([translation.get(axis) for axis in ("x", "y", "z")], dtype=np.float64)
    quaternion = np.asarray([rotation.get(axis) for axis in ("x", "y", "z", "w")], dtype=np.float64)
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 1e-9:
        raise RuntimeError(f"invalid Planner TF response for {source} -> {target}: {data}")
    deployed_response = np.eye(4, dtype=np.float64)
    deployed_response[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    deployed_response[:3, 3] = xyz
    return np.linalg.inv(deployed_response)


def planner_target_transforms(
    planner: Any, args: argparse.Namespace, bridge_base_frame: str, bridge_tool_frame: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed transforms bracketing the dynamic bridge tool pose."""
    if bool(getattr(args, "planner_direct_pose", False)):
        return np.eye(4), np.eye(4)
    planning_base_bridge_base = _planner_lookup_target_source(
        planner, bridge_base_frame, args.planner_base_frame
    )
    bridge_tool_planning_tool = _planner_lookup_target_source(
        planner, args.planner_tool_frame, bridge_tool_frame
    )
    return planning_base_bridge_base, bridge_tool_planning_tool


def deployment_pose_to_planner_pose(
    pose: Sequence[float], bridge_base: np.ndarray, bridge_tool: np.ndarray,
    planning_base_bridge_base: np.ndarray, bridge_tool_planning_tool: np.ndarray,
    *, direct_grasp_center: bool = False, direct_pose: bool = False,
) -> tuple[list[float], list[float]]:
    if direct_pose:
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise RuntimeError("direct Planner pose must contain six finite xyz/rotation-vector values")
        return values[:3].tolist(), Rotation.from_rotvec(values[3:]).as_quat().tolist()
    # In direct-center mode ``pose`` is already the desired pose of the
    # Planner IK frame (left_eef), expressed in the legacy SDK base.  Only the
    # fixed base-frame alignment is needed.  Applying bridge_tool or
    # bridge_tool_planning_tool here would reintroduce the retired gripper TCP
    # offset and move the new open-jaw center away from the GraspNet center.
    if direct_grasp_center:
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise RuntimeError("GraspNet center pose must contain six finite xyz/rotation-vector values")
        sdk_base_center = np.eye(4, dtype=np.float64)
        sdk_base_center[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
        sdk_base_center[:3, 3] = values[:3]
        planning_base_tool = planning_base_bridge_base @ bridge_base @ sdk_base_center
        return (
            planning_base_tool[:3, 3].tolist(),
            Rotation.from_matrix(planning_base_tool[:3, :3]).as_quat().tolist(),
        )

    bridge_position, bridge_orientation = deployment_pose_to_bridge_pose(pose, bridge_base, bridge_tool)
    bridge_base_tool = np.eye(4, dtype=np.float64)
    bridge_base_tool[:3, :3] = Rotation.from_quat(bridge_orientation).as_matrix()
    bridge_base_tool[:3, 3] = bridge_position
    planning_base_tool = (
        planning_base_bridge_base @ bridge_base_tool @ bridge_tool_planning_tool
    )
    return planning_base_tool[:3, 3].tolist(), Rotation.from_matrix(planning_base_tool[:3, :3]).as_quat().tolist()


def _validate_planner_args(args: argparse.Namespace) -> None:
    if args.arm_id != LEFT_ARM_ID or args.planner_group != LEFT_GROUP:
        raise RuntimeError("this GraspNet deployment supports arm1 only; use --planner-group left_arm")
    for name in ("planner_max_vel", "planner_max_acc"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in ("planner_trajectory_port", "planner_tf_port", "planner_waypoint_port"):
        value = int(getattr(args, name))
        if not 1 <= value <= 65535:
            raise RuntimeError(f"--{name.replace('_', '-')} must be a valid TCP port")
    if bool(getattr(args, "planner_direct_grasp_center", False)) and args.planner_tool_frame != "left_eef":
        raise RuntimeError(
            "--planner-direct-grasp-center requires --planner-tool-frame left_eef"
        )
    if bool(getattr(args, "planner_direct_pose", False)) and not bool(getattr(args, "planner_direct_grasp_center", False)):
        raise RuntimeError("--planner-direct-pose requires --planner-direct-grasp-center")


def resolve_planner_tool_frame(args: argparse.Namespace, frame_ids: Sequence[str]) -> str:
    """Choose an available left-hand TCP without silently falling back to flange IK."""
    available = set(frame_ids)
    requested = str(args.planner_tool_frame)
    if requested in available:
        return requested
    # The currently deployed MoveIt chain has left_hand_transfer_end as its
    # tip.  Older local waypoint files used tool-base/end names which are not
    # published by this Planner instance.
    legacy_tool_frames = {"left_hand_tool_end_link", "left_hand_tool_base"}
    if requested in legacy_tool_frames and "left_hand_transfer_end" in available:
        print(
            f"-> Planner frame {requested} is unavailable; "
            "using available Cartesian TCP left_hand_transfer_end"
        )
        args.planner_tool_frame = "left_hand_transfer_end"
        return args.planner_tool_frame
    raise RuntimeError(
        f"Planner IK frame {requested!r} is absent; available left-hand/tool frames: "
        f"{sorted(frame for frame in available if 'left' in frame or 'tool' in frame)}"
    )


def plan_cartesian(
    planner: Any, args: argparse.Namespace, pose: Sequence[float], mode: str, label: str,
    bridge_base: np.ndarray, bridge_tool: np.ndarray, planning_base_bridge_base: np.ndarray,
    bridge_tool_planning_tool: np.ndarray,
) -> dict[str, Any]:
    position, orientation = deployment_pose_to_planner_pose(
        pose, bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool,
        direct_grasp_center=bool(getattr(args, "planner_direct_grasp_center", False)),
        direct_pose=bool(getattr(args, "planner_direct_pose", False)),
    )
    print(f"-> Planner {mode.upper()} {label}: frame={args.planner_base_frame}, xyz={np.round(position, 4).tolist()}")
    if mode == "rrtstar" or _allow_right_hand_tall_contact():
        # Match PlannerRuntime, used by the rest of the Loop.  The Parm SDK's
        # public Cartesian helper rejects "rrtstar", but the ordinary 5604
        # Planner exposes this named pipeline.  Use exactly one normal cart
        # waypoint; the server performs IK and collision-checked joint-space
        # RRTstar planning to the pregrasp pose.
        def request_with_collision_policy() -> dict[str, Any]:
            payload = {
                "waypoints": [{
                    "name": f"graspnet_{mode}_pregrasp",
                    "type": "cart",
                    "group": args.planner_group,
                    "planner": mode,
                    "max_velocity_scaling_factor": float(args.planner_max_vel),
                    "max_acceleration_scaling_factor": float(args.planner_max_acc),
                    "frame_id": args.planner_base_frame,
                    "ik_frame": args.planner_tool_frame,
                    "position": position,
                    "orientation": orientation,
                }],
                "topp": {
                    "max_vel_factor": float(args.planner_max_vel),
                    "max_acc_factor": float(args.planner_max_acc),
                    "final_phase_ratio": 0.05,
                    "final_phase_oversample": 2.0,
                },
            }
            return planner._trajectory.send_payload(
                _with_right_hand_tall_contact(payload), timeout=30.0
            )

        result = retry_task_creation(
            request_with_collision_policy,
            label=f"GraspNet {label} {mode.upper()} 笛卡尔规划",
        )
    else:
        result = retry_task_creation(
            lambda: planner.plan_cart_trajectory(
                group=args.planner_group,
                planner=mode,
                frame_id=args.planner_base_frame,
                ik_frame=args.planner_tool_frame,
                position=position,
                orientation=orientation,
                max_vel=float(args.planner_max_vel),
                max_acc=float(args.planner_max_acc),
            ),
            label=f"GraspNet {label} {mode.upper()} 笛卡尔规划",
        )
    return validate_trajectory(result, label)


def plan_cartesian_from_joint_state(
    planner: Any,
    args: argparse.Namespace,
    pose: Sequence[float],
    mode: str,
    label: str,
    bridge_base: np.ndarray,
    bridge_tool: np.ndarray,
    planning_base_bridge_base: np.ndarray,
    bridge_tool_planning_tool: np.ndarray,
    start_joints_deg: Sequence[float],
) -> dict[str, Any]:
    """Plan a Cartesian target from an explicit offline joint start state.

    This mirrors Module 3's offline planner request.  It is used only to
    reject a GraspNet candidate before the arm moves to its pregrasp pose;
    execution still re-plans from the live state immediately before motion.
    """
    values = [float(value) for value in start_joints_deg]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{label} offline start must contain seven finite joint degrees")
    position, orientation = deployment_pose_to_planner_pose(
        pose, bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool,
        direct_grasp_center=bool(getattr(args, "planner_direct_grasp_center", False)),
        direct_pose=bool(getattr(args, "planner_direct_pose", False)),
    )
    print(
        f"-> Planner offline {mode.upper()} {label}: "
        f"start(deg)={np.round(values, 3).tolist()}, "
        f"frame={args.planner_base_frame}, xyz={np.round(position, 4).tolist()}"
    )
    def request_offline() -> dict[str, Any]:
        payload = {
            "start_state": {
                "joint_names": LEFT_JOINT_NAMES,
                "joint_values": values,
            },
            "waypoints": [{
                "name": "graspnet_offline_cartesian_target",
                "type": "cart",
                "group": args.planner_group,
                "planner": mode,
                "max_velocity_scaling_factor": float(args.planner_max_vel),
                "max_acceleration_scaling_factor": float(args.planner_max_acc),
                "frame_id": args.planner_base_frame,
                "ik_frame": args.planner_tool_frame,
                "position": position,
                "orientation": orientation,
            }],
            "topp": {
                "max_vel_factor": float(args.planner_max_vel),
                "max_acc_factor": float(args.planner_max_acc),
                "final_phase_ratio": 0.05,
                "final_phase_oversample": 2.0,
            },
        }
        return planner._trajectory.send_payload(
            _with_right_hand_tall_contact(payload), timeout=30.0
        )

    result = retry_task_creation(
        request_offline,
        label=f"GraspNet {label} {mode.upper()} 离线笛卡尔规划",
    )
    return validate_trajectory(result, label)


def plan_joint(planner: Any, args: argparse.Namespace, joints: Sequence[float], label: str) -> dict[str, Any]:
    values = [float(value) for value in joints]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{label} must contain seven finite joint values")
    print(f"-> Planner RRTStar {label}: joints(deg)={np.round(values, 3).tolist()}")
    result = retry_task_creation(
        lambda: planner._trajectory.send_payload(_with_right_hand_tall_contact({
            "waypoints": [{
                "name": f"graspnet_{label}_rrtstar",
                "type": "joint",
                "group": args.planner_group,
                "planner": "rrtstar",
                "max_velocity_scaling_factor": float(args.planner_max_vel),
                "max_acceleration_scaling_factor": float(args.planner_max_acc),
                "joint_names": LEFT_JOINT_NAMES,
                "joint_values": values,
            }],
            "topp": {
                "max_vel_factor": float(args.planner_max_vel),
                "max_acc_factor": float(args.planner_max_acc),
            },
        }), timeout=30.0),
        label=f"GraspNet {label} RRTSTAR 关节规划",
    )
    return validate_trajectory(result, label)


def preflight_candidate_plans(
    args: argparse.Namespace, candidate_plans: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plan candidate approach and descent poses without moving.

    Each entry must contain ``plan``, ``rank`` and
    ``parallel_jaw_symmetry_rotation_deg``.  The caller orders entries by
    GraspNet score after choosing the nearest 0/180-degree parallel-jaw
    equivalent for each grasp.  A candidate is retained only when both its
    live-state -> pregrasp plan and its offline pregrasp -> grasp descent plan
    are valid.  A failed planning request is candidate-local; a missing TF or
    Planner connection remains a hard fault.
    """
    _validate_planner_args(args)
    if not candidate_plans:
        return [], []
    bridge_base, bridge_tool, bridge_base_frame, bridge_tool_frame = load_bridge(args)
    reachable_items: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with connect_planner(args, "candidate reachability preflight") as planner:
        frame_ids = planner.list_frame_ids()
        resolve_planner_tool_frame(args, frame_ids)
        required = {
            args.planner_base_frame, args.planner_tool_frame,
            bridge_base_frame, bridge_tool_frame,
        }
        missing = sorted(required.difference(frame_ids))
        if missing:
            raise RuntimeError(f"Planner TF is missing required frames: {missing}")
        planning_base_bridge_base, bridge_tool_planning_tool = planner_target_transforms(
            planner, args, bridge_base_frame, bridge_tool_frame
        )
        for item in candidate_plans:
            rank = int(item["rank"])
            symmetry = float(item["parallel_jaw_symmetry_rotation_deg"])
            label = f"candidate rank {rank}, parallel-jaw symmetry {symmetry:.0f}deg pregrasp"
            print(f"-> Planner reachability check: {label}")
            phase = "pregrasp"
            try:
                trajectory = plan_cartesian(
                    planner, args, item["plan"]["pregrasp_eef_pose"],
                    args.planner_pregrasp_mode, label,
                    bridge_base, bridge_tool,
                    planning_base_bridge_base, bridge_tool_planning_tool,
                )
                pregrasp_end = trajectory["trajectory"][-1]
                phase = "offline_grasp_descent"
                descent = plan_cartesian_from_joint_state(
                    planner, args, item["plan"]["grasp_eef_pose"],
                    args.planner_descend_mode,
                    f"candidate rank {rank}, parallel-jaw symmetry {symmetry:.0f}deg offline grasp descent",
                    bridge_base, bridge_tool,
                    planning_base_bridge_base, bridge_tool_planning_tool,
                    pregrasp_end,
                )
            except Exception as exc:
                if not isinstance(exc, PlannerNoTrajectoryError) and not is_candidate_unreachable_error(exc):
                    raise
                message = str(exc)
                print(f"   unreachable: {message}")
                records.append({
                    "rank": rank,
                    "source_grasp_index": int(item["source_grasp_index"]),
                    "grasp_source": str(item["grasp_source"]),
                    "score": float(item["score"]),
                    "parallel_jaw_symmetry_rotation_deg": symmetry,
                    "reachable": False,
                    "phase": phase,
                    "reason": message,
                })
                continue
            records.append({
                "rank": rank,
                "source_grasp_index": int(item["source_grasp_index"]),
                "grasp_source": str(item["grasp_source"]),
                "score": float(item["score"]),
                "parallel_jaw_symmetry_rotation_deg": symmetry,
                "reachable": True,
                "pregrasp_waypoints": int(len(trajectory["trajectory"])),
                "offline_grasp_descent_waypoints": int(len(descent["trajectory"])),
                "offline_grasp_descent_start_joints_deg": pregrasp_end,
            })
            reachable_items.append(item)
            print("   pregrasp and offline grasp descent reachable; retained without robot motion")
    return reachable_items, records


def validate_trajectory(result: Any, label: str) -> dict[str, Any]:
    data = (result or {}).get("data") or {}
    trajectory = data.get("trajectory")
    arm_id = int(data.get("arm_id", -1))
    frequency = float(data.get("sample_frequency", 0.0))
    if arm_id != LEFT_ARM_ID:
        raise RuntimeError(f"Planner returned wrong arm_id for {label}: {result}")
    if not isinstance(trajectory, list) or not trajectory:
        raise PlannerNoTrajectoryError(
            f"Planner did not return a valid arm1 trajectory for {label}: {result}"
        )
    if not math.isfinite(frequency) or frequency <= 0:
        raise RuntimeError(f"Planner returned invalid sample_frequency for {label}: {frequency}")
    if any(len(point) != 7 or not all(math.isfinite(float(value)) for value in point) for point in trajectory):
        raise RuntimeError(f"Planner returned malformed joint trajectory for {label}")
    print(f"   planned {len(trajectory)} waypoints @ {frequency:.2f} Hz")
    return {"arm_id": arm_id, "trajectory": trajectory, "sample_frequency": frequency}


def execute_trajectory(args: argparse.Namespace, trajectory: dict[str, Any], label: str) -> None:
    if args.planner_plan_only:
        print(f"-> --planner-plan-only: {label} was not executed")
        return
    request = Request(
        args.planner_execute_url,
        data=json.dumps(trajectory).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    from tool.left_motion_right_servo_lock import left_arm_motion_guard
    try:
        # The endpoint returns only after the submitted trajectory completes.
        # Hold the shared lock for its physical execution, not for planning.
        with left_arm_motion_guard(label):
            with urlopen(request, timeout=300.0) as response:
                raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"Planner execution endpoint returned HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach Planner execution endpoint {args.planner_execute_url}") from exc
    try:
        outcome = json.loads(raw)
    except json.JSONDecodeError:
        outcome = {"raw_response": raw}
    if outcome.get("success") is False:
        raise RuntimeError(f"Planner trajectory execution failed for {label}: {outcome}")
    print(f"-> Planner trajectory completed: {label}")


def move_to_observation_with_planner(args: argparse.Namespace) -> bool:
    """Replace Version5's Arm.move_j observation motion with Planner RRTStar."""
    if not args.move_observation:
        print("-> observation motion skipped; ensure the D405 is already at the calibrated observation pose")
        return True
    _validate_planner_args(args)
    print("-> D405 observation motion: confirmation bypassed by Version7 flow")
    with connect_planner(args, "D405 observation planning") as planner:
        trajectory = plan_joint(
            planner, args, args.observation_joints_deg, "D405 observation pose"
        )
        if args.planner_plan_only:
            print(
                "-> --planner-plan-only: observation trajectory was not executed; "
                "capture is stopped because the camera pose is not guaranteed."
            )
            return False
        execute_trajectory(args, trajectory, "D405 observation pose")
    return True


def check_pregrasp_distance(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    """Reject an unexpectedly distant Cartesian grasp before opening the gripper.

    Legacy plans target SDK ``eef_pose`` and are compared with the SDK state.
    Direct-center plans target Planner ``left_eef`` instead, so compare both
    target and live tool position in the Planner base frame.  Mixing those two
    tool frames would bias this safety distance by the flange-to-gripper offset.
    """
    maximum = float(args.max_grasp_distance_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("--max-grasp-distance-m must be a positive finite value")
    target = np.asarray(plan.get("pregrasp_eef_pose", [])[:3], dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise RuntimeError("grasp plan has no finite pregrasp EEF xyz")

    if bool(getattr(args, "planner_direct_grasp_center", False)):
        bridge_base, bridge_tool, bridge_base_frame, bridge_tool_frame = load_bridge(args)
        with connect_planner(args, "post-visualization pregrasp safety check") as planner:
            frame_ids = planner.list_frame_ids()
            resolve_planner_tool_frame(args, frame_ids)
            required = {
                args.planner_base_frame,
                args.planner_tool_frame,
                bridge_base_frame,
                bridge_tool_frame,
            }
            missing = sorted(required.difference(frame_ids))
            if missing:
                raise RuntimeError(f"Planner TF is missing required frames: {missing}")
            planning_base_bridge_base, bridge_tool_planning_tool = planner_target_transforms(
                planner, args, bridge_base_frame, bridge_tool_frame
            )
            target_position, _ = deployment_pose_to_planner_pose(
                plan["pregrasp_eef_pose"],
                bridge_base,
                bridge_tool,
                planning_base_bridge_base,
                bridge_tool_planning_tool,
                direct_grasp_center=True,
                direct_pose=bool(getattr(args, "planner_direct_pose", False)),
            )
            current_tool = _planner_lookup_target_source(
                planner, args.planner_tool_frame, args.planner_base_frame
            )[:3, 3]
        distance = float(np.linalg.norm(np.asarray(target_position) - current_tool))
        print(
            f"-> pregrasp distance for {args.planner_tool_frame} in "
            f"{args.planner_base_frame}: {distance:.3f} m (limit {maximum:.3f} m)"
        )
        if distance > maximum:
            calibration_hint = (
                "the direct Planner wrist calibration and selected grasp"
                if bool(getattr(args, "planner_direct_pose", False))
                else "the bridge calibration and selected grasp"
            )
            raise PregraspDistanceRejected(
                "pregrasp target exceeds --max-grasp-distance-m; verify observation pose, "
                f"camera calibration, {calibration_hint} before retrying"
            )
        return

    from parm_sdk import Arm

    with Arm(host=args.arm_host) as arm:
        state = arm.get_state() or {}
    arm_state = (state.get("state") or {}).get("left") or {}
    if str(state.get("system_state", "")).lower() != "running" or arm_state.get("enabled") is not True:
        raise RuntimeError("arm1 is not RUNNING and enabled")
    current = np.asarray((arm_state.get("eef_pose") or {}).get("xyz"), dtype=np.float64)
    if current.shape != (3,) or not np.all(np.isfinite(current)):
        raise RuntimeError("arm state does not contain a finite arm1 EEF xyz")
    distance = float(np.linalg.norm(target - current))
    print(f"-> pregrasp distance in arm1_base: {distance:.3f} m (limit {maximum:.3f} m)")
    if distance > maximum:
        raise PregraspDistanceRejected(
            "pregrasp target exceeds --max-grasp-distance-m; verify observation pose, "
            "camera calibration, and selected grasp before retrying"
        )


def execute_grasp_with_planner(args: argparse.Namespace, plan: dict[str, Any]) -> bool:
    """Version5 execution replacement; detection and plan creation remain untouched."""
    if not (args.execute_grasp and args.confirm_real_grasp):
        print("-> grasp plan is dry-run; add --execute-grasp --confirm-real-grasp to enable Planner execution")
        return False
    _validate_planner_args(args)
    try:
        check_pregrasp_distance(args, plan)
    except PregraspDistanceRejected as exc:
        raise CandidatePlanRejected("pregrasp_distance", exc) from exc
    bridge_base, bridge_tool, bridge_base_frame, bridge_tool_frame = load_bridge(args)
    auto_confirm = args.auto_confirm_real_grasp
    if auto_confirm is None:
        # Keep the existing loop environment variable as a backward-compatible
        # fallback. The CLI flag always wins, including its explicit --no form.
        auto_confirm = os.environ.get("LOOP_AUTO_CONFIRM_GRASPNET_EXECUTION") == "1"
    if auto_confirm:
        answer = "y"
        print(
            "Open3D preview is closed. "
            "--auto-confirm-real-grasp 已自动确认真实抓取。"
        )
    else:
        answer = input("Open3D preview is closed. 输入 y 通过 Planner 执行真实抓取，其他输入取消：").strip().lower()
    if answer not in {"y", "yes"}:
        print(
            "-> Planner grasp execution declined; no gripper or trajectory command was sent. "
            "Returning to observation for fresh inference."
        )
        raise RetryGrasp("operator_declined")
    weight_api = None
    sensor_context: Any = nullcontext(None)
    if args.grasp_weight_check:
        from parm_sdk import ForceSensor

        weight_api = load_weight_check_api(args)
        sensor_context = ForceSensor(
            host=args.force_sensor_host,
            hand=args.force_sensor_hand,
        )

    with connect_planner(args, "fresh grasp replanning") as planner, sensor_context as force_sensor:
        frame_ids = planner.list_frame_ids()
        resolve_planner_tool_frame(args, frame_ids)
        required = {
            args.planner_base_frame, args.planner_tool_frame,
            bridge_base_frame, bridge_tool_frame,
        }
        missing = sorted(required.difference(frame_ids))
        if missing:
            raise RuntimeError(f"Planner TF is missing required frames: {missing}")
        planning_base_bridge_base, bridge_tool_planning_tool = planner_target_transforms(
            planner, args, bridge_base_frame, bridge_tool_frame
        )
        if args.planner_plan_only:
            plan_cartesian(planner, args, plan["pregrasp_eef_pose"], args.planner_pregrasp_mode, "pregrasp", bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool)
            plan_cartesian(planner, args, plan["grasp_eef_pose"], args.planner_descend_mode, "vertical grasp descent", bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool)
            print("-> --planner-plan-only: no gripper or trajectory command was sent")
            return False
        try:
            pregrasp_trajectory = plan_cartesian(
                planner, args, plan["pregrasp_eef_pose"], args.planner_pregrasp_mode,
                "pregrasp", bridge_base, bridge_tool,
                planning_base_bridge_base, bridge_tool_planning_tool,
            )
        except Exception as exc:
            if not isinstance(exc, PlannerNoTrajectoryError) and not is_candidate_unreachable_error(exc):
                raise
            raise CandidatePlanRejected("pregrasp", exc) from exc

        baseline = None
        weight_thresholds = None
        if weight_api is not None:
            if args.weight_check_samples < 3:
                raise RuntimeError("--weight-check-samples must be at least 3")
            if args.weight_check_timeout_s <= 0 or args.weight_check_settle_s < 0:
                raise RuntimeError(
                    "--weight-check-timeout-s must be positive and "
                    "--weight-check-settle-s cannot be negative"
                )
            _, threshold_type, collect_force_window, _ = weight_api
            weight_thresholds = build_weight_thresholds(args, threshold_type)
            print(
                "-> grasp weight thresholds: "
                f"EMPTY<= {weight_thresholds.empty_max_g:.1f} g, "
                f"SINGLE={weight_thresholds.single_min_g:.1f}.."
                f"{weight_thresholds.single_max_g:.1f} g, "
                f"MULTIPLE>= {weight_thresholds.multiple_min_g:.1f} g, "
                f"max MAD={weight_thresholds.max_axis_mad_n:.3f} N"
            )
            try:
                force_sensor.wait_first(timeout=float(args.weight_check_timeout_s))
            except Exception as exc:
                raise ForceSensorDataError("stream", exc) from exc

        # Record the empty reference before descent.  The loaded sample must be
        # collected after returning to this exact pose; raw sensor-frame force
        # vectors cannot be compared across different wrist attitudes.
        permit_value = os.environ.get("LOOP_GRASPNET_EXECUTION_PERMIT_FILE")
        if permit_value:
            permit_path = Path(permit_value)
            print(
                "-> 抓取点预测和预抓取规划已完成；等待总控状态=0后执行真实抓取："
                f"{permit_path}",
                flush=True,
            )
            while True:
                if permit_path.is_file():
                    content = permit_path.read_text(encoding="utf-8").strip().lower()
                    if content == "allow":
                        print("-> 总控已放行；开始执行预测出的预抓取轨迹。", flush=True)
                        break
                    raise RuntimeError(
                        f"invalid GraspNet execution permit content in {permit_path}: "
                        f"{content!r}"
                    )
                time.sleep(0.2)
        move_gripper(args, float(plan["opening_percent"]), "set planned opening")
        execute_trajectory(args, pregrasp_trajectory, "pregrasp")
        if weight_api is not None:
            time.sleep(float(args.weight_check_settle_s))
            try:
                baseline = collect_force_window(
                    force_sensor,
                    sample_count=int(args.weight_check_samples),
                    timeout_s=float(args.weight_check_timeout_s),
                )
            except Exception as exc:
                raise ForceSensorDataError("baseline", exc) from exc
            print(
                "-> open-gripper force baseline at pregrasp: "
                f"median={[round(float(value), 4) for value in baseline.median_force_n]} N, "
                f"MAD={[round(float(value), 4) for value in baseline.axis_mad_n]} N"
            )
        try:
            grasp_trajectory = plan_cartesian(
                planner, args, plan["grasp_eef_pose"], args.planner_descend_mode,
                "vertical grasp descent", bridge_base, bridge_tool,
                planning_base_bridge_base, bridge_tool_planning_tool,
            )
        except Exception as exc:
            if not isinstance(exc, PlannerNoTrajectoryError) and not is_candidate_unreachable_error(exc):
                raise
            raise CandidatePlanRejected("grasp_descent", exc) from exc
        execute_trajectory(args, grasp_trajectory, "grasp descent")
        close_gripper(args)
        if weight_api is not None:
            execute_trajectory(
                args,
                plan_cartesian(
                    planner,
                    args,
                    plan["pregrasp_eef_pose"],
                    "lin",
                    "short post-grasp lift for weight check",
                    bridge_base,
                    bridge_tool,
                    planning_base_bridge_base,
                    bridge_tool_planning_tool,
                ),
                "short post-grasp lift for weight check",
            )
            grasp_class, _, _, verify_grasp_after_lift = weight_api
            try:
                result = verify_grasp_after_lift(
                    force_sensor,
                    baseline,
                    weight_thresholds,
                    settle_s=float(args.weight_check_settle_s),
                    sample_count=int(args.weight_check_samples),
                    timeout_s=float(args.weight_check_timeout_s),
                )
            except Exception as exc:
                print(
                    "-> force data unavailable for same-pose weight check; releasing and "
                    "starting a fresh GraspNet loop"
                )
                move_gripper(args, 100.0, "release after same-pose weight-check failure")
                raise RetryGrasp("UNKNOWN") from exc
            result = apply_contiguous_single_band(
                result,
                weight_thresholds,
                grasp_class,
                bool(getattr(args, "weight_contiguous_single_band", False)),
            )
            print_weight_result(result)
            if result.grasp_class is not grasp_class.SINGLE:
                print(
                    f"-> same-pose weight check rejected {result.grasp_class.value}; releasing and "
                    "starting a fresh GraspNet loop"
                )
                move_gripper(args, 100.0, "release rejected grasp at pregrasp")
                raise RetryGrasp(result.grasp_class.value, result)
            args.verified_weight_handoff = {
                "classification": result.grasp_class.value,
                "mass_g": float(result.mass_g),
                "baseline": {
                    "median_force_n": [float(value) for value in result.baseline.median_force_n],
                    "axis_mad_n": [float(value) for value in result.baseline.axis_mad_n],
                    "sample_count": int(result.baseline.sample_count),
                },
                "loaded_at_pregrasp": {
                    "median_force_n": [float(value) for value in result.loaded.median_force_n],
                    "axis_mad_n": [float(value) for value in result.loaded.axis_mad_n],
                    "sample_count": int(result.loaded.sample_count),
                },
                "thresholds": {
                    "empty_max_g": float(weight_thresholds.empty_max_g),
                    "single_min_g": float(weight_thresholds.single_min_g),
                    "single_max_g": float(weight_thresholds.single_max_g),
                    "multiple_min_g": float(weight_thresholds.multiple_min_g),
                    "max_axis_mad_n": float(weight_thresholds.max_axis_mad_n),
                    "gravity_m_s2": float(weight_thresholds.gravity_m_s2),
                },
                "force_sensor_host": str(args.force_sensor_host),
                "force_sensor_hand": int(args.force_sensor_hand),
                "sample_count": int(args.weight_check_samples),
                "timeout_s": float(args.weight_check_timeout_s),
                "settle_s": float(args.weight_check_settle_s),
            }
            print(
                "-> exactly one object verified at the same pregrasp pose; "
                "continuing to graspup"
            )
        # The object has now been lifted off the table and classified at the
        # same wrist pose as the empty baseline.  Move to the configured
        # handoff pose only after that classification succeeds.
        execute_trajectory(
            args,
            plan_joint(
                planner,
                args,
                args.postgrasp_lift_joints_deg,
                "post-grasp lift",
            ),
            "post-grasp lift",
        )
        require_left_gripper_holding(args, phase="graspup")
        if weight_api is not None:
            _, _, collect_force_window, _ = weight_api
            try:
                time.sleep(float(args.weight_check_settle_s))
                graspup_loaded = collect_force_window(
                    force_sensor,
                    sample_count=int(args.weight_check_samples),
                    timeout_s=float(args.weight_check_timeout_s),
                )
            except Exception as exc:
                print(
                    "-> force data unavailable at graspup; releasing and "
                    "starting a fresh GraspNet loop"
                )
                move_gripper(args, 100.0, "release after graspup force-signature failure")
                raise RetryGrasp("UNKNOWN") from exc
            if graspup_loaded.max_axis_mad_n > weight_thresholds.max_axis_mad_n:
                print(
                    "-> unstable force data at graspup; releasing and "
                    "starting a fresh GraspNet loop"
                )
                move_gripper(args, 100.0, "release after unstable graspup force signature")
                raise RetryGrasp("UNKNOWN")
            max_delta_n = max(
                0.20,
                0.5
                * float(result.mass_g)
                * float(weight_thresholds.gravity_m_s2)
                / 1000.0,
            )
            graspup_load_signature = {
                "median_force_n": [
                    float(value) for value in graspup_loaded.median_force_n
                ],
                "axis_mad_n": [
                    float(value) for value in graspup_loaded.axis_mad_n
                ],
                "sample_count": int(graspup_loaded.sample_count),
                "max_force_delta_n": float(max_delta_n),
                "max_axis_mad_n": float(weight_thresholds.max_axis_mad_n),
            }
            args.verified_weight_handoff["graspup_load_signature"] = (
                graspup_load_signature
            )
            # Retain the transitional field for existing logs and readers.
            args.verified_weight_handoff["loaded_at_graspup"] = {
                **graspup_load_signature,
                "mass_g": float(result.mass_g),
                "classification": result.grasp_class.value,
            }
            print(
                "-> graspup held-object force signature: "
                f"median={[round(float(value), 4) for value in graspup_loaded.median_force_n]} N, "
                f"max allowed delta={max_delta_n:.3f} N"
            )
        if bool(getattr(args, "stop_after_graspup", False)):
            print(
                "-> Version7 stop-after-graspup: object remains clamped at the post-grasp/graspup pose; "
                "placement and gripper release are skipped for Module 2 inspection"
            )
            return True
        execute_trajectory(args, plan_joint(planner, args, args.place_joints_deg, "placement"), "placement")
        move_gripper(args, 100.0, "open")
        print("-> object released at Planner placement pose")
        return False


def preview_pregrasp_with_planner(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if not args.confirm_real_pregrasp:
        raise RuntimeError("--preview-pregrasp requires --confirm-real-pregrasp")
    _validate_planner_args(args)
    bridge_base, bridge_tool, bridge_base_frame, bridge_tool_frame = load_bridge(args)
    if os.environ.get("LOOP_AUTO_CONFIRM_MOTION") == "1":
        answer = "y"
        print("自动确认通过 Planner 移动到预抓取位姿。")
    else:
        answer = input("输入 y 仅通过 Planner 移动到预抓取位姿，其他输入取消：").strip().lower()
    if answer not in {"y", "yes"}:
        print("-> Planner pregrasp preview cancelled; no robot or gripper command was sent")
        return
    with connect_planner(args, "pregrasp preview replanning") as planner:
        frame_ids = planner.list_frame_ids()
        resolve_planner_tool_frame(args, frame_ids)
        required = {
            args.planner_base_frame, args.planner_tool_frame,
            bridge_base_frame, bridge_tool_frame,
        }
        missing = sorted(required.difference(frame_ids))
        if missing:
            raise RuntimeError(f"Planner TF is missing required frames: {missing}")
        planning_base_bridge_base, bridge_tool_planning_tool = planner_target_transforms(
            planner, args, bridge_base_frame, bridge_tool_frame
        )
        trajectory = plan_cartesian(planner, args, plan["pregrasp_eef_pose"], args.planner_pregrasp_mode, "pregrasp preview", bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool)
        if args.planner_plan_only:
            print("-> --planner-plan-only: no gripper or trajectory command was sent")
            return
        move_gripper(args, float(plan["opening_percent"]), "set preview opening")
        execute_trajectory(args, trajectory, "pregrasp preview")


_VERSION5_BUILD_PARSER = version5.build_parser
version5.build_parser = build_parser
version5.move_to_observation = move_to_observation_with_planner
version5.execute_grasp = execute_grasp_with_planner
version5.preview_pregrasp = preview_pregrasp_with_planner


def main() -> None:
    attempt = 1
    while True:
        print(f"\n=== traditional GraspNet attempt {attempt} ===")
        try:
            version5.main()
        except RetryGrasp as exc:
            attempt += 1
            print(f"-> retry requested after weight classification {exc}")
            continue
        return


if __name__ == "__main__":
    main()
