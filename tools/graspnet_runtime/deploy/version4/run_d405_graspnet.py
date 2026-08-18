#!/usr/bin/env python3
"""Version 4: grasp with arm1 wrist D405, then lift, place, and release."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEPLOY_ROOT = ROOT / "deploy"
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))
from planner_motion import DEFAULT_EXECUTE_URL, DEFAULT_PLANNER_HOST, plan_joint_rrtstar, plan_legacy_axis_angle_pose
VERSION1 = ROOT / "deploy" / "version1" / "run_dual_camera_inference.py"
DEFAULT_WRIST_CALIBRATION = ROOT / "deploy" / "calibration" / "output" / "wrist1" / "calibration_wrist1.json"
DEFAULT_CHECKPOINT = ROOT / "checkpoint" / "checkpoint-rs.tar"
DEFAULT_TCP_TRANSLATION = ROOT / "deploy" / "calibration" / "output" / "tcp_translation2" / "calibration_translation2.json"
DEFAULT_OBSERVATION_JOINTS_DEG = [ -6.25,-21.348, 12.599, 78.268,   4.039,  56.382, -11.374]
# After closing on the object, first move to this known raised configuration,
# then move to the placement configuration and open the gripper.
DEFAULT_POSTGRASP_LIFT_JOINTS_DEG = [
    -6.288,
    -38.805,
    12.556,
    72.427,
    4.059,
    67.331,
    -11.391,
]
DEFAULT_PLACE_JOINTS_DEG = [
    -62.016,
    37.312,
    -23.766,
    55.214,
    1.200,
    -0.677,
    78.087,
]
DEFAULT_LINEAR_SPEED_MPS = 0.05
DEFAULT_ANGULAR_SPEED_RADPS = 0.30
DEFAULT_MAX_LINEAR_ACCEL_MPS2 = 0.10
DEFAULT_MAX_ANGULAR_ACCEL_RADPS2 = 0.60


def build_parser() -> argparse.ArgumentParser:
    default_display = os.environ.get("DISPLAY")
    if not default_display and Path("/tmp/.X11-unix/X1").exists():
        default_display = ":1"
    parser = argparse.ArgumentParser(
        description="Capture arm1 wrist D405 and run calibrated GraspNet inference only."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--wrist-calibration", default=str(DEFAULT_WRIST_CALIBRATION))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--arm-id", type=int, default=1, choices=(1,))
    parser.add_argument("--planner-host", default=DEFAULT_PLANNER_HOST)
    parser.add_argument("--execute-url", default=DEFAULT_EXECUTE_URL)
    parser.add_argument("--d405-serial", default="260322270600")
    parser.add_argument("--d405-frame-count", type=int, default=5, help="D405 frames fused by per-pixel median")
    parser.add_argument("--d405-postprocess", dest="d405_postprocess", action="store_true", default=True)
    parser.add_argument("--no-d405-postprocess", dest="d405_postprocess", action="store_false")
    parser.add_argument("--d405-hole-filling", action="store_true")
    parser.add_argument("--d405-z-min", type=float, default=0.10)
    parser.add_argument("--d405-z-max", type=float, default=1.20)
    parser.add_argument("--d405-x-abs-max", type=float, default=0.80)
    parser.add_argument("--d405-y-abs-max", type=float, default=0.80)
    parser.add_argument(
        "--collision-thresh",
        type=float,
        default=0.01,
        help="GraspNet collision threshold; set 0 to inspect unfiltered candidates",
    )
    parser.add_argument(
        "--collision-voxel-size",
        "--voxel-size",
        dest="collision_voxel_size",
        type=float,
        default=0.01,
        help="Collision point-cloud voxel size in meters (default: 0.01)",
    )
    parser.add_argument(
        "--collision-approach-dist",
        "--approach-dist",
        dest="collision_approach_dist",
        type=float,
        default=0.05,
        help=("collision-check gripper approach distance in meters (default: 0.05); "
              "set <= 0 to disable only the swept approach volume while keeping finger/bottom checks"),
    )
    parser.add_argument(
        "--collision-table-thickness",
        type=float,
        default=0.0,
        help="virtual tabletop thickness used only by collision checking (meters)",
    )
    parser.add_argument("--collision-table-plane-dist", type=float, default=0.005)
    parser.add_argument("--collision-table-min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--collision-table-min-normal-z", type=float, default=0.5)
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Maximum number of grasp poses shown in Open3D (default: 50)",
    )
    parser.add_argument(
        "--table-clearance-m",
        type=float,
        default=0.015,
        help="reject grasp centers closer than this height above the automatically estimated tabletop",
    )
    parser.add_argument(
        "--disable-tabletop-filter",
        action="store_true",
        help="do not reject candidates using the estimated tabletop plane",
    )
    parser.add_argument(
        "--table-height-percentile",
        type=float,
        default=5.0,
        help="low Z percentile of the transformed D405 cloud used as the horizontal tabletop estimate",
    )
    parser.add_argument(
        "--table-xy-bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="optional tabletop rectangle in arm1_base meters; rejects grasps whose jaw endpoints leave it",
    )
    parser.add_argument(
        "--table-edge-margin-m",
        type=float,
        default=0.02,
        help="inset applied to --table-xy-bounds before accepting a jaw endpoint",
    )
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="aligned D405 color/depth ROI used for GraspNet point-cloud sampling",
    )
    parser.add_argument("--move-observation", dest="move_observation", action="store_true", default=True)
    parser.add_argument(
        "--skip-observation-move",
        dest="move_observation",
        action="store_false",
        help="do not move to the fixed D405 observation pose (diagnostic use only)",
    )
    parser.add_argument("--movej-speed", type=float, default=0.1)
    parser.add_argument(
        "--postgrasp-lift-joints-deg",
        type=float,
        nargs=7,
        default=DEFAULT_POSTGRASP_LIFT_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="joint configuration used to raise the grasped object before placement",
    )
    parser.add_argument(
        "--place-joints-deg",
        type=float,
        nargs=7,
        default=DEFAULT_PLACE_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="joint configuration at which the gripper opens to release the object",
    )
    parser.add_argument(
        "--postgrasp-movej-speed",
        type=float,
        default=0.1,
        help="speed used for the lift and placement move_j commands",
    )
    parser.add_argument("--tcp-translation-calibration", default=str(DEFAULT_TCP_TRANSLATION))
    parser.add_argument("--gripper-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--pregrasp-height-m", type=float, default=0.05)
    parser.add_argument("--gripper-host", default="localhost")
    parser.add_argument(
        "--gripper-hand",
        type=int,
        default=1,
        choices=(1, 2),
        help="SDK hand index; arm1 is the left arm and uses hand 1",
    )
    parser.add_argument(
        "--gripper-open-width-m",
        type=float,
        default=0.078,
        help="physical jaw width represented by Gripper position=100 percent",
    )
    parser.add_argument("--gripper-width-margin-m", type=float, default=0.005)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    parser.add_argument(
        "--gripper-close-force",
        type=float,
        default=100.0,
        help="closing force percentage; separate from the opening/planned-width force (default: 100)",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=80.0,
        help="left gripper opening/closing speed percentage (default: 80)",
    )
    parser.add_argument("--gripper-command-retries", type=int, default=3)
    parser.add_argument("--gripper-retry-delay-s", type=float, default=1.0)
    parser.add_argument("--auto-open-gripper", dest="auto_open_gripper", action="store_true", default=True)
    parser.add_argument("--no-auto-open-gripper", dest="auto_open_gripper", action="store_false")
    parser.add_argument("--pregrasp-duration-s", type=float, default=3.0)
    parser.add_argument("--descend-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--cartesian-speed-factor",
        type=float,
        default=1.0,
        help="custom move_to_pose speed scale; velocity scales by factor and acceleration by factor squared",
    )
    parser.add_argument("--servo-hz", type=float, default=50.0, choices=(10.0, 20.0, 25.0, 50.0, 100.0))
    parser.add_argument("--pose-settle-timeout-s", type=float, default=5.0)
    parser.add_argument("--pose-position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--pose-orientation-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--max-grasp-distance-m", type=float, default=0.80)
    parser.add_argument(
        "--preview-pregrasp",
        action="store_true",
        help="move only to the corrected pregrasp pose after visualization",
    )
    parser.add_argument("--execute-grasp", action="store_true")
    parser.add_argument("--confirm-real-grasp", action="store_true")
    parser.add_argument(
        "--observation-joints-deg",
        type=float,
        nargs=7,
        default=DEFAULT_OBSERVATION_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument(
        "--vis",
        dest="vis",
        action="store_true",
        default=True,
        help="open the official Open3D point-cloud and grasp visualization (default: on)",
    )
    parser.add_argument(
        "--no-vis",
        dest="vis",
        action="store_false",
        help="do not open the Open3D point-cloud and grasp visualization",
    )
    parser.add_argument(
        "--display",
        default=default_display,
        help="X11 display used for Open3D visualization (default: current DISPLAY or :1)",
    )
    parser.add_argument(
        "--save-visualization",
        dest="save_visualization",
        action="store_true",
        default=True,
        help="save a color-image grasp overlay after inference (default: on)",
    )
    parser.add_argument(
        "--no-save-visualization",
        dest="save_visualization",
        action="store_false",
        help="do not save the color-image grasp overlay",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="also open a GUI window for the saved grasp overlay",
    )
    return parser


def load_wrist_calibration(path: Path) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"wrist calibration does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("T_eef_wrist1"), dtype=np.float64)
    if matrix.size == 0:
        raise RuntimeError(f"{path} does not contain T_eef_wrist1")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"invalid T_eef_wrist1 in {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"invalid homogeneous bottom row in T_eef_wrist1: {path}")
    return matrix


def choose_nearest_equivalent(
    target: float, current: float, *, lower: float | None = None, upper: float | None = None
) -> float:
    """Choose the nearest equivalent target, optionally constrained to command limits."""
    candidates = [target + 360.0 * turns for turns in range(-2, 3)]
    if lower is not None or upper is not None:
        candidates = [
            value
            for value in candidates
            if (lower is None or value >= lower) and (upper is None or value <= upper)
        ]
    if not candidates:
        raise RuntimeError("目标不存在控制器范围内的 360 度等效角")
    return min(candidates, key=lambda value: abs(value - current))


def move_to_observation(args: argparse.Namespace) -> bool:
    if not args.move_observation:
        print("-> observation move skipped; using current arm1 pose")
        return True
    joints = np.asarray(args.observation_joints_deg, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError("observation joints must contain 7 finite values")
    if not np.isfinite(args.movej_speed) or args.movej_speed <= 0:
        raise RuntimeError("--movej-speed must be a positive finite value")
    print("-> moving arm1 to D405 observation pose")
    print("   joints(deg):", [round(float(v), 3) for v in joints])
    plan_joint_rrtstar(joints.tolist(), planner_host=args.planner_host, speed=float(args.movej_speed),
                   execute_url=args.execute_url, execute=True)
    print("-> observation move completed")
    return True


def move_to_joint_configuration(
    args: argparse.Namespace, joints_deg: list[float] | np.ndarray, label: str
) -> None:
    """Move arm1 to a checked joint configuration, choosing safe 360-degree equivalents."""
    joints = np.asarray(joints_deg, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"{label} joints must contain 7 finite values")
    if not np.isfinite(args.postgrasp_movej_speed) or args.postgrasp_movej_speed <= 0:
        raise RuntimeError("--postgrasp-movej-speed must be a positive finite value")
    print(f"-> moving arm1 to {label}")
    print("   requested joints(deg):", [round(float(value), 3) for value in joints])
    plan_joint_rrtstar(joints.tolist(), planner_host=args.planner_host,
                   speed=float(args.postgrasp_movej_speed), execute_url=args.execute_url, execute=True)
    print(f"-> {label} completed")


def run_inference(args: argparse.Namespace, output_dir: Path) -> Path:
    cmd = [
        args.python_bin,
        str(VERSION1),
        "--camera",
        "d405",
        "--apply_calibration",
        "--checkpoint_path",
        str(args.checkpoint_path),
        "--num_point",
        str(args.num_point),
        "--num_view",
        str(args.num_view),
        "--wrist_calibration",
        str(args.wrist_calibration),
        "--arm_host",
        str(args.arm_host),
        "--arm_id",
        str(args.arm_id),
        "--d405_serial",
        str(args.d405_serial),
        "--d405_frame_count",
        str(args.d405_frame_count),
        "--d405_z_min",
        str(args.d405_z_min),
        "--d405_z_max",
        str(args.d405_z_max),
        "--d405_x_abs_max",
        str(args.d405_x_abs_max),
        "--d405_y_abs_max",
        str(args.d405_y_abs_max),
        "--collision_thresh",
        str(args.collision_thresh),
        "--voxel_size",
        str(args.collision_voxel_size),
        "--approach_dist",
        str(args.collision_approach_dist),
        "--collision_table_thickness",
        str(args.collision_table_thickness),
        "--collision_table_plane_dist",
        str(args.collision_table_plane_dist),
        "--collision_table_min_inlier_ratio",
        str(args.collision_table_min_inlier_ratio),
        "--collision_table_min_normal_z",
        str(args.collision_table_min_normal_z),
        "--top_k",
        str(args.top_k),
        "--output_dir",
        str(output_dir),
    ]
    if args.d405_postprocess:
        cmd.append("--d405_postprocess")
    if args.d405_hole_filling:
        cmd.append("--d405_hole_filling")
    env = os.environ.copy()
    if args.roi is not None:
        cmd.extend(["--roi", *(str(value) for value in args.roi)])
    print("-> D405 GraspNet:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    runs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(
            f"expected exactly one inference run under {output_dir}, found {len(runs)}"
        )
    if not (runs[0] / "d405" / "meta.json").exists():
        raise RuntimeError(f"inference output is incomplete: {runs[0]}")
    if not (runs[0] / "d405" / "grasps.npy").exists():
        raise RuntimeError(f"inference output is incomplete: {runs[0]}")
    return runs[0]


def grasp_result(run_dir: Path, grasp: np.ndarray) -> dict:
    return {
        "coordinate_frame": "arm1_base",
        "score": float(grasp[0]),
        "width_m": float(grasp[1]),
        "height_m": float(grasp[2]),
        "depth_m": float(grasp[3]),
        "rotation_matrix": grasp[4:13].reshape(3, 3).tolist(),
        "translation_m": grasp[13:16].tolist(),
        "run_dir": str(run_dir),
    }


def table_safe_grasp_indices(args: argparse.Namespace, run_dir: Path, grasps: np.ndarray) -> np.ndarray:
    """Keep grasps above the tabletop and, when configured, inside its XY footprint."""
    if grasps.ndim != 2 or grasps.shape[1] < 16:
        return np.empty(0, dtype=np.int64)
    if args.disable_tabletop_filter:
        indices = np.arange(len(grasps), dtype=np.int64)
        (run_dir / "tabletop_filter.json").write_text(
            json.dumps({"enabled": False, "input_grasps": int(len(grasps)), "kept": int(len(grasps))}, indent=2),
            encoding="utf-8",
        )
        print(f"[d405] tabletop filter disabled: retained={len(indices)}")
        return indices
    if args.table_clearance_m < 0 or not 0.0 <= args.table_height_percentile <= 100.0:
        raise RuntimeError("table clearance must be >= 0 and table height percentile must be in [0, 100]")

    grasp_dir = run_dir / "d405"
    meta = json.loads((grasp_dir / "meta.json").read_text(encoding="utf-8"))
    cloud_camera = np.asarray(np.load(grasp_dir / "point_cloud.npy"), dtype=np.float64)
    transform = np.asarray(meta.get("T_base_camera"), dtype=np.float64)
    if cloud_camera.ndim != 2 or cloud_camera.shape[1] != 3 or transform.shape != (4, 4):
        raise RuntimeError("cannot estimate tabletop: D405 point cloud or T_base_camera is unavailable")
    cloud_base = cloud_camera @ transform[:3, :3].T + transform[:3, 3]
    table_z = float(np.percentile(cloud_base[:, 2], args.table_height_percentile))
    mask = grasps[:, 15] >= table_z + float(args.table_clearance_m)
    height_kept = int(np.count_nonzero(mask))

    if args.table_xy_bounds is not None:
        x_min, x_max, y_min, y_max = (float(value) for value in args.table_xy_bounds)
        margin = float(args.table_edge_margin_m)
        if x_min >= x_max or y_min >= y_max or margin < 0:
            raise RuntimeError("invalid --table-xy-bounds or --table-edge-margin-m")
        # Check both jaw tips, not only the predicted center, so a wide grasp
        # cannot hook the tabletop edge while its center remains in bounds.
        opening_axis = grasps[:, 4:13].reshape(-1, 3, 3)[:, :, 1]
        half_opening = grasps[:, 1:2] * 0.5
        centers_xy = grasps[:, 13:15]
        endpoint_offsets_xy = opening_axis[:, :2] * half_opening
        endpoints_xy = np.stack((centers_xy - endpoint_offsets_xy, centers_xy + endpoint_offsets_xy), axis=1)
        in_bounds = (
            (endpoints_xy[:, :, 0] >= x_min + margin)
            & (endpoints_xy[:, :, 0] <= x_max - margin)
            & (endpoints_xy[:, :, 1] >= y_min + margin)
            & (endpoints_xy[:, :, 1] <= y_max - margin)
        ).all(axis=1)
        mask &= in_bounds

    indices = np.flatnonzero(mask).astype(np.int64)
    report = {
        "estimated_table_z_base_m": table_z,
        "height_percentile": float(args.table_height_percentile),
        "minimum_center_height_above_table_m": float(args.table_clearance_m),
        "table_xy_bounds_base_m": args.table_xy_bounds,
        "table_edge_margin_m": float(args.table_edge_margin_m),
        "input_grasps": int(len(grasps)),
        "kept_after_height": height_kept,
        "kept": int(len(indices)),
    }
    (run_dir / "tabletop_filter.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "[d405] tabletop filter: estimated_z={:+.4f}m clearance={:.3f}m candidates={} "
        "height_kept={} retained={}".format(
            table_z, args.table_clearance_m, len(grasps), height_kept, len(indices)
        )
    )
    if args.table_xy_bounds is not None:
        print("[d405] tabletop XY bounds enabled:", [round(float(value), 4) for value in args.table_xy_bounds])
    return indices


def summarize_top_grasp(run_dir: Path, grasps: np.ndarray) -> dict | None:
    grasp_dir = run_dir / "d405"
    meta = json.loads((grasp_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("coordinate_frame") != "arm1_base":
        raise RuntimeError("D405 output is not in arm1_base coordinates")
    if grasps.ndim != 2 or grasps.shape[0] == 0 or grasps.shape[1] < 16:
        print("[d405] no valid grasps survived filtering; no top_grasp.json was created")
        return None
    result = grasp_result(run_dir, grasps[int(np.argmax(grasps[:, 0]))])
    (run_dir / "top_grasp.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "[d405] top grasp: score={:.4f} width={:.4f} center=({:+.4f}, {:+.4f}, {:+.4f})".format(
            result["score"], result["width_m"], *result["translation_m"]
        )
    )
    print("[d405] result:", run_dir / "top_grasp.json")
    return result


def select_executable_grasp(
    args: argparse.Namespace, run_dir: Path, grasps: np.ndarray, source_indices: np.ndarray
) -> dict | None:
    if grasps.ndim != 2 or grasps.shape[0] == 0 or grasps.shape[1] < 16:
        return None
    max_predicted_width = float(args.gripper_open_width_m) - float(args.gripper_width_margin_m)
    if max_predicted_width <= 0:
        raise RuntimeError("gripper open width must exceed the configured width margin")
    feasible_indices = np.flatnonzero(grasps[:, 1] <= max_predicted_width)
    feasible = grasps[feasible_indices]
    if len(feasible) == 0:
        print(
            "[d405] no grasp fits the physical jaw limit: predicted width must be <= "
            f"{max_predicted_width:.4f} m for {args.gripper_open_width_m:.4f} m open width "
            f"and {args.gripper_width_margin_m:.4f} m margin"
        )
        return None
    selected_local_index = int(feasible_indices[int(np.argmax(feasible[:, 0]))])
    selected = grasp_result(run_dir, grasps[selected_local_index])
    selected["source_grasp_index"] = int(source_indices[selected_local_index])
    (run_dir / "selected_executable_grasp.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    print(
        "[d405] selected executable grasp: score={:.4f} width={:.4f} m".format(
            selected["score"], selected["width_m"]
        )
    )
    return selected


def load_tcp_translation(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("point_is_grasp_tcp") is not True:
        raise RuntimeError(
            "the TCP calibration does not certify that its point is the gripper pinch/grasp center; "
            "real grasp execution is blocked until a T_eef_grasp_tcp calibration is provided"
        )
    transform = np.asarray(
        payload.get("T_eef_grasp_tcp", payload.get("T_eef_fixed_point")), dtype=np.float64
    )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError(f"invalid T_eef_grasp_tcp in {path}")
    return transform


def grasp_tcp_is_certified(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("point_is_grasp_tcp") is True


def build_execution_plan(
    args: argparse.Namespace,
    top_grasp: dict,
    current_eef_rotation: Rotation | None = None,
    parallel_jaw_symmetry_rotation_deg: float | None = None,
) -> dict:
    if args.pregrasp_height_m <= 0 or args.gripper_open_width_m <= 0:
        raise RuntimeError("pregrasp height and gripper open width must be positive")
    requested_width = float(top_grasp["width_m"]) + float(args.gripper_width_margin_m)
    if requested_width <= 0 or requested_width > args.gripper_open_width_m:
        raise RuntimeError(
            "predicted width + margin is outside the gripper range: "
            f"{requested_width:.4f} m > {args.gripper_open_width_m:.4f} m"
        )
    R_grasp = np.asarray(top_grasp["rotation_matrix"], dtype=np.float64)
    if R_grasp.shape != (3, 3) or not np.allclose(R_grasp.T @ R_grasp, np.eye(3), atol=1e-4):
        raise RuntimeError("predicted grasp rotation matrix is invalid")
    # GraspNet: +X approach, +Y opening, +Z side. Pivot calibration measured
    # the pinch TCP at +X of the EEF, so these approach axes must coincide.
    # The fixed 65-degree installation offset is consequently around local X
    # (the approach axis), not around EEF Z.
    R_gripper = R_grasp
    nominal_eef = R_gripper @ Rotation.from_euler(
        "x", float(args.gripper_yaw_offset_deg), degrees=True
    ).as_matrix().T
    symmetry_rotation_deg = 0.0
    rotation_to_current_deg = None
    if parallel_jaw_symmetry_rotation_deg is not None:
        if float(parallel_jaw_symmetry_rotation_deg) not in (0.0, 180.0):
            raise RuntimeError("parallel_jaw_symmetry_rotation_deg must be 0 or 180")
        selected = int(float(parallel_jaw_symmetry_rotation_deg) == 180.0)
        equivalent_eef = nominal_eef @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        candidates = (nominal_eef, equivalent_eef)
        R_eef = candidates[selected]
        symmetry_rotation_deg = 180.0 * selected
        if current_eef_rotation is not None:
            rotation_to_current_deg = math.degrees(
                (Rotation.from_matrix(R_eef) * current_eef_rotation.inv()).magnitude()
            )
    elif current_eef_rotation is not None:
        equivalent_eef = nominal_eef @ Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        candidates = (nominal_eef, equivalent_eef)
        errors = [
            math.degrees((Rotation.from_matrix(candidate) * current_eef_rotation.inv()).magnitude())
            for candidate in candidates
        ]
        selected = int(np.argmin(errors))
        R_eef = candidates[selected]
        symmetry_rotation_deg = 180.0 * selected
        rotation_to_current_deg = errors[selected]
    else:
        R_eef = nominal_eef
    center = np.asarray(top_grasp["translation_m"], dtype=np.float64)
    pregrasp_tcp = center + np.array([0.0, 0.0, float(args.pregrasp_height_m)])
    direct_grasp_center = bool(getattr(args, "planner_direct_grasp_center", False))
    if direct_grasp_center:
        # The current gripper's Planner ``left_eef`` frame is the open-jaw
        # center used by GraspNet.  The old pivot calibration measured the
        # closed tips of a different gripper and must not affect this target.
        grasp_tcp_offset_eef = np.zeros(3, dtype=np.float64)
        pregrasp_eef = pregrasp_tcp.copy()
        grasp_eef = center.copy()
        tcp_translation_calibration = None
        target_frame = "left_eef"
    else:
        T_eef_grasp_tcp = load_tcp_translation(Path(args.tcp_translation_calibration))
        grasp_tcp_offset_eef = T_eef_grasp_tcp[:3, 3]
        pregrasp_eef = pregrasp_tcp - R_eef @ grasp_tcp_offset_eef
        grasp_eef = center - R_eef @ grasp_tcp_offset_eef
        tcp_translation_calibration = str(args.tcp_translation_calibration)
        target_frame = "sdk_eef_pose"
    rotvec = Rotation.from_matrix(R_eef).as_rotvec()
    return {
        "coordinate_frame": "arm1_base",
        "score": float(top_grasp["score"]),
        "grasp_center_base_m": center.tolist(),
        "gripper_rotation_base": R_gripper.tolist(),
        "eef_rotation_base": R_eef.tolist(),
        "eef_axis_angle_rad": rotvec.tolist(),
        "orientation_mode": "full_graspnet_rotation_with_rigid_flange_yaw_offset",
        "parallel_jaw_symmetry_rotation_deg": symmetry_rotation_deg,
        "rotation_from_capture_eef_deg": rotation_to_current_deg,
        "eef_tcp_axis_base": R_eef[:, 0].tolist(),
        "grasp_approach_axis_base": R_grasp[:, 0].tolist(),
        "pregrasp_grasp_tcp_base_m": pregrasp_tcp.tolist(),
        "pregrasp_eef_pose": np.r_[pregrasp_eef, rotvec].tolist(),
        "grasp_eef_pose": np.r_[grasp_eef, rotvec].tolist(),
        "predicted_width_m": float(top_grasp["width_m"]),
        "opening_width_m": requested_width,
        "opening_percent": 100.0 * requested_width / float(args.gripper_open_width_m),
        "width_margin_m": float(args.gripper_width_margin_m),
        "gripper_open_width_m": float(args.gripper_open_width_m),
        "pregrasp_height_m": float(args.pregrasp_height_m),
        "yaw_offset_deg": float(args.gripper_yaw_offset_deg),
        "planner_direct_grasp_center": direct_grasp_center,
        "target_frame": target_frame,
        "tcp_translation_calibration": tcp_translation_calibration,
        "assumption": "rigid flange mounting: full GraspNet roll/pitch/yaw is followed; yaw offset is the configured tool mounting offset",
    }


def capture_eef_rotation(run_dir: Path, T_eef_wrist: np.ndarray) -> Rotation:
    """Recover the EEF orientation at capture from T_base_camera and hand-eye calibration."""
    meta = json.loads((run_dir / "d405" / "meta.json").read_text(encoding="utf-8"))
    T_base_camera = np.asarray(meta.get("T_base_camera"), dtype=np.float64)
    if T_base_camera.shape != (4, 4):
        raise RuntimeError("cannot recover capture EEF orientation: T_base_camera is unavailable")
    T_base_eef = T_base_camera @ np.linalg.inv(T_eef_wrist)
    return Rotation.from_matrix(T_base_eef[:3, :3])


def show_official_visualization(
    args: argparse.Namespace,
    run_dir: Path,
    grasp_indices: np.ndarray,
    selected_source_index: int | None = None,
    execution_plan: dict | None = None,
) -> None:
    if not args.vis:
        return
    if not args.display:
        raise RuntimeError("Open3D visualization requires --display :1 or --no-vis")
    os.environ["DISPLAY"] = str(args.display)
    os.environ.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))
    import open3d as o3d
    from graspnetAPI import GraspGroup

    grasp_dir = run_dir / "d405"
    points = np.asarray(np.load(grasp_dir / "point_cloud.npy"), dtype=np.float64)
    colors_path = grasp_dir / "point_colors.npy"
    colors = np.asarray(np.load(colors_path), dtype=np.float64) if colors_path.exists() else None
    camera_grasps = np.asarray(np.load(grasp_dir / "grasps_camera.npy"), dtype=np.float64)
    if camera_grasps.ndim != 2 or np.any(grasp_indices < 0) or np.any(grasp_indices >= len(camera_grasps)):
        raise RuntimeError("camera-frame grasp data does not match tabletop-filtered base grasps")
    ordered_indices = grasp_indices[np.argsort(camera_grasps[grasp_indices, 0])[::-1]]
    display_indices = ordered_indices[: args.top_k].tolist()
    if selected_source_index is not None and selected_source_index not in display_indices:
        display_indices.append(selected_source_index)
    grasps = GraspGroup(camera_grasps[np.asarray(display_indices, dtype=np.int64)])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None and colors.shape == points.shape and np.all(np.isfinite(colors)):
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    else:
        print("[d405] point colors unavailable in this historical run; using gray cloud")
        cloud.paint_uniform_color([0.62, 0.62, 0.62])
    grippers = grasps.to_open3d_geometry_list()
    for grasp_index, gripper in zip(display_indices, grippers):
        gripper.paint_uniform_color([1.0, 0.75, 0.0])

    selected_label = ""
    if execution_plan is not None:
        # GraspGroup geometry is camera-frame. Render the real commanded EEF
        # orientation at the grasp TCP, rather than the pre-offset network pose.
        meta = json.loads((grasp_dir / "meta.json").read_text(encoding="utf-8"))
        T_base_camera = np.asarray(meta.get("T_base_camera"), dtype=np.float64)
        if T_base_camera.shape != (4, 4):
            raise RuntimeError("cannot visualize execution plan: T_base_camera is unavailable")
        R_camera_base = T_base_camera[:3, :3].T
        t_camera_base = -R_camera_base @ T_base_camera[:3, 3]
        planned = np.array(camera_grasps[int(selected_source_index)], copy=True)
        R_eef_base = np.asarray(execution_plan["eef_rotation_base"], dtype=np.float64)
        tcp_base = np.asarray(execution_plan["grasp_center_base_m"], dtype=np.float64)
        planned[4:13] = (R_camera_base @ R_eef_base).reshape(-1)
        planned[13:16] = R_camera_base @ tcp_base + t_camera_base
        planned_gripper = GraspGroup(planned[None, :]).to_open3d_geometry_list()[0]
        planned_gripper.paint_uniform_color([0.0, 1.0, 0.0])
        grippers.append(planned_gripper)
        selected_label = " plus corrected execution pose in green"
    elif selected_source_index is not None:
        for grasp_index, gripper in zip(display_indices, grippers):
            if grasp_index == selected_source_index:
                gripper.paint_uniform_color([0.0, 1.0, 0.0])
        selected_label = " plus selected executable grasp in green"
    print("[d405] Open3D: showing %d tabletop-safe grasps%s" % (len(grasps), selected_label))
    o3d.visualization.draw_geometries([cloud, *grippers], window_name="Version4 D405 GraspNet")


def _check_arm_ready(state: dict, arm_id: int) -> dict:
    if str(state.get("system_state", "")).lower() != "running":
        raise RuntimeError(f"arm system_state is not RUNNING: {state.get('system_state')}")
    key = "left" if arm_id == 1 else "right"
    arm_state = (state.get("state") or {}).get(key) or {}
    if arm_state.get("enabled") is not True:
        raise RuntimeError(f"arm{arm_id} is not enabled")
    return arm_state


def _eef_pose_command(arm_state: dict) -> np.ndarray:
    raw_pose = arm_state.get("eef_pose") or {}
    if not isinstance(raw_pose, dict):
        raise RuntimeError("arm state does not contain a structured eef_pose")
    xyz = np.asarray(raw_pose.get("xyz"), dtype=np.float64)
    rpy = np.asarray(raw_pose.get("rpy"), dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.all(np.isfinite(np.r_[xyz, rpy])):
        raise RuntimeError("arm state does not contain finite EEF xyz/rpy")
    return np.r_[xyz, Rotation.from_euler("xyz", rpy).as_rotvec()]


def _smoothstep5(progress: float) -> float:
    """Quintic timing used by the validated local custom move_to_pose."""
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def _minimum_scurve_duration(distance: float, max_speed: float, max_acceleration: float) -> float:
    """Duration required for a quintic path to stay within velocity and acceleration limits."""
    if distance <= 1e-12:
        return 0.0
    speed_duration = 1.875 * distance / max_speed
    acceleration_duration = math.sqrt((10.0 / math.sqrt(3.0)) * distance / max_acceleration)
    return max(speed_duration, acceleration_duration)


def move_absolute_pose(args: argparse.Namespace, pose: list[float], duration_s: float) -> None:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)) or duration_s <= 0:
        raise RuntimeError("invalid absolute EEF pose or motion duration")
    factor = float(args.cartesian_speed_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise RuntimeError("--cartesian-speed-factor must be > 0")
    print("-> planning Cartesian LIN grasp motion (legacy servo disabled)")
    plan_legacy_axis_angle_pose(
        values.tolist(), planner_type="lin", planner_host=args.planner_host,
        speed=min(1.0, 0.1 * factor), execute_url=args.execute_url, execute=True,
    )


def close_collector_gripper_session(args: argparse.Namespace) -> None:
    """Close the long-lived collector-compatible gripper RPC client."""
    gripper = getattr(args, "_collector_gripper_client", None)
    setattr(args, "_collector_gripper_client", None)
    setattr(args, "_collector_gripper_initialized", False)
    if gripper is not None:
        try:
            gripper.close()
        except Exception:
            pass


def collector_gripper_session(args: argparse.Namespace):
    """Return the same long-lived synchronous Gripper client used by collector."""
    gripper = getattr(args, "_collector_gripper_client", None)
    if gripper is not None:
        return gripper
    from parm_sdk import Gripper

    # New gripper drivers publish authoritative state only on gripper.state.
    # Keep one client for the whole Version7 run, but subscribe to port 7702.
    gripper = Gripper(host=args.gripper_host, status_port=7702)
    setattr(args, "_collector_gripper_client", gripper)
    print(
        f"-> collector-compatible gripper session opened: host={args.gripper_host}, "
        f"hand={args.gripper_hand}, status_port=7702"
    )
    return gripper


def wait_for_gripper_hand_state(gripper, hand: int, timeout_s: float = 2.0) -> dict | None:
    """Wait through transient schema-3 PUB frames that omit one device."""
    deadline = time.monotonic() + timeout_s
    while True:
        state = gripper.get_state().get(hand)
        if isinstance(state, dict) and state:
            return state
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def move_gripper(
    args: argparse.Namespace, position_percent: float, label: str, *, force: float | None = None
) -> dict:
    """Send one gripper action through collector's persistent RPC pattern."""

    attempts = int(args.gripper_command_retries)
    if attempts < 1 or args.gripper_retry_delay_s < 0:
        raise RuntimeError("invalid gripper retry settings")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            gripper = collector_gripper_session(args)
            state = wait_for_gripper_hand_state(gripper, args.gripper_hand)
            if state is None:
                raise RuntimeError(f"gripper hand {args.gripper_hand} is unavailable")
            motion_state = str(state.get("motion_state", "")).strip().lower()
            if motion_state in {"dropped", "error"} or state.get("error"):
                raise RuntimeError(
                    f"gripper hand {args.gripper_hand} is unavailable or faulted: {state}; "
                    "arm1 normally uses --gripper-hand 1"
                )
            # Schema-3 reports a completed, auto-initialized gripper as
            # motion_state='arrived', not initialized=True.  Treat both as
            # usable so every open/close does not reset the physical gripper.
            ready = bool(state.get("initialized")) or motion_state in {
                "ready", "arrived", "idle", "gripping"
            }
            if not ready:
                gripper.init(args.gripper_hand)
            setattr(args, "_collector_gripper_initialized", True)
            move_kwargs = {
                "force": args.gripper_force if force is None else force,
                "mode": "sync",
                "preempt": False,
            }
            if args.gripper_speed is not None:
                move_kwargs["speed"] = args.gripper_speed
            response = gripper.move(args.gripper_hand, position_percent, **move_kwargs)
            final_state = wait_for_gripper_hand_state(gripper, args.gripper_hand)
            reported_state = (response.get("data") or {}) if isinstance(response, dict) else {}
            state = final_state or reported_state
            if not state:
                raise RuntimeError("gripper command returned no final device state")
            if state.get("motion_state") == "dropped":
                raise RuntimeError("gripper became dropped/unresponsive after the command")
            details = ", ".join(
                f"{key}={state[key]!r}"
                for key in ("position", "motion_state", "initialized", "fault", "error")
                if key in state
            )
            print(
                f"-> gripper {label} completed (target={position_percent:.1f}%, "
                f"force={(args.gripper_force if force is None else force):.1f}%); reported {details}"
            )
            return state
        except Exception as exc:
            last_error = exc
            # Match collector's long-lived-client behavior in the healthy
            # case, but reconnect after a failed RPC so a broken ZMQ session
            # does not poison the retry attempts.
            close_collector_gripper_session(args)
            if attempt == attempts:
                break
            print(f"-> gripper {label} attempt {attempt}/{attempts} failed: {exc}; retrying")
            time.sleep(float(args.gripper_retry_delay_s))
    raise RuntimeError(f"gripper {label} failed after {attempts} attempts") from last_error


def open_gripper(args: argparse.Namespace) -> None:
    move_gripper(args, 100.0, "open")


def close_gripper(args: argparse.Namespace) -> dict:
    return move_gripper(args, 0.0, "close", force=float(args.gripper_close_force))


def execute_grasp(args: argparse.Namespace, plan: dict) -> None:
    if not (args.execute_grasp and args.confirm_real_grasp):
        print("-> grasp plan is dry-run; add --execute-grasp --confirm-real-grasp to enable execution")
        return
    from parm_sdk import Arm

    with Arm(host=args.arm_host) as arm:
        arm_state = _check_arm_ready(arm.get_state() or {}, args.arm_id)
        current_xyz = np.asarray((arm_state.get("eef_pose") or {}).get("xyz"), dtype=np.float64)
    pregrasp_xyz = np.asarray(plan["pregrasp_eef_pose"][:3], dtype=np.float64)
    if current_xyz.shape != (3,) or np.linalg.norm(pregrasp_xyz - current_xyz) > args.max_grasp_distance_m:
        raise RuntimeError("pregrasp target is unavailable or exceeds --max-grasp-distance-m")
    move_gripper(args, plan["opening_percent"], "set planned opening")
    print("-> moving to predicted XY/rotation at grasp Z + 5 cm")
    move_absolute_pose(args, plan["pregrasp_eef_pose"], args.pregrasp_duration_s)
    print("-> descending vertically with fixed XY and rotation")
    move_absolute_pose(args, plan["grasp_eef_pose"], args.descend_duration_s)
    close_gripper(args)
    # The raised pose prevents carrying the object at the grasp height.  The
    # placement pose is reached with the gripper still closed; release happens
    # only after that move completes.
    move_to_joint_configuration(args, args.postgrasp_lift_joints_deg, "post-grasp lift pose")
    move_to_joint_configuration(args, args.place_joints_deg, "placement pose")
    open_gripper(args)
    print("-> object released at placement pose")


def preview_pregrasp(args: argparse.Namespace, plan: dict) -> None:
    """Move only to the safe pregrasp pose to visually validate physical alignment."""
    from parm_sdk import Arm

    with Arm(host=args.arm_host) as arm:
        arm_state = _check_arm_ready(arm.get_state() or {}, args.arm_id)
        current_xyz = np.asarray((arm_state.get("eef_pose") or {}).get("xyz"), dtype=np.float64)
    pregrasp_xyz = np.asarray(plan["pregrasp_eef_pose"][:3], dtype=np.float64)
    if current_xyz.shape != (3,) or np.linalg.norm(pregrasp_xyz - current_xyz) > args.max_grasp_distance_m:
        raise RuntimeError("pregrasp target is unavailable or exceeds --max-grasp-distance-m")
    move_gripper(args, plan["opening_percent"], "set preview opening")
    print("-> preview: moving to corrected pregrasp only; no descent and no close")
    move_absolute_pose(args, plan["pregrasp_eef_pose"], args.pregrasp_duration_s)
    print("-> pregrasp preview reached; gripper remains open")


def save_grasp_visualization(
    run_dir: Path,
    grasp_indices: np.ndarray,
    *,
    selected_source_index: int | None = None,
    show: bool = False,
) -> Path:
    """Draw top camera-frame grasps on the captured D405 color image."""
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to save the grasp visualization") from exc

    grasp_dir = run_dir / "d405"
    image_path = grasp_dir / "color.png"
    grasps_path = grasp_dir / "grasps_camera.npy"
    intrinsics_path = grasp_dir / "color_intrinsics.npy"
    if not image_path.exists() or not grasps_path.exists() or not intrinsics_path.exists():
        raise RuntimeError(f"incomplete D405 visualization assets under {grasp_dir}")

    canvas = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    all_grasps = np.asarray(np.load(grasps_path), dtype=np.float64)
    if np.any(grasp_indices < 0) or np.any(grasp_indices >= len(all_grasps)):
        raise RuntimeError("camera-frame grasp data does not match tabletop-filtered base grasps")
    grasps = all_grasps[grasp_indices]
    intrinsics = np.asarray(np.load(intrinsics_path), dtype=np.float64)
    if canvas is None or intrinsics.shape != (3, 3):
        raise RuntimeError(f"invalid D405 visualization assets under {grasp_dir}")
    distortion_path = grasp_dir / "color_distortion.npy"
    distortion = np.asarray(np.load(distortion_path), dtype=np.float64) if distortion_path.exists() else None

    valid = []
    for source_index, grasp in zip(grasp_indices, grasps):
        if grasp.shape[0] < 16 or not np.all(np.isfinite(grasp[:16])):
            continue
        center = grasp[13:16]
        if center[2] <= 1e-6:
            continue
        valid.append((int(source_index), grasp))
    valid.sort(key=lambda item: float(item[1][0]), reverse=True)

    height, width = canvas.shape[:2]
    for rank, (source_index, grasp) in enumerate(valid[:20], start=1):
        center = grasp[13:16]
        rotation = grasp[4:13].reshape(3, 3)
        # GraspNet uses R[:, 0] as approach and R[:, 1] as jaw-opening axes.
        points = np.vstack((center, center + rotation[:, 0] * 0.08, center + rotation[:, 1] * 0.04))
        projected, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intrinsics, distortion
        )
        pixels = np.rint(projected.reshape(-1, 2)).astype(int)
        u, v = pixels[0]
        if not (0 <= u < width and 0 <= v < height):
            continue
        is_selected = source_index == selected_source_index
        color = (0, 255, 0) if is_selected else (0, 215, 255)
        cv2.circle(canvas, (u, v), 8 if is_selected else 5, color, 2, cv2.LINE_AA)
        cv2.line(canvas, tuple(pixels[0]), tuple(pixels[1]), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, tuple(pixels[0]), tuple(pixels[2]), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{rank}:{float(grasp[0]):.2f}",
            (u + 8, max(18, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "D405 grasps (green=selected, red=approach, blue=opening)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output = run_dir / "grasp_overlay.png"
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"failed to write visualization: {output}")
    print("[d405] visualization:", output)

    if show:
        if not os.environ.get("DISPLAY"):
            print("[d405] DISPLAY is not set; saved visualization without opening a window")
        else:
            cv2.imshow("version3 D405 grasp", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return output


def main() -> None:
    args = build_parser().parse_args()
    if args.preview_pregrasp and args.execute_grasp:
        raise RuntimeError("--preview-pregrasp and --execute-grasp cannot be used together")
    T_eef_wrist = load_wrist_calibration(Path(args.wrist_calibration))
    output_root = Path(args.output_dir) if args.output_dir else HERE.parent / "output"
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.auto_open_gripper:
        # Keep the jaws clear while perception and motion planning run.  Closing is
        # only permitted after both verified servo stages have completed.
        open_gripper(args)
    if not move_to_observation(args):
        return
    run_dir = run_inference(args, output_dir)
    raw_grasps = np.asarray(np.load(run_dir / "d405" / "grasps.npy"), dtype=np.float64)
    tabletop_indices = table_safe_grasp_indices(args, run_dir, raw_grasps)
    filtered_grasps = raw_grasps[tabletop_indices]
    top_grasp = summarize_top_grasp(run_dir, filtered_grasps)
    if top_grasp is None:
        if args.save_visualization:
            save_grasp_visualization(run_dir, tabletop_indices, show=args.show_visualization)
        print("-> version4 D405 inference finished without a tabletop-safe grasp")
    else:
        selected_grasp = select_executable_grasp(args, run_dir, filtered_grasps, tabletop_indices)
        selected_source_index = (
            None if selected_grasp is None else int(selected_grasp["source_grasp_index"])
        )
        # Once an executable grasp is selected, preview only that exact grasp.
        # The other collision/table/width candidates are diagnostic data, not
        # robot motion targets, and would make the preview misleading.
        preview_indices = (
            tabletop_indices
            if selected_source_index is None
            else np.asarray([selected_source_index], dtype=np.int64)
        )
        if args.save_visualization:
            save_grasp_visualization(
                run_dir,
                preview_indices,
                selected_source_index=selected_source_index,
                show=args.show_visualization,
            )
        if selected_grasp is None:
            show_official_visualization(args, run_dir, tabletop_indices)
            print("-> version4 D405 inference finished without a physically executable grasp")
            return
        if (
            not bool(getattr(args, "planner_direct_grasp_center", False))
            and not grasp_tcp_is_certified(Path(args.tcp_translation_calibration))
        ):
            show_official_visualization(args, run_dir, preview_indices, selected_source_index)
            print(
                "-> selected grasp is visualization-only: current TCP calibration is an arbitrary "
                "gripper point, not the certified pinch/grasp TCP; real execution is blocked"
            )
            return
        plan = build_execution_plan(
            args,
            selected_grasp,
            capture_eef_rotation(run_dir, T_eef_wrist),
        )
        plan_path = run_dir / "grasp_execution_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print("[d405] grasp execution plan:", plan_path)
        print(
            "[d405] parallel-jaw symmetry={}deg, capture-to-target rotation={:.1f}deg".format(
                plan["parallel_jaw_symmetry_rotation_deg"],
                plan["rotation_from_capture_eef_deg"],
            )
        )
        print(
            "[d405] opening={:.1f}% ({:.4f} m), pregrasp grasp TCP={}".format(
                plan["opening_percent"],
                plan["opening_width_m"],
                [round(value, 4) for value in plan["pregrasp_grasp_tcp_base_m"]],
            )
        )
        show_official_visualization(args, run_dir, preview_indices, selected_source_index, plan)
        if args.preview_pregrasp:
            preview_pregrasp(args, plan)
        else:
            execute_grasp(args, plan)


if __name__ == "__main__":
    main()
