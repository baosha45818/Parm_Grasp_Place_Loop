#!/usr/bin/env python3
"""Remove the recorded PC1 left arm/gripper from each RGB-D near-depth overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .remove_pc2_urdf_gripper_depth import (
    MODULE_DIR, URDF_ROOT, load_self_surface_points, process_capture,
)


PC1_DIR = MODULE_DIR / "output" / "pc1"
PC2_DIR = MODULE_DIR / "output" / "pc2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URDF-remove recorded PC1 arm/gripper depth.")
    parser.add_argument("--pc1-dir", type=Path, default=PC1_DIR)
    parser.add_argument("--output-dir", type=Path, default=PC1_DIR / "near_depth_without_gripper")
    parser.add_argument("--camera-pose-snapshot", type=Path,
                        help="Snapshot containing fixed T_body_base_chest_camera; newest PC2 snapshot by default")
    parser.add_argument("--near-min-m", type=float, default=0.150)
    parser.add_argument("--near-max-m", type=float, default=0.300)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.035,
                        help="Depth agreement for the initial URDF self-pixel test (m)")
    parser.add_argument("--mesh-samples", type=int, default=200000)
    parser.add_argument("--dilate-px", type=int, default=2,
                        help="Legacy uniform expansion; unused while adaptive expansion is enabled")
    parser.add_argument("--dilate-near-px", type=int, default=3,
                        help="Expansion for the nearest third of the requested depth range")
    parser.add_argument("--dilate-mid-px", type=int, default=7,
                        help="Expansion for the middle third of the requested depth range")
    parser.add_argument("--dilate-far-px", type=int, default=12,
                        help="Expansion for the farthest third of the requested depth range")
    parser.add_argument("--expansion-depth-tolerance-m", type=float, default=0.015,
                        help="Strict local depth agreement required for newly expanded pixels (m)")
    parser.add_argument("--adaptive-depth-expansion", action="store_true",
                        help="Enable optional depth-adaptive expansion; disabled by default to protect workpieces")
    return parser.parse_args()


def newest_camera_pose_snapshot() -> Path:
    snapshots = sorted(PC2_DIR.glob("urdf_gripper_snapshot_*.json"), key=lambda path: path.stat().st_mtime)
    if not snapshots:
        raise FileNotFoundError("No PC2 URDF snapshot with chest-camera pose exists")
    return snapshots[-1]


def tool_matrix(tool_pose: dict) -> list[list[float]]:
    translation = tool_pose["translation_m"]
    quaternion = tool_pose["quaternion_xyzw"]
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_quat([quaternion[axis] for axis in "xyzw"]).as_matrix()
    matrix[:3, 3] = [translation[axis] for axis in "xyz"]
    return matrix.tolist()


def snapshot_for_capture(capture: Path, T_body_camera: object) -> tuple[dict, list[float]]:
    state_path = capture / "robot_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("robot_state.json is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "ok":
        raise RuntimeError(f"robot state unavailable: {state.get('error', 'unknown error')}")
    joints = [float(value) for value in state["left_arm_joint_position_deg"]]
    if len(joints) != 7:
        raise ValueError("left_arm_joint_position_deg must contain seven values")
    gripper = state.get("left_gripper_state") or {}
    opening_percent = float(gripper["position"])
    if not 0.0 <= opening_percent <= 100.0:
        raise ValueError(f"Invalid gripper opening percent: {opening_percent}")
    # URDF 0.65 rad is fully closed; the service reports 0=closed, 100=open.
    # Use the contemporaneous recorded opening instead of a fixed closed pose.
    active_joint_rad = 0.65 * (1.0 - opening_percent / 100.0)
    return {
        "urdf_root": str(URDF_ROOT),
        "T_body_base_chest_camera": T_body_camera,
        "T_body_base_left_hand_transfer_end": tool_matrix(state["T_body_base_left_hand_transfer_end"]),
        "urdf_active_joint_rad": active_joint_rad,
        "recorded_gripper_position_percent": opening_percent,
    }, joints


def main() -> None:
    args = parse_args()
    pose_path = args.camera_pose_snapshot or newest_camera_pose_snapshot()
    camera_snapshot = json.loads(pose_path.read_text(encoding="utf-8"))
    T_body_camera = camera_snapshot["T_body_base_chest_camera"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for capture in sorted(path for path in args.pc1_dir.glob("rgbd_*") if path.is_dir()):
        try:
            snapshot, joints = snapshot_for_capture(capture, T_body_camera)
            points = load_self_surface_points(URDF_ROOT, float(snapshot["urdf_active_joint_rad"]),
                                             args.mesh_samples, joints)
            result = process_capture(capture, args.output_dir, snapshot, points, args)
            result["status"] = "ok"
            result["gripper_opening_percent"] = snapshot["recorded_gripper_position_percent"]
        except Exception as exc:
            result = {"capture": capture.name, "status": "skipped", "reason": str(exc)}
        records.append(result)
        print(f"{capture.name}: {result['status']}")
    (args.output_dir / "summary.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
    print(f"PC1 near-depth overlays: {args.output_dir}")


if __name__ == "__main__":
    main()
