#!/usr/bin/env python3
"""Remove URDF-confirmed left-arm pixels from a saved head-D435 near-depth overlay."""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from tool.inspection_support.legacy_inspect.remove_pc2_urdf_gripper_depth import (  # noqa: E402
    URDF_ROOT,
    load_self_surface_points,
    origin_matrix,
    parse_vector,
    projected_gripper_depth,
)


MODULE_ROOT = HERE.parent.parent
DEFAULT_CAPTURE = MODULE_ROOT / "output/head_rgbd_20260728_121840"
DEFAULT_ALIGNMENT = (
    PROJECT_ROOT / "calibration/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch15_yaw35_kinematic.json"
)
DEFAULT_OUTPUT = MODULE_ROOT / "output/near_depth_filtered/head_rgbd_20260728_121840_arm_removed_overlay.png"
DEFAULT_JOINTS = [-14.704, 36.499, -75.262, 92.138, -91.44, 50.882, 88.0]


def body_to_left_arm_base() -> np.ndarray:
    """Fixed body_base -> left_arm_base transform from the deployed URDF."""
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.0, -1.3614, -1.5708]).as_matrix()
    transform[:3, 3] = [-5.9285e-05, 0.10253, -0.0014458]
    return transform


def transfer_to_arm_base(urdf_root: Path, joints_deg: list[float]) -> np.ndarray:
    """Compute T_left_hand_transfer_end_left_arm_base using the supplied joints."""
    root = ET.parse(urdf_root / "anchor_config/description/urdf/arm/left/left_arm.urdf.xacro").getroot()
    joint_by_child: dict[str, tuple[str, np.ndarray]] = {}
    for element in root.findall(".//joint"):
        name = element.get("name", "").replace("${prefix}", "left_arm_")
        parent = element.find("parent").get("link", "").replace("${prefix}", "left_arm_")
        child = element.find("child").get("link", "").replace("${prefix}", "left_arm_")
        transform = origin_matrix(element.find("origin"))
        match = re.fullmatch(r"left_arm_joint([1-7])", name)
        if match:
            axis_element = element.find("axis")
            axis = parse_vector(axis_element.get("xyz") if axis_element is not None else None, (0, 0, 1))
            transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(
                axis * np.deg2rad(joints_deg[int(match.group(1)) - 1])
            ).as_matrix()
        joint_by_child[child] = (parent, transform)

    transfer_base_to_transfer_end = np.eye(4)
    transfer_base_to_transfer_end[:3, :3] = Rotation.from_euler("xyz", [-1.5708, 0.0, -1.5708]).as_matrix()
    transfer_base_to_transfer_end[:3, 3] = [0.0495, 0.0, 0.0002853]
    poses = {"left_arm_end_link": np.linalg.inv(transfer_base_to_transfer_end)}
    current = "left_arm_end_link"
    while current in joint_by_child:
        parent, parent_to_child = joint_by_child[current]
        poses[parent] = poses[current] @ np.linalg.inv(parent_to_child)
        current = parent
    if "left_arm_base" not in poses:
        raise RuntimeError("left_arm_base could not be resolved from URDF")
    return poses["left_arm_base"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--joints-deg", type=float, nargs=7, default=DEFAULT_JOINTS)
    parser.add_argument("--near-m", type=float, default=0.10)
    parser.add_argument("--far-m", type=float, default=0.30)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.035)
    parser.add_argument("--mesh-samples", type=int, default=240000)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 <= args.near_m < args.far_m or args.depth_tolerance_m <= 0:
        parser.error("invalid depth range or tolerance")

    meta = json.loads((args.capture_dir / "metadata.json").read_text())
    rgb = cv2.imread(str(args.capture_dir / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(args.capture_dir / "depth_aligned_000.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
        raise RuntimeError("invalid aligned RGB-D capture")
    intrinsics = meta["color_intrinsics"]
    K = np.array([[intrinsics["fx"], 0.0, intrinsics["ppx"]], [0.0, intrinsics["fy"], intrinsics["ppy"]], [0.0, 0.0, 1.0]])
    depth_m = depth.astype(np.float32) * float(meta["depth_scale_m"])

    alignment = json.loads(args.alignment.read_text())
    T_body_camera = np.asarray(alignment["T_body_base_head_camera"], dtype=float)
    T_transfer_arm_base = transfer_to_arm_base(URDF_ROOT, [float(value) for value in args.joints_deg])
    T_body_transfer = body_to_left_arm_base() @ np.linalg.inv(T_transfer_arm_base)
    T_camera_transfer = np.linalg.inv(T_body_camera) @ T_body_transfer
    arm_points = load_self_surface_points(URDF_ROOT, 0.6398063554, args.mesh_samples, [float(value) for value in args.joints_deg])
    rendered_depth = projected_gripper_depth(arm_points, T_camera_transfer, K, depth.shape)
    arm = np.isfinite(rendered_depth) & (depth_m > 0) & (np.abs(depth_m - rendered_depth) <= args.depth_tolerance_m)
    arm = cv2.dilate(arm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    selected = (depth_m >= args.near_m) & (depth_m <= args.far_m) & ~arm

    overlay = rgb.copy()
    yellow = np.array((0, 255, 255), dtype=np.float32)
    overlay[selected] = np.rint(rgb[selected].astype(np.float32) * (1.0 - args.alpha) + yellow * args.alpha).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f"could not write {args.output}")
    print(json.dumps({"output": str(args.output), "arm_pixels_removed": int(arm.sum()), "near_pixels_retained": int(selected.sum())}))


if __name__ == "__main__":
    main()
