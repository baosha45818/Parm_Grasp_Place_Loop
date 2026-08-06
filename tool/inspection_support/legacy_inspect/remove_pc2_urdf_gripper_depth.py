#!/usr/bin/env python3
"""Remove the PC2 left end-effector assembly from near-depth layers via URDF.

The script projects the closed left-gripper URDF at a saved PC2 TCP pose into
the chest-camera image.  A projected pixel is removed only when its measured
depth agrees with the rendered gripper depth, which avoids deleting a visible
workpiece behind the hand merely because the two overlap in image coordinates.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy import ndimage
from scipy.spatial.transform import Rotation


MODULE_DIR = Path(__file__).resolve().parent
PC2_DIR = MODULE_DIR / "output" / "pc2"
URDF_ROOT = Path(__file__).resolve().parents[3] / "third_party/anchor_urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URDF-project and remove the PC2 left-gripper depth.")
    parser.add_argument("--pc2-dir", type=Path, default=PC2_DIR)
    parser.add_argument("--snapshot", type=Path, help="Saved URDF gripper snapshot; newest one is the default")
    parser.add_argument("--output-dir", type=Path, default=PC2_DIR / "near_depth_without_gripper")
    parser.add_argument("--near-min-m", type=float, default=0.110)
    parser.add_argument("--near-max-m", type=float, default=0.280)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.035,
                        help="Measured/projected depth agreement required to call a pixel gripper")
    parser.add_argument("--mesh-samples", type=int, default=200000,
                        help="Total sampled URDF surface points for software depth rendering")
    parser.add_argument("--pc2-joints-deg", type=float, nargs=7,
                        default=[-10.496, -31.557, -4.878, 135.978, -99.147, 5.249, 45.166],
                        help="Fixed PC2 left-arm joints 1..7 in degrees; used for terminal-arm URDF links")
    parser.add_argument("--dilate-px", type=int, default=2,
                        help="Small projected-mask expansion to absorb calibration/model error")
    return parser.parse_args()


def newest_snapshot(pc2_dir: Path) -> Path:
    candidates = sorted(pc2_dir.glob("urdf_gripper_snapshot_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No urdf_gripper_snapshot_*.json under {pc2_dir}")
    return candidates[-1]


def matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return result


def parse_vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    clean = text.replace("${pi}", str(np.pi)).replace("${-pi}", str(-np.pi))
    values = [float(item) for item in clean.split()]
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    origin = np.eye(4, dtype=np.float64)
    if element is None:
        return origin
    origin[:3, :3] = Rotation.from_euler("xyz", parse_vector(element.get("rpy"), (0, 0, 0))).as_matrix()
    origin[:3, 3] = parse_vector(element.get("xyz"), (0, 0, 0))
    return origin


def description_root(urdf_root: Path) -> Path:
    """Accept either the legacy package root or a direct description root."""
    root = Path(urdf_root)
    if (root / "urdf").is_dir() and (root / "meshes").is_dir():
        return root
    legacy = root / "anchor_config/description"
    if (legacy / "urdf").is_dir() and (legacy / "meshes").is_dir():
        return legacy
    raise FileNotFoundError(f"URDF description root not found under {root}")


def resolve_mesh(uri: str, urdf_root: Path) -> Path:
    for prefix in (
        "package://anchor_config/description/",
        "package://welding_config/description/",
    ):
        if uri.startswith(prefix):
            return description_root(urdf_root) / uri.removeprefix(prefix)
    raise ValueError(f"Unsupported mesh URI: {uri}")


def joint_transform(joint: dict, active_angle: float) -> np.ndarray:
    transform = joint["origin"].copy()
    if joint["type"] in {"revolute", "continuous"}:
        multiplier, offset = joint["mimic"]
        transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(
            joint["axis"] * (active_angle * multiplier + offset)
        ).as_matrix()
    return transform


def load_self_surface_points(
    urdf_root: Path, active_angle: float, samples: int, pc2_joints_deg: list[float],
    arm_kinematics_path: Path | None = None,
) -> np.ndarray:
    """Resolve the hand plus all fixed-pose left-arm URDF links."""
    description = description_root(urdf_root)
    xacro_path = description / "urdf/hand_tool/left/left_hand.urdf.xacro"
    root = ET.parse(xacro_path).getroot()
    prefix = "left_hand_"
    mesh_by_link: dict[str, list[tuple[Path, np.ndarray]]] = {}
    for link in root.findall(".//link"):
        name = link.get("name", "").replace("${prefix}", prefix)
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            scale = parse_vector(mesh.get("scale"), (1, 1, 1))
            visuals.append((resolve_mesh(mesh.get("filename"), urdf_root), scale))
        if visuals:
            mesh_by_link[name] = visuals

    joints = []
    adjacency: dict[str, list[tuple[str, int, bool]]] = {}
    for element in root.findall(".//joint"):
        parent = element.find("parent").get("link", "").replace("${prefix}", prefix)
        child = element.find("child").get("link", "").replace("${prefix}", prefix)
        mimic = element.find("mimic")
        multiplier = float(mimic.get("multiplier", "1")) if mimic is not None else 0.0
        offset = float(mimic.get("offset", "0")) if mimic is not None else 0.0
        # The sole non-mimic movable joint is the URDF active finger joint.
        if element.get("name", "").replace("${prefix}", prefix) == "left_hand_gripper_finger1_joint":
            multiplier = 1.0
        joint = {
            "parent": parent, "child": child, "type": element.get("type", "fixed"),
            "origin": origin_matrix(element.find("origin")),
            "axis": parse_vector(element.find("axis").get("xyz") if element.find("axis") is not None else None, (0, 0, 1)),
            "mimic": (multiplier, offset),
        }
        index = len(joints)
        joints.append(joint)
        adjacency.setdefault(parent, []).append((child, index, True))
        adjacency.setdefault(child, []).append((parent, index, False))

    link_pose = {"left_hand_transfer_end": np.eye(4, dtype=np.float64)}
    pending = ["left_hand_transfer_end"]
    while pending:
        source = pending.pop()
        for target, index, forward in adjacency.get(source, []):
            if target in link_pose:
                continue
            transform = joint_transform(joints[index], active_angle)
            link_pose[target] = link_pose[source] @ (transform if forward else np.linalg.inv(transform))
            pending.append(target)

    meshes = []
    weighted = []
    for link, entries in mesh_by_link.items():
        if link not in link_pose:
            continue
        for mesh_path, scale in entries:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if mesh.is_empty():
                continue
            mesh.scale(1.0, center=(0, 0, 0))
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
            mesh.transform(link_pose[link])
            meshes.append(mesh)
            weighted.append(max(mesh.get_surface_area(), 1e-9))
    # The visible residual can originate from any left-arm link, not only the
    # terminal wrist assembly.  PC2 fixes all seven arm joints, so reconstruct
    # every arm link by traversing the arm URDF backwards from the known end.
    T_transfer_base_transfer_end = np.eye(4, dtype=np.float64)
    T_transfer_base_transfer_end[:3, :3] = Rotation.from_euler(
        "xyz", [-1.5708, 0.0, -1.5708]
    ).as_matrix()
    T_transfer_base_transfer_end[:3, 3] = [0.0495, 0.0, 0.0002853]
    T_tool_arm_end = np.linalg.inv(T_transfer_base_transfer_end)
    arm_root = ET.parse(
        description / "urdf/arm/left/left_arm.urdf.xacro"
    ).getroot()
    calibrated_origins = {}
    if arm_kinematics_path is not None:
        calibration = yaml.safe_load(Path(arm_kinematics_path).read_text(encoding="utf-8"))
        calibrated_origins = calibration.get("kinematics", {})
    arm_joints = {}
    for element in arm_root.findall(".//joint"):
        name = element.get("name", "").replace("${prefix}", "left_arm_")
        parent = element.find("parent").get("link", "").replace("${prefix}", "left_arm_")
        child = element.find("child").get("link", "").replace("${prefix}", "left_arm_")
        axis_element = element.find("axis")
        axis = parse_vector(axis_element.get("xyz") if axis_element is not None else None, (0, 0, 1))
        origin = element.find("origin")
        if origin is None and name in calibrated_origins:
            item = calibrated_origins[name]
            origin = ET.Element("origin", {
                "xyz": f"{item['x']} {item['y']} {item['z']}",
                "rpy": f"{item['roll']} {item['pitch']} {item['yaw']}",
            })
        transform = origin_matrix(origin)
        match = re.fullmatch(r"left_arm_joint([1-7])", name)
        if match:
            transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(
                axis * np.deg2rad(pc2_joints_deg[int(match.group(1)) - 1])
            ).as_matrix()
        arm_joints[child] = (parent, transform)
    arm_pose = {"left_arm_end_link": T_tool_arm_end}
    current = "left_arm_end_link"
    while current in arm_joints:
        parent, transform = arm_joints[current]
        arm_pose[parent] = arm_pose[current] @ np.linalg.inv(transform)
        current = parent
    for link_name in ("left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4",
                      "left_arm_link5", "left_arm_link6", "left_arm_link7"):
        transform = arm_pose.get(link_name)
        if transform is None:
            raise RuntimeError(f"Could not resolve terminal-arm pose for {link_name}")
        mesh_name = f"{link_name.removeprefix('left_arm_')}.STL"
        mesh = o3d.io.read_triangle_mesh(str(
            description / "meshes/arm/left" / mesh_name
        ))
        if mesh.is_empty():
            raise RuntimeError(f"Could not read terminal-arm URDF mesh: {mesh_name}")
        mesh.transform(transform)
        meshes.append(mesh)
        weighted.append(max(mesh.get_surface_area(), 1e-9))
    if not meshes:
        raise RuntimeError("No left-gripper URDF meshes were resolved")
    weights = np.asarray(weighted, dtype=np.float64)
    counts = np.maximum(500, np.rint(samples * weights / weights.sum()).astype(int))
    return np.concatenate([
        np.asarray(mesh.sample_points_uniformly(int(count)).points)
        for mesh, count in zip(meshes, counts, strict=True)
    ])


def projected_gripper_depth(points_tool: np.ndarray, T_camera_tool: np.ndarray, K: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    points = (T_camera_tool[:3, :3] @ points_tool.T).T + T_camera_tool[:3, 3]
    valid = points[:, 2] > 0.02
    points = points[valid]
    u = np.rint(K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    zbuffer = np.full(shape, np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, (v[inside], u[inside]), points[inside, 2].astype(np.float32))
    return zbuffer


def process_capture(capture_dir: Path, output_dir: Path, snapshot: dict, points_tool: np.ndarray, args: argparse.Namespace) -> dict:
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    rgb = cv2.imread(str(capture_dir / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = np.load(capture_dir / "depth_aligned_000.npy", allow_pickle=False)
    if rgb is None or depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"Invalid RGB-D pair in {capture_dir}")
    K = np.asarray(metadata["color_intrinsics"], dtype=np.float64)
    scale = float(metadata["depth_scale_m_per_unit"])
    T_camera_tool = np.linalg.inv(matrix(snapshot["T_body_base_chest_camera"], "camera pose")) @ matrix(
        snapshot["T_body_base_left_hand_transfer_end"], "tool pose"
    )
    rendered_depth = projected_gripper_depth(points_tool, T_camera_tool, K, depth.shape)
    measured_depth = depth.astype(np.float32) * scale
    projected = np.isfinite(rendered_depth)
    # Permit a small positive/negative error; actual occluding gripper pixels
    # must agree with the URDF-predicted range.
    gripper = projected & (measured_depth > 0) & (
        np.abs(measured_depth - rendered_depth) <= args.depth_tolerance_m
    )
    # PC1 optionally uses a depth-adaptive expansion: calibration/model error
    # occupies more pixels on the farther parts of the arm, while a small
    # expansion near the camera protects a potentially grasped workpiece.
    if getattr(args, "adaptive_depth_expansion", False):
        split_near = args.near_min_m + (args.near_max_m - args.near_min_m) / 3.0
        split_far = args.near_min_m + 2.0 * (args.near_max_m - args.near_min_m) / 3.0
        expanded = np.zeros_like(gripper)
        for low, high, radius in (
            (args.near_min_m, split_near, args.dilate_near_px),
            (split_near, split_far, args.dilate_mid_px),
            (split_far, args.near_max_m + 1e-6, args.dilate_far_px),
        ):
            confirmed = gripper & (measured_depth >= low) & (measured_depth < high)
            expanded |= confirmed
            if radius:
                candidate = cv2.dilate(
                    confirmed.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)
                ) > 0
                # A 2-D dilation alone can cross from a gripper edge straight
                # onto the grasped workpiece.  Keep an added pixel only if its
                # measured depth also agrees with its nearest confirmed URDF
                # surface point.  Original confirmed pixels remain untouched.
                _, nearest = ndimage.distance_transform_edt(
                    ~confirmed, return_indices=True
                )
                nearest_rendered = rendered_depth[nearest[0], nearest[1]]
                depth_consistent = (
                    (measured_depth > 0)
                    & np.isfinite(nearest_rendered)
                    & (np.abs(measured_depth - nearest_rendered) <= args.expansion_depth_tolerance_m)
                )
                expanded |= candidate & depth_consistent
        gripper = expanded
    elif args.dilate_px:
        gripper = cv2.dilate(gripper.astype(np.uint8), np.ones((2 * args.dilate_px + 1,) * 2, np.uint8)) > 0
    near = (measured_depth >= args.near_min_m) & (measured_depth <= args.near_max_m) & ~gripper
    count, labels, stats, _ = cv2.connectedComponentsWithStats(near.astype(np.uint8), connectivity=8)
    if count > 1:
        near = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    else:
        near[:] = False
    overlay = rgb.copy()
    overlay[near] = (0, 140, 255)
    rendered = cv2.addWeighted(overlay, 0.58, rgb, 0.42, 0.0)
    cv2.putText(rendered, f"near {args.near_min_m:.3f}-{args.near_max_m:.3f} m | URDF self removed: {gripper.mean():.1%}",
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(rendered, f"near {args.near_min_m:.3f}-{args.near_max_m:.3f} m | URDF self removed: {gripper.mean():.1%}",
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = capture_dir.name
    overlay_path = output_dir / f"{stem}_near_depth_without_gripper.png"
    if not cv2.imwrite(str(overlay_path), rendered):
        raise RuntimeError(f"Cannot write {overlay_path}")
    return {"capture": capture_dir.name, "overlay": overlay_path.name}


def main() -> None:
    args = parse_args()
    if not (0 < args.near_min_m < args.near_max_m and args.depth_tolerance_m > 0 and args.mesh_samples > 0):
        raise ValueError("Invalid depth range, tolerance, or mesh sample count")
    snapshot_path = args.snapshot or newest_snapshot(args.pc2_dir)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    angle = float(snapshot["urdf_active_joint_rad"])
    points_tool = load_self_surface_points(
        Path(snapshot.get("urdf_root", URDF_ROOT)), angle, args.mesh_samples, args.pc2_joints_deg
    )
    captures = sorted(path for path in args.pc2_dir.glob("rgbd_*") if path.is_dir())
    results = [process_capture(path, args.output_dir, snapshot, points_tool, args) for path in captures]
    print(f"Rendered {len(results)} RGB/depth overlays with {len(points_tool)} URDF surface points into {args.output_dir}")


if __name__ == "__main__":
    main()
