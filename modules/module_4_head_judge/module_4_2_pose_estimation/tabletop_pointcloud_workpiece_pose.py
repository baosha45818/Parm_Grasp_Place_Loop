#!/usr/bin/env python3
"""Capture head RGB-D, seed SAM2 from tabletop geometry, and write mask poses.

Each invocation creates the next numeric capture directory under ``runs`` in
this module.  The directory is self-contained:
RGB, XYZ, depth-derived seed masks, the final SAM2 mask, and the mask-pose
JSON are all saved together.

The pose JSON preserves the body-base XY, tail-to-head yaw, and quadrant
conventions of the established RGB-mask pipeline.  RGB-D is used only to find
reliable SAM2 foreground prompts.

Example:
    python3 tabletop_pointcloud_workpiece_pose.py --workpiece 2
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import json
import math
import os
import socket
import struct
import subprocess
import sys
import shutil
import tempfile
import time
import http.client
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

MODULE_ROOT = Path(__file__).resolve().parent
PARM_PROJECT_ROOT = MODULE_ROOT.parents[2]
if str(PARM_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARM_PROJECT_ROOT))
RUNTIME_PYTHON = Path(os.environ.get(
    "POSE_ESTIMATION_PYTHON",
    str(PARM_PROJECT_ROOT / ".venv/bin/python"),
))
if not RUNTIME_PYTHON.is_file():
    raise RuntimeError(f"main.py runtime is missing: {RUNTIME_PYTHON}")
# Do not resolve the executable here: both the system interpreter and a venv
# normally resolve to the same underlying binary, but only the venv supplies
# this module's OpenCV, Open3D, RealSense and SDK packages.
if Path(sys.executable).absolute() != RUNTIME_PYTHON.absolute():
    os.execv(str(RUNTIME_PYTHON), [str(RUNTIME_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import cv2
import numpy as np
import open3d as o3d
from tools.icp import full_6d_point_to_plane_icp

# These helpers were previously imported from mask_workpiece_pose.py.  Keep a
# local copy so this file can run as a standalone tabletop pose pipeline.
# Default to the legacy SDK-to-planner alignment calibration.  It is valid
# only at this head observation pose in the current SDK command convention.
DEFAULT_CALIBRATION = MODULE_ROOT / "calibration/sdk_base_to_planner_alignment.json"
CALIBRATED_HEAD_PITCH_DEG = 38.5
CALIBRATED_HEAD_YAW_DEG = 0.0
CALIBRATED_TEACH_DISPLAY_PITCH_DEG = 38.5
DEFAULT_PLANE_Z_M = -0.391809

# Report-only pose-confidence policy.  These defaults are deliberately kept
# separate from ICP optimization: they identify ambiguous or weak estimates
# without changing the selected transform or blocking a downstream grasp.
POSE_CONFIDENCE_THRESHOLDS = {
    "high_min_observed_points": 400,
    "acceptable_min_observed_points": 300,
    "high_min_fitness": 0.97,
    "acceptable_min_fitness": 0.93,
    "high_max_rmse_mm": 4.2,
    "acceptable_max_rmse_mm": 4.8,
    "high_max_depth_score_mm": 4.5,
    "acceptable_max_depth_score_mm": 6.25,
    "high_min_face_margin_mm": 1.0,
    "acceptable_min_face_margin_mm": 0.35,
    "high_max_near_optimal_center_deviation_mm": 3.0,
    "high_max_near_optimal_rotation_deviation_deg": 2.0,
    "acceptable_max_near_optimal_center_deviation_mm": 5.0,
    "acceptable_max_near_optimal_rotation_deviation_deg": 4.0,
    "reject_near_optimal_center_deviation_mm": 8.0,
    "reject_near_optimal_rotation_deviation_deg": 6.0,
}

# Keep the CAD assets explicit here so this script remains runnable after the
# former head_pointcloud_icp_pose helper has been removed.
WORKPIECE_1_DIR = MODULE_ROOT / "assets/workpiece_1"
WORKPIECE_2_DIR = MODULE_ROOT / "assets/workpiece_2"
WORKPIECE_ASSETS = {
    "1": (
        WORKPIECE_1_DIR / "workpiece_1.stl",
        WORKPIECE_1_DIR / "workpiece1.semantic.json",
    ),
    "2": (
        WORKPIECE_2_DIR / "workpiece_2.stl",
        WORKPIECE_2_DIR / "workpiece2.semantic.json",
    ),
}
DEFAULT_WORKPIECE = "2"
DEFAULT_STL, DEFAULT_SEMANTIC_CALIBRATION = WORKPIECE_ASSETS[DEFAULT_WORKPIECE]


DEFAULT_HEAD_SERIAL = "261222076212"
DEFAULT_CAPTURE_ROOT = PARM_PROJECT_ROOT / "output/module_4_2"
SAM2_SOCKET = Path(os.environ.get(
    "POSE_ESTIMATION_SAM2_SOCKET",
    str(PARM_PROJECT_ROOT / "tool/sam2-service/runtime/sam2-service.sock"),
))
SAM2_BIN = Path(os.environ.get(
    "POSE_ESTIMATION_SAM2_BIN",
    str(PARM_PROJECT_ROOT / "tool/sam2-service/.venv/bin/sam2-service"),
))
# Trapezoidal tabletop region marked by the operator on workpiece_1 capture
# 584. Vertices are clockwise in the 1280x720 head RGB frame, and follow the
# table edges so floor and surrounding hardware are not treated as workpieces.
DEFAULT_TABLETOP_ROI_POLYGON_PX = np.asarray(
    ((121, 156), (1252, 167), (1255, 684), (25, 679)), dtype=np.float64,
)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Load either legacy or current head eye-to-hand calibration schema.

    The new ChArUco output names its transform ``T_base_head_camera`` while
    the former planner-alignment export used ``T_body_base_head_camera``.
    Both mean p_base = T_base_head_camera @ p_head_camera.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in (None, "ACCEPTED"):
        raise RuntimeError(f"calibration is not accepted: {path}")
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(payload.get("distortion_coeffs", []), dtype=np.float64).reshape(-1)
    transform_key = next(
        (key for key in ("T_base_head_camera", "T_body_base_head_camera") if key in payload), None
    )
    if transform_key is None:
        raise RuntimeError(
            f"calibration lacks T_base_head_camera/T_body_base_head_camera: {path}"
        )
    transform = np.asarray(payload[transform_key], dtype=np.float64)
    image_size = tuple(int(value) for value in payload["image_size"])
    if (camera_matrix.shape != (3, 3) or transform.shape != (4, 4)
            or len(image_size) != 2 or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0))):
        raise RuntimeError(f"invalid calibration payload: {path}")
    return camera_matrix, distortion, transform, image_size


def _pixel_to_base_xy(pixel: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray,
                      t_base_camera: np.ndarray, plane_z: float) -> np.ndarray:
    normalized = cv2.undistortPoints(
        np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2), camera_matrix, distortion
    ).reshape(2)
    ray_base = t_base_camera[:3, :3] @ np.array([normalized[0], normalized[1], 1.0])
    camera_origin = t_base_camera[:3, 3]
    if abs(ray_base[2]) < 1e-9:
        raise RuntimeError("pixel ray is parallel to the workpiece plane")
    distance = (plane_z - camera_origin[2]) / ray_base[2]
    if distance <= 0:
        raise RuntimeError("workpiece plane is behind the camera; check --plane-z and calibration")
    return (camera_origin + distance * ray_base)[:2]


def _angle_and_quadrant(direction_xy: np.ndarray) -> tuple[float, int]:
    angle_deg = math.degrees(math.atan2(float(direction_xy[1]), float(direction_xy[0]))) % 360.0
    return float(angle_deg), int(angle_deg // 90.0) + 1


def _rotated_box_axis(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("component has no contour")
    rectangle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    centre, corners = np.asarray(rectangle[0], dtype=np.float64), cv2.boxPoints(rectangle).astype(np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    index = int(np.argmax(lengths))
    if lengths[index] <= 0:
        raise RuntimeError("component rotated rectangle has zero length")
    return centre, edges[index] / lengths[index], corners


def component_pose(component: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray,
                   t_base_camera: np.ndarray, plane_z: float) -> dict:
    """Mask-only diagnostics retained for JSON compatibility and debug drawing."""
    rows, columns = np.where(component)
    points_xy = np.column_stack((columns, rows)).astype(np.float64)
    if len(points_xy) < 3:
        raise RuntimeError("component has too few pixels")
    center_pixel = np.mean(points_xy, axis=0)
    rectangle_center, coarse_axis, corners = _rotated_box_axis(component)
    coordinates = (points_xy - rectangle_center) @ coarse_axis
    low, high = np.percentile(coordinates, (2.5, 97.5))
    tail_candidates, head_candidates = points_xy[coordinates <= low], points_xy[coordinates >= high]
    if len(tail_candidates) < 1 or len(head_candidates) < 1:
        raise RuntimeError("component has no usable head/tail pixels")
    # The wider endpoint is the head.  This is diagnostic only: ICP/CAD
    # overwrites the final head/tail and yaw fields below.
    normal = np.array([-coarse_axis[1], coarse_axis[0]])
    tail_width = float(np.ptp(tail_candidates @ normal))
    head_width = float(np.ptp(head_candidates @ normal))
    if tail_width > head_width:
        tail_candidates, head_candidates = head_candidates, tail_candidates
        tail_width, head_width = head_width, tail_width
    tail_pixel, head_pixel = tail_candidates.mean(axis=0), head_candidates.mean(axis=0)
    direction = head_pixel - tail_pixel
    long_length = float(np.linalg.norm(direction))
    if long_length <= 1e-9:
        raise RuntimeError("component head-to-tail direction is zero")
    base_angle_deg, base_quadrant = _angle_and_quadrant(
        _pixel_to_base_xy(head_pixel, camera_matrix, distortion, t_base_camera, plane_z)
        - _pixel_to_base_xy(tail_pixel, camera_matrix, distortion, t_base_camera, plane_z)
    )
    center_base = _pixel_to_base_xy(center_pixel, camera_matrix, distortion, t_base_camera, plane_z)
    return {
        "center_pixel_xy": center_pixel.tolist(), "center_base_xy_m": center_base.tolist(),
        "center_base_xy_mm": (center_base * 1000.0).tolist(),
        "grasp_center_pixel_xy": ((tail_pixel + head_pixel) / 2.0).tolist(),
        "head_pixel_xy": head_pixel.tolist(), "tail_pixel_xy": tail_pixel.tolist(),
        "head_direction_base_angle_deg": base_angle_deg, "head_direction_base_quadrant": base_quadrant,
        "head_width_px": head_width, "tail_width_px": tail_width,
        "head_to_tail_width_ratio": float(head_width / max(tail_width, 1e-6)),
        "area_px": int(len(points_xy)), "rotated_box_pixel_corners_xy": corners.tolist(),
    }


def write_icp_workpiece_frame(rgb: np.ndarray, poses: list[dict], camera_matrix: np.ndarray,
                              distortion: np.ndarray, output: Path, roi_polygon: np.ndarray,
                              axis_length_m: float = 0.045) -> None:
    """Overlay the ROI and each ICP-estimated semantic workpiece frame on RGB."""
    image = rgb.copy()
    polygon = np.rint(roi_polygon).astype(np.int32)
    cv2.polylines(image, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
    label = tuple(polygon[0] + np.array((6, 24)))
    cv2.putText(image, "ROI", label, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "ROI", label, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 1, cv2.LINE_AA)
    colors = ((0, 0, 255), (0, 200, 0), (255, 80, 0))  # BGR: X red, Y green, Z blue
    for index, pose in enumerate(poses, start=1):
        # ``workpiece`` always means the configured semantic frame here, not
        # the raw STL axes.  For workpiece 2 this applies diag(-1, -1, 1), so
        # using the STL transform would reverse the displayed X and Y arrows.
        transform = np.asarray(pose["T_head_camera_workpiece_semantic"], dtype=np.float64)
        origin = transform[:3, 3]
        endpoints = np.stack([origin + axis_length_m * transform[:3, axis] for axis in range(3)])
        pixels, _ = cv2.projectPoints(
            np.vstack((origin, endpoints)), np.zeros(3), np.zeros(3), camera_matrix, distortion,
        )
        pixels = pixels.reshape(-1, 2)
        origin_px = tuple(np.rint(pixels[0]).astype(int))
        cv2.circle(image, origin_px, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, origin_px, 4, (255, 255, 255), -1, cv2.LINE_AA)
        for axis, (color, name) in enumerate(zip(colors, ("X", "Y", "Z"))):
            endpoint_px = tuple(np.rint(pixels[axis + 1]).astype(int))
            cv2.arrowedLine(image, origin_px, endpoint_px, color, 3, cv2.LINE_AA, tipLength=0.16)
            cv2.putText(image, name, (endpoint_px[0] + 5, endpoint_px[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"ICP frame #{index}", (origin_px[0] + 8, origin_px[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write {output}")


def load_semantic_calibration(path: Path) -> np.ndarray:
    """Load the semantic-frame-to-STL transform used by the pose output."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    transform = np.asarray(payload["T_stl_workpiece_semantic"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError(f"invalid semantic calibration: {path}")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise RuntimeError(f"semantic calibration is not a homogeneous transform: {path}")
    return transform


def load_positive_face_axis(path: Path) -> np.ndarray:
    """Return the semantic direction which points out of the front face."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    axis_name = str(payload.get("positive_face", {}).get("upward_axis", "+Z"))
    axes = {
        "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
    }
    if axis_name not in axes:
        raise RuntimeError(
            f"positive_face.upward_axis must be one of {sorted(axes)}, got {axis_name!r}: {path}"
        )
    return np.asarray(axes[axis_name], dtype=np.float64)


def load_stl_triangles(path: Path) -> np.ndarray:
    """Load binary or ASCII STL triangles and normalize millimetre CAD to metres."""
    blob = Path(path).read_bytes()
    triangles: np.ndarray
    if len(blob) >= 84:
        count = struct.unpack("<I", blob[80:84])[0]
        if 84 + count * 50 == len(blob):
            triangles = np.empty((count, 3, 3), dtype=np.float64)
            offset = 84
            for index in range(count):
                values = struct.unpack("<12fH", blob[offset:offset + 50])
                triangles[index] = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
                offset += 50
        else:
            triangles = _load_ascii_stl_triangles(blob)
    else:
        triangles = _load_ascii_stl_triangles(blob)
    if len(triangles) == 0:
        raise RuntimeError(f"STL has no triangles: {path}")
    # This workpiece is hand-sized.  Exports whose extent exceeds one metre
    # are interpreted as millimetres, matching the former shared helper.
    if float(np.ptp(triangles.reshape(-1, 3), axis=0).max()) > 1.0:
        triangles *= 1e-3
    return triangles


def _load_ascii_stl_triangles(blob: bytes) -> np.ndarray:
    vertices: list[list[float]] = []
    for line in blob.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices or len(vertices) % 3:
        raise RuntimeError("STL has no valid triangle vertices")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def sample_stl_surface(path: Path, count: int, seed: int = 0) -> np.ndarray:
    """Area-weighted uniform samples over the STL surface."""
    triangles = load_stl_triangles(path)
    v0 = triangles[:, 0]
    edge_1, edge_2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    areas = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
    if not np.any(areas > 0):
        raise RuntimeError(f"STL has zero total area: {path}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(triangles), size=int(count), replace=True, p=areas / areas.sum())
    r1, r2 = rng.random(int(count)), rng.random(int(count))
    root_r1 = np.sqrt(r1)
    return (
        (1.0 - root_r1)[:, None] * triangles[indices, 0]
        + (root_r1 * (1.0 - r2))[:, None] * triangles[indices, 1]
        + (root_r1 * r2)[:, None] * triangles[indices, 2]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=DEFAULT_HEAD_SERIAL, help="head D435i serial number")
    parser.add_argument("--head-host", default=os.environ.get("PARM_HEAD_HOST", "localhost"))
    parser.add_argument(
        "--head-pitch-deg", type=float, default=CALIBRATED_HEAD_PITCH_DEG,
        help="capture pitch in this script's SDK convention; must remain 38.5 for the default calibration",
    )
    parser.add_argument(
        "--head-yaw-deg", type=float, default=CALIBRATED_HEAD_YAW_DEG,
        help="capture yaw in SDK/teach-console convention; must remain 0 for the default calibration",
    )
    parser.add_argument(
        "--head-settle-s", type=float, default=2.0,
        help="seconds to wait after the head move before depth capture (default: 2)",
    )
    parser.add_argument(
        "--head-position-tolerance-deg", type=float, default=0.5,
        help="skip the head move/settle when pitch and yaw are already within this tolerance (default: 0.5)",
    )
    parser.add_argument(
        "--skip-head-move", action="store_true",
        help="do not command the head; intended only for offline/debug captures",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--depth-fuse-frames", type=int, default=15,
        help="median-fuse this many aligned depth frames (default: 15)",
    )
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--plane-z", type=float, default=DEFAULT_PLANE_Z_M)
    parser.add_argument(
        "--roi", nargs=4, type=float, default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="optional normalized rectangular ROI override; the default is the capture-379 tabletop trapezoid",
    )
    parser.add_argument("--plane-distance-mm", type=float, default=4.0)
    parser.add_argument("--plane-samples", type=int, default=40000)
    parser.add_argument("--plane-iters", type=int, default=1200)
    parser.add_argument("--seed-height-mm", type=float, default=10.0)
    parser.add_argument("--seed-max-gray", type=int, default=90)
    parser.add_argument("--min-seed-area", type=int, default=90)
    parser.add_argument("--min-mask-area", type=int, default=350)
    parser.add_argument("--min-pose-area", type=int, default=500)
    parser.add_argument(
        "--workpiece", choices=tuple(WORKPIECE_ASSETS), default=DEFAULT_WORKPIECE,
        help="select bundled CAD/semantic-frame assets (default: 2)",
    )
    parser.add_argument("--stl", type=Path,
                        help="override the STL selected by --workpiece")
    parser.add_argument("--semantic-calibration", type=Path,
                        help="override the semantic JSON selected by --workpiece")
    parser.add_argument(
        "--positive-face-axis-stl", nargs=3, type=float, default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="deprecated compatibility option; full-6DoF ICP no longer constrains any STL face to the table",
    )
    parser.add_argument("--model-samples", type=int, default=14000)
    parser.add_argument("--render-samples", type=int, default=90000)
    parser.add_argument("--max-observed-points", type=int, default=5000)
    parser.add_argument("--min-icp-points", type=int, default=120)
    parser.add_argument("--icp-iters", type=int, default=45)
    parser.add_argument(
        "--icp-workers", type=int, default=4,
        help="cKDTree workers per ICP hypothesis; four hypotheses run concurrently (default: 4)",
    )
    parser.add_argument("--icp-max-corr-mm", type=float, default=9.0)
    parser.add_argument("--icp-global-voxel-mm", type=float, default=5.0,
                        help="voxel size for full-6DoF FPFH/RANSAC initialization")
    parser.add_argument("--icp-ransac-iters", type=int, default=100000,
                        help="maximum iterations for FPFH/RANSAC initialization")
    parser.add_argument("--icp-global-seeds", type=int, default=16,
                        help="number of deterministic FPFH/RANSAC global initializations; side-on workpieces need broader seeding")
    parser.add_argument("--minimum-z-up-dot", type=float, default=0.90,
                        help="minimum absolute dot product between configured positive-face axis and the table-up normal (default 0.90 permits at most about 25.8 degrees from table-up)")
    parser.add_argument("--max-table-penetration-mm", type=float, default=10.0,
                        help="reject CAD poses that place any sampled surface this far below the fitted table; 10 mm absorbs RGB-D/table-plane bias")
    parser.add_argument(
        "--minimum-face-margin-mm", type=float, default=0.35,
        help="minimum depth-score advantage required to classify FRONT or BACK; otherwise OTHER",
    )
    parser.add_argument(
        "--near-optimal-score-window-mm", type=float, default=0.5,
        help="diagnostic window above a same-side best depth score for pose-consensus reporting",
    )
    parser.add_argument(
        "--low-confidence-retries", type=int, default=2,
        help="additional RGB-D captures after a LOW pose-confidence result; LOW attempts are discarded",
    )
    parser.add_argument("--sam2-url", default=f"unix://{SAM2_SOCKET}")
    args = parser.parse_args()
    preset_stl, preset_semantic = WORKPIECE_ASSETS[args.workpiece]
    if args.stl is None:
        args.stl = preset_stl
    if args.semantic_calibration is None:
        args.semantic_calibration = preset_semantic
    return args


def normalized_roi_bounds(roi: tuple[float, float, float, float] | list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) for value in roi)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("--roi must satisfy 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    return int(math.floor(x0 * width)), int(math.floor(y0 * height)), int(math.ceil(x1 * width)), int(math.ceil(y1 * height))


def tabletop_roi_mask(roi: tuple[float, float, float, float] | list[float] | None,
                      width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the default perspective tabletop mask, or an explicit rectangular override."""
    if roi is not None:
        left, top, right, bottom = normalized_roi_bounds(roi, width, height)
        polygon = np.asarray(((left, top), (right - 1, top), (right - 1, bottom - 1), (left, bottom - 1)), dtype=np.int32)
    else:
        scale = np.asarray((width / 1280.0, height / 720.0), dtype=np.float64)
        polygon = np.rint(DEFAULT_TABLETOP_ROI_POLYGON_PX * scale).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_8)
    return mask.astype(bool), polygon


def create_run_dir(root: Path) -> tuple[int, Path]:
    """Atomically reserve the next numeric run directory without overwriting captures."""
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdecimal()]
    index = max(existing, default=0) + 1
    while True:
        run_dir = root / str(index)
        try:
            run_dir.mkdir()
            return index, run_dir
        except FileExistsError:
            index += 1


def import_parm_sdk():
    """Use the same SDK import fallback as push_T/arm_head_teach.py."""
    try:
        return importlib.import_module("parm_sdk")
    except ImportError:
        override = os.environ.get("PARM_SDK_WHEEL", "").strip()
        candidates = [Path(override)] if override else []
        candidates.extend(sorted(Path("/home/dainanjun/Public/parm_sdk_1.0.0").glob("parm_sdk-*.whl"), reverse=True))
        for wheel in candidates:
            if wheel.is_file():
                sys.path.insert(0, str(wheel))
                try:
                    return importlib.import_module("parm_sdk")
                except ImportError:
                    continue
    raise ImportError("无法导入 parm_sdk；请安装 SDK 或设置 PARM_SDK_WHEEL。")


def move_head_to_capture_pose(args: argparse.Namespace) -> dict[str, object]:
    """Move the head to the calibrated capture pose before opening the camera."""
    if args.skip_head_move:
        return {"commanded": False, "reason": "--skip-head-move"}
    if not -8.0 <= args.head_pitch_deg <= 45.0:
        raise ValueError("--head-pitch-deg must be in [-8, 45] for the installed parm_sdk")
    if not -60.0 <= args.head_yaw_deg <= 60.0:
        raise ValueError("--head-yaw-deg must be in [-60, 60]")
    if args.head_settle_s < 0:
        raise ValueError("--head-settle-s cannot be negative")
    if args.head_position_tolerance_deg < 0:
        raise ValueError("--head-position-tolerance-deg cannot be negative")
    sdk = import_parm_sdk()
    head = sdk.Head(host=args.head_host)
    try:
        initial_raw: dict[str, float] | None = None
        initial_sdk: dict[str, float] | None = None
        initially_moving: bool | None = None
        try:
            state = head.get_state()
            initial_raw = {
                "pitch_deg": float(state["pitch"]["position"]),
                "yaw_deg": float(state["yaw"]["position"]),
            }
            # This script uses parm_sdk 20260811: get_state() and move_to()
            # use the same convention (positive pitch looks down about +Y).
            # The older teach console displayed -45 for this physical pose
            # because its SDK negated pitch before sending the command.
            initial_sdk = initial_raw.copy()
            moving_value = state.get("controller", {}).get("is_moving")
            initially_moving = bool(moving_value) if moving_value is not None else None
        except (AttributeError, KeyError, TypeError, ValueError):
            initial_raw = initial_sdk = None
            initially_moving = None

        tolerance = float(args.head_position_tolerance_deg)
        already_at_target = (
            initial_sdk is not None
            and initially_moving is False
            and abs(initial_sdk["pitch_deg"] - float(args.head_pitch_deg)) <= tolerance
            and abs(initial_sdk["yaw_deg"] - float(args.head_yaw_deg)) <= tolerance
        )
        if already_at_target:
            return {
                "commanded": False,
                "reason": "already_at_target",
                "host": args.head_host,
                "requested_pitch_deg": float(args.head_pitch_deg),
                "requested_yaw_deg": float(args.head_yaw_deg),
                "position_tolerance_deg": tolerance,
                "settle_s": float(args.head_settle_s),
                "settle_s_applied": 0.0,
                "initial_state_raw_deg": initial_raw,
                "initial_state_sdk_deg": initial_sdk,
                "initially_moving": initially_moving,
            }

        head.move_to(pitch=float(args.head_pitch_deg), yaw=float(args.head_yaw_deg))
        if args.head_settle_s:
            time.sleep(args.head_settle_s)
        observed_raw: dict[str, float] | None = None
        observed_sdk: dict[str, float] | None = None
        try:
            state = head.get_state()
            observed_raw = {
                "pitch_deg": float(state["pitch"]["position"]),
                "yaw_deg": float(state["yaw"]["position"]),
            }
            observed_sdk = observed_raw.copy()
        except (AttributeError, KeyError, TypeError, ValueError):
            observed_raw = observed_sdk = None
    finally:
        try:
            head.close()
        except Exception:
            pass
    return {
        "commanded": True, "host": args.head_host,
        "requested_pitch_deg": float(args.head_pitch_deg), "requested_yaw_deg": float(args.head_yaw_deg),
        "position_tolerance_deg": tolerance,
        "settle_s": float(args.head_settle_s), "settle_s_applied": float(args.head_settle_s),
        "initial_state_raw_deg": initial_raw, "initial_state_sdk_deg": initial_sdk,
        "initially_moving": initially_moving,
        "observed_state_deg": observed_raw,
        "observed_state_raw_deg": observed_raw, "observed_state_sdk_deg": observed_sdk,
    }


def validate_calibration_capture_pose(args: argparse.Namespace) -> None:
    """Prevent an accidental capture at a pose incompatible with the default extrinsic."""
    try:
        using_default = args.calibration.resolve() == DEFAULT_CALIBRATION.resolve()
    except OSError:
        using_default = args.calibration == DEFAULT_CALIBRATION
    if using_default and (
        not math.isclose(args.head_pitch_deg, CALIBRATED_HEAD_PITCH_DEG, abs_tol=1e-9)
        or not math.isclose(args.head_yaw_deg, CALIBRATED_HEAD_YAW_DEG, abs_tol=1e-9)
    ):
        raise ValueError(
            "the default ChArUco calibration is valid only at the confirmed "
            f"physical low-head pose (this SDK: pitch={CALIBRATED_HEAD_PITCH_DEG:g}, "
            f"yaw={CALIBRATED_HEAD_YAW_DEG:g}; old teach display: "
            f"pitch={CALIBRATED_TEACH_DISPLAY_PITCH_DEG:g}); provide a matching --calibration "
            "before overriding --head-pitch-deg/--head-yaw-deg"
        )


def capture_rgbd(args: argparse.Namespace, run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required; run this script with `uv run` from the project root") from exc
    if min(args.width, args.height, args.fps, args.warmup_frames, args.depth_fuse_frames) <= 0:
        raise ValueError("capture dimensions, FPS, and frame counts must be positive")
    serials = sorted(device.get_info(rs.camera_info.serial_number) for device in rs.context().devices)
    if args.serial not in serials:
        raise RuntimeError(f"head camera {args.serial} was not found; connected cameras: {serials}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=5000)
        depth_samples: list[np.ndarray] = []
        rgb: np.ndarray | None = None
        for _ in range(args.depth_fuse_frames):
            frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("camera returned an incomplete aligned RGB-D frame")
            rgb = np.asanyarray(color_frame.get_data()).copy()
            sample = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            sample[(sample == 0) | (sample == np.iinfo(np.uint16).max)] = np.nan
            depth_samples.append(sample)
        if rgb is None:
            raise RuntimeError("camera did not return an RGB frame")
        depth = np.rint(np.nan_to_num(np.nanmedian(np.stack(depth_samples), axis=0), nan=0.0)).astype(np.uint16)
        depth_profile = frames.get_depth_frame().profile.as_video_stream_profile()
        intr = depth_profile.intrinsics
        scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    rows, columns = np.indices(depth.shape, dtype=np.float32)
    z = depth.astype(np.float32) * scale_m
    xyz = np.empty((*depth.shape, 3), dtype=np.float32)
    xyz[:, :, 0] = (columns - float(intr.ppx)) * z / float(intr.fx)
    xyz[:, :, 1] = (rows - float(intr.ppy)) * z / float(intr.fy)
    xyz[:, :, 2] = z
    xyz[depth == 0] = np.nan
    if not cv2.imwrite(str(run_dir / "rgb.png"), rgb):
        raise RuntimeError("failed to write rgb.png")
    np.savez_compressed(
        run_dir / "rgbd.npz", rgb=rgb, depth_raw=depth, xyz_m=xyz,
        intrinsics=np.asarray([intr.fx, intr.fy, intr.ppx, intr.ppy], dtype=np.float64),
        depth_scale_m=np.asarray(scale_m), serial=np.asarray(args.serial),
        timestamp_unix_s=np.asarray(time.time()),
    )
    metadata: dict[str, object] = {
        "capture_index": None, "captured_at": datetime.now(timezone.utc).isoformat(),
        "serial": args.serial, "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
        "depth_scale_m": scale_m, "depth_fusion": f"median of {args.depth_fuse_frames} aligned frames",
        "intrinsics": {"fx": float(intr.fx), "fy": float(intr.fy), "ppx": float(intr.ppx), "ppy": float(intr.ppy)},
    }
    return rgb, depth, xyz, metadata


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError("zero-length table normal")
    return np.asarray(vector, dtype=np.float64) / norm


def fit_table_plane(points: np.ndarray, distance_m: float, samples: int, iterations: int) -> np.ndarray:
    """Exact Open3D plane-estimation sequence used by sam2_tabletop_masks.py."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 500:
        raise RuntimeError("too few valid RGB-D points to estimate tabletop")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    coarse = cloud.voxel_down_sample(0.003)
    plane, _ = coarse.segment_plane(float(distance_m), 3, int(iterations))
    plane = np.asarray(plane, dtype=np.float64)
    normal = normalize(plane[:3])
    if float(normal @ (-points.mean(axis=0))) < 0.0:
        plane *= -1.0
    return plane


def distributed_positive_points(labels: np.ndarray, label: int, count: int = 3) -> list[tuple[int, int]]:
    """Return distributed positive SAM2 prompts for one seed component/workpiece."""
    rows, columns = np.nonzero(labels == label)
    points = np.column_stack((columns, rows))
    if len(points) == 0:
        return []
    chosen = [int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))]
    nearest = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(1, min(int(count), len(points))):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
    return [tuple(int(value) for value in points[index]) for index in chosen]


def build_seed_groups(rgb: np.ndarray, xyz: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    """Return table-height/darkness seeds and one point-only SAM2 prompt group per workpiece."""
    height, width = rgb.shape[:2]
    # Equivalent to main.py's depth condition: 0 < raw depth < 65535.
    valid = np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0.0) & (xyz[:, :, 2] < 65.0)
    plane = fit_table_plane(xyz[valid], args.plane_distance_mm * 1e-3, args.plane_samples, args.plane_iters)
    heights = xyz @ plane[:3] + plane[3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    roi, _ = tabletop_roi_mask(args.roi, width, height)
    raw_seed = valid & roi & (heights >= args.seed_height_mm * 1e-3) & (gray <= args.seed_max_gray)
    cleaned = cv2.morphologyEx(raw_seed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cleaned = cv2.dilate(cleaned, np.ones((5, 5), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    kept = [label for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= args.min_seed_area]
    # A connected component is one physical workpiece.  Keep components
    # separate: each receives its own point-only SAM2 request.
    groups = [{"seed_label": int(label), "positives_xy": distributed_positive_points(labels, label)} for label in kept]
    return raw_seed, cleaned.astype(bool), groups, plane


def retain_prompt_components(mask: np.ndarray, positives: list[tuple[int, int]], seed_mask: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    selected = {int(labels[y, x]) for x, y in positives if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0] and labels[y, x] > 0}
    if not selected and count > 1:
        overlaps = [int(np.count_nonzero((labels == label) & seed_mask)) for label in range(1, count)]
        selected = {1 + int(np.argmax(overlaps))}
    return np.isin(labels, list(selected)) if selected else np.zeros_like(mask, dtype=bool)


def ensure_sam2_service() -> None:
    """Start the same resident SAM2 service used by desktop_grasp/main.py."""
    if SAM2_SOCKET.exists():
        return
    if not SAM2_BIN.is_file():
        raise RuntimeError(f"SAM2 service binary is missing: {SAM2_BIN}")
    log = DEFAULT_CAPTURE_ROOT / "sam2-service.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        subprocess.Popen(
            [str(SAM2_BIN), "--socket", str(SAM2_SOCKET), "--backend", "efficientvit", "--queue-capacity", "4", "--request-timeout-s", "90"],
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    for _ in range(120):
        if SAM2_SOCKET.exists():
            return
        time.sleep(0.5)
    raise RuntimeError(f"SAM2 service did not create {SAM2_SOCKET}; inspect {log}")


def prompt_segment(rgb: np.ndarray, positives: list[tuple[int, int]], sam2_url: str) -> np.ndarray:
    """Segment from positive points only, using the resident HTTP-over-UDS SAM2 API."""
    payload = io.BytesIO()
    # The deployed service requires a ``box`` archive entry for backwards
    # compatibility, but does not send it to SAM2.  A full-image value makes
    # its post-processing crop a no-op, so no spatial box constraint is used.
    height, width = rgb.shape[:2]
    np.savez_compressed(payload, image_bgr=rgb, points=np.asarray(positives, dtype=np.int32), labels=np.ones(len(positives), dtype=np.int32), box=np.asarray([0, 0, width - 1, height - 1], dtype=np.int32))
    endpoint = urlparse(sam2_url)
    if endpoint.scheme != "unix" or not endpoint.path:
        raise ValueError("--sam2-url must be a unix:// socket URL")
    class UnixConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(90.0)
            self.sock.connect(endpoint.path)
    connection = UnixConnection("sam2-service", timeout=90.0)
    try:
        connection.request("POST", "/v1/prompt-segment-sam2", body=payload.getvalue(), headers={"Content-Type": "application/x-sam2-prompt-npz"})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM2 returned HTTP {response.status}: {body.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return np.asarray(archive["mask"] > 0, dtype=bool)


def sam2_mask(rgb: np.ndarray, seed_mask: np.ndarray, groups: list[dict[str, object]], roi: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, object]], str]:
    if not groups:
        return np.zeros(rgb.shape[:2], dtype=bool), [], "not-run-no-seed-groups"
    ensure_sam2_service()
    combined = np.zeros(rgb.shape[:2], dtype=bool)
    diagnostics: list[dict[str, object]] = []
    for group in groups:
        selected = retain_prompt_components(prompt_segment(rgb, group["positives_xy"], args.sam2_url) & roi, group["positives_xy"], seed_mask)
        accepted = int(np.count_nonzero(selected)) >= args.min_mask_area
        if accepted:
            combined |= selected
        diagnostics.append({**group, "mask_pixels": int(np.count_nonzero(selected)), "accepted": accepted})
    return combined, diagnostics, "sam2-service/efficientvit"


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def planar_basis(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    centered -= (centered @ normal)[:, None] * normal
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    first = normalize(vectors[:, -1] - normal * np.dot(vectors[:, -1], normal))
    return first, normalize(np.cross(normal, first))


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = normalize(axis)
    cosine, sine, remainder = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([[cosine+x*x*remainder, x*y*remainder-z*sine, x*z*remainder+y*sine], [y*x*remainder+z*sine, cosine+y*y*remainder, y*z*remainder-x*sine], [z*x*remainder-y*sine, z*y*remainder+x*sine, cosine+z*z*remainder]])


def fpfh_features(points: np.ndarray, voxel_m: float) -> tuple[o3d.geometry.PointCloud, object]:
    """Compute the deterministic geometry/features shared by all RANSAC seeds."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down = cloud.voxel_down_sample(voxel_m)
    if len(down.points) < 40:
        raise RuntimeError("too few points after global-registration voxel downsampling")
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 2.5, max_nn=40))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 5.0, max_nn=100)
    )
    return down, feature


def fpfh_ransac_initial(source: np.ndarray, target: np.ndarray, voxel_m: float, iterations: int,
                        seed: int = 0, *, prepared_features: tuple | None = None) -> np.ndarray:
    """Find a full-6DoF CAD-to-observation initial pose without using the table plane.

    The workpiece may lean on the table, so neither STL Z nor any STL plane is
    constrained to the table normal.  FPFH/RANSAC supplies a global 3-D seed;
    the custom ICP below then refines all three translation and rotation axes.
    """
    o3d.utility.random.seed(int(seed))
    if prepared_features is None:
        source_down, source_feature = fpfh_features(source, voxel_m)
        target_down, target_feature = fpfh_features(target, voxel_m)
    else:
        source_down, source_feature, target_down, target_feature = prepared_features
    # A thin, partly observed workpiece often has too few bidirectional FPFH
    # matches, so Open3D deliberately falls back to its unfiltered matches.
    # Keep that behavior unchanged while hiding the repetitive fallback notice.
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_feature, target_feature, True, voxel_m * 3.0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.80),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_m * 3.0),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(int(iterations), 0.999),
        )
    if result.fitness < 0.03 or not np.isfinite(result.inlier_rmse):
        raise RuntimeError(f"FPFH/RANSAC could not establish a CAD pose (fitness={result.fitness:.3f})")
    return np.asarray(result.transformation, dtype=np.float64)


def semantic_half_turn_initials(initial: np.ndarray, stl_semantic: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return the FPFH seed plus 180° semantic-frame alternatives.

    FPFH descriptors can confuse the two sides of this thin, partly occluded
    bracket.  The alternatives preserve a proper right-handed rotation and
    give ICP a seed for the opposite Z-facing interpretation.
    """
    turns = (("fpfh", np.eye(3)), ("flip_about_semantic_x", np.diag((1.0, -1.0, -1.0))),
             ("flip_about_semantic_y", np.diag((-1.0, 1.0, -1.0))),
             ("flip_about_semantic_z", np.diag((-1.0, -1.0, 1.0))))
    inverse = np.linalg.inv(stl_semantic)
    candidates = []
    for name, rotation_semantic in turns:
        candidate = initial.copy()
        candidate[:3, :3] = initial[:3, :3] @ stl_semantic[:3, :3] @ rotation_semantic @ inverse[:3, :3]
        candidates.append((name, candidate))
    return candidates


def depth_score(T: np.ndarray, cad_render: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, scale: float) -> float:
    """Score a pose by z-buffer rendering the CAD against observed depth.

    A previous 2-D nearest-neighbour approximation let multiple CAD surfaces
    on the same camera ray participate in scoring.  In particular, a surface
    hidden behind the front face could accidentally explain a depth pixel.
    Here each CAD sample is rasterised into its nearest pixel and ``minimum.at``
    retains only the smallest camera-Z value: the physically visible surface.
    """
    height, width = depth.shape[:2]
    points = transform_points(T, cad_render)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)]
    if len(points) == 0:
        return float("inf")
    columns = np.rint(K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]).astype(np.intp)
    rows = np.rint(K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]).astype(np.intp)
    inside = (0 <= columns) & (columns < width) & (0 <= rows) & (rows < height)
    if not np.any(inside):
        return float("inf")

    # Z-buffer: for every image pixel, retain the nearest CAD surface only.
    rendered_z = np.full((height, width), np.inf, dtype=np.float64)
    pixel_indices = rows[inside] * width + columns[inside]
    np.minimum.at(rendered_z.ravel(), pixel_indices, points[inside, 2])

    observed = mask.astype(bool) & (depth > 0)
    rendered = np.isfinite(rendered_z)
    overlap = observed & rendered
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count < 80:
        return float("inf")
    observed_z = depth[overlap].astype(np.float64) * scale
    residual = np.abs(rendered_z[overlap] - observed_z)

    # Score both depth agreement and visible-silhouette agreement.  The latter
    # penalises CAD pixels outside the SAM2 component as well as observed
    # component pixels that the rendered CAD does not cover.
    union_count = int(np.count_nonzero(observed | rendered))
    coverage_error = 1.0 - overlap_count / union_count
    return float(np.mean(np.minimum(residual, 0.020)) + 0.004 * coverage_error)


def semantic_landmarks(vertices: np.ndarray, stl_semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(stl_semantic)
    semantic = transform_points(inverse, vertices)
    low, high = np.percentile(semantic[:, 0], (2.5, 97.5))
    # Workpiece 2 defines semantic origin as the STL bounding-box centre,
    # rather than as a mesh-vertex centroid.
    center = np.zeros(3, dtype=np.float64)
    tail, head = semantic[semantic[:, 0] <= low].mean(axis=0), semantic[semantic[:, 0] >= high].mean(axis=0)
    return center, tail, head


def matrix_to_rpy_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic ZYX yaw/pitch/roll as roll, pitch, yaw in radians."""
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return roll, pitch, yaw


def downsample_points(points: np.ndarray, voxel_m: float = .003, maximum: int = 5000) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64); points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0: return points.reshape(0, 3)
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    points = points[np.sort(first)]
    return points[:maximum]


def icp_pose_for_mask(component: np.ndarray, xyz: np.ndarray, depth: np.ndarray, depth_scale: float, camera_matrix: np.ndarray, distortion: np.ndarray, base_camera: np.ndarray, table_plane: np.ndarray, cad: np.ndarray, cad_render: np.ndarray, stl_semantic: np.ndarray, positive_face_axis: np.ndarray, landmarks: tuple[np.ndarray, np.ndarray, np.ndarray, float], args: argparse.Namespace, diagnostic: dict) -> dict:
    observed = downsample_points(xyz[component], maximum=args.max_observed_points)
    if len(observed) < args.min_icp_points: raise RuntimeError(f"insufficient mask depth: {len(observed)} points")
    table_plane = np.asarray(table_plane, dtype=np.float64)
    table_normal = normalize(table_plane[:3])
    voxel_m = args.icp_global_voxel_mm * 1e-3
    source_down, source_feature = fpfh_features(cad, voxel_m)
    target_down, target_feature = fpfh_features(observed, voxel_m)
    prepared_features = (source_down, source_feature, target_down, target_feature)
    observed_center = np.median(observed, axis=0)

    def refine(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, float, float]:
        initialization, candidate = item
        candidate[:3, 3] = observed_center
        transform, rmse, fitness = full_6d_point_to_plane_icp(
            cad, observed, candidate,
            max_iterations=args.icp_iters,
            max_corr_m=args.icp_max_corr_mm * 1e-3,
        )
        return initialization, transform, rmse, fitness

    trials: dict[str, list[dict[str, object]]] = {"FRONT": [], "BACK": []}
    # The four semantic half-turn hypotheses are independent.  Run them in a
    # fixed-size pool while keeping RANSAC seeds sequential (Open3D's RNG is
    # process-global).  executor.map preserves the original candidate order.
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="icp-hypothesis") as executor:
        for seed in range(args.icp_global_seeds):
            initial = fpfh_ransac_initial(
                cad, observed, voxel_m, args.icp_ransac_iters, seed,
                prepared_features=prepared_features,
            )
            refined = executor.map(refine, semantic_half_turn_initials(initial, stl_semantic))
            for initialization, transform, rmse, fitness in refined:
                # STL (and semantic) origin is the defined workpiece centre.
                # It was anchored at the robust observed-cloud centre before
                # local ICP so a fold cannot be mistaken for the whole part.
                T_camera_semantic = transform @ stl_semantic
                positive_face_camera = T_camera_semantic[:3, :3] @ positive_face_axis
                face_up_dot = float(positive_face_camera @ table_normal)
                origin_pixel, _ = cv2.projectPoints(
                    T_camera_semantic[:3, 3][None, :], np.zeros(3), np.zeros(3), camera_matrix, distortion,
                )
                ox, oy = np.rint(origin_pixel.reshape(2)).astype(int)
                origin_in_mask = (
                    0 <= ox < component.shape[1] and 0 <= oy < component.shape[0] and bool(component[oy, ox])
                )
                table_height = transform_points(transform, cad) @ table_normal + table_plane[3]
                penetration_m = float(max(0.0, -np.min(table_height)))
                if not origin_in_mask or penetration_m > args.max_table_penetration_mm * 1e-3:
                    continue
                side = (
                    "FRONT" if face_up_dot >= args.minimum_z_up_dot else
                    "BACK" if face_up_dot <= -args.minimum_z_up_dot else None
                )
                if side is not None:
                    score = depth_score(transform, cad_render, depth, component, camera_matrix, depth_scale)
                    if np.isfinite(score):
                        trials[side].append({
                            "score": score,
                            "initialization": f"fpfh_seed_{seed}/{initialization}",
                            "transform": transform,
                            "rmse": rmse,
                            "fitness": fitness,
                            "face_up_dot": face_up_dot,
                            "origin_pixel": [int(ox), int(oy)],
                            "penetration_m": penetration_m,
                        })
    best = {
        side: min(candidates, key=lambda candidate: float(candidate["score"]))
        for side, candidates in trials.items() if candidates
    }
    if not best:
        raise RuntimeError(
            "no full-6DoF ICP pose satisfies either configured face direction, origin, and table constraints; "
            "check the mask, the CAD semantic frame, or --minimum-z-up-dot"
        )
    if len(best) == 2:
        front_score, back_score = float(best["FRONT"]["score"]), float(best["BACK"]["score"])
        margin_m = abs(front_score - back_score)
        orientation = min(best, key=lambda side: float(best[side]["score"])) if margin_m >= args.minimum_face_margin_mm * 1e-3 else "OTHER"
    else:
        margin_m = None
        orientation = "OTHER"
    selected_side = orientation if orientation in best else min(best, key=lambda side: float(best[side]["score"]))
    selected = best[selected_side]
    score = float(selected["score"])
    initialization = str(selected["initialization"])
    T_camera_stl = np.asarray(selected["transform"], dtype=np.float64)
    rmse = float(selected["rmse"])
    fitness = float(selected["fitness"])
    face_up_dot = float(selected["face_up_dot"])
    origin_pixel = list(selected["origin_pixel"])
    penetration_m = float(selected["penetration_m"])

    def candidate_pose(candidate: dict[str, object]) -> tuple[np.ndarray, list[float]]:
        transform = np.asarray(candidate["transform"], dtype=np.float64)
        semantic = base_camera @ transform @ stl_semantic
        rpy = matrix_to_rpy_zyx(semantic[:3, :3])
        return semantic, [float(math.degrees(value)) for value in rpy]

    # A low local ICP residual can still leave several materially different
    # poses nearly tied.  Preserve the within-face leaderboard and the spread
    # of candidates close to the best score as a confidence diagnostic.
    candidate_confidence: dict[str, object] = {}
    score_window_m = args.near_optimal_score_window_mm * 1e-3
    for side, candidates in trials.items():
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda candidate: float(candidate["score"]))
        best_candidate = ranked[0]
        best_transform, best_rpy = candidate_pose(best_candidate)
        best_score = float(best_candidate["score"])
        top_three = []
        for rank, candidate in enumerate(ranked[:3], start=1):
            semantic, rpy = candidate_pose(candidate)
            relative_rotation = best_transform[:3, :3].T @ semantic[:3, :3]
            cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
            top_three.append({
                "rank": rank,
                "depth_score_mm": float(candidate["score"]) * 1000.0,
                "score_gap_from_best_mm": (float(candidate["score"]) - best_score) * 1000.0,
                "center_base_m": semantic[:3, 3].tolist(),
                "rpy_deg": rpy,
                "center_deviation_from_best_mm": float(np.linalg.norm(semantic[:3, 3] - best_transform[:3, 3])) * 1000.0,
                "rotation_deviation_from_best_deg": math.degrees(math.acos(cosine)),
                "initialization": str(candidate["initialization"]),
            })
        near_optimal = [candidate for candidate in ranked if float(candidate["score"]) <= best_score + score_window_m]
        near_position_deviations_mm, near_rotation_deviations_deg = [], []
        near_rpys = []
        for candidate in near_optimal:
            semantic, rpy = candidate_pose(candidate)
            relative_rotation = best_transform[:3, :3].T @ semantic[:3, :3]
            cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
            near_position_deviations_mm.append(float(np.linalg.norm(semantic[:3, 3] - best_transform[:3, 3])) * 1000.0)
            near_rotation_deviations_deg.append(math.degrees(math.acos(cosine)))
            near_rpys.append(rpy)
        candidate_confidence[side] = {
            "candidate_count": len(ranked),
            "top_three": top_three,
            "top_three_score_gaps_mm": [entry["score_gap_from_best_mm"] for entry in top_three],
            "near_optimal_score_window_mm": args.near_optimal_score_window_mm,
            "near_optimal_candidate_count": len(near_optimal),
            "near_optimal_max_center_deviation_from_best_mm": max(near_position_deviations_mm),
            "near_optimal_max_rotation_deviation_from_best_deg": max(near_rotation_deviations_deg),
            "near_optimal_rpy_component_span_deg": (
                (np.max(np.asarray(near_rpys), axis=0) - np.min(np.asarray(near_rpys), axis=0)).tolist()
            ),
        }

    selected_candidate_confidence = candidate_confidence[selected_side]
    near_count = int(selected_candidate_confidence["near_optimal_candidate_count"])
    near_center_deviation_mm = float(selected_candidate_confidence["near_optimal_max_center_deviation_from_best_mm"])
    near_rotation_deviation_deg = float(selected_candidate_confidence["near_optimal_max_rotation_deviation_from_best_deg"])
    score_mm = score * 1000.0
    rmse_mm = rmse * 1000.0
    face_margin_mm = None if margin_m is None else margin_m * 1000.0
    thresholds = POSE_CONFIDENCE_THRESHOLDS
    low_reasons: list[str] = []
    if orientation == "OTHER":
        low_reasons.append("front_back_orientation_ambiguous")
    if len(observed) < thresholds["acceptable_min_observed_points"]:
        low_reasons.append("too_few_observed_points")
    if fitness < thresholds["acceptable_min_fitness"]:
        low_reasons.append("low_icp_fitness")
    if rmse_mm > thresholds["acceptable_max_rmse_mm"]:
        low_reasons.append("high_icp_rmse")
    if score_mm > thresholds["acceptable_max_depth_score_mm"]:
        low_reasons.append("high_depth_score")
    if (face_margin_mm is not None
            and face_margin_mm < thresholds["acceptable_min_face_margin_mm"]):
        low_reasons.append("insufficient_front_back_score_margin")
    if (near_count >= 2 and (
            near_center_deviation_mm > thresholds["reject_near_optimal_center_deviation_mm"]
            or near_rotation_deviation_deg > thresholds["reject_near_optimal_rotation_deviation_deg"]
    )):
        low_reasons.append("near_optimal_candidates_disagree")
    high_quality = (
        orientation != "OTHER"
        and len(observed) >= thresholds["high_min_observed_points"]
        and fitness >= thresholds["high_min_fitness"]
        and rmse_mm <= thresholds["high_max_rmse_mm"]
        and score_mm <= thresholds["high_max_depth_score_mm"]
        and face_margin_mm is not None
        and face_margin_mm >= thresholds["high_min_face_margin_mm"]
        and (near_count == 1 or (
            near_center_deviation_mm <= thresholds["high_max_near_optimal_center_deviation_mm"]
            and near_rotation_deviation_deg <= thresholds["high_max_near_optimal_rotation_deviation_deg"]
        ))
    )
    advisories: list[str] = []
    if near_count >= 2 and (
            near_center_deviation_mm > thresholds["acceptable_max_near_optimal_center_deviation_mm"]
            or near_rotation_deviation_deg > thresholds["acceptable_max_near_optimal_rotation_deviation_deg"]
    ):
        advisories.append("near_optimal_candidate_variation")
    top_three = selected_candidate_confidence["top_three"]
    if len(top_three) >= 2 and float(top_three[1]["score_gap_from_best_mm"]) < args.near_optimal_score_window_mm:
        advisories.append("runner_up_inside_near_optimal_score_window")
    pose_confidence = {
        "level": "LOW" if low_reasons else ("HIGH" if high_quality else "MEDIUM"),
        "recommended_for_grasp": not low_reasons,
        "report_only": True,
        "low_reasons": low_reasons,
        "advisories": advisories,
        "thresholds": thresholds,
        "checks": {
            "orientation": orientation,
            "observed_points": int(len(observed)),
            "icp_fitness": fitness,
            "icp_rmse_mm": rmse_mm,
            "depth_score_mm": score_mm,
            "front_back_score_margin_mm": face_margin_mm,
            "near_optimal_candidate_count": near_count,
            "near_optimal_max_center_deviation_mm": near_center_deviation_mm,
            "near_optimal_max_rotation_deviation_deg": near_rotation_deviation_deg,
        },
    }
    T_base_semantic = base_camera @ T_camera_stl @ stl_semantic
    T_camera_semantic = T_camera_stl @ stl_semantic
    center_s, tail_s, head_s = landmarks
    center_b, tail_b, head_b = transform_points(T_base_semantic, np.stack((center_s, tail_s, head_s)))
    center_c, tail_c, head_c = transform_points(T_camera_semantic, np.stack((center_s, tail_s, head_s)))
    pixels, _ = cv2.projectPoints(np.stack((center_c, head_c, tail_c)), np.zeros(3), np.zeros(3), camera_matrix, distortion)
    center_px, head_px, tail_px = pixels.reshape(-1, 2)
    direction = head_b - tail_b
    horizontal_direction = direction[:2]
    horizontal_length = float(np.linalg.norm(horizontal_direction))
    if horizontal_length < 1e-7:
        raise RuntimeError("workpiece semantic +X is nearly vertical; horizontal yaw is undefined")
    yaw = math.degrees(math.atan2(float(horizontal_direction[1]), float(horizontal_direction[0]))) % 360.0
    quadrant = int(yaw // 90.0) + 1
    roll, pitch, yaw_rpy = matrix_to_rpy_zyx(T_base_semantic[:3, :3])
    # Retain mask-only measurements (area/box/width) as diagnostics, but replace every pose field with ICP/CAD geometry.
    record = dict(diagnostic)
    face_hypotheses = {
        side: {
            "depth_score_mm": float(candidate["score"]) * 1000,
            "rmse_mm": float(candidate["rmse"]) * 1000,
            "fitness": float(candidate["fitness"]),
            "face_up_dot_table_normal": float(candidate["face_up_dot"]),
            "initialization": str(candidate["initialization"]),
        }
        for side, candidate in best.items()
    }
    record.update({
        "center_pixel_xy": center_px.tolist(), "head_pixel_xy": head_px.tolist(),
        "tail_pixel_xy": tail_px.tolist(), "center_base_xy_m": center_b[:2].tolist(),
        "center_base_xy_mm": (center_b[:2] * 1000).tolist(), "head_base_xy_m": head_b[:2].tolist(),
        "tail_base_xy_m": tail_b[:2].tolist(), "head_direction_base_vector": direction.tolist(),
        "head_direction_base_xy_unit": (horizontal_direction / horizontal_length).tolist(),
        "head_direction_base_angle_deg": yaw, "head_direction_base_quadrant": quadrant,
        "robot_base_xy_yaw": {"x_m": float(center_b[0]), "y_m": float(center_b[1]),
            "x_mm": float(center_b[0] * 1000), "y_mm": float(center_b[1] * 1000),
            "yaw_rad": math.radians(yaw), "yaw_deg": yaw, "yaw_quadrant": quadrant,
            "yaw_definition": "horizontal projection of CAD semantic +X (tail to head), estimated by full-6DoF mask-local point-cloud ICP"},
        "robot_base_xyz_rpy": {"x_m": float(center_b[0]), "y_m": float(center_b[1]), "z_m": float(center_b[2]),
            "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw_rpy, "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw_rpy) % 360.0,
            "rpy_convention": "intrinsic ZYX (yaw, then pitch, then roll)"},
        "pose_body_base": {"x_m": float(center_b[0]), "y_m": float(center_b[1]), "z_m": float(center_b[2]),
            "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw_rpy, "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw_rpy) % 360.0,
            "rpy_convention": "intrinsic ZYX (yaw, then pitch, then roll)", "rotation_matrix": T_base_semantic[:3, :3].tolist()},
        "orientation": orientation,
        "object_side": {"FRONT": "positive_face_full_6dof", "BACK": "reverse_face_full_6dof", "OTHER": "ambiguous_face_full_6dof"}[orientation],
        "orientation_score_margin_mm": None if margin_m is None else margin_m * 1000,
        "required_orientation_margin_mm": args.minimum_face_margin_mm,
        "face_hypotheses": face_hypotheses, "candidate_confidence": candidate_confidence,
        "pose_confidence": pose_confidence,
        "icp": {"rmse_mm": rmse * 1000, "fitness": fitness, "depth_score_mm": score * 1000,
            "observed_points": int(len(observed)), "initial_hypotheses": int(args.icp_global_seeds * 4),
            "initialization": initialization, "positive_face_axis_semantic": positive_face_axis.tolist(),
            "positive_face_up_dot_table_normal": face_up_dot, "origin_pixel_xy": origin_pixel,
            "table_penetration_mm": penetration_m * 1000,
            "origin_constraint": "projected semantic origin must lie in the SAM2 workpiece mask"},
        "T_head_camera_stl": T_camera_stl.tolist(), "T_head_camera_workpiece": T_camera_semantic.tolist(),
        "T_head_camera_workpiece_semantic": T_camera_semantic.tolist(),
        "T_body_base_workpiece_semantic": T_base_semantic.tolist(),
    })
    return record


def write_seed_debug(rgb: np.ndarray, raw_seed: np.ndarray, cleaned_seed: np.ndarray, groups: list[dict[str, object]], output: Path) -> None:
    debug = rgb.copy()
    debug[raw_seed] = (0.45 * debug[raw_seed] + np.array([0, 180, 255])).astype(np.uint8)
    debug[cleaned_seed] = (0.40 * debug[cleaned_seed] + np.array([0, 255, 0])).astype(np.uint8)
    for index, group in enumerate(groups, 1):
        for point in group["positives_xy"]:
            cv2.circle(debug, tuple(point), 5, (0, 0, 255), -1, cv2.LINE_AA)
        anchor = tuple(int(value) for value in group["positives_xy"][0])
        cv2.putText(debug, f"SAM2 #{index}", (anchor[0] + 8, max(18, anchor[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), debug):
        raise RuntimeError(f"failed to write {output}")


def result_payload(mask_path: Path, calibration: Path, plane_z: float, poses: list[dict], rejected: list[dict],
                   *, workpiece: str, stl: Path, semantic_calibration: Path) -> dict[str, object]:
    return {
        "mask": str(mask_path), "calibration": str(calibration), "coordinate_frame": "body_base", "plane_z_m": plane_z,
        "angle_convention": {
            "base": "tail_to_head; body_base +X toward +Y, counter-clockwise",
            "image": "tail_to_head; right/up is quadrant I, left/up II, left/down III, right/down IV",
        },
        "workpiece_asset": workpiece,
        "stl": str(stl),
        "semantic_calibration": str(semantic_calibration),
        "workpiece_semantic_frame": "Configured by semantic_calibration; p_stl = T_stl_workpiece_semantic @ p_workpiece_semantic; +Z=+X cross +Y",
        "workpieces": poses, "rejected_components": rejected,
    }


def main() -> int:
    args = parse_args()
    if (min(args.plane_distance_mm, args.seed_height_mm, args.min_seed_area, args.min_mask_area,
            args.min_pose_area, args.icp_workers) <= 0
            or args.minimum_face_margin_mm < 0 or args.near_optimal_score_window_mm < 0
            or args.low_confidence_retries < 0):
        raise ValueError("thresholds and minimum areas must be positive")
    for path in (args.calibration, args.stl, args.semantic_calibration):
        if not path.is_file(): raise FileNotFoundError(path)
    workpiece_capture_root = args.capture_root / f"workpiece_{args.workpiece}"
    workpiece_capture_root.mkdir(parents=True, exist_ok=True)
    validate_calibration_capture_pose(args)
    head_move = move_head_to_capture_pose(args)
    camera_matrix, distortion, t_base_camera, calibration_size = load_calibration(args.calibration)
    triangles = load_stl_triangles(args.stl)
    vertices = triangles.reshape(-1, 3)
    stl_semantic = load_semantic_calibration(args.semantic_calibration)
    positive_face_axis = load_positive_face_axis(args.semantic_calibration)
    landmarks = semantic_landmarks(vertices, stl_semantic)
    cad = sample_stl_surface(args.stl, args.model_samples)
    cad_render = sample_stl_surface(args.stl, args.render_samples)

    def low_attempt_key(attempt_poses: list[dict]) -> tuple[float, float, float, float]:
        """Lower is better when no attempt has reached MEDIUM confidence."""
        penalties = []
        for pose in attempt_poses:
            confidence = pose["pose_confidence"]
            checks, thresholds = confidence["checks"], confidence["thresholds"]
            penalties.append(sum((
                max(0.0, (thresholds["acceptable_min_observed_points"] - checks["observed_points"])
                    / thresholds["acceptable_min_observed_points"]),
                max(0.0, (thresholds["acceptable_min_fitness"] - checks["icp_fitness"])
                    / thresholds["acceptable_min_fitness"]),
                max(0.0, (checks["icp_rmse_mm"] - thresholds["acceptable_max_rmse_mm"])
                    / thresholds["acceptable_max_rmse_mm"]),
                max(0.0, (checks["depth_score_mm"] - thresholds["acceptable_max_depth_score_mm"])
                    / thresholds["acceptable_max_depth_score_mm"]),
                max(0.0, (checks["near_optimal_max_center_deviation_mm"]
                    - thresholds["reject_near_optimal_center_deviation_mm"])
                    / thresholds["reject_near_optimal_center_deviation_mm"]),
                max(0.0, (checks["near_optimal_max_rotation_deviation_deg"]
                    - thresholds["reject_near_optimal_rotation_deviation_deg"])
                    / thresholds["reject_near_optimal_rotation_deviation_deg"]),
                1.0 if checks["orientation"] == "OTHER" else 0.0,
            )))
        return (
            float(sum(len(pose["pose_confidence"]["low_reasons"]) for pose in attempt_poses)),
            float(sum(penalties)),
            -float(sum(pose["icp"]["fitness"] for pose in attempt_poses)),
            float(sum(pose["icp"]["depth_score_mm"] for pose in attempt_poses)),
        )

    def back_icp_key(attempt_poses: list[dict]) -> tuple[float, float, float]:
        """Rank two independently captured BACK results; lower is better.

        ``depth_score_mm`` is the same score used to choose the winning ICP
        hypothesis within a capture.  RMSE and fitness make the choice
        deterministic when the depth scores are tied.
        """
        return (
            float(sum(pose["icp"]["depth_score_mm"] for pose in attempt_poses)),
            float(sum(pose["icp"]["rmse_mm"] for pose in attempt_poses)),
            -float(sum(pose["icp"]["fitness"] for pose in attempt_poses)),
        )

    # Keep only the best rejected attempt in a private temporary directory;
    # numeric output directories remain reserved for accepted/fallback results.
    with tempfile.TemporaryDirectory(prefix=".pending_pose_best_", dir=workpiece_capture_root) as best_name, \
            tempfile.TemporaryDirectory(prefix=".pending_pose_back_", dir=workpiece_capture_root) as back_name:
        best_dir = Path(best_name)
        back_dir = Path(back_name)
        best_key: tuple[float, float, float, float] | None = None
        best_pose_json: dict | None = None
        best_metadata: dict | None = None
        best_pose_count = 0
        first_back_key: tuple[float, float, float] | None = None
        first_back_pose_json: dict | None = None
        first_back_metadata: dict | None = None
        first_back_pose_count = 0
        # A BACK detection consumes one additional capture for confirmation;
        # low-confidence retries retain their existing allowance.
        attempt = 1
        attempt_limit = args.low_confidence_retries + 1
        while attempt <= attempt_limit:
            # Do not expose rejected captures in the numeric output sequence.
            # TemporaryDirectory removes images/point clouds after each attempt.
            with tempfile.TemporaryDirectory(prefix=".pending_pose_", dir=workpiece_capture_root) as pending_name:
                pending_dir = Path(pending_name)
                rgb, depth, xyz, capture_metadata = capture_rgbd(args, pending_dir)
                if calibration_size != (rgb.shape[1], rgb.shape[0]):
                    raise RuntimeError(f"capture resolution {rgb.shape[1]}x{rgb.shape[0]} differs from calibration {calibration_size}")
                raw_seed, cleaned_seed, groups, plane = build_seed_groups(rgb, xyz, args)
                cv2.imwrite(str(pending_dir / "seed_raw.png"), raw_seed.astype(np.uint8) * 255)
                cv2.imwrite(str(pending_dir / "seed_cleaned.png"), cleaned_seed.astype(np.uint8) * 255)
                write_seed_debug(rgb, raw_seed, cleaned_seed, groups, pending_dir / "seed_prompts.png")
                roi, roi_polygon = tabletop_roi_mask(args.roi, rgb.shape[1], rgb.shape[0])
                mask, group_diagnostics, device = sam2_mask(rgb, cleaned_seed, groups, roi, args)
                pending_mask_path = pending_dir / "mask.png"
                if not cv2.imwrite(str(pending_mask_path), mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"failed to write {pending_mask_path}")
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                labels_to_pose = [label for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= args.min_pose_area]
                # Area order only sets the ICP processing order.  The public
                # workpieces[] order is assigned after all valid poses have
                # been estimated, using body_base Y (right to left).
                labels_to_pose.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
                poses, rejected = [], []
                for label in labels_to_pose:
                    try:
                        component = labels == label
                        diagnostic = component_pose(component, camera_matrix, distortion, t_base_camera, args.plane_z)
                        poses.append(icp_pose_for_mask(component, xyz, depth, float(capture_metadata["depth_scale_m"]), camera_matrix, distortion, t_base_camera, plane, cad, cad_render, stl_semantic, positive_face_axis, landmarks, args, diagnostic))
                    except RuntimeError as exc:
                        rejected.append({"label": int(label), "area_px": int(stats[label, cv2.CC_STAT_AREA]), "reason": str(exc)})
                if not poses:
                    print(f"[retry] attempt={attempt}/{args.low_confidence_retries + 1} no valid pose discarded")
                    attempt += 1
                    continue

                # The calibrated head camera maps image-right to decreasing
                # body_base Y.  Publish the rightmost workpiece first, so
                # workpieces[0] is the rightmost valid object and subsequent
                # entries proceed from right to left.  Use X only as a stable
                # tie-breaker when two centres have the same Y coordinate.
                poses.sort(key=lambda pose: (
                    float(pose["center_base_xy_m"][1]),
                    float(pose["center_base_xy_m"][0]),
                ))

                write_icp_workpiece_frame(
                    rgb, poses, camera_matrix, distortion, pending_dir / "rgb_icp_workpiece_frame.png",
                    roi_polygon,
                )
                pose_json = result_payload(
                    pending_mask_path, args.calibration, args.plane_z, poses, rejected,
                    workpiece=args.workpiece, stl=args.stl, semantic_calibration=args.semantic_calibration,
                )
                metadata = {
                    **capture_metadata, "head_move": head_move, "confidence_attempt": attempt,
                    "low_confidence_retries_allowed": args.low_confidence_retries,
                    "confidence_fallback_used": False,
                    "calibration": str(args.calibration), "table_plane_head_camera": plane.tolist(),
                    "roi_normalized_rect_override": list(args.roi) if args.roi is not None else None,
                    "roi_polygon_px": roi_polygon.tolist(), "seed_height_mm": args.seed_height_mm, "seed_max_gray": args.seed_max_gray,
                    "seed_groups": groups, "sam2_groups": group_diagnostics, "sam2_device": device, "pose_source": "mask-local point cloud + STL multi-start full-6DoF ICP",
                    "mask_foreground_pixels": int(np.count_nonzero(mask)), "pose_components": len(poses),
                    "files": {"rgb": "rgb.png", "rgbd": "rgbd.npz", "seed_raw": "seed_raw.png", "seed_cleaned": "seed_cleaned.png", "seed_prompts": "seed_prompts.png", "mask": "mask.png", "pose_json": "mask_pose.json", "icp_workpiece_frame": "rgb_icp_workpiece_frame.png"},
                }
                # ``poses`` was sorted above from image-right to image-left.
                # Module 8.1 operates workpieces[0], so only its orientation
                # controls BACK confirmation.  A non-target workpiece being
                # BACK must not trigger another capture or block the target.
                is_back_detection = bool(poses) and poses[0]["orientation"] == "BACK"
                if is_back_detection:
                    # Choose between the two confirmation captures by the
                    # rightmost target's ICP quality only; other workpieces
                    # are diagnostics and must not influence this decision.
                    current_back_key = back_icp_key([poses[0]])
                    if first_back_pose_json is None:
                        # Do not publish a single BACK detection: a fresh
                        # RGB-D capture is required before declaring BACK.
                        shutil.copytree(pending_dir, back_dir, dirs_exist_ok=True)
                        first_back_key = current_back_key
                        first_back_pose_json = pose_json
                        first_back_metadata = metadata
                        first_back_pose_count = len(poses)
                        attempt_limit += 1
                        print(f"[retry] attempt={attempt} BACK detected; capturing once more for confirmation")
                        attempt += 1
                        continue

                    # Both independent captures are BACK.  Save the one with
                    # the better ICP score, rather than always saving the
                    # confirmation capture.
                    use_current = first_back_key is None or current_back_key < first_back_key
                    selected_dir = pending_dir if use_current else back_dir
                    selected_pose_json = pose_json if use_current else first_back_pose_json
                    selected_metadata = metadata if use_current else first_back_metadata
                    selected_pose_count = len(poses) if use_current else first_back_pose_count
                    if selected_metadata is None:
                        raise RuntimeError("BACK confirmation metadata is missing")
                    run_index, run_dir = create_run_dir(workpiece_capture_root)
                    for artifact in selected_dir.iterdir():
                        os.replace(artifact, run_dir / artifact.name)
                    selected_pose_json["mask"] = str(run_dir / "mask.png")
                    selected_pose_json["back_face_confirmation"] = {
                        "confirmed": True,
                        "captures_checked": 2,
                        "selection": "second_capture" if use_current else "first_capture",
                        "first_capture_icp_key": list(first_back_key) if first_back_key is not None else None,
                        "second_capture_icp_key": list(current_back_key),
                        "key_definition": "sum(depth_score_mm), sum(rmse_mm), negative sum(fitness); lower is better",
                    }
                    pose_path = run_dir / "mask_pose.json"
                    pose_path.write_text(json.dumps(selected_pose_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    selected_metadata["capture_index"] = run_index
                    selected_metadata["back_face_confirmation"] = selected_pose_json["back_face_confirmation"]
                    (run_dir / "metadata.json").write_text(json.dumps(selected_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(f"[done] capture={run_index} dir={run_dir} workpieces={selected_pose_count} pose_json={pose_path} back_confirmed=2 icp_selection={selected_pose_json['back_face_confirmation']['selection']}")
                    return 0
                if first_back_pose_json is not None:
                    # BACK confirmation is strictly consecutive.  Once the
                    # confirmation capture says FRONT or OTHER, discard the
                    # earlier BACK and require two new consecutive BACK
                    # observations before returning BACK.
                    first_back_key = None
                    first_back_pose_json = None
                    first_back_metadata = None
                    first_back_pose_count = 0
                    shutil.rmtree(back_dir)
                    back_dir.mkdir()
                low_poses = [pose for pose in poses if pose["pose_confidence"]["level"] == "LOW"]
                if low_poses:
                    reasons = ",".join("+".join(pose["pose_confidence"]["low_reasons"]) for pose in low_poses)
                    key = low_attempt_key(poses)
                    if best_key is None or key < best_key:
                        shutil.copytree(pending_dir, best_dir, dirs_exist_ok=True)
                        best_key, best_pose_json, best_metadata, best_pose_count = key, pose_json, metadata, len(poses)
                    print(f"[retry] attempt={attempt}/{args.low_confidence_retries + 1} low-confidence pose discarded: {reasons}")
                    attempt += 1
                    continue

                run_index, run_dir = create_run_dir(workpiece_capture_root)
                for artifact in pending_dir.iterdir():
                    os.replace(artifact, run_dir / artifact.name)
                capture_metadata["capture_index"] = run_index
                pose_json["mask"] = str(run_dir / "mask.png")
                pose_path = run_dir / "mask_pose.json"
                pose_path.write_text(json.dumps(pose_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                metadata["capture_index"] = run_index
                (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"[done] capture={run_index} dir={run_dir} workpieces={len(poses)} pose_json={pose_path}")
                return 0

            attempt += 1

        if best_pose_json is None or best_metadata is None:
            raise RuntimeError(
                f"no valid pose after {args.low_confidence_retries + 1} captures; "
                "no result was written to the numeric output directory"
            )
        run_index, run_dir = create_run_dir(workpiece_capture_root)
        for artifact in best_dir.iterdir():
            os.replace(artifact, run_dir / artifact.name)
        best_pose_json["mask"] = str(run_dir / "mask.png")
        pose_path = run_dir / "mask_pose.json"
        pose_path.write_text(json.dumps(best_pose_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        best_metadata["capture_index"] = run_index
        best_metadata["confidence_fallback_used"] = True
        best_metadata["confidence_fallback_selection_key"] = list(best_key) if best_key is not None else None
        (run_dir / "metadata.json").write_text(json.dumps(best_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[done] capture={run_index} dir={run_dir} workpieces={best_pose_count} pose_json={pose_path} confidence_fallback=LOW")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
