#!/usr/bin/env python3
"""工件1：由 Module 4-1 诊断结果生成反面抓取位姿。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.grasp.toml_config import load_config_mapping

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
LOOP_ROOT = MODULE_ROOT.parents[3]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))

from tools.stl_model import load_stl_triangles_m

DEFAULT_CONFIG = MODULE_ROOT / "config/xiangxin_workpiece_1_grasp.toml"


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("zero-length direction")
    return vector / norm


def widest_head_in_stl(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centered = vertices - vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(values)[::-1]
    long_axis = normalize(vectors[:, order[0]])
    width_axis = normalize(vectors[:, order[1]])
    long_projection = centered @ long_axis
    width_projection = centered @ width_axis
    edges = np.linspace(long_projection.min(), long_projection.max(), 42)
    best = None
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (long_projection >= low) & (long_projection < high)
        if np.count_nonzero(selected) < 12:
            continue
        width = float(np.percentile(width_projection[selected], 95) - np.percentile(width_projection[selected], 5))
        center = vertices[selected].mean(axis=0)
        if best is None or width > best[0]:
            best = (width, center)
    if best is None:
        raise RuntimeError("STL does not contain enough vertices to locate the wide head")
    return best[1], long_axis, width_axis, best[0]


def make_grasp_frame(width_axis_camera: np.ndarray, table_normal_camera: np.ndarray) -> np.ndarray:
    x_axis = -normalize(table_normal_camera)
    y_axis = normalize(width_axis_camera - x_axis * np.dot(width_axis_camera, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] *= -1
    return rotation


def pose_record(rotation: np.ndarray, translation: np.ndarray) -> dict:
    return {
        "translation_m": np.asarray(translation).round(7).tolist(),
        "rotation_matrix": np.asarray(rotation).round(8).tolist(),
    }


def project(K: np.ndarray, point: np.ndarray) -> tuple[int, int] | None:
    if point[2] <= 1e-6:
        return None
    uv = K @ point
    return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))


def draw_grasp_overlay(rgb: np.ndarray, K: np.ndarray, records: list[dict]) -> np.ndarray:
    canvas = rgb.copy()
    for record in records:
        pose = record["grasp_pose"]
        origin = np.asarray(pose["translation_m"], dtype=float)
        rotation = np.asarray(pose["rotation_matrix"], dtype=float)
        p0 = project(K, origin)
        if p0 is None:
            continue
        cv2.circle(canvas, p0, 7, (0, 255, 255), 2, cv2.LINE_AA)
        for axis, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            p1 = project(K, origin + rotation[:, axis] * .04)
            if p1 is not None:
                cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=.18)
    return canvas


def reverse_grasp_record_from_icp(
    T_camera_object: np.ndarray,
    table_normal_camera: np.ndarray,
    K: np.ndarray,
    vertices: np.ndarray,
    config: dict,
    *,
    object_key: str = "sam2-0",
    quality: dict | None = None,
) -> dict:
    transform = np.asarray(T_camera_object, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_camera_object must be a finite 4x4 matrix")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    raw_head, long_axis, width_axis, _ = widest_head_in_stl(vertices)
    if config.get("grasp_point_stl_m") is not None:
        raw_head = np.asarray(config["grasp_point_stl_m"], dtype=float)
    if config.get("grasp_width_axis_stl") is not None:
        width_axis = normalize(config["grasp_width_axis_stl"])
    origin = vertices.mean(axis=0)
    projection = (vertices - origin) @ long_axis
    head_projection = float((raw_head - origin) @ long_axis)
    inward = long_axis if abs(head_projection - projection.min()) < abs(head_projection - projection.max()) else -long_axis
    outward_axis = normalize(config.get("grasp_outward_axis_stl", -inward))
    object_center = rotation @ origin + translation
    head_center = rotation @ raw_head + translation
    points_to_camera = normalize(-object_center)
    head_towards_camera = bool(np.dot(rotation @ outward_axis, points_to_camera) > 0)
    outward_offset = float(config["wide_head_outward_grasp_offset_m"])
    inward_offset = float(config["wide_head_inward_grasp_offset_m"])
    offset_axis = -outward_axis if head_towards_camera else outward_axis
    head = raw_head + offset_axis * (inward_offset if head_towards_camera else outward_offset)
    head_pixel, center_pixel = project(K, head_center), project(K, object_center)
    if head_pixel is None or center_pixel is None:
        raise RuntimeError("grasp landmark cannot be projected into the image")
    image_left = head_pixel[0] < center_pixel[0]
    lateral = float(config["wide_head_image_left_grasp_lateral_offset_m"] if image_left else config["wide_head_image_right_grasp_lateral_offset_m"])
    grasp_rotation = make_grasp_frame(rotation @ width_axis, table_normal_camera)
    center = rotation @ head + translation + grasp_rotation[:, 1] * lateral
    record = {
        "object_side": "reverse",
        "object_key": object_key,
        "wide_head_outward_grasp_offset_m": outward_offset,
        "wide_head_inward_grasp_offset_m": inward_offset,
        "grasp_lateral_offset_m": lateral,
        "grasp_lateral_offset_axis": "grasp_frame_+Y",
        "wide_head_towards_camera": head_towards_camera,
        "wide_head_image_direction": "left" if image_left else "right",
        "wide_head_grasp_offset_direction": (
            "inward" if head_towards_camera else "outward"
        ),
        "wide_head_grasp_offset_m": (
            inward_offset if head_towards_camera else outward_offset
        ),
        "grasp_frame_convention": "version7_graspnet:+X_approach_down,+Y_jaw_opening,+Z_side",
        "grasp_pose": pose_record(grasp_rotation, center),
        "pregrasp_pose": pose_record(grasp_rotation, center - grasp_rotation[:, 0] * .08),
    }
    if quality:
        record.update(quality)
    return record


def predict(capture_dir: Path, diagnostics_path: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    reverse = [item for item in diagnostics if item.get("classification") == "reverse"]
    if len(diagnostics) != 1 or len(reverse) != 1:
        raise RuntimeError("Module 5-1 仅接受 Module 4 判定出的唯一 BACK 工件")
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], dtype=float)
    config = load_config_mapping(config_path)
    stl = Path(config["stl"])
    if not stl.is_absolute():
        stl = LOOP_ROOT / stl
    item = reverse[0]
    reverse_quality = item["reverse"]
    record = reverse_grasp_record_from_icp(
        np.asarray(reverse_quality["T_camera_object"], dtype=float),
        np.asarray(item["table_normal_camera"], dtype=float), K,
        load_stl_triangles_m(stl).reshape(-1, 3), config,
        object_key=str(item.get("object_key", "sam2-0")),
        quality={
            "icp_fitness": reverse_quality.get("icp_fitness"),
            "icp_rmse_m": reverse_quality.get("icp_rmse_m"),
            "depth_match_fraction": (reverse_quality.get("render") or {}).get("match_fraction"),
        },
    )
    output = capture_dir / "sam2_masks/grasp_poses.json"
    output.write_text(json.dumps([record], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rgb = cv2.imread(str(capture_dir / "rgb_000.png"), cv2.IMREAD_COLOR)
    if rgb is None or not cv2.imwrite(str(capture_dir / "sam2_masks/grasp_overlay.png"), draw_grasp_overlay(rgb, K, [record])):
        raise RuntimeError("cannot write grasp overlay")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--orientation-diagnostics", type=Path, required=True)
    parser.add_argument("--grasp-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output = predict(args.capture_dir.resolve(), args.orientation_diagnostics.resolve(), args.grasp_config.resolve())
    print(f"WORKPIECE_1_GRASP_POSES={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
