#!/usr/bin/env python3
"""Estimate workpiece-2 FRONT/BACK and pose from one existing RGB-D capture.

SAM2 supplies one or more mask-local observed point clouds.  Each cloud is
globally initialized with FPFH/RANSAC, expanded into four semantic half-turn
hypotheses, and refined by unrestricted point-to-plane ICP.  The winning ICP
transform is the single source for both the face label and the published pose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
PROJECT_ROOT = MODULE_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.module_4_head_judge.module_4_1_buffer_state_estimation import (  # noqa: E402
    full6d_pose_solver as pose_solver,
)
from tools.inspection_support.stl_model import load_cad_pointcloud_template  # noqa: E402
from .geometry import (  # noqa: E402
    camera_matrix,
    depth_to_xyz,
    load_transform,
)


ASSET_ROOT = MODULE_ROOT / "assets/workpiece_2"
STL_PATH = PROJECT_ROOT / "tools/workpiece_2/workpiece_2.stl"
CAD_TEMPLATE_PATH = PROJECT_ROOT / "tools/workpiece_2/cad_pointcloud_template.npz"
SEMANTIC_PATH = ASSET_ROOT / "workpiece2.semantic.json"
CONFIG_PATH = MODULE_ROOT / "configs/workpiece_2.json"
BODY_HEAD_PATH = (
    Path(os.environ.get(
        "PARM_BODY_HEAD_CALIBRATION",
        PROJECT_ROOT / "tools/calibration/head_d435/xiangxin_reverse_observation.json",
    )).expanduser().resolve()
)


def _camera_matrix(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    return camera_matrix(metadata)


def _depth_xyz(depth: np.ndarray, matrix: np.ndarray, scale_m: float) -> np.ndarray:
    return depth_to_xyz(depth, matrix, scale_m)


def _load_base_camera() -> np.ndarray:
    return load_transform(BODY_HEAD_PATH)


def _solver_args(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        max_observed_points=int(config.get("max_observed_points", 5000)),
        min_icp_points=int(config.get("minimum_icp_points", 120)),
        icp_global_voxel_mm=float(config["reverse_full_6d_global_voxel_m"]) * 1000.0,
        icp_global_seeds=int(config["reverse_full_6d_global_seeds"]),
        icp_ransac_iters=int(config["reverse_full_6d_ransac_iterations"]),
        icp_iters=int(config["reverse_full_6d_icp_iterations"]),
        icp_max_corr_mm=float(config["reverse_full_6d_icp_max_correspondence_m"]) * 1000.0,
        minimum_z_up_dot=float(config.get("minimum_face_alignment", 0.90)),
        max_table_penetration_mm=float(config["reverse_full_6d_max_table_penetration_m"]) * 1000.0,
        minimum_face_margin_mm=float(config.get("minimum_face_margin_mm", 0.35)),
        near_optimal_score_window_mm=float(config.get("near_optimal_score_window_mm", 0.5)),
    )


def _deduplicate_masks(masks: np.ndarray, minimum_iou: float = 0.75) -> np.ndarray:
    """Collapse repeated SAM2 prompts that segmented the same physical object."""
    kept: list[np.ndarray] = []
    for raw_mask in masks:
        mask = raw_mask.astype(bool)
        duplicate = False
        for existing in kept:
            union = int(np.count_nonzero(mask | existing))
            overlap = int(np.count_nonzero(mask & existing))
            if union and overlap / union >= minimum_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
    return np.asarray(kept, dtype=np.uint8)


def _promote_single_face(record: dict, config: dict) -> None:
    """Accept one strong face when the opposite face has no physical candidate."""
    hypotheses = record["face_hypotheses"]
    if record["orientation"] != "OTHER" or len(hypotheses) != 1:
        return
    side, candidate = next(iter(hypotheses.items()))
    if not (
        float(candidate["fitness"]) >= float(config["single_face_minimum_fitness"])
        and float(candidate["rmse_mm"]) <= float(config["single_face_maximum_rmse_mm"])
        and float(candidate["depth_score_mm"]) <= float(config["single_face_maximum_depth_score_mm"])
    ):
        return
    record["orientation"] = side
    record["object_side"] = (
        "positive_face_full_6dof" if side == "FRONT" else "reverse_face_full_6dof"
    )
    confidence = record["pose_confidence"]
    confidence["low_reasons"] = [
        reason for reason in confidence["low_reasons"]
        if reason not in {"front_back_orientation_ambiguous", "insufficient_front_back_score_margin"}
    ]
    confidence["checks"]["orientation"] = side
    confidence["single_face_absolute_gate"] = {
        "accepted": True,
        "side": side,
        "reason": "opposite face produced no candidate satisfying physical geometry constraints",
    }
    confidence["recommended_for_grasp"] = not confidence["low_reasons"]
    confidence["level"] = "LOW" if confidence["low_reasons"] else "MEDIUM"


def _classification(record: dict) -> tuple[str, bool, bool]:
    raw_orientation = str(record["orientation"])
    low_reasons = list(record["pose_confidence"]["low_reasons"])
    pose_only_failures = [
        reason for reason in low_reasons
        if reason not in {"front_back_orientation_ambiguous", "insufficient_front_back_score_margin"}
    ]
    pose_valid = not pose_only_failures
    orientation_valid = raw_orientation in {"FRONT", "BACK"} and not low_reasons
    if not orientation_valid:
        return "unknown", pose_valid, False
    return ("positive" if raw_orientation == "FRONT" else "reverse"), pose_valid, True


def _legacy_face_hypothesis(candidate: dict) -> dict:
    """Expose the familiar Module-4 diagnostic names without inventing depth matches."""
    return {
        "icp_fitness": float(candidate["fitness"]),
        "icp_rmse_m": float(candidate["rmse_mm"]) * 1e-3,
        "depth_score_m": float(candidate["depth_score_mm"]) * 1e-3,
        "face_up_dot_table_normal": float(candidate["face_up_dot_table_normal"]),
        "initialization": str(candidate["initialization"]),
    }


def _diagnostic(
    index: int, record: dict, table_normal: np.ndarray,
    calibrated_head_pose_deg: dict | None,
) -> dict:
    classification, pose_valid, orientation_valid = _classification(record)
    hypotheses = record["face_hypotheses"]
    result = {
        "object_key": f"sam2-{index}",
        "classification": classification,
        "raw_orientation": record["orientation"],
        "score_margin_m": (
            None if record["orientation_score_margin_mm"] is None
            else float(record["orientation_score_margin_mm"]) * 1e-3
        ),
        "table_normal_camera": np.asarray(table_normal, dtype=float).round(9).tolist(),
        "pose": {
            "valid": pose_valid,
            "orientation_valid": orientation_valid,
            "method": "fpfh_ransac_multi_start_full_6d_point_to_plane_icp",
            "T_head_camera_workpiece": record["T_head_camera_workpiece"],
            "T_body_base_workpiece": record["T_body_base_workpiece_semantic"],
            "body_base_xyz_rpy": record["robot_base_xyz_rpy"],
            "calibration": str(BODY_HEAD_PATH),
            "calibrated_head_pose_deg": calibrated_head_pose_deg,
        },
        "pose_confidence": record["pose_confidence"],
        "candidate_confidence": record["candidate_confidence"],
        "icp": record["icp"],
        "face_hypotheses": hypotheses,
    }
    if "FRONT" in hypotheses:
        result["positive"] = _legacy_face_hypothesis(hypotheses["FRONT"])
    if "BACK" in hypotheses:
        result["reverse"] = _legacy_face_hypothesis(hypotheses["BACK"])
    return result


def _write_overlay(rgb: np.ndarray, masks: np.ndarray, diagnostics: list[dict], path: Path) -> None:
    image = rgb.copy()
    colors = {"positive": (0, 180, 0), "reverse": (0, 0, 220), "unknown": (0, 180, 255)}
    for mask, diagnostic in zip(masks, diagnostics):
        selected = mask.astype(bool)
        color = colors[diagnostic["classification"]]
        tint = np.empty_like(image)
        tint[:] = color
        image[selected] = (0.55 * image[selected] + 0.45 * tint[selected]).astype(np.uint8)
        rows, columns = np.nonzero(selected)
        if len(rows):
            pose = diagnostic.get("icp") or {}
            label = (
                f"{diagnostic['classification']} "
                f"f={float(pose.get('fitness', math.nan)):.2f} "
                f"d={float(pose.get('depth_score_mm', math.nan)):.1f}mm"
            )
            cv2.putText(image, label, (int(columns.mean()), int(rows.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法写入: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    capture_dir = args.capture_dir.expanduser().resolve()
    masks_dir = args.masks_dir.expanduser().resolve()
    required = [
        capture_dir / "metadata.json", capture_dir / "rgb_000.png",
        capture_dir / "depth_aligned_000.png", masks_dir / "masks.npz",
        masks_dir / "meta.json", STL_PATH, CAD_TEMPLATE_PATH, SEMANTIC_PATH,
        args.config, BODY_HEAD_PATH,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少工件2状态估计输入: " + ", ".join(map(str, missing)))

    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    masks_metadata = json.loads((masks_dir / "meta.json").read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rgb = cv2.imread(str(capture_dir / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture_dir / "depth_aligned_000.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
        raise RuntimeError("工件2 RGB-D 无法读取或尺寸不一致")
    masks = _deduplicate_masks(np.load(masks_dir / "masks.npz")["masks"])
    matrix, distortion = _camera_matrix(metadata)
    xyz = _depth_xyz(depth, matrix, float(metadata["depth_scale_m"]))
    table_plane = np.asarray(masks_metadata["table_plane_camera"], dtype=np.float64)
    base_camera = _load_base_camera()
    body_head_payload = json.loads(BODY_HEAD_PATH.read_text(encoding="utf-8"))

    triangles = pose_solver.load_stl_triangles(STL_PATH)
    vertices = triangles.reshape(-1, 3)
    stl_semantic = pose_solver.load_semantic_calibration(SEMANTIC_PATH)
    positive_face_axis = pose_solver.load_positive_face_axis(SEMANTIC_PATH)
    landmarks = pose_solver.semantic_landmarks(vertices, stl_semantic)
    cad, cad_render = load_cad_pointcloud_template(CAD_TEMPLATE_PATH)
    solver_args = _solver_args(config)

    diagnostics: list[dict] = []
    accepted_masks: list[np.ndarray] = []
    for index, raw_mask in enumerate(masks):
        component = raw_mask.astype(bool)
        try:
            record = pose_solver.icp_pose_for_mask(
                component, xyz, depth, float(metadata["depth_scale_m"]),
                matrix, distortion, base_camera, table_plane, cad, cad_render,
                stl_semantic, positive_face_axis, landmarks, solver_args,
                {"area_px": int(np.count_nonzero(component))},
            )
            _promote_single_face(record, config)
            diagnostic = _diagnostic(
                index, record, table_plane[:3],
                body_head_payload.get("head_pose_deg"),
            )
        except RuntimeError as exc:
            diagnostic = {
                "object_key": f"sam2-{index}",
                "classification": "unknown",
                "reason": str(exc),
                "pose": {"valid": False, "orientation_valid": False},
            }
        diagnostics.append(diagnostic)
        accepted_masks.append(component)

    diagnostics_path = masks_dir / "orientation_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_overlay(rgb, np.asarray(accepted_masks), diagnostics, masks_dir / "orientation_overlay.png")
    print(json.dumps({
        "workpiece_id": 2,
        "objects": len(diagnostics),
        "diagnostics": str(diagnostics_path),
        "orientation_overlay": str(masks_dir / "orientation_overlay.png"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
