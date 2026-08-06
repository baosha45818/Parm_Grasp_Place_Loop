#!/usr/bin/env python3
"""Inference-only D405 point-cloud thickening for production workpiece 2.

The physical capture cloud is deliberately never changed.  This module writes
an adjacent, normal-extruded cloud solely for the GraspNet service request.
It is enabled only by the Module 1-1 workpiece-2 branch and only extrudes
surfaces whose local plane is at least the configured angle from the table.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np


class Workpiece2CloudThickeningError(RuntimeError):
    """The inference-only cloud could not be constructed safely."""


def validate_args(args) -> None:
    """Validate the production-only option set before any hardware action."""
    if not args.enable_workpiece2_cloud_thickening:
        return
    if args.workpiece_id != "2":
        raise RuntimeError(
            "--enable-workpiece2-cloud-thickening requires --workpiece-id 2"
        )
    if not args.disable_femto:
        raise RuntimeError(
            "workpiece-2 cloud thickening is D405-only; pass --disable-femto"
        )

    offsets = np.asarray(args.workpiece2_cloud_thickening_offsets_mm, dtype=np.float64)
    if len(offsets) == 0 or not np.all(np.isfinite(offsets) & (offsets > 0.0)):
        raise RuntimeError(
            "--workpiece2-cloud-thickening-offsets-mm must contain finite positive values"
        )
    if int(args.workpiece2_cloud_thickening_max_source_points) < 100:
        raise RuntimeError("--workpiece2-cloud-thickening-max-source-points must be at least 100")
    for name in (
        "workpiece2_cloud_thickening_plane_distance_m",
        "workpiece2_cloud_thickening_min_height_m",
        "workpiece2_cloud_thickening_max_height_m",
        "workpiece2_cloud_thickening_normal_radius_m",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be finite and positive")
    if (
        float(args.workpiece2_cloud_thickening_min_height_m)
        >= float(args.workpiece2_cloud_thickening_max_height_m)
    ):
        raise RuntimeError("workpiece-2 cloud-thickening minimum height must be below maximum height")
    if int(args.workpiece2_cloud_thickening_normal_max_nn) < 3:
        raise RuntimeError("--workpiece2-cloud-thickening-normal-max-nn must be at least 3")
    angle = float(args.workpiece2_cloud_thickening_min_plane_angle_deg)
    if not np.isfinite(angle) or not 0.0 <= angle <= 90.0:
        raise RuntimeError("--workpiece2-cloud-thickening-min-plane-angle-deg must be within [0, 90]")


def _choose_table_side(signed_distance: np.ndarray, args) -> float:
    """Return the table-normal side occupied by the elevated workpiece."""
    low = float(args.workpiece2_cloud_thickening_min_height_m)
    high = float(args.workpiece2_cloud_thickening_max_height_m)
    positive = int(np.count_nonzero((signed_distance >= low) & (signed_distance <= high)))
    negative = int(np.count_nonzero((-signed_distance >= low) & (-signed_distance <= high)))
    if max(positive, negative) < 100:
        raise Workpiece2CloudThickeningError(
            "table fit found fewer than 100 geometrically elevated points; "
            "refusing to fabricate an inference cloud"
        )
    return 1.0 if positive >= negative else -1.0


def build_thickened_inference_cloud(raw_cloud_path: Path, args) -> tuple[Path, dict]:
    """Create a separate augmented cloud and a reproducibility record.

    The original raw D405 cloud is retained for object-reference, candidate
    support, collision, and execution logic.  Only the service request is
    redirected to the returned file.
    """
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - deployment dependency check
        raise Workpiece2CloudThickeningError(
            "workpiece-2 cloud thickening requires Open3D for normal estimation"
        ) from exc

    raw_cloud_path = Path(raw_cloud_path)
    points = np.asarray(np.load(raw_cloud_path), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise Workpiece2CloudThickeningError(f"invalid point-cloud shape: {raw_cloud_path}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 1000:
        raise Workpiece2CloudThickeningError(f"invalid or too-small raw cloud: {raw_cloud_path}")

    started = time.monotonic()
    o3d.utility.random.seed(0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, inliers = cloud.segment_plane(
        distance_threshold=float(args.workpiece2_cloud_thickening_plane_distance_m),
        ransac_n=3,
        num_iterations=500,
    )
    table_normal = np.asarray(plane[:3], dtype=np.float64)
    normal_norm = float(np.linalg.norm(table_normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-9 or len(inliers) < 500:
        raise Workpiece2CloudThickeningError("RANSAC could not establish a dominant table plane")
    table_normal /= normal_norm
    signed = points @ table_normal + float(plane[3]) / normal_norm
    side = _choose_table_side(signed, args)
    height = side * signed
    selected_indices = np.flatnonzero(
        (height >= float(args.workpiece2_cloud_thickening_min_height_m))
        & (height <= float(args.workpiece2_cloud_thickening_max_height_m))
    )
    if len(selected_indices) < 100:
        raise Workpiece2CloudThickeningError("too few above-table points for workpiece-2 thickening")
    source_before_cap = int(len(selected_indices))
    cap = int(args.workpiece2_cloud_thickening_max_source_points)
    if len(selected_indices) > cap:
        selected_indices = selected_indices[
            np.linspace(0, len(selected_indices) - 1, cap, dtype=np.int64)
        ]
    selected = points[selected_indices]

    normal_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected))
    normal_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(args.workpiece2_cloud_thickening_normal_radius_m),
            max_nn=int(args.workpiece2_cloud_thickening_normal_max_nn),
        )
    )
    normals = np.asarray(normal_cloud.normals, dtype=np.float64)
    normal_lengths = np.linalg.norm(normals, axis=1)
    usable = np.isfinite(normals).all(axis=1) & (normal_lengths > 1e-6)
    selected, normals = selected[usable], normals[usable]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    if len(selected) < 100:
        raise Workpiece2CloudThickeningError("normal estimation left too few usable elevated points")

    # abs removes the arbitrary Open3D normal sign: horizontal planes are 0°,
    # vertical planes are 90°.  Therefore only the requested steep surfaces
    # (>=65° from the table) contribute virtual points.
    cosine_to_table = np.clip(np.abs(normals @ table_normal), 0.0, 1.0)
    plane_angle_deg = np.degrees(np.arccos(cosine_to_table))
    steep = plane_angle_deg >= float(args.workpiece2_cloud_thickening_min_plane_angle_deg)
    selected, normals, plane_angle_deg = selected[steep], normals[steep], plane_angle_deg[steep]
    if len(selected) < 100:
        raise Workpiece2CloudThickeningError(
            "too few local surfaces meet --workpiece2-cloud-thickening-min-plane-angle-deg; "
            "refusing to fabricate an inference cloud"
        )

    offsets_m = np.unique(
        np.asarray(args.workpiece2_cloud_thickening_offsets_mm, dtype=np.float64)
    ) / 1000.0
    virtual_layers = [
        selected + sign * offset * normals
        for offset in offsets_m
        for sign in (-1.0, 1.0)
    ]
    augmented = np.concatenate((points, *virtual_layers), axis=0).astype(np.float32)
    output_path = raw_cloud_path.with_name("point_cloud_workpiece2_thickened.npy")
    np.save(output_path, augmented)
    record = {
        "mode": "workpiece2_inference_only_normal_extrusion",
        "workpiece_id": "2",
        "raw_point_cloud": str(raw_cloud_path),
        "inference_point_cloud": str(output_path),
        "raw_points": int(len(points)),
        "table_inliers": int(len(inliers)),
        "above_table_points_before_cap": source_before_cap,
        "selected_steep_surface_points": int(len(selected)),
        "virtual_points": int(len(augmented) - len(points)),
        "inference_points": int(len(augmented)),
        "offsets_mm_each_side": [float(value) for value in offsets_m * 1000.0],
        "table_plane_point_cloud_frame": [float(value) for value in plane],
        "selected_table_normal_sign": int(side),
        "height_band_m": [
            float(args.workpiece2_cloud_thickening_min_height_m),
            float(args.workpiece2_cloud_thickening_max_height_m),
        ],
        "minimum_plane_to_table_angle_deg": float(args.workpiece2_cloud_thickening_min_plane_angle_deg),
        "selected_plane_to_table_angle_deg": {
            "min": float(np.min(plane_angle_deg)),
            "median": float(np.median(plane_angle_deg)),
            "max": float(np.max(plane_angle_deg)),
        },
        "elapsed_s": time.monotonic() - started,
        "safety_note": (
            "raw point_cloud.npy remains unchanged; only GraspNet inference receives "
            "the virtual layers, while later candidate filters retain physical D405 points"
        ),
    }
    output_path.with_name("point_cloud_workpiece2_thickening.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return output_path, record
