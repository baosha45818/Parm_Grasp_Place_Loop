#!/usr/bin/env python3
"""Offline RGB evidence for workpiece-2 steep-surface candidate ranking.

This program never opens a camera or connects to any robot service.  It reads
one saved D405 inference directory plus the corresponding filtered GraspNet
candidates, recomputes the physical >=65-degree surface samples in a temporary
directory, and writes a before/after RGB overlay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workpiece2_cloud_thickening import (  # noqa: E402
    build_thickened_inference_cloud,
    rank_candidate_indices_by_steep_surface,
)


def _project(points_base: np.ndarray, T_camera_base: np.ndarray, K: np.ndarray) -> np.ndarray:
    # Row-vector form of the column-vector transform T_camera_base.
    points_camera = points_base @ T_camera_base[:3, :3].T + T_camera_base[:3, 3]
    positive = points_camera[:, 2] > 1e-6
    pixels = np.full((len(points_base), 2), -1, dtype=np.int32)
    values = points_camera[positive]
    pixels[positive, 0] = np.rint(K[0, 0] * values[:, 0] / values[:, 2] + K[0, 2]).astype(np.int32)
    pixels[positive, 1] = np.rint(K[1, 1] * values[:, 1] / values[:, 2] + K[1, 2]).astype(np.int32)
    return pixels


def _draw_grasp(draw: ImageDraw.ImageDraw, grasp: np.ndarray, T_camera_base: np.ndarray, K: np.ndarray, color: str, label: str) -> None:
    rotation = np.asarray(grasp[4:13], dtype=np.float64).reshape(3, 3)
    center = np.asarray(grasp[13:16], dtype=np.float64)
    half_width = max(.005, min(.050, float(grasp[1]) * .5))
    points = np.vstack((
        center,
        center + .040 * rotation[:, 0],
        center + half_width * rotation[:, 1] - .014 * rotation[:, 2],
        center + half_width * rotation[:, 1] + .014 * rotation[:, 2],
        center - half_width * rotation[:, 1] - .014 * rotation[:, 2],
        center - half_width * rotation[:, 1] + .014 * rotation[:, 2],
    ))
    pixels = _project(points, T_camera_base, K)
    if np.any(pixels[:2] < 0):
        return
    draw.ellipse((*((pixels[0] - 7).tolist()), *((pixels[0] + 7).tolist())), outline=color, width=3)
    draw.line(tuple(pixels[1].tolist() + pixels[0].tolist()), fill=color, width=3)
    for first, second in ((2, 3), (4, 5)):
        if np.all(pixels[[first, second]] >= 0):
            draw.line(tuple(pixels[first].tolist() + pixels[second].tolist()), fill=color, width=5)
    draw.text(tuple((pixels[0] + 8).tolist()), label, fill=color, stroke_width=1, stroke_fill="black")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d405-assets", type=Path, required=True)
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--edge-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angle-deg", type=float, default=65.0)
    parser.add_argument("--contact-distance-m", type=float, default=.015)
    parser.add_argument("--min-score-ratio", type=float, default=.60)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    cli = _args()
    assets = cli.d405_assets.resolve()
    grasps = np.asarray(np.load(cli.grasps), dtype=np.float64)
    edge = json.loads(cli.edge_filter.read_text(encoding="utf-8"))
    indices = np.asarray([row["source_grasp_index"] for row in edge["candidates"] if row.get("accepted")], dtype=np.int64)
    meta = json.loads((assets / "meta.json").read_text(encoding="utf-8"))
    T_base_camera = np.asarray(meta["T_base_camera"], dtype=np.float64)
    T_camera_base = np.linalg.inv(T_base_camera)
    K = np.asarray(np.load(assets / "color_intrinsics.npy"), dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="w2-steep-priority-") as temp:
        temp_root = Path(temp)
        shutil.copy2(assets / "point_cloud.npy", temp_root / "point_cloud.npy")
        shutil.copy2(assets / "meta.json", temp_root / "meta.json")
        ranking_args = SimpleNamespace(
            enable_workpiece2_cloud_thickening=True, workpiece_id="2", disable_femto=True,
            workpiece2_cloud_thickening_offsets_mm=[1.5, 3.0],
            workpiece2_cloud_thickening_max_source_points=10000,
            workpiece2_cloud_thickening_plane_distance_m=.003,
            workpiece2_cloud_thickening_min_height_m=.004,
            workpiece2_cloud_thickening_max_height_m=.120,
            workpiece2_cloud_thickening_normal_radius_m=.010,
            workpiece2_cloud_thickening_normal_max_nn=30,
            workpiece2_cloud_thickening_min_plane_angle_deg=float(cli.angle_deg),
            workpiece2_steep_surface_priority=True,
            workpiece2_steep_surface_contact_distance_m=float(cli.contact_distance_m),
            workpiece2_steep_surface_min_score_ratio=float(cli.min_score_ratio),
        )
        _, cloud_record = build_thickened_inference_cloud(temp_root / "point_cloud.npy", ranking_args)
        ranked, record = rank_candidate_indices_by_steep_surface(ranking_args, temp_root, grasps, indices)
        steep_base = np.asarray(np.load(temp_root / "workpiece2_steep_surface_points.npy"), dtype=np.float64)
        steep_base = steep_base @ T_base_camera[:3, :3].T + T_base_camera[:3, 3]

    raw = indices[np.argsort(grasps[indices, 0])[::-1]]
    rgb = Image.open(assets / "color.png").convert("RGB")
    left, right = rgb.copy(), rgb.copy()
    for image in (left, right):
        draw = ImageDraw.Draw(image)
        points = _project(steep_base[::max(1, len(steep_base) // 900)], T_camera_base, K)
        for u, v in points:
            if 0 <= u < image.width and 0 <= v < image.height:
                draw.point((int(u), int(v)), fill="#00d9ff")
    for rank, index in enumerate(raw[:cli.top_k], start=1):
        _draw_grasp(ImageDraw.Draw(left), grasps[int(index)], T_camera_base, K, "#ff9d00", f"raw #{rank}: {grasps[int(index), 0]:.3f}")
    for rank, index in enumerate(ranked[:cli.top_k], start=1):
        _draw_grasp(ImageDraw.Draw(right), grasps[int(index)], T_camera_base, K, "#32ff66", f"steep #{rank}: {grasps[int(index), 0]:.3f}")
    canvas = Image.new("RGB", (rgb.width * 2, rgb.height + 32), "black")
    canvas.paste(left, (0, 32)); canvas.paste(right, (rgb.width, 32))
    title = ImageDraw.Draw(canvas)
    title.text((8, 8), "Raw GraspNet score order (orange); cyan = physical >=65deg surfaces", fill="white")
    title.text((rgb.width + 8, 8), "Steep-surface high-score priority (green)", fill="white")
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(cli.output)
    summary = {
        "output": str(cli.output), "steep_surface_points": cloud_record["selected_steep_surface_points"],
        "raw_indices": [int(value) for value in raw[:cli.top_k]],
        "ranked_indices": [int(value) for value in ranked[:cli.top_k]],
        "ranking": record,
    }
    cli.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("output", "steep_surface_points", "raw_indices", "ranked_indices")}, indent=2))


if __name__ == "__main__":
    main()
