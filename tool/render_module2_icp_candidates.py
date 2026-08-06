#!/usr/bin/env python3
"""将历史 Module 2 ICP 的低 RMSE 候选投影回原始 RGB 图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_2_left_inspect_apart.standalone_inspect_global import (  # noqa: E402
    CONFIG_PATH,
    WORKPIECE_2_R_OBJECT_STL,
    WORKPIECE_2_STL_PATH,
    _projected_mask_consistency,
    load_object_triangles,
)
from tool.inspection_support.registration import make_pca_initial_guesses  # noqa: E402
from tool.inspection_support.stl_model import sample_surface_points  # noqa: E402


def candidates(
    cad: np.ndarray, observed: np.ndarray, cfg: dict, mask: np.ndarray, K: np.ndarray,
) -> list[tuple[dict, np.ndarray]]:
    cad_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
    observed_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=float(cfg["observed_voxel_m"]) * 4.0, max_nn=40
    )
    cad_cloud.estimate_normals(search)
    observed_cloud.estimate_normals(search)
    output: list[tuple[dict, np.ndarray]] = []
    for method, estimator in (
        ("point_to_plane", o3d.pipelines.registration.TransformationEstimationPointToPlane()),
        ("point_to_point", o3d.pipelines.registration.TransformationEstimationPointToPoint()),
    ):
        for initialization, initial in make_pca_initial_guesses(cad, observed):
            result = o3d.pipelines.registration.registration_icp(
                cad_cloud,
                observed_cloud,
                float(cfg["icp_max_corr_m"]),
                initial,
                estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=int(cfg["icp_max_iters"])
                ),
            )
            if result.fitness <= 0.0 or not np.isfinite(result.inlier_rmse):
                continue
            metric = {
                "method": method,
                "initialization": initialization,
                "fitness": float(result.fitness),
                "rmse_m": float(result.inlier_rmse),
            }
            metric.update(_projected_mask_consistency(
                cad, np.asarray(result.transformation), mask, K,
                int(cfg["icp_projection_dilation_px"]),
            ))
            metric["failed_gates"] = [
                name
                for name, passed in (
                    ("fitness", metric["fitness"] >= float(cfg["quality_min_fitness"])),
                    ("rmse", metric["rmse_m"] <= float(cfg["quality_max_rmse_m"])),
                    ("mask_near", metric["mask_near_cad_ratio"] >= float(cfg["icp_min_mask_near_cad_ratio"])),
                    ("iou", metric["projected_iou"] >= float(cfg["icp_min_projected_iou"])),
                    ("median_px", metric["mask_to_cad_median_px"] <= float(cfg["icp_max_mask_to_cad_median_px"])),
                )
                if not passed
            ]
            # Normalize every unmet gate by its configured threshold, so this
            # offline ranking means "closest to the actual online acceptance".
            metric["quality_gap"] = float(
                max(0.0, (float(cfg["quality_min_fitness"]) - metric["fitness"]) / float(cfg["quality_min_fitness"]))
                + max(0.0, (metric["rmse_m"] - float(cfg["quality_max_rmse_m"])) / float(cfg["quality_max_rmse_m"]))
                + max(0.0, (float(cfg["icp_min_mask_near_cad_ratio"]) - metric["mask_near_cad_ratio"]) / float(cfg["icp_min_mask_near_cad_ratio"]))
                + max(0.0, (float(cfg["icp_min_projected_iou"]) - metric["projected_iou"]) / float(cfg["icp_min_projected_iou"]))
                + max(0.0, (metric["mask_to_cad_median_px"] - float(cfg["icp_max_mask_to_cad_median_px"])) / float(cfg["icp_max_mask_to_cad_median_px"]))
            )
            output.append((metric, np.asarray(result.transformation)))
    return output


def draw_candidate(
    image: np.ndarray,
    mask: np.ndarray,
    cad: np.ndarray,
    transform: np.ndarray,
    K: np.ndarray,
    metric: dict,
) -> np.ndarray:
    panel = image.copy()
    red = np.zeros_like(panel)
    red[:, :, 2] = 255
    selected = mask > 0
    panel[selected] = (panel[selected] * 0.55 + red[selected] * 0.45).astype(np.uint8)
    points = (transform[:3, :3] @ cad.T).T + transform[:3, 3]
    points = points[points[:, 2] > 1e-5]
    pixels_h = (K @ points.T).T
    pixels = np.rint(pixels_h[:, :2] / pixels_h[:, 2:3]).astype(np.int32)
    height, width = panel.shape[:2]
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    for u, v in pixels[inside][::max(1, int(np.count_nonzero(inside) // 5000))]:
        cv2.circle(panel, (int(u), int(v)), 1, (0, 255, 255), -1)
    label = (
        f"{metric['method']} / {metric['initialization']} | "
        f"fitness={metric['fitness']:.4f}, RMSE={metric['rmse_m'] * 1000.0:.2f}mm"
    )
    gate_text = (
        f"failed={','.join(metric['failed_gates']) or 'none'} | "
        f"mask-near={metric['mask_near_cad_ratio']:.3f}, IoU={metric['projected_iou']:.3f}, "
        f"median={metric['mask_to_cad_median_px']:.1f}px, gap={metric['quality_gap']:.3f}"
    )
    cv2.rectangle(panel, (8, 8), (min(width - 8, 1250), 82), (0, 0, 0), -1)
    cv2.putText(panel, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(panel, gate_text, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("pc1", "pc2"), required=True)
    parser.add_argument("--workpiece", choices=("1", "2"), default="1")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--rank", choices=("rmse", "quality"), default="rmse")
    parser.add_argument("--camera-intrinsics", nargs=9, type=float, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(str(source / f"{args.view}_rgb.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(source / f"{args.view}_mask.png"), cv2.IMREAD_GRAYSCALE)
    points = np.load(source / f"{args.view}_points_camera.npy")
    if rgb is None or mask is None:
        raise RuntimeError(f"{args.view} 缺少 RGB 或 mask")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.workpiece == "2":
        cfg.update({"stl_path": str(WORKPIECE_2_STL_PATH), "R_object_stl": WORKPIECE_2_R_OBJECT_STL})
    cad = sample_surface_points(
        load_object_triangles(cfg), int(cfg["cad_sample_points"]), int(cfg["seed"])
    )
    K = np.asarray(args.camera_intrinsics, dtype=float).reshape(3, 3)
    ranked = candidates(cad, points, cfg, mask, K)
    if args.rank == "rmse":
        ranked.sort(key=lambda item: item[0]["rmse_m"])
        filename = f"{args.view}_lowest_rmse_candidates.png"
    else:
        ranked.sort(key=lambda item: (item[0]["quality_gap"], item[0]["rmse_m"]))
        filename = f"{args.view}_closest_quality_candidates.png"
    chosen = ranked[:args.top_n]
    panels = [draw_candidate(rgb, mask, cad, transform, K, metric) for metric, transform in chosen]
    output_path = output / filename
    if not cv2.imwrite(str(output_path), np.vstack(panels)):
        raise RuntimeError(f"无法保存: {output_path}")
    (output / f"{args.view}_{args.rank}_ranked_candidates.json").write_text(
        json.dumps([metric for metric, _ in chosen], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
