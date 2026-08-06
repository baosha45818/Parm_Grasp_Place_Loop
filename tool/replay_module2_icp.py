#!/usr/bin/env python3
"""离线复算已保存 Module 2 的 ICP，并输出新门限下的候选指标和可视化。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_2_left_inspect_apart.standalone_inspect_global import (  # noqa: E402
    CONFIG_PATH,
    WORKPIECE_2_R_OBJECT_STL,
    WORKPIECE_2_STL_PATH,
    _icp_pose,
    _projected_mask_consistency,
    _plot_pose_estimation,
    _save_pose_overlay,
    load_object_triangles,
)
from tool.inspection_support.stl_model import sample_surface_points  # noqa: E402
from tool.inspection_support.registration import make_pca_initial_guesses  # noqa: E402


def best_rejected_transform(
    cad: np.ndarray,
    observed: np.ndarray,
    cfg: dict,
    selection_mask: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Repeat ICP only to retain the best rejected transform for visualization."""
    cad_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
    observed_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=float(cfg["observed_voxel_m"]) * 4.0, max_nn=40
    )
    cad_cloud.estimate_normals(search)
    observed_cloud.estimate_normals(search)
    candidates: list[tuple[dict, np.ndarray]] = []
    for method_name, estimator in (
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
            item = {
                "method": method_name,
                "initialization": initialization,
                "fitness": float(result.fitness),
                "rmse_m": float(result.inlier_rmse),
            }
            item.update(_projected_mask_consistency(
                cad,
                np.asarray(result.transformation),
                selection_mask,
                camera_intrinsics,
                int(cfg["icp_projection_dilation_px"]),
            ))
            candidates.append((item, np.asarray(result.transformation)))
    return max(
        candidates,
        key=lambda pair: (
            pair[0]["mask_near_cad_ratio"],
            pair[0]["projected_iou"],
            pair[0]["fitness"],
            -pair[0]["rmse_m"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workpiece", choices=("1", "2"), default="1")
    parser.add_argument("--view", choices=("pc1", "pc2"), default="pc1")
    parser.add_argument("--camera-intrinsics", nargs=9, type=float, default=None)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    result_path = source / "result.json"
    view = args.view
    points_path = source / f"{view}_points_camera.npy"
    mask_path = source / f"{view}_mask.png"
    rgb_path = source / f"{view}_rgb.png"
    missing = [path for path in (points_path, mask_path, rgb_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"离线复算缺少输入: {missing}")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True)

    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    registration = result.get("registration") or {}
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.workpiece == "2":
        cfg.update({
            "stl_path": str(WORKPIECE_2_STL_PATH),
            "R_object_stl": WORKPIECE_2_R_OBJECT_STL,
            "object_origin_stl_m": [0.0, 0.0, 0.0],
        })
    points = np.load(points_path)
    selection_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if selection_mask is None:
        raise RuntimeError(f"无法读取 mask: {mask_path}")
    if registration.get("camera_intrinsics") is not None:
        camera_intrinsics = np.asarray(registration["camera_intrinsics"], dtype=float)
    elif args.camera_intrinsics is not None:
        camera_intrinsics = np.asarray(args.camera_intrinsics, dtype=float).reshape(3, 3)
    else:
        raise RuntimeError("失败运行没有保存内参；请传 --camera-intrinsics")

    shutil.copy2(rgb_path, output / f"{view}_rgb.png")
    shutil.copy2(mask_path, output / f"{view}_mask.png")
    triangles = load_object_triangles(cfg)
    cad = sample_surface_points(
        triangles, int(cfg["cad_sample_points"]), int(cfg["seed"])
    )

    try:
        T_camera_object, metrics = _icp_pose(
            cad, points, cfg, selection_mask, camera_intrinsics
        )
        status = "accepted"
    except Exception as exc:
        metrics = getattr(exc, "metrics", {"error": f"{type(exc).__name__}: {exc}"})
        status = "rejected"
        T_camera_object = None

    T_body_camera = None
    if result.get("T_camera_object") is not None and result.get("T_body_base_object") is not None:
        old_camera_object = np.asarray(result["T_camera_object"], dtype=float)
        old_body_object = np.asarray(result["T_body_base_object"], dtype=float)
        T_body_camera = old_body_object @ np.linalg.inv(old_camera_object)
    report: dict[str, object] = {
        "source_dir": str(source),
        "view": view,
        "workpiece": args.workpiece,
        "stl_path": cfg["stl_path"],
        "status": status,
        "new_quality_thresholds": {
            "quality_min_fitness": cfg["quality_min_fitness"],
            "quality_max_rmse_m": cfg["quality_max_rmse_m"],
        },
        "old_registration": (
            {"fitness": registration["fitness"], "rmse_m": registration["rmse_m"]}
            if registration else None
        ),
        "new_registration": metrics,
    }
    if T_camera_object is None:
        # A rejected replay has no valid output pose.  Visualize the strongest
        # rejected candidate, rather than silently reusing another workpiece's
        # historical result.
        rejected_candidate, T_camera_object = best_rejected_transform(
            cad, points, cfg, selection_mask, camera_intrinsics
        )
        report["visualization_transform_source"] = "best_rejected_icp_candidate"
        report["visualized_rejected_candidate"] = rejected_candidate
    else:
        report["visualization_transform_source"] = "new_accepted_icp_pose"

    if T_camera_object is not None:
        T_body_object = T_body_camera @ T_camera_object if T_body_camera is not None else None
        overlay = None
        if T_body_object is not None:
            overlay = _save_pose_overlay(
                output, view, cad, T_camera_object, T_body_object, camera_intrinsics
            )
        camera_3d = output / "pose_estimation_camera_3d.png"
        body_3d = output / "pose_estimation_body_base_3d.png"
        _plot_pose_estimation(
            camera_3d, points, cad, T_camera_object, frame_name="camera", seed=int(cfg["seed"])
        )
        if T_body_camera is not None and T_body_object is not None:
            body_points = (T_body_camera[:3, :3] @ points.T).T + T_body_camera[:3, 3]
            _plot_pose_estimation(
                body_3d, body_points, cad, T_body_object, frame_name="body_base", seed=int(cfg["seed"])
            )
        report["T_camera_object"] = T_camera_object.tolist()
        if T_body_object is not None:
            report["T_body_base_object"] = T_body_object.tolist()
        report["visualizations"] = {
            "camera_3d": str(camera_3d),
        }
        if overlay is not None:
            report["visualizations"]["overlay"] = str(overlay)
        if T_body_object is not None:
            report["visualizations"]["body_base_3d"] = str(body_3d)

        attempts = metrics.get("attempts", []) if isinstance(metrics, dict) else []
        if attempts:
            labels = [f"{item['method']}\n{item['initialization']}" for item in attempts]
            rmses_mm = [float(item["rmse_m"]) * 1000.0 for item in attempts]
            figure, axes = plt.subplots(figsize=(14, 5), dpi=150)
            bars = axes.bar(range(len(attempts)), rmses_mm, color="#b94a48")
            axes.axhline(float(cfg["quality_max_rmse_m"]) * 1000.0, color="#222222", linestyle="--", label="RMSE threshold")
            axes.set_xticks(range(len(attempts)), labels, rotation=35, ha="right", fontsize=8)
            axes.set_ylabel("ICP RMSE (mm)")
            axes.set_title("Strict replay: all ICP candidates rejected")
            axes.legend()
            for bar, value in zip(bars, rmses_mm):
                axes.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
            figure.tight_layout()
            summary = output / "strict_icp_rmse_candidates.png"
            figure.savefig(summary)
            plt.close(figure)
            report["visualizations"]["strict_rmse_candidates"] = str(summary)

    (output / "replay_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
