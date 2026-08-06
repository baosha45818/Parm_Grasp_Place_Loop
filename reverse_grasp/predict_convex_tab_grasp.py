#!/usr/bin/env python3
"""Project the operator-selected convex-tab grasp point using a verified reverse pose.

This is deliberately a prediction/visualization step only: no robot, head, or gripper
command is issued.  Inputs may be read anywhere, but every new artifact is constrained
to ``reverse_grasp``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def output_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        path.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须位于 {HERE}: {path}") from exc
    return path


def project(K: np.ndarray, point_camera: np.ndarray) -> tuple[int, int]:
    if point_camera[2] <= 1e-6:
        raise RuntimeError("抓取点不在相机前方")
    pixel = K @ point_camera
    return int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """Remove small ICP numerical shear before using object axes for point clouds."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    result = u @ vt
    if np.linalg.det(result) < 0.:
        u[:, -1] *= -1.
        result = u @ vt
    return result


def load_object_offset(path: Path | None) -> tuple[np.ndarray, dict]:
    """Read an optional persistent manual correction in the workpiece frame."""
    if path is None:
        return np.zeros(3, dtype=float), {
            "enabled": False,
            "frame_id": "workpiece2_top_view",
            "offset_object_m": [0.0, 0.0, 0.0],
        }
    resolved = path.expanduser().resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    values = np.asarray(data.get("offset_object_m"), dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"抓取偏置必须是三个有限米值: {resolved}")
    frame = data.get("frame_id")
    if frame != "workpiece2_top_view":
        raise ValueError(f"抓取偏置坐标系必须是 workpiece2_top_view，实际为 {frame!r}")
    return values, {
        "enabled": True,
        "path": str(resolved),
        "frame_id": frame,
        "offset_object_m": values.round(8).tolist(),
    }


def draw_tab_cloud(scene_camera: np.ndarray, tab_camera: np.ndarray, selected_camera: np.ndarray,
                   approach_camera: np.ndarray, path: Path) -> None:
    rng = np.random.default_rng(11)
    scene = scene_camera[rng.choice(len(scene_camera), min(8000, len(scene_camera)), replace=False)]
    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(scene[:, 0], scene[:, 1], scene[:, 2], s=.25, c="#999999", alpha=.20, label="SAM workpiece cloud")
    axis.scatter(tab_camera[:, 0], tab_camera[:, 1], tab_camera[:, 2], s=2.0, c="#ff8c00", alpha=.85, label="+X convex-tab cloud")
    axis.scatter(*selected_camera, s=70, c="red", marker="o", label="measured grasp surface point")
    axis.quiver(*selected_camera, *approach_camera, length=.045, color="b", arrow_length_ratio=.16, label="approach / object +Z")
    axis.set_xlabel("camera X (m)")
    axis.set_ylabel("camera Y (m)")
    axis.set_zlabel("camera Z (m)")
    axis.set_title("SAM point cloud selected convex tab")
    axis.legend(loc="upper right")
    all_points = np.vstack((scene, tab_camera, selected_camera[None, :]))
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    pad = np.maximum((hi - lo) * .12, .025)
    axis.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    axis.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    axis.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    axis.set_box_aspect((1, 1, .6))
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True, help="fixed_reverse_pose.json")
    parser.add_argument("--capture-dir", type=Path, required=True, help="对应 RGB-D capture 目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refine-pixel", type=int, nargs=2, metavar=("U", "V"), default=None,
                        help="可选：凸起在 RGB 中的人工确认像素；以局部深度反投影校正抓取点")
    parser.add_argument("--auto-search-radius-px", type=int, default=50,
                        help="自动凸起深度校正时，在 CAD 投影点周围搜索的半径")
    parser.add_argument("--tab-x-extreme-percentile", type=float, default=80.0,
                        help="候选限定为工件 +X 最外端的百分位阈值")
    parser.add_argument("--tab-height-percentile", type=float, default=70.0,
                        help="在 +X 端候选中保留离桌面最高的百分位阈值")
    parser.add_argument("--grasp-down-offset-m", type=float, default=.010,
                        help="实际夹爪 TCP 沿工件 +Z（向桌面）下移量，默认 10 mm")
    parser.add_argument("--grasp-offset-config", type=Path, default=None,
                        help="可选：手动确认的工件坐标系抓取偏置 JSON；未传入则不加偏置")
    args = parser.parse_args()

    pose_path = args.pose.expanduser().resolve()
    capture = args.capture_dir.expanduser().resolve()
    output = output_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    if not pose_path.is_file() or not capture.is_dir():
        raise FileNotFoundError("pose 或 capture-dir 不存在")
    output.mkdir(parents=True)
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    config = json.loads((HERE / "workpiece_2_orientation_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("无法读取 rgb_000.png")

    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]])
    T = np.asarray(pose["T_camera_object"], dtype=float)
    T[:3, :3] = nearest_rotation(T[:3, :3])
    raw_table_normal = np.asarray(pose.get("table_normal_camera", []), dtype=float)
    table_normal = normalize(raw_table_normal) if raw_table_normal.shape == (3,) else None
    point_object = np.asarray(config["grasp_point_object_m"], dtype=float)
    point_camera = T[:3, :3] @ point_object + T[:3, 3]
    pixel = project(K, point_camera)
    refinement = {"method": "cad_landmark_from_fixed_icp", "pixel": list(pixel)}
    depth = cv2.imread(str(capture / "depth_aligned_000.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("无法读取 depth_aligned_000.png")
    if args.refine_pixel is not None:
        u, v = args.refine_pixel
        if not (3 <= u < depth.shape[1] - 3 and 3 <= v < depth.shape[0] - 3):
            raise ValueError("refine-pixel 超出有效深度图范围")
        window = depth[v - 3:v + 4, u - 3:u + 4]
        valid = window[window > 0]
        if len(valid) < 12:
            raise RuntimeError("凸起像素附近没有足够的有效深度")
        depth_m = float(np.median(valid)) * float(metadata["depth_scale_m"])
        point_camera = np.array([(u - intr["ppx"]) * depth_m / intr["fx"],
                                 (v - intr["ppy"]) * depth_m / intr["fy"], depth_m])
        point_object = T[:3, :3].T @ (point_camera - T[:3, 3])
        pixel = (u, v)
        refinement = {
            "method": "operator_pixel_plus_local_median_depth",
            "pixel": list(pixel), "median_depth_m": depth_m, "window": "7x7",
            "note": "This corrects the grasp-contact XY from the locally observed convex tab; the fixed ICP pose remains the orientation source.",
        }
    else:
        # The red-circle tab is the workpiece's +X extreme.  Select it from
        # the freshly generated SAM mask in the ICP object frame, rather than
        # merely choosing a close-depth point around the CAD projection (which
        # can jump to a nearby hole or mounting plate).
        # The fresh pipeline stores the raw SAM output for audit and, when
        # available, a table-plane-cleaned derivative for geometry.  The tab
        # selector must use the same clean point set as ICP.
        masks_file = capture / "sam2_masks_without_table" / "masks.npz"
        if not masks_file.is_file():
            masks_file = capture / "sam2_masks" / "masks.npz"
        mask_index = int(pose.get("mask_index", 0))
        if not masks_file.is_file():
            raise FileNotFoundError(f"自动凸起选择缺少 SAM mask: {masks_file}")
        masks = np.load(masks_file)["masks"]
        if not 0 <= mask_index < len(masks):
            raise RuntimeError(f"ICP mask_index={mask_index} 不在 SAM masks 范围内")
        sam_mask = masks[mask_index].astype(bool) & (depth > 0)
        ys, xs = np.nonzero(sam_mask)
        if len(xs) < 150:
            raise RuntimeError("SAM 工件 mask 的有效深度点不足，无法自动定位凸起")
        depths_m = depth[ys, xs].astype(float) * float(metadata["depth_scale_m"])
        observed_camera = np.column_stack(((xs - intr["ppx"]) * depths_m / intr["fx"],
                                           (ys - intr["ppy"]) * depths_m / intr["fy"], depths_m))
        # Row-vector form of p_object = R^T(p_camera-t).
        observed_object = (observed_camera - T[:3, 3]) @ T[:3, :3]
        if table_normal is None:
            raise RuntimeError("自动翘起结构选择需要 table_normal_camera")
        if not (50.0 <= args.tab_x_extreme_percentile < 100.0 and 50.0 <= args.tab_height_percentile < 100.0):
            raise ValueError("tab-x-extreme-percentile 和 tab-height-percentile 必须在 [50,100) 内")
        # First constrain the point cloud to the +X outer end. Then retain the
        # upper part of that end only, measured along table->camera normal.
        # This is the geometric definition of the raised +X structure.
        x_threshold = float(np.percentile(observed_object[:, 0], args.tab_x_extreme_percentile))
        x_extreme = observed_object[:, 0] >= x_threshold
        height_along_table_normal = observed_camera @ table_normal
        height_threshold = float(np.percentile(height_along_table_normal[x_extreme], args.tab_height_percentile))
        raised_x_extreme = x_extreme & (height_along_table_normal >= height_threshold)
        tab_mask = np.zeros(depth.shape, dtype=np.uint8)
        tab_mask[ys[raised_x_extreme], xs[raised_x_extreme]] = 1
        count, labels, stats, centers = cv2.connectedComponentsWithStats(tab_mask, connectivity=8)
        components = [(int(stats[index, cv2.CC_STAT_AREA]), index)
                      for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= 20]
        if not components:
            raise RuntimeError("SAM mask 中没有足够大的工件 +X 端凸起区域")
        area, index = max(components)
        cu, cv = centers[index]
        u, v = int(round(cu)), int(round(cv))
        component_indices = labels[ys, xs] == index
        tab_camera = observed_camera[component_indices]
        tab_object = observed_object[component_indices]
        # The coordinate-wise median is robust to edge pixels and holes; pick
        # the nearest real measured point so the TCP never targets empty space.
        tab_center_camera = np.median(tab_camera, axis=0)
        medoid_index = int(np.argmin(np.sum((tab_camera - tab_center_camera) ** 2, axis=1)))
        point_camera = tab_camera[medoid_index]
        point_object = tab_object[medoid_index]
        depth_m = float(point_camera[2])
        pixel = (u, v)
        refinement = {
            "method": "sam_mask_icp_positive_x_raised_structure_center",
            "pixel": list(pixel), "median_depth_m": depth_m,
            "object_x_percentile": float(args.tab_x_extreme_percentile), "object_x_threshold_m": x_threshold,
            "height_percentile_within_positive_x": float(args.tab_height_percentile),
            "height_threshold_table_normal_m": height_threshold,
            "component_area_px": area,
            "component_centroid_pixel": [round(float(cu), 3), round(float(cv), 3)],
            "point_selection": "3d_component_medoid",
            "note": "Automatic grasp selection: largest raised coherent SAM point-cloud structure within the ICP workpiece +X outer end; its 3-D medoid supplies the contact point.",
        }
    if args.grasp_down_offset_m < 0 or args.grasp_down_offset_m > .030:
        raise ValueError("grasp-down-offset-m 必须在 [0, 0.030] m")
    manual_offset_object, manual_offset_info = load_object_offset(args.grasp_offset_config)
    surface_point_camera = point_camera.copy()
    surface_point_object = point_object.copy()
    # Use the fitted tabletop normal for physical down, exactly as Module 5-1
    # does. The normal is oriented table->camera, hence its negative is the
    # real approach/down direction. Object +Z is only a fallback.
    approach_camera = -table_normal if table_normal is not None else normalize(T[:3, 2])
    point_camera = point_camera + approach_camera * float(args.grasp_down_offset_m)
    point_object = T[:3, :3].T @ (point_camera - T[:3, 3])
    nominal_grasp_point_object = point_object.copy()
    nominal_grasp_point_camera = point_camera.copy()
    # Apply the learned correction in the newly estimated object frame, so it
    # follows the workpiece when its tabletop yaw changes between captures.
    point_object = point_object + manual_offset_object
    point_camera = T[:3, :3] @ point_object + T[:3, 3]
    
    # Object -Z faces up/out of the table for this confirmed tabletop-reverse placement.
    outward_camera = -approach_camera
    visible = image.copy()
    cv2.circle(visible, pixel, 13, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(visible, pixel, 3, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(visible, "convex-tab grasp", (pixel[0] + 16, pixel[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(visible, "convex-tab grasp", (pixel[0] + 16, pixel[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1, cv2.LINE_AA)
    overlay = output / "convex_tab_grasp_overlay.png"
    if not cv2.imwrite(str(overlay), visible):
        raise RuntimeError(f"无法写入 {overlay}")
    pointcloud_artifact = None
    pointcloud_3d_artifact = None
    if args.refine_pixel is None:
        pointcloud_artifact = output / "convex_tab_pointcloud.npz"
        np.savez_compressed(pointcloud_artifact,
                            sam_workpiece_points_camera_m=observed_camera,
                            convex_tab_points_camera_m=tab_camera,
                            selected_surface_point_camera_m=surface_point_camera,
                            selected_surface_point_object_m=surface_point_object)
        pointcloud_3d_artifact = output / "convex_tab_pointcloud_3d.png"
        draw_tab_cloud(observed_camera, tab_camera, surface_point_camera, approach_camera, pointcloud_3d_artifact)

    payload = {
        "schema_version": 1,
        "event": "workpiece_2_convex_tab_grasp_prediction",
        "source_pose": str(pose_path),
        "frame_id": pose["frame_id"],
        "object_frame": pose["object_frame"],
        "grasp_contact_point_object_m": point_object.round(8).tolist(),
        "grasp_contact_point_camera_m": point_camera.round(8).tolist(),
        "surface_point_object_m_before_down_offset": surface_point_object.round(8).tolist(),
        "surface_point_camera_m_before_down_offset": surface_point_camera.round(8).tolist(),
        "grasp_down_offset_m": float(args.grasp_down_offset_m),
        "nominal_grasp_contact_point_object_m_before_manual_offset": nominal_grasp_point_object.round(8).tolist(),
        "nominal_grasp_contact_point_camera_m_before_manual_offset": nominal_grasp_point_camera.round(8).tolist(),
        "manual_grasp_offset": manual_offset_info,
        "image_pixel": list(pixel),
        "outward_direction_camera": outward_camera.round(8).tolist(),
        "approach_direction_camera": approach_camera.round(8).tolist(),
        "approach_direction_source": "negative_table_normal_camera" if table_normal is not None else "object_positive_z_fallback",
        "placement_constraint": "confirmed tabletop reverse: object +Z points into the table; object -Z faces the camera/outward",
        "grasp_point_definition": config["grasp_point_definition"],
        "local_refinement": refinement,
        "artifacts": {"overlay": str(overlay), "pointcloud_npz": str(pointcloud_artifact) if pointcloud_artifact else None,
                      "pointcloud_3d": str(pointcloud_3d_artifact) if pointcloud_3d_artifact else None},
        "created_at": time.time(),
    }
    result = output / "convex_tab_grasp_prediction.json"
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CONVEX_TAB_GRASP={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
