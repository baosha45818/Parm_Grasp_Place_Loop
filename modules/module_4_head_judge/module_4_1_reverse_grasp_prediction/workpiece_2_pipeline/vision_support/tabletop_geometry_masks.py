#!/usr/bin/env python3
"""仅用 RGB-D 桌面平面分离工件点云，不调用 SAM2。

在已标定的头部观察位，安全 ROI 中的主平面就是桌面。对完整深度点云做
RANSAC 平面拟合后，所有位于桌面上方的有效深度像素均视为候选工件；按
连通域拆成独立 mask，供后续 CAD/ICP 直接使用。输出仍使用 ``masks.npz``，
以兼容现有 ICP 接口。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent


def depth_to_xyz(depth: np.ndarray, intrinsic: np.ndarray, scale_m: float) -> np.ndarray:
    yy, xx = np.indices(depth.shape)
    z = depth.astype(float) * scale_m
    return np.dstack(((xx - intrinsic[0, 2]) * z / intrinsic[0, 0],
                      (yy - intrinsic[1, 2]) * z / intrinsic[1, 1], z))


def fit_table_plane(points: np.ndarray) -> np.ndarray:
    """Return a plane whose normal points from the table towards the camera."""
    o3d.utility.random.seed(17)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(.003)
    if len(cloud.points) < 200:
        raise RuntimeError("有效深度点不足，无法拟合桌面平面")
    plane, _ = cloud.segment_plane(distance_threshold=.004, ransac_n=3, num_iterations=1600)
    plane = np.asarray(plane, dtype=float)
    plane /= max(float(np.linalg.norm(plane[:3])), 1e-12)
    # The camera origin is [0, 0, 0].  A positive signed height should point
    # from the tabletop towards the camera, where the workpiece is located.
    if plane[3] < 0.0:
        plane *= -1.0
    return plane


def main() -> int:
    workspace = json.loads((MODULE_ROOT / "config/tabletop_workspace.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--height-mm", type=float, default=7.0,
                        help="保留高于桌面的最小高度；默认 7 mm")
    parser.add_argument("--min-component-pixels", type=int, default=300,
                        help="保留的最小深度连通域像素数")
    parser.add_argument("--roi", nargs=4, type=int, default=workspace["safe_roi_xywh"],
                        metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()
    if args.height_mm <= 0.0 or args.min_component_pixels < 1:
        parser.error("--height-mm 必须 > 0，--min-component-pixels 必须 >= 1")

    capture = args.capture_dir.expanduser().resolve()
    output = (args.out or capture / "geometry_masks").expanduser().resolve()
    metadata_path = capture / "metadata.json"
    rgb_path = capture / "rgb_000.png"
    depth_path = capture / "depth_aligned_000.png"
    if not all(path.is_file() for path in (metadata_path, rgb_path, depth_path)):
        raise FileNotFoundError("capture 缺少 metadata.json、rgb_000.png 或 depth_aligned_000.png")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
        raise RuntimeError("RGB-D 图像无法读取或尺寸不一致")
    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]])
    xyz = depth_to_xyz(depth, K, float(metadata["depth_scale_m"]))
    valid = (depth > 0) & np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > .20) & (xyz[:, :, 2] < 2.50)
    if np.count_nonzero(valid) < 1000:
        raise RuntimeError("有效深度像素不足，无法分离桌面")
    plane = fit_table_plane(xyz[valid])
    height_m = xyz @ plane[:3] + plane[3]

    x, y, width, height = args.roi
    roi = np.zeros(depth.shape, dtype=bool)
    roi[max(0, y):min(depth.shape[0], y + height), max(0, x):min(depth.shape[1], x + width)] = True
    foreground = valid & roi & (height_m >= args.height_mm / 1000.0)
    # Remove isolated depth noise only.  Do not use RGB or SAM-derived rules:
    # every non-table connected component remains available to ICP.
    foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    masks: list[np.ndarray] = []
    components: list[dict] = []
    overlay = rgb.copy()
    rng = np.random.default_rng(17)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < args.min_component_pixels:
            continue
        mask = labels == index
        masks.append(mask.astype(np.uint8))
        ys, xs = np.nonzero(mask)
        color = rng.integers(70, 255, 3, dtype=np.uint8)
        overlay[mask] = (.50 * overlay[mask] + .50 * color).astype(np.uint8)
        components.append({
            "mask_index": len(masks) - 1,
            "pixels": area,
            "bbox_xywh": [int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]),
                             int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])],
            "centroid_xy": [float(xs.mean()), float(ys.mean())],
            "height_median_m": float(np.median(height_m[mask])),
            "height_max_m": float(np.max(height_m[mask])),
        })

    output.mkdir(parents=True)
    cv2.imwrite(str(output / "geometry_masks_overlay.png"), overlay)
    plane_overlay = rgb.copy()
    plane_pixels = valid & roi & (np.abs(height_m) <= .004)
    plane_overlay[plane_pixels] = (.55 * plane_overlay[plane_pixels] + .45 * np.array([0, 220, 0])).astype(np.uint8)
    cv2.imwrite(str(output / "table_plane_inliers_overlay.png"), plane_overlay)
    np.savez_compressed(output / "masks.npz", masks=np.asarray(masks, dtype=np.uint8))
    (output / "meta.json").write_text(json.dumps({
        "method": "ransac_table_plane_subtraction_no_sam2",
        "plane_camera": plane.tolist(),
        "height_threshold_m": args.height_mm / 1000.0,
        "safe_roi_xywh": [int(value) for value in args.roi],
        "valid_depth_pixels": int(np.count_nonzero(valid)),
        "foreground_pixels_before_components": int(np.count_nonzero(foreground)),
        "components": components,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GEOMETRY_MASKS={output}")
    print(f"TABLE_PLANE={plane.tolist()}")
    print(f"GEOMETRY_COMPONENTS={len(masks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
