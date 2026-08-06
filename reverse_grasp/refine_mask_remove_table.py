#!/usr/bin/env python3
"""Remove tabletop pixels accidentally included in an existing SAM2 mask.

This is deliberately an offline, non-destructive post-process.  It reads an
existing RGB-D capture and SAM ``masks.npz`` and writes a new mask directory
under ``reverse_grasp``; the original SAM result is never changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tool.inspection_support.mask_depth import depth_to_xyz


def under_project(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"--out 必须位于项目目录 {ROOT}: {path}") from exc
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height-mm", type=float, default=1.5,
                        help="仅保留高于拟合桌面的深度像素；默认 1.5 mm")
    parser.add_argument("--edge-dilate-px", type=int, default=2,
                        help="以受支持的前景为种子，在原 SAM mask 内补回的边缘像素半径")
    return parser.parse_args()


def table_plane(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fit the dominant tabletop plane and orient its normal toward the camera."""
    import open3d as o3d

    points = xyz[valid]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, _ = cloud.voxel_down_sample(.003).segment_plane(.004, 3, 1200)
    plane = np.asarray(plane, dtype=float)
    plane /= np.linalg.norm(plane[:3])
    if np.dot(plane[:3], -points.mean(axis=0)) < 0:
        plane *= -1
    return plane


def main() -> int:
    args = parse_args()
    capture = args.capture_dir.expanduser().resolve()
    source = args.masks_dir.expanduser().resolve()
    out = under_project(args.out)
    required = [capture / "metadata.json", capture / "rgb_000.png",
                capture / "depth_aligned_000.png", source / "masks.npz"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists():
        raise FileExistsError(f"输出目录已存在: {out}")

    meta = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    rgb = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture / "depth_aligned_000.png"), cv2.IMREAD_UNCHANGED)
    intr = meta["color_intrinsics"]
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.]])
    xyz = depth_to_xyz(depth, K, float(meta["depth_scale_m"]))
    valid = (depth > 0) & (depth < 65535)
    plane = table_plane(xyz, valid)
    height = xyz @ plane[:3] + plane[3]
    raw_masks = np.load(source / "masks.npz")["masks"].astype(bool)

    refined = []
    diagnostics = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * args.edge_dilate_px + 1, 2 * args.edge_dilate_px + 1))
    for index, raw in enumerate(raw_masks):
        # Table plane points have height ~= 0.  The workpiece is above it in
        # this camera convention.  Invalid-depth pixels cannot support ICP.
        supported = raw & valid & (height >= args.height_mm / 1000.0)
        # Keep a narrow in-mask halo only adjacent to genuine raised geometry.
        # This restores object contours but cannot grow into remote tabletop.
        halo = cv2.dilate(supported.astype(np.uint8), kernel).astype(bool) & raw & valid
        mask = supported | halo
        if int(mask.sum()) < 120:
            raise RuntimeError(f"mask {index}: 桌面剔除后仅剩 {int(mask.sum())} 个深度点；请降低 --height-mm")
        refined.append(mask.astype(np.uint8))
        diagnostics.append({"mask_index": index, "raw_pixels": int(raw.sum()),
                            "valid_raw_pixels": int((raw & valid).sum()),
                            "retained_pixels": int(mask.sum()),
                            "removed_table_pixels": int((raw & valid & ~mask).sum())})

    out.mkdir(parents=True)
    refined_np = np.asarray(refined, dtype=np.uint8)
    np.savez_compressed(out / "masks.npz", masks=refined_np)
    raw_overlay = rgb.copy()
    refined_overlay = rgb.copy()
    removed_overlay = rgb.copy()
    for mask_index, raw in enumerate(raw_masks):
        mask = refined_np[mask_index].astype(bool)
        raw_overlay[raw] = (0.45 * raw_overlay[raw] + 0.55 * np.array([0, 165, 255])).astype(np.uint8)
        refined_overlay[mask] = (0.35 * refined_overlay[mask] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
        removed = raw & valid & ~mask
        removed_overlay[removed] = (0.25 * removed_overlay[removed] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
        removed_overlay[mask] = (0.55 * removed_overlay[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(str(out / "raw_sam_overlay.png"), raw_overlay)
    cv2.imwrite(str(out / "sam_mask_without_table_overlay.png"), refined_overlay)
    cv2.imwrite(str(out / "removed_table_diagnostic_overlay.png"), removed_overlay)
    (out / "meta.json").write_text(json.dumps({
        "source_masks_dir": str(source), "capture_dir": str(capture),
        "method": "dominant_table_plane_height_filter_with_in_mask_edge_halo",
        "table_plane_camera": plane.tolist(), "height_threshold_m": args.height_mm / 1000.0,
        "edge_dilate_px": args.edge_dilate_px, "masks": diagnostics,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"REFINED_MASKS={out}")
    print(f"OVERLAY={out / 'sam_mask_without_table_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
