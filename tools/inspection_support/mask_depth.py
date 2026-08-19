"""Build object point clouds from a SAM2 mask and aligned RealSense depth."""

from __future__ import annotations

import cv2
import numpy as np
import warnings


def valid_median_depth(
    depth_frames: list[np.ndarray],
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median only valid per-pixel samples; never let missing zero frames erase data."""
    if not depth_frames:
        raise ValueError("depth_frames must not be empty")
    stack = np.stack([np.asarray(frame, dtype=np.uint16) for frame in depth_frames])
    min_units = float(min_depth_m) / float(depth_scale)
    max_units = float(max_depth_m) / float(depth_scale)
    valid = (stack >= min_units) & (stack <= max_units)
    support = np.count_nonzero(valid, axis=0).astype(np.uint16)
    masked = np.where(valid, stack.astype(np.float32), np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = np.nanmedian(masked, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.rint(median).astype(np.uint16), support


def depth_to_xyz(depth_u16: np.ndarray, K: np.ndarray, depth_scale: float) -> np.ndarray:
    h, w = depth_u16.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    z = depth_u16.astype(np.float32) * float(depth_scale)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def mask_depth_to_points_cam(mask_u8: np.ndarray,
                             depth_u16: np.ndarray,
                             K: np.ndarray,
                             depth_scale: float,
                             min_depth_m: float = 0.05,
                             max_depth_m: float = 1.2,
                             erode_px: int = 1) -> np.ndarray:
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    xyz = depth_to_xyz(depth_u16, K, depth_scale)
    z = xyz[..., 2]
    valid = (mask > 0) & np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    pts = xyz[valid].astype(np.float64)
    return pts


def mask_depth_diagnostics(mask_u8: np.ndarray,
                           depth_u16: np.ndarray,
                           depth_scale: float,
                           min_depth_m: float = 0.05,
                           max_depth_m: float = 1.2,
                           erode_px: int = 1) -> dict:
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask_used = cv2.erode(mask, k, iterations=1)
    else:
        mask_used = mask

    depth_m = np.asarray(depth_u16, dtype=np.float32) * float(depth_scale)
    raw_valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 1e-6)
    valid = (
        (mask_used > 0)
        & np.isfinite(depth_m)
        & (depth_m >= float(min_depth_m))
        & (depth_m <= float(max_depth_m))
    )

    ys, xs = np.nonzero(mask > 0)
    if len(xs) > 0:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    else:
        bbox = [0, 0, 0, 0]

    stats: dict[str, float | int | list[int] | np.ndarray] = {
        "mask_px": int(np.count_nonzero(mask)),
        "mask_used_px": int(np.count_nonzero(mask_used)),
        "mask_bbox_xywh": bbox,
        "raw_depth_valid_px": int(np.count_nonzero(raw_valid)),
        "depth_valid_px": int(np.count_nonzero(valid)),
        "depth_valid_ratio": float(np.count_nonzero(valid) / max(np.count_nonzero(mask_used), 1)),
        "valid_mask_u8": (valid.astype(np.uint8) * 255),
    }
    if np.any(raw_valid):
        vals = depth_m[raw_valid]
        stats["raw_depth_min_m"] = float(vals.min())
        stats["raw_depth_median_m"] = float(np.median(vals))
        stats["raw_depth_max_m"] = float(vals.max())
    else:
        stats["raw_depth_min_m"] = 0.0
        stats["raw_depth_median_m"] = 0.0
        stats["raw_depth_max_m"] = 0.0
    if np.any(valid):
        vals = depth_m[valid]
        stats["depth_min_m"] = float(vals.min())
        stats["depth_median_m"] = float(np.median(vals))
        stats["depth_max_m"] = float(vals.max())
    else:
        stats["depth_min_m"] = 0.0
        stats["depth_median_m"] = 0.0
        stats["depth_max_m"] = 0.0
    return stats


def voxel_downsample_np(points_xyz: np.ndarray, voxel_m: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0 or voxel_m <= 0:
        return pts
    keys = np.floor(pts / float(voxel_m)).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(unique_idx)]


def remove_radius_outliers_np(points_xyz: np.ndarray,
                              z_mad_scale: float = 4.0) -> np.ndarray:
    """Cheap depth-tail cleanup for a segmented object cloud."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 20:
        return pts
    med = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist)))
    if mad < 1e-9:
        return pts
    keep = dist <= np.median(dist) + float(z_mad_scale) * 1.4826 * mad
    return pts[keep]
