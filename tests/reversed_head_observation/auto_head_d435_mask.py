"""Non-interactive SAM2 mask generation for the fixed reversed-head observation pose."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tool.inspection_support.mask_depth import (
    mask_depth_diagnostics,
    mask_depth_to_points_cam,
    remove_radius_outliers_np,
    valid_median_depth,
    voxel_downsample_np,
)
from tool.inspection_support.sam2_icp import prompt_segment


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "head_d435_auto_mask_config.json"


def _scale_point(point: list[float], width: int, height: int) -> list[int]:
    return [int(round(float(point[0]) * (width - 1))), int(round(float(point[1]) * (height - 1)))]


def automatic_prompts(image: np.ndarray, config: dict) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Scale fixed-pose normalized prompt geometry to this captured image."""
    height, width = image.shape[:2]
    roi = config["roi_xyxy_normalized"]
    box = _scale_point(roi[:2], width, height) + _scale_point(roi[2:], width, height)
    positives = [_scale_point(point, width, height) for point in config["positive_points_normalized"]]
    negatives = [_scale_point(point, width, height) for point in config["negative_points_normalized"]]
    return box, positives, negatives


def capture_auto_view(camera, cfg: dict, name: str, output: Path, sam2_url: str, prompt_config: Path = DEFAULT_CONFIG):
    """Capture RGB-D and produce the same artifacts as capture_view(), without UI input."""
    prompt_spec = json.loads(prompt_config.read_text(encoding="utf-8"))
    frames = [camera.read() for _ in range(int(cfg["capture_frames"]))]
    frame = frames[-1]
    if cfg.get("depth_aggregation") == "valid_median":
        depth, depth_support = valid_median_depth(
            [item.depth for item in frames], frame.depth_scale,
            float(cfg["min_depth_m"]), float(cfg["max_depth_m"]),
        )
    else:
        stack = np.stack([item.depth for item in frames])
        depth = np.median(stack, axis=0).astype(np.uint16)
        depth_support = np.count_nonzero(stack, axis=0).astype(np.uint16)
    if depth.shape != frame.color.shape[:2]:
        raise RuntimeError("RGB/depth are not aligned; automatic RGB prompts are unsafe")

    # At the calibrated handoff observation pose the workpiece is stationary
    # in image coordinates.  Use the two approved prompt locations from the
    # overlay directly instead of re-detecting RGB-D components on each run.
    # SAM2 still produces the mask; only its prompts are fixed automatically.
    box, positives, negatives = automatic_prompts(frame.color, prompt_spec)
    if len(positives) != 2:
        raise ValueError("自动标点配置必须恰好包含工件正点 1 和 2")
    mask = prompt_segment(frame.color, box, positives, negatives, sam2_url)
    if not np.any(mask):
        raise RuntimeError("automatic SAM2 mask is empty")
    diagnostics = mask_depth_diagnostics(
        mask, depth, frame.depth_scale, min_depth_m=float(cfg["min_depth_m"]),
        max_depth_m=float(cfg["max_depth_m"]), erode_px=int(cfg["erode_px"]),
    )
    points = mask_depth_to_points_cam(
        mask, depth, frame.depth_intrinsics, frame.depth_scale,
        min_depth_m=float(cfg["min_depth_m"]), max_depth_m=float(cfg["max_depth_m"]),
        erode_px=int(cfg["erode_px"]),
    )
    raw_point_count = int(len(points))
    points = remove_radius_outliers_np(points, float(cfg.get("outlier_mad_scale", 4.0)))
    filtered_point_count = int(len(points))
    points = voxel_downsample_np(points, float(cfg["observed_voxel_m"]))

    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / f"{name}_rgb.png"), frame.color)
    cv2.imwrite(str(output / f"{name}_mask.png"), mask)
    cv2.imwrite(str(output / f"{name}_valid_depth.png"), diagnostics["valid_mask_u8"])
    support_visual = np.clip(depth_support.astype(np.float32) * (255.0 / max(int(cfg["capture_frames"]), 1)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(output / f"{name}_depth_support.png"), support_visual)
    overlay = frame.color.copy()
    color = np.asarray(prompt_spec.get("mask_color_bgr", [0, 255, 0]), dtype=np.float32)
    alpha = float(prompt_spec.get("mask_alpha", 0.45))
    selected = mask > 0
    overlay[selected] = np.rint(frame.color[selected].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
    for index, point in enumerate(positives, 1):
        cv2.circle(overlay, tuple(point), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(overlay, str(index), (point[0] + 9, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    for point in negatives:
        cv2.drawMarker(overlay, tuple(point), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.imwrite(str(output / f"{name}_mask_overlay.png"), overlay)
    np.save(output / f"{name}_depth.npy", depth)
    np.save(output / f"{name}_depth_support.npy", depth_support)
    np.save(output / f"{name}_points_camera.npy", points)
    mask_depth = depth[mask > 0]
    nonzero = mask_depth[mask_depth > 0]
    report = {
        "mask_mode": "fixed_points_1_2_sam2",
        "prompt_config": str(prompt_config),
        "roi_xyxy": box,
        "positive_points": positives,
        "negative_points": negatives,
        "mask_pixels": int(selected.sum()),
        "depth_scale_m_per_unit": float(frame.depth_scale),
        "camera_intrinsics": frame.depth_intrinsics.tolist(),
        "mask_raw_depth": None if not nonzero.size else {"min": float(nonzero.min()), "median": float(np.median(nonzero)), "max": float(nonzero.max())},
        "raw_point_count": raw_point_count,
        "outlier_filtered_point_count": filtered_point_count,
        "valid_point_count": int(len(points)),
        "depth_valid_ratio": float(diagnostics["depth_valid_ratio"]),
        "depth_aggregation": str(cfg.get("depth_aggregation", "raw_median")),
        "capture_frames": int(cfg["capture_frames"]),
    }
    (output / f"{name}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return points, report
