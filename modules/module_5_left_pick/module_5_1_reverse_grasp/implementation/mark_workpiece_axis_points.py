#!/usr/bin/env python3
"""Segment the held workpiece and mark three evenly spaced 2-D axis points."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve()
MODULE_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from sam2_client import prompt_segment  # noqa: E402


DEFAULT_CAPTURE = MODULE_ROOT / "output/head_rgbd_20260728_121840"
DEFAULT_OUTPUT = MODULE_ROOT / "output/near_depth_filtered/head_rgbd_20260728_121840_workpiece_axis_3_points.png"
DEFAULT_MASK_OVERLAY = MODULE_ROOT / "output/near_depth_filtered/head_rgbd_20260728_121840_workpiece_mask_overlay.png"


def point_on_mask(mask: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    """Return the mask pixel closest to a continuous image-space target."""
    yy, xx = np.nonzero(mask)
    candidates = np.column_stack((xx, yy)).astype(float)
    index = int(np.argmin(np.sum((candidates - target) ** 2, axis=1)))
    return int(candidates[index, 0]), int(candidates[index, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--socket", default="unix:///tmp/sam2-service.sock")
    parser.add_argument("--box", type=int, nargs=4, default=[340, 25, 580, 280], metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--positive", type=int, nargs=2, action="append", default=None, metavar=("X", "Y"))
    parser.add_argument("--negative", type=int, nargs=2, action="append", default=None, metavar=("X", "Y"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-overlay-output", type=Path, default=DEFAULT_MASK_OVERLAY)
    args = parser.parse_args()

    positives = args.positive or [[450, 95]]
    negatives = args.negative or []
    rgb = cv2.imread(str(args.capture_dir / "rgb_000.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"cannot read RGB image from {args.capture_dir}")
    mask = prompt_segment(rgb, list(args.box), positives, negatives, args.socket) > 0
    # The prompted component must be connected to the explicitly selected point.
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    seed_x, seed_y = positives[0]
    if not (0 <= seed_x < rgb.shape[1] and 0 <= seed_y < rgb.shape[0]):
        raise ValueError("positive point lies outside the image")
    label = int(labels[seed_y, seed_x])
    if label == 0:
        raise RuntimeError("SAM2 mask does not contain the selected workpiece point")
    mask = labels == label
    yy, xx = np.nonzero(mask)
    if len(xx) < 100:
        raise RuntimeError("workpiece mask is too small to estimate an axis")

    pixels = np.column_stack((xx, yy)).astype(float)
    center = pixels.mean(axis=0)
    _, _, vectors = np.linalg.svd(pixels - center, full_matrices=False)
    axis = vectors[0]
    projection = (pixels - center) @ axis
    # Keep points inside the mask and away from unstable contour extrema.
    positions = np.quantile(projection, [0.15, 0.50, 0.85])
    axis_points = [point_on_mask(mask, center + axis * value) for value in positions]

    result = rgb.copy()
    line_start = center + axis * np.quantile(projection, 0.05)
    line_end = center + axis * np.quantile(projection, 0.95)
    cv2.line(result, tuple(np.rint(line_start).astype(int)), tuple(np.rint(line_end).astype(int)), (255, 255, 0), 2, cv2.LINE_AA)
    for index, point in enumerate(axis_points, 1):
        cv2.circle(result, point, 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(result, str(index), (point[0] + 12, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise RuntimeError(f"cannot write {args.output}")
    mask_overlay = rgb.copy()
    green = np.array((0, 255, 0), dtype=np.float32)
    mask_overlay[mask] = np.rint(
        rgb[mask].astype(np.float32) * 0.55 + green * 0.45
    ).astype(np.uint8)
    for index, point in enumerate(positives, 1):
        cv2.circle(mask_overlay, tuple(point), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(mask_overlay, str(index), (point[0] + 11, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    for point in negatives:
        cv2.circle(mask_overlay, tuple(point), 8, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.line(mask_overlay, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(mask_overlay, (point[0] - 4, point[1] + 4), (point[0] + 4, point[1] - 4), (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(args.mask_overlay_output), mask_overlay):
        raise RuntimeError(f"cannot write {args.mask_overlay_output}")
    print({"output": str(args.output), "mask_overlay": str(args.mask_overlay_output), "axis_points_xy": axis_points, "mask_pixels": int(mask.sum())})


if __name__ == "__main__":
    main()
