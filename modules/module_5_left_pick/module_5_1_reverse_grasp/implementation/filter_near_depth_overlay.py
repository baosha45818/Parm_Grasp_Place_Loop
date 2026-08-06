#!/usr/bin/env python3
"""Render unchanged RGB images with the selected depth band in yellow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAPTURE_ROOT = MODULE_ROOT / "output"


def depth_band_overlay(
    rgb: np.ndarray, depth: np.ndarray, scale_m: float, near_m: float, far_m: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep RGB unchanged except for a transparent yellow depth overlay."""
    depth_m = depth.astype(np.float32) * float(scale_m)
    valid = (depth > 0) & (depth_m >= near_m) & (depth_m <= far_m)
    mask = (valid.astype(np.uint8) * 255)

    overlay = rgb.copy()
    yellow = np.array((0, 255, 255), dtype=np.float32)  # BGR
    overlay[valid] = np.rint(
        rgb[valid].astype(np.float32) * (1.0 - alpha) + yellow * alpha
    ).astype(np.uint8)
    return overlay, mask


def process_capture(capture_dir: Path, output_dir: Path, near_m: float, far_m: float, alpha: float) -> dict[str, object]:
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    scale_m = float(metadata["depth_scale_m"])
    rgb_path = capture_dir / "rgb_000.png"
    depth_path = capture_dir / "depth_aligned_000.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise RuntimeError(f"unable to read RGB-D pair in {capture_dir}")
    if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"unexpected aligned RGB-D shape/type in {capture_dir}")

    overlay, mask = depth_band_overlay(rgb, depth, scale_m, near_m, far_m, alpha)
    overlay_path = output_dir / f"{capture_dir.name}_near_depth_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"failed to write near-depth results for {capture_dir}")

    valid = mask > 0
    return {"capture": capture_dir.name, "overlay": overlay_path.name, "selected_pixels": int(np.count_nonzero(valid))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CAPTURE_ROOT / "near_depth_filtered")
    parser.add_argument("--near-m", type=float, default=0.15, help="nearest retained depth, in metres")
    parser.add_argument("--far-m", type=float, default=0.40, help="farthest retained depth, in metres")
    parser.add_argument("--alpha", type=float, default=0.45, help="yellow overlay opacity in [0, 1]")
    args = parser.parse_args()
    if not 0.0 <= args.near_m < args.far_m:
        parser.error("require 0 <= --near-m < --far-m")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")

    captures = sorted(
        path for path in args.capture_root.glob("head_rgbd_*")
        if (path / "metadata.json").is_file() and (path / "rgb_000.png").is_file() and (path / "depth_aligned_000.png").is_file()
    )
    if not captures:
        raise SystemExit(f"no complete head_rgbd_* capture found under {args.capture_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [process_capture(path, args.output_dir, args.near_m, args.far_m, args.alpha) for path in captures]
    print(json.dumps(reports, ensure_ascii=False))
    print(args.output_dir)


if __name__ == "__main__":
    main()
