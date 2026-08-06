#!/usr/bin/env python3
"""Overlay a depth-coloured candidate mask on its captured RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--view", default="pc1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument(
        "--show-missing-depth", action="store_true",
        help="Draw selected zero/invalid-depth candidate pixels in magenta.",
    )
    args = parser.parse_args()

    prefix = args.run_dir / args.view
    rgb = cv2.imread(str(prefix.with_name(prefix.name + "_rgb.png")), cv2.IMREAD_COLOR)
    candidate = cv2.imread(
        str(prefix.with_name(prefix.name + "_processed_near.png")), cv2.IMREAD_GRAYSCALE
    )
    depth = np.load(prefix.with_name(prefix.name + "_depth.npy"), allow_pickle=False)
    if rgb is None or candidate is None or depth.shape != rgb.shape[:2]:
        raise RuntimeError("Missing or inconsistent RGB, candidate mask, or depth")

    candidate_mask = candidate > 0
    selected = candidate_mask & np.isfinite(depth) & (depth > 0)
    if not selected.any():
        raise RuntimeError("No valid depth remains after the URDF prompt filter")
    depth_m = depth.astype(np.float32) * args.depth_scale
    low, high = np.percentile(depth_m[selected], (2, 98))
    high = max(float(high), float(low) + 1e-4)
    indexed = np.zeros(depth.shape, dtype=np.uint8)
    indexed[selected] = np.clip((depth_m[selected] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(indexed, cv2.COLORMAP_TURBO)
    overlay = rgb.copy()
    overlay[selected] = (0.35 * rgb[selected] + 0.65 * colour[selected]).astype(np.uint8)
    missing = candidate_mask & ~np.isfinite(depth) | (candidate_mask & (depth <= 0))
    if args.show_missing_depth and missing.any():
        overlay[missing] = (0.35 * rgb[missing] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.rectangle(overlay, (0, 0), (610, 50), (0, 0, 0), -1)
    cv2.putText(overlay, "Depth candidates after URDF prompt exclusion", (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Turbo: {low:.3f} m (blue) to {high:.3f} m (red)", (12, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if args.show_missing_depth and missing.any():
        cv2.putText(overlay, f"Magenta: {int(missing.sum())} expanded missing-depth pixels", (620, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"wrote {args.output}; candidate_pixels={int(selected.sum())}")


if __name__ == "__main__":
    main()
