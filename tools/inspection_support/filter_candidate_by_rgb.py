#!/usr/bin/env python3
"""Apply an RGB brightness exclusion to saved filtered depth candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", default="pc1")
    parser.add_argument("--max-rgb-threshold", type=int, default=40)
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="After RGB exclusion, retain only the largest 8-connected candidate region.",
    )
    args = parser.parse_args()
    prefix = args.view
    rgb = cv2.imread(str(args.run_dir / f"{prefix}_rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(args.run_dir / f"{prefix}_depth.npy", allow_pickle=False)
    candidate = cv2.imread(str(args.run_dir / f"{prefix}_processed_near.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or candidate is None or rgb.shape[:2] != depth.shape or candidate.shape != depth.shape:
        raise RuntimeError("Missing or inconsistent RGB-D/candidate files")
    # Strictly remove pixels whose brightest B/G/R channel is below the cutoff.
    # This deliberately keeps exactly max(B,G,R) == threshold.
    dark = np.max(rgb, axis=2) < args.max_rgb_threshold
    kept = (candidate > 0) & ~dark
    before_lcc = int(kept.sum())
    if args.keep_largest_component and kept.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(kept.astype(np.uint8), 8)
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept = labels == largest_label
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / f"{prefix}_rgb.png"), rgb)
    cv2.imwrite(str(args.output_dir / f"{prefix}_processed_near.png"), kept.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output_dir / f"{prefix}_removed_dark_rgb.png"), ((candidate > 0) & dark).astype(np.uint8) * 255)
    np.save(args.output_dir / f"{prefix}_depth.npy", depth)
    print(
        f"input={int((candidate > 0).sum())} "
        f"removed_rgb_lt_{args.max_rgb_threshold}={int(((candidate > 0) & dark).sum())} "
        f"after_rgb={before_lcc} kept={int(kept.sum())} "
        f"largest_component={args.keep_largest_component}"
    )


if __name__ == "__main__":
    main()
