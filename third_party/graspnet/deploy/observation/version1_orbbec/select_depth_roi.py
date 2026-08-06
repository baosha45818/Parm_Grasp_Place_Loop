#!/usr/bin/env python3
"""Select a depth-pixel ROI for one saved Orbbec observation; no robot control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args()
    meta_path = args.capture_dir / "meta.json"
    depth = np.load(args.capture_dir / "depth.npy")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scale = float(meta["depth_scale_m"])
    depth_m = depth.astype(np.float32) * scale
    valid = depth_m > 0
    if depth.ndim != 2 or not np.any(valid):
        raise RuntimeError("depth.npy must be a valid 2D depth image")
    low, high = np.percentile(depth_m[valid], [2, 98])
    image = cv2.applyColorMap((np.clip((depth_m - low) / max(high - low, 1e-3), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    x, y, w, h = (int(v) for v in cv2.selectROI("Orbbec depth ROI", image, showCrosshair=True, fromCenter=False))
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled")
    meta["roi_xywh"] = [x, y, w, h]
    meta["roi_coordinate_frame"] = "depth_pixels"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("saved", meta["roi_xywh"])


if __name__ == "__main__":
    main()
