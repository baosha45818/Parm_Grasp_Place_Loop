#!/usr/bin/env python3
"""Render a diagnostic overlay separating retained object and URDF-arm depth."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--view", default="pc1")
    parser.add_argument("--pre-sam2", action="store_true",
                        help="use the pre-SAM2 depth-band candidate instead of the final SAM2 mask")
    parser.add_argument("--near-min-m", type=float,
                        help="rebuild the pre-SAM2 candidate with this lower depth bound")
    parser.add_argument("--near-max-m", type=float,
                        help="rebuild the pre-SAM2 candidate with this upper depth bound")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="metres per unit of the saved uint16 depth image")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def depth_colour(depth_m: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = depth_m[selected]
    low, high = np.percentile(values, (2, 98))
    high = max(float(high), float(low) + 1e-4)
    encoded = np.zeros(depth_m.shape, dtype=np.uint8)
    encoded[selected] = np.clip((depth_m[selected] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO), float(low), float(high)


def label(image: np.ndarray, text: str, xy: tuple[int, int]) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    prefix = args.run_dir / args.view
    rgb = cv2.imread(str(prefix.with_name(prefix.name + "_rgb.png")), cv2.IMREAD_COLOR)
    if (args.near_min_m is None) != (args.near_max_m is None):
        raise ValueError("--near-min-m and --near-max-m must be supplied together")
    if (args.near_min_m is not None and not args.pre_sam2):
        raise ValueError("depth-band rebuilding is only valid with --pre-sam2")
    mask_suffix = "_processed_near.png" if args.pre_sam2 else "_mask.png"
    object_mask = cv2.imread(str(prefix.with_name(prefix.name + mask_suffix)), cv2.IMREAD_GRAYSCALE)
    arm_mask = cv2.imread(str(prefix.with_name(prefix.name + "_urdf_self_removed.png")), cv2.IMREAD_GRAYSCALE)
    raw_depth = np.load(prefix.with_name(prefix.name + "_depth.npy"), allow_pickle=False)
    if rgb is None or object_mask is None or arm_mask is None:
        raise RuntimeError("Missing RGB, final object mask, or URDF self-pixel mask")
    if raw_depth.shape != rgb.shape[:2]:
        raise RuntimeError("Depth and RGB dimensions differ")

    depth_m = raw_depth.astype(np.float32) * args.depth_scale
    valid = raw_depth > 0
    arm = (arm_mask > 0) & valid
    # `processed_near` is the pre-SAM2 depth-band candidate (largest connected
    # component after removing depth-confirmed URDF pixels).  `mask` is the
    # final SAM2 result.  In both cases remove only depth-confirmed URDF
    # pixels, rather than every 2-D URDF projection pixel.
    if args.near_min_m is None:
        candidate = object_mask > 0
    else:
        near = (depth_m >= args.near_min_m) & (depth_m <= args.near_max_m) & ~arm
        count, labels, stats, _ = cv2.connectedComponentsWithStats(near.astype(np.uint8), 8)
        candidate = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))) if count > 1 else near
    object_only = candidate & valid & ~arm

    object_colour, object_low, object_high = depth_colour(depth_m, object_only)
    arm_colour, arm_low, arm_high = depth_colour(depth_m, arm)

    object_panel = np.zeros_like(rgb)
    object_panel[object_only] = object_colour[object_only]
    arm_panel = np.zeros_like(rgb)
    arm_panel[arm] = arm_colour[arm]
    combined = rgb.copy()
    combined[object_only] = (0.40 * combined[object_only] + 0.60 * object_colour[object_only]).astype(np.uint8)
    combined[arm] = (0.40 * combined[arm] + 0.60 * arm_colour[arm]).astype(np.uint8)

    depth_band = (f" [{args.near_min_m:.3f}, {args.near_max_m:.3f}] m"
                  if args.near_min_m is not None else "")
    candidate_name = ("Pre-SAM2 non-URDF depth candidate" + depth_band) if args.pre_sam2 else "Retained workpiece depth"
    label(object_panel, f"{candidate_name}: {object_only.sum()} px", (20, 38))
    label(object_panel, f"Turbo: {object_low:.3f}--{object_high:.3f} m", (20, 70))
    label(arm_panel, f"URDF-confirmed arm/tool depth: {arm.sum()} px", (20, 38))
    label(arm_panel, f"Turbo: {arm_low:.3f}--{arm_high:.3f} m", (20, 70))
    combined_name = "Pre-SAM2 depth candidate + URDF-confirmed arm/tool" if args.pre_sam2 else "Depth overlay: workpiece + URDF-confirmed arm/tool"
    label(combined, combined_name, (20, 38))
    label(combined, "Each class uses its own Turbo depth scale; see lower panels", (20, 70))
    label(rgb, "Captured RGB", (20, 38))

    canvas = np.vstack((np.hstack((rgb, combined)), np.hstack((object_panel, arm_panel))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"wrote {args.output}")
    print(f"object_pixels={object_only.sum()} arm_pixels={arm.sum()} overlap_removed={(object_mask > 0).astype(bool)[arm].sum()}")


if __name__ == "__main__":
    main()
