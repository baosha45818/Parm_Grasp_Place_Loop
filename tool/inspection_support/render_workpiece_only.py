#!/usr/bin/env python3
"""Render final SAM2 workpiece masks and depth without robot overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--view", default="pc1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Write only the RGB image with the final mask overlaid.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="On an overlay-only output, draw SAM2 ROI plus positive/negative prompts from replay_report.json.",
    )
    parser.add_argument(
        "--raw-mask", action="store_true",
        help="Overlay the direct SAM2 response (*_sam2_raw_mask.png) instead of the final post-processed mask.",
    )
    return parser.parse_args()


def text(image: np.ndarray, value: str, y: int) -> None:
    cv2.putText(image, value, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, value, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    cfg = args()
    prefix = cfg.run_dir / cfg.view
    rgb = cv2.imread(str(prefix.with_name(prefix.name + "_rgb.png")), cv2.IMREAD_COLOR)
    mask_name = "_sam2_raw_mask.png" if cfg.raw_mask else "_mask.png"
    mask = cv2.imread(str(prefix.with_name(prefix.name + mask_name)), cv2.IMREAD_GRAYSCALE)
    depth = np.load(prefix.with_name(prefix.name + "_depth.npy"), allow_pickle=False)
    if rgb is None or mask is None or depth.shape != rgb.shape[:2]:
        raise RuntimeError("Missing or inconsistent RGB, final mask, or depth")

    mask_pixels = mask > 0
    selected = mask_pixels & (depth > 0)
    if not mask_pixels.any() and not cfg.overlay_only:
        raise RuntimeError("Final mask is empty")
    if not selected.any() and not cfg.overlay_only:
        raise RuntimeError("Final mask has no valid depth pixels")
    depth_m = depth.astype(np.float32) * cfg.depth_scale
    if selected.any():
        low, high = np.percentile(depth_m[selected], (2, 98))
        high = max(float(high), float(low) + 1e-4)
    else:
        low, high = 0.0, 1.0
    encoded = np.zeros(depth.shape, np.uint8)
    encoded[selected] = np.clip((depth_m[selected] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    depth_panel = np.zeros_like(rgb)
    colour = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
    depth_panel[selected] = colour[selected]
    mask_panel = np.zeros_like(rgb)
    mask_panel[selected] = (255, 255, 255)
    overlay = rgb.copy()
    # Prompt overlays are 2-D SAM2 diagnostics.  Show the complete SAM2 mask
    # in red, including pixels whose depth camera measurement is invalid.
    overlay[mask_pixels] = (
        0.35 * overlay[mask_pixels] + 0.65 * np.array([0, 0, 255])
    ).astype(np.uint8)

    if cfg.show_prompts:
        report_path = cfg.run_dir / "replay_report.json"
        if not report_path.exists():
            raise RuntimeError("--show-prompts requires replay_report.json")
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        box = report.get("roi_box_xyxy")
        if box and len(box) == 4:
            x0, y0, x1, y1 = (int(value) for value in box)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
        for x, y in report.get("positive_points", []):
            cv2.circle(overlay, (int(x), int(y)), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 10, (0, 0, 0), 2, cv2.LINE_AA)
        for x, y in report.get("negative_points", []):
            cv2.circle(overlay, (int(x), int(y)), 8, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 10, (0, 0, 0), 2, cv2.LINE_AA)
        text(overlay, "mask=red  ROI=yellow  positive=green  negative=blue", 38)

    if cfg.overlay_only:
        output = overlay
    else:
        text(rgb, "Captured RGB", 38)
        text(overlay, "Final SAM2 workpiece mask only", 38)
        text(mask_panel, f"Workpiece mask: {selected.sum()} valid-depth px", 38)
        text(depth_panel, "Workpiece depth only (Turbo)", 38)
        text(depth_panel, f"{low:.3f}--{high:.3f} m", 70)
        output = np.vstack((np.hstack((rgb, overlay)), np.hstack((mask_panel, depth_panel))))
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(cfg.output), output):
        raise RuntimeError(f"Could not write {cfg.output}")
    print(f"wrote {cfg.output}; workpiece_pixels={selected.sum()}")


if __name__ == "__main__":
    main()
