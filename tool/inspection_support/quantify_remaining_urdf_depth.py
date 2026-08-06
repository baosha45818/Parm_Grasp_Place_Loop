#!/usr/bin/env python3
"""Quantify measured-vs-URDF depth in a remaining PC1 candidate region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.inspection_support.replay_pc1_sam2_simple import _rendered_depth


def _stats(values: np.ndarray) -> dict[str, float | int]:
    if not values.size:
        return {"pixels": 0}
    return {
        "pixels": int(values.size),
        "min_mm": round(float(values.min()) * 1000, 1),
        "median_mm": round(float(np.median(values)) * 1000, 1),
        "p90_mm": round(float(np.percentile(values, 90)) * 1000, 1),
        "max_mm": round(float(values.max()) * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rgb = cv2.imread(str(args.source_run / "pc1_rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(args.source_run / "pc1_depth.npy", allow_pickle=False)
    candidate = cv2.imread(str(args.candidate_run / "pc1_processed_near.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or candidate is None or rgb.shape[:2] != depth.shape or candidate.shape != depth.shape:
        raise RuntimeError("Inconsistent RGB, depth, or candidate mask")
    result = json.loads((args.source_run / "result.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "modules/module_2_left_inspect_apart/inspect_config.json").read_text())
    rendered = _rendered_depth(result, cfg, depth)
    measured = depth.astype(np.float32) * 0.001
    valid = (candidate > 0) & (measured > 0)
    x0, y0, x1, y1 = args.roi
    roi = np.zeros(depth.shape, dtype=bool)
    roi[max(0, y0):min(depth.shape[0], y1), max(0, x0):min(depth.shape[1], x1)] = True
    target = valid & roi
    projected = target & np.isfinite(rendered)
    unprojected = target & ~np.isfinite(rendered)
    error = np.abs(measured - rendered)
    predicted_minus_measured = rendered - measured
    max_rgb = np.max(rgb, axis=2)
    black = max_rgb <= 20
    black_tolerance = 0.250 + np.clip((rendered - 0.200) / 0.050, 0.0, 1.0) * 0.050
    report = {
        "roi_xyxy": args.roi,
        "remaining_candidate_pixels": int(target.sum()),
        "urdf_projected_remaining_pixels": int(projected.sum()),
        "urdf_unprojected_remaining_pixels": int(unprojected.sum()),
        "projected_depth_error": _stats(error[projected]),
        "predicted_minus_measured": _stats(predicted_minus_measured[projected]),
        "predicted_at_least_125mm_farther_pixels": int(
            (projected & (predicted_minus_measured >= 0.125)).sum()
        ),
        "rgb_channel_max": {
            "min": int(max_rgb[target].min()) if target.any() else None,
            "median": float(np.median(max_rgb[target])) if target.any() else None,
            "p90": float(np.percentile(max_rgb[target], 90)) if target.any() else None,
            "max": int(max_rgb[target].max()) if target.any() else None,
        },
        "near_black_max_rgb_le_20_pixels": int((target & black).sum()),
        "would_match_35mm_pixels": int((projected & (error <= 0.035)).sum()),
        "would_match_black_dynamic_tolerance_pixels": int(
            (projected & black & (error <= black_tolerance)).sum()
        ),
        "measured_depth": _stats(measured[target]),
        "predicted_depth": _stats(rendered[projected]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
