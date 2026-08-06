#!/usr/bin/env python3
"""Compare grayscale and HSV statistics for two PC1 candidate regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _stats(values: np.ndarray) -> dict[str, float | int]:
    if not values.size:
        return {"pixels": 0}
    return {
        "pixels": int(values.size),
        "min": int(values.min()),
        "p10": round(float(np.percentile(values, 10)), 1),
        "median": round(float(np.median(values)), 1),
        "p90": round(float(np.percentile(values, 90)), 1),
        "max": int(values.max()),
    }


def _mask(shape: tuple[int, int], roi: list[int], candidate: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = roi
    region = np.zeros(shape, dtype=bool)
    region[max(0, y0):min(shape[0], y1), max(0, x0):min(shape[1], x1)] = True
    return region & (candidate > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm-roi", type=int, nargs=4, required=True)
    parser.add_argument("--workpiece-roi", type=int, nargs=4, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    rgb = cv2.imread(str(args.run_dir / "pc1_rgb.png"), cv2.IMREAD_COLOR)
    candidate = cv2.imread(str(args.run_dir / "pc1_processed_near.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or candidate is None or rgb.shape[:2] != candidate.shape:
        raise RuntimeError("Missing or inconsistent RGB/candidate mask")
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    arm = _mask(gray.shape, args.arm_roi, candidate)
    workpiece = _mask(gray.shape, args.workpiece_roi, candidate)
    report = {
        "arm_roi_xyxy": args.arm_roi,
        "workpiece_roi_xyxy": args.workpiece_roi,
        "arm": {"gray": _stats(gray[arm]), "saturation": _stats(hsv[:, :, 1][arm]), "value": _stats(hsv[:, :, 2][arm])},
        "workpiece": {"gray": _stats(gray[workpiece]), "saturation": _stats(hsv[:, :, 1][workpiece]), "value": _stats(hsv[:, :, 2][workpiece])},
    }
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for roi, label, colour in ((args.arm_roi, "arm residual", (0, 0, 255)), (args.workpiece_roi, "workpiece", (0, 255, 0))):
        x0, y0, x1, y1 = roi
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(canvas, label, (x0, max(22, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), canvas)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
