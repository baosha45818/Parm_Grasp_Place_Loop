#!/usr/bin/env python3
"""Fill every enclosed hole in saved PC1/PC2 final masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_2_left_inspect_apart import standalone_inspect as inspect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    changed = 0
    for run in sorted(path for path in args.output_dir.iterdir() if path.is_dir()):
        for view in ("pc1", "pc2"):
            mask_path = run / f"{view}_mask.png"
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read {mask_path}")
            filled, report = inspect._fill_small_enclosed_holes(mask)
            if not cv2.imwrite(str(mask_path), filled):
                raise RuntimeError(f"Could not write {mask_path}")
            report_path = run / "replay_report.json"
            if report_path.exists():
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                payload["final_internal_hole_fill"] = report
                report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            changed += 1
    print(f"filled_final_masks={changed}")


if __name__ == "__main__":
    main()
