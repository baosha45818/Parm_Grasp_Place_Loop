#!/usr/bin/env python3
"""离线补齐历史 Inspect 运行中被 ICP 质量门限拒绝的最佳候选位姿图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_2_left_inspect_apart.standalone_inspect_global import (  # noqa: E402
    CONFIG_PATH,
    WORKPIECE_2_R_OBJECT_STL,
    WORKPIECE_2_STL_PATH,
    ICPRegistrationError,
    _icp_pose,
    _save_rejected_icp_visualizations,
    load_object_triangles,
    sample_surface_points,
)


DEFAULT_INTRINSICS = np.array([
    [751.8960571289062, 0.0, 632.2559204101562],
    [0.0, 751.3369750976562, 359.5553283691406],
    [0.0, 0.0, 1.0],
], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--workpiece", choices=("1", "2"), default="2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/module_2_left_inspect_apart",
    )
    return parser.parse_args()


def load_cfg(workpiece: str) -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if workpiece == "2":
        cfg.update({
            "stl_path": str(WORKPIECE_2_STL_PATH),
            "R_object_stl": WORKPIECE_2_R_OBJECT_STL,
            "object_origin_stl_m": [0.0, 0.0, 0.0],
        })
    return cfg


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit 必须大于 0")
    cfg = load_cfg(args.workpiece)
    triangles = load_object_triangles(cfg)
    cad = sample_surface_points(
        triangles, int(cfg["cad_sample_points"]), int(cfg["seed"])
    )
    candidates: list[tuple[float, Path, str]] = []
    for run_dir in args.output_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name[:8].isdigit():
            continue
        for view in ("pc1", "pc2"):
            if (
                (run_dir / f"{view}_points_camera.npy").is_file()
                and (run_dir / f"{view}_mask.png").is_file()
                and (run_dir / f"{view}_rgb.png").is_file()
                and not (run_dir / f"{view}_icp_best_rejected_pose_visualization.png").exists()
            ):
                candidates.append((
                    (run_dir / f"{view}_points_camera.npy").stat().st_mtime,
                    run_dir,
                    view,
                ))
    candidates.sort(reverse=True)

    completed: list[dict] = []
    for _, run_dir, view in candidates:
        if len(completed) >= args.limit:
            break
        points = np.load(run_dir / f"{view}_points_camera.npy")
        if len(points) < int(cfg["min_registration_points"]):
            continue
        mask = cv2.imread(str(run_dir / f"{view}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        try:
            _icp_pose(cad, points, cfg, mask, DEFAULT_INTRINSICS)
        except ICPRegistrationError as exc:
            artifacts = _save_rejected_icp_visualizations(
                run_dir, view, cad, points, cfg, DEFAULT_INTRINSICS, exc.metrics
            )
            completed.append({"run_dir": str(run_dir), "view": view, **artifacts})
            print(f"[补齐] {run_dir.name}/{view}: {artifacts['overlay']}", flush=True)

    summary = args.output_dir / "rejected_icp_visualization_backfill.json"
    summary.write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 已补齐 {len(completed)} 个被拒 ICP 位姿图；汇总: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
