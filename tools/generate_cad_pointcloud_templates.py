#!/usr/bin/env python3
"""Generate deterministic persistent CAD point-cloud templates for Module 4."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.inspection_support.stl_model import (  # noqa: E402
    load_stl_triangles_m,
    sample_surface_points,
)


ICP_POINT_COUNT = 14_000
RENDER_POINT_COUNT = 90_000
ICP_SEED = 3
RENDER_SEED = 4
WORKPIECES = {
    "1": PROJECT_ROOT / "tools/workpiece_1/welding_up_side_1118_only_2_small_screws_removed.stl",
    "2": PROJECT_ROOT / "tools/workpiece_2/workpiece_2.stl",
}


def generate(workpiece_id: str) -> Path:
    stl_path = WORKPIECES[workpiece_id]
    triangles = load_stl_triangles_m(stl_path)
    icp_points = sample_surface_points(triangles, ICP_POINT_COUNT, ICP_SEED)
    render_points = sample_surface_points(triangles, RENDER_POINT_COUNT, RENDER_SEED)
    output = stl_path.parent / "cad_pointcloud_template.npz"
    pending = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        pending,
        schema_version=np.int64(1),
        workpiece_id=np.asarray(workpiece_id),
        source_stl=np.asarray(str(stl_path.relative_to(PROJECT_ROOT))),
        source_stl_sha256=np.asarray(hashlib.sha256(stl_path.read_bytes()).hexdigest()),
        units=np.asarray("m"),
        icp_seed=np.int64(ICP_SEED),
        render_seed=np.int64(RENDER_SEED),
        icp_points_m=icp_points,
        render_points_m=render_points,
    )
    pending.replace(output)
    print(
        f"workpiece={workpiece_id} output={output} "
        f"icp_points={len(icp_points)} render_points={len(render_points)}"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()
    selected = WORKPIECES if args.workpiece == "all" else (args.workpiece,)
    for workpiece_id in selected:
        generate(workpiece_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
