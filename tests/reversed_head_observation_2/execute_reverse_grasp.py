#!/usr/bin/env python3
"""Run the tabletop reverse-grasp pipeline with the workpiece-2 CAD."""

from pathlib import Path
import sys

from ._delegate import delegate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIRMATION = "--confirm-workpiece-2-reverse-grasp-geometry"


if __name__ == "__main__":
    executing = "--execute" in sys.argv[1:]
    confirmed = CONFIRMATION in sys.argv[1:]
    if executing and not confirmed:
        raise SystemExit(
            "拒绝执行：请先用 --preflight-only 核验工件2 grasp_overlay.png/"
            "grasp_3d.png；确认后增加 --confirm-workpiece-2-reverse-grasp-geometry。"
        )
    if confirmed:
        sys.argv.remove(CONFIRMATION)
    delegate(
        "modules.module_5_left_pick.module_5_1_reverse_grasp.main",
        (
            "--stl", str(ROOT / "tool/workpiece_2/workpiece_2.stl"),
            "--orientation-config", str(HERE / "reverse_grasp_config.json"),
        ),
    )

