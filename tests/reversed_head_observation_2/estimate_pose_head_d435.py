#!/usr/bin/env python3
"""Estimate workpiece-2 pose using its STL and documented object frame."""

from pathlib import Path

from ._delegate import delegate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


if __name__ == "__main__":
    delegate(
        "tests.reversed_head_observation.estimate_pose_head_d435",
        (
            "--config", str(HERE / "head_d435_pose_config.json"),
            "--auto-mask-config", str(HERE / "head_d435_auto_mask_config.json"),
            "--output-dir", str(ROOT / "output/reversed_head_pose_d435_workpiece_2"),
            "--mask-mode", "interactive",
            "--no-auto-gripper-prompt",
        ),
    )
