#!/usr/bin/env python3
"""Reprocess a saved workpiece-2 capture with the correct model frame."""

from pathlib import Path

from ._delegate import delegate


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    delegate(
        "tests.reversed_head_observation.reprocess_head_d435_result",
        ("--config", str(HERE / "head_d435_pose_config.json")),
    )

