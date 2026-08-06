#!/usr/bin/env python3
"""Move left-held workpiece 2 to the bounded right-handoff region."""

from pathlib import Path

from ._delegate import delegate


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    delegate(
        "tests.reversed_head_observation.move_object_center",
        (
            "--config", str(HERE / "handoff_config.json"),
            "--cartesian-planner", "rrtstar",
            "--fixed-handoff-orientation",
        ),
    )
