#!/usr/bin/env python3
"""15 mm/s trial profile for the right-arm Hole A wiggle insertion.

This launcher deliberately reuses the current, validated insertion logic from
``place_hole_a_wiggle.py``.  It sets the insertion legs to 15 mm/s with
90 mm/s² acceleration.  The endpoint error requirement remains 0.5 mm, but
the short confirmation window is 0.45 s to avoid unnecessary fallback to the
full endpoint confirmation.  A dedicated 0.06 planner profile is used for
the collision-checked return to the taught joint start.  Explicit command-line
values still win.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import runpy
import sys


SOURCE = Path(__file__).with_name("place_hole_a_wiggle.py")
FAST_RETURN_CONFIG = SOURCE.parents[2] / "config.hole_a_first_live_fast_return.json"
PROFILE_DEFAULTS = [
    "--with-wiggle",
    "--planner-config", str(FAST_RETURN_CONFIG),
    "--fast-linear-speed-mm-s", "15.0",
    "--fast-linear-accel-mm-s2", "90.0",
    "--slow-linear-speed-mm-s", "15.0",
    "--slow-linear-accel-mm-s2", "90.0",
    "--quick-endpoint-wait-s", "0.45",
]


def has_output_argument(arguments: list[str]) -> bool:
    return any(argument == "--output" or argument.startswith("--output=") for argument in arguments)


def main() -> None:
    supplied_args = sys.argv[1:]
    # Put profile defaults before user arguments: argparse uses the last
    # occurrence, so an operator can still override either value explicitly.
    default_output: list[str] = []
    if not has_output_argument(supplied_args):
        default_output = [
            "--output",
            str(Path("logs") / f"hole_a_wiggle_new_{datetime.now():%Y%m%d_%H%M%S}.log"),
        ]
    sys.argv = [str(SOURCE), *PROFILE_DEFAULTS, *default_output, *supplied_args]
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
