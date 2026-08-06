#!/usr/bin/env python3
"""7 mm/s continuous straight-line Hole A insertion-search trial.

The established force and torque checks are reused, but lateral Y+/Y− wiggles
and per-millimetre endpoint waits are disabled.  World-X commands are streamed
continuously and force is sampled at each 1-mm checkpoint.  This experimental
mode has no lateral alignment correction or lateral insertion signature.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import runpy
import sys


SOURCE = Path(__file__).with_name("place_hole_a_wiggle.py")
FAST_RETURN_CONFIG = SOURCE.parents[2] / "config.hole_a_first_live_fast_return.json"
PROFILE_DEFAULTS = [
    "--straight-line",
    "--continuous-straight-search",
    "--continuous-force-monitor-hz", "50.0",
    "--continuous-force-output-hz", "10.0",
    "--planner-config", str(FAST_RETURN_CONFIG),
    "--fast-linear-speed-mm-s", "7.0",
    "--fast-linear-accel-mm-s2", "20.0",
    "--slow-linear-speed-mm-s", "7.0",
    "--slow-linear-accel-mm-s2", "20.0",
    "--quick-endpoint-wait-s", "0.45",
]


def has_output_argument(arguments: list[str]) -> bool:
    return any(argument == "--output" or argument.startswith("--output=") for argument in arguments)


def main() -> None:
    supplied_args = sys.argv[1:]
    default_output: list[str] = []
    if not has_output_argument(supplied_args):
        default_output = [
            "--output",
            str(Path("logs") / f"hole_a_wiggle_new_siggle_{datetime.now():%Y%m%d_%H%M%S}.log"),
        ]
    sys.argv = [str(SOURCE), *PROFILE_DEFAULTS, *default_output, *supplied_args]
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
