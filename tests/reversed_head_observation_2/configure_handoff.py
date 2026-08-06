#!/usr/bin/env python3
"""Generate workpiece-2 attached-collision calibration files."""

from pathlib import Path
import sys

from ._delegate import delegate


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments:
        delegate("tests.reversed_head_observation.configure_handoff")
    command, *rest = arguments
    if command == "set-target":
        sys.argv[1:] = [
            command,
            "--input-config", str(HERE / "handoff_config.json"),
            "--output-config", str(HERE / "handoff_config.ready.json"),
            *rest,
        ]
    else:
        sys.argv[1:] = arguments
    delegate("tests.reversed_head_observation.configure_handoff")

