#!/usr/bin/env python3
"""Compatibility launcher for module 7's hole-A commissioning script.

The implementation now belongs with module 7.  Keep this tiny launcher so
existing operator commands using ``tool/run_hole_a_insertion.py`` continue to
work while they are migrated.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_9_right_place.run_hole_a_insertion import main


if __name__ == "__main__":
    raise SystemExit(main())
