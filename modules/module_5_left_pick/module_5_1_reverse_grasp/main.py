#!/usr/bin/env python3
"""Stable Module 5 entry point for tabletop reverse-side left pick."""
from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path


if __name__ == "__main__":
    if os.environ.get("PARM_MODULAR_REVERSE_PICK") == "1":
        modular_entry = Path(__file__).resolve().parents[1] / "run_head_judge_then_pick.py"
        print(f"[module-5] 模块化反抓链入口: {modular_entry}", flush=True)
        runpy.run_path(str(modular_entry), run_name="__main__")
        raise SystemExit(0)
    implementation = Path(__file__).with_name("implementation")
    sys.path.insert(0, str(implementation))
    runpy.run_path(str(implementation / "main.py"), run_name="__main__")
