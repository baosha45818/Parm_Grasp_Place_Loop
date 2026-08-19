#!/usr/bin/env python3
"""Module 10 实机兼容入口：执行放后松爪、抬升并回右臂 Idle。

当前复用旧总控已经验证的实现；入口稳定在 modules 内，后续可以在不改变
模块化总控的情况下替换内部实现。
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "modules/module_10_right_idle/right_place_and_idle.py"


if __name__ == "__main__":
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
