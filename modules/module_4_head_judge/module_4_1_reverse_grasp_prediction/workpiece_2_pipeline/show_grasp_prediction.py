#!/usr/bin/env python3
"""Open an image in the desktop viewer; fall back to printing its path."""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--image", type=Path, required=True)
p.add_argument("--title", default="grasp prediction")
a = p.parse_args()
if not a.image.is_file():
    raise FileNotFoundError(a.image)
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    print(f"未检测到图形桌面，请查看图片：{a.image}")
    raise SystemExit(0)
try:
    subprocess.Popen(["xdg-open", str(a.image)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    input(f"已请求桌面查看器打开：{a.image}\n查看后按 Enter 继续。")
except (FileNotFoundError, OSError, EOFError):
    print(f"GUI 不可用，请查看图片：{a.image}")
