#!/usr/bin/env python3
"""Move the head, capture RGB, segment workpieces, and write pose JSON."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time


MODULE_ROOT = Path(__file__).resolve().parent
CAPTURE_SCRIPT = MODULE_ROOT / "camera/capture_head_rgb.py"
SEGMENT_SCRIPT = MODULE_ROOT / "seg/segment_image.py"
POSE_SCRIPT = MODULE_ROOT / "seg/mask_workpiece_pose.py"
OUTPUT_DIR = MODULE_ROOT.parents[2] / "output/4_2"
SEGMENT_NO_WORKPIECE_EXIT_CODE = 10
NO_WORKPIECE_EXIT_CODE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-host", default=os.environ.get("PARM_HEAD_HOST", "localhost"))
    parser.add_argument("--head-pitch", type=float, default=38.5)
    parser.add_argument("--head-yaw", type=float, default=0.0)
    parser.add_argument("--head-settle-s", type=float, default=1.0)
    parser.add_argument("--serial", default="261222076212", help="head D435i serial number")
    parser.add_argument("--roi", nargs=4, type=float, default=(0.08, 0.20, 0.98, 0.90))
    parser.add_argument("--outlier-mad-scale", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prefix", default="head_pose")
    parser.add_argument("--skip-head-move", action="store_true", help="do not issue a head motion command")
    return parser.parse_args()


def import_sdk():
    try:
        return importlib.import_module("parm_sdk")
    except ImportError:
        wheel = Path("/home/dainanjun/Public/parm_sdk_1.0.0/parm_sdk-1.0.0+20260717.191601-py3-none-any.whl")
        if wheel.is_file():
            sys.path.insert(0, str(wheel))
            return importlib.import_module("parm_sdk")
    raise RuntimeError("无法导入 parm_sdk；请安装 SDK 或设置 PYTHONPATH。")


def set_head(args: argparse.Namespace) -> None:
    if args.skip_head_move:
        print("[head] skipped by --skip-head-move")
        return
    sdk = import_sdk()
    head = sdk.Head(host=args.head_host)
    try:
        head.move_to(pitch=args.head_pitch, yaw=args.head_yaw)
        if args.head_settle_s > 0:
            time.sleep(args.head_settle_s)
    finally:
        head.close()


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=MODULE_ROOT, check=True)


def main() -> int:
    args = parse_args()
    if args.head_settle_s < 0:
        raise ValueError("--head-settle-s must be non-negative")
    for path in (CAPTURE_SCRIPT, SEGMENT_SCRIPT, POSE_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)

    set_head(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = set(args.output_dir.glob(f"{args.prefix}_*.png"))
    run([
        sys.executable, str(CAPTURE_SCRIPT), "--serial", args.serial,
        "--output-dir", str(args.output_dir), "--prefix", args.prefix,
    ])
    created = sorted(set(args.output_dir.glob(f"{args.prefix}_*.png")) - before)
    if len(created) != 1:
        raise RuntimeError(f"expected exactly one captured RGB file, found: {created}")
    image = created[0]
    mask = image.with_name(f"{image.stem}_mask.png")
    pose = image.with_name(f"{image.stem}_mask_pose.json")
    debug = image.with_name(f"{image.stem}_mask_pose.png")
    try:
        run([
            sys.executable, str(SEGMENT_SCRIPT), str(image), str(mask), "--roi",
            *map(str, args.roi), "--outlier-mad-scale", str(args.outlier_mad_scale),
        ])
    except subprocess.CalledProcessError as exc:
        if exc.returncode == SEGMENT_NO_WORKPIECE_EXIT_CODE:
            print("[stop] 桌面没有工件，右臂无法抓取。")
            return NO_WORKPIECE_EXIT_CODE
        raise
    run([sys.executable, str(POSE_SCRIPT), str(mask), "--output", str(pose), "--debug-image", str(debug)])
    print(f"[done] image={image}\n[done] mask={mask}\n[done] pose_json={pose}\n[done] debug_image={debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
