#!/usr/bin/env python3
"""采集头部 D435i RGB-D，并立即运行 workpiece_2 完整反抓视觉预测。

该入口只打开相机；不会移动云台、机械臂或夹爪。调用前必须由上游将云台和
工件置于反抓观察位。采集、SAM2、ICP 和所有日志只写入本目录。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAPTURE_IMPL = ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/implementation"
FULL_PIPELINE = HERE / "run_full_visual_prediction.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="261722072445", help="头部 D435i 序列号")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--depth-fuse-frames", type=int, default=15)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FULL_PIPELINE.is_file():
        raise RuntimeError(f"完整视觉入口不存在: {FULL_PIPELINE}")
    if str(CAPTURE_IMPL) not in sys.path:
        sys.path.insert(0, str(CAPTURE_IMPL))
    from capture_head_rgbd import capture

    capture_root = HERE / "captures"
    capture_root.mkdir(parents=True, exist_ok=True)
    capture_dir = capture(SimpleNamespace(
        serial=args.serial,
        output_dir=capture_root,
        frames=1,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        timeout_ms=args.timeout_ms,
        depth_fuse_frames=args.depth_fuse_frames,
        preview_min_m=.20,
        preview_max_m=2.50,
    ))
    subprocess.run(
        [str(Path(sys.executable)), str(FULL_PIPELINE), "--capture-dir", str(capture_dir)],
        check=True,
    )
    print(f"REVERSE_GRASP_CAPTURE={capture_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
