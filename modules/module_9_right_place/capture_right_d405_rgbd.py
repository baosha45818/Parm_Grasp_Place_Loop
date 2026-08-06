#!/usr/bin/env python3
"""Capture one RGB-D frame from the right-wrist RealSense D405."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_RIGHT_D405_SERIAL = "230422273719"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取右腕 D405 的一帧 RGB 和对齐深度图")
    parser.add_argument("--serial", default=DEFAULT_RIGHT_D405_SERIAL, help="右腕 D405 序列号")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # On the right-wrist D405, the only common 1280x720 RGB + depth profile
    # runs at 5 FPS. Keep these as CLI options for lower-resolution profiles.
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=15)
    args = parser.parse_args()

    if min(args.width, args.height, args.fps) <= 0 or args.warmup_frames < 0:
        parser.error("宽、高、fps 必须为正数，warmup 帧数不能为负")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "缺少图像依赖。请用含 numpy、opencv、pyrealsense2 的采集环境运行，例如：\n"
            "  /home/dainanjun/Public/Parm_Grasp_Place_Loop/.venv/bin/python "
            "modules/module_9_right_place/capture_right_d405_rgbd.py"
        ) from exc

    from tool.inspection_support.robot_camera import D405Camera

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = output_dir / f"right_d405_{stamp}"
    with D405Camera(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        enable_postprocess=False,
        camera_name="right_d405",
    ) as camera:
        frame = camera.read()

    rgb_path = prefix.with_name(f"{prefix.name}_rgb.png")
    depth_path = prefix.with_name(f"{prefix.name}_depth_aligned.png")
    depth_visual_path = prefix.with_name(f"{prefix.name}_depth_visual.png")
    metadata_path = prefix.with_name(f"{prefix.name}_metadata.json")
    _write_image(cv2, rgb_path, frame.color)
    _write_image(cv2, depth_path, frame.depth)
    depth_visual, valid_depth = _make_depth_visual(cv2, np, frame.depth, frame.depth_scale)
    _write_image(cv2, depth_visual_path, depth_visual)

    metadata = {
        "camera_name": frame.camera_name,
        "serial": frame.serial,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "color_shape": list(frame.color.shape),
        "depth_shape": list(frame.depth.shape),
        "depth_dtype": str(frame.depth.dtype),
        "depth_scale_m": frame.depth_scale,
        "color_intrinsics": frame.color_intrinsics.tolist(),
        "depth_is_aligned_to_color": True,
        "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
        "valid_depth_min_m": (
            float(frame.depth[valid_depth].min() * frame.depth_scale)
            if np.any(valid_depth)
            else None
        ),
        "valid_depth_max_m": (
            float(frame.depth[valid_depth].max() * frame.depth_scale)
            if np.any(valid_depth)
            else None
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[right-d405] capture complete")
    print(f"  rgb: {rgb_path}")
    print(f"  depth: {depth_path}")
    print(f"  depth visual: {depth_visual_path}")
    print(f"  metadata: {metadata_path}")
    return 0


def _write_image(cv2, path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法写入图像：{path}")


def _make_depth_visual(cv2, np, depth, depth_scale: float):
    """Make a display-only depth map; the raw uint16 depth is saved separately."""
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8), valid
    depth_m = depth.astype(np.float32) * depth_scale
    clipped = np.clip(depth_m, 0.15, 1.0)
    normalized = ((clipped - 0.15) / 0.85 * 255.0).astype(np.uint8)
    normalized[~valid] = 0
    visual = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    visual[~valid] = 0
    return visual, valid


if __name__ == "__main__":
    raise SystemExit(main())
