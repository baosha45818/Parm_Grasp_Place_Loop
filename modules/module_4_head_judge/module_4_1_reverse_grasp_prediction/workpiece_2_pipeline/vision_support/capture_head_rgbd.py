#!/usr/bin/env python3
"""Capture RGB-D frames from the PARM head RealSense D435i.

The script only opens the camera; it sends no commands to the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("OpenCV is required: install opencv-python in the active environment.") from exc

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("pyrealsense2 is required in the active Python environment.") from exc


DEFAULT_HEAD_SERIAL = "261222076212"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, object]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def make_depth_preview(depth: np.ndarray, depth_scale_m: float, minimum_m: float, maximum_m: float) -> np.ndarray:
    """Return an 8-bit BGR image for convenient viewing of a raw Z16 depth frame."""
    if maximum_m <= minimum_m:
        raise ValueError("--preview-max-m must be greater than --preview-min-m")
    depth_m = depth.astype(np.float32) * depth_scale_m
    valid = (depth_m >= minimum_m) & (depth_m <= maximum_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth_m[valid] - minimum_m) / (maximum_m - minimum_m) * 255.0, 0, 255
    ).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def capture_fused_rgbd(
    pipeline: rs.pipeline,
    align: rs.align,
    fuse_frames: int,
    timeout_ms: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Median-fuse static-scene depth frames while retaining the newest RGB frame."""
    depth_samples: list[np.ndarray] = []
    color: np.ndarray | None = None
    timestamp_ms = 0.0
    for _ in range(fuse_frames):
        frames = align.process(pipeline.wait_for_frames(timeout_ms=timeout_ms))
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Camera returned an incomplete RGB-D frame")
        color = np.asanyarray(color_frame.get_data()).copy()
        timestamp_ms = color_frame.get_timestamp()
        sample = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        # 0 is invalid. 65535 can be emitted by alignment for out-of-FOV pixels.
        sample[(sample == 0) | (sample == np.iinfo(np.uint16).max)] = np.nan
        depth_samples.append(sample)

    fused = np.nanmedian(np.stack(depth_samples), axis=0)
    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    return color, np.rint(fused).astype(np.uint16), timestamp_ms


def capture(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir / datetime.now().strftime("head_rgbd_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    context = rs.context()
    connected = {
        device.get_info(rs.camera_info.serial_number): device
        for device in context.devices
    }
    if args.serial not in connected:
        raise RuntimeError(
            f"Head D435i {args.serial} is not connected. "
            f"Connected serials: {sorted(connected)}"
        )

    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline = rs.pipeline(context)
    align = rs.align(rs.stream.color)

    try:
        profile = pipeline.start(config)
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=args.timeout_ms)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale_m = float(depth_sensor.get_depth_scale())
        saved_frames: list[dict[str, object]] = []

        for frame_index in range(args.frames):
            color, depth, timestamp_ms = capture_fused_rgbd(
                pipeline, align, args.depth_fuse_frames, args.timeout_ms
            )
            color_path = output_dir / f"rgb_{frame_index:03d}.png"
            depth_path = output_dir / f"depth_aligned_{frame_index:03d}.png"
            preview_path = output_dir / f"depth_preview_{frame_index:03d}.png"
            if not cv2.imwrite(str(color_path), color):
                raise RuntimeError(f"Failed to write {color_path}")
            if not cv2.imwrite(str(depth_path), depth):
                raise RuntimeError(f"Failed to write {depth_path}")
            if not cv2.imwrite(
                str(preview_path),
                make_depth_preview(depth, depth_scale_m, args.preview_min_m, args.preview_max_m),
            ):
                raise RuntimeError(f"Failed to write {preview_path}")
            saved_frames.append(
                {
                    "index": frame_index,
                    "rgb": color_path.name,
                    "depth_aligned": depth_path.name,
                    "depth_preview": preview_path.name,
                    "device_timestamp_ms": timestamp_ms,
                }
            )
            print(f"Saved {color_path.name}, {depth_path.name}, and {preview_path.name}")

        metadata = {
            "camera_role": "head",
            "camera_model": "Intel RealSense D435i",
            "serial": args.serial,
            "captured_at": datetime.now().astimezone().isoformat(),
            "depth_alignment": "depth aligned to color",
            "depth_format": "uint16 PNG; multiply values by depth_scale_m to obtain metres",
            "depth_fusion": f"median of {args.depth_fuse_frames} aligned depth frames",
            "depth_preview": (
                f"8-bit false-colour PNG, clipped to {args.preview_min_m:g}-{args.preview_max_m:g} m; "
                "black pixels are invalid or outside this range"
            ),
            "depth_scale_m": depth_scale_m,
            "color_intrinsics": intrinsics_to_dict(color_profile.get_intrinsics()),
            "frames": saved_frames,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB + aligned depth from the PARM head D435i.")
    parser.add_argument("--serial", default=DEFAULT_HEAD_SERIAL, help="RealSense serial number")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=1, help="Number of RGB-D pairs to save")
    parser.add_argument("--width", type=int, default=848, help="RGB and depth width (D435i shared RGB-D mode: 848)")
    parser.add_argument("--height", type=int, default=480, help="RGB and depth height (D435i shared RGB-D mode: 480)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--depth-fuse-frames", type=int, default=15, help="Aligned depth frames median-fused per saved image")
    parser.add_argument("--preview-min-m", type=float, default=0.20, help="Nearest depth rendered in the preview")
    parser.add_argument("--preview-max-m", type=float, default=2.50, help="Farthest depth rendered in the preview")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.depth_fuse_frames < 1:
        parser.error("--depth-fuse-frames must be at least 1")
    return args


if __name__ == "__main__":
    try:
        destination = capture(parse_args())
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Capture complete: {destination}")
