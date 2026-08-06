#!/usr/bin/env python3
"""Capture RGB images from the PARM head-mounted RealSense camera.

The camera is identified by serial number instead of USB port, so reconnecting
it does not change which device the script uses.  The default serial comes
from the working PARM data collector configuration.

Examples:
    # Save one image to my_robot/camera/captures/head_rgb/
    uv run python my_robot/camera/capture_head_rgb.py

    # Continuously publish the latest RGB frame at 10 Hz. Stop with Ctrl-C.
    uv run python my_robot/camera/capture_head_rgb.py --preview

    # Capture 100 frames without opening a GUI window.
    uv run python my_robot/camera/capture_head_rgb.py --count 100 --interval 0.1
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# Head D435i serial from Public/data_collector/parm_workpiece_collector.py.
DEFAULT_HEAD_SERIAL = "261222076212"
DEFAULT_CAPTURE_ROOT = Path(__file__).resolve().parent / "captures"
DEFAULT_OUTPUT_DIR = DEFAULT_CAPTURE_ROOT / "head_rgb"
DEFAULT_PREVIEW_OUTPUT = DEFAULT_CAPTURE_ROOT / "head_rgb_latest.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB images from the PARM head RealSense camera")
    parser.add_argument("--serial", default=DEFAULT_HEAD_SERIAL, help=f"RealSense serial number (default: {DEFAULT_HEAD_SERIAL})")
    parser.add_argument("--width", type=int, default=1280, help="requested color width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="requested color height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="requested color FPS (default: 30)")
    parser.add_argument("--warmup-frames", type=int, default=15, help="frames to discard for auto-exposure (default: 15)")
    parser.add_argument("--count", type=int, default=1, help="number of frames to save; ignored in --preview mode (default: 1)")
    parser.add_argument("--interval", type=float, default=0.0, help="minimum seconds between saves in batch mode (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"directory for saved PNGs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--prefix", default="head_rgb", help="saved filename prefix (default: head_rgb)")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="continuously write the latest frame; press s to save a snapshot, Ctrl-C to stop",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=30.0,
        help="latest-frame write rate for --preview (default: 10)",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        help=f"latest-frame PNG path (default: {DEFAULT_PREVIEW_OUTPUT})",
    )
    return parser.parse_args()


def _load_dependencies():
    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "Missing camera dependency. From the project root run: uv sync\n"
            "(the environment needs the 'pyrealsense2' package)."
        ) from exc
    return cv2, rs


def _connected_serials(rs) -> list[str]:
    ctx = rs.context()
    return sorted(device.get_info(rs.camera_info.serial_number) for device in ctx.devices)


def _save_frame(cv2, image, output_dir: Path, prefix: str, index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"{prefix}_{stamp}_{index:06d}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")
    print(f"[saved] {path}")
    return path


def _write_latest_frame(cv2, image: np.ndarray, output_path: Path) -> None:
    """Atomically replace the image consumed by a live-preview reader."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if not cv2.imwrite(str(temp_path), image):
        raise RuntimeError(f"failed to write preview image: {temp_path}")
    os.replace(temp_path, output_path)


class _TerminalKeyReader:
    """Read single terminal key presses without a GUI window when possible."""

    def __init__(self) -> None:
        self._enabled = False
        self._stdin_fd: int | None = None
        self._original_settings = None

    def __enter__(self) -> "_TerminalKeyReader":
        if not sys.stdin.isatty():
            print("[preview] stdin is not a terminal; latest-frame publishing only")
            return self
        import termios
        import tty

        self._stdin_fd = sys.stdin.fileno()
        self._original_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._enabled = True
        return self

    def read_key(self) -> str | None:
        if not self._enabled or self._stdin_fd is None:
            return None
        readable, _, _ = select.select([self._stdin_fd], [], [], 0)
        return sys.stdin.read(1) if readable else None

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._stdin_fd is not None and self._original_settings is not None:
            import termios

            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._original_settings)


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("--width, --height, and --fps must be positive")
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.interval < 0:
        raise ValueError("--interval cannot be negative")
    if args.preview_fps <= 0:
        raise ValueError("--preview-fps must be positive")

    cv2, rs = _load_dependencies()
    connected = _connected_serials(rs)
    if args.serial not in connected:
        raise RuntimeError(
            f"head camera (serial={args.serial}) was not found. Connected RealSense serials: {connected}"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(
            f"[camera] serial={args.serial} "
            f"color={color_profile.width()}x{color_profile.height()}@{color_profile.fps()}"
        )

        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=5000)

        saved = 0
        next_save_time = 0.0
        if args.preview:
            preview_path = args.preview_output or DEFAULT_PREVIEW_OUTPUT
            period = 1.0 / args.preview_fps
            next_preview_time = 0.0
            print(
                f"[preview] writing {preview_path} at {args.preview_fps:g} Hz; "
                f"press s to save snapshots in {args.output_dir}; Ctrl-C to stop"
            )
            with _TerminalKeyReader() as key_reader:
                while True:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    image = np.asanyarray(color.get_data())
                    now = time.monotonic()
                    if now >= next_preview_time:
                        _write_latest_frame(cv2, image, preview_path)
                        next_preview_time = now + period
                    if key_reader.read_key() in {"s", "S"}:
                        _save_frame(cv2, image, args.output_dir, args.prefix, saved)
                        saved += 1
        else:
            while saved < args.count:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color = frames.get_color_frame()
                if not color:
                    continue
                now = time.monotonic()
                if now < next_save_time:
                    continue
                image = np.asanyarray(color.get_data())
                _save_frame(cv2, image, args.output_dir, args.prefix, saved)
                saved += 1
                next_save_time = now + args.interval
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
