"""Minimal RGB adapter for the fixed-pose PARM head RealSense D435i."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np


def _import_realsense():
    """Match the existing robot camera adapters' SDK-environment fallback."""
    try:
        return importlib.import_module("pyrealsense2")
    except ImportError:
        pass
    candidates = [
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        Path("/home/dainanjun/qiuguolin/graspnet/.venv/lib/python3.10/site-packages"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            return importlib.import_module("pyrealsense2")
        except ImportError:
            continue
    raise RuntimeError(
        "pyrealsense2 is not importable. Expected it in the active environment "
        "or /home/dainanjun/Public/data_collector/.venv."
    )


@dataclass
class HeadRGBFrame:
    color: np.ndarray
    color_intrinsics: np.ndarray
    serial: str
    color_distortion: np.ndarray | None


class HeadD435iCamera:
    """Color-only RealSense source used by the head eye-to-hand calibration.

    Chessboard pose estimation only needs the color image and its calibrated
    stream profile, so depth is intentionally not enabled.
    """

    def __init__(self, serial: str, width: int = 1280, height: int = 720,
                 fps: int = 30, warmup_frames: int = 15):
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self._pipe = None
        self._rs = None
        self._intrinsics = None
        self._distortion = None

    def open(self) -> "HeadD435iCamera":
        rs = _import_realsense()
        connected = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().devices
        ]
        if self.serial not in connected:
            raise RuntimeError(
                f"head D435i serial {self.serial} not connected; available: {connected}"
            )
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._intrinsics = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        coeffs = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1)
        self._distortion = coeffs if coeffs.size else None
        self._rs = rs
        print(
            f"[camera] head D435i serial={self.serial} "
            f"color={color_profile.width()}x{color_profile.height()}@{color_profile.fps()}"
        )
        for _ in range(max(0, self.warmup_frames)):
            self._pipe.wait_for_frames(timeout_ms=2000)
        return self

    def read(self, timeout_ms: int = 2000) -> HeadRGBFrame:
        if self._pipe is None or self._intrinsics is None:
            raise RuntimeError("HeadD435iCamera is not open")
        color = self._pipe.wait_for_frames(timeout_ms=timeout_ms).get_color_frame()
        if not color:
            raise RuntimeError("head D435i returned no color frame")
        return HeadRGBFrame(
            color=np.asanyarray(color.get_data()).copy(),
            color_intrinsics=np.array(self._intrinsics, copy=True),
            serial=self.serial,
            color_distortion=None if self._distortion is None else np.array(self._distortion, copy=True),
        )

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
        self._pipe = None

    def __enter__(self) -> "HeadD435iCamera":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
