"""Color-only adapter for the chest-mounted Orbbec Femto Bolt."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import cv2
import numpy as np


DEFAULT_CHEST_SERIAL = "CL8654103PJ"


def _import_orbbec():
    try:
        return importlib.import_module("pyorbbecsdk")
    except ImportError:
        pass
    for site_packages in (
        Path("/home/dainanjun/Public/data_collector/.venv/lib/python3.10/site-packages"),
        Path("/home/dainanjun/qiuguolin/graspnet/.venv-orbbec2-uv/lib/python3.10/site-packages"),
    ):
        if site_packages.is_dir() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
        try:
            return importlib.import_module("pyorbbecsdk")
        except ImportError:
            continue
    raise RuntimeError(
        "pyorbbecsdk is not importable. Use /home/dainanjun/Public/data_collector/.venv/bin/python "
        "or install the Orbbec SDK."
    )


@dataclass
class ChestRGBFrame:
    color: np.ndarray
    color_intrinsics: np.ndarray
    serial: str
    color_distortion: np.ndarray | None


def _select_color_profile(ob, profiles, width: int, height: int, fps: int):
    formats = [ob.OBFormat.BGR, ob.OBFormat.RGB, ob.OBFormat.MJPG, ob.OBFormat.YUYV, ob.OBFormat.NV12]
    for fmt in formats:
        try:
            return profiles.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            pass
    candidates = [profiles.get_stream_profile_by_index(i).as_video_stream_profile() for i in range(profiles.get_count())]
    if not candidates:
        raise RuntimeError("Orbbec color stream profile list is empty")
    return max(
        candidates,
        key=lambda p: (
            int(p.get_fps() == fps),
            int(p.get_width() == width and p.get_height() == height),
            int(p.get_format() in formats),
            int(p.get_width()) * int(p.get_height()),
        ),
    )


def _frame_to_bgr(ob, frame) -> np.ndarray:
    width, height = int(frame.get_width()), int(frame.get_height())
    data, fmt = np.asarray(frame.get_data()), frame.get_format()
    if fmt == ob.OBFormat.BGR:
        return np.asarray(data, dtype=np.uint8).reshape(height, width, 3).copy()
    if fmt == ob.OBFormat.RGB:
        return np.asarray(data, dtype=np.uint8).reshape(height, width, 3)[:, :, ::-1].copy()
    if fmt == ob.OBFormat.MJPG:
        image = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode Orbbec MJPG color frame")
        return image
    if fmt == ob.OBFormat.YUYV:
        return cv2.cvtColor(np.asarray(data, dtype=np.uint8).reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == ob.OBFormat.NV12:
        return cv2.cvtColor(np.asarray(data, dtype=np.uint8).reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    raise RuntimeError(f"unsupported Orbbec color format: {fmt}")


class ChestOrbbecCamera:
    """Color stream and active-profile intrinsics for eye-to-hand calibration."""

    def __init__(self, serial: str = DEFAULT_CHEST_SERIAL, width: int = 1280, height: int = 720,
                 fps: int = 30, warmup_frames: int = 15):
        self.serial, self.width, self.height, self.fps = str(serial), int(width), int(height), int(fps)
        self.warmup_frames = int(warmup_frames)
        self._ob = self._pipeline = self._intrinsics = self._distortion = None

    def open(self) -> "ChestOrbbecCamera":
        ob = _import_orbbec()
        context = ob.Context()
        devices = context.query_devices()
        serials = [devices.get_device_serial_number_by_index(i) for i in range(devices.get_count())]
        if self.serial not in serials:
            raise RuntimeError(f"chest Orbbec serial {self.serial} not connected; available: {serials}")
        pipeline = ob.Pipeline(devices.get_device_by_serial_number(self.serial))
        profile = _select_color_profile(
            ob, pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR), self.width, self.height, self.fps
        )
        config = ob.Config()
        config.disable_all_stream()
        config.enable_stream(profile)
        pipeline.start(config)
        intrinsic = profile.get_intrinsic()
        self._intrinsics = np.array(
            [[intrinsic.fx, 0.0, intrinsic.cx], [0.0, intrinsic.fy, intrinsic.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        distortion = getattr(intrinsic, "distortion", None)
        try:
            values = np.asarray(distortion, dtype=np.float64).reshape(-1)
            self._distortion = values if values.size else None
        except Exception:
            self._distortion = None
        self._ob, self._pipeline = ob, pipeline
        print(f"[camera] chest Orbbec serial={self.serial} color={profile.get_width()}x{profile.get_height()}@{profile.get_fps()}")
        for _ in range(max(0, self.warmup_frames)):
            pipeline.wait_for_frames(2000)
        return self

    def read(self, timeout_ms: int = 2000) -> ChestRGBFrame:
        if self._pipeline is None or self._intrinsics is None:
            raise RuntimeError("ChestOrbbecCamera is not open")
        frames = self._pipeline.wait_for_frames(int(timeout_ms))
        color = None if frames is None else frames.get_color_frame()
        if color is None:
            raise RuntimeError("chest Orbbec returned no color frame")
        return ChestRGBFrame(_frame_to_bgr(self._ob, color), self._intrinsics.copy(), self.serial,
                             None if self._distortion is None else self._distortion.copy())

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
