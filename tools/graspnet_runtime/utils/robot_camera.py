"""Camera adapters for running GraspNet on robot-mounted cameras."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import sys

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional for MJPG/YUYV decode
    cv2 = None


def _candidate_site_packages() -> list[str]:
    home = Path.home()
    candidates = [
        Path(__file__).resolve().parents[1] / ".venv" / "lib" / "python3.10" / "site-packages",
        home / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        home / ".cache" / "uv" / "archive-v0" / "LMLMzNtZdRLNZsl2",
        home / ".cache" / "uv" / "archive-v0" / "WyVhwr5Copbpd_9y",
    ]
    return [str(path) for path in candidates if path.exists()]


def _import_optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        pass

    for candidate in _candidate_site_packages():
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


rs = _import_optional_module("pyrealsense2")
ob = _import_optional_module("pyorbbecsdk")


DEFAULT_D405_SERIAL = "260322270600"
DEFAULT_FEMTO_BOLT_SERIAL = "CL8L56301LK"


@dataclass
class RGBDFrame:
    color: np.ndarray
    depth: np.ndarray
    color_intrinsics: np.ndarray
    depth_intrinsics: np.ndarray
    depth_scale: float
    camera_name: str
    serial: str
    color_distortion: np.ndarray | None = None
    depth_distortion: np.ndarray | None = None
    depth_to_color_rotation: np.ndarray | None = None
    depth_to_color_translation: np.ndarray | None = None
    # Filled by deployment code when it selects a colorization projection.
    colorization_mode: str | None = None
    colorization_rotation: np.ndarray | None = None
    colorization_translation: np.ndarray | None = None
    # Device timestamps are intentionally retained separately from host wall
    # time: USB cameras do not necessarily share the host clock epoch.
    color_device_timestamp_ms: float | None = None
    depth_device_timestamp_ms: float | None = None


def list_realsense_devices() -> list[dict[str, str]]:
    if rs is None:
        return []
    devices = []
    for dev in rs.context().devices:
        devices.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
            }
        )
    return devices


def list_orbbec_devices() -> list[str]:
    if ob is None:
        return []
    ctx = ob.Context()
    device_list = ctx.query_devices()
    return [
        device_list.get_device_serial_number_by_index(i)
        for i in range(device_list.get_count())
    ]


def _intrinsics_to_matrix(profile) -> np.ndarray:
    intr = profile.get_intrinsics()
    return np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _profile_distortion(profile) -> np.ndarray | None:
    coeffs = getattr(profile.get_intrinsics(), "coeffs", None)
    if coeffs is None:
        return None
    values = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    return values.copy() if values.size else None


def _orbbec_intrinsics_to_matrix(intr) -> np.ndarray:
    return np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _orbbec_distortion(intr) -> np.ndarray | None:
    for name in ("distortion", "distortion_coeffs", "coeffs"):
        value = getattr(intr, name, None)
        if value is None:
            continue
        try:
            values = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if values.size:
            return values.copy()
    return None


def _orbbec_timestamp_ms(frame) -> float | None:
    """Best-effort device timestamp without trusting its epoch as host time."""
    try:
        value = float(frame.get_timestamp())
    except Exception:
        return None
    return value if np.isfinite(value) and value >= 0.0 else None


def _orbbec_extrinsic_to_rotation_translation(extrinsic) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(extrinsic.rot, dtype=np.float64).reshape(3, 3)
    transform = np.asarray(extrinsic.transform, dtype=np.float64).reshape(-1)
    if transform.size == 3:
        translation = transform
    elif transform.size == 12:
        translation = transform.reshape(3, 4)[:, 3]
    elif transform.size == 16:
        translation = transform.reshape(4, 4)[:3, 3]
    else:
        raise RuntimeError(
            f"unexpected Orbbec extrinsic translation shape: {transform.shape}"
        )
    return rotation, translation.reshape(3)


def _orbbec_profile_candidates(profile_list) -> list:
    return [
        profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
        for i in range(profile_list.get_count())
    ]


def _select_orbbec_video_profile(
    profile_list,
    requested_width: int,
    requested_height: int,
    requested_fps: int,
    preferred_formats: list,
):
    for fmt in preferred_formats:
        try:
            return profile_list.get_video_stream_profile(
                requested_width,
                requested_height,
                fmt,
                requested_fps,
            )
        except Exception:
            continue

    candidates = _orbbec_profile_candidates(profile_list)
    if not candidates:
        raise RuntimeError("Orbbec stream profile list is empty")

    def _sort_key(profile):
        same_fps = int(profile.get_fps() == requested_fps)
        same_wh = int(
            profile.get_width() == requested_width
            and profile.get_height() == requested_height
        )
        same_fmt = int(profile.get_format() in preferred_formats)
        pixels = int(profile.get_width()) * int(profile.get_height())
        return (same_fps, same_wh, same_fmt, pixels, int(profile.get_fps()))

    return max(candidates, key=_sort_key)


def _orbbec_frame_to_bgr(frame) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    fmt = frame.get_format()
    data = np.asarray(frame.get_data())

    if fmt == ob.OBFormat.BGR:
        return np.asarray(data, dtype=np.uint8).reshape(height, width, 3).copy()
    if fmt == ob.OBFormat.RGB:
        rgb = np.asarray(data, dtype=np.uint8).reshape(height, width, 3)
        return rgb[:, :, ::-1].copy()
    if fmt == ob.OBFormat.MJPG:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to decode Orbbec MJPG frames")
        decoded = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode Orbbec MJPG frame")
        return decoded
    if fmt == ob.OBFormat.YUYV:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to decode Orbbec YUYV frames")
        yuyv = np.asarray(data, dtype=np.uint8).reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if fmt == ob.OBFormat.NV12:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to decode Orbbec NV12 frames")
        nv12 = np.asarray(data, dtype=np.uint8).reshape(height * 3 // 2, width)
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    raise RuntimeError(f"unsupported Orbbec color format: {fmt}")


def _orbbec_depth_to_u16(frame) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    data = np.asarray(frame.get_data())
    if data.dtype == np.uint16 and data.size == width * height:
        return data.reshape(height, width).copy()
    return np.asarray(data, dtype=np.uint8).view(np.uint16).reshape(height, width).copy()


class D405Camera:
    def __init__(
        self,
        serial: str = DEFAULT_D405_SERIAL,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 15,
        enable_postprocess: bool = False,
        enable_hole_filling: bool = False,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self.enable_postprocess = bool(enable_postprocess)
        self.enable_hole_filling = bool(enable_hole_filling)
        self._ctx = None
        self._pipe = None
        self._align = None
        self._color_intrinsics = None
        self._color_distortion = None
        self._depth_scale = None
        self._spatial_filter = None
        self._temporal_filter = None
        self._hole_filling_filter = None

    def open(self) -> "D405Camera":
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")
        self._ctx = rs.context()
        connected = {
            dev.get_info(rs.camera_info.serial_number): dev for dev in self._ctx.devices
        }
        if self.serial not in connected:
            raise RuntimeError(
                f"D405 serial {self.serial} not connected. "
                f"Available RealSense devices: {list_realsense_devices()}"
            )
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        cfg.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self._pipe = rs.pipeline(self._ctx)
        try:
            profile = self._pipe.start(cfg)
            self._align = rs.align(rs.stream.color)
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._color_intrinsics = _intrinsics_to_matrix(color_profile)
            self._color_distortion = _profile_distortion(color_profile)
            self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            if self.enable_postprocess:
                # These filters preserve the aligned 640x480 profile. Do not add
                # decimation here because it would invalidate ROI pixel coordinates.
                self._spatial_filter = rs.spatial_filter()
                self._temporal_filter = rs.temporal_filter()
                if self.enable_hole_filling:
                    self._hole_filling_filter = rs.hole_filling_filter()
            for _ in range(max(0, self.warmup_frames)):
                try:
                    self._pipe.wait_for_frames(timeout_ms=2000)
                except Exception:
                    break
        except Exception:
            # 直接采集没有 Broker 代为回收；任一配置或启动异常都必须立即
            # stop pipeline，避免当前 Module 1 进程留下半开的 D405 句柄。
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
        self._pipe = None
        self._align = None
        self._spatial_filter = None
        self._temporal_filter = None
        self._hole_filling_filter = None
        self._ctx = None

    def __enter__(self) -> "D405Camera":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self, timeout_ms: int = 2000) -> RGBDFrame:
        if self._pipe is None or self._align is None:
            raise RuntimeError("D405Camera is not open")
        frames = self._pipe.wait_for_frames(timeout_ms=timeout_ms)
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D405 returned incomplete color/depth frames")
        if self._spatial_filter is not None:
            depth = self._spatial_filter.process(depth)
            depth = self._temporal_filter.process(depth)
            if self._hole_filling_filter is not None:
                depth = self._hole_filling_filter.process(depth)
        return RGBDFrame(
            color=np.asanyarray(color.get_data()).copy(),
            depth=np.asanyarray(depth.get_data()).copy(),
            color_intrinsics=np.array(self._color_intrinsics, copy=True),
            depth_intrinsics=np.array(self._color_intrinsics, copy=True),
            depth_scale=float(self._depth_scale),
            camera_name="d405",
            serial=self.serial,
            color_distortion=None if self._color_distortion is None else np.array(self._color_distortion, copy=True),
            depth_distortion=None,
        )


class FemtoBoltCamera:
    def __init__(
        self,
        serial: str = DEFAULT_FEMTO_BOLT_SERIAL,
        color_width: int = 1280,
        color_height: int = 720,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
        warmup_frames: int = 15,
    ):
        self.serial = serial
        self.color_width = int(color_width)
        self.color_height = int(color_height)
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self._ctx = None
        self._pipe = None
        self._color_intrinsics = None
        self._color_distortion = None
        self._depth_intrinsics = None
        self._depth_distortion = None
        self._depth_to_color_rotation = None
        self._depth_to_color_translation = None
        self._align_mode = None
        self._align_filter = None

    def open(self) -> "FemtoBoltCamera":
        if ob is None:
            raise RuntimeError("pyorbbecsdk is not installed")
        self._ctx = ob.Context()
        device_list = self._ctx.query_devices()
        available = [
            device_list.get_device_serial_number_by_index(i)
            for i in range(device_list.get_count())
        ]
        if self.serial not in available:
            raise RuntimeError(
                f"Femto Bolt serial {self.serial} not connected. "
                f"Available Orbbec devices: {available}"
            )
        device = device_list.get_device_by_serial_number(self.serial)
        self._pipe = ob.Pipeline(device)
        cfg = ob.Config()
        cfg.disable_all_stream()

        color_profiles = self._pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        depth_profiles = self._pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        color_profile = _select_orbbec_video_profile(
            color_profiles,
            self.color_width,
            self.color_height,
            self.fps,
            [ob.OBFormat.BGR, ob.OBFormat.RGB, ob.OBFormat.MJPG, ob.OBFormat.YUYV, ob.OBFormat.NV12],
        )
        depth_profile = _select_orbbec_video_profile(
            depth_profiles,
            self.depth_width,
            self.depth_height,
            self.fps,
            [ob.OBFormat.Z16, ob.OBFormat.Y16],
        )

        cfg.enable_stream(color_profile)
        cfg.enable_stream(depth_profile)

        self._pipe.enable_frame_sync()
        started = False
        start_errors: list[str] = []
        for align_mode in (ob.OBAlignMode.HW_MODE, ob.OBAlignMode.SW_MODE, ob.OBAlignMode.DISABLE):
            try:
                cfg.set_align_mode(align_mode)
                self._pipe.start(cfg)
                self._align_mode = align_mode
                started = True
                break
            except Exception as exc:
                start_errors.append(f"{align_mode}: {exc}")
        if not started:
            raise RuntimeError("Femto Bolt failed to start with any align mode: " + " | ".join(start_errors))

        self._color_intrinsics = _orbbec_intrinsics_to_matrix(color_profile.get_intrinsic())
        self._color_distortion = _orbbec_distortion(color_profile.get_intrinsic())
        self._depth_intrinsics = _orbbec_intrinsics_to_matrix(depth_profile.get_intrinsic())
        self._depth_distortion = _orbbec_distortion(depth_profile.get_intrinsic())
        rotation, translation = _orbbec_extrinsic_to_rotation_translation(
            depth_profile.get_extrinsic_to(color_profile)
        )
        self._depth_to_color_rotation = rotation
        self._depth_to_color_translation = translation
        try:
            self._align_filter = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM)
        except Exception:
            self._align_filter = None
        print(f"[camera] femto_bolt align mode: {self._align_mode}")
        for _ in range(max(0, self.warmup_frames)):
            try:
                self._pipe.wait_for_frames(2000)
            except Exception:
                break
        return self

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
        self._pipe = None
        self._ctx = None
        self._align_filter = None

    def __enter__(self) -> "FemtoBoltCamera":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self, timeout_ms: int = 2000) -> RGBDFrame:
        if self._pipe is None:
            raise RuntimeError("FemtoBoltCamera is not open")
        frames = self._pipe.wait_for_frames(timeout_ms)
        if self._align_filter is not None:
            try:
                aligned_frames = self._align_filter.process(frames)
                if aligned_frames is not None:
                    frames = aligned_frames
            except Exception:
                pass
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if color is None or depth is None:
            raise RuntimeError("Femto Bolt returned incomplete color/depth frames")
        color_bgr = _orbbec_frame_to_bgr(color)
        depth_u16 = _orbbec_depth_to_u16(depth)
        depth_intrinsics = np.array(self._depth_intrinsics, copy=True)
        depth_to_color_rotation = np.array(self._depth_to_color_rotation, copy=True)
        depth_to_color_translation = np.array(self._depth_to_color_translation, copy=True)
        depth_distortion = np.array(self._depth_distortion, copy=True) if self._depth_distortion is not None else None

        if depth_u16.shape == color_bgr.shape[:2]:
            # SDK-aligned depth behaves like "depth already in color camera pixels".
            depth_intrinsics = np.array(self._color_intrinsics, copy=True)
            depth_to_color_rotation = None
            depth_to_color_translation = None
            depth_distortion = None

        return RGBDFrame(
            color=color_bgr,
            depth=depth_u16,
            color_intrinsics=np.array(self._color_intrinsics, copy=True),
            depth_intrinsics=depth_intrinsics,
            depth_scale=float(depth.get_depth_scale()),
            camera_name="femto_bolt",
            serial=self.serial,
            color_distortion=None if self._color_distortion is None else np.array(self._color_distortion, copy=True),
            depth_distortion=depth_distortion,
            depth_to_color_rotation=depth_to_color_rotation,
            depth_to_color_translation=depth_to_color_translation,
            color_device_timestamp_ms=_orbbec_timestamp_ms(color),
            depth_device_timestamp_ms=_orbbec_timestamp_ms(depth),
        )
