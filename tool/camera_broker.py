"""Strict client for the Module 0 Camera Broker capture-bundle contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import uuid

import numpy as np

from tool.inspection_support.robot_camera import RGBDFrame


SCHEMA_VERSION = 1


class FrameContractError(RuntimeError):
    """The Broker returned a completed-looking bundle that violates its ABI."""


def _request(socket_path: str, payload: dict, timeout_s: float = 30.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_s)
        connection.connect(socket_path)
        connection.sendall((json.dumps(payload) + "\n").encode())
        reply = b""
        while not reply.endswith(b"\n"):
            block = connection.recv(65536)
            if not block:
                raise RuntimeError("Camera Broker closed the reply early")
            reply += block
    response = json.loads(reply.decode())
    if not response.get("ok"):
        raise RuntimeError(
            "Camera Broker request failed "
            f"[{response.get('error_code', 'INTERNAL')}]: {response.get('error')}"
        )
    return response["result"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class BrokerFemtoCamera:
    """Drop-in capture source for Module 2; it never opens an SDK device."""

    def __init__(self, socket_path: str, *, profile_id: str = "inspect_femto_v1"):
        self.socket_path = socket_path
        self.profile_id = profile_id

    def __enter__(self) -> "BrokerFemtoCamera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def assert_compatible(self, inspect_config: dict) -> None:
        """Reject silent drift between Module 2 calibration/config and Broker profile."""
        status = _request(self.socket_path, {"op": "status"})
        if status.get("schema_version") != SCHEMA_VERSION:
            raise FrameContractError("Camera Broker schema version mismatch")
        spec = (status.get("profiles") or {}).get(self.profile_id)
        camera = (status.get("camera_config") or {}).get("femto_bolt")
        if not isinstance(spec, dict) or not isinstance(camera, dict):
            raise FrameContractError("Camera Broker does not expose the required Femto profile")
        expected = {
            "camera": "femto_bolt",
            "frame_count": int(inspect_config["capture_frames"]),
        }
        actual = {"camera": spec.get("camera"), "frame_count": spec.get("frame_count")}
        if actual != expected:
            raise FrameContractError(f"inspect profile mismatch: broker={actual}, module={expected}")
        for key in ("serial", "color_width", "color_height", "depth_width", "depth_height", "warmup_frames"):
            expected_value = inspect_config["camera_serial"] if key == "serial" else inspect_config.get(key)
            if expected_value is not None and str(camera.get(key)) != str(expected_value):
                raise FrameContractError(
                    f"inspect camera configuration mismatch for {key}: "
                    f"broker={camera.get(key)!r}, module={expected_value!r}"
                )

    def capture(self, *, purpose: str) -> RGBDFrame:
        request_id = str(uuid.uuid4())
        result = _request(
            self.socket_path,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "op": "capture_bundle",
                "camera": "femto_bolt",
                "profile_id": self.profile_id,
                "consumer": "module_2_left_inspect_apart",
                "purpose": purpose,
            },
            timeout_s=60.0,
        )
        if result.get("schema_version") != SCHEMA_VERSION or result.get("request_id") != request_id:
            raise FrameContractError("Camera Broker response version or request id mismatch")
        manifest_path = Path(str(result["manifest_path"])).resolve()
        if result.get("sha256") != _sha256(manifest_path):
            raise FrameContractError("Camera Broker manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("state") != "complete":
            raise FrameContractError("capture bundle is not a completed schema-v1 artifact")
        if manifest.get("request", {}).get("request_id") != request_id:
            raise FrameContractError("capture bundle request id mismatch")
        frames = manifest.get("frames") or {}
        if self.profile_id.startswith("inspect_femto"):
            if int(frames.get("request_flush_count", 0)) < 1:
                raise FrameContractError("Femto bundle lacks request-boundary stale-frame flush")
            def parse_increasing_timestamp_series(key: str) -> list[float]:
                timestamps = frames.get(key)
                if not isinstance(timestamps, list) or not timestamps:
                    raise FrameContractError(f"Femto bundle lacks {key}")
                try:
                    parsed = [float(value) for value in timestamps]
                except (TypeError, ValueError) as exc:
                    raise FrameContractError(f"Femto bundle has invalid {key}") from exc
                if not all(np.isfinite(value) and value >= 0.0 for value in parsed):
                    raise FrameContractError(f"Femto bundle has non-finite {key}")
                if any(later <= earlier for earlier, later in zip(parsed, parsed[1:])):
                    raise FrameContractError(f"Femto bundle has non-increasing {key}")
                return parsed

            parsed_timestamps = parse_increasing_timestamp_series("color_device_timestamps_ms")
            depth_timestamps = parse_increasing_timestamp_series("depth_device_timestamps_ms")
            if len(parsed_timestamps) != int(frames.get("accepted_count", -1)) or len(depth_timestamps) != len(parsed_timestamps):
                raise FrameContractError("Femto device timestamp count does not match accepted frames")
            try:
                selected_timestamp = float(frames.get("selected_color_device_timestamp_ms"))
            except (TypeError, ValueError) as exc:
                raise FrameContractError("Femto bundle lacks selected color timestamp") from exc
            if selected_timestamp != parsed_timestamps[-1]:
                raise FrameContractError("Femto selected color timestamp is not the last accepted frame")
            try:
                selected_depth_timestamp = float(frames.get("selected_depth_device_timestamp_ms"))
            except (TypeError, ValueError) as exc:
                raise FrameContractError("Femto bundle lacks selected depth timestamp") from exc
            if selected_depth_timestamp != depth_timestamps[-1]:
                raise FrameContractError("Femto selected depth timestamp is not the last accepted frame")
        geometry = manifest.get("geometry") or {}
        if geometry.get("color_order") != "BGR" or geometry.get("depth_dtype") != "uint16":
            raise FrameContractError("unsupported camera bundle pixel format")
        scale = float(geometry.get("depth_scale_m_per_unit", 0.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise FrameContractError("camera bundle has invalid depth_scale_m_per_unit")
        root = manifest_path.parent
        for name, expected in (manifest.get("files") or {}).items():
            path = root / name
            if not path.is_file() or _sha256(path) != expected:
                raise FrameContractError(f"camera bundle checksum mismatch: {name}")
        color = np.load(root / "color.npy")
        depth = np.load(root / "depth.npy")
        color_intrinsics = np.load(root / "color_intrinsics.npy")
        depth_intrinsics = np.load(root / "depth_intrinsics.npy")
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise FrameContractError(f"invalid BGR color array: {color.shape} {color.dtype}")
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise FrameContractError(f"invalid depth array: {depth.shape} {depth.dtype}")
        for label, value in (("color_intrinsics", color_intrinsics), ("depth_intrinsics", depth_intrinsics)):
            if value.shape != (3, 3) or not np.all(np.isfinite(value)):
                raise FrameContractError(f"invalid {label}")
        registration = geometry.get("registration")
        rotation = translation = None
        transform_path = root / "T_depth_color.npy"
        if registration == "separate_optical_frames":
            if not transform_path.is_file():
                raise FrameContractError("separate depth geometry requires T_depth_color.npy")
            transform = np.load(transform_path)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise FrameContractError("invalid T_depth_color.npy")
            rotation, translation = transform[:3, :3], transform[:3, 3]
        elif registration != "depth_in_color_geometry":
            raise FrameContractError(f"unknown depth registration: {registration!r}")
        load_optional = lambda name: np.load(root / name) if (root / name).is_file() else None
        return RGBDFrame(
            color=color, depth=depth, color_intrinsics=color_intrinsics, depth_intrinsics=depth_intrinsics,
            depth_scale=scale, camera_name="femto_bolt", serial=str(manifest["camera"]["serial"]),
            color_distortion=load_optional("color_distortion.npy"),
            depth_distortion=load_optional("depth_distortion.npy"),
            depth_to_color_rotation=rotation, depth_to_color_translation=translation,
        )
