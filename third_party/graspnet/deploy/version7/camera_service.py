#!/usr/bin/env python3
"""Persistent D405/Femto RGB-D capture service for Version7."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import socketserver
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image


BROKER_SCHEMA_VERSION = 1

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "utils", ROOT / "deploy", ROOT / "deploy" / "calibration"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from calibration_common import _import_parm_sdk, read_robot_base_to_eef  # noqa: E402
from robot_camera import DEFAULT_D405_SERIAL, DEFAULT_FEMTO_BOLT_SERIAL, D405Camera, FemtoBoltCamera  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent Version7 RGB-D camera service")
    p.add_argument("--socket-path", default="/tmp/graspnet-v7-camera.sock")
    p.add_argument("--allow-root", default=str(ROOT / "deploy" / "version7" / "jobs"))
    p.add_argument(
        "--capture-root",
        default=None,
        help="Broker-owned immutable capture bundles (default: <allow-root>/captures)",
    )
    p.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    p.add_argument("--arm-id", type=int, default=1)
    p.add_argument("--d405-serial", default=DEFAULT_D405_SERIAL)
    p.add_argument("--d405-width", type=int, default=640)
    p.add_argument("--d405-height", type=int, default=480)
    p.add_argument("--d405-fps", type=int, choices=(5, 15, 30), default=30)
    p.add_argument("--d405-postprocess", action="store_true", default=True)
    p.add_argument("--no-d405-postprocess", dest="d405_postprocess", action="store_false")
    p.add_argument("--d405-hole-filling", action="store_true")
    p.add_argument("--femto-serial", default=DEFAULT_FEMTO_BOLT_SERIAL)
    p.add_argument("--femto-color-width", type=int, default=1280)
    p.add_argument("--femto-color-height", type=int, default=720)
    p.add_argument("--femto-depth-width", type=int, default=640)
    p.add_argument("--femto-depth-height", type=int, default=480)
    p.add_argument("--femto-warmup-frames", type=int, default=3)
    p.add_argument(
        "--femto-request-flush-frames", type=int, default=30,
        help="discard this many Femto frames at every capture request boundary",
    )
    p.add_argument("--femto-frame-read-retries", type=int, default=10,
                   help="retry incomplete Femto frames before rejecting a request")
    p.add_argument("--min-valid-depth-ratio", type=float, default=0.01)
    p.add_argument("--max-capture-bundles", type=int, default=200)
    p.add_argument("--disable-femto", action="store_true",
                   help="open D405 only; reject Femto capture requests")
    p.add_argument(
        "--no-arm-state", action="store_true",
        help="camera-only mode: never open the SDK Arm or add legacy T_base_eef metadata",
    )
    return p.parse_args()


def depth_scale_m(frame) -> float:
    scale = float(frame.depth_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("camera returned an invalid depth scale")
    return scale * .001 if frame.camera_name == "femto_bolt" and scale >= 1.0 else scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class CameraEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.d405_lock = threading.Lock()
        self.femto_lock = threading.Lock()
        self.stack = ExitStack()
        self.d405 = self.stack.enter_context(D405Camera(serial=args.d405_serial, width=args.d405_width,
                                                        height=args.d405_height, fps=args.d405_fps,
                                                        enable_postprocess=args.d405_postprocess,
                                                        enable_hole_filling=args.d405_hole_filling))
        self.femto = None
        if not args.disable_femto:
            self.femto = self.stack.enter_context(FemtoBoltCamera(serial=args.femto_serial,
                                                                   color_width=args.femto_color_width,
                                                                   color_height=args.femto_color_height,
                                                                   depth_width=args.femto_depth_width,
                                                                   depth_height=args.femto_depth_height,
                                                                   warmup_frames=args.femto_warmup_frames))
        self.arm = None
        if not args.no_arm_state:
            self.arm = self.stack.enter_context(_import_parm_sdk().Arm(host=args.arm_host))
        print("-> Version7 camera service ready: D405{} is open".format(
            " only" if args.disable_femto else " and Femto"
        ), flush=True)

    def close(self) -> None:
        self.stack.close()

    def status(self) -> dict:
        status = {"arm": {"host": self.args.arm_host, "id": self.args.arm_id,
                          "legacy_eef_pose_enabled": self.arm is not None},
                  "d405": {"serial": self.args.d405_serial, "width": self.args.d405_width,
                            "height": self.args.d405_height, "fps": self.args.d405_fps,
                            "postprocess": self.args.d405_postprocess,
                            "hole_filling": self.args.d405_hole_filling}}
        if self.femto is not None:
            status["femto_bolt"] = {"serial": self.args.femto_serial,
                                     "color_width": self.args.femto_color_width, "color_height": self.args.femto_color_height,
                                     "depth_width": self.args.femto_depth_width, "depth_height": self.args.femto_depth_height,
                                     "warmup_frames": self.args.femto_warmup_frames,
                                     "request_flush_frames": self.args.femto_request_flush_frames,
                                     "frame_read_retries": self.args.femto_frame_read_retries}
        return status

    def profile(self, profile_id: str) -> tuple[str, int]:
        """Return the fixed device and frame count for a public Broker profile."""
        profiles = {
            "graspnet_d405_v1": ("d405", 1),
            "graspnet_femto_v1": ("femto_bolt", 1),
            # Preserve Module 2's existing median-of-five capture behaviour.
            "inspect_femto_v1": ("femto_bolt", 5),
        }
        try:
            camera, frame_count = profiles[profile_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown Camera Broker profile: {profile_id}") from exc
        if camera == "femto_bolt" and self.femto is None:
            raise RuntimeError(f"profile {profile_id} requires Femto, but Femto is disabled")
        return camera, frame_count

    def profile_fingerprint(self, profile_id: str) -> str:
        camera, frame_count = self.profile(profile_id)
        payload = {"profile_id": profile_id, "camera": camera, "frame_count": frame_count,
                   "camera_config": self.status()[camera]}
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def profile_specs(self) -> dict:
        specs = {}
        for profile_id in ("graspnet_d405_v1", "graspnet_femto_v1", "inspect_femto_v1"):
            try:
                camera, frame_count = self.profile(profile_id)
            except RuntimeError:
                continue
            specs[profile_id] = {
                "camera": camera,
                "frame_count": frame_count,
                "fingerprint_sha256": self.profile_fingerprint(profile_id),
            }
        return specs

    def _read_complete_femto(self):
        """Hide the short incomplete-frame burst emitted just after stream start."""
        assert self.femto is not None
        last_error: Exception | None = None
        for attempt in range(self.args.femto_frame_read_retries + 1):
            try:
                return self.femto.read()
            except RuntimeError as exc:
                last_error = exc
                if "incomplete color/depth" not in str(exc) or attempt >= self.args.femto_frame_read_retries:
                    raise
                time.sleep(0.04)
        raise RuntimeError("Femto complete-frame retry exhausted") from last_error

    def probe(self, profile_id: str, require_eef_pose: bool = False) -> dict:
        """Read one RGB-D frame without writing a bundle or controlling the arm."""
        camera, _ = self.profile(profile_id)
        lock = self.d405_lock if camera == "d405" else self.femto_lock
        source = self.d405 if camera == "d405" else self.femto
        with lock:
            frame = self._read_complete_femto() if camera == "femto_bolt" else source.read()
        depth = np.asarray(frame.depth)
        color = np.asarray(frame.color)
        ratio = float(np.count_nonzero(depth) / depth.size) if depth.size else 0.0
        valid = (
            color.dtype == np.uint8 and color.ndim == 3 and color.shape[2] == 3
            and depth.dtype == np.uint16 and depth.ndim == 2
            and np.asarray(frame.color_intrinsics).shape == (3, 3)
            and np.asarray(frame.depth_intrinsics).shape == (3, 3)
            and np.all(np.isfinite(frame.color_intrinsics)) and np.all(np.isfinite(frame.depth_intrinsics))
            and np.isfinite(depth_scale_m(frame)) and ratio >= float(self.args.min_valid_depth_ratio)
        )
        if not valid:
            raise RuntimeError("QUALITY_REJECTED: camera readiness probe failed")
        result = {"profile_id": profile_id, "camera": camera, "valid_depth_ratio": ratio}
        if require_eef_pose:
            if camera != "d405":
                raise RuntimeError("EEF pose probe is only valid for the D405 profile")
            if self.arm is None:
                raise RuntimeError("legacy T_base_eef is disabled in camera-only mode")
            T_base_eef = read_robot_base_to_eef(self.arm, self.args.arm_id)
            result["T_base_eef"] = np.asarray(T_base_eef, dtype=float).tolist()
        return result

    def capture(self, request: dict) -> dict:
        camera = str(request["camera"])
        output = Path(request["output_dir"])
        count = int(request.get("frame_count", 1))
        if count < 1:
            raise RuntimeError("frame_count must be positive")
        if camera == "d405":
            with self.d405_lock:
                started_unix_s = time.time()
                started_monotonic_s = time.monotonic()
                frames = [self.d405.read() for _ in range(count)]
                frame = frames[-1]
                if count > 1:
                    frame = replace(frame, depth=np.median(np.stack([item.depth for item in frames]), axis=0).astype(frame.depth.dtype))
                require_eef_pose = bool(request.get("require_eef_pose", self.arm is not None))
                if require_eef_pose:
                    if self.arm is None:
                        raise RuntimeError("legacy T_base_eef requested from camera-only service")
                    T_base_eef = read_robot_base_to_eef(self.arm, self.args.arm_id)
                else:
                    T_base_eef = None
        elif camera == "femto_bolt":
            if self.femto is None:
                raise RuntimeError("Femto is disabled for this Version7 camera service")
            with self.femto_lock:
                started_unix_s = time.time()
                started_monotonic_s = time.monotonic()
                # A persistent Orbbec pipeline can retain frames while no
                # consumer is reading it.  Warmup at service startup does not
                # solve that request-boundary backlog, so discard a bounded
                # number before accepting this request's frames.
                # Do not retain discarded RGB-D arrays: 30 full Femto frames
                # are large enough to create an avoidable transient memory spike.
                for _ in range(self.args.femto_request_flush_frames):
                    self._read_complete_femto()
                flushed_count = self.args.femto_request_flush_frames
                frames = [self._read_complete_femto() for _ in range(count)]
                color_timestamps = [item.color_device_timestamp_ms for item in frames]
                depth_timestamps = [item.depth_device_timestamp_ms for item in frames]
                if any(value is None for value in color_timestamps + depth_timestamps):
                    raise RuntimeError("STALE_FRAME: Femto did not provide device timestamps for every accepted frame")
                color_timestamps = [float(value) for value in color_timestamps]
                depth_timestamps = [float(value) for value in depth_timestamps]
                if any(
                    later <= earlier for earlier, later in zip(color_timestamps, color_timestamps[1:])
                ) or any(
                    later <= earlier for earlier, later in zip(depth_timestamps, depth_timestamps[1:])
                ):
                    raise RuntimeError(
                        "STALE_FRAME: Femto device timestamps did not advance after request-boundary flush"
                    )
                frame = frames[-1]
                if count > 1:
                    frame = replace(
                        frame,
                        depth=np.median(
                            np.stack([item.depth for item in frames]), axis=0
                        ).astype(frame.depth.dtype),
                    )
                T_base_eef = None
        else:
            raise RuntimeError("camera must be d405 or femto_bolt")
        finished_unix_s = time.time()
        output.mkdir(parents=True, exist_ok=False)
        np.save(output / "color.npy", frame.color)
        np.save(output / "depth.npy", frame.depth)
        np.save(output / "depth_intrinsics.npy", frame.depth_intrinsics)
        np.save(output / "color_intrinsics.npy", frame.color_intrinsics)
        if frame.depth_distortion is not None:
            np.save(output / "depth_distortion.npy", np.asarray(frame.depth_distortion, dtype=np.float64))
        if frame.color_distortion is not None:
            np.save(output / "color_distortion.npy", np.asarray(frame.color_distortion, dtype=np.float64))
        if frame.depth_to_color_rotation is not None and frame.depth_to_color_translation is not None:
            T_depth_color = np.eye(4, dtype=np.float64)
            T_depth_color[:3, :3] = np.asarray(frame.depth_to_color_rotation, dtype=np.float64).reshape(3, 3)
            T_depth_color[:3, 3] = np.asarray(frame.depth_to_color_translation, dtype=np.float64).reshape(3)
            np.save(output / "T_depth_color.npy", T_depth_color)
        Image.fromarray(frame.color[:, :, ::-1]).save(output / "color.png")
        depth_scale_value_m = depth_scale_m(frame)
        meta = {
            "camera_name": frame.camera_name, "serial": frame.serial,
            "depth_scale_m": depth_scale_value_m, "depth_scale_raw": float(frame.depth_scale),
            "color_shape_hw": list(frame.color.shape[:2]), "depth_shape_hw": list(frame.depth.shape[:2]),
            "frame_count": count,
            "has_depth_distortion": frame.depth_distortion is not None,
            "has_color_distortion": frame.color_distortion is not None,
            "capture_started_unix_s": started_unix_s,
            "capture_finished_unix_s": finished_unix_s,
            "capture_elapsed_s": finished_unix_s - started_unix_s,
        }
        if camera == "femto_bolt":
            meta["femto_request_flush_frames"] = flushed_count
            meta["femto_color_device_timestamps_ms"] = color_timestamps
            meta["femto_depth_device_timestamps_ms"] = depth_timestamps
            meta["femto_selected_color_device_timestamp_ms"] = frame.color_device_timestamp_ms
            meta["femto_selected_depth_device_timestamp_ms"] = frame.depth_device_timestamp_ms
        if T_base_eef is not None:
            meta["T_base_eef"] = np.asarray(T_base_eef, dtype=float).tolist()
        (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"capture_dir": str(output), "camera": camera, "frame_count": count,
                "depth_shape_hw": meta["depth_shape_hw"], "capture_started_monotonic_s": started_monotonic_s,
                "capture_elapsed_s": finished_unix_s - started_unix_s}
    def capture_pair(self, d405_request: dict, femto_request: dict, max_start_skew_s: float) -> dict:
        """Capture from independent devices concurrently while keeping each SDK single-threaded."""
        d405_request = dict(d405_request, camera="d405")
        femto_request = dict(femto_request, camera="femto_bolt")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="v7-camera") as workers:
            d405 = workers.submit(self.capture, d405_request)
            femto = workers.submit(self.capture, femto_request)
            d405_result, femto_result = d405.result(), femto.result()
        start_skew_s = abs(d405_result["capture_started_monotonic_s"] - femto_result["capture_started_monotonic_s"])
        if start_skew_s > max_start_skew_s:
            raise RuntimeError(f"D405/Femto capture start skew {start_skew_s:.3f}s exceeds {max_start_skew_s:.3f}s")
        return {"d405": d405_result, "femto_bolt": femto_result, "start_skew_s": start_skew_s}


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline().decode())
            response = self.server.handle_request(request)  # type: ignore[attr-defined]
            self.wfile.write((json.dumps({"ok": True, "result": response}) + "\n").encode())
        except Exception as exc:
            message = str(exc)
            error_code = "INTERNAL"
            if message.startswith("QUALITY_REJECTED:"):
                error_code, message = message.split(":", 1)
                message = message.strip()
            elif "unknown Camera Broker profile" in message:
                error_code = "UNKNOWN_PROFILE"
            elif "requires request_id" in message or "unsupported schema_version" in message:
                error_code = "INVALID_REQUEST"
            self.wfile.write((json.dumps({"ok": False, "error_code": error_code, "error": message}) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    def __init__(self, socket_path: str, engine: CameraEngine, allow_root: Path, capture_root: Path):
        super().__init__(socket_path, Handler)
        self.engine, self.allow_root = engine, allow_root.resolve()
        self.capture_root = capture_root.resolve()
        self.capture_root_lock = threading.Lock()

    def _prune_capture_bundles(self, protected: Path) -> None:
        limit = int(self.engine.args.max_capture_bundles)
        if limit < 1:
            raise RuntimeError("max_capture_bundles must be positive")
        completed = sorted(
            (path for path in self.capture_root.iterdir()
             if path.is_dir() and not path.name.startswith(".") and (path / "manifest.json").is_file()),
            key=lambda path: path.name,
            reverse=True,
        )
        removable = [path for path in completed if path != protected]
        for path in removable[max(0, limit - 1):]:
            shutil.rmtree(path)

    def _capture_bundle(self, request: dict) -> dict:
        if int(request.get("schema_version", -1)) != BROKER_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported schema_version: {request.get('schema_version')!r}")
        request_id = str(request.get("request_id") or "").strip()
        if not request_id:
            raise RuntimeError("capture_bundle requires request_id")
        profile_id = str(request.get("profile_id") or "").strip()
        camera, frame_count = self.engine.profile(profile_id)
        requested_camera = request.get("camera")
        if requested_camera is not None and str(requested_camera) != camera:
            raise RuntimeError(f"profile {profile_id} belongs to {camera}, not {requested_camera}")
        capture_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{time.time_ns() % 1_000_000_000:09d}Z_{request_id}"
        final_dir = self.capture_root / capture_id
        temporary_dir = self.capture_root / ("." + capture_id + ".partial")
        self.capture_root.mkdir(parents=True, exist_ok=True)
        try:
            result = self.engine.capture({"camera": camera, "output_dir": str(temporary_dir), "frame_count": frame_count})
            meta = json.loads((temporary_dir / "meta.json").read_text(encoding="utf-8"))
            files = {name: _sha256(temporary_dir / name) for name in (
                "color.npy", "depth.npy", "color_intrinsics.npy", "depth_intrinsics.npy"
            )}
            for optional in ("color_distortion.npy", "depth_distortion.npy", "T_depth_color.npy"):
                if (temporary_dir / optional).is_file():
                    files[optional] = _sha256(temporary_dir / optional)
            depth = np.load(temporary_dir / "depth.npy", mmap_mode="r")
            color = np.load(temporary_dir / "color.npy", mmap_mode="r")
            color_intrinsics = np.load(temporary_dir / "color_intrinsics.npy", mmap_mode="r")
            depth_intrinsics = np.load(temporary_dir / "depth_intrinsics.npy", mmap_mode="r")
            if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
                raise RuntimeError("QUALITY_REJECTED: invalid BGR color array")
            if depth.dtype != np.uint16 or depth.ndim != 2:
                raise RuntimeError("QUALITY_REJECTED: invalid raw depth array")
            if color_intrinsics.shape != (3, 3) or depth_intrinsics.shape != (3, 3) or not (
                np.all(np.isfinite(color_intrinsics)) and np.all(np.isfinite(depth_intrinsics))
            ):
                raise RuntimeError("QUALITY_REJECTED: invalid camera intrinsics")
            if not np.isfinite(float(meta["depth_scale_m"])) or float(meta["depth_scale_m"]) <= 0.0:
                raise RuntimeError("QUALITY_REJECTED: invalid depth scale")
            valid_depth_ratio = float(np.count_nonzero(depth) / depth.size) if depth.size else 0.0
            manifest = {
                "schema_version": BROKER_SCHEMA_VERSION,
                "state": "complete",
                "capture_id": capture_id,
                "request": {"request_id": request_id, "consumer": request.get("consumer"),
                            "purpose": request.get("purpose"), "profile_id": profile_id},
                "camera": {"name": camera, "serial": meta["serial"],
                           "profile_fingerprint_sha256": self.engine.profile_fingerprint(profile_id)},
                "frames": {"requested_count": frame_count, "accepted_count": frame_count,
                           "depth_aggregation": "median_uint16" if frame_count > 1 else "single",
                           "color_selection": "last_accepted",
                           "request_flush_count": int(meta.get("femto_request_flush_frames", 0)),
                           "color_device_timestamps_ms": meta.get("femto_color_device_timestamps_ms"),
                           "depth_device_timestamps_ms": meta.get("femto_depth_device_timestamps_ms"),
                           "selected_color_device_timestamp_ms": meta.get("femto_selected_color_device_timestamp_ms"),
                           "selected_depth_device_timestamp_ms": meta.get("femto_selected_depth_device_timestamp_ms")},
                "geometry": {"color_order": "BGR", "depth_dtype": "uint16",
                             "depth_scale_m_per_unit": meta["depth_scale_m"],
                             "color_shape_hw": meta["color_shape_hw"], "depth_shape_hw": meta["depth_shape_hw"],
                             "registration": "depth_in_color_geometry" if not (temporary_dir / "T_depth_color.npy").is_file() else "separate_optical_frames"},
                "timing": {"capture_started_unix_ns": int(meta["capture_started_unix_s"] * 1e9),
                           "capture_finished_unix_ns": int(meta["capture_finished_unix_s"] * 1e9)},
                "quality": {"valid_depth_ratio": valid_depth_ratio,
                            "checks": {"intrinsics_finite": True, "depth_nonempty": bool(np.count_nonzero(depth))}},
                "files": files,
            }
            manifest_path = temporary_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.rename(temporary_dir, final_dir)
            with self.capture_root_lock:
                self._prune_capture_bundles(final_dir)
            completed_manifest = final_dir / "manifest.json"
            return {"schema_version": BROKER_SCHEMA_VERSION, "request_id": request_id,
                    "capture_id": capture_id, "manifest_path": str(completed_manifest),
                    "sha256": _sha256(completed_manifest), "capture": result}
        except Exception:
            # A partial directory is never a valid result.  Keep it for diagnosis;
            # clients only accept the atomically renamed final directory.
            raise
    def handle_request(self, request: dict) -> dict:
        if request.get("op") == "status":
            return {"schema_version": BROKER_SCHEMA_VERSION, "allow_root": str(self.allow_root),
                    "capture_root": str(self.capture_root), "camera_config": self.engine.status(),
                    "profiles": self.engine.profile_specs()}
        if request.get("op") == "probe":
            if int(request.get("schema_version", -1)) != BROKER_SCHEMA_VERSION:
                raise RuntimeError(f"unsupported schema_version: {request.get('schema_version')!r}")
            return self.engine.probe(
                str(request.get("profile_id") or ""),
                require_eef_pose=bool(request.get("require_eef_pose", False)),
            )
        if request.get("op") == "capture_bundle":
            return self._capture_bundle(request)
        requests = [request]
        if request.get("op") == "capture_pair":
            requests = [request["d405"], request["femto_bolt"]]
        for item in requests:
            output = Path(item["output_dir"]).resolve()
            if self.allow_root not in (output, *output.parents):
                raise RuntimeError(f"output_dir must stay under camera allow-root: {self.allow_root}")
        if request.get("op") == "capture_pair":
            max_start_skew_s = float(request.get("max_start_skew_s", .2))
            if not np.isfinite(max_start_skew_s) or max_start_skew_s < 0:
                raise RuntimeError("max_start_skew_s must be finite and non-negative")
            return self.engine.capture_pair(request["d405"], request["femto_bolt"], max_start_skew_s)
        return self.engine.capture(request)


def main() -> None:
    args = parse_args()
    if (args.femto_warmup_frames < 0 or args.femto_request_flush_frames < 0
            or args.femto_frame_read_retries < 0
            or not 0.0 <= args.min_valid_depth_ratio <= 1.0 or args.max_capture_bundles < 1):
        raise ValueError("invalid Camera Broker warmup, depth-ratio, or retention configuration")
    allow_root = Path(args.allow_root).resolve(); allow_root.mkdir(parents=True, exist_ok=True)
    capture_root = Path(args.capture_root).resolve() if args.capture_root else allow_root / "captures"
    socket_path = Path(args.socket_path)
    if socket_path.exists() or socket_path.is_symlink():
        raise RuntimeError(f"refusing to replace existing camera socket: {socket_path}; stop the old service or remove a confirmed stale socket")
    engine = CameraEngine(args)
    try:
        with Server(str(socket_path), engine, allow_root, capture_root) as server:
            os.chmod(socket_path, 0o600)
            print(f"-> camera service listening on {socket_path}", flush=True)
            server.serve_forever()
    finally:
        engine.close()


if __name__ == "__main__": main()
