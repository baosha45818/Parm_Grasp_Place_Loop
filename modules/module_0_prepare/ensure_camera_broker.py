#!/usr/bin/env python3
"""Start or validate the singleton camera Broker for a loop run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
V7_DIR = ROOT / "tools/graspnet_runtime/deploy/version7"
SOCKET = os.environ.get("CAMERA_BROKER_SOCKET", "/tmp/graspnet-v7-camera.sock")


def _request(payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2.0)
        connection.connect(SOCKET)
        connection.sendall((json.dumps(payload) + "\n").encode())
        response = b""
        while not response.endswith(b"\n"):
            block = connection.recv(65536)
            if not block:
                raise RuntimeError("Camera Broker closed health-check response early")
            response += block
    payload = json.loads(response.decode())
    if not payload.get("ok"):
        raise RuntimeError("Camera Broker health check failed: " + str(payload.get("error")))
    return payload["result"]


def _status() -> dict:
    return _request({"op": "status"})


def _remove_confirmed_stale_socket() -> None:
    path = Path(SOCKET)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not stat.S_ISSOCK(path.lstat().st_mode):
        raise RuntimeError(f"refusing to remove non-socket Camera Broker path: {path}")
    listeners = subprocess.run(
        ["ss", "-xlH"], check=False, capture_output=True, text=True
    )
    if listeners.returncode == 0 and str(path) in listeners.stdout:
        raise RuntimeError(f"Camera Broker socket has a live listener but did not answer health check: {path}")
    path.unlink()
    print(f"[module-0] removed confirmed stale Camera Broker socket: {path}", flush=True)


def _restart_owned_broker() -> None:
    """Restart only the project-owned Broker before any robot task is scheduled."""
    pid_path = V7_DIR / "logs" / "camera_service.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Camera Broker failed the EEF preflight and cannot be safely restarted "
            f"because its owned PID cannot be verified: {pid_path}"
        ) from exc
    if "camera_service.py" not in command:
        raise RuntimeError(
            "Camera Broker failed the EEF preflight but the recorded PID does not belong "
            f"to camera_service.py: pid={pid}"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            _status()
        except OSError:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"Camera Broker pid={pid} did not stop within 10s")
    _remove_confirmed_stale_socket()
    print(f"[module-0] restarted Camera Broker after failed D405 EEF preflight: pid={pid}", flush=True)


def _start_broker(
    timeout_s: float, *, require_femto: bool, require_d405_eef_pose: bool,
    expected_d405_serial: str | None, expected_d405_fps: int | None,
) -> dict:
    environment = dict(os.environ)
    # Module 1's direct-Planner ICP path owns only D405.  Do not overwrite
    # that request with the legacy all-camera Broker default.
    environment["GRASPNET_V7_DISABLE_FEMTO"] = "0" if require_femto else "1"
    environment["GRASPNET_V7_DIRECT_PLANNER"] = "0" if require_d405_eef_pose else "1"
    if expected_d405_serial is not None:
        environment["GRASPNET_V7_D405_SERIAL"] = str(expected_d405_serial)
    if expected_d405_fps is not None:
        environment["GRASPNET_V7_D405_FPS"] = str(int(expected_d405_fps))
    environment["GRASPNET_V7_CAMERA_SOCKET"] = SOCKET
    subprocess.run(["bash", str(V7_DIR / "start_camera_service.sh")], check=True, env=environment)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return _status()
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Camera Broker did not become ready at {SOCKET} within {timeout_s}s")
            time.sleep(0.25)


def _validate_broker(
    status: dict, *, require_d405_eef_pose: bool = True, expected_d405_serial: str | None = None,
    expected_d405_fps: int | None = None, require_femto: bool = True,
) -> None:
    """Validate the requested camera set and D405-to-robot dependency."""
    if status.get("schema_version") != 1:
        raise RuntimeError("Camera Broker ABI mismatch; restart it with the current project version")
    cameras = status.get("camera_config") or {}
    if not require_femto and "femto_bolt" in cameras:
        raise RuntimeError(
            "Camera Broker has Femto enabled, but this D405-only flow must not "
            "occupy the chest camera"
        )
    required_cameras = {"d405"}
    if require_femto:
        required_cameras.add("femto_bolt")
    missing_cameras = sorted(required_cameras.difference(cameras))
    if missing_cameras:
        raise RuntimeError(f"Camera Broker is missing required cameras: {missing_cameras}")
    profiles = status.get("profiles") or {}
    if expected_d405_serial is not None:
        actual = str((cameras.get("d405") or {}).get("serial", ""))
        if actual != str(expected_d405_serial):
            raise RuntimeError(
                f"Camera Broker D405 serial is {actual!r}; expected {expected_d405_serial!r}"
            )
    if expected_d405_fps is not None:
        actual_fps = (cameras.get("d405") or {}).get("fps")
        if actual_fps != int(expected_d405_fps):
            raise RuntimeError(
                f"Camera Broker D405 FPS is {actual_fps!r}; expected {int(expected_d405_fps)!r}"
            )
    profile_ids = ["graspnet_d405_v1"]
    if require_femto:
        profile_ids.extend(("graspnet_femto_v1", "inspect_femto_v1"))
    for profile_id in profile_ids:
        if profile_id not in profiles:
            raise RuntimeError(f"Camera Broker lacks required profile: {profile_id}")
        result = _request({
            "op": "probe", "schema_version": 1, "profile_id": profile_id,
            "require_eef_pose": profile_id == "graspnet_d405_v1" and require_d405_eef_pose,
        })
        if profile_id == "graspnet_d405_v1" and require_d405_eef_pose and "T_base_eef" not in result:
            raise RuntimeError("Camera Broker D405 probe did not return T_base_eef")


def ensure_camera_broker(
    timeout_s: float = 75.0, *, require_d405_eef_pose: bool = True,
    expected_d405_serial: str | None = None, expected_d405_fps: int | None = None,
    require_femto: bool = True,
) -> dict:
    try:
        status = _status()
        started = False
    except OSError:
        _remove_confirmed_stale_socket()
        status = _start_broker(
            timeout_s,
            require_femto=require_femto,
            require_d405_eef_pose=require_d405_eef_pose,
            expected_d405_serial=expected_d405_serial,
            expected_d405_fps=expected_d405_fps,
        )
        started = True
    try:
        _validate_broker(
            status, require_d405_eef_pose=require_d405_eef_pose,
            expected_d405_serial=expected_d405_serial,
            expected_d405_fps=expected_d405_fps,
            require_femto=require_femto,
        )
    except RuntimeError:
        if started:
            raise
        # A live old Broker may not implement the EEF probe, while a Broker
        # created before a driver restart can retain a stale SDK subscriber.
        _restart_owned_broker()
        status = _start_broker(
            timeout_s,
            require_femto=require_femto,
            require_d405_eef_pose=require_d405_eef_pose,
            expected_d405_serial=expected_d405_serial,
            expected_d405_fps=expected_d405_fps,
        )
        started = True
        _validate_broker(
            status, require_d405_eef_pose=require_d405_eef_pose,
            expected_d405_serial=expected_d405_serial,
            expected_d405_fps=expected_d405_fps,
            require_femto=require_femto,
        )
    print(f"[module-0] Camera Broker {'started' if started else 'reused'}: {SOCKET}", flush=True)
    return status


def require_camera_broker() -> dict:
    """Validate an already-running Broker without starting or restarting it."""
    status = _status()
    _validate_broker(status)
    print(f"[module-0] Camera Broker prestarted and ready: {SOCKET}", flush=True)
    return status


def stop_camera_broker(timeout_s: float = 10.0) -> bool:
    """Stop only the project-owned broker and release its camera devices for GCS."""
    pid_path = V7_DIR / "logs" / "camera_service.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except FileNotFoundError:
        # The broker may already have exited; remove only a confirmed stale socket.
        _remove_confirmed_stale_socket()
        pid_path.unlink(missing_ok=True)
        return False
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot validate Camera Broker PID file: {pid_path}") from exc
    if "camera_service.py" not in command:
        raise RuntimeError(f"refusing to stop non-Camera-Broker pid={pid}")

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while Path(f"/proc/{pid}").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Camera Broker pid={pid} did not stop within {timeout_s}s")
        time.sleep(0.1)
    _remove_confirmed_stale_socket()
    pid_path.unlink(missing_ok=True)
    print(f"[module-0] Camera Broker stopped; camera devices released for GCS: pid={pid}", flush=True)
    return True


if __name__ == "__main__":
    ensure_camera_broker()
