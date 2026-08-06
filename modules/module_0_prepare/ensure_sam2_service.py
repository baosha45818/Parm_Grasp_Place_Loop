#!/usr/bin/env python3
"""Module 0 ownership and readiness checks for the resident SAM2 service."""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import socket
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOCKET = Path(os.environ.get(
    "SAM2_SERVICE_SOCKET",
    "/home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service/runtime/sam2-service.sock",
))
START_SCRIPT = ROOT / "tool/sam2-service/start_uds_server.sh"
LOG = ROOT / "output/services/sam2-service.log"
RECOVERY_LOCK = SOCKET.with_name(f"{SOCKET.name}.recovery.lock")


def sam2_url() -> str:
    return f"unix://{SOCKET}"


def _health(socket_path: Path = SOCKET) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5.0)
        connection.connect(str(socket_path))
        connection.sendall(b"GET /healthz HTTP/1.1\r\nHost: sam2-service\r\nConnection: close\r\n\r\n")
        response = b""
        while True:
            block = connection.recv(65536)
            if not block:
                break
            response += block
    header, separator, body = response.partition(b"\r\n\r\n")
    if not separator or not header.startswith(b"HTTP/1.1 200"):
        raise RuntimeError(f"SAM2 health check returned invalid HTTP response: {header[:120]!r}")
    payload = json.loads(body.decode())
    if payload.get("service") != "sam2-service" or payload.get("started") is not True:
        raise RuntimeError(f"SAM2 service is not ready: {payload}")
    return payload


def _remove_confirmed_stale_socket() -> None:
    if not SOCKET.exists() and not SOCKET.is_symlink():
        return
    try:
        socket_mode = SOCKET.lstat().st_mode
    except FileNotFoundError:
        return
    if SOCKET.is_symlink() or not stat.S_ISSOCK(socket_mode):
        raise RuntimeError(f"refusing to remove non-socket SAM2 path: {SOCKET}")
    listeners = subprocess.run(["ss", "-xlH"], check=False, capture_output=True, text=True)
    if listeners.returncode == 0 and str(SOCKET) in listeners.stdout:
        raise RuntimeError(f"SAM2 socket has a live listener but fails health checks: {SOCKET}")
    try:
        SOCKET.unlink()
    except FileNotFoundError:
        return
    print(f"[module-0] removed confirmed stale SAM2 socket: {SOCKET}", flush=True)


def require_sam2_service() -> dict:
    """Return a healthy service; managed Loop may recover a missing process."""
    try:
        return _health()
    except OSError as exc:
        if os.environ.get("PARM_SAM2_AUTO_RECOVER") != "1":
            raise
        print(
            "[module-0] SAM2 health endpoint disappeared; "
            f"recovering the resident service: {exc}",
            flush=True,
        )
        return ensure_sam2_service()


def ensure_sam2_service(timeout_s: float = 180.0) -> dict:
    try:
        payload = _health()
        started = False
    except OSError:
        # Multiple business workers may notice the same lost socket.  Serialize
        # stale-socket removal and startup, then recheck after acquiring the
        # lock because another worker may already have completed recovery.
        SOCKET.parent.mkdir(parents=True, exist_ok=True)
        with RECOVERY_LOCK.open("a+", encoding="utf-8") as recovery_lock:
            fcntl.flock(recovery_lock.fileno(), fcntl.LOCK_EX)
            try:
                payload = _health()
                started = False
            except OSError:
                _remove_confirmed_stale_socket()
                if not START_SCRIPT.is_file():
                    raise RuntimeError(
                        f"SAM2 start script is unavailable: {START_SCRIPT}"
                    )
                LOG.parent.mkdir(parents=True, exist_ok=True)
                service_env = os.environ.copy()
                service_env["SAM2_SERVICE_SOCKET"] = str(SOCKET)
                with LOG.open("ab") as log:
                    subprocess.Popen(
                        ["/bin/bash", str(START_SCRIPT)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=service_env,
                    )
                deadline = time.monotonic() + timeout_s
                while True:
                    try:
                        payload = _health()
                        started = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "SAM2 did not become ready within "
                                f"{timeout_s}s; inspect {LOG}"
                            )
                        time.sleep(0.5)
    backend = payload.get("backend")
    if backend != "efficientvit":
        raise RuntimeError(f"SAM2 backend mismatch: expected efficientvit, got {backend!r}")
    print(f"[module-0] SAM2 {'started' if started else 'reused'}: {SOCKET}", flush=True)
    return payload


if __name__ == "__main__":
    ensure_sam2_service()
