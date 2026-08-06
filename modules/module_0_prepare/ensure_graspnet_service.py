#!/usr/bin/env python3
"""Module 0 ownership and readiness checks for persistent Version7 GraspNet."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
V7_DIR = ROOT / "third_party/graspnet/deploy/version7"
SOCKET = Path(os.environ.get("GRASPNET_V7_SOCKET", "/tmp/graspnet-v7.sock"))
JOBS_ROOT = Path(os.environ.get("GRASPNET_V7_JOBS_ROOT", str(V7_DIR / "jobs"))).resolve()
NUM_VIEW = int(os.environ.get("GRASPNET_V7_NUM_VIEW", "300"))
PYTHON = ROOT / "third_party/graspnet/.venv/bin/python"
SERVICE = V7_DIR / "graspnet_service.py"
CHECKPOINT = ROOT / "third_party/graspnet/checkpoint/checkpoint-rs.tar"
RUNTIME_DIR = V7_DIR / "runtime"
PID_FILE = Path(os.environ.get("GRASPNET_SERVICE_PID_FILE", str(RUNTIME_DIR / "graspnet-service.pid")))
LOG = Path(os.environ.get("GRASPNET_SERVICE_LOG", str(RUNTIME_DIR / "graspnet-service.log")))


def _status() -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5.0)
        connection.connect(str(SOCKET))
        connection.sendall(b'{"op":"status"}\n')
        response = b""
        while not response.endswith(b"\n"):
            block = connection.recv(65536)
            if not block:
                raise RuntimeError("GraspNet service closed health-check response early")
            response += block
    payload = json.loads(response.decode())
    if not payload.get("ok"):
        raise RuntimeError("GraspNet health check failed: " + str(payload.get("error")))
    return payload["result"]


def _remove_confirmed_stale_socket() -> None:
    if not SOCKET.exists() and not SOCKET.is_symlink():
        return
    if SOCKET.is_symlink() or not stat.S_ISSOCK(SOCKET.lstat().st_mode):
        raise RuntimeError(f"refusing to remove non-socket GraspNet path: {SOCKET}")
    listeners = subprocess.run(["ss", "-xlH"], check=False, capture_output=True, text=True)
    if listeners.returncode == 0 and str(SOCKET) in listeners.stdout:
        raise RuntimeError(f"GraspNet socket has a live listener but fails health checks: {SOCKET}")
    SOCKET.unlink()
    print(f"[module-0] removed confirmed stale GraspNet socket: {SOCKET}", flush=True)


def require_graspnet_service() -> dict:
    return _validate(_status())


def _validate(payload: dict) -> dict:
    if Path(str(payload.get("allow_root", ""))).resolve() != JOBS_ROOT:
        raise RuntimeError(f"GraspNet allow-root mismatch: {payload.get('allow_root')} != {JOBS_ROOT}")
    if int(payload.get("num_view", -1)) != NUM_VIEW:
        raise RuntimeError(f"GraspNet num_view mismatch: {payload.get('num_view')} != {NUM_VIEW}")
    return payload


def ensure_graspnet_service(timeout_s: float = 300.0) -> dict:
    try:
        payload = _validate(_status())
        started = False
    except OSError:
        _remove_confirmed_stale_socket()
        missing = [path for path in (PYTHON, SERVICE, CHECKPOINT) if not path.is_file()]
        if missing:
            raise RuntimeError("GraspNet startup dependency missing: " + ", ".join(map(str, missing)))
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        with LOG.open("ab") as log:
            subprocess.Popen(
                [str(PYTHON), str(SERVICE), "--checkpoint-path", str(CHECKPOINT),
                 "--num-view", str(NUM_VIEW), "--socket-path", str(SOCKET),
                 "--allow-root", str(JOBS_ROOT), "--pid-file", str(PID_FILE)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment,
            )
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                payload = _validate(_status())
                started = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"GraspNet did not become ready within {timeout_s}s; inspect {LOG}")
                time.sleep(1.0)
    print(f"[module-0] GraspNet {'started' if started else 'reused'}: {SOCKET}", flush=True)
    return payload


if __name__ == "__main__":
    ensure_graspnet_service()
