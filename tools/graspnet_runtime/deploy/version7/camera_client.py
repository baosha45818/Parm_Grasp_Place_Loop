"""Synchronous Unix-socket client shared by Version7 camera requests."""
from __future__ import annotations
import json
import socket

def _request(socket_path: str, request: dict, timeout_s: float = 30.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s); conn.connect(socket_path)
        conn.sendall((json.dumps(request) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            part = conn.recv(65536)
            if not part: raise RuntimeError("camera service closed response early")
            data += part
    response = json.loads(data.decode())
    if not response.get("ok"): raise RuntimeError("camera service failed: " + str(response.get("error")))
    return response["result"]


def capture(socket_path: str, request: dict, timeout_s: float = 30.0) -> dict:
    return _request(socket_path, request, timeout_s)


def status(socket_path: str, timeout_s: float = 5.0) -> dict:
    return _request(socket_path, {"op": "status"}, timeout_s)


def capture_pair(socket_path: str, d405_request: dict, femto_request: dict, timeout_s: float = 30.0,
                 max_start_skew_s: float = .2) -> dict:
    return _request(socket_path, {"op": "capture_pair", "d405": d405_request, "femto_bolt": femto_request,
                                  "max_start_skew_s": max_start_skew_s}, timeout_s)
