"""Small synchronous client for the Version7 local GraspNet service."""
from __future__ import annotations

import json
import socket


def infer(socket_path: str, request: dict, timeout_s: float = 120.0) -> dict:
    return _request(socket_path, request, timeout_s)


def infer_batch(socket_path: str, requests: list[dict], timeout_s: float = 120.0) -> list[dict]:
    result = _request(socket_path, {"requests": requests}, timeout_s)
    # 服务端为单请求保留了历史响应格式：直接返回一条结果；只有真正批量时
    # 才包在 {"batch": [...]} 中。D405-only 正好会走单请求路径。
    if "batch" in result:
        batch = result["batch"]
        if not isinstance(batch, list):
            raise RuntimeError(f"GraspNet service returned invalid batch: {result!r}")
        return batch
    if len(requests) == 1:
        return [result]
    raise RuntimeError(f"GraspNet service returned no batch for {len(requests)} requests: {result!r}")


def status(socket_path: str, timeout_s: float = 5.0) -> dict:
    return _request(socket_path, {"op": "status"}, timeout_s)


def _request(socket_path: str, request: dict, timeout_s: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(socket_path)
        conn.sendall((json.dumps(request) + "\n").encode("utf-8"))
        payload = b""
        while not payload.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                raise RuntimeError("GraspNet service closed the response early")
            payload += chunk
    response = json.loads(payload.decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError("GraspNet service failed: " + str(response.get("error")))
    return response["result"]
