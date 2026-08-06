from __future__ import annotations
import json, socket
def estimate(socket_path: str, request: dict, timeout_s: float = 120.0) -> dict:
    return _request(socket_path, request, timeout_s)


def status(socket_path: str, timeout_s: float = 5.0) -> dict:
    return _request(socket_path, {"op": "status"}, timeout_s)


def _request(socket_path: str, request: dict, timeout_s: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s); conn.connect(socket_path); conn.sendall((json.dumps(request)+"\n").encode()); data=b""
        while not data.endswith(b"\n"):
            part=conn.recv(65536)
            if not part: raise RuntimeError("pose service closed response early")
            data += part
    response=json.loads(data.decode())
    if not response.get("ok"): raise RuntimeError("pose service failed: "+str(response.get("error")))
    return response["result"]
