"""Minimal SAM2 prompt client shared by the Left Pick perception pipeline."""
from __future__ import annotations

import http.client
import io
import socket
from urllib.parse import urlparse

import numpy as np


def prompt_segment(
    image: np.ndarray,
    box: list[int],
    positives: list[list[int]],
    negatives: list[list[int]],
    sam2_url: str,
) -> np.ndarray:
    """Request a prompt-segmentation mask from the local SAM2 service."""
    if not positives:
        raise ValueError("SAM2 segmentation requires at least one positive point")
    points = np.asarray(positives + negatives, dtype=np.int32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    payload = io.BytesIO()
    np.savez_compressed(payload, image_bgr=image, points=points, labels=labels, box=np.asarray(box))

    endpoint = urlparse(sam2_url)
    if endpoint.scheme == "unix":
        if not endpoint.path:
            raise ValueError(f"SAM2 Unix socket path is missing: {sam2_url!r}")

        class UnixConnection(http.client.HTTPConnection):
            def connect(self) -> None:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(endpoint.path)

        connection = UnixConnection("sam2-service", timeout=90.0)
        request_path = "/v1/prompt-segment-sam2"
    elif endpoint.scheme in {"http", "https"} and endpoint.hostname:
        connection_type = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(endpoint.hostname, port=endpoint.port, timeout=90.0)
        request_path = (endpoint.path or "/") + (f"?{endpoint.query}" if endpoint.query else "")
    else:
        raise ValueError(f"SAM2 URL must use unix:// or http(s)://: {sam2_url!r}")
    try:
        connection.request(
            "POST", request_path, body=payload.getvalue(),
            headers={"Content-Type": "application/x-sam2-prompt-npz"},
        )
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM2 returned HTTP {response.status}: {body.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return (archive["mask"] > 0).astype(np.uint8) * 255
