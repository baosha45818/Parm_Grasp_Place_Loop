"""读取 ASCII 或二进制 STL，并按尺寸自动归一到米。"""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def load_stl_triangles_m(stl_path: str | Path) -> np.ndarray:
    """返回形状为 ``(N, 3, 3)`` 的米制三角面片。"""
    path = Path(stl_path)
    blob = path.read_bytes()
    triangles: np.ndarray | None = None
    if len(blob) >= 84:
        triangle_count = struct.unpack("<I", blob[80:84])[0]
        if 84 + triangle_count * 50 == len(blob):
            records = np.frombuffer(
                blob, dtype=np.dtype([
                    ("normal", "<f4", 3), ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]), offset=84, count=triangle_count,
            )
            triangles = np.asarray(records["vertices"], dtype=np.float64)
    if triangles is None:
        vertices = []
        for line in blob.decode("utf-8", errors="ignore").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(value) for value in fields[1:]])
        if not vertices or len(vertices) % 3:
            raise RuntimeError(f"无法解析 STL: {path}")
        triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    extent = float(np.ptp(triangles.reshape(-1, 3), axis=0).max())
    if extent > 2.0:
        triangles = triangles / 1000.0
    if triangles.size == 0 or not np.all(np.isfinite(triangles)):
        raise RuntimeError(f"STL 没有有效三角面: {path}")
    return triangles
