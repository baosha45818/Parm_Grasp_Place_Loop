"""Load the workpiece STL and sample a CAD point cloud without extra deps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np

THIS_ROOT = Path(__file__).resolve().parents[2]
PLACE_ROOT = THIS_ROOT
# 工件 CAD 与公共工具均已从旧 tool/ 迁至 tools/。
WORKPIECE_DIR = THIS_ROOT / "tools"


@dataclass
class HoleInfo:
    name: str
    center_stl_m: np.ndarray
    axis_stl: np.ndarray
    radius_m: float


@dataclass
class WorkpieceModel:
    info_path: Path
    stl_path: Path
    triangles_m: np.ndarray
    centroid_m: np.ndarray
    holes: dict[str, HoleInfo]

    @property
    def vertices_m(self) -> np.ndarray:
        return self.triangles_m.reshape(-1, 3)

    @property
    def bbox_min_m(self) -> np.ndarray:
        return self.vertices_m.min(axis=0)

    @property
    def bbox_max_m(self) -> np.ndarray:
        return self.vertices_m.max(axis=0)


def _resolve_stl_path(info_path: Path, stl_value: str) -> Path:
    raw = Path(stl_value)
    if raw.is_absolute():
        return raw
    candidates = [
        info_path.parent / raw,
        WORKPIECE_DIR / raw,
        WORKPIECE_DIR / raw.name,
        PLACE_ROOT / raw,
        THIS_ROOT / raw,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def load_workpiece_model(info_path: str | Path) -> WorkpieceModel:
    info_path = Path(info_path)
    data = json.loads(info_path.read_text())
    stl_path = _resolve_stl_path(info_path, data["stl_path"])
    triangles = load_stl_triangles_m(stl_path)
    holes: dict[str, HoleInfo] = {}
    for h in data.get("holes", []):
        axis = np.asarray(h["axis_stl"], dtype=np.float64)
        axis /= max(np.linalg.norm(axis), 1e-12)
        holes[str(h["name"])] = HoleInfo(
            name=str(h["name"]),
            center_stl_m=np.asarray(h["center_stl_m"], dtype=np.float64),
            axis_stl=axis,
            radius_m=float(h.get("radius_m", 0.0)),
        )
    return WorkpieceModel(
        info_path=info_path,
        stl_path=stl_path,
        triangles_m=triangles,
        centroid_m=np.asarray(data["centroid_m"], dtype=np.float64),
        holes=holes,
    )


def load_stl_triangles_m(stl_path: str | Path) -> np.ndarray:
    """Return triangles as (N,3,3) meters.

    Some CAD exports are millimeters, but this project's current STL is already
    in meters. Auto-detect from the bounding-box size so we do not silently scale
    the model by 1000x.
    """
    stl_path = Path(stl_path)
    blob = stl_path.read_bytes()
    triangles_mm: np.ndarray
    if len(blob) >= 84:
        n_tri = struct.unpack("<I", blob[80:84])[0]
        expected = 84 + n_tri * 50
        if expected == len(blob):
            arr = np.empty((n_tri, 3, 3), dtype=np.float64)
            off = 84
            for i in range(n_tri):
                vals = struct.unpack("<12fH", blob[off:off + 50])
                arr[i] = np.asarray(vals[3:12], dtype=np.float64).reshape(3, 3)
                off += 50
            triangles_mm = arr
        else:
            triangles_mm = _load_ascii_stl_triangles_mm(blob.decode("utf-8", errors="ignore"))
    else:
        triangles_mm = _load_ascii_stl_triangles_mm(blob.decode("utf-8", errors="ignore"))
    extent = triangles_mm.reshape(-1, 3).ptp(axis=0)
    # A hand-size workpiece in meters is < 1.0; in millimeters it is tens.
    if float(np.max(extent)) > 1.0:
        return triangles_mm * 1e-3
    return triangles_mm


def triangles_in_object_frame(
    triangles_stl_m: np.ndarray,
    *,
    R_object_stl: object | None = None,
    object_origin_stl_m: object | None = None,
) -> np.ndarray:
    """Express STL triangles in a configured right-handed object frame.

    ``R_object_stl`` maps column vectors from the raw STL frame to the
    application object frame.  The optional origin is expressed in the raw
    STL frame.  Omitting both values preserves the historical behaviour.
    """
    triangles = np.asarray(triangles_stl_m, dtype=np.float64).reshape(-1, 3, 3)
    rotation = (
        np.eye(3, dtype=np.float64)
        if R_object_stl is None
        else np.asarray(R_object_stl, dtype=np.float64)
    )
    origin = (
        np.zeros(3, dtype=np.float64)
        if object_origin_stl_m is None
        else np.asarray(object_origin_stl_m, dtype=np.float64)
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("R_object_stl must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("R_object_stl must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("R_object_stl must be right-handed (determinant +1)")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("object_origin_stl_m must contain three finite values")
    # Points are stored as row vectors, hence the transpose here.
    return (triangles - origin) @ rotation.T


def _load_ascii_stl_triangles_mm(text: str) -> np.ndarray:
    verts: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(verts) % 3 != 0 or not verts:
        raise RuntimeError("STL has no valid triangles")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def sample_surface_points(triangles_m: np.ndarray,
                          n_points: int = 12000,
                          seed: int = 0) -> np.ndarray:
    tris = np.asarray(triangles_m, dtype=np.float64).reshape(-1, 3, 3)
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    probs = areas / max(float(areas.sum()), 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tris), size=int(n_points), replace=True, p=probs)
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    s1 = np.sqrt(r1)
    a = 1.0 - s1
    b = s1 * (1.0 - r2)
    c = s1 * r2
    pts = (
        a[:, None] * tris[idx, 0]
        + b[:, None] * tris[idx, 1]
        + c[:, None] * tris[idx, 2]
    )
    return pts.astype(np.float64)


def load_cad_pointcloud_template(
    template_path: str | Path,
    *,
    expected_icp_points: int = 14000,
    expected_render_points: int = 90000,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a persistent CAD surface point-cloud template.

    Both returned arrays are expressed in the raw STL frame and use metres.
    """
    template_path = Path(template_path)
    with np.load(template_path, allow_pickle=False) as archive:
        schema_version = int(np.asarray(archive["schema_version"]).item())
        units = str(np.asarray(archive["units"]).item())
        icp_points = np.asarray(archive["icp_points_m"], dtype=np.float64)
        render_points = np.asarray(archive["render_points_m"], dtype=np.float64)
    if schema_version != 1:
        raise RuntimeError(
            f"unsupported CAD point-cloud template schema {schema_version}: {template_path}"
        )
    if units != "m":
        raise RuntimeError(f"CAD point-cloud template units must be 'm': {template_path}")
    expected_shapes = (
        (icp_points, (int(expected_icp_points), 3), "icp_points_m"),
        (render_points, (int(expected_render_points), 3), "render_points_m"),
    )
    for points, expected_shape, name in expected_shapes:
        if points.shape != expected_shape or not np.isfinite(points).all():
            raise RuntimeError(
                f"invalid {name} in {template_path}: expected finite {expected_shape}, "
                f"got {points.shape}"
            )
    return icp_points, render_points
