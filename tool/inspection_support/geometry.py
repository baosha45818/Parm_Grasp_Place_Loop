"""Small SE(3) helpers used by the point-cloud pose pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_T_json(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    return np.asarray(data["T"], dtype=np.float64)


def transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def invert_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def pose6_to_T(pose6) -> np.ndarray:
    p = np.asarray(pose6, dtype=np.float64).reshape(6)
    return make_T(Rotation.from_euler("xyz", p[3:]).as_matrix(), p[:3])


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:] = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return pose


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(3)
    bb = np.asarray(b, dtype=np.float64).reshape(3)
    aa /= max(np.linalg.norm(aa), 1e-12)
    bb /= max(np.linalg.norm(bb), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0))))


def point_to_line_distance_m(point_xyz: np.ndarray,
                             line_point_xyz: np.ndarray,
                             line_dir_xyz: np.ndarray) -> float:
    p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    p0 = np.asarray(line_point_xyz, dtype=np.float64).reshape(3)
    d = np.asarray(line_dir_xyz, dtype=np.float64).reshape(3)
    d /= max(np.linalg.norm(d), 1e-12)
    return float(np.linalg.norm(np.cross(p - p0, d)))


def rotation_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape(3)
    bb = np.asarray(b, dtype=np.float64).reshape(3)
    aa /= max(np.linalg.norm(aa), 1e-12)
    bb /= max(np.linalg.norm(bb), 1e-12)
    v = np.cross(aa, bb)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(trial, aa)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(aa, trial)
        axis /= max(np.linalg.norm(axis), 1e-12)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rotation_from_z_axis_and_x_hint(z_axis: np.ndarray,
                                    x_hint: np.ndarray) -> np.ndarray:
    """Build a right-handed rotation whose local z matches z_axis.

    The remaining in-plane degree of freedom is resolved by projecting x_hint
    onto the plane orthogonal to z_axis. This removes the ambiguous free yaw
    that appears when we only constrain a hole axis.
    """
    z = np.asarray(z_axis, dtype=np.float64).reshape(3)
    xh = np.asarray(x_hint, dtype=np.float64).reshape(3)
    z /= max(np.linalg.norm(z), 1e-12)

    x = xh - z * float(np.dot(xh, z))
    if np.linalg.norm(x) < 1e-9:
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(trial, z)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = trial - z * float(np.dot(trial, z))
    x /= max(np.linalg.norm(x), 1e-12)

    y = np.cross(z, x)
    y /= max(np.linalg.norm(y), 1e-12)
    x = np.cross(y, z)
    x /= max(np.linalg.norm(x), 1e-12)

    R = np.stack([x, y, z], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1.0
    return R


def pose_gap_mm_deg(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    trans_mm = float(np.linalg.norm((T_b[:3, 3] - T_a[:3, 3]) * 1000.0))
    rot_a = Rotation.from_matrix(T_a[:3, :3])
    rot_b = Rotation.from_matrix(T_b[:3, :3])
    rot_deg = float(np.degrees((rot_b * rot_a.inv()).magnitude()))
    return trans_mm, rot_deg


def load_peg_top_world(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    peg = data["peg"]
    if "peg_top_base_xyz_m" in peg:
        return np.asarray(peg["peg_top_base_xyz_m"], dtype=np.float64)
    gl = float(peg.get("calibration_gripper_length_m", 0.17))
    return np.array(
        [
            float(peg["x_base_m"]),
            float(peg["y_base_m"]),
            float(peg["peg_top_flange_z_m"]) - gl,
        ],
        dtype=np.float64,
    )
