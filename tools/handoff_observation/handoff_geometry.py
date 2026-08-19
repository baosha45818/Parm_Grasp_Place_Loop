"""Rigid-transform helpers for the reversed-workpiece direct handoff.

The convention used throughout this module is ``p_A = T_A_B @ p_B``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from loop_types import ObjectPose


def validate_transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def object_pose_to_transform(pose: dict[str, Any], name: str) -> np.ndarray:
    position = np.asarray(pose.get("position_m", pose.get("position")), dtype=float)
    quaternion = np.asarray(
        pose.get("orientation_xyzw", pose.get("orientation")), dtype=float
    )
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError(f"{name} must contain xyz position and xyzw quaternion")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError(f"{name} quaternion has zero norm")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    matrix[:3, 3] = position
    return matrix


def planner_tf_target_source(
    response: dict[str, Any], source_frame: str, target_frame: str
) -> np.ndarray:
    """Convert the deployed Planner TF response to ``T_target_source``.

    The deployed service serializes the opposite transform in its payload.
    This behaviour is also handled by the existing reverse-regrasp script.
    Frame labels are checked before applying the required inverse.
    """
    data = (response or {}).get("data") or {}
    if data.get("source_frame") != source_frame or data.get("target_frame") != target_frame:
        raise RuntimeError(
            f"unexpected Planner TF response for {source_frame}->{target_frame}: {data}"
        )
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    try:
        xyz = [float(translation[key]) for key in "xyz"]
        xyzw = [float(rotation[key]) for key in "xyzw"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Planner TF response: {response}") from exc
    serialized = np.eye(4)
    serialized[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    serialized[:3, 3] = xyz
    return validate_transform(np.linalg.inv(serialized), f"T_{target_frame}_{source_frame}")


def compute_handoff_targets(
    T_base_object_current: np.ndarray,
    T_base_left_tcp_current: np.ndarray,
    T_base_object_handoff: np.ndarray,
    T_right_tcp_object: np.ndarray,
    right_pregrasp_offset_tcp_m: list[float],
) -> dict[str, np.ndarray]:
    """Compute the dynamic left target and fixed right grasp/pregrasp targets."""
    T_base_object_current = validate_transform(
        T_base_object_current, "T_base_object_current"
    )
    T_base_left_tcp_current = validate_transform(
        T_base_left_tcp_current, "T_base_left_tcp_current"
    )
    T_base_object_handoff = validate_transform(
        T_base_object_handoff, "T_base_object_handoff"
    )
    T_right_tcp_object = validate_transform(T_right_tcp_object, "T_right_tcp_object")
    offset = np.asarray(right_pregrasp_offset_tcp_m, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("right_pregrasp_offset_tcp_m must contain three finite values")
    if float(np.linalg.norm(offset)) < 0.01:
        raise ValueError("right pregrasp offset must be at least 10 mm")

    T_left_tcp_object = np.linalg.inv(T_base_left_tcp_current) @ T_base_object_current
    T_base_left_tcp_handoff = T_base_object_handoff @ np.linalg.inv(T_left_tcp_object)
    T_base_right_tcp_grasp = T_base_object_handoff @ np.linalg.inv(T_right_tcp_object)
    T_grasp_pregrasp = np.eye(4)
    T_grasp_pregrasp[:3, 3] = offset
    T_base_right_tcp_pregrasp = T_base_right_tcp_grasp @ T_grasp_pregrasp
    return {
        "T_left_tcp_object": validate_transform(T_left_tcp_object, "T_left_tcp_object"),
        "T_base_left_tcp_handoff": validate_transform(
            T_base_left_tcp_handoff, "T_base_left_tcp_handoff"
        ),
        "T_base_right_tcp_grasp": validate_transform(
            T_base_right_tcp_grasp, "T_base_right_tcp_grasp"
        ),
        "T_base_right_tcp_pregrasp": validate_transform(
            T_base_right_tcp_pregrasp, "T_base_right_tcp_pregrasp"
        ),
    }


def compute_right_tail_grasp_targets(
    T_base_object: np.ndarray,
    tail_grasp_point_object_m: list[float],
    pinch_point_in_right_tcp_m: list[float],
    pregrasp_distance_m: float,
    *,
    jaw_symmetry_deg: float = 0.0,
    approach_mode: str = "from_tail",
) -> dict[str, np.ndarray]:
    """Build right-palm poses that place its pinch centre on the object tail.

    Object ``+X`` is the tail direction and object ``+Y`` is the width to be
    clamped.  The deployed right gripper opens along TCP ``X`` and reaches
    from the palm toward its pinch centre along TCP ``+Z``.
    """
    T_base_object = validate_transform(T_base_object, "T_base_object")
    grasp_object = np.asarray(tail_grasp_point_object_m, dtype=float)
    pinch_tcp = np.asarray(pinch_point_in_right_tcp_m, dtype=float)
    if grasp_object.shape != (3,) or not np.all(np.isfinite(grasp_object)):
        raise ValueError("tail grasp point must contain three finite values")
    if pinch_tcp.shape != (3,) or not np.all(np.isfinite(pinch_tcp)):
        raise ValueError("right TCP pinch point must contain three finite values")
    if not np.isfinite(pregrasp_distance_m) or pregrasp_distance_m < 0.01:
        raise ValueError("right pregrasp distance must be at least 10 mm")
    if jaw_symmetry_deg not in (0.0, 180.0):
        raise ValueError("jaw symmetry must be 0 or 180 degrees")
    valid_approaches = {
        "from_tail",
        "top_down",
        "tail_above_20",
        "tail_above_30",
        "tail_above_45",
        "tail_above_60",
        "tail_below_20",
        "tail_below_30",
        "tail_below_45",
        "tail_below_60",
        "slide_from_body_30",
        "slide_from_body_45",
        "slide_from_body_60",
    }
    if approach_mode not in valid_approaches:
        raise ValueError(f"unsupported approach mode: {approach_mode!r}")

    R_base_object = T_base_object[:3, :3]
    object_x = R_base_object[:, 0]
    object_y = R_base_object[:, 1]
    object_z = R_base_object[:, 2]

    # TCP +Z points from the palm to the pinch centre; TCP +X is jaw opening.
    # Prefer approaching from outside the tail. A top-down candidate is also
    # provided because the side pregrasp can be outside the right arm workspace.
    if approach_mode == "from_tail":
        tcp_z = -object_x
    elif approach_mode == "top_down":
        tcp_z = -object_z
    elif approach_mode.startswith("slide_from_body_"):
        elevation_deg = float(approach_mode.rsplit("_", 1)[1])
        elevation_rad = np.deg2rad(elevation_deg)
        # With the jaws fully open, approach from the object's body/front side:
        # the two fingers straddle the object width and slide along its length
        # until the pinch centre reaches the tail. This is not an outside-tail
        # approach and therefore requires explicit visual clearance confirmation.
        tcp_z = (
            np.cos(elevation_rad) * object_x
            + np.sin(elevation_rad) * object_z
        )
        tcp_z /= np.linalg.norm(tcp_z)
    else:
        elevation_deg = float(approach_mode.rsplit("_", 1)[1])
        elevation_rad = np.deg2rad(elevation_deg)
        vertical_sign = -1.0 if approach_mode.startswith("tail_above_") else 1.0
        # Pregrasp is always outside the physical +X tail:
        #   pregrasp - grasp = -tcp_z * distance
        # Its object-X component must therefore be positive. The two families
        # approach diagonally from above or below without crossing the object.
        tcp_z = (
            -np.cos(elevation_rad) * object_x
            + vertical_sign * np.sin(elevation_rad) * object_z
        )
        tcp_z /= np.linalg.norm(tcp_z)
    if jaw_symmetry_deg == 0.0:
        tcp_x = object_y
        tcp_y = np.cross(tcp_z, tcp_x)
    else:
        tcp_x = -object_y
        tcp_y = np.cross(tcp_z, tcp_x)
    R_base_right_tcp = np.column_stack((tcp_x, tcp_y, tcp_z))

    grasp_point_base = (
        T_base_object[:3, 3] + R_base_object @ grasp_object
    )
    T_base_right_tcp_grasp = np.eye(4)
    T_base_right_tcp_grasp[:3, :3] = R_base_right_tcp
    T_base_right_tcp_grasp[:3, 3] = (
        grasp_point_base - R_base_right_tcp @ pinch_tcp
    )

    # Move the whole palm farther outside the tail, opposite TCP +Z.
    T_base_right_tcp_pregrasp = T_base_right_tcp_grasp.copy()
    T_base_right_tcp_pregrasp[:3, 3] -= tcp_z * pregrasp_distance_m
    return {
        "T_base_right_tcp_grasp": validate_transform(
            T_base_right_tcp_grasp, "T_base_right_tcp_grasp"
        ),
        "T_base_right_tcp_pregrasp": validate_transform(
            T_base_right_tcp_pregrasp, "T_base_right_tcp_pregrasp"
        ),
        "grasp_point_base_m": grasp_point_base,
    }


def transform_to_object_pose(matrix: np.ndarray) -> ObjectPose:
    matrix = validate_transform(matrix, "pose transform")
    return ObjectPose(
        frame_id="body_base",
        position=matrix[:3, 3].tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def transform_debug(matrix: np.ndarray) -> dict[str, Any]:
    matrix = validate_transform(matrix, "debug transform")
    rotation = Rotation.from_matrix(matrix[:3, :3])
    return {
        "matrix": matrix.tolist(),
        "position_m": matrix[:3, 3].tolist(),
        "orientation_xyzw": rotation.as_quat().tolist(),
        "rpy_xyz_deg": rotation.as_euler("xyz", degrees=True).tolist(),
    }


def transform_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return translation metres and rotation degrees between two poses."""
    T_a = validate_transform(T_a, "T_a")
    T_b = validate_transform(T_b, "T_b")
    delta = np.linalg.inv(T_a) @ T_b
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    rotation_deg = float(
        np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
    )
    return translation_m, rotation_deg
