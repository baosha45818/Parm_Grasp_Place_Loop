#!/usr/bin/env python3
"""Propagate a fixed-pose D435 extrinsic through the nominal head URDF chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_REFERENCE = Path(
    "/home/dainanjun/Public/Parm_Grasp_Place_Loop/calibration/sdk_base_alignment/"
    "sdk_base_to_planner_alignment.json"
)
DEFAULT_OUTPUT = Path(
    "/home/dainanjun/Public/Parm_Grasp_Place_Loop/calibration/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch2_yaw26p5_kinematic.json"
)


def transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler(axis, angle_rad).as_matrix()
    return result


def nominal_body_head_cam(command_pitch_deg: float, command_yaw_deg: float) -> np.ndarray:
    """Return nominal T_body_base_head_cam from the deployed PARM URDF.

    URDF head_joint_link1 is yaw about +Z and head_joint_link2 is pitch about
    +Y.  The command pitch sign is used here.  Although the deployed Head SDK
    reports pitch feedback with the opposite sign, the 38.5-degree calibrated
    camera translation agrees with the URDF at joint +38.5 degrees.
    """
    yaw = np.deg2rad(float(command_yaw_deg))
    pitch_urdf = np.deg2rad(float(command_pitch_deg))
    T_base_link1 = transform([0.0, 0.0, 0.039718], [0.0, 0.0, 0.0]) @ axis_rotation("z", yaw)
    T_link1_link2 = (
        transform([0.00081107, -0.041492, 0.07], [0.0, 0.0, 0.019545])
        @ axis_rotation("y", pitch_urdf)
    )
    T_link2_cam = transform([0.08305, 0.04376, 0.11778], [0.0, 0.1745, 0.0])
    return T_base_link1 @ T_link1_link2 @ T_link2_cam


def propagate_extrinsic(
    T_body_camera_reference: np.ndarray,
    reference_pitch_deg: float,
    reference_yaw_deg: float,
    target_pitch_deg: float,
    target_yaw_deg: float,
) -> np.ndarray:
    reference = np.asarray(T_body_camera_reference, dtype=float)
    if reference.shape != (4, 4) or not np.all(np.isfinite(reference)):
        raise ValueError("reference extrinsic must be a finite 4x4 matrix")
    model_reference = nominal_body_head_cam(reference_pitch_deg, reference_yaw_deg)
    model_target = nominal_body_head_cam(target_pitch_deg, target_yaw_deg)
    return model_target @ np.linalg.inv(model_reference) @ reference


def derive_payload(
    reference: dict[str, Any], target_pitch_deg: float, target_yaw_deg: float
) -> dict[str, Any]:
    reference_pose = reference.get("head_pose_deg") or {}
    reference_pitch = float(reference_pose["pitch"])
    reference_yaw = float(reference_pose["yaw"])
    target = propagate_extrinsic(
        np.asarray(reference["T_body_base_head_camera"], dtype=float),
        reference_pitch,
        reference_yaw,
        target_pitch_deg,
        target_yaw_deg,
    )
    payload = dict(reference)
    payload.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "head_pose_deg": {
            "pitch": float(target_pitch_deg),
            "yaw": float(target_yaw_deg),
        },
        "T_body_base_head_camera": target.tolist(),
        "extrinsic_derivation": {
            "method": "reference_extrinsic_propagated_through_deployed_head_urdf",
            "reference_head_pose_deg": {
                "pitch": reference_pitch,
                "yaw": reference_yaw,
            },
            "target_head_pose_deg": {
                "pitch": float(target_pitch_deg),
                "yaw": float(target_yaw_deg),
            },
            "pitch_mapping": "urdf_joint_rad = command_pitch_deg",
            "yaw_mapping": "urdf_joint_rad = command_yaw_deg",
            "urdf_joints": ["head_joint_link1", "head_joint_link2", "head_joint_cam"],
            "warning": (
                "KINEMATIC_DERIVATION_NOT_AN_INDEPENDENT_CAMERA_CALIBRATION; "
                "accuracy depends on nominal URDF joint origins and zero offsets"
            ),
        },
    })
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--target-pitch", type=float, default=2.0)
    parser.add_argument("--target-yaw", type=float, default=26.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    payload = derive_payload(reference, args.target_pitch, args.target_yaw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": "KINEMATIC_ALIGNMENT_GENERATED",
        "output": str(args.output.resolve()),
        "head_pose_deg": payload["head_pose_deg"],
        "T_body_base_head_camera": payload["T_body_base_head_camera"],
        "warning": payload["extrinsic_derivation"]["warning"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
