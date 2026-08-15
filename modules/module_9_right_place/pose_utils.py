"""Pose-format and frame-axis helpers."""

from __future__ import annotations

import math

from move_by_joints import rpy_to_rotvec


def command_for(target_rpy: list[float]) -> list[float]:
    return target_rpy[:3] + rpy_to_rotvec(target_rpy[3:])


def local_z_in_world(rpy: list[float]) -> list[float]:
    """Unit +Z of right_hand_transfer_end expressed in SDK world coordinates."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Third column of Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr]
