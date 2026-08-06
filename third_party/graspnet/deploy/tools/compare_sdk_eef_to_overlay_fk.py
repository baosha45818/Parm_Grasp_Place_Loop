#!/usr/bin/env python3
"""Read SDK once and compare its EEF pose with overlay-URDF FK at the same q.

This deliberately does not query Planner TF or /joint_states.  It isolates
the calibrated overlay kinematics from a potentially incorrect Planner joint
state bridge.  No robot command is sent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from compare_sdk_planner_saved_poses import sdk_transform, planner_transform, solve_fixed_frame_alignment  # noqa: E402


def read_sdk_state(host: str) -> tuple[np.ndarray, np.ndarray]:
    import importlib
    for candidate in (Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages", REPO / ".venv" / "lib" / "python3.10" / "site-packages"):
        if candidate.exists(): sys.path.insert(0, str(candidate))
    sdk = importlib.import_module("parm_sdk")
    with sdk.Arm(host=host) as arm:
        left = ((arm.get_state() or {}).get("state") or {}).get("left") or {}
    return sdk_transform(left["eef_pose"]), np.radians(np.asarray(left["position"], dtype=float))


def overlay_fk(kinematics: Path, q_rad: np.ndarray) -> np.ndarray:
    payload = yaml.safe_load(kinematics.read_text(encoding="utf-8"))["kinematics"]
    T = np.eye(4)
    for index, angle in enumerate(q_rad, 1):
        row = payload[f"left_arm_joint{index}"]
        origin = np.eye(4)
        origin[:3, :3] = Rotation.from_euler("xyz", [row["roll"], row["pitch"], row["yaw"]]).as_matrix()
        origin[:3, 3] = [row["x"], row["y"], row["z"]]
        joint = np.eye(4); joint[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        T = T @ origin @ joint
    T[0, 3] += T[0, 0] * .084
    T[1, 3] += T[1, 0] * .084
    T[2, 3] += T[2, 0] * .084
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--kinematics", type=Path, default=Path("/home/dainanjun/Public/plucky_hjt_parm/arm_configs/arm_kinematics/left.yaml"))
    args = parser.parse_args()
    pairs = []
    for path in sorted(args.samples_dir.glob("sample_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        pairs.append((sdk_transform(item["sdk_left_eef_pose_after_raw"]), planner_transform(item)))
    _, A, B = solve_fixed_frame_alignment(pairs)
    sdk_eef, q = read_sdk_state(args.arm_host)
    fk = overlay_fk(args.kinematics, q)
    predicted = A @ sdk_eef @ B
    error = np.linalg.inv(fk) @ predicted
    print(json.dumps({
        "sdk_joint_deg": np.degrees(q).round(6).tolist(),
        "overlay_fk_T_left_arm_base_end_link": fk.round(12).tolist(),
        "sdk_eef_expressed_as_left_arm_base_end_link": predicted.round(12).tolist(),
        "same_q_error": {"translation_mm": float(np.linalg.norm(error[:3, 3]) * 1000), "rotation_deg": float(Rotation.from_matrix(error[:3, :3]).magnitude() * 180 / np.pi)},
    }, indent=2))


if __name__ == "__main__":
    main()
