#!/usr/bin/env python3
"""Read-only live SDK EEF vs GraspNet Planner end-link verification.

The fixed SDK-base/end-link definitions are loaded from the broad-workspace
SDK kinematics fit.  No trajectory is planned or executed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def import_sdk():
    for candidate in (
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        REPO / ".venv" / "lib" / "python3.10" / "site-packages",
    ):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
        for name in tuple(sys.modules):
            if name == "parm_sdk" or name.startswith("parm_sdk."):
                del sys.modules[name]
        sdk = importlib.import_module("parm_sdk")
        if hasattr(sdk, "Arm") and hasattr(sdk, "Planner"):
            return sdk
    raise RuntimeError("parm_sdk Arm/Planner client is unavailable")


def matrix(rows) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(4, 4)


def sdk_pose(state: dict) -> np.ndarray:
    pose = state["state"]["left"]["eef_pose"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose["rpy"]).as_matrix()
    T[:3, 3] = pose["xyz"]
    return T


def planner_pose(planner) -> np.ndarray:
    response = planner.lookup_transform("left_arm_base", "left_arm_end_link")
    data = response["data"]
    if data.get("source_frame") != "left_arm_base" or data.get("target_frame") != "left_arm_end_link":
        raise RuntimeError(f"unexpected Planner TF: {data}")
    q, p = data["rotation"], data["translation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


def delta(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    relative = np.linalg.inv(before) @ after
    return (float(np.linalg.norm(relative[:3, 3]) * 1000),
            float(Rotation.from_matrix(relative[:3, :3]).magnitude() * 180 / np.pi))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path,
                        default=REPO / "deploy/calibration/output/left_kinematics_sdk_20260728_candidate.yaml")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--planner-trajectory-port", type=int, default=5605)
    parser.add_argument("--planner-tf-port", type=int, default=5614)
    parser.add_argument("--planner-waypoint-port", type=int, default=5615)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    fit = yaml.safe_load(args.alignment.read_text(encoding="utf-8"))
    A = matrix(fit["T_planner_base_sdk_base"])
    B = matrix(fit["T_sdk_eef_planner_end_link"])
    sdk = import_sdk()
    print("Read-only: no trajectory will be planned or executed.")
    print(f"alignment: {args.alignment}")
    errors = []
    with sdk.Arm(host=args.arm_host) as arm, sdk.Planner(
        host=args.planner_host, trajectory_port=args.planner_trajectory_port,
        tf_port=args.planner_tf_port, waypoint_port=args.planner_waypoint_port,
    ) as planner:
        for index in range(args.samples):
            before = sdk_pose(arm.get_state())
            actual = planner_pose(planner)
            after = sdk_pose(arm.get_state())
            motion_mm, motion_deg = delta(before, after)
            predicted = A @ before @ B
            err_mm, err_deg = delta(actual, predicted)
            errors.append((err_mm, err_deg))
            print(f"#{index + 1}: SDK->Planner error={err_mm:.3f} mm / {err_deg:.3f} deg; "
                  f"SDK motion during read={motion_mm:.3f} mm / {motion_deg:.3f} deg")
            time.sleep(0.1)
    values = np.asarray(errors)
    print(json.dumps({
        "translation_median_mm": float(np.median(values[:, 0])),
        "translation_max_mm": float(np.max(values[:, 0])),
        "rotation_median_deg": float(np.median(values[:, 1])),
        "rotation_max_deg": float(np.max(values[:, 1])),
    }, indent=2))


if __name__ == "__main__":
    main()
