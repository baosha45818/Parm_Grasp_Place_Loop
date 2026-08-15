#!/usr/bin/env python3
"""Compare SDK and Planner end poses recorded at the same static arm poses.

Fits only the two unavoidable fixed frame definitions (SDK base->Planner base
and SDK EEF->Planner end link), then reports the remaining pose-dependent
error.  It never connects to hardware or commands motion.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE))


def pose_to_transform(values: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    T[:3, 3] = values[:3]
    return T


def solve_fixed_frame_alignment(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Fit fixed SDK-base/EEF frame definitions and retain pose-dependent residual."""
    def residual(values: np.ndarray) -> np.ndarray:
        T_planner_base_sdk_base = pose_to_transform(values[:6])
        T_sdk_eef_planner_end = pose_to_transform(values[6:])
        rows = []
        for T_sdk_base_eef, T_planner_base_end in pairs:
            error = np.linalg.inv(T_planner_base_end) @ T_planner_base_sdk_base @ T_sdk_base_eef @ T_sdk_eef_planner_end
            rows.extend(error[:3, 3])
            rows.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec())
        return np.asarray(rows)

    # A zero seed already corresponds to coincident frame names; use just a
    # few deterministic alternatives to avoid expensive unrelated searches.
    starts = [np.zeros(12)]
    for yaw in (-np.pi / 2, np.pi / 2, np.pi):
        seed = np.zeros(12); seed[5] = yaw
        starts.append(seed)
    result = min((least_squares(residual, seed, max_nfev=2500) for seed in starts), key=lambda item: item.cost)
    rows = residual(result.x).reshape(-1, 6)
    trans = np.linalg.norm(rows[:, :3], axis=1) * 1000.0
    rot = np.linalg.norm(rows[:, 3:], axis=1) * 180.0 / np.pi
    return {
        "sample_count": len(pairs),
        "translation_rms_mm": float(np.sqrt(np.mean(trans ** 2))),
        "translation_max_mm": float(np.max(trans)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rot ** 2))),
        "rotation_max_deg": float(np.max(rot)),
    }, pose_to_transform(result.x[:6]), pose_to_transform(result.x[6:])


def import_sdk_with_planner():
    root = HERE.parents[2]
    for candidate in (
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        root / ".venv" / "lib" / "python3.10" / "site-packages",
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


def read_sdk_eef(arm) -> np.ndarray:
    raw = ((arm.get_state() or {}).get("state") or {}).get("left", {}).get("eef_pose")
    if not isinstance(raw, dict) or not isinstance(raw.get("xyz"), list) or not isinstance(raw.get("rpy"), list):
        raise RuntimeError(f"unexpected SDK left eef_pose: {raw}")
    return sdk_transform(raw)


def sdk_transform(raw: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", np.asarray(raw["rpy"], dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(raw["xyz"], dtype=float)
    return T


def planner_transform(sample: dict) -> np.ndarray:
    data = sample["planner_tf_after_raw"]["direct"]["data"]
    q, p = data["rotation"], data["translation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    # Planner labels the request direction but serializes the inverse matrix.
    # These saved records were captured with lookup(base, end), so invert to
    # obtain canonical T_left_arm_base_left_arm_end_link.
    return np.linalg.inv(T)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="also read the current SDK/Planner pose; read-only")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--planner-host", default="localhost")
    parser.add_argument("--planner-trajectory-port", type=int, default=5605)
    parser.add_argument("--planner-tf-port", type=int, default=5614)
    parser.add_argument("--planner-waypoint-port", type=int, default=5615)
    args = parser.parse_args()
    pairs = []
    for path in sorted(args.samples_dir.glob("sample_*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        raw = sample.get("sdk_left_eef_pose_after_raw")
        if not isinstance(raw, dict) or "rpy" not in raw or "xyz" not in raw:
            raise RuntimeError(f"{path}: missing saved raw SDK eef_pose")
        pairs.append((sdk_transform(raw), planner_transform(sample)))
    metrics, T_planner_base_sdk_base, T_sdk_eef_planner_end = solve_fixed_frame_alignment(pairs)
    # Validation without refitting each held-out pose exposes a misleading
    # in-sample alignment caused by accidentally stale or duplicated samples.
    split = len(pairs) // 2
    train_metrics, train_A, train_B = solve_fixed_frame_alignment(pairs[:split])
    test_rows = []
    for S, P in pairs[split:]:
        error = np.linalg.inv(P) @ train_A @ S @ train_B
        test_rows.append((np.linalg.norm(error[:3, 3]) * 1000.0, Rotation.from_matrix(error[:3, :3]).magnitude() * 180.0 / np.pi))
    test = np.asarray(test_rows)
    report = {
        "samples_dir": str(args.samples_dir), "metrics": metrics,
        "fixed_frame_alignment": {
            "T_planner_base_sdk_base": T_planner_base_sdk_base.round(12).tolist(),
            "T_sdk_eef_planner_end_link": T_sdk_eef_planner_end.round(12).tolist(),
        },
        "holdout_validation_first10_fit_last10": {
            "fit_metrics": train_metrics,
            "translation_rms_mm": float(np.sqrt(np.mean(test[:, 0] ** 2))),
            "translation_max_mm": float(np.max(test[:, 0])),
            "rotation_rms_deg": float(np.sqrt(np.mean(test[:, 1] ** 2))),
            "rotation_max_deg": float(np.max(test[:, 1])),
        },
    }
    if args.live:
        sdk = import_sdk_with_planner()
        with sdk.Arm(host=args.arm_host) as arm, sdk.Planner(
            host=args.planner_host, trajectory_port=args.planner_trajectory_port,
            tf_port=args.planner_tf_port, waypoint_port=args.planner_waypoint_port,
        ) as planner:
            sdk_before = read_sdk_eef(arm)
            time.sleep(.03)
            # Request end->base then invert the serialized payload to obtain
            # canonical T_base_end, matching calibrate_planner_tf_bridge.py.
            data = planner.lookup_transform("left_arm_end_link", "left_arm_base")["data"]
            q, p = data["rotation"], data["translation"]
            planner_live = np.eye(4)
            planner_live[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
            planner_live[:3, 3] = [p["x"], p["y"], p["z"]]
            planner_live = np.linalg.inv(planner_live)
            time.sleep(.03)
            sdk_after = read_sdk_eef(arm)
        sdk_motion_mm = float(np.linalg.norm(sdk_after[:3, 3] - sdk_before[:3, 3]) * 1000)
        sdk_motion_deg = float(Rotation.from_matrix(sdk_before[:3, :3].T @ sdk_after[:3, :3]).magnitude() * 180 / np.pi)
        predicted = T_planner_base_sdk_base @ sdk_after @ T_sdk_eef_planner_end
        error = np.linalg.inv(planner_live) @ predicted
        report["live"] = {
            "SDK_T_base_eef": sdk_after.round(12).tolist(),
            "Planner_T_left_arm_base_end_link": planner_live.round(12).tolist(),
            "SDK_pose_expressed_in_Planner_frames": predicted.round(12).tolist(),
            "error_after_fixed_frame_alignment": {
                "translation_mm": float(np.linalg.norm(error[:3, 3]) * 1000),
                "rotation_deg": float(Rotation.from_matrix(error[:3, :3]).magnitude() * 180 / np.pi),
            },
            "SDK_motion_during_read": {"translation_mm": sdk_motion_mm, "rotation_deg": sdk_motion_deg},
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
