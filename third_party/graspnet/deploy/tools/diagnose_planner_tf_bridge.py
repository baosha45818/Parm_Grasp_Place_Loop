#!/usr/bin/env python3
"""Read-only validation of the accepted SDK-eef_pose -> Planner TF bridge.

This is deliberately a diagnostic, not a calibration or motion tool.  At a
static arm pose it reads both sources used by the touch-board workflow:

  predicted T_planner_base_planner_tool
    = T_planner_base_legacy_arm1_base @ T_legacy_eef @ T_eef_planner_tool

and the live Planner TF for the same tool.  Their difference is the fixed
coordinate-chain error that would be present before any D405 or TCP transform
is applied.  It never plans, executes, enables, teaches, or moves the robot.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
CALIBRATION_DIR = REPO_ROOT / "deploy" / "calibration"
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_DIR))

from calibration_common import read_robot_base_to_eef  # noqa: E402
from calibrate_planner_tf_bridge import transform_from_planner_lookup  # noqa: E402


DEFAULT_BRIDGE = (
    REPO_ROOT / "deploy" / "calibration" / "output" / "planner_tf_bridge.json"
)


def import_sdk_with_planner():
    """Load the SDK build that exposes both Arm and Planner.

    The project virtualenv has an older Arm-only parm_sdk.  The deployed
    Planner lives in the data_collector virtualenv, which is also what the
    production Cartesian motion adapter requires.
    """
    candidates = [
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        REPO_ROOT / ".venv" / "lib" / "python3.10" / "site-packages",
    ]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        sys.path.insert(0, str(candidate))
        # A failed first import must not pin the Arm-only package in sys.modules.
        for name in tuple(sys.modules):
            if name == "parm_sdk" or name.startswith("parm_sdk."):
                del sys.modules[name]
        try:
            sdk = importlib.import_module("parm_sdk")
            if hasattr(sdk, "Arm") and hasattr(sdk, "Planner"):
                return sdk
            errors.append(f"{candidate}: parm_sdk has no Planner")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("cannot import a parm_sdk with Arm and Planner: " + " | ".join(errors))


def load_bridge(path: Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ACCEPTED":
        raise RuntimeError(f"{path} is not an ACCEPTED Planner TF bridge")
    T_planner_base_legacy_base = np.asarray(payload.get("T_planner_base_base"), dtype=np.float64)
    T_eef_planner_tool = np.asarray(payload.get("T_eef_planner_tool"), dtype=np.float64)
    if T_planner_base_legacy_base.shape != (4, 4) or T_eef_planner_tool.shape != (4, 4):
        raise RuntimeError(f"{path} must contain two 4x4 bridge transforms")
    planner_base = str(payload.get("planner_base_frame") or "").strip()
    planner_tool = str(payload.get("planner_tool_frame") or "").strip()
    if not planner_base or not planner_tool:
        raise RuntimeError(f"{path} does not declare planner_base_frame/planner_tool_frame")
    return T_planner_base_legacy_base, T_eef_planner_tool, planner_base, planner_tool


def pose_error(T_actual: np.ndarray, T_predicted: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return actual^-1 * predicted, translation norm mm, rotation magnitude deg."""
    T_error = np.linalg.inv(T_actual) @ T_predicted
    translation_mm = float(np.linalg.norm(T_error[:3, 3]) * 1000.0)
    rotation_deg = float(Rotation.from_matrix(T_error[:3, :3]).magnitude() * 180.0 / np.pi)
    return T_error, translation_mm, rotation_deg


def take_sample(arm, planner, args: argparse.Namespace, bridge: tuple[np.ndarray, np.ndarray, str, str]) -> dict:
    T_planner_base_legacy_base, T_eef_planner_tool, planner_base, planner_tool = bridge
    before = read_robot_base_to_eef(arm, args.arm_id)
    lookup = planner.lookup_transform(planner_tool, planner_base)
    T_live = transform_from_planner_lookup(lookup, planner_tool, planner_base)
    after = read_robot_base_to_eef(arm, args.arm_id)
    motion_mm = float(np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000.0)
    motion_deg = float(Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).magnitude() * 180.0 / np.pi)
    T_predicted = T_planner_base_legacy_base @ after @ T_eef_planner_tool
    T_error, translation_mm, rotation_deg = pose_error(T_live, T_predicted)
    return {
        "T_legacy_arm1_base_eef": after.tolist(),
        "T_planner_base_tool_live": T_live.tolist(),
        "T_planner_base_tool_predicted": T_predicted.tolist(),
        "T_live_inverse_predicted": T_error.tolist(),
        "translation_error_mm": translation_mm,
        "rotation_error_deg": rotation_deg,
        "arm_motion_during_read_mm": motion_mm,
        "arm_motion_during_read_deg": motion_deg,
        "static": motion_mm <= args.max_motion_mm and motion_deg <= args.max_motion_deg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SDK/Planner TF bridge diagnostic")
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--arm-id", type=int, choices=(1, 2), default=1)
    parser.add_argument("--planner-host", default="localhost")
    parser.add_argument("--planner-trajectory-port", type=int, default=5605,
                        help="GraspNet dedicated Planner trajectory port")
    parser.add_argument("--planner-tf-port", type=int, default=5614,
                        help="GraspNet dedicated Planner TF port")
    parser.add_argument("--planner-waypoint-port", type=int, default=5615,
                        help="GraspNet dedicated Planner waypoint port")
    parser.add_argument("--samples", type=int, default=10, help="static paired reads (default: 10)")
    parser.add_argument("--interval-s", type=float, default=0.15)
    parser.add_argument("--max-motion-mm", type=float, default=0.3)
    parser.add_argument("--max-motion-deg", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if args.samples < 1 or args.interval_s < 0:
        parser.error("--samples must be positive and --interval-s must be non-negative")

    bridge = load_bridge(args.bridge)
    _, _, planner_base, planner_tool = bridge
    print("Read-only diagnostic: no trajectory will be planned or executed.")
    print(f"bridge: {args.bridge}")
    print(f"compare: SDK eef_pose -> {planner_base}/{planner_tool} (samples={args.samples})")

    sdk = import_sdk_with_planner()
    samples: list[dict] = []
    with sdk.Arm(host=args.arm_host) as arm, sdk.Planner(
        host=args.planner_host,
        trajectory_port=args.planner_trajectory_port,
        tf_port=args.planner_tf_port,
        waypoint_port=args.planner_waypoint_port,
    ) as planner:
        available = planner.list_frame_ids()
        for frame in (planner_base, planner_tool):
            if frame not in available:
                raise RuntimeError(f"Planner frame {frame!r} is absent")
        for index in range(args.samples):
            sample = take_sample(arm, planner, args, bridge)
            samples.append(sample)
            print(
                f"#{index + 1:02d}: bridge error={sample['translation_error_mm']:.3f} mm / "
                f"{sample['rotation_error_deg']:.3f} deg; "
                f"arm during read={sample['arm_motion_during_read_mm']:.3f} mm / "
                f"{sample['arm_motion_during_read_deg']:.3f} deg"
            )
            if index + 1 < args.samples:
                time.sleep(args.interval_s)

    stable = [sample for sample in samples if sample["static"]]
    if not stable:
        raise RuntimeError("all reads occurred while the arm was moving; hold it stationary and run again")
    translation = np.asarray([sample["translation_error_mm"] for sample in stable])
    rotation = np.asarray([sample["rotation_error_deg"] for sample in stable])
    report = {
        "bridge": str(args.bridge),
        "planner_base_frame": planner_base,
        "planner_tool_frame": planner_tool,
        "sample_count": len(samples),
        "static_sample_count": len(stable),
        "translation_error_mm": {
            "median": float(np.median(translation)), "max": float(np.max(translation)),
        },
        "rotation_error_deg": {
            "median": float(np.median(rotation)), "max": float(np.max(rotation)),
        },
        "samples": samples,
    }
    print(
        "RESULT (static samples): "
        f"median={report['translation_error_mm']['median']:.3f} mm / "
        f"{report['rotation_error_deg']['median']:.3f} deg, "
        f"max={report['translation_error_mm']['max']:.3f} mm / "
        f"{report['rotation_error_deg']['max']:.3f} deg"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
