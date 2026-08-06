#!/usr/bin/env python3
"""Read-only convention/timestamp check for SDK eef_pose and Planner TF.

No motion, enable, planner trajectory, teach, or gripper command is sent.
For each sample this records raw SDK state plus both directions of the same
Planner TF lookup, then verifies that the two raw transforms are inverses.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]


def import_sdk():
    candidates = [
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        REPO_ROOT / ".venv" / "lib" / "python3.10" / "site-packages",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        sys.path.insert(0, str(candidate))
        for name in tuple(sys.modules):
            if name == "parm_sdk" or name.startswith("parm_sdk."):
                del sys.modules[name]
        sdk = importlib.import_module("parm_sdk")
        if hasattr(sdk, "Arm") and hasattr(sdk, "Planner"):
            return sdk
    raise RuntimeError("cannot import a parm_sdk with both Arm and Planner")


def raw_transform(response: dict) -> tuple[np.ndarray, dict]:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([rotation[key] for key in "xyzw"]).as_matrix()
    T[:3, 3] = [translation[key] for key in "xyz"]
    return T, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--planner-host", default="localhost")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "deploy" / "calibration" / "output" / "sdk_planner_pose_pair_check.json")
    args = parser.parse_args()
    if args.samples < 1 or args.interval_s < 0:
        parser.error("--samples must be positive and --interval-s non-negative")

    sdk = import_sdk()
    rows: list[dict] = []
    print("Read-only check: querying raw SDK eef_pose and both Planner TF directions.")
    with sdk.Arm(host=args.arm_host) as arm, sdk.Planner(host=args.planner_host) as planner:
        for index in range(args.samples):
            before = arm.get_state() or {}
            forward_response = planner.lookup_transform("left_arm_base", "left_arm_end_link")
            reverse_response = planner.lookup_transform("left_arm_end_link", "left_arm_base")
            after = arm.get_state() or {}
            T_forward, forward = raw_transform(forward_response)
            T_reverse, reverse = raw_transform(reverse_response)
            closure = T_forward @ T_reverse
            closure_mm = float(np.linalg.norm(closure[:3, 3]) * 1000.0)
            closure_deg = float(Rotation.from_matrix(closure[:3, :3]).magnitude() * 180.0 / np.pi)
            left_before = ((before.get("state") or {}).get("left") or {})
            left_after = ((after.get("state") or {}).get("left") or {})
            row = {
                "sdk_timestamp_before": before.get("timestamp"),
                "sdk_timestamp_after": after.get("timestamp"),
                "sdk_left_eef_pose_before_raw": left_before.get("eef_pose"),
                "sdk_left_eef_pose_after_raw": left_after.get("eef_pose"),
                "sdk_left_position_before_deg": left_before.get("position"),
                "sdk_left_position_after_deg": left_after.get("position"),
                "planner_forward_raw": forward_response,
                "planner_reverse_raw": reverse_response,
                "planner_forward_timestamp": forward.get("timestamp"),
                "planner_reverse_timestamp": reverse.get("timestamp"),
                "raw_forward_times_reverse_closure_mm": closure_mm,
                "raw_forward_times_reverse_closure_deg": closure_deg,
                "raw_T_left_arm_base_left_arm_end_link": T_forward.tolist(),
            }
            rows.append(row)
            print(
                f"#{index + 1}: raw forward*reverse closure={closure_mm:.6f} mm/{closure_deg:.6f} deg; "
                f"SDK timestamps={row['sdk_timestamp_before']!r}->{row['sdk_timestamp_after']!r}; "
                f"Planner timestamps={row['planner_forward_timestamp']!r},{row['planner_reverse_timestamp']!r}"
            )
            if index + 1 < args.samples:
                time.sleep(args.interval_s)
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
