#!/usr/bin/env python3
"""Calibrate the read-only bridge from the deployment base frame to Planner TF.

No robot motion, enable, teach, or gripper API is used.  Each sample pairs the
deployment transform ``T_base_eef`` with Planner's live
``T_planner_base_planner_tool``.  Several manually chosen static arm poses are
then used to solve the two fixed transforms:

    T_planner_base_planner_tool = T_planner_base_base
                                     @ T_base_eef
                                     @ T_eef_planner_tool

Camera calibrations remain expressed in the existing ``base`` frame.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
CALIBRATION_DIR = REPO_ROOT / "deploy" / "calibration"
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_DIR))

from calibration_common import read_robot_base_to_eef  # noqa: E402


DEFAULT_KINEMATICS = Path("/home/dainanjun/Public/plucky_hjt_parm/arm_configs/arm_kinematics/left.yaml")


def import_parm_sdk():
    """Import an SDK build that exposes both Arm and Planner.

    The project virtualenv may contain an Arm-only SDK while the deployed
    Planner client is installed in data_collector's virtualenv.  A bridge
    capture must use a single SDK build that has both interfaces.
    """
    candidates = [
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
        Path.home() / "Public" / "parm_sdk_1.0.0" / "parm_sdk-1.0.0+20260717.191601-py3-none-any.whl",
        REPO_ROOT / ".venv" / "lib" / "python3.10" / "site-packages",
    ]
    errors: list[str] = []
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        # Do not let a previously imported Arm-only package pin this process.
        for name in tuple(sys.modules):
            if name == "parm_sdk" or name.startswith("parm_sdk."):
                del sys.modules[name]
        try:
            sdk = importlib.import_module("parm_sdk")
            if hasattr(sdk, "Arm") and hasattr(sdk, "Planner"):
                return sdk
            errors.append(f"{candidate}: parm_sdk has no Arm/Planner pair")
        except Exception as exc:  # Try every supported local SDK location.
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("parm_sdk with Arm and Planner is not importable: " + " | ".join(errors))


def transform_from_planner_lookup(response: dict, source: str, target: str) -> np.ndarray:
    """Normalize the deployed inverse TF payload to ``T_target_source``."""
    data = response.get("data") or {}
    if data.get("source_frame") != source or data.get("target_frame") != target:
        raise RuntimeError(f"unexpected Planner TF response: {data}")
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = np.array([translation.get(axis) for axis in ("x", "y", "z")], dtype=np.float64)
    quat = np.array([rotation.get(axis) for axis in ("x", "y", "z", "w")], dtype=np.float64)
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-9:
        raise RuntimeError(f"invalid Planner TF response: {data}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quat).as_matrix()
    result[:3, 3] = xyz
    return np.linalg.inv(result)


def pose_to_transform(values: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    result[:3, 3] = values[:3]
    return result


def transform_to_pose(transform: np.ndarray) -> np.ndarray:
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]


def sdk_joint_deg(state: dict, arm_id: int) -> np.ndarray:
    """Return the encoder angles paired with the SDK EEF pose in this state."""
    key = "left" if int(arm_id) == 1 else "right"
    values = np.asarray(((state or {}).get("state") or {}).get(key, {}).get("position"), dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"SDK state has no finite seven-axis {key} joint position: {state}")
    return values


def overlay_fk(kinematics_path: Path, joint_deg: np.ndarray) -> np.ndarray:
    """The calibrated left-URDF FK for the exact SDK feedback angles.

    This independently checks that the Domain-60 Planner TF is built from the
    same joint feedback and calibrated left-arm geometry before a bridge
    sample is accepted.  The final 84 mm fixed end-link origin is present in
    the calibrated xacro after J7.
    """
    rows = yaml.safe_load(kinematics_path.read_text(encoding="utf-8"))["kinematics"]
    T = np.eye(4, dtype=np.float64)
    for index, angle in enumerate(np.radians(joint_deg), start=1):
        row = rows[f"left_arm_joint{index}"]
        origin = np.eye(4, dtype=np.float64)
        origin[:3, :3] = Rotation.from_euler(
            "xyz", [row["roll"], row["pitch"], row["yaw"]]
        ).as_matrix()
        origin[:3, 3] = [row["x"], row["y"], row["z"]]
        joint = np.eye(4, dtype=np.float64)
        joint[:3, :3] = Rotation.from_rotvec([0.0, 0.0, float(angle)]).as_matrix()
        T = T @ origin @ joint
    T[:3, 3] += T[:3, :3] @ np.array([0.084, 0.0, 0.0])
    return T


def pose_error(T_actual: np.ndarray, T_predicted: np.ndarray) -> tuple[float, float]:
    error = np.linalg.inv(T_actual) @ T_predicted
    return (
        float(np.linalg.norm(error[:3, 3]) * 1000.0),
        float(Rotation.from_matrix(error[:3, :3]).magnitude() * 180.0 / np.pi),
    )


def load_samples(samples_dir: Path, max_fk_error_mm: float, max_fk_error_deg: float) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(samples_dir.glob("sample_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_eef = np.asarray(payload["T_base_eef"], dtype=np.float64)
        planner_base_tool = np.asarray(payload["T_planner_base_tool"], dtype=np.float64)
        if base_eef.shape != (4, 4) or planner_base_tool.shape != (4, 4):
            raise RuntimeError(f"{path} does not contain 4x4 transforms")
        # New captures contain an explicit FK/Planner-TF check.  Do not fit a
        # base bridge to samples where the Planner is using stale or different
        # joint feedback; that would hide an FK/joint-state fault as a bridge.
        check = payload.get("overlay_fk_vs_planner_tf")
        if check is not None:
            mm, deg = float(check.get("translation_mm", np.inf)), float(check.get("rotation_deg", np.inf))
            if mm > max_fk_error_mm or deg > max_fk_error_deg:
                raise RuntimeError(
                    f"{path}: overlay FK and Planner TF disagree {mm:.3f}mm/{deg:.3f}deg "
                    f"(limits {max_fk_error_mm:.3f}mm/{max_fk_error_deg:.3f}deg); "
                    "fix Planner joint feedback/URDF before fitting a bridge"
                )
        pairs.append((base_eef, planner_base_tool))
    return pairs


def solve_bridge(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(pairs) < 5:
        raise RuntimeError("need at least five samples captured at substantially different arm poses")

    relative_rotations = [
        Rotation.from_matrix((np.linalg.inv(pairs[0][0]) @ base_eef)[:3, :3]).magnitude()
        for base_eef, _ in pairs[1:]
    ]
    if len(relative_rotations) < 2 or np.degrees(max(relative_rotations)) < 20.0:
        raise RuntimeError("sample poses lack rotational diversity; capture at least 20 degrees of EEF rotation")

    def residual(values: np.ndarray) -> np.ndarray:
        planner_base_base = pose_to_transform(values[:6])
        eef_tool = pose_to_transform(values[6:])
        rows = []
        for base_eef, planner_base_tool in pairs:
            error = np.linalg.inv(planner_base_tool) @ planner_base_base @ base_eef @ eef_tool
            # One metre and one radian have comparable optimizer scale here.
            rows.extend(error[:3, 3])
            rows.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec())
        return np.asarray(rows, dtype=np.float64)

    # The hand-eye equation has local minima.  Deterministic diverse seeds make
    # the offline calibration robust without ever commanding the robot.
    rng = np.random.default_rng(20260720)
    starts = [np.zeros(12, dtype=np.float64)]
    for _ in range(24):
        seed = np.zeros(12, dtype=np.float64)
        seed[:3] = rng.uniform(-0.6, 0.6, size=3)
        seed[3:6] = rng.uniform(-np.pi, np.pi, size=3)
        seed[6:9] = rng.uniform(-0.25, 0.25, size=3)
        seed[9:12] = rng.uniform(-np.pi, np.pi, size=3)
        starts.append(seed)
    result = min(
        (least_squares(residual, start, method="trf", max_nfev=4000) for start in starts),
        key=lambda candidate: candidate.cost,
    )
    if not result.success:
        raise RuntimeError(f"bridge solve failed: {result.message}")
    planner_base_base = pose_to_transform(result.x[:6])
    eef_tool = pose_to_transform(result.x[6:])
    rows = residual(result.x).reshape(-1, 6)
    translation_mm = np.linalg.norm(rows[:, :3], axis=1) * 1000.0
    rotation_deg = np.linalg.norm(rows[:, 3:], axis=1) * 180.0 / np.pi
    metrics = {
        "sample_count": len(pairs),
        "translation_rms_mm": float(np.sqrt(np.mean(translation_mm**2))),
        "translation_max_mm": float(np.max(translation_mm)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_deg**2))),
        "rotation_max_deg": float(np.max(rotation_deg)),
        "optimizer_cost": float(result.cost),
    }
    return planner_base_base, eef_tool, metrics


def capture_sample(args: argparse.Namespace) -> None:
    sdk = import_parm_sdk()
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    with sdk.Arm(host=args.arm_host) as arm, sdk.Planner(
        host=args.planner_host, trajectory_port=args.planner_trajectory_port,
        tf_port=args.planner_tf_port, waypoint_port=args.planner_waypoint_port,
    ) as planner:
        frame_ids = planner.list_frame_ids()
        for frame in (args.planner_base_frame, args.planner_tool_frame):
            if frame not in frame_ids:
                raise RuntimeError(f"Planner frame {frame!r} is absent; available frames: {frame_ids}")
        # The deployed service labels this source -> target but serializes the
        # inverse payload; transform_from_planner_lookup normalizes it.
        before_state = arm.get_state()
        before = read_robot_base_to_eef(arm, args.arm_id)
        lookup = planner.lookup_transform(args.planner_tool_frame, args.planner_base_frame)
        T_planner_base_tool = transform_from_planner_lookup(
            lookup, args.planner_tool_frame, args.planner_base_frame
        )
        after_state = arm.get_state()
        after = read_robot_base_to_eef(arm, args.arm_id)
    translation_change_mm = float(np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000.0)
    rotation_change_deg = float(
        Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).magnitude() * 180.0 / np.pi
    )
    if translation_change_mm > 1.0 or rotation_change_deg > 0.5:
        raise RuntimeError(
            "robot moved while reading this sample "
            f"({translation_change_mm:.2f} mm, {rotation_change_deg:.2f} deg); recapture while static"
        )
    T_base_eef = after
    q_deg = sdk_joint_deg(after_state, args.arm_id)
    T_overlay_fk = overlay_fk(args.kinematics, q_deg)
    fk_error_mm, fk_error_deg = pose_error(T_planner_base_tool, T_overlay_fk)
    if fk_error_mm > args.max_fk_error_mm or fk_error_deg > args.max_fk_error_deg:
        raise RuntimeError(
            "Planner FK/feedback check failed: calibrated overlay FK at the SDK joint angles "
            f"differs from live Planner TF by {fk_error_mm:.3f}mm/{fk_error_deg:.3f}deg "
            f"(limits {args.max_fk_error_mm:.3f}mm/{args.max_fk_error_deg:.3f}deg). "
            "Do not capture a bridge sample until Planner joint feedback and URDF are correct."
        )

    paths = sorted(args.samples_dir.glob("sample_*.json"))
    sample_path = args.samples_dir / f"sample_{len(paths) + 1:03d}.json"
    payload = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "arm_host": args.arm_host,
        "arm_id": args.arm_id,
        "planner_host": args.planner_host,
        "planner_base_frame": args.planner_base_frame,
        "planner_tool_frame": args.planner_tool_frame,
        "T_base_eef": T_base_eef.tolist(),
        "T_planner_base_tool": T_planner_base_tool.tolist(),
        "sdk_joint_deg": q_deg.tolist(),
        "T_overlay_fk_left_arm_base_end_link": T_overlay_fk.tolist(),
        "overlay_fk_vs_planner_tf": {
            "translation_mm": fk_error_mm,
            "rotation_deg": fk_error_deg,
            "kinematics": str(args.kinematics),
            "meaning": "same SDK q -> calibrated overlay FK versus live 5605 Planner TF",
        },
        "planner_lookup": lookup,
        "arm_motion_during_read": {
            "translation_change_mm": translation_change_mm,
            "rotation_change_deg": rotation_change_deg,
        },
    }
    sample_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"SDK q -> overlay FK vs Planner TF: {fk_error_mm:.3f} mm / {fk_error_deg:.3f} deg")
    print(f"saved {sample_path}")
    print("This command only read SDK state and Planner TF; it did not send a robot command.")


def solve_and_write(args: argparse.Namespace) -> None:
    pairs = load_samples(args.samples_dir, args.max_fk_error_mm, args.max_fk_error_deg)
    T_planner_base_base, T_eef_tool, metrics = solve_bridge(pairs)
    accepted = metrics["translation_max_mm"] <= args.max_translation_error_mm and metrics["rotation_max_deg"] <= args.max_rotation_error_deg
    payload = {
        "status": "ACCEPTED" if accepted else "REJECTED",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "formula": "T_planner_base_tool = T_planner_base_base @ T_base_eef @ T_eef_planner_tool",
        "source_base_frame": "base",
        "source_eef_frame": "eef_pose",
        "planner_base_frame": args.planner_base_frame,
        "planner_tool_frame": args.planner_tool_frame,
        "T_planner_base_base": T_planner_base_base.tolist(),
        "T_eef_planner_tool": T_eef_tool.tolist(),
        "metrics": metrics,
        "limits": {
            "max_translation_error_mm": args.max_translation_error_mm,
            "max_rotation_error_deg": args.max_rotation_error_deg,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "metrics": metrics}, indent=2))
    print(f"wrote {args.output}")
    if not accepted:
        raise RuntimeError("bridge residual is too large; capture more diverse static poses and solve again")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only base <-> Planner TF bridge calibration")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true", help="capture one read-only Arm/Planner TF pair")
    parser.add_argument("--solve", action="store_true", help="solve a bridge from saved samples")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "deploy" / "calibration" / "output" / "planner_tf_bridge.json")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--arm-id", type=int, choices=(1, 2), default=1)
    parser.add_argument("--planner-host", default="localhost")
    parser.add_argument("--planner-trajectory-port", type=int, default=5605,
                        help="GraspNet calibrated Planner trajectory port")
    parser.add_argument("--planner-tf-port", type=int, default=5614,
                        help="GraspNet calibrated Planner TF port")
    parser.add_argument("--planner-waypoint-port", type=int, default=5615,
                        help="GraspNet calibrated Planner waypoint port")
    parser.add_argument("--planner-base-frame", default="base_link")
    parser.add_argument("--planner-tool-frame", default="tool0")
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS,
                        help="calibrated left-arm YAML used for same-q FK validation")
    parser.add_argument("--max-fk-error-mm", type=float, default=0.5,
                        help="reject capture/solve if same-q overlay FK differs from live Planner TF above this")
    parser.add_argument("--max-fk-error-deg", type=float, default=0.1,
                        help="reject capture/solve if same-q overlay FK differs from live Planner TF above this")
    parser.add_argument("--max-translation-error-mm", type=float, default=3.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=2.0)
    args = parser.parse_args()
    if args.capture == args.solve:
        parser.error("specify exactly one of --capture or --solve")
    if args.max_fk_error_mm <= 0 or args.max_fk_error_deg <= 0:
        parser.error("--max-fk-error-mm and --max-fk-error-deg must be positive")
    if args.capture:
        capture_sample(args)
    else:
        solve_and_write(args)


if __name__ == "__main__":
    main()
