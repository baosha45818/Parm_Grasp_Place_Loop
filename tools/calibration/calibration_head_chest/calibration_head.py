#!/usr/bin/env python3
"""Eye-to-hand calibration for a camera installed at a named robot position."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration_common import (
    capture_samples, load_samples, refresh_board_poses,
    review_board_corners, save_json, solve_laser_robust, transform_to_json,
)
from chest_orbbec_camera import ChestOrbbecCamera
from head_d435i_camera import HeadD435iCamera


CODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CODE_DIR / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "realsense_d435i_head_charuco.json"
OUTPUT_ROOT = CODE_DIR / "outputs"
SAMPLE_CAMERA_KEY = "camera"

def _load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {label} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _load_referenced_config(setup_path: Path, reference: object, label: str) -> dict:
    if not isinstance(reference, str) or not reference:
        raise RuntimeError(f"{label} reference must be a non-empty relative path")
    target = (setup_path.parent / reference).resolve()
    return _load_json(target, label)


def load_config(path: Path) -> dict:
    """Load the four user-maintained calibration settings.

    Camera model and position are labels only. The adapter is detected from
    the connected device matching ``camera_serial`` rather than inferred from
    either label.
    """
    config = _load_json(path, "calibration config")
    if {"camera_model", "camera_serial", "position", "board_config"} <= config.keys():
        board = _load_referenced_config(path, config["board_config"], "board config")
        config = {
            "camera_model": config["camera_model"],
            "default_serial": config["camera_serial"],
            "position": config["position"],
            "default_board": board,
            "config_format": "simple_v3",
        }
    else:
        # Retained for existing private configurations.
        config["config_format"] = "legacy"

    if config["config_format"] == "legacy":
        config.setdefault("position", config.get("camera_name"))
    required = {"camera_model", "default_serial", "position", "default_board"}
    if missing := sorted(required - config.keys()):
        raise RuntimeError(f"calibration config missing fields: {missing}")
    if not isinstance(config["default_board"], dict):
        raise RuntimeError("calibration config field default_board must be an object")
    for key in ("camera_model", "default_serial", "position"):
        if not isinstance(config[key], str) or not config[key]:
            raise RuntimeError(f"calibration config field {key!r} must be a non-empty string")
    return config


def parse_args() -> tuple[argparse.Namespace, dict]:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    selected, _ = selector.parse_known_args()
    config = load_config(selected.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="calibration setup JSON, e.g. configs/realsense_d435i_head_charuco.json",
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument(
        "--data", type=Path,
        default=OUTPUT_ROOT / selected.config.stem,
        help="sample/result directory (defaults to outputs/<config filename>)",
    )
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--arm-id", type=int, default=1)
    parser.add_argument("--gripper-host", default="192.168.3.11")
    parser.add_argument("--gripper-hand", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    parser.add_argument("--disable-gripper-toggle", action="store_true")
    parser.add_argument("--serial", default=str(config["default_serial"]))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--head-pitch-deg", type=float, default=0.0, help="recorded for a position with gimbal_pose enabled")
    parser.add_argument("--head-yaw-deg", type=float, default=0.0, help="recorded for a position with gimbal_pose enabled")
    parser.add_argument("--teach-free", dest="teach_free", action="store_true", default=True)
    parser.add_argument("--drag-teach", dest="teach_free", action="store_false")
    parser.add_argument("--max-translation-rms-mm", type=float, default=5.0)
    parser.add_argument("--max-translation-max-mm", type=float, default=15.0)
    parser.add_argument("--max-rotation-rms-deg", type=float, default=1.0)
    parser.add_argument("--max-rotation-max-deg", type=float, default=3.0)
    parser.add_argument("--max-outlier-fraction", type=float, default=0.25)
    parser.add_argument("--min-accepted-samples", type=int, default=10)
    parser.add_argument("--exclude-index", type=int, action="append", default=[])
    parser.add_argument("--intrinsics", type=Path, help="optional JSON with camera_matrix and distortion_coeffs")
    parser.add_argument("--review-corners", dest="review_corners", action="store_true", default=True)
    parser.add_argument("--skip-corner-review", dest="review_corners", action="store_false")
    return parser.parse_args(), config


class AutoDetectedCamera:
    """Open the SDK that recognises the configured serial number."""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self._camera = None

    def __enter__(self):
        failures = []
        for factory in (HeadD435iCamera, ChestOrbbecCamera):
            camera = factory(self.serial, self.width, self.height, self.fps)
            try:
                self._camera = camera.open()
                return self._camera
            except Exception as exc:
                camera.close()
                failures.append(f"{factory.__name__}: {exc}")
        raise RuntimeError(f"no supported camera found for serial {self.serial!r}: " + "; ".join(failures))

    def __exit__(self, exc_type, exc, tb):
        if self._camera is not None:
            self._camera.close()


def create_camera(serial: str, width: int, height: int, fps: int) -> AutoDetectedCamera:
    return AutoDetectedCamera(serial, width, height, fps)


def main() -> None:
    args, config = parse_args()
    if not args.collect and not args.solve:
        raise SystemExit("choose --collect and/or --solve")
    if min(args.width, args.height, args.fps, args.samples) <= 0:
        raise SystemExit("--width, --height, --fps, and --samples must be positive")
    position = str(config["position"])
    board = dict(config["default_board"])
    if args.collect:
        print(f"[{position}] Keep this camera rigidly fixed throughout this session.")
        capture_samples(
            create_camera(args.serial, args.width, args.height, args.fps),
            SAMPLE_CAMERA_KEY, args.arm_host, args.arm_id, board, args.data, args.samples, args.teach_free,
            gripper_host=None if args.disable_gripper_toggle else args.gripper_host,
            gripper_hand=args.gripper_hand, gripper_force=args.gripper_force,
        )
    if not args.solve:
        return

    samples = load_samples(args.data)
    excluded = set(args.exclude_index)
    available = {int(sample["index"]) for sample in samples}
    if missing := sorted(excluded - available):
        raise RuntimeError(f"excluded sample indices not found: {missing}")
    samples = [sample for sample in samples if int(sample["index"]) not in excluded]
    intrinsics_source = "active camera profile"
    if args.intrinsics is not None:
        payload = json.loads(args.intrinsics.read_text(encoding="utf-8"))
        matrix = np.asarray(payload.get("camera_matrix"), dtype=np.float64)
        distortion = np.asarray(payload.get("distortion_coeffs"), dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or distortion.size < 4:
            raise RuntimeError(f"invalid camera intrinsics file: {args.intrinsics}")
        for sample in samples:
            sample["camera_matrix"] = matrix.tolist()
            sample["distortion_coeffs"] = distortion.tolist()
        intrinsics_source = str(args.intrinsics)
    if args.review_corners and board.get("type") not in {"aruco_grid", "charuco"}:
        samples = review_board_corners(samples, args.data, board, camera_name=SAMPLE_CAMERA_KEY)
    samples = refresh_board_poses(samples, args.data, SAMPLE_CAMERA_KEY, board)
    T_base_camera, T_eef_board, rms, used, rejected, rows = solve_laser_robust(
        samples, max_outlier_fraction=args.max_outlier_fraction,
        min_samples=args.min_accepted_samples, camera_name=SAMPLE_CAMERA_KEY,
    )
    translation = [row["translation_mm"] for row in rows]
    rotation = [row["rotation_deg"] for row in rows]
    quality = {
        "translation_rms_mm": float(np.sqrt(np.mean(np.square(translation)))),
        "translation_max_mm": float(max(translation)),
        "rotation_rms_deg": float(np.sqrt(np.mean(np.square(rotation)))),
        "rotation_max_deg": float(max(rotation)),
        "thresholds": {
            "translation_rms_mm": args.max_translation_rms_mm,
            "translation_max_mm": args.max_translation_max_mm,
            "rotation_rms_deg": args.max_rotation_rms_deg,
            "rotation_max_deg": args.max_rotation_max_deg,
        },
        "samples": rows,
    }
    accepted = (
        len(used) >= args.min_accepted_samples
        and quality["translation_rms_mm"] <= args.max_translation_rms_mm
        and quality["translation_max_mm"] <= args.max_translation_max_mm
        and quality["rotation_rms_deg"] <= args.max_rotation_rms_deg
        and quality["rotation_max_deg"] <= args.max_rotation_max_deg
    )
    result = {
        "camera_name": position,
        "position": position,
        "camera_model": config["camera_model"],
        "camera_serial": args.serial,
        "calibration_type": config["calibration_type"],
        "board": board,
        "sample_count": len(used),
        "input_sample_count": len(samples),
        "rejected_indices": rejected,
        "explicitly_excluded_indices": sorted(excluded),
        "camera_matrix": samples[0].get("camera_matrix"),
        "distortion_coeffs": samples[0].get("distortion_coeffs"),
        "intrinsics_source": "active camera profile" if args.intrinsics is None else intrinsics_source,
        "image_size": samples[0].get("image_size"),
        "robot_pose_source": "parm_sdk state.<arm>.eef_pose",
        "corner_assignment": (
            "ChArUco corner IDs" if board.get("type") == "charuco"
            else "ArUco marker IDs" if board.get("type") == "aruco_grid"
            else "manual_review" if args.review_corners else "printed_text_strip"
        ),
        "rms_residual": rms,
        "residual": quality,
        "T_base_camera": transform_to_json(T_base_camera),
        "T_eef_board": transform_to_json(T_eef_board),
        "status": "ACCEPTED" if accepted else "REJECTED_QUALITY_GATE",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    result["mount_pose_deg"] = {"pitch": args.head_pitch_deg, "yaw": args.head_yaw_deg}
    result_stem = f"calibration_{args.config.stem}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / (
        f"{result_stem}.json" if accepted else f"{result_stem}_rejected.json"
    )
    save_json(output, result)
    if not accepted:
        raise RuntimeError(
            f"{position} calibration rejected: kept={len(used)}/{len(samples)}, "
            f"translation rms/max={quality['translation_rms_mm']:.1f}/{quality['translation_max_mm']:.1f}mm, "
            f"rotation rms/max={quality['rotation_rms_deg']:.2f}/{quality['rotation_max_deg']:.2f}deg; "
            f"diagnostic saved to {output}"
        )
    print(f"saved {output}; rms={rms:.6f}")


if __name__ == "__main__":
    main()
