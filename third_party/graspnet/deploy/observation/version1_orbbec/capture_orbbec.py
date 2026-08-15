#!/usr/bin/env python3
"""Save one Femto observation after arm1 has already reached ``go_pc``.

This program never sends an arm or gripper command.  It only reads arm1's EEF
pose so the captured point cloud can later be transformed into EEF coordinates.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

GRASPNET_ROOT = Path("/home/dainanjun/qiuguolin/graspnet")
if str(GRASPNET_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASPNET_ROOT))
if str(GRASPNET_ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(GRASPNET_ROOT / "utils"))

from robot_camera import DEFAULT_FEMTO_BOLT_SERIAL, FemtoBoltCamera


def depth_scale_meters(frame) -> float:
    raw = float(frame.depth_scale)
    if not np.isfinite(raw) or raw <= 0:
        raise RuntimeError(f"invalid depth scale: {raw}")
    return raw * 1e-3 if frame.camera_name == "femto_bolt" and raw >= 1.0 else raw


def read_T_base_eef(host: str) -> np.ndarray:
    from parm_sdk import Arm

    with Arm(host=host) as arm:
        state = arm.get_state() or {}
    arm_state = (state.get("state") or {}).get("left") or {}
    if str(state.get("system_state", "")).lower() != "running" or arm_state.get("enabled") is not True:
        raise RuntimeError("arm1 is not RUNNING and enabled")
    pose = arm_state.get("eef_pose") or {}
    xyz, rpy = np.asarray(pose.get("xyz"), dtype=np.float64), np.asarray(pose.get("rpy"), dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.all(np.isfinite(np.r_[xyz, rpy])):
        raise RuntimeError("arm state has no finite arm1 eef_pose")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Femto RGB-D at the already-reached go_pc pose")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--femto-serial", default=DEFAULT_FEMTO_BOLT_SERIAL)
    parser.add_argument("--frames", type=int, default=3, help="median-fuse this many depth frames")
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    T_base_eef = read_T_base_eef(args.arm_host)
    with FemtoBoltCamera(serial=args.femto_serial) as camera:
        frames = [camera.read() for _ in range(args.frames)]
    reference = frames[-1]
    if any(frame.depth.shape != reference.depth.shape for frame in frames):
        raise RuntimeError("camera depth shape changed during capture")
    depth = np.median(np.stack([frame.depth for frame in frames]), axis=0).astype(reference.depth.dtype)
    out = args.output_root / "captures" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "go_pc"
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "depth.npy", depth)
    np.save(out / "depth_intrinsics.npy", reference.depth_intrinsics)
    Image.fromarray(reference.color[:, :, ::-1]).save(out / "color.png")
    (out / "meta.json").write_text(json.dumps({
        "depth_scale_m": depth_scale_meters(reference),
        "depth_scale_raw": float(reference.depth_scale),
        "T_base_eef": T_base_eef.tolist(),
        "camera_serial": reference.serial,
        "observation_pose": "go_pc",
        "color_shape_hw": list(reference.color.shape[:2]),
        "depth_shape_hw": list(reference.depth.shape[:2]),
        "depth_in_color_pixels": reference.depth.shape[:2] == reference.color.shape[:2]
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")
    print("next: create a depth-aligned SAM2 mask.npy and run classify_orbbec.py")


if __name__ == "__main__":
    main()
