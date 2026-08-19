#!/usr/bin/env python3
"""Enumerate and capture RGB-D from a selected connected camera.

Examples:
  python capture_selected_rgbd.py --list
  python capture_selected_rgbd.py --camera 0
  python capture_selected_rgbd.py --camera 1 --frames 3 --fuse-frames 5

The numbering shown by ``--list`` is the only numbering accepted by
``--camera``.  Every capture is written into its own timestamped subdirectory;
existing captures and source images are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.inspection_support.robot_camera import (
    D405Camera,
    FemtoBoltCamera,
    list_orbbec_devices,
    list_realsense_devices,
)
from loop_config import LoopConfig
from tools.planner_runtime import PlannerRuntime


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB + aligned depth from a numbered camera.")
    parser.add_argument("--list", action="store_true", help="List connected cameras and exit")
    parser.add_argument("--camera", type=int, help="Camera index printed by --list")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=1, help="Number of RGB-D pairs to save")
    parser.add_argument("--fuse-frames", type=int, default=5, help="Valid depth frames median-fused per pair")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280, help="Requested RGB width")
    parser.add_argument("--height", type=int, default=720, help="Requested RGB height")
    parser.add_argument("--depth-width", type=int, default=640, help="Requested depth width for Orbbec")
    parser.add_argument("--depth-height", type=int, default=480, help="Requested depth height for Orbbec")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--record-robot-state", action=argparse.BooleanOptionalAction, default=True,
        help="Record left-arm, gripper and TCP state into robot_state.json (default: enabled)",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--joint-unit", choices=("deg", "rad"), default="deg",
                        help="Unit returned by the arm state service; snapshot is always written in degrees")
    args = parser.parse_args()
    if not args.list and args.camera is None:
        parser.error("Specify --list or --camera <index>")
    if args.frames < 1 or args.fuse_frames < 1 or args.warmup_frames < 0:
        parser.error("--frames and --fuse-frames must be >= 1; --warmup-frames must be >= 0")
    return args


def connected_cameras() -> list[dict[str, str]]:
    """Return one stable, displayable index across both supported SDKs."""
    cameras: list[dict[str, str]] = []
    for device in list_realsense_devices():
        cameras.append({"backend": "realsense", "serial": device["serial"], "name": device["name"]})
    for serial in list_orbbec_devices():
        cameras.append({"backend": "orbbec", "serial": serial, "name": "Orbbec RGB-D camera"})
    return cameras


def print_cameras(cameras: list[dict[str, str]]) -> None:
    if not cameras:
        print("No supported RGB-D cameras are connected.")
        return
    print("index  backend     serial             name")
    for index, camera in enumerate(cameras):
        print(f"{index:>5}  {camera['backend']:<10}  {camera['serial']:<17}  {camera['name']}")


def make_camera(spec: dict[str, str], args: argparse.Namespace):
    if spec["backend"] == "realsense":
        # RealSense alignment is performed by D405Camera internally.  Its name
        # is configurable, so it is safe to use with any RGB-D RealSense model.
        return D405Camera(
            serial=spec["serial"], width=args.width, height=args.height, fps=args.fps,
            warmup_frames=args.warmup_frames, depth_width=args.width, depth_height=args.height,
            camera_name=f"realsense_{spec['serial']}",
        )
    return FemtoBoltCamera(
        serial=spec["serial"], color_width=args.width, color_height=args.height,
        depth_width=args.depth_width, depth_height=args.depth_height, fps=args.fps,
        warmup_frames=args.warmup_frames,
    )


def read_fused_frame(camera, fuse_frames: int, timeout_ms: int):
    """Keep only complete frames and median-fuse non-zero depth pixels."""
    frames = []
    errors = []
    for _ in range(max(fuse_frames * 5, 10)):
        try:
            frame = camera.read(timeout_ms=timeout_ms)
            if frame.depth.shape != frame.color.shape[:2] or not np.any(frame.depth):
                errors.append("unaligned or empty depth")
                continue
            frames.append(frame)
            if len(frames) == fuse_frames:
                break
        except Exception as exc:
            errors.append(str(exc))
    if len(frames) != fuse_frames:
        raise RuntimeError(f"Only received {len(frames)}/{fuse_frames} complete RGB-D frames; {errors[-3:]}")
    raw = np.stack([frame.depth.astype(np.float32) for frame in frames])
    raw[raw == 0] = np.nan
    depth = np.nanmedian(raw, axis=0)
    depth = np.nan_to_num(depth, nan=0.0).astype(np.uint16)
    return frames[-1], depth


def depth_preview(depth: np.ndarray, scale_m: float) -> tuple[np.ndarray, list[float]]:
    depth_m = depth.astype(np.float32) * scale_m
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        raise RuntimeError("Captured depth has no valid pixels")
    near, far = (float(value) for value in np.percentile(depth_m[valid], [2, 98]))
    far = max(far, near + 0.001)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = np.clip((depth_m[valid] - near) * 255.0 / (far - near), 0, 255).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview, [near, far]


def read_left_joints_deg(arm: object, joint_unit: str) -> list[float]:
    state = arm.get_state() or {}
    left = ((state.get("state") or {}).get("left") or {})
    joints = np.asarray(left.get("position") or [], dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"Invalid left-arm state: {left}")
    if joint_unit == "rad":
        joints = np.rad2deg(joints)
    return [float(value) for value in joints]


def json_safe(value: object) -> object:
    """Convert SDK response values into JSON-safe Python data."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def capture_robot_state(args: argparse.Namespace) -> dict:
    """Read a best-effort state snapshot without ever discarding RGB-D data."""
    snapshot: dict[str, object] = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "status": "ok",
        "base_frame": "body_base",
        "tool_frame": "left_hand_transfer_end",
        "joint_unit_written": "deg",
    }
    planner = PlannerRuntime(LoopConfig(
        planner_host=args.planner_host, execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base", left_group="left_arm", right_group="right_arm",
        left_tool_frame="left_hand_transfer_end", right_tool_frame="right_hand_transfer_end",
        max_velocity=0.1, max_acceleration=0.1, idle_poll_seconds=0.2, live=True, poses={},
    ))
    try:
        from parm_sdk import Arm, Gripper

        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        with Arm(host=args.arm_host) as arm, Gripper(host=args.gripper_host) as gripper:
            snapshot["left_arm_joint_position_deg"] = read_left_joints_deg(arm, args.joint_unit)
            snapshot["left_gripper_state"] = json_safe(gripper.get_state().get(1))
            transform_response = planner.lookup_transform("body_base", "left_hand_transfer_end")
            data = (transform_response or {}).get("data") or transform_response or {}
            translation = data.get("translation") or {}
            rotation = data.get("rotation") or {}
            snapshot["T_body_base_left_hand_transfer_end"] = {
                "translation_m": {axis: float(translation[axis]) for axis in "xyz"},
                "quaternion_xyzw": {axis: float(rotation[axis]) for axis in "xyzw"},
            }
    except Exception as exc:
        snapshot["status"] = "unavailable"
        snapshot["error"] = str(exc)
    finally:
        planner.close()
    return snapshot


def capture(spec: dict[str, str], args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = args.output_dir / f"rgbd_{timestamp}_{spec['backend']}_{spec['serial']}"
    destination.mkdir(parents=True, exist_ok=False)
    camera = make_camera(spec, args)
    saved = []
    try:
        camera.open()
        for index in range(args.frames):
            frame, depth = read_fused_frame(camera, args.fuse_frames, args.timeout_ms)
            preview, preview_range = depth_preview(depth, float(frame.depth_scale))
            rgb_name = f"rgb_{index:03d}.png"
            depth_name = f"depth_aligned_{index:03d}.png"
            preview_name = f"depth_preview_{index:03d}.png"
            for path, image in ((destination / rgb_name, frame.color), (destination / depth_name, depth),
                                (destination / preview_name, preview)):
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"Failed to save {path}")
            np.save(destination / f"depth_aligned_{index:03d}.npy", depth)
            saved.append({"index": index, "rgb": rgb_name, "depth": depth_name,
                          "depth_npy": f"depth_aligned_{index:03d}.npy", "preview": preview_name,
                          "preview_range_m": preview_range, "valid_depth_ratio": float(np.mean(depth > 0))})
    finally:
        camera.close()

    robot_state = None
    if args.record_robot_state:
        robot_state = capture_robot_state(args)
        (destination / "robot_state.json").write_text(
            json.dumps(robot_state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    metadata = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "camera": spec,
        "depth_alignment": "depth pixels are aligned to RGB pixels",
        "depth_format": "uint16; multiply by depth_scale_m_per_unit for metres",
        "depth_scale_m_per_unit": float(frame.depth_scale),
        "color_intrinsics": frame.color_intrinsics.tolist(),
        "depth_intrinsics": frame.depth_intrinsics.tolist(),
        "fusion": f"median of {args.fuse_frames} valid frames",
        "frames": saved,
        "robot_state_snapshot": (
            {"file": "robot_state.json", "status": robot_state["status"]}
            if robot_state is not None else {"recorded": False}
        ),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    cameras = connected_cameras()
    if args.list:
        print_cameras(cameras)
        return
    if args.camera < 0 or args.camera >= len(cameras):
        print_cameras(cameras)
        raise ValueError(f"Invalid --camera {args.camera}")
    selected = cameras[args.camera]
    print(f"Capturing camera {args.camera}: {selected['backend']} {selected['serial']} ({selected['name']})")
    destination = capture(selected, args)
    print(f"Capture complete: {destination}")


if __name__ == "__main__":
    main()
