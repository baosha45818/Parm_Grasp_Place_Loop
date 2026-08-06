#!/usr/bin/env python3
"""Classify a held workpiece from one go_pc Femto RGB-D capture.

The classification is geometric: CAD-to-observation ICP determines the CAD
face pointing down at go_pc.  Routing is then read from the local config.  No
robot, gripper or camera command is sent by this offline program.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/dainanjun/qiuguolin/graspnet")
VERSION6 = ROOT / "deploy" / "version6"
for path in (ROOT, VERSION6):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from workpiece_pose import (
    ViewCapture,
    RegistrationQualityError,
    capture_points_in_eef,
    classify_held_workpiece_faces,
    estimate_eef_to_object_from_clouds,
    validate_transform,
)

HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_capture(path: Path) -> ViewCapture:
    meta = load_json(path / "meta.json")
    depth = np.load(path / "depth.npy")
    intrinsics = np.load(path / "depth_intrinsics.npy")
    mask = np.load(path / "mask.npy")
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise RuntimeError("mask.npy and depth.npy must be same-size 2D arrays")
    if intrinsics.shape != (3, 3):
        raise RuntimeError("depth_intrinsics.npy must be 3x3")
    roi = meta.get("roi_xywh")
    if not isinstance(roi, list) or len(roi) != 4:
        raise RuntimeError("meta.json needs depth-pixel roi_xywh before classification")
    return ViewCapture(
        name="go_pc",
        depth_u16=depth,
        depth_intrinsics=intrinsics,
        depth_scale_m=float(meta["depth_scale_m"]),
        roi_xywh=tuple(int(value) for value in roi),
        T_base_eef=validate_transform(meta["T_base_eef"], "T_base_eef"),
        object_mask_u8=mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline go_pc Orbbec grasp-class routing")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config_template.json")
    parser.add_argument("--laser-calibration", type=Path,
                        default=ROOT / "deploy/calibration/output/laser/calibration_laser.json")
    parser.add_argument("--output-root", type=Path, default=HERE / "output")
    args = parser.parse_args()
    cfg, calibration = load_json(args.config), load_json(args.laser_calibration)
    if calibration.get("status") != "ACCEPTED":
        raise RuntimeError("laser calibration status must be ACCEPTED")
    T_base_laser = validate_transform(calibration["T_base_laser"], "T_base_laser")
    capture = load_capture(args.capture_dir)
    if not bool(load_json(args.capture_dir / "meta.json").get("depth_in_color_pixels", False)):
        raise RuntimeError("capture is not color/depth aligned; do not use a color-derived SAM2 mask")
    point_cfg = cfg["point_cloud"]
    cloud = capture_points_in_eef(capture, T_base_laser,
        min_depth_m=float(point_cfg["min_depth_m"]), max_depth_m=float(point_cfg["max_depth_m"]),
        erode_px=int(point_cfg["erode_px"]), voxel_m=float(point_cfg["observed_voxel_m"]))
    if len(cloud) < int(point_cfg["min_points_per_view"]):
        raise RegistrationQualityError(f"go_pc has only {len(cloud)} usable object points")
    T_eef_object, metrics, _ = estimate_eef_to_object_from_clouds(
        [cloud], [capture], Path(cfg["workpiece"]["stl_path"]), point_cfg)
    faces = classify_held_workpiece_faces(
        T_eef_object, capture.T_base_eef, cfg["workpiece"]["face_axes_stl"],
        ["front", "back", "left", "right"], 0.01)
    route = cfg["routing_by_down_face"][faces["down_face"]]
    result = {
        "T_eef_object": T_eef_object.tolist(),
        "registration": metrics,
        "held_orientation": faces,
        "route": route,
        "safety": "offline classification only; no robot action was sent"
    }
    output = args.output_root / "classification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"class={route['grasp_class']} next_step={route['next_step']}")
    print(f"saved {output / 'result.json'}")


if __name__ == "__main__":
    main()
