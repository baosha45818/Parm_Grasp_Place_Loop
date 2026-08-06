#!/usr/bin/env python3
"""Re-run ICP candidate selection from a saved head-D435 capture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(__file__).resolve().parents[2] / ".venv/bin/python"
if __name__ == "__main__" and Path(sys.executable) != PYTHON:
    os.execv(
        str(PYTHON),
        [str(PYTHON), "-m", "tests.reversed_head_observation.reprocess_head_d435_result", *sys.argv[1:]],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tool.inspection_support.sam2_icp import (
    icp_pose,
    save_pose_visualization,
)
from tool.inspection_support.stl_model import (
    load_stl_triangles_m,
    sample_surface_points,
    triangles_in_object_frame,
)


DEFAULT_CONFIG = Path(__file__).with_name("head_d435_pose_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.result.resolve().parent
    result = json.loads(args.result.read_text(encoding="utf-8"))
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    points_path = run_dir / "head_d435_points_camera.npy"
    mask_path = run_dir / "head_d435_mask.png"
    points = np.load(points_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot read {mask_path}")
    intrinsics = np.asarray(result["point_report"]["camera_intrinsics"], dtype=float)
    triangles = triangles_in_object_frame(
        load_stl_triangles_m(Path(cfg["stl_path"])),
        R_object_stl=cfg.get("R_object_stl"),
        object_origin_stl_m=cfg.get("object_origin_stl_m"),
    )
    cad = sample_surface_points(
        triangles, int(cfg["cad_sample_points"]), int(cfg["seed"])
    )
    T_camera_object, metrics = icp_pose(
        cad,
        points,
        cfg,
        selection_mask=mask,
        camera_intrinsics=intrinsics,
    )

    frame_id = (result.get("object_pose") or {}).get("frame_id")
    if frame_id == "body_base":
        alignment_path = Path(result["transform_info"]["calibration_path"])
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        T_output_object = (
            np.asarray(alignment["T_body_base_head_camera"], dtype=float)
            @ T_camera_object
        )
    elif frame_id == "head_d435i_color_optical_frame":
        T_output_object = T_camera_object
    else:
        raise RuntimeError(f"unsupported output frame {frame_id!r}")

    visualization = save_pose_visualization(
        run_dir,
        "head_d435",
        cad,
        T_camera_object,
        T_output_object,
        frame_id,
        intrinsics,
        output_filename="pose_estimation_visualization_corrected.png",
    )
    corrected = dict(result)
    corrected.update({
        "reprocessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reprocessed_from": str(args.result.resolve()),
        "object_pose": {
            "frame_id": frame_id,
            "position_m": T_output_object[:3, 3].tolist(),
            "orientation_xyzw": Rotation.from_matrix(
                T_output_object[:3, :3]
            ).as_quat().tolist(),
        },
        "T_camera_object": T_camera_object.tolist(),
        "T_output_object": T_output_object.tolist(),
        "registration": metrics,
        "pose_visualization": str(visualization),
        "model_frame": {
            "stl_path": str(Path(cfg["stl_path"]).resolve()),
            "R_object_stl": cfg.get("R_object_stl", np.eye(3).tolist()),
            "object_origin_stl_m": cfg.get("object_origin_stl_m", [0.0, 0.0, 0.0]),
        },
    })
    output = run_dir / "result_corrected.json"
    output.write_text(
        json.dumps(corrected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": "REPROCESSED_WITH_MASK_OVERLAP_SELECTION",
        "output": str(output),
        "visualization": str(visualization),
        "object_pose": corrected["object_pose"],
        "registration": metrics,
        "robot_moved": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
