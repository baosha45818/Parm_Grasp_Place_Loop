#!/usr/bin/env python3
"""Visualize the convex-tab grasp candidates on the source head RGB image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
REVERSE_GRASP_ROOT = HERE.parent


def inside_here(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        # Manual candidate preparation stores per-capture visualizations below
        # reverse_grasp/runs, not only below reverse_grasp/grasp.
        path.relative_to(REVERSE_GRASP_ROOT)
    except ValueError as exc:
        raise ValueError(f"输出必须在 {REVERSE_GRASP_ROOT}: {path}") from exc
    return path


def project(K: np.ndarray, point: np.ndarray) -> tuple[int, int] | None:
    if point[2] <= 1e-6:
        return None
    p = K @ point
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


def find_capture_dir(source_pose: Path) -> Path:
    """Find capture beside either legacy or fresh-pipeline ICP output."""
    for candidate in source_pose.parents:
        if (candidate / "metadata.json").is_file():
            return candidate
        sibling_capture = candidate / "capture"
        if (sibling_capture / "metadata.json").is_file():
            return sibling_capture
    raise RuntimeError(f"无法从位姿文件定位 RGB-D capture 目录: {source_pose}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finger-half-gap-m", type=float, default=.018,
                        help="仅用于图示的半开度，并非夹爪命令")
    parser.add_argument("--pregrasp-m", type=float, default=.060)
    args = parser.parse_args()
    output = inside_here(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction = json.loads(args.prediction_json.expanduser().resolve().read_text(encoding="utf-8"))
    source_pose_path = Path(prediction["source_pose"]).expanduser().resolve()
    source_pose = json.loads(source_pose_path.read_text(encoding="utf-8"))
    capture = find_capture_dir(source_pose_path)
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("无法读取源 RGB")
    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0., intr["ppx"]], [0., intr["fy"], intr["ppy"]], [0., 0., 1.]])
    T_obj = np.asarray(source_pose["T_camera_object"], dtype=float)
    contact = np.asarray(prediction["grasp_contact_point_camera_m"], dtype=float)
    table_normal = np.asarray(source_pose.get("table_normal_camera", []), dtype=float)
    approach0 = (-table_normal / np.linalg.norm(table_normal)
                 if table_normal.shape == (3,) else T_obj[:3, 2] / np.linalg.norm(T_obj[:3, 2]))
    jaw = T_obj[:3, 0] / np.linalg.norm(T_obj[:3, 0])
    # Make a stable right-handed display basis; grasp implementation uses this same basis.
    side = np.cross(approach0, jaw); side /= np.linalg.norm(side)
    jaw = np.cross(side, approach0); jaw /= np.linalg.norm(jaw)
    canvas = image.copy()
    p = project(K, contact)
    if p is None:
        raise RuntimeError("抓取点不在相机前方")
    cv2.circle(canvas, p, 8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "grasp TCP", (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
    left, right = contact + jaw * args.finger_half_gap_m, contact - jaw * args.finger_half_gap_m
    for point, color, label in ((left, (0, 255, 0), "left finger (+object X)"),
                                (right, (0, 0, 255), "right finger (-object X)")):
        uv = project(K, point)
        if uv:
            cv2.drawMarker(canvas, uv, color, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (uv[0] + 6, uv[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    colors = [(255, 255, 255), (255, 0, 255), (0, 165, 255)]
    candidates = []
    for angle, color in zip((90., 15., -15.), colors):
        tilt = float(np.copysign(90. - abs(angle), angle))
        R = np.column_stack((approach0, jaw, side)) @ Rotation.from_rotvec([0., np.deg2rad(tilt), 0.]).as_matrix()
        pre = contact - R[:, 0] * args.pregrasp_m
        uv = project(K, pre)
        if uv:
            cv2.arrowedLine(canvas, uv, p, color, 2, cv2.LINE_AA, tipLength=.13)
            cv2.putText(canvas, f"{angle:+.0f} deg approach", (uv[0] + 5, uv[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        candidates.append({"table_approach_angle_deg": angle, "side_tilt_about_tcp_y_deg": tilt,
                           "T_camera_tcp": {"translation_m": contact.round(8).tolist(), "rotation_matrix": R.round(8).tolist()},
                           "pregrasp_tcp_camera_m": pre.round(8).tolist()})
    cv2.putText(canvas, "green: left finger +object X | red: right finger -object X", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "green: left finger +object X | red: right finger -object X", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    overlay = output / "grasp_prediction_overlay.png"
    cv2.imwrite(str(overlay), canvas)
    details = output / "grasp_prediction_camera_poses.json"
    details.write_text(json.dumps({"prediction_json": str(args.prediction_json.expanduser().resolve()),
                                   "left_finger_axis_camera": jaw.round(8).tolist(),
                                   "right_finger_axis_camera": (-jaw).round(8).tolist(),
                                   "candidates": candidates, "overlay": str(overlay)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"GRASP_OVERLAY={overlay}\nGRASP_CAMERA_POSES={details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
