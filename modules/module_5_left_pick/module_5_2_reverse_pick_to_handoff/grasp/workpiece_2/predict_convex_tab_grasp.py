#!/usr/bin/env python3
"""反面分支工件2：根据已确认的反面位姿投影固定 CAD 凸耳抓取点。

本步骤只消费 Module 4 的物体位姿，不控制机器人、云台或夹爪。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent


def output_path(path: Path) -> Path:
    """限制运行产物写在本抓取适配器目录内。"""
    path = path.expanduser().resolve()
    try:
        path.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须位于 {HERE}: {path}") from exc
    return path


def project(K: np.ndarray, point_camera: np.ndarray) -> tuple[int, int]:
    """把相机系三维点投影到彩色图像。"""
    if point_camera[2] <= 1e-6:
        raise RuntimeError("抓取点不在相机前方")
    pixel = K @ point_camera
    return int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))


def normalize(vector: np.ndarray) -> np.ndarray:
    """归一化方向向量。"""
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """去除 ICP 矩阵中的微小数值剪切。"""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    result = u @ vt
    if np.linalg.det(result) < 0.:
        u[:, -1] *= -1.
        result = u @ vt
    return result


def pose_from_module4_result(path: Path) -> tuple[dict, Path]:
    """提取 Module 4-1 已确认的唯一 BACK 位姿，不再运行第二次 ICP。"""
    result = json.loads(path.read_text(encoding="utf-8"))
    objects = result.get("objects") or []
    if (
        result.get("event") != "module_4_1_buffer_state_estimated"
        or str(result.get("workpiece_id")) != "2"
        or result.get("decision") != "BACK"
        or len(objects) != 1
    ):
        raise RuntimeError("工件2抓取只接受 Module 4-1 发布的唯一 BACK 结果")
    diagnostic = objects[0].get("diagnostics") or {}
    pose = diagnostic.get("pose") or {}
    if not pose.get("valid") or not pose.get("orientation_valid"):
        raise RuntimeError("Module 4-1 工件2位姿未通过质量门")
    transform = pose.get("T_head_camera_workpiece")
    table_normal = diagnostic.get("table_normal_camera")
    if transform is None or table_normal is None:
        raise RuntimeError("Module 4-1 工件2结果缺少相机系位姿或桌面法向")
    return {
        "T_camera_object": transform,
        "table_normal_camera": table_normal,
        "frame_id": "head_camera",
        "object_frame": "workpiece_2_semantic",
        "icp": diagnostic.get("icp") or {},
        "pose_confidence": diagnostic.get("pose_confidence") or {},
        "classification": diagnostic.get("classification"),
        "capture_dir": result["capture_dir"],
        "source_module4_result": str(path),
    }, Path(result["capture_dir"]).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--module4-result", type=Path, help="Module 4-1 正反+位姿报告")
    source.add_argument("--pose", type=Path, help="旧 fixed_reverse_pose.json 兼容入口")
    parser.add_argument("--capture-dir", type=Path, help="旧 pose 入口对应 RGB-D capture 目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grasp-down-offset-m", type=float, default=-.005,
                        help="实际夹爪 TCP 沿桌面法向偏移量；正值向下、负值向上，默认向上 5 mm")
    args = parser.parse_args()

    source_path = (args.module4_result or args.pose).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Module 4 位姿结果不存在: {source_path}")
    if args.module4_result:
        pose, capture = pose_from_module4_result(source_path)
    else:
        if args.capture_dir is None:
            parser.error("--pose 必须同时提供 --capture-dir")
        pose = json.loads(source_path.read_text(encoding="utf-8"))
        capture = args.capture_dir.expanduser().resolve()
    output = output_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    if not capture.is_dir():
        raise FileNotFoundError(f"Module 4 capture-dir 不存在: {capture}")
    output.mkdir(parents=True)
    if args.module4_result:
        normalized_pose = output / "module_4_1_object_pose.json"
        normalized_pose.write_text(
            json.dumps(pose, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        source_path = normalized_pose
    config = json.loads((HERE / "workpiece_2_grasp.json").read_text(encoding="utf-8"))
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("无法读取 rgb_000.png")

    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]])
    T = np.asarray(pose["T_camera_object"], dtype=float)
    T[:3, :3] = nearest_rotation(T[:3, :3])
    raw_table_normal = np.asarray(pose.get("table_normal_camera", []), dtype=float)
    if raw_table_normal.shape != (3,):
        raise RuntimeError("Module 4 位姿缺少 table_normal_camera")
    table_normal = normalize(raw_table_normal)
    point_object = np.asarray(config["grasp_point_object_m"], dtype=float)
    point_camera = T[:3, :3] @ point_object + T[:3, 3]
    pixel = project(K, point_camera)
    refinement = {"method": "cad_landmark_from_fixed_icp", "pixel": list(pixel)}
    if not (capture / "depth_aligned_000.png").is_file():
        raise RuntimeError("缺少 depth_aligned_000.png")
    if not -.030 <= args.grasp_down_offset_m <= .030:
        raise ValueError("grasp-down-offset-m 必须在 [-0.030, 0.030] m")
    surface_point_camera = point_camera.copy()
    surface_point_object = point_object.copy()
    # Use the fitted tabletop normal for physical down, exactly as Module 5-1
    # does. The normal is oriented table->camera, hence its negative is the
    # real approach/down direction.
    approach_camera = -table_normal
    point_camera = point_camera + approach_camera * float(args.grasp_down_offset_m)
    point_object = T[:3, :3].T @ (point_camera - T[:3, 3])
    point_camera = T[:3, :3] @ point_object + T[:3, 3]

    # Object -Z faces up/out of the table for this confirmed tabletop-reverse placement.
    outward_camera = -approach_camera
    visible = image.copy()
    cv2.circle(visible, pixel, 13, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(visible, pixel, 3, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(visible, "convex-tab grasp", (pixel[0] + 16, pixel[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(visible, "convex-tab grasp", (pixel[0] + 16, pixel[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1, cv2.LINE_AA)
    overlay = output / "convex_tab_grasp_overlay.png"
    if not cv2.imwrite(str(overlay), visible):
        raise RuntimeError(f"无法写入 {overlay}")
    payload = {
        "schema_version": 1,
        "event": "workpiece_2_convex_tab_grasp_prediction",
        "source_pose": str(source_path),
        "source_module4_result": (
            str(args.module4_result.expanduser().resolve())
            if args.module4_result else None
        ),
        "frame_id": pose["frame_id"],
        "object_frame": pose["object_frame"],
        "grasp_contact_point_object_m": point_object.round(8).tolist(),
        "grasp_contact_point_camera_m": point_camera.round(8).tolist(),
        "surface_point_object_m_before_down_offset": surface_point_object.round(8).tolist(),
        "surface_point_camera_m_before_down_offset": surface_point_camera.round(8).tolist(),
        "grasp_down_offset_m": float(args.grasp_down_offset_m),
        "image_pixel": list(pixel),
        "outward_direction_camera": outward_camera.round(8).tolist(),
        "approach_direction_camera": approach_camera.round(8).tolist(),
        "approach_direction_source": "negative_table_normal_camera",
        "placement_constraint": "confirmed tabletop reverse: object +Z points into the table; object -Z faces the camera/outward",
        "grasp_point_definition": config["grasp_point_definition"],
        "local_refinement": refinement,
        "artifacts": {"overlay": str(overlay), "pointcloud_npz": None,
                      "pointcloud_3d": None},
        "created_at": time.time(),
    }
    result = output / "convex_tab_grasp_prediction.json"
    result.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CONVEX_TAB_GRASP={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
