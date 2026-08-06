#!/usr/bin/env python3
"""从 workpiece_2 的 ICP 位姿生成并核验左臂反抓点。

本工具只读取已完成的 D435/SAM2/ICP ``result.json``，只在本目录写入
预测 JSON 和覆盖图；绝不连接 Planner、机械臂、夹爪或云台。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "workpiece_2_reverse_grasp.json"


def output_path(path: Path) -> Path:
    """Resolve an artifact path and keep every write inside this directory."""
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出文件必须位于 {HERE}：{resolved}") from exc
    return resolved


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("zero-length vector")
    return vector / length


def pose_record(rotation: np.ndarray, translation: np.ndarray) -> dict[str, list]:
    return {
        "translation_m": np.asarray(translation, float).round(8).tolist(),
        "rotation_matrix": np.asarray(rotation, float).round(8).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="workpiece_2 ICP result.json")
    parser.add_argument("--output", type=Path, default=HERE / "latest_prediction.json")
    parser.add_argument("--overlay", type=Path, default=HERE / "latest_grasp_overlay.png")
    parser.add_argument("--log", type=Path, default=HERE / "reverse_grasp_prediction.log.jsonl",
                        help="追加式预测日志；必须位于 reverse_grasp/ 内")
    parser.add_argument(
        "--table-normal-body", nargs=3, type=float, default=[0.0, 0.0, 1.0],
        metavar=("NX", "NY", "NZ"), help="body_base 中由桌面指向上方的单位法向；默认 +Z",
    )
    parser.add_argument("--confirm-grasp-point", action="store_true",
                        help="记录操作者已审阅本次覆盖图；仍不会执行任何运动")
    return parser.parse_args()


def load_result(path: Path, config: dict) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    model = result.get("model_frame", {})
    expected = Path(config["stl_path"]).resolve()
    actual = Path(str(model.get("stl_path", ""))).resolve()
    if actual != expected:
        raise RuntimeError(f"ICP 模型不匹配：期望 {expected}，得到 {actual}")
    if result.get("object_pose", {}).get("frame_id") != "body_base":
        raise RuntimeError("result.json 必须输出到 body_base，不能把相机系抓取点直接用于机械臂")
    matrix = np.asarray(result.get("T_output_object"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("result.json 缺少有效 T_output_object")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise RuntimeError("T_output_object 不是齐次变换")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) < 0.99:
        raise RuntimeError("T_output_object 的旋转部分无效")
    gate = config["quality_gate"]
    registration = result.get("registration", {})
    values = {
        "fitness": float(registration.get("fitness", -1)),
        "rmse_m": float(registration.get("rmse_m", float("inf"))),
        "mask_overlap_ratio": float(registration.get("mask_overlap_ratio", -1)),
    }
    if (values["fitness"] < gate["minimum_icp_fitness"]
            or values["rmse_m"] > gate["maximum_icp_rmse_m"]
            or values["mask_overlap_ratio"] < gate["minimum_mask_overlap_ratio"]):
        raise RuntimeError(f"ICP 质量门未通过：{values}，阈值={gate}")
    return result


def draw_overlay(result: dict, point_camera: np.ndarray, output: Path) -> str | None:
    """将预测点投影到原 ICP RGB 图；缺少图像时不影响 JSON 预测。"""
    try:
        import cv2
    except ImportError:
        return None
    report = result.get("point_report", {})
    intr = report.get("camera_intrinsics")
    if intr is None:
        return None
    result_path = Path(result["_source_path"])
    candidates = (result_path.parent / "head_d435_rgb.png", result_path.parent / "rgb_000.png")
    image_path = next((p for p in candidates if p.is_file()), None)
    if image_path is None or point_camera[2] <= 1e-6:
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    K = np.asarray(intr, dtype=float)
    u, v, w = K @ point_camera
    if abs(w) < 1e-9:
        return None
    xy = (int(round(u / w)), int(round(v / w)))
    cv2.circle(image, xy, 9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(image, xy, (0, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.putText(image, "LEFT GRASP TCP (review required)", (max(4, xy[0] - 180), max(25, xy[1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"无法写入覆盖图：{output}")
    return str(output.resolve())


def main() -> int:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result_path = args.result.expanduser().resolve()
    result = load_result(result_path, config)
    result["_source_path"] = str(result_path)

    T_body_object = np.asarray(result["T_output_object"], dtype=float)
    T_camera_object = np.asarray(result.get("T_camera_object"), dtype=float)
    if T_camera_object.shape != (4, 4):
        raise RuntimeError("result.json 缺少用于覆盖图的 T_camera_object")
    R_object_stl = np.asarray(config["R_object_stl"], dtype=float)
    point_stl = np.asarray(config["left_grasp_point_stl_m"], dtype=float)
    point_object = R_object_stl @ point_stl
    jaw_object = unit(R_object_stl @ np.asarray(config["jaw_opening_axis_stl"], dtype=float))
    point_body = T_body_object[:3, :3] @ point_object + T_body_object[:3, 3]
    point_camera = T_camera_object[:3, :3] @ point_object + T_camera_object[:3, 3]

    table_normal = unit(np.asarray(args.table_normal_body, dtype=float))
    approach_down = -table_normal
    jaw_body = T_body_object[:3, :3] @ jaw_object
    jaw_body = unit(jaw_body - approach_down * np.dot(jaw_body, approach_down))
    side_axis = unit(np.cross(approach_down, jaw_body))
    jaw_body = unit(np.cross(side_axis, approach_down))
    R_grasp = np.column_stack((approach_down, jaw_body, side_axis))
    if np.linalg.det(R_grasp) < 0:
        R_grasp[:, 1] *= -1
    pregrasp = point_body - approach_down * float(config["pregrasp_clearance_m"])

    overlay = draw_overlay(result, point_camera, output_path(args.overlay))
    payload = {
        "schema_version": 1,
        "event": "workpiece_2_reverse_grasp_predicted",
        "execution": "FORBIDDEN_BY_THIS_TOOL",
        "source_result": str(result_path),
        "workpiece": {
            "stl_path": config["stl_path"], "object_frame": config["object_frame"],
            "grasp_point_stl_m": point_stl.tolist(),
            "grasp_point_object_m": point_object.round(8).tolist(),
            "grasp_point_basis": config["left_grasp_point_basis"],
        },
        "icp_quality": {key: result["registration"][key] for key in ("fitness", "rmse_m", "mask_overlap_ratio")},
        "body_base": {
            "grasp_pose": pose_record(R_grasp, point_body),
            "pregrasp_pose": pose_record(R_grasp, pregrasp),
            "table_normal_up": table_normal.round(8).tolist(),
            "grasp_frame_convention": "+X=vertical_approach_down,+Y=jaw_opening,+Z=side",
        },
        "camera": {"grasp_point_m": point_camera.round(8).tolist(), "overlay": overlay},
        "operator_confirmation": bool(args.confirm_grasp_point),
        "created_at": time.time(),
    }
    output = output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log = output_path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "event": payload["event"],
            "created_at": payload["created_at"],
            "source_result": payload["source_result"],
            "output": str(output),
            "overlay": overlay,
            "operator_confirmation": payload["operator_confirmation"],
            "icp_quality": payload["icp_quality"],
            "grasp_point_body_m": payload["body_base"]["grasp_pose"]["translation_m"],
        }, ensure_ascii=False) + "\n")
    print(f"REVERSE_GRASP_PREDICTION={output}")
    print(f"LEFT_GRASP_POINT_BODY_M={payload['body_base']['grasp_pose']['translation_m']}")
    print("已生成预测；必须人工核验覆盖图。本工具不会、也不能执行抓取。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
