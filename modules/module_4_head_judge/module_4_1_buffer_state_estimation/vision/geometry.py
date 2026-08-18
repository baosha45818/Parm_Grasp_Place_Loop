"""工件 1/2 共用的 RGB-D、桌面 ROI 与 SAM2 prompt 几何工具。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def camera_matrix(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """从采集 metadata 构造彩色相机内参与畸变参数。"""
    intrinsics = metadata["color_intrinsics"]
    matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(intrinsics.get("coeffs", [0.0] * 5), dtype=np.float64)
    return matrix, coefficients


def depth_to_xyz(depth: np.ndarray, matrix: np.ndarray, scale_m: float) -> np.ndarray:
    """把与彩色图对齐的 Z16 深度图转换为逐像素相机坐标。"""
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    z = depth.astype(np.float64) * scale_m
    xyz = np.empty((*depth.shape, 3), dtype=np.float64)
    xyz[..., 0] = (columns - matrix[0, 2]) * z / matrix[0, 0]
    xyz[..., 1] = (rows - matrix[1, 2]) * z / matrix[1, 1]
    xyz[..., 2] = z
    xyz[depth <= 0] = np.nan
    return xyz


def workspace_roi_mask(workspace: dict, image_shape: tuple[int, ...]) -> np.ndarray:
    """按当前分辨率栅格化标定的桌面安全多边形。"""
    height, width = image_shape[:2]
    reference_width, reference_height = workspace["reference_color_size_wh"]
    polygon = np.asarray(workspace["safe_roi_polygon_px"], dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("safe_roi_polygon_px must contain at least three [x, y] points")
    polygon = np.rint(
        polygon * [width / reference_width, height / reference_height]
    ).astype(np.int32)
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi, [polygon], 1)
    return roi


def farthest_points(binary: np.ndarray, count: int) -> list[list[int]]:
    """在二值区域中以“中心点 + 最远点采样”生成分散提示点。"""
    rows, columns = np.nonzero(binary)
    points = np.column_stack((columns, rows))
    if not len(points):
        return []
    take = min(int(count), len(points))
    mean = points.mean(axis=0)
    chosen = [int(np.argmin(np.sum((points - mean) ** 2, axis=1)))]
    nearest = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(1, take):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
    return [[int(value) for value in points[index]] for index in chosen]


def retain_prompt_components(
    mask: np.ndarray,
    positives: list[list[int]],
    seed_mask: np.ndarray,
) -> np.ndarray:
    """只保留包含正提示的 SAM2 连通域，并提供 seed-overlap 回退。"""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    selected = {
        int(labels[y, x])
        for x, y in positives
        if 0 <= x < labels.shape[1]
        and 0 <= y < labels.shape[0]
        and labels[y, x] > 0
    }
    if not selected and count > 1:
        overlaps = [
            int(np.count_nonzero((labels == label) & seed_mask))
            for label in range(1, count)
        ]
        selected = {1 + int(np.argmax(overlaps))}
    return (
        np.isin(labels, list(selected))
        if selected
        else np.zeros_like(mask, dtype=bool)
    )


def load_transform(path: Path) -> np.ndarray:
    """读取兼容的新旧头相机外参字段。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = next(
        (
            candidate
            for candidate in (
                "T_body_base_head_camera",
                "T_base_head_camera",
                "T",
            )
            if candidate in payload
        ),
        None,
    )
    if key is None:
        raise ValueError(f"头部标定缺少 base <- head_camera 变换: {path}")
    transform = np.asarray(payload[key], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"无效的 body/head 相机外参: {path}")
    return transform

