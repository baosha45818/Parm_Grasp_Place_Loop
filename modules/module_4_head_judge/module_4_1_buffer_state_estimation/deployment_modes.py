"""Module 4-1 现场模式：原子绑定头部相机、标定和采集参数。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[2]
CALIBRATION_ROOT = PROJECT_ROOT / "tools/calibration/head_d435"
CONFIG_ROOT = MODULE_ROOT / "configs"


@dataclass(frozen=True)
class HeadD435Mode:
    name: str
    camera_serial: str
    calibration_path: Path
    calibration_type: str
    calibration_image_size: tuple[int, int]
    capture_image_sizes: dict[str, tuple[int, int]]
    fps: int
    workspace_paths: dict[str, Path]

    def image_size(self, workpiece_id: str | int) -> tuple[int, int]:
        key = str(workpiece_id)
        try:
            return self.capture_image_sizes[key]
        except KeyError as exc:
            raise ValueError(f"不支持的工件编号: {key}") from exc

    def workspace_path(self, workpiece_id: str | int) -> Path:
        key = str(workpiece_id)
        try:
            return self.workspace_paths[key]
        except KeyError as exc:
            raise ValueError(f"不支持的工件编号: {key}") from exc


_MODE_FILES = {
    "xiangxin": CALIBRATION_ROOT / "xiangxin_reverse_observation.json",
    "gzhu": CALIBRATION_ROOT / "gzhu_reverse_observation.json",
}


def _finite_transform(payload: dict[str, Any], path: Path) -> None:
    matrix = payload.get("T_body_base_head_camera")
    if not isinstance(matrix, list) or len(matrix) != 4 or any(
        not isinstance(row, list) or len(row) != 4 for row in matrix
    ):
        raise RuntimeError(f"头部标定不是 4x4 齐次矩阵: {path}")
    try:
        values = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"头部标定矩阵包含非数值: {path}") from exc
    if not all(math.isfinite(value) for row in values for value in row):
        raise RuntimeError(f"头部标定矩阵包含非有限值: {path}")
    if values[3] != [0.0, 0.0, 0.0, 1.0]:
        raise RuntimeError(f"头部标定齐次矩阵末行无效: {path}")


def load_mode(name: str) -> HeadD435Mode:
    """读取并验证一套不可拆分的现场相机/标定配置。"""
    try:
        path = _MODE_FILES[name]
    except KeyError as exc:
        raise ValueError(f"--mode 必须是 xiangxin 或 gzhu，当前为 {name!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"现场 {name} 的头部 D435 标定不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != name:
        raise RuntimeError(f"现场名与标定内容不一致: {path}")
    serial = str(payload.get("camera_serial", "")).strip()
    if not serial:
        raise RuntimeError(f"头部标定缺少 camera_serial: {path}")
    _finite_transform(payload, path)
    try:
        calibration_size = tuple(int(v) for v in payload["image_size"])
        capture = payload["capture"]
        capture_sizes = {
            key: tuple(int(v) for v in capture[f"workpiece_{key}_image_size"])
            for key in ("1", "2")
        }
        fps = int(capture["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"头部标定缺少有效采集参数: {path}") from exc
    if len(calibration_size) != 2 or any(len(size) != 2 for size in capture_sizes.values()):
        raise RuntimeError(f"image_size 必须为 [width, height]: {path}")
    if min(*calibration_size, fps, *(v for size in capture_sizes.values() for v in size)) < 1:
        raise RuntimeError(f"头部采集尺寸或 fps 无效: {path}")
    workspace_paths = {
        "1": CONFIG_ROOT / f"{name}_workpiece_1_workspace.json",
        "2": CONFIG_ROOT / "workpiece_2_workspace.json",
    }
    missing = [workspace for workspace in workspace_paths.values() if not workspace.is_file()]
    if missing:
        raise FileNotFoundError("Module 4-1 桌面工作区配置不存在: " + ", ".join(map(str, missing)))
    return HeadD435Mode(
        name=name,
        camera_serial=serial,
        calibration_path=path.resolve(),
        calibration_type=str(payload.get("calibration_type", "")),
        calibration_image_size=(calibration_size[0], calibration_size[1]),
        capture_image_sizes=capture_sizes,
        fps=fps,
        workspace_paths=workspace_paths,
    )


def apply_mode_environment(mode: HeadD435Mode, workpiece_id: str | int) -> None:
    """让所有 Module 4-1 子进程使用同一现场，禁止标定与相机串用。"""
    width, height = mode.image_size(workpiece_id)
    os.environ.update({
        "PARM_DEPLOYMENT_MODE": mode.name,
        "PARM_HEAD_D435_SERIAL": mode.camera_serial,
        "PARM_HEAD_D435_WIDTH": str(width),
        "PARM_HEAD_D435_HEIGHT": str(height),
        "PARM_HEAD_D435_FPS": str(mode.fps),
        "PARM_BODY_HEAD_CALIBRATION": str(mode.calibration_path),
        "PARM_HEAD_TABLETOP_WORKSPACE": str(mode.workspace_path(workpiece_id)),
    })


def calibration_payload(mode: HeadD435Mode) -> dict[str, Any]:
    return json.loads(mode.calibration_path.read_text(encoding="utf-8"))
