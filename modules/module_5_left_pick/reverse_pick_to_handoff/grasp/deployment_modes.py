"""反面抓取分支的现场模式绑定与标定一致性校验。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[3]
CALIBRATION_DIR = PROJECT_ROOT / "tools/calibration/head_d435"
MODULE_4_1_CONFIG_DIR = (
    PROJECT_ROOT
    / "modules/module_4_head_judge/module_4_1_buffer_state_estimation/configs"
)


@dataclass(frozen=True)
class DeploymentMode:
    """一套不可拆分的现场硬件、相机和坐标标定参数。"""

    name: str
    camera_serial: str
    camera_width: int
    camera_height: int
    camera_fps: int
    calibration_image_width: int
    calibration_image_height: int
    calibration_kind: str
    body_head_calibration: Path
    tabletop_workspace: Path
    workpiece_2_tabletop_workspace: Path
    workpiece_2_camera_width: int
    workpiece_2_camera_height: int
    arm_host: str
    gripper_host: str
    force_sensor_host: str
    head_host: str
    head_feedback_pitch_sign: int
    workpiece_1_body_z_down_m: float
    workpiece_1_grasp_config: Path
    planner_host: str = "127.0.0.1"


MODES = {
    "gzhu": DeploymentMode(
        name="gzhu",
        camera_serial="261222076212",
        camera_width=1280,
        camera_height=720,
        camera_fps=30,
        calibration_image_width=1280,
        calibration_image_height=720,
        calibration_kind="sdk_bridge",
        body_head_calibration=CALIBRATION_DIR / "gzhu_reverse_observation.json",
        tabletop_workspace=MODULE_4_1_CONFIG_DIR / "gzhu_workpiece_1_workspace.json",
        workpiece_2_tabletop_workspace=MODULE_4_1_CONFIG_DIR / "workpiece_2_workspace.json",
        workpiece_2_camera_width=1280,
        workpiece_2_camera_height=720,
        arm_host="192.168.88.1",
        gripper_host="127.0.0.1",
        force_sensor_host="127.0.0.1",
        head_host="127.0.0.1",
        head_feedback_pitch_sign=-1,
        workpiece_1_body_z_down_m=0.020,
        workpiece_1_grasp_config=MODULE_DIR / "config/gzhu_workpiece_1_grasp.json",
    ),
    "xiangxin": DeploymentMode(
        name="xiangxin",
        camera_serial="261722072445",
        camera_width=848,
        camera_height=480,
        camera_fps=30,
        calibration_image_width=1280,
        calibration_image_height=720,
        calibration_kind="planner_direct",
        body_head_calibration=CALIBRATION_DIR / "xiangxin_reverse_observation.json",
        tabletop_workspace=MODULE_4_1_CONFIG_DIR / "xiangxin_workpiece_1_workspace.json",
        workpiece_2_tabletop_workspace=MODULE_4_1_CONFIG_DIR / "workpiece_2_workspace.json",
        workpiece_2_camera_width=1280,
        workpiece_2_camera_height=720,
        arm_host="192.168.3.11",
        gripper_host="192.168.3.11",
        force_sensor_host="192.168.3.22",
        head_host="192.168.3.22",
        head_feedback_pitch_sign=1,
        workpiece_1_body_z_down_m=0.005,
        workpiece_1_grasp_config=MODULE_DIR / "config/workpiece_1_grasp.json",
    ),
}


def load_mode(name: str) -> DeploymentMode:
    """加载模式，并在任何硬件访问前验证相机与标定没有串用。"""
    try:
        mode = MODES[name]
    except KeyError as exc:
        raise ValueError(f"--mode 必须是 gzhu 或 xiangxin，当前为 {name!r}") from exc
    path = mode.body_head_calibration
    if not path.is_file():
        raise FileNotFoundError(f"模式 {name} 的头部手眼标定不存在: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "mode": mode.name,
        "camera_serial": mode.camera_serial,
        "calibration_type": mode.calibration_kind,
        "image_size": [mode.calibration_image_width, mode.calibration_image_height],
    }
    mismatches = [
        f"{key}={payload.get(key)!r}（应为 {value!r}）"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    capture = payload.get("capture") or {}
    expected_capture = {
        "fps": mode.camera_fps,
        "workpiece_1_image_size": [mode.camera_width, mode.camera_height],
        "workpiece_2_image_size": [
            mode.workpiece_2_camera_width, mode.workpiece_2_camera_height,
        ],
    }
    mismatches.extend(
        f"capture.{key}={capture.get(key)!r}（应为 {value!r}）"
        for key, value in expected_capture.items()
        if capture.get(key) != value
    )
    matrix = payload.get("T_body_base_head_camera")
    if not isinstance(matrix, list) or len(matrix) != 4 or any(
        not isinstance(row, list) or len(row) != 4 for row in matrix
    ):
        mismatches.append("T_body_base_head_camera 不是 4x4 矩阵")
    elif any(not math.isfinite(float(value)) for row in matrix for value in row):
        mismatches.append("T_body_base_head_camera 包含非有限数值")
    elif [float(value) for value in matrix[3]] != [0.0, 0.0, 0.0, 1.0]:
        mismatches.append("T_body_base_head_camera 齐次矩阵末行无效")
    pose = payload.get("head_pose_deg")
    if not isinstance(pose, dict) or not {"pitch", "yaw"} <= pose.keys():
        mismatches.append("head_pose_deg 缺少 pitch/yaw")
    if mismatches:
        raise RuntimeError(f"模式 {name} 标定绑定错误: " + "; ".join(mismatches))
    for label, workspace in (
        ("工件1", mode.tabletop_workspace),
        ("工件2", mode.workpiece_2_tabletop_workspace),
    ):
        if not workspace.is_file():
            raise FileNotFoundError(f"模式 {name} 的{label}桌面 ROI 不存在: {workspace}")
    if not mode.workpiece_1_grasp_config.is_file():
        raise FileNotFoundError(
            f"模式 {name} 的工件1抓取点配置不存在: {mode.workpiece_1_grasp_config}"
        )
    grasp_config = json.loads(mode.workpiece_1_grasp_config.read_text(encoding="utf-8"))
    if grasp_config.get("mode") != name:
        raise RuntimeError(
            f"模式 {name} 与工件1抓取点配置不匹配: {mode.workpiece_1_grasp_config}"
        )
    return mode


def apply_mode_environment(mode: DeploymentMode, workpiece: str) -> None:
    """把模式端点写入子进程环境，确保 PlannerRuntime 也使用同一现场。"""
    if workpiece == "1":
        width, height = mode.camera_width, mode.camera_height
        workspace = mode.tabletop_workspace
    elif workpiece == "2":
        width, height = mode.workpiece_2_camera_width, mode.workpiece_2_camera_height
        workspace = mode.workpiece_2_tabletop_workspace
    else:
        raise ValueError(f"workpiece 必须是 '1' 或 '2'，当前为 {workpiece!r}")
    values = {
        "PARM_DEPLOYMENT_MODE": mode.name,
        "PARM_ARM_HOST": mode.arm_host,
        "PARM_GRIPPER_HOST": mode.gripper_host,
        "PARM_FORCE_SENSOR_HOST": mode.force_sensor_host,
        "PARM_HEAD_HOST": mode.head_host,
        "PARM_PLANNER_HOST": mode.planner_host,
        "PARM_HEAD_D435_SERIAL": mode.camera_serial,
        "PARM_HEAD_D435_WIDTH": str(width),
        "PARM_HEAD_D435_HEIGHT": str(height),
        "PARM_HEAD_D435_FPS": str(mode.camera_fps),
        "PARM_HEAD_FEEDBACK_PITCH_SIGN": str(mode.head_feedback_pitch_sign),
        "PARM_BODY_HEAD_CALIBRATION": str(mode.body_head_calibration),
        "PARM_HEAD_TABLETOP_WORKSPACE": str(workspace),
    }
    os.environ.update(values)


def calibration_payload(mode: DeploymentMode) -> dict[str, Any]:
    """返回已经通过模式一致性检查的标定 JSON。"""
    load_mode(mode.name)
    return json.loads(mode.body_head_calibration.read_text(encoding="utf-8"))
