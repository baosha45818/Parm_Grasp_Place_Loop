"""读取两类工件左臂反抓和右臂交接的 TOML 预设位置。"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POSITIONS_PATH = Path(__file__).with_name("reverse_grasp_handoff_positions.toml")


def _parse_toml_subset(path: Path) -> dict[str, Any]:
    """解析本项目位置文件所需的 TOML 子集，兼容 Python 3.10。"""
    result: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = result
            for part in line[1:-1].split("."):
                if not part or not part.replace("_", "").isalnum():
                    raise RuntimeError(f"无效 TOML 分区 {path}:{line_number}")
                section = section.setdefault(part, {})
            continue
        if section is None or "=" not in line:
            raise RuntimeError(f"无效 TOML 配置行 {path}:{line_number}")
        key, raw_value = (item.strip() for item in line.split("=", 1))
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f"无法解析 TOML 值 {path}:{line_number}") from exc
        section[key] = value
    return result


def _vector(section: dict[str, Any], key: str, length: int, label: str) -> list[float]:
    raw = section.get(key)
    if not isinstance(raw, list) or len(raw) != length:
        raise RuntimeError(f"{label}.{key} 必须包含 {length} 个数值")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{label}.{key} 包含非有限数值")
    return values


def _positive(section: dict[str, Any], key: str, label: str) -> float:
    value = float(section.get(key, float("nan")))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label}.{key} 必须为正数")
    return value


def load_workpiece_handoff_profile(
    workpiece: str,
    path: Path = POSITIONS_PATH,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """返回按部署模式绑定的工件反抓/交接配置。

    关节运动的公共部分仍按工件维护；云台角、手眼标定、自动标点以及少数
    现场专用右臂位置必须由 ``xiangxin``/``gzhu`` 分开选择，禁止静默混用。
    """
    if workpiece not in {"1", "2"}:
        raise ValueError(f"workpiece 必须是 '1' 或 '2'，当前为 {workpiece!r}")
    resolved = path.expanduser().resolve()
    raw = _parse_toml_subset(resolved)
    selected_mode = (mode or os.environ.get("PARM_DEPLOYMENT_MODE", "xiangxin")).strip()
    if selected_mode not in {"xiangxin", "gzhu"}:
        raise ValueError(
            "反抓交接配置的 mode 必须是 xiangxin 或 gzhu，"
            f"当前为 {selected_mode!r}"
        )
    deployment = ((raw.get("deployment") or {}).get(selected_mode) or {})
    deployment_observation = dict(deployment.get("left_observation") or {})
    if not deployment_observation:
        raise RuntimeError(f"位置配置缺少 deployment.{selected_mode}.left_observation")
    section = (((raw.get(f"workpiece_{workpiece}") or {})))
    if not isinstance(section, dict):
        raise RuntimeError(f"位置配置缺少 workpiece_{workpiece}: {resolved}")

    left_ready = dict(section.get("left_reverse_grasp") or {})
    left_label = f"workpiece_{workpiece}.left_reverse_grasp"
    left_ready["ready_joints_deg"] = _vector(left_ready, "ready_joints_deg", 7, left_label)
    left_ready["arrival_tolerance_deg"] = _positive(left_ready, "arrival_tolerance_deg", left_label)
    left_ready["arrival_timeout_s"] = _positive(left_ready, "arrival_timeout_s", left_label)
    left_ready["post_close_lift_m"] = _positive(left_ready, "post_close_lift_m", left_label)
    max_attempts = left_ready.get("max_grasp_attempts")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise RuntimeError(f"{left_label}.max_grasp_attempts 必须为不小于 1 的整数")
    left_ready["max_grasp_attempts"] = max_attempts

    left_handoff = dict(section.get("left_handoff") or {})
    handoff_label = f"workpiece_{workpiece}.left_handoff"
    for key in ("prepare_joints_deg", "observation_joints_deg", "left_return_joints_deg"):
        left_handoff[key] = _vector(left_handoff, key, 7, handoff_label)
    left_handoff["handoff_center_body_m"] = _vector(left_handoff, "handoff_center_body_m", 3, handoff_label)
    for key in ("axis_tolerance_deg", "center_tolerance_mm"):
        left_handoff[key] = _positive(left_handoff, key, handoff_label)
    deployment_left = dict(
        deployment.get(f"workpiece_{workpiece}_left_handoff") or {}
    )
    # 部署记录可冻结一整套已实测交接点；它覆盖公共工件字段，避免后续为另一
    # 个现场调整公共参数时污染该现场的已验证轨迹。
    for key in (
        "prepare_joints_deg", "observation_joints_deg", "left_return_joints_deg",
    ):
        if key in deployment_left:
            left_handoff[key] = _vector(
                deployment_left, key, 7,
                f"deployment.{selected_mode}.workpiece_{workpiece}_left_handoff",
            )
    if "handoff_center_body_m" in deployment_left:
        left_handoff["handoff_center_body_m"] = _vector(
            deployment_left, "handoff_center_body_m", 3,
            f"deployment.{selected_mode}.workpiece_{workpiece}_left_handoff",
        )
    for key in ("axis_tolerance_deg", "center_tolerance_mm"):
        if key in deployment_left:
            left_handoff[key] = _positive(
                deployment_left, key,
                f"deployment.{selected_mode}.workpiece_{workpiece}_left_handoff",
            )
    # 云台角/标定不是工件参数。只允许从 deployment.<mode> 读取，避免旧配置
    # 把广珠和祥鑫的观察位、自动标点合并为同一组值。
    try:
        left_handoff["head_pitch_deg"] = float(deployment_observation["head_pitch_deg"])
        left_handoff["head_yaw_deg"] = float(deployment_observation["head_yaw_deg"])
        deployment_alignment = ROOT / str(deployment_observation["alignment"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"deployment.{selected_mode}.left_observation 缺少有效云台角或标定"
        ) from exc
    if not deployment_alignment.is_file():
        raise FileNotFoundError(
            f"deployment.{selected_mode}.left_observation.alignment 不存在: "
            f"{deployment_alignment}"
        )
    left_handoff["alignment"] = deployment_alignment.resolve()
    auto_mask_config = ROOT / str(deployment_observation.get("auto_mask_config", ""))
    if not auto_mask_config.is_file():
        raise FileNotFoundError(
            f"deployment.{selected_mode}.left_observation.auto_mask_config 不存在: "
            f"{auto_mask_config}"
        )

    right_handoff = dict(section.get("right_handoff") or {})
    right_label = f"workpiece_{workpiece}.right_handoff"
    for key in ("grasp_joints_deg", "place_ready_joints_deg"):
        right_handoff[key] = _vector(right_handoff, key, 7, right_label)
    if workpiece == "1":
        right_handoff["place_ready_tolerance_deg"] = _positive(right_handoff, "place_ready_tolerance_deg", right_label)
    else:
        for key in ("prepare_joints_deg", "pregrasp_joints_deg"):
            right_handoff[key] = _vector(right_handoff, key, 7, right_label)
        for key in ("post_release_clearance_minus_x_m", "post_release_clearance_plus_z_m"):
            right_handoff[key] = _positive(right_handoff, key, right_label)
    deployment_right = dict(
        deployment.get(f"workpiece_{workpiece}_right_handoff") or {}
    )
    for key in ("grasp_joints_deg", "place_ready_joints_deg"):
        if key in deployment_right:
            right_handoff[key] = _vector(
                deployment_right, key, 7,
                f"deployment.{selected_mode}.workpiece_{workpiece}_right_handoff",
            )
    if workpiece == "1" and "place_ready_tolerance_deg" in deployment_right:
        right_handoff["place_ready_tolerance_deg"] = _positive(
            deployment_right, "place_ready_tolerance_deg",
            f"deployment.{selected_mode}.workpiece_{workpiece}_right_handoff",
        )

    # 保持 Module 5-2/8-2 的原字段形状，避免因配置迁移改变运动逻辑。
    return {
        "schema_version": 1,
        "profile_id": (
            f"{selected_mode}_workpiece_{workpiece}_reverse_grasp_handoff_toml_v2"
        ),
        "deployment_mode": selected_mode,
        "workpiece": workpiece,
        "left_reverse_grasp": left_ready,
        "left_observation": {
            "prepare_joints_deg": list(left_handoff["prepare_joints_deg"]),
            "observation_joints_deg": list(left_handoff["observation_joints_deg"]),
            "head_pose_deg": {"pitch": left_handoff["head_pitch_deg"], "yaw": left_handoff["head_yaw_deg"]},
            "alignment": left_handoff["alignment"],
            "auto_mask_config": auto_mask_config.resolve(),
        },
        "handoff": {
            "center_body_m": list(left_handoff["handoff_center_body_m"]),
            "axis_tolerance_deg": left_handoff["axis_tolerance_deg"],
            "center_tolerance_mm": left_handoff["center_tolerance_mm"],
        },
        "right_handoff": right_handoff,
        "left_return_joints_deg": list(left_handoff["left_return_joints_deg"]),
        "source": resolved,
    }
