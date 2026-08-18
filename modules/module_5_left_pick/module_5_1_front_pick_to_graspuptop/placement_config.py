"""读取并校验从旧 Module 3 迁入的工件放置 TOML 参数。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "1": HERE / "workpiece_1_placement_parameters.toml",
    "2": HERE / "workpiece_2_placement_parameters.toml",
}


def _parse_value(path: Path, line_number: int, raw_value: str):
    value = raw_value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"TOML 配置无法解析: {path}:{line_number}: {raw_value!r}"
        ) from exc


def _load_toml(path: Path) -> dict:
    """读取旧配置使用的受限 TOML 子集，兼容项目的 Python 3.10 环境。"""
    payload: dict[str, dict] = {}
    current: dict | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            if not section_name or "." in section_name or section_name in payload:
                raise ValueError(
                    f"TOML 配置分组无效: {path}:{line_number}: {line!r}"
                )
            current = {}
            payload[section_name] = current
            continue
        if current is None or "=" not in line:
            raise ValueError(
                f"TOML 配置只支持分组内 key = value: {path}:{line_number}"
            )
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key or key in current:
            raise ValueError(
                f"TOML 配置键无效或重复: {path}:{line_number}: {key!r}"
            )
        current[key] = _parse_value(path, line_number, raw_value)
    return payload


@dataclass(frozen=True)
class PlacementParameters:
    """经过校验的放置参数；字段名与旧 Module 3 保持一致。"""

    BODY_BASE_Z_FACE_SELECTION_MAX_ANGLE_DEG: float
    ALLOWED_HANDOFF_WEIGHT_CLASSES: tuple[str, ...]
    USE_MULTIPLE_HOLDING_MIN_G: bool
    ROTATION_CONVENTION: str
    TARGET_XY_RADIUS_M: float
    XY_RADIUS_FACTORS: list[float]
    FRONT_TARGET_OBJECT_XYZ_M: list[float]
    FRONT_TARGET_Z_CANDIDATES_M: list[float]
    FRONT_RELEASE_Z_CANDIDATES_M: list[float]
    FRONT_YAW_CANDIDATES_DEG: list[float]
    FRONT_ROLL_CANDIDATES_DEG: list[float]
    FRONT_PITCH_CANDIDATES_DEG: list[float]
    FRONT_BASE_RPY_DEG: list[float]
    REVERSED_TARGET_OBJECT_XYZ_M: list[float]
    REVERSED_TARGET_Z_CANDIDATES_M: list[float]
    REVERSED_RELEASE_Z_CANDIDATES_M: list[float]
    REVERSED_YAW_CANDIDATES_DEG: list[float]
    REVERSED_ROLL_CANDIDATES_DEG: list[float]
    REVERSED_PITCH_CANDIDATES_DEG: list[float]
    REVERSED_BASE_RPY_DEG: list[float]
    DIRECT_OBJECT_TARGET_XYZ_M: list[float]
    DIRECT_OBJECT_TARGET_Z_CANDIDATES_M: list[float]
    DIRECT_OBJECT_RELEASE_Z_CANDIDATES_M: list[float]
    DIRECT_LEFT_EEF_TARGET_XYZ_M: list[float]
    DIRECT_LEFT_EEF_TARGET_Z_CANDIDATES_M: list[float]
    DIRECT_LEFT_EEF_RELEASE_Z_CANDIDATES_M: list[float]


def _numbers(
    section: dict, key: str, *, count: int | None = None, allow_empty: bool = False
) -> list[float]:
    raw = section.get(key)
    if not isinstance(raw, list) or (not raw and not allow_empty):
        qualifier = "数值列表" if allow_empty else "非空数值列表"
        raise ValueError(f"放置配置 {key} 必须是{qualifier}")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"放置配置 {key} 包含非有限数值")
    if count is not None and len(values) != count:
        raise ValueError(f"放置配置 {key} 必须恰好有 {count} 个数值")
    return values


def _section(payload: dict, name: str) -> dict:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"放置配置缺少 [{name}] 分组")
    return value


def load_workpiece_parameters(workpiece: str) -> PlacementParameters:
    """读取指定工件的 TOML，并在生成规划候选前完成校验。"""
    try:
        path = CONFIG_PATHS[str(workpiece)]
    except KeyError as exc:
        raise ValueError(f"未知工件类型: {workpiece!r}") from exc
    payload = _load_toml(path)

    shared = _section(payload, "shared")
    front = _section(payload, "front")
    reversed_face = _section(payload, "reversed")
    direct_object_place = _section(payload, "direct object place")
    direct_left_eef_place = _section(payload, "direct left eef place")
    convention = shared.get("rotation_convention")
    if convention != "local_rpy":
        raise ValueError(
            "放置配置 rotation_convention 目前只支持 'local_rpy'，"
            f"实际为 {convention!r}"
        )
    weights = shared.get("allowed_handoff_weight_classes")
    if (
        not isinstance(weights, list)
        or not weights
        or not all(isinstance(value, str) for value in weights)
    ):
        raise ValueError("放置配置 allowed_handoff_weight_classes 必须是非空字符串列表")
    return PlacementParameters(
        BODY_BASE_Z_FACE_SELECTION_MAX_ANGLE_DEG=float(
            shared["body_base_z_face_selection_max_angle_deg"]
        ),
        ALLOWED_HANDOFF_WEIGHT_CLASSES=tuple(weights),
        USE_MULTIPLE_HOLDING_MIN_G=bool(shared["use_multiple_holding_min_g"]),
        ROTATION_CONVENTION=convention,
        TARGET_XY_RADIUS_M=float(shared["target_xy_radius_m"]),
        XY_RADIUS_FACTORS=_numbers(shared, "xy_radius_factors"),
        FRONT_TARGET_OBJECT_XYZ_M=_numbers(front, "target_object_xyz_m", count=3),
        FRONT_TARGET_Z_CANDIDATES_M=_numbers(front, "target_z_candidates_m"),
        FRONT_RELEASE_Z_CANDIDATES_M=_numbers(
            front, "release_z_candidates_m", allow_empty=True
        ),
        FRONT_YAW_CANDIDATES_DEG=_numbers(front, "yaw_candidates_deg"),
        FRONT_ROLL_CANDIDATES_DEG=_numbers(front, "roll_candidates_deg"),
        FRONT_PITCH_CANDIDATES_DEG=_numbers(front, "pitch_candidates_deg"),
        FRONT_BASE_RPY_DEG=_numbers(front, "base_rpy_deg", count=3),
        REVERSED_TARGET_OBJECT_XYZ_M=_numbers(
            reversed_face, "target_object_xyz_m", count=3
        ),
        REVERSED_TARGET_Z_CANDIDATES_M=_numbers(
            reversed_face, "target_z_candidates_m"
        ),
        REVERSED_RELEASE_Z_CANDIDATES_M=_numbers(
            reversed_face, "release_z_candidates_m", allow_empty=True
        ),
        REVERSED_YAW_CANDIDATES_DEG=_numbers(reversed_face, "yaw_candidates_deg"),
        REVERSED_ROLL_CANDIDATES_DEG=_numbers(reversed_face, "roll_candidates_deg"),
        REVERSED_PITCH_CANDIDATES_DEG=_numbers(
            reversed_face, "pitch_candidates_deg"
        ),
        REVERSED_BASE_RPY_DEG=_numbers(reversed_face, "base_rpy_deg", count=3),
        DIRECT_OBJECT_TARGET_XYZ_M=_numbers(
            direct_object_place, "target_object_xyz_m", count=3
        ),
        DIRECT_OBJECT_TARGET_Z_CANDIDATES_M=_numbers(
            direct_object_place, "target_z_candidates_m"
        ),
        DIRECT_OBJECT_RELEASE_Z_CANDIDATES_M=_numbers(
            direct_object_place, "release_z_candidates_m", allow_empty=True
        ),
        DIRECT_LEFT_EEF_TARGET_XYZ_M=_numbers(
            direct_left_eef_place, "target_left_eef_xyz_m", count=3
        ),
        DIRECT_LEFT_EEF_TARGET_Z_CANDIDATES_M=_numbers(
            direct_left_eef_place, "target_z_candidates_m"
        ),
        DIRECT_LEFT_EEF_RELEASE_Z_CANDIDATES_M=_numbers(
            direct_left_eef_place, "release_z_candidates_m", allow_empty=True
        ),
    )

