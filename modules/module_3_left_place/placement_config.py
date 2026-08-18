"""读取 Module3 唯一业务配置：固定 Buffer TCP 放置目标。"""

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


@dataclass(frozen=True)
class BufferPlacementParameters:
    """某类工件的 Buffer 水平目标与下探 Z 候选。"""
    target_left_eef_xyz_m: tuple[float, float, float]
    target_z_candidates_m: tuple[float, ...]
    release_z_candidates_m: tuple[float, ...]


def _parse_value(path: Path, line_number: int, raw_value: str):
    """安全解析一个配置字面量，并保留准确的报错位置。"""
    try:
        return ast.literal_eval(raw_value.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Buffer配置无法解析: {path}:{line_number}: {raw_value!r}"
        ) from exc


def _load_buffer_section(path: Path) -> dict:
    """读取受限的 ``[buffer]`` 配置段，拒绝未知段和重复键。"""
    section: str | None = None
    values: dict = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section != "buffer":
                raise ValueError(
                    f"Module 3配置只允许 [buffer]: {path}:{line_number}: {line!r}"
                )
            continue
        if section != "buffer" or "=" not in line:
            raise ValueError(
                f"Buffer配置只支持 [buffer] 下的 key = value: {path}:{line_number}"
            )
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key or key in values:
            raise ValueError(
                f"Buffer配置键无效或重复: {path}:{line_number}: {key!r}"
            )
        values[key] = _parse_value(path, line_number, raw_value)
    return values


def _numbers(
    values: dict,
    key: str,
    *,
    count: int | None = None,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    """读取并校验一个有限浮点数列表。"""
    raw = values.get(key)
    if not isinstance(raw, list) or (not raw and not allow_empty):
        qualifier = "数值列表" if allow_empty else "非空数值列表"
        raise ValueError(f"Buffer配置 {key} 必须是{qualifier}")
    parsed = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"Buffer配置 {key} 包含非有限数值")
    if count is not None and len(parsed) != count:
        raise ValueError(f"Buffer配置 {key} 必须恰好有 {count} 个数值")
    return parsed


def load_workpiece_parameters(workpiece: str) -> BufferPlacementParameters:
    """加载并完整校验指定工件的 Buffer 放置参数。"""
    try:
        path = CONFIG_PATHS[str(workpiece)]
    except KeyError as exc:
        raise ValueError(f"未知工件类型: {workpiece!r}") from exc
    values = _load_buffer_section(path)
    required = {
        "target_left_eef_xyz_m",
        "target_z_candidates_m",
        "release_z_candidates_m",
    }
    unexpected = set(values) - required
    missing = required - set(values)
    if unexpected or missing:
        raise ValueError(
            f"Buffer配置键不匹配: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    xyz = _numbers(values, "target_left_eef_xyz_m", count=3)
    return BufferPlacementParameters(
        target_left_eef_xyz_m=(xyz[0], xyz[1], xyz[2]),
        target_z_candidates_m=_numbers(values, "target_z_candidates_m"),
        release_z_candidates_m=_numbers(
            values, "release_z_candidates_m", allow_empty=True
        ),
    )
