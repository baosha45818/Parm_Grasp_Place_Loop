"""定义拆分后 Module 2 Inspect 与 Module 3 Left Place 的文件协议。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TERMINAL_OUTCOMES = {
    "success",
    "insufficient_point_cloud_discarded",
    "pc1_pc2_registration_failed_recovered",
    "held_object_dropped",
    "planning_timeout_recovered_at_graspup",
}


def write_stage_result(path: Path, payload: dict[str, Any]) -> None:
    """原子写入结果，避免总控读到半个 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": SCHEMA_VERSION, **payload}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_stage_result(path: Path, *, expected_stage: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"不支持的阶段结果版本: {payload.get('schema_version')!r}")
    if payload.get("stage") != expected_stage:
        raise RuntimeError(
            f"阶段结果来源错误: expected={expected_stage!r}, "
            f"actual={payload.get('stage')!r}"
        )
    if payload.get("outcome") not in TERMINAL_OUTCOMES:
        raise RuntimeError(f"未知阶段结果: {payload.get('outcome')!r}")
    return payload


def validate_inspection_success(payload: dict[str, Any]) -> None:
    if payload.get("outcome") != "success":
        raise RuntimeError(f"Inspect 未成功，不能进入 Left Place: {payload}")
    for name in (
        "T_body_base_object",
        "T_tool_object",
        "T_body_left_eef_at_inspect",
        "T_left_eef_object",
    ):
        matrix = payload.get(name)
        if (
            not isinstance(matrix, list)
            or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
        ):
            raise RuntimeError(f"Inspect 结果缺少合法{name}")
