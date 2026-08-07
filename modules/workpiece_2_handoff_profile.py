"""兼容入口：工件2交接位置现统一从 TOML 配置读取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.reverse_grasp_handoff_positions import (
    POSITIONS_PATH,
    load_workpiece_handoff_profile,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = POSITIONS_PATH


def load_workpiece_2_handoff_profile(
    path: Path = PROFILE_PATH,
) -> dict[str, Any]:
    """返回工件2交接配置；path 参数仅保留旧调用兼容。"""
    if path.expanduser().resolve() != PROFILE_PATH.resolve():
        raise ValueError("工件2交接位置已统一到 reverse_grasp_handoff_positions.toml")
    return load_workpiece_handoff_profile("2")


def workpiece_2_right_prepare_joints_deg() -> list[float]:
    return list(
        load_workpiece_2_handoff_profile()["right_handoff"][
            "prepare_joints_deg"
        ]
    )
