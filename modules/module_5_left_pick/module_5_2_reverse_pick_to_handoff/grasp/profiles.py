"""反面抓取分支：加载并校验两类工件的抓取和预备位配置。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.grasp.toml_config import load_config_mapping


MODULE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = MODULE_DIR / "config"
DEFAULT_PROFILES = {
    "1": CONFIG_DIR / "xiangxin_workpiece_1_grasp.toml",
    "2": CONFIG_DIR / "workpiece_2.json",
}
EXPECTED_ADAPTERS = {
    "1": "unique_back_prediction",
    "2": "convex_tab_prediction",
}
REVERSE_READY_CONFIG = CONFIG_DIR / "reverse_ready.json"


@dataclass(frozen=True)
class ReverseGraspProfile:
    workpiece: str
    adapter: str
    path: Path
    perception: dict[str, Any]
    execution: dict[str, Any]


def _positive_number(section: dict[str, Any], key: str, label: str) -> float:
    value = float(section.get(key, float("nan")))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label}.{key} 必须为正数")
    return value


def _approach_angles(section: dict[str, Any], label: str) -> list[float]:
    raw = section.get("approach_angles_deg")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{label}.approach_angles_deg 必须是非空数组")
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) or not 0.0 < abs(value) <= 90.0 for value in values):
        raise RuntimeError(f"{label}.approach_angles_deg 必须在 [-90, 90] 内且不能为 0")
    return values


def load_reverse_grasp_profile(
    workpiece: str, profile_path: Path | None = None,
) -> ReverseGraspProfile:
    """Return one validated profile; relative asset paths are profile-relative."""
    if workpiece not in DEFAULT_PROFILES:
        raise ValueError(f"workpiece 必须是 '1' 或 '2'，当前为 {workpiece!r}")
    path = (profile_path or DEFAULT_PROFILES[workpiece]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"反抓配置不存在: {path}")
    payload = load_config_mapping(path)
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError(f"不支持的反抓配置版本: {path}")
    if str(payload.get("workpiece")) != workpiece:
        raise RuntimeError(f"反抓配置工件编号与 --workpiece 不一致: {path}")
    adapter = str(payload.get("adapter") or "")
    if adapter != EXPECTED_ADAPTERS[workpiece]:
        raise RuntimeError(
            f"workpiece {workpiece} 必须使用 adapter={EXPECTED_ADAPTERS[workpiece]!r}"
        )
    perception = dict(payload.get("perception") or {})
    execution = dict(payload.get("execution") or {})
    execution["approach_angles_deg"] = _approach_angles(execution, "execution")
    if workpiece == "1":
        orientation = (path.parent / str(perception.get("orientation_config", ""))).resolve()
        if not orientation.is_file():
            raise FileNotFoundError(f"工件1 CAD 朝向配置不存在: {orientation}")
        perception["orientation_config"] = orientation
    else:
        for key in ("capture_snapshots", "depth_fuse_frames", "planning_inflight"):
            value = perception.get(key) if key != "planning_inflight" else execution.get(key)
            if not isinstance(value, int) or value < 1:
                raise RuntimeError(f"工件2配置 {key} 必须为正整数")
        execution["pregrasp_m"] = _positive_number(execution, "pregrasp_m", "execution")
        down_offset = float(execution.get("grasp_down_offset_m", float("nan")))
        if not math.isfinite(down_offset) or not -0.030 <= down_offset <= 0.030:
            raise RuntimeError("工件2配置 grasp_down_offset_m 必须在 [-0.030, 0.030] m")
        execution["grasp_down_offset_m"] = down_offset
        execution["close_force"] = _positive_number(execution, "close_force", "execution")
    return ReverseGraspProfile(
        workpiece=workpiece,
        adapter=adapter,
        path=path,
        perception=perception,
        execution=execution,
    )


def load_reverse_ready_profile(workpiece: str) -> dict[str, Any]:
    """只读取 Module 5-1 所需的反抓预备位，不触发交接模块配置校验。"""
    if workpiece not in {"1", "2"}:
        raise ValueError(f"workpiece 必须是 '1' 或 '2'，当前为 {workpiece!r}")
    payload = json.loads(REVERSE_READY_CONFIG.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError(f"不支持的反抓预备位配置版本: {REVERSE_READY_CONFIG}")
    section = dict((payload.get("workpieces") or {}).get(workpiece) or {})
    joints = section.get("ready_joints_deg")
    if not isinstance(joints, list) or len(joints) != 7:
        raise RuntimeError(f"工件 {workpiece} 反抓预备位必须包含 7 个关节角")
    section["ready_joints_deg"] = [float(value) for value in joints]
    if any(not math.isfinite(value) for value in section["ready_joints_deg"]):
        raise RuntimeError(f"工件 {workpiece} 反抓预备位包含非有限数值")
    for key in ("arrival_tolerance_deg", "arrival_timeout_s", "post_close_lift_m"):
        section[key] = _positive_number(section, key, f"workpiece_{workpiece}")
    attempts = section.get("max_grasp_attempts")
    if not isinstance(attempts, int) or attempts < 1:
        raise RuntimeError(f"工件 {workpiece} max_grasp_attempts 必须为正整数")
    return section
