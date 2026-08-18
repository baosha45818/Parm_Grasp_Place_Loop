"""Module 3 四条固定主路径的本地轨迹缓存。

此文件只负责缓存文件的校验、读取与原子写入；不连接规划器、不控制机械臂。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


# 目录名沿用现场约定的拼写。这里只保存四条“固定工位主路径”，不保存
# 松爪下探、抓取或其它恢复动作，避免把临时动作错误复用于下一件工件。
TRAJECTORY_CACHE_ROOT = Path(__file__).resolve().parent / "trajactory"
TRAJECTORY_CACHE_SCHEMA_VERSION = 1
_ALLOWED_ROUTES = frozenset(
    {
        "graspuptop_to_buffer",
        "buffer_to_graspuptop",
        "graspuptop_to_middle",
        "middle_to_graspuptop",
    }
)


def trajectory_cache_path(workpiece: str, route: str) -> Path:
    """返回某工件、某固定主路径唯一对应的本地轨迹文件。"""
    if route not in _ALLOWED_ROUTES:
        raise ValueError(f"不允许缓存非 Module 3 主路径: {route}")
    return TRAJECTORY_CACHE_ROOT / f"workpiece_{workpiece}" / f"{route}.json"


def trajectory_cache_signature(payload: dict) -> str:
    """把当前工位/规划参数压缩成缓存兼容性标识。"""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cached_planner_response(cache_payload: dict) -> dict:
    """严格校验磁盘轨迹，并恢复为 ``PlannerRuntime`` 可执行的响应格式。"""
    trajectory = cache_payload.get("trajectory")
    sample_frequency = cache_payload.get("sample_frequency_hz")
    if (
        not isinstance(trajectory, list)
        or not trajectory
        or any(
            not isinstance(point, list)
            or len(point) != 7
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in point)
            for point in trajectory
        )
        or not isinstance(sample_frequency, (int, float))
        or not np.isfinite(sample_frequency)
        or sample_frequency <= 0.0
    ):
        raise ValueError("轨迹缓存格式无效")
    return {
        "data": {
            "arm_id": 1,
            "trajectory": [[float(value) for value in point] for point in trajectory],
            "sample_frequency": float(sample_frequency),
        }
    }


def load_trajectory_cache(
    *,
    workpiece: str,
    route: str,
    compatibility: dict,
) -> tuple[dict | None, str]:
    """读取兼容的成功轨迹；文件缺失、损坏或配置改变都视为缓存未命中。"""
    path = trajectory_cache_path(workpiece, route)
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("根节点不是对象")
        if payload.get("schema_version") != TRAJECTORY_CACHE_SCHEMA_VERSION:
            return None, "schema_changed"
        if payload.get("workpiece") != str(workpiece) or payload.get("route") != route:
            return None, "route_or_workpiece_mismatch"
        if payload.get("compatibility_signature") != trajectory_cache_signature(compatibility):
            return None, "configuration_changed"
        response = _cached_planner_response(payload)
        return {"path": path, "payload": payload, "response": response}, "hit"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[轨迹缓存] 忽略无效文件 {path}: {type(exc).__name__}: {exc}", flush=True)
        return None, "invalid"


def save_trajectory_cache(
    *,
    workpiece: str,
    route: str,
    compatibility: dict,
    planned_response: dict,
    selected_candidate: dict | None = None,
) -> Path:
    """在一次真实轨迹执行并到位后，原子写入可复用的主路径轨迹。"""
    response = _cached_planner_response(
        {
            "trajectory": ((planned_response.get("data") or {}).get("trajectory")),
            "sample_frequency_hz": (
                (planned_response.get("data") or {}).get("sample_frequency", 10.0)
            ),
        }
    )
    data = response["data"]
    path = trajectory_cache_path(workpiece, route)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRAJECTORY_CACHE_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workpiece": str(workpiece),
        "route": route,
        "compatibility": compatibility,
        "compatibility_signature": trajectory_cache_signature(compatibility),
        "trajectory": data["trajectory"],
        "sample_frequency_hz": data["sample_frequency"],
        # 出站轨迹需要恢复原选中的固定目标，回程则不需要此字段。
        "selected_candidate": selected_candidate,
        "saved_after": "planner_execution_succeeded_and_joint_target_verified",
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(f"[轨迹缓存] 已保存 {route}: {path}", flush=True)
    return path
