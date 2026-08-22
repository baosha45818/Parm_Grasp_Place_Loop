"""Module 9 固定 ``place-ready -> 搜索前关节起点`` 路径的本地轨迹缓存。

缓存只允许在右臂刚完成 Module 8 交接、且实际关节角仍与缓存首点吻合时回放。
文件格式和 Module 3 的 ``trajactory`` 缓存保持同一安全原则：首次轨迹必须由
在线规划实际执行并确认到位后才能写入；不匹配时不能盲目播放。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time


TRAJECTORY_CACHE_ROOT = Path(__file__).resolve().parent / "trajactory"
SCHEMA_VERSION = 1
ROUTE = "place_ready_to_initial"


def cache_path(workpiece: str) -> Path:
    """Return the one fixed initial-route cache for a workpiece."""
    return TRAJECTORY_CACHE_ROOT / f"workpiece_{workpiece}" / f"{ROUTE}.json"


def compatibility_signature(compatibility: dict) -> str:
    canonical = json.dumps(
        compatibility, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def planner_config_digest(path: Path) -> str:
    """Bind a cache to the planner configuration that produced it."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"无法读取 Module 9 规划配置以校验轨迹缓存: {path}") from exc


def build_compatibility(*, target_joints_deg: list[float], planner: str, planner_config: Path) -> dict:
    return {
        "arm_id": 2,
        "route": ROUTE,
        "target_joints_deg": [float(value) for value in target_joints_deg],
        "planner": str(planner),
        "planner_config": str(planner_config.resolve()),
        "planner_config_sha256": planner_config_digest(planner_config),
    }


def _response_from_payload(payload: dict) -> dict:
    trajectory = payload.get("trajectory")
    sample_frequency = payload.get("sample_frequency_hz")
    if (
        not isinstance(trajectory, list)
        or not trajectory
        or any(
            not isinstance(point, list)
            or len(point) != 7
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in point
            )
            for point in trajectory
        )
        or not isinstance(sample_frequency, (int, float))
        or not math.isfinite(float(sample_frequency))
        or float(sample_frequency) <= 0.0
    ):
        raise ValueError("轨迹缓存格式无效")
    return {
        "data": {
            "arm_id": 2,
            "trajectory": [[float(value) for value in point] for point in trajectory],
            "sample_frequency": float(sample_frequency),
        }
    }


def load(*, workpiece: str, compatibility: dict) -> tuple[dict | None, str]:
    """Load a compatible right-arm trajectory, never accepting stale metadata."""
    path = cache_path(workpiece)
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("根节点不是对象")
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None, "schema_changed"
        if payload.get("workpiece") != str(workpiece) or payload.get("route") != ROUTE:
            return None, "route_or_workpiece_mismatch"
        if payload.get("compatibility_signature") != compatibility_signature(compatibility):
            return None, "configuration_changed"
        return {"path": path, "response": _response_from_payload(payload)}, "hit"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[轨迹缓存] 忽略无效文件 {path}: {type(exc).__name__}: {exc}", flush=True)
        return None, "invalid"


def save(*, workpiece: str, compatibility: dict, planned_response: dict) -> Path:
    """Atomically persist a trajectory only after execution/arrival verification."""
    response = _response_from_payload(
        {
            "trajectory": (planned_response.get("data") or {}).get("trajectory"),
            "sample_frequency_hz": (planned_response.get("data") or {}).get(
                "sample_frequency", 10.0
            ),
        }
    )
    data = response["data"]
    path = cache_path(workpiece)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workpiece": str(workpiece),
        "route": ROUTE,
        "compatibility": compatibility,
        "compatibility_signature": compatibility_signature(compatibility),
        "trajectory": data["trajectory"],
        "sample_frequency_hz": data["sample_frequency"],
        "saved_after": "planner_execution_succeeded_and_right_joint_target_verified",
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(f"[轨迹缓存] 已保存 {ROUTE}: {path}", flush=True)
    return path
