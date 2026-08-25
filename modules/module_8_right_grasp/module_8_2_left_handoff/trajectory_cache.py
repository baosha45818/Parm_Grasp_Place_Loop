"""Module 8.2 right-arm Idle-to-pregrasp trajectory cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time


CACHE_ROOT = Path(__file__).resolve().parent / "trajactory"
SCHEMA_VERSION = 1
ROUTE = "idle_to_pregrasp"


def path(workpiece: str) -> Path:
    return CACHE_ROOT / f"workpiece_{workpiece}" / f"{ROUTE}.json"


def _signature(value: dict) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compatibility(*, mode: str, profile_id: str, grasp_joints_deg: list[float]) -> dict:
    return {
        "arm_id": 2, "route": ROUTE, "mode": mode, "profile_id": profile_id,
        "grasp_joints_deg": [float(value) for value in grasp_joints_deg],
        "pregrasp_offset": {"axis": "object-minus-x", "distance_m": 0.05},
        "planner": "rrtstar",
    }


def _response(payload: dict) -> dict:
    trajectory, frequency = payload.get("trajectory"), payload.get("sample_frequency_hz")
    if (
        not isinstance(trajectory, list) or not trajectory
        or any(not isinstance(point, list) or len(point) != 7 or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in point) for point in trajectory)
        or not isinstance(frequency, (int, float)) or not math.isfinite(float(frequency)) or float(frequency) <= 0.0
    ):
        raise ValueError("轨迹缓存格式无效")
    return {"data": {"arm_id": 2, "trajectory": [[float(v) for v in p] for p in trajectory], "sample_frequency": float(frequency)}}


def load(*, workpiece: str, compatibility_data: dict) -> tuple[dict | None, str]:
    destination = path(workpiece)
    if not destination.is_file():
        return None, "missing"
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("根节点不是对象")
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None, "schema_changed"
        if payload.get("workpiece") != str(workpiece) or payload.get("route") != ROUTE:
            return None, "route_or_workpiece_mismatch"
        if payload.get("compatibility_signature") != _signature(compatibility_data):
            return None, "configuration_changed"
        return {"path": destination, "response": _response(payload)}, "hit"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[轨迹缓存] 忽略无效文件 {destination}: {type(exc).__name__}: {exc}", flush=True)
        return None, "invalid"


def save(*, workpiece: str, compatibility_data: dict, response: dict) -> Path:
    normalized = _response({
        "trajectory": (response.get("data") or {}).get("trajectory"),
        "sample_frequency_hz": (response.get("data") or {}).get("sample_frequency", 10.0),
    })
    data, destination = normalized["data"], path(workpiece)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION, "workpiece": str(workpiece), "route": ROUTE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "compatibility": compatibility_data, "compatibility_signature": _signature(compatibility_data),
        "trajectory": data["trajectory"], "sample_frequency_hz": data["sample_frequency"],
        "saved_after": "rrtstar_execution_succeeded_and_right_joint_target_verified",
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(f"[轨迹缓存] 已保存 {ROUTE}: {destination}", flush=True)
    return destination
