"""Load and validate the production workpiece-2 high-handoff profile."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = Path(__file__).with_name("workpiece_2_handoff_profile.json")


def _finite_vector(
    payload: dict[str, Any], key: str, length: int
) -> list[float]:
    values = payload.get(key)
    if not isinstance(values, list) or len(values) != length:
        raise RuntimeError(f"工件2交接profile的 {key} 必须包含{length}个数值")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"工件2交接profile的 {key} 包含非有限数值")
    return result


def load_workpiece_2_handoff_profile(
    path: Path = PROFILE_PATH,
) -> dict[str, Any]:
    """Return a normalized profile shared by Module 5-2, 8-2 and the loop."""
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("workpiece") != "2":
        raise RuntimeError(f"无效的工件2交接profile：{resolved}")
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise RuntimeError("工件2交接profile缺少 profile_id")

    left_observation = dict(payload.get("left_observation") or {})
    left_observation["prepare_joints_deg"] = _finite_vector(
        left_observation, "prepare_joints_deg", 7
    )
    left_observation["observation_joints_deg"] = _finite_vector(
        left_observation, "observation_joints_deg", 7
    )
    head_pose = dict(left_observation.get("head_pose_deg") or {})
    for axis in ("pitch", "yaw"):
        value = float(head_pose.get(axis, float("nan")))
        if not math.isfinite(value):
            raise RuntimeError(f"工件2交接profile缺少有效云台{axis}")
        head_pose[axis] = value
    alignment = ROOT / str(left_observation.get("alignment", ""))
    if not alignment.is_file():
        raise FileNotFoundError(f"工件2观察外参不存在：{alignment}")
    left_observation["head_pose_deg"] = head_pose
    left_observation["alignment"] = alignment.resolve()

    handoff = dict(payload.get("handoff") or {})
    handoff["center_body_m"] = _finite_vector(handoff, "center_body_m", 3)
    for key in ("axis_tolerance_deg", "center_tolerance_mm"):
        value = float(handoff.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"工件2交接profile的 {key} 必须为正数")
        handoff[key] = value

    right_handoff = dict(payload.get("right_handoff") or {})
    for key in (
        "prepare_joints_deg",
        "pregrasp_joints_deg",
        "grasp_joints_deg",
        "place_ready_joints_deg",
    ):
        right_handoff[key] = _finite_vector(right_handoff, key, 7)
    for key in (
        "post_release_clearance_minus_x_m",
        "post_release_clearance_plus_z_m",
    ):
        value = float(right_handoff.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"工件2交接profile的 {key} 必须为正数")
        right_handoff[key] = value

    normalized = dict(payload)
    normalized["left_observation"] = left_observation
    normalized["handoff"] = handoff
    normalized["right_handoff"] = right_handoff
    normalized["left_return_joints_deg"] = _finite_vector(
        payload, "left_return_joints_deg", 7
    )
    normalized["source"] = resolved
    return normalized


def workpiece_2_right_prepare_joints_deg() -> list[float]:
    return list(
        load_workpiece_2_handoff_profile()["right_handoff"][
            "prepare_joints_deg"
        ]
    )
