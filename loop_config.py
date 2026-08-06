"""读取 Loop 配置，并校验所有笛卡尔目标都使用 base 坐标系。"""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path

from loop_types import ObjectPose, PoseMap


@dataclass(frozen=True)
class LoopConfig:
    planner_host: str
    execute_url: str
    base_frame: str
    left_group: str
    right_group: str
    left_tool_frame: str
    right_tool_frame: str
    max_velocity: float
    max_acceleration: float
    idle_poll_seconds: float
    live: bool
    poses: PoseMap
    hole_a_insertion: "HoleAInsertionConfig | None" = None


@dataclass(frozen=True)
class HoleAInsertionConfig:
    """Four-waypoint hole-A path and its measured right-wrist FT signature."""

    step_size_m: float
    baseline_samples: int
    sample_timeout_seconds: float
    ty_contact_rise_nm: float
    fz_peak_max_n: float
    fz_recovery_n: float
    fz_final_abs_max_n: float
    max_force_delta_n: float
    max_torque_delta_nm: float
    release_after_success: bool
    preinsert_joints_deg: list[float] | None
    driver_waypoints: list[list[float]] | None


def _load_hole_a_insertion(raw: object) -> HoleAInsertionConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("hole_a_insertion must be an object")
    raw_joints = raw.get("preinsert_joints_deg")
    if raw_joints is not None:
        if not isinstance(raw_joints, list) or len(raw_joints) != 7:
            raise ValueError("hole_a_insertion.preinsert_joints_deg must contain seven values")
        preinsert_joints_deg = [float(value) for value in raw_joints]
        if not all(math.isfinite(value) for value in preinsert_joints_deg):
            raise ValueError("hole_a_insertion.preinsert_joints_deg must be finite")
    else:
        preinsert_joints_deg = None
    raw_driver_waypoints = raw.get("driver_waypoints")
    if raw_driver_waypoints is not None:
        if not isinstance(raw_driver_waypoints, list) or len(raw_driver_waypoints) != 4:
            raise ValueError("hole_a_insertion.driver_waypoints must contain four xyz+rpy poses")
        driver_waypoints = []
        for index, waypoint in enumerate(raw_driver_waypoints):
            if not isinstance(waypoint, list) or len(waypoint) != 6:
                raise ValueError(f"driver_waypoints[{index}] must contain xyz+rpy")
            parsed = [float(value) for value in waypoint]
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError(f"driver_waypoints[{index}] must be finite")
            driver_waypoints.append(parsed)
    else:
        driver_waypoints = None
    config = HoleAInsertionConfig(
        step_size_m=float(raw["step_size_m"]),
        baseline_samples=int(raw["baseline_samples"]),
        sample_timeout_seconds=float(raw["sample_timeout_seconds"]),
        ty_contact_rise_nm=float(raw["ty_contact_rise_nm"]),
        fz_peak_max_n=float(raw["fz_peak_max_n"]),
        fz_recovery_n=float(raw["fz_recovery_n"]),
        fz_final_abs_max_n=float(raw["fz_final_abs_max_n"]),
        max_force_delta_n=float(raw["max_force_delta_n"]),
        max_torque_delta_nm=float(raw["max_torque_delta_nm"]),
        release_after_success=bool(raw.get("release_after_success", True)),
        preinsert_joints_deg=preinsert_joints_deg,
        driver_waypoints=driver_waypoints,
    )
    if (
        not math.isfinite(config.step_size_m)
        or config.step_size_m <= 0.0
        or config.baseline_samples < 1
        or not math.isfinite(config.sample_timeout_seconds)
        or config.sample_timeout_seconds <= 0.0
    ):
        raise ValueError("hole_a_insertion step size, sample count and timeout must be positive")
    for name in (
        "ty_contact_rise_nm",
        "fz_recovery_n",
        "fz_final_abs_max_n",
        "max_force_delta_n",
        "max_torque_delta_nm",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"hole_a_insertion.{name} must be positive and finite")
    if not math.isfinite(config.fz_peak_max_n):
        raise ValueError("hole_a_insertion.fz_peak_max_n must be finite")
    return config


def load_config(path: Path, *, live: bool) -> LoopConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_frame = raw.get("base_frame", "body_base")
    if base_frame != "body_base":
        raise ValueError("Cartesian base_frame must be exactly 'body_base'")

    poses: PoseMap = {}
    for name, value in raw.get("poses", {}).items():
        if value is None:
            poses[name] = None
            continue
        if value.get("frame_id", base_frame) != "body_base":
            raise ValueError(f"pose {name!r} must use frame_id='body_base'")
        poses[name] = ObjectPose(
            frame_id="body_base",
            position=list(value["position"]),
            orientation=list(value["orientation"]),
        )

    return LoopConfig(
        planner_host=raw.get("planner_host", "127.0.0.1"),
        execute_url=raw.get(
            "execute_url", "http://127.0.0.1:5667/middle/waypoint/execute"
        ),
        base_frame=base_frame,
        left_group=raw.get("left_group", "left_arm"),
        right_group=raw.get("right_group", "right_arm"),
        left_tool_frame=raw.get("left_tool_frame", "left_hand_tool_end_link"),
        right_tool_frame=raw.get("right_tool_frame", "right_hand_tool_end_link"),
        max_velocity=float(raw.get("max_velocity", 0.1)),
        max_acceleration=float(raw.get("max_acceleration", 0.1)),
        idle_poll_seconds=float(raw.get("idle_poll_seconds", 0.2)),
        live=live,
        poses=poses,
        hole_a_insertion=_load_hole_a_insertion(raw.get("hole_a_insertion")),
    )
