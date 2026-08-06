"""Append right-arm planner attempts in the loop's trajectory-record format."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import threading
from typing import Any


_LOCK = threading.Lock()
_LOOP_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if parent.name == "Parm_Grasp_Place_Loop"),
    Path(__file__).resolve().parents[2],
)
_DEFAULT_ROOT = _LOOP_ROOT / "output" / "right_trajectories"
_ENV_DIR = "PARM_RIGHT_TRAJECTORY_RECORD_DIR"
_ENV_MODULE = "PARM_RIGHT_TRAJECTORY_MODULE"


def start_run(module: str) -> Path:
    """Create one timestamped directory and pass it to child processes."""
    directory = _DEFAULT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ[_ENV_DIR] = str(directory)
    os.environ[_ENV_MODULE] = module
    return directory


def _record_dir() -> Path:
    configured = os.environ.get(_ENV_DIR)
    if configured:
        return Path(configured)
    directory = _DEFAULT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ[_ENV_DIR] = str(directory)
    return directory


def capture_tool_pose(planner: Any, frame_id: str, ik_frame: str) -> dict[str, Any] | None:
    """Read the planner's current IK-frame pose without making motion fail on diagnostics."""
    try:
        data = (planner.lookup_transform(frame_id, ik_frame).get("data") or {})
        translation = data.get("translation") or {}
        rotation = data.get("rotation") or {}
        position = [float(translation[axis]) for axis in "xyz"]
        orientation = [float(rotation[axis]) for axis in "xyzw"]
        return {
            "frame_id": frame_id,
            "ik_frame": ik_frame,
            "position_m": position,
            "orientation_xyzw": orientation,
        }
    except Exception:
        return None


def record_plan(
    *,
    module: str,
    stage: str,
    status: str,
    planner: str,
    request_type: str,
    target: dict[str, Any],
    response: dict[str, Any] | None = None,
    error: BaseException | str | None = None,
    start_joints_deg: list[float] | None = None,
    start_tool_pose: dict[str, Any] | None = None,
) -> None:
    """Write one successful or failed planner RPC without affecting robot motion."""
    try:
        data = (response or {}).get("data") or {}
        trajectory = data.get("trajectory")
        if not isinstance(trajectory, list):
            trajectory = []
        planned_start = trajectory[0] if trajectory else None
        planned_end = trajectory[-1] if trajectory else None
        target_kind = "joint_target" if "joint_values_deg" in target else "tool_pose_target"
        to = {"kind": target_kind, **target}
        end_tool_pose = (
            {
                "frame_id": target.get("frame_id"),
                "ik_frame": target.get("ik_frame"),
                "position_m": target.get("position_m"),
                "orientation_xyzw": target.get("orientation_xyzw"),
            }
            if target_kind == "tool_pose_target"
            else None
        )
        effective_module = os.environ.get(_ENV_MODULE, module)
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "status": status,
            "arm": "right",
            "module": effective_module,
            "stage": stage,
            "planner": planner,
            "request_type": request_type,
            "start_joints_deg": start_joints_deg or planned_start,
            "target": target,
            "from": {
                "source": "planned_trajectory_start" if planned_start else "unavailable_before_failed_plan",
                "joint_angles_deg": start_joints_deg or planned_start,
                "tool_pose": start_tool_pose,
            },
            "to": to,
            "start_eef_pose": start_tool_pose,
            "end_eef_pose": end_tool_pose,
            "logical_motion": {"from": stage, "to": target_kind},
            "planned_start_joints_deg": planned_start,
            "planned_end_joints_deg": planned_end,
            "trajectory_joint_deg": trajectory if status == "success" else None,
            "sample_frequency_hz": data.get("sample_frequency"),
            "planner_arm_id": data.get("arm_id"),
            "error": None if error is None else (
                str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
            ),
        }
        directory = _record_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"right_planning_attempts_pid_{os.getpid()}.jsonl"
        with _LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as record_error:
        print(f"[right-planner-record] 写入失败: {type(record_error).__name__}: {record_error}", flush=True)
