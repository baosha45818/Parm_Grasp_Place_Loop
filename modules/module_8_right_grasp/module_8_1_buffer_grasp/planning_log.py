"""Process-safe structured logging shared by the workpiece pipeline children."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any


PLAN_LOG_ENV = "WORKPIECE_PIPELINE_EVENT_LOG"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def planner_start_pose(planner: object, frame_id: str, ik_frame: str) -> dict[str, Any] | None:
    """Read the planner-side start transform without making logging fatal."""
    try:
        result = planner.lookup_transform(frame_id, ik_frame)
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict):
            return None
        translation, rotation = data.get("translation"), data.get("rotation")
        if not isinstance(translation, dict) or not isinstance(rotation, dict):
            return None
        return {
            "frame_id": frame_id,
            "ik_frame": ik_frame,
            "position_m": [float(translation[key]) for key in ("x", "y", "z")],
            "orientation_xyzw": [float(rotation[key]) for key in ("x", "y", "z", "w")],
        }
    except Exception:
        return None


def append_event(event: dict[str, Any]) -> None:
    """Append one JSON object atomically across child processes and threads."""
    destination = os.environ.get(PLAN_LOG_ENV, "").strip()
    if not destination:
        return
    payload = {"timestamp_utc": utc_now(), "pid": os.getpid(), **event}
    path = Path(destination)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Diagnostics must never alter robot-control behavior.
        return


def planning_event(
    *, stage: str, method: str, start: dict[str, Any] | None,
    target: dict[str, Any], started_at: float, success: bool,
    failure_reason: str | None = None, **details: Any,
) -> None:
    append_event({
        "event": "planner_attempt",
        "stage": stage,
        "planning_method": method,
        "start": start,
        "target": target,
        "success": bool(success),
        "failure_reason": failure_reason,
        "duration_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        **details,
    })
