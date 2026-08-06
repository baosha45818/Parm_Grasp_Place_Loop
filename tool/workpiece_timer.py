#!/usr/bin/env python3
"""Persist per-workpiece module timings across the dual-arm processes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
from time import time
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
TIME_ROOT = ROOT / "output" / "time"
WORKPIECE_ENV = "PARM_WORKPIECE_ID"
RUN_ENV = "PARM_LOOP_RUN_ID"


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def _run_id() -> str:
    return _safe_name(os.environ.get(RUN_ENV, "manual"))


def _path(workpiece_id: str) -> Path:
    return TIME_ROOT / _run_id() / f"{_safe_name(workpiece_id)}.json"


def _empty(workpiece_id: str, now: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": _run_id(),
        "workpiece_id": workpiece_id,
        "started_at": _utc(now),
        "finished_at": None,
        "total_duration_seconds": None,
        "outcome": "IN_PROGRESS",
        "modules": {},
        "active": {},
    }


def _read(path: Path, workpiece_id: str, now: float) -> dict[str, Any]:
    if not path.is_file():
        return _empty(workpiece_id, now)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty(workpiece_id, now)
    return value if isinstance(value, dict) else _empty(workpiece_id, now)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _locked_update(workpiece_id: str, update) -> dict[str, Any]:
    now = time()
    path = _path(workpiece_id)
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        payload = _read(path, workpiece_id, now)
        update(payload, now)
        _write(path, payload)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return payload


def begin(workpiece_id: str, *, metadata: dict[str, Any] | None = None) -> Path:
    """Create the immutable logical workpiece record if it does not exist."""
    def update(payload: dict[str, Any], _now: float) -> None:
        if metadata:
            payload.setdefault("metadata", {}).update(metadata)

    _locked_update(workpiece_id, update)
    return _path(workpiece_id)


def start(workpiece_id: str, module: str, *, scope: str, detail: str = "") -> None:
    def update(payload: dict[str, Any], now: float) -> None:
        modules = payload.setdefault("modules", {})
        entry = modules.setdefault(module, {"attempts": [], "total_duration_seconds": 0.0})
        key = f"{module}:{scope}"
        active = payload.setdefault("active", {})
        if key in active:
            return
        attempt = {
            "started_at": _utc(now),
            "finished_at": None,
            "duration_seconds": None,
            "status": "RUNNING",
            "detail": detail,
        }
        entry["attempts"].append(attempt)
        active[key] = {"module": module, "attempt_index": len(entry["attempts"]) - 1, "started_at": now}

    _locked_update(workpiece_id, update)


def finish(workpiece_id: str, module: str, *, scope: str, status: str, detail: str = "") -> None:
    def update(payload: dict[str, Any], now: float) -> None:
        active = payload.setdefault("active", {})
        key = f"{module}:{scope}"
        running = active.pop(key, None)
        if running is None:
            return
        entry = payload["modules"][module]
        attempt = entry["attempts"][int(running["attempt_index"])]
        duration = max(0.0, now - float(running["started_at"]))
        attempt.update({
            "finished_at": _utc(now),
            "duration_seconds": duration,
            "status": status,
        })
        if detail:
            attempt["detail"] = detail
        entry["total_duration_seconds"] = sum(
            float(item["duration_seconds"] or 0.0) for item in entry["attempts"]
        )

    _locked_update(workpiece_id, update)


def complete(workpiece_id: str, *, outcome: str, detail: str = "") -> None:
    def update(payload: dict[str, Any], now: float) -> None:
        # A process can be interrupted while one of its child modules is still
        # running. Close those attempts here so every record has a duration.
        active = payload.setdefault("active", {})
        interrupted_status = "ABORTED" if outcome == "ABORTED" else "INTERRUPTED"
        for key, running in list(active.items()):
            module = str(running["module"])
            entry = payload.get("modules", {}).get(module)
            if not isinstance(entry, dict):
                active.pop(key, None)
                continue
            attempts = entry.get("attempts", [])
            index = int(running["attempt_index"])
            if 0 <= index < len(attempts):
                attempts[index].update({
                    "finished_at": _utc(now),
                    "duration_seconds": max(0.0, now - float(running["started_at"])),
                    "status": interrupted_status,
                })
            active.pop(key, None)
        for entry in payload.get("modules", {}).values():
            if isinstance(entry, dict):
                entry["total_duration_seconds"] = sum(
                    float(item.get("duration_seconds") or 0.0)
                    for item in entry.get("attempts", [])
                )
        started = datetime.fromisoformat(payload["started_at"]).timestamp()
        payload["finished_at"] = _utc(now)
        payload["total_duration_seconds"] = max(0.0, now - started)
        payload["outcome"] = outcome
        if detail:
            payload["outcome_detail"] = detail

    _locked_update(workpiece_id, update)


def active_workpiece() -> str | None:
    value = os.environ.get(WORKPIECE_ENV, "").strip()
    return value or None


@contextmanager
def module_timer(module: str, *, workpiece_id: str | None = None, detail: str = "") -> Iterator[None]:
    """Time one module attempt; child processes inherit ``PARM_WORKPIECE_ID``."""
    item = workpiece_id or active_workpiece()
    if item is None:
        yield
        return
    scope = f"pid-{os.getpid()}-{threading.get_ident()}"
    start(item, module, scope=scope, detail=detail)
    try:
        yield
    except BaseException as exc:
        finish(item, module, scope=scope, status="FAILURE", detail=f"{type(exc).__name__}: {exc}")
        raise
    else:
        finish(item, module, scope=scope, status="SUCCESS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("begin", "start", "finish", "complete"))
    parser.add_argument("--workpiece-id", default=active_workpiece())
    parser.add_argument("--module")
    parser.add_argument("--scope", default=f"pid-{os.getpid()}")
    parser.add_argument("--status", default="SUCCESS")
    parser.add_argument("--detail", default="")
    args = parser.parse_args()
    if not args.workpiece_id:
        raise ValueError("需要 --workpiece-id 或 PARM_WORKPIECE_ID")
    if args.action == "begin":
        print(begin(args.workpiece_id))
    elif args.action == "start":
        if not args.module:
            raise ValueError("start 需要 --module")
        start(args.workpiece_id, args.module, scope=args.scope, detail=args.detail)
    elif args.action == "finish":
        if not args.module:
            raise ValueError("finish 需要 --module")
        finish(args.workpiece_id, args.module, scope=args.scope, status=args.status, detail=args.detail)
    else:
        complete(args.workpiece_id, outcome=args.status, detail=args.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
