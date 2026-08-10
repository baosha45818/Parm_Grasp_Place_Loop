"""Small, dependency-free timing recorder for Module 5-1 executions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, time


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class ExecutionTiming:
    """Record wall-clock metadata and monotonic durations for one execution."""

    def __init__(self) -> None:
        self._started_wall = time()
        self._started_monotonic = perf_counter()
        self._active: dict[str, tuple[float, float]] = {}
        self.stages: list[dict] = []

    def start(self, name: str) -> None:
        if name in self._active:
            raise RuntimeError(f"计时阶段已开始: {name}")
        self._active[name] = (time(), perf_counter())

    def finish(self, name: str, *, status: str = "SUCCESS") -> None:
        started_wall, started_monotonic = self._active.pop(name)
        finished_wall = time()
        self.stages.append({
            "name": name,
            "started_at": _utc(started_wall),
            "finished_at": _utc(finished_wall),
            "duration_seconds": max(0.0, perf_counter() - started_monotonic),
            "status": status,
        })

    def complete(self, status: str) -> dict:
        for name in list(self._active):
            self.finish(name, status="INTERRUPTED")
        finished_wall = time()
        return {
            "started_at": _utc(self._started_wall),
            "finished_at": _utc(finished_wall),
            "total_duration_seconds": max(0.0, perf_counter() - self._started_monotonic),
            "status": status,
            "stages": self.stages,
        }

    def write(self, path: Path, status: str) -> None:
        path.write_text(
            json.dumps(self.complete(status), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
