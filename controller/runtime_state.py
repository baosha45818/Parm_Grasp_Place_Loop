"""Buffer、左右臂及重试计数的持久化状态；只有 controller 可写。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class RuntimeState:
    run_id: str
    workpiece: str
    buffer: str = "EMPTY"
    middle: str = "EMPTY"
    # 反面工件离开 Buffer 后的独立工位状态；禁止再复用 buffer=HANDOFF_READY。
    handoff: str = "EMPTY"
    left_arm: str = "IDLE"
    right_arm: str = "IDLE"
    orientation: str | None = None
    buffer_reset_reason: str | None = None
    last_module: str | None = None
    retries: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RuntimeState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))
