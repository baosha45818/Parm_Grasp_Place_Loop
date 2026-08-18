"""新总控唯一使用的模块结果协议。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import json


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    WAIT = "WAIT"
    OPERATOR_REQUIRED = "OPERATOR_REQUIRED"
    FAULT = "FAULT"
    CANCELLED = "CANCELLED"


@dataclass
class ModuleResult:
    module: str
    status: Status
    outcome: str
    run_id: str
    outputs: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    source_result: str | None = None
    schema_version: int = 1

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
