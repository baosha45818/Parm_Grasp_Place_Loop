"""控制器拥有的跨进程资源锁。模块本身不再承担调度互斥。"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator


LOCK_NAMES = {"BUFFER", "LEFT_ARM", "RIGHT_ARM", "HEAD_D435", "LEFT_D405", "HANDOFF_ZONE"}


@contextmanager
def acquire(root: Path, names: tuple[str, ...]) -> Iterator[None]:
    if unknown := set(names) - LOCK_NAMES:
        raise ValueError(f"未知资源锁: {sorted(unknown)}")
    root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for name in sorted(set(names)):
            handle = (root / f"{name.lower()}.lock").open("a+")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
