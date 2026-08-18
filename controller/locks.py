"""控制器拥有的跨进程资源锁。模块本身不再承担调度互斥。"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import threading
import time
from typing import Iterator


# Buffer 与交接位都是持久化业务状态，不是运行期资源锁；它们的工位约束由
# RuntimeState + scheduler_rules 负责。这里只保留真实的独占硬件资源。
LOCK_NAMES = {"LEFT_ARM", "RIGHT_ARM", "HEAD_D435", "LEFT_D405"}


class LockCancelled(RuntimeError):
    """总控停止时，不应继续等待尚被其他模块占用的资源。"""


@contextmanager
def acquire(
    root: Path,
    names: tuple[str, ...],
    *,
    stop_event: threading.Event | None = None,
    poll_seconds: float = 0.1,
) -> Iterator[None]:
    """按固定顺序获取资源锁，并在停止请求到达时立即放弃等待。"""
    if unknown := set(names) - LOCK_NAMES:
        raise ValueError(f"未知资源锁: {sorted(unknown)}")
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds 必须为正")
    root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for name in sorted(set(names)):
            handle = (root / f"{name.lower()}.lock").open("a+")
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        raise LockCancelled(
                            f"停止请求已到达，不再等待资源锁 {name}"
                        )
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if stop_event is not None:
                            stop_event.wait(poll_seconds)
                        else:
                            time.sleep(poll_seconds)
            except BaseException:
                handle.close()
                raise
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
