"""Retry only transient planner task-creation failures.

The trajectory service can occasionally accept a TCP request but reject it
before IK or path search starts with ``Failed to create planning task``.  That
is a service-side lifecycle failure, not a geometric planning result.  Keep
the retry rule narrow so collision and IK failures remain visible immediately.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


ResultT = TypeVar("ResultT")
TASK_CREATION_ATTEMPTS = 5
TASK_CREATION_RETRY_DELAYS_S = (1.0, 2.0, 3.0, 5.0)
TASK_CREATION_FAILURE_TEXT = "Failed to create planning task"


def is_task_creation_failure(exc: Exception) -> bool:
    """Return true only for the planner's pre-IK task-allocation failure."""
    return TASK_CREATION_FAILURE_TEXT in str(exc)


def retry_task_creation(
    request: Callable[[], ResultT],
    *,
    label: str,
) -> ResultT:
    """Run one planning request with bounded backoff for task creation only."""
    for attempt in range(1, TASK_CREATION_ATTEMPTS + 1):
        try:
            return request()
        except Exception as exc:
            if not is_task_creation_failure(exc) or attempt == TASK_CREATION_ATTEMPTS:
                raise
            delay_s = TASK_CREATION_RETRY_DELAYS_S[attempt - 1]
            print(
                f"[planner] {label} 创建规划任务失败；{delay_s:.1f}s 后重试 "
                f"{attempt + 1}/{TASK_CREATION_ATTEMPTS}",
                flush=True,
            )
            time.sleep(delay_s)
    raise AssertionError("unreachable")
