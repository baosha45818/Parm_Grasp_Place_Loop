"""左臂 Planner 与右臂 ServoPose 的兼容上下文。

当前双臂流程允许两类运动并行执行，因此这里不再施加跨进程文件锁。
保留原有上下文管理器接口，避免各业务模块因导入路径变化而失效。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def left_arm_motion_guard(label: str) -> Iterator[None]:
    """兼容旧调用：左臂 Planner 轨迹不再阻塞右臂 ServoPose。"""
    del label
    yield


@contextmanager
def right_servo_guard() -> Iterator[None]:
    """兼容旧调用：右臂 ServoPose 不再阻塞左臂 Planner 轨迹。"""
    yield
