"""Force/torque frame validation and small vector operations."""

from __future__ import annotations

import math


def vector(frame: dict, key: str) -> list[float]:
    values = frame.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError(f"无效力传感器帧 {key}: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"力传感器 {key} 存在非有限值")
    return result


def read_wrench(sensor: object, timeout: float) -> tuple[list[float], list[float]]:
    """Return the SDK subscriber's currently cached force frame."""
    frame = sensor.wait_first(timeout=timeout)  # type: ignore[attr-defined]
    return vector(frame, "filtered_force"), vector(frame, "filtered_torque")


def mean(rows: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(3)]


def norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("无法计算空采样集的中位数")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
