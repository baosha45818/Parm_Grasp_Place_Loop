"""用抓取前后的力差估算工件重量并判断抓空、单件或多件。

该文件不依赖 ``parm_sdk``，便于单元测试。调用方只需传入一个实现
``get_state() -> dict | None`` 的对象（真实运行时即 ``ForceSensor``）。
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence


class ForceSensorLike(Protocol):
    def get_state(self) -> dict | None:
        """返回 ForceSensor 的最近一帧。"""


class GraspClass(str, Enum):
    EMPTY = "EMPTY"
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"
    UNKNOWN = "UNKNOWN"


class ForceDataError(RuntimeError):
    """在规定时间内未取得足够的有效力数据。"""


@dataclass(frozen=True)
class WeightThresholds:
    """重量分类阈值，单位为克；稳定性阈值单位为牛顿。"""

    empty_max_g: float
    single_min_g: float
    single_max_g: float
    multiple_min_g: float
    max_axis_mad_n: float = 0.15
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        values = (
            self.empty_max_g,
            self.single_min_g,
            self.single_max_g,
            self.multiple_min_g,
            self.max_axis_mad_n,
            self.gravity_m_s2,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("所有阈值必须是有限正数")
        if not (
            self.empty_max_g
            < self.single_min_g
            <= self.single_max_g
            < self.multiple_min_g
        ):
            raise ValueError(
                "阈值必须满足 empty_max_g < single_min_g <= "
                "single_max_g < multiple_min_g"
            )


@dataclass(frozen=True)
class ForceWindow:
    median_force_n: tuple[float, float, float]
    axis_mad_n: tuple[float, float, float]
    sample_count: int

    @property
    def max_axis_mad_n(self) -> float:
        return max(self.axis_mad_n)


@dataclass(frozen=True)
class GraspWeightResult:
    grasp_class: GraspClass
    mass_g: float
    delta_force_n: tuple[float, float, float]
    baseline: ForceWindow
    loaded: ForceWindow
    reason: str


def _valid_force(frame: object) -> tuple[float, float, float] | None:
    if not isinstance(frame, dict):
        return None
    force = frame.get("filtered_force")
    if not isinstance(force, Sequence) or isinstance(force, (str, bytes)):
        return None
    if len(force) != 3:
        return None
    try:
        parsed = tuple(float(value) for value in force)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in parsed):
        return None
    return parsed  # type: ignore[return-value]


def summarize_force(samples: Sequence[Sequence[float]]) -> ForceWindow:
    """以逐轴中位数和 MAD 汇总一个静止采样窗口。"""

    if not samples:
        raise ValueError("samples 不能为空")
    if any(len(sample) != 3 for sample in samples):
        raise ValueError("每个力样本必须恰好包含三个轴")

    axes = [[float(sample[axis]) for sample in samples] for axis in range(3)]
    medians = tuple(statistics.median(axis) for axis in axes)
    mads = tuple(
        statistics.median(abs(value - median) for value in axis)
        for axis, median in zip(axes, medians)
    )
    return ForceWindow(
        median_force_n=medians,  # type: ignore[arg-type]
        axis_mad_n=mads,  # type: ignore[arg-type]
        sample_count=len(samples),
    )


def collect_force_window(
    sensor: ForceSensorLike,
    *,
    sample_count: int = 31,
    timeout_s: float = 2.0,
    poll_interval_s: float = 0.005,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ForceWindow:
    """采集不同时间戳的有效帧，超时则抛出 ``ForceDataError``。"""

    if sample_count < 3:
        raise ValueError("sample_count 至少为 3")
    if timeout_s <= 0 or poll_interval_s < 0:
        raise ValueError("timeout_s 必须大于 0，poll_interval_s 不能小于 0")

    deadline = monotonic() + timeout_s
    samples: list[tuple[float, float, float]] = []
    last_timestamp: object = object()

    while len(samples) < sample_count and monotonic() < deadline:
        frame = sensor.get_state()
        force = _valid_force(frame)
        if force is not None:
            timestamp = frame.get("timestamp") if isinstance(frame, dict) else None
            # 有时间戳时避免把 SDK 返回的同一“最近帧”重复计入窗口。
            if timestamp is None or timestamp != last_timestamp:
                samples.append(force)
                last_timestamp = timestamp
        if len(samples) < sample_count:
            sleep(poll_interval_s)

    if len(samples) < sample_count:
        raise ForceDataError(
            f"力传感器采样超时：需要 {sample_count} 帧，仅收到 "
            f"{len(samples)} 帧有效的新数据"
        )
    return summarize_force(samples)


def classify_grasp_weight(
    baseline: ForceWindow,
    loaded: ForceWindow,
    thresholds: WeightThresholds,
) -> GraspWeightResult:
    """比较同姿态的空载/抓取窗口，并返回保守的重量分类。"""

    delta = tuple(
        loaded_value - baseline_value
        for baseline_value, loaded_value in zip(
            baseline.median_force_n, loaded.median_force_n
        )
    )
    mass_g = math.sqrt(sum(value * value for value in delta)) * 1000.0
    mass_g /= thresholds.gravity_m_s2

    max_mad = max(baseline.max_axis_mad_n, loaded.max_axis_mad_n)
    if max_mad > thresholds.max_axis_mad_n:
        grasp_class = GraspClass.UNKNOWN
        reason = (
            f"采样不稳定：最大轴 MAD={max_mad:.3f} N，"
            f"允许值={thresholds.max_axis_mad_n:.3f} N"
        )
    elif mass_g <= thresholds.empty_max_g:
        grasp_class = GraspClass.EMPTY
        reason = f"估算重量 {mass_g:.1f} g 不高于抓空上限"
    elif thresholds.single_min_g <= mass_g <= thresholds.single_max_g:
        grasp_class = GraspClass.SINGLE
        reason = f"估算重量 {mass_g:.1f} g 位于单件区间"
    elif mass_g >= thresholds.multiple_min_g:
        grasp_class = GraspClass.MULTIPLE
        reason = f"估算重量 {mass_g:.1f} g 不低于多件下限"
    else:
        grasp_class = GraspClass.UNKNOWN
        reason = f"估算重量 {mass_g:.1f} g 位于阈值迟滞区"

    return GraspWeightResult(
        grasp_class=grasp_class,
        mass_g=mass_g,
        delta_force_n=delta,  # type: ignore[arg-type]
        baseline=baseline,
        loaded=loaded,
        reason=reason,
    )


def verify_grasp_after_lift(
    sensor: ForceSensorLike,
    baseline: ForceWindow,
    thresholds: WeightThresholds,
    *,
    settle_s: float = 0.3,
    sample_count: int = 31,
    timeout_s: float = 2.0,
    poll_interval_s: float = 0.005,
    sleep: Callable[[float], None] = time.sleep,
) -> GraspWeightResult:
    """抓取上移离开桌面并停止后，采集载荷窗口并完成分类。

    调用本函数前，规划器必须已经同步完成抬升动作。``baseline`` 应在同一
    抓取周期、同一抬升检查姿态下用空夹爪采集。函数不会控制机械臂或夹爪。
    """

    if settle_s < 0:
        raise ValueError("settle_s 不能小于 0")
    sleep(settle_s)
    loaded = collect_force_window(
        sensor,
        sample_count=sample_count,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        sleep=sleep,
    )
    return classify_grasp_weight(baseline, loaded, thresholds)
