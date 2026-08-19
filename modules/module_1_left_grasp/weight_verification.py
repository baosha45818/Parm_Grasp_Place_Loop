"""Module1 抓取抬升后的力传感器质量门控。

底层物理分类器复用已验证的力窗口实现；本适配层负责 Module1 策略：只有
稳定分类为 ``SINGLE`` 的抓取才能生成供下游模块使用的 handoff。
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterator


LOOP_ROOT = Path(__file__).resolve().parents[2]


class WeightQualityRejected(RuntimeError):
    """表示空抓、多抓、分类未知或力数据不可用。"""

    def __init__(self, grasp_class: str, message: str):
        """保存分类字符串，供上层决定重试或终止。"""
        super().__init__(message)
        self.grasp_class = str(grasp_class)


@dataclass(frozen=True)
class WeightCheckConfig:
    """力传感器连接、采样及单件重量阈值配置。"""
    host: str = "192.168.3.22"
    hand: int = 1
    samples: int = 31
    timeout_s: float = 3.0
    settle_s: float = 0.3
    empty_max_g: float = 50.0
    single_min_g: float = 50.000000001
    single_max_g: float = 129.999999999
    multiple_min_g: float = 130.0
    max_axis_mad_n: float = 0.15


def _api() -> tuple[Any, Any, Any, Any]:
    """延迟加载公共重量分类 API，避免纯规划路径连接硬件。"""
    if str(LOOP_ROOT) not in sys.path:
        sys.path.insert(0, str(LOOP_ROOT))
    from tools.grasp_weight_check import (
        GraspClass,
        WeightThresholds,
        collect_force_window,
        verify_grasp_after_lift,
    )

    return GraspClass, WeightThresholds, collect_force_window, verify_grasp_after_lift


def _thresholds(config: WeightCheckConfig) -> Any:
    """将 Module1 配置转换为公共分类器阈值对象。"""
    _, thresholds_type, _, _ = _api()
    return thresholds_type(
        empty_max_g=config.empty_max_g,
        single_min_g=config.single_min_g,
        single_max_g=config.single_max_g,
        multiple_min_g=config.multiple_min_g,
        max_axis_mad_n=config.max_axis_mad_n,
    )


@contextmanager
def force_sensor(config: WeightCheckConfig) -> Iterator[Any]:
    """连接左侧力传感器并确认数据流可用。"""
    if config.samples < 3 or config.timeout_s <= 0 or config.settle_s < 0:
        raise ValueError("invalid weight-check sampling settings")
    from parm_sdk import ForceSensor

    with ExitStack() as stack:
        try:
            sensor = stack.enter_context(ForceSensor(host=config.host, hand=config.hand))
        except Exception as exc:
            raise WeightQualityRejected(
                "UNKNOWN", f"ICP force sensor connection failed: {exc}"
            ) from exc
        try:
            sensor.wait_first(timeout=config.timeout_s)
        except Exception as exc:
            raise WeightQualityRejected(
                "UNKNOWN", f"ICP force sensor stream failed: {exc}"
            ) from exc
        yield sensor


def collect_empty_baseline(sensor: Any, config: WeightCheckConfig) -> Any:
    """在预抓取腕部位姿采集夹爪空载基线。"""
    _, _, collect_force_window, _ = _api()
    return collect_force_window(
        sensor, sample_count=config.samples, timeout_s=config.timeout_s
    )


def verify_single_after_lift(sensor: Any, baseline: Any, config: WeightCheckConfig) -> dict[str, object]:
    """验证抬升后为单件；通过则返回 handoff 载荷，否则拒绝抓取。"""
    grasp_class, _, _, verify_grasp_after_lift = _api()
    thresholds = _thresholds(config)
    result = verify_grasp_after_lift(
        sensor,
        baseline,
        thresholds,
        settle_s=config.settle_s,
        sample_count=config.samples,
        timeout_s=config.timeout_s,
    )
    print(
        "-> ICP grasp weight result: "
        f"class={result.grasp_class.value}, mass={result.mass_g:.1f} g, "
        f"reason={result.reason}",
        flush=True,
    )
    if result.grasp_class is not grasp_class.SINGLE:
        raise WeightQualityRejected(
            result.grasp_class.value,
            f"ICP grasp weight gate rejected {result.grasp_class.value}: {result.reason}",
        )
    return {
        "weight_check_enabled": True,
        "weight_check": {
            "classification": result.grasp_class.value,
            "mass_g": float(result.mass_g),
            "reason": result.reason,
            "baseline": {
                "median_force_n": list(result.baseline.median_force_n),
                "axis_mad_n": list(result.baseline.axis_mad_n),
                "sample_count": result.baseline.sample_count,
            },
            "loaded_at_pregrasp": {
                "median_force_n": list(result.loaded.median_force_n),
                "axis_mad_n": list(result.loaded.axis_mad_n),
                "sample_count": result.loaded.sample_count,
            },
            "thresholds": {
                "empty_max_g": thresholds.empty_max_g,
                "single_min_g": thresholds.single_min_g,
                "single_max_g": thresholds.single_max_g,
                "multiple_min_g": thresholds.multiple_min_g,
                "max_axis_mad_n": thresholds.max_axis_mad_n,
                "gravity_m_s2": thresholds.gravity_m_s2,
            },
            "sample_count": config.samples,
            "timeout_s": config.timeout_s,
            "settle_s": config.settle_s,
        },
        "force_sensor_host": config.host,
        "force_sensor_hand": config.hand,
    }


def verify_graspup_force_signature(
    sensor: Any, handoff: dict[str, object], config: WeightCheckConfig,
) -> None:
    """执行与 Version7 一致的 graspup 后稳定性门控并写入 handoff。"""
    _, _, collect_force_window, _ = _api()
    thresholds = _thresholds(config)
    try:
        import time

        time.sleep(config.settle_s)
        loaded = collect_force_window(
            sensor, sample_count=config.samples, timeout_s=config.timeout_s
        )
    except Exception as exc:
        raise WeightQualityRejected(
            "UNKNOWN", f"ICP graspup force signature unavailable: {exc}"
        ) from exc
    if loaded.max_axis_mad_n > thresholds.max_axis_mad_n:
        raise WeightQualityRejected(
            "UNKNOWN",
            "ICP graspup force signature is unstable: "
            f"MAD={loaded.max_axis_mad_n:.3f}N > {thresholds.max_axis_mad_n:.3f}N",
        )
    weight_check = handoff.get("weight_check")
    if not isinstance(weight_check, dict):
        raise RuntimeError("ICP handoff is missing weight_check")
    signature = {
        "median_force_n": list(loaded.median_force_n),
        "axis_mad_n": list(loaded.axis_mad_n),
        "sample_count": loaded.sample_count,
        "max_axis_mad_n": thresholds.max_axis_mad_n,
    }
    weight_check["graspup_load_signature"] = signature
    # 保留 Version7 兼容字段名，避免已有下游读取器失效。
    weight_check["loaded_at_graspup"] = {
        **signature,
        "mass_g": weight_check["mass_g"],
        "classification": weight_check["classification"],
    }
    print(
        "-> ICP graspup held-object force signature: "
        f"median={[round(float(value), 4) for value in loaded.median_force_n]} N",
        flush=True,
    )
