"""Post-lift force-sensor quality gate for the traditional ICP grasp.

The physical classifier is shared with the previously validated force-window
implementation.  This adapter owns Module 1-3's policy: only a stable SINGLE
classification may be handed to Module 2.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterator


LOOP_ROOT = Path(__file__).resolve().parents[3]


class WeightQualityRejected(RuntimeError):
    """The grasp is empty, multiple, unknown, or force data is unusable."""

    def __init__(self, grasp_class: str, message: str):
        super().__init__(message)
        self.grasp_class = str(grasp_class)


@dataclass(frozen=True)
class WeightCheckConfig:
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
    if str(LOOP_ROOT) not in sys.path:
        sys.path.insert(0, str(LOOP_ROOT))
    from tests.grasp_weight_check import (  # type: ignore[import-not-found]
        GraspClass,
        WeightThresholds,
        collect_force_window,
        verify_grasp_after_lift,
    )

    return GraspClass, WeightThresholds, collect_force_window, verify_grasp_after_lift


def _thresholds(config: WeightCheckConfig) -> Any:
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
    """Open the left force sensor and ensure its stream is live."""
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
    """Collect the open-gripper reference at the pregrasp wrist pose."""
    _, _, collect_force_window, _ = _api()
    return collect_force_window(
        sensor, sample_count=config.samples, timeout_s=config.timeout_s
    )


def verify_single_after_lift(sensor: Any, baseline: Any, config: WeightCheckConfig) -> dict[str, object]:
    """Return a Module-2 handoff payload or reject the physical grasp."""
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
    """Match Version7's post-graspup force-stability gate and persist it."""
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
    # Retain the Version7-compatible name for existing readers.
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
