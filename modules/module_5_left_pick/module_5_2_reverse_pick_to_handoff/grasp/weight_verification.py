"""反面抓取分支工件2使用的力传感器二值持件确认。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
WEIGHT_CONFIG = ROOT / "config" / "grasp_weight_thresholds.conf"
POST_CLOSE_CHECK_LIFT_M = 0.03


def _config_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in WEIGHT_CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise RuntimeError(f"重量配置格式错误: {raw_line!r}")
        values[key.strip()] = value.strip()
    return values


def _thresholds() -> tuple[Any, int, float, float]:
    """读取共享采样参数；反抓仅使用其中的抓空上限。"""
    from tools.grasp_weight_check import WeightThresholds

    if not WEIGHT_CONFIG.is_file():
        raise FileNotFoundError(f"共享重量配置不存在: {WEIGHT_CONFIG}")
    values = _config_values()
    try:
        return (
            WeightThresholds(
                empty_max_g=float(values["WEIGHT_EMPTY_MAX_G"]),
                single_min_g=float(values["WEIGHT_SINGLE_MIN_G"]),
                single_max_g=float(values["WEIGHT_SINGLE_MAX_G"]),
                multiple_min_g=float(values["WEIGHT_MULTIPLE_MIN_G"]),
                max_axis_mad_n=float(values["WEIGHT_MAX_AXIS_MAD_N"]),
            ),
            int(values["WEIGHT_CHECK_SAMPLES"]),
            float(values["WEIGHT_CHECK_TIMEOUT_S"]),
            float(values["WEIGHT_CHECK_SETTLE_S"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"共享重量配置无效: {WEIGHT_CONFIG}") from exc


def _collect_window(force_sensor_host: str):
    from parm_sdk import ForceSensor
    from tools.grasp_weight_check import collect_force_window

    _, sample_count, timeout_s, settle_s = _thresholds()
    with ForceSensor(host=force_sensor_host, hand=1) as sensor:
        sensor.wait_first(timeout=timeout_s)
        time.sleep(settle_s)
        return collect_force_window(sensor, sample_count=sample_count, timeout_s=timeout_s)


def capture_open_gripper_baseline(force_sensor_host: str = "192.168.3.22"):
    """在抓取位、闭夹爪前采集空载三轴力基线。"""
    baseline = _collect_window(force_sensor_host)
    print(f"[反抓力] 抓取位空载基线: median={list(baseline.median_force_n)} N", flush=True)
    return baseline


def verify_object_after_lift(
    baseline: Any,
    force_sensor_host: str = "192.168.3.22",
    *,
    minimum_held_mass_g: float | None = None,
) -> dict[str, Any]:
    """上提 3 cm 且到位后判断是否有工件，不区分单件或多件。"""
    from tools.grasp_weight_check import classify_grasp_weight

    thresholds, _, _, _ = _thresholds()
    minimum_mass_g = (
        thresholds.empty_max_g
        if minimum_held_mass_g is None
        else float(minimum_held_mass_g)
    )
    if minimum_mass_g <= 0.0:
        raise ValueError("minimum_held_mass_g 必须为正数")
    try:
        loaded = _collect_window(force_sensor_host)
        result = classify_grasp_weight(baseline, loaded, thresholds)
        # 反抓只关心“抓空”与“有工件”：不再使用 SINGLE/MULTIPLE 的边界。
        stable = max(result.baseline.max_axis_mad_n, result.loaded.max_axis_mad_n) <= thresholds.max_axis_mad_n
        # ``minimum_mass_g`` 可为某条抓取流程设置比全局“抓空”阈值更
        # 贴合工件的二值门限。例如工件 1 的反抓要求重量至少为 40 g。
        accepted = stable and result.mass_g >= minimum_mass_g
        if stable and not accepted:
            reason = (
                f"估算重量 {result.mass_g:.1f} g 小于持件下限 "
                f"{minimum_mass_g:.1f} g"
            )
        else:
            reason = result.reason
        payload = {
            "classification": "HELD" if accepted else ("EMPTY" if stable else "UNKNOWN"),
            "accepted": accepted,
            "mass_g": float(result.mass_g),
            "raw_weight_class": result.grasp_class.value,
            "reason": reason,
            "empty_max_g": thresholds.empty_max_g,
            "minimum_held_mass_g": minimum_mass_g,
            "check_lift_m": POST_CLOSE_CHECK_LIFT_M,
        }
    except Exception as exc:
        payload = {
            "classification": "UNKNOWN",
            "accepted": False,
            "mass_g": None,
            "reason": f"{type(exc).__name__}: {exc}",
            "check_lift_m": POST_CLOSE_CHECK_LIFT_M,
        }
    print(
        "[反抓力] 上提3cm后持件判定: "
        f"class={payload['classification']} mass={payload['mass_g']}g "
        f"accepted={payload['accepted']} reason={payload['reason']}",
        flush=True,
    )
    return payload
