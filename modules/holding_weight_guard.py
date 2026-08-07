"""同一 graspup 姿态下的左手持件重量复检。"""

from __future__ import annotations

import math
import time
from typing import Any


def verify_graspup_signature(weight_check: dict[str, Any], label: str) -> dict[str, Any] | None:
    """与 Module 1 在 graspup 记录的持件力签名比较。

    该比较只允许在同一 graspup 关节位使用；跨姿态的重力方向变化不能用
    原始三轴力向量直接比较。没有新式签名的旧 handoff 返回 ``None``，由旧
    调用方保留其兼容逻辑。
    """
    signature = weight_check.get("graspup_load_signature")
    if not isinstance(signature, dict):
        return None
    try:
        from parm_sdk import ForceSensor
        from tests.grasp_weight_check import ForceWindow, collect_force_window

        timeout_s = float(weight_check["timeout_s"])
        settle_s = float(weight_check["settle_s"])
        with ForceSensor(
            host=str(weight_check["force_sensor_host"]),
            hand=int(weight_check["force_sensor_hand"]),
        ) as sensor:
            sensor.wait_first(timeout=timeout_s)
            time.sleep(settle_s)
            loaded = collect_force_window(
                sensor,
                sample_count=int(weight_check["sample_count"]),
                timeout_s=timeout_s,
            )
        reference = ForceWindow(
            median_force_n=tuple(float(value) for value in signature["median_force_n"]),
            axis_mad_n=tuple(float(value) for value in signature["axis_mad_n"]),
            sample_count=int(signature["sample_count"]),
        )
        force_delta_n = math.sqrt(sum(
            (current - expected) ** 2
            for current, expected in zip(
                loaded.median_force_n, reference.median_force_n,
            )
        ))
        max_force_delta_n = float(signature["max_force_delta_n"])
        max_axis_mad_n = float(signature["max_axis_mad_n"])
        stable = loaded.max_axis_mad_n <= max(reference.max_axis_mad_n, max_axis_mad_n)
        return {
            "checkpoint": label,
            "classification": "GRASPUP_SIGNATURE",
            "mass_g": None,
            "force_delta_n": force_delta_n,
            "max_force_delta_n": max_force_delta_n,
            "loaded_max_axis_mad_n": float(loaded.max_axis_mad_n),
            "passed": bool(stable and force_delta_n <= max_force_delta_n),
            "reason": "与 Module 1 graspup 持件力签名比较",
        }
    except Exception as exc:
        return {
            "checkpoint": label,
            "classification": "UNKNOWN",
            "mass_g": None,
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
