#!/usr/bin/env python3
"""Run ``place_hole_a_wiggle.py`` from a reviewed JSON profile.

The profile only contains values that map to the target script's public CLI.
Real hardware is deliberately enabled only by this launcher's ``--live`` and
``--confirm-live`` flags; a JSON edit alone can never start robot motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT = Path(__file__).with_name("place_hole_a_wiggle.py")

# Scalar CLI options accepted in profile.parameters.  Keeping this explicit is
# intentional: a typo must fail before it can affect a robot run.
SCALAR_KEYS = {
    "initial_motion", "planner_config", "initial_joint_planner", "arm_host", "force_host", "search_axis", "search_direction", "retry_lift_mm",
    "max_retry_lifts", "max_consecutive_back_target_timeouts", "same_target_retries",
    "back_step_mm", "wiggle_half_amplitude_mm", "continuous_force_monitor_hz",
    "continuous_force_output_hz", "continuous_release_frames",
    "continuous_release_fx_rise_tolerance_n", "continuous_release_ty_rise_tolerance_nm",
    "fast_linear_speed_mm_s", "fast_linear_accel_mm_s2", "slow_after_back_mm",
    "slow_linear_speed_mm_s", "slow_linear_accel_mm_s2", "quick_endpoint_tolerance_mm",
    "quick_endpoint_wait_s", "max_back_mm", "baseline_samples", "sample_timeout",
    "max_force_delta_n", "max_torque_delta_nm", "entry_pre_fx_n", "entry_pre_ty_nm",
    "entry_fx_drop_n", "entry_ty_drop_nm", "entry_ty_abs_nm", "entry_relaxed_pre_fx_n",
    "entry_relaxed_pre_ty_nm", "entry_relaxed_fx_drop_n", "entry_relaxed_ty_drop_nm",
    "entry_release_pre_force_n", "entry_release_pre_torque_nm", "entry_release_force_n",
    "entry_release_torque_nm", "entry_release_force_drop_n", "entry_release_torque_drop_nm",
    "entry_release_confirm_samples", "entry_release_confirm_interval_s",
    "entry_resultant_pre_force_n", "entry_resultant_pre_torque_nm",
    "entry_resultant_force_n", "entry_resultant_torque_nm",
    "entry_resultant_force_drop_n", "entry_resultant_torque_drop_nm",
    "entry_resultant_release_frames",
    "release_events_required",
    "entry_fx_reversal_pre_n", "entry_fx_reversal_current_n",
    "entry_fx_reversal_drop_n", "entry_fx_reversal_confirm_frames", "entry_fx_reversal_peak_window_mm",
    "wiggle_entry_pre_fx_n", "wiggle_entry_pre_ty_nm", "wiggle_entry_fx_drop_n",
    "wiggle_entry_ty_drop_nm", "min_entry_back_mm", "output",
}
BOOLEAN_KEYS = {
    "reset_right_task_before_servo", "straight_line", "continuous_straight_search", "single_continuous_search",
}
SPECIAL_LIST_KEYS = {"initial_right_joints_deg"}


def read_profile(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 配置 {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("配置必须是对象，且 schema_version 必须为 1")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("parameters 必须是对象")
    unknown = set(parameters) - SCALAR_KEYS - BOOLEAN_KEYS - SPECIAL_LIST_KEYS
    if unknown:
        raise ValueError(f"parameters 包含不支持的字段: {', '.join(sorted(unknown))}")
    return parameters


def profile_command(parameters: dict[str, Any], config_path: Path, *, live: bool) -> list[str]:
    command = [sys.executable, str(SCRIPT)]
    for key in sorted(SCALAR_KEYS & set(parameters)):
        value = parameters[key]
        if isinstance(value, (dict, list, bool)) or value is None:
            raise ValueError(f"parameters.{key} 必须是字符串或数字")
        if key in {"planner_config", "output"}:
            value = str((config_path.parent / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(value)
        command.extend(["--" + key.replace("_", "-"), str(value)])

    for key in BOOLEAN_KEYS & set(parameters):
        value = parameters[key]
        if not isinstance(value, bool):
            raise ValueError(f"parameters.{key} 必须是 true 或 false")
        if key == "reset_right_task_before_servo":
            if value:
                command.append("--reset-right-task-before-servo")
        elif key == "straight_line":
            command.append("--straight-line" if value else "--with-wiggle")
        elif key == "continuous_straight_search":
            command.append("--continuous-straight-search" if value else "--no-continuous-straight-search")
        elif key == "single_continuous_search" and value:
            command.append("--single-continuous-search")

    if "initial_right_joints_deg" in parameters:
        joints = parameters["initial_right_joints_deg"]
        if not isinstance(joints, list) or len(joints) != 7 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in joints):
            raise ValueError("parameters.initial_right_joints_deg 必须是 7 个数值组成的数组")
        command.extend(["--initial-right-joints-deg", *(str(value) for value in joints)])
    if live:
        command.extend(["--live", "--confirm-live"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="按 JSON profile 启动孔 A wiggle 搜索")
    parser.add_argument("--config", type=Path, required=True, help="profile JSON 路径")
    parser.add_argument("--live", action="store_true", help="允许连接真实硬件")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--print-command", action="store_true", help="仅打印最终命令，不执行")
    args = parser.parse_args()
    if args.live != args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    try:
        parameters = read_profile(args.config.resolve())
        command = profile_command(parameters, args.config.resolve(), live=args.live)
    except ValueError as exc:
        parser.error(str(exc))
    print("执行配置：", args.config.resolve())
    print("最终命令：", subprocess.list2cmdline(command))
    if args.print_command:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
