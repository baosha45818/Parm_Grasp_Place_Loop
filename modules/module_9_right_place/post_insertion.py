"""Post-insertion motions for the right-arm Hole-A placement flow."""

from __future__ import annotations

from functools import partial
import time

from move_by_joints import current_pose, rotvec_to_rpy
from motion_primitives import move_continuous_linear_step
from placement_session import PlacementSession
from right_arm_state import wait_for_right_task_stopped
from servo_worker import move_continuous_with_recovery
from wrench_utils import read_wrench


class PostEntryFxRiseReached(RuntimeError):
    """The post-insertion descent reached its raw sensor-Fx stop threshold."""


def monitor_descend_raw_fx(
    alpha: float,
    *,
    session: PlacementSession,
    descend_fx_base: float,
) -> None:
    """Stop the active post-entry descent when raw Fx rises past its limit."""
    assert session.arm is not None and session.sensor is not None and session.log_event is not None
    args, arm, sensor, log_event = session.args, session.arm, session.sensor, session.log_event
    force, _ = read_wrench(sensor, args.sample_timeout)
    delta_fx = force[0] - descend_fx_base
    descended_mm = alpha * args.post_entry_descend_mm
    now = time.monotonic()
    if session.last_descend_output_at is None or now - session.last_descend_output_at >= 0.1:
        session.last_descend_output_at = now
        print(f"[post_entry_descend] down≈{descended_mm:.2f}mm raw dFx={delta_fx:.2f}N")
    if delta_fx >= args.post_entry_descend_fx_rise_n:
        arm.stop_servo_pose(2)  # type: ignore[attr-defined]
        session.continuous_servo_stopped = True
        log_event(
            "post_entry_descend_fx_stop",
            descended_mm=round(descended_mm, 3), raw_fx_n=force[0],
            raw_fx_baseline_n=descend_fx_base, raw_delta_fx_n=delta_fx,
            threshold_n=args.post_entry_descend_fx_rise_n,
        )
        raise PostEntryFxRiseReached()


def post_entry_relief_if_needed(session: PlacementSession, sdk: object) -> None:
    """Back off after entry, then descend while monitoring raw Fx if enabled."""
    assert session.arm is not None and session.sensor is not None and session.log_file is not None and session.log_event is not None
    args, arm, sensor, log_file, log_event = session.args, session.arm, session.sensor, session.log_file, session.log_event
    if session.servo_started:
        if not session.continuous_servo_stopped:
            arm.stop_servo_pose(2)  # type: ignore[attr-defined]
            session.continuous_servo_stopped = True
        wait_for_right_task_stopped(arm, log_file, confirmed_event="right_pre_relief_stop_confirmed", timeout_event="right_pre_relief_stop_wait_timeout", cancel_if_still_running_after_s=0.5)
    if args.post_entry_relief_mm > 0.0:
        actual = current_pose(arm, 2)
        relief_start = actual[:3] + rotvec_to_rpy(actual[3:])
        relief_target = list(relief_start)
        relief_target[session.search_axis_index] -= session.search_direction_sign * args.post_entry_relief_mm / 1000.0
        print(f"进洞确认后，沿世界 {args.search_axis.upper()}{'−' if session.search_direction_sign > 0.0 else '+'} 固定后拉 {args.post_entry_relief_mm:.3f}mm。")
        arm.start_servo_pose(2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0)  # type: ignore[attr-defined]
        session.continuous_servo_stopped = False
        try:
            move_continuous_with_recovery(session, sdk, move_continuous_linear_step, "post_entry_relief", relief_start, relief_target, linear_speed_mps=args.post_entry_relief_speed_mm_s / 1000.0, linear_accel_mps2=args.post_entry_linear_accel_mm_s2 / 1000.0, initial_speed_mps=0.0)
        finally:
            arm.stop_servo_pose(2)  # type: ignore[attr-defined]
            wait_for_right_task_stopped(arm, log_file, confirmed_event="right_post_entry_relief_stop_confirmed", timeout_event="right_post_entry_relief_stop_wait_timeout", cancel_if_still_running_after_s=0.5)
            session.continuous_servo_stopped = True
        log_event("post_entry_relief_completed", distance_mm=args.post_entry_relief_mm)
    if args.post_entry_descend_mm <= 0.0:
        return
    descend_start_raw = current_pose(arm, 2)
    descend_start = descend_start_raw[:3] + rotvec_to_rpy(descend_start_raw[3:])
    descend_target = list(descend_start)
    descend_target[2] -= args.post_entry_descend_mm / 1000.0
    monitoring_enabled = args.post_entry_descend_fx_rise_n > 0.0
    descend_fx_base = None
    if monitoring_enabled:
        time.sleep(0.25)
        samples: list[float] = []
        for index in range(args.baseline_samples):
            samples.append(read_wrench(sensor, args.sample_timeout)[0][0])
            if index + 1 < args.baseline_samples:
                time.sleep(0.05)
        descend_fx_base = sum(samples) / len(samples)
    print(f"进洞确认后，直接沿世界 Z− 下降 {args.post_entry_descend_mm:.3f}mm；" + (f"raw Fx 上升 {args.post_entry_descend_fx_rise_n:.2f}N 即停止。" if monitoring_enabled else "不读取力传感器。"))
    log_event("post_entry_descend_start", max_distance_mm=args.post_entry_descend_mm, raw_fx_monitoring_enabled=monitoring_enabled, raw_fx_baseline_n=descend_fx_base, raw_fx_rise_threshold_n=args.post_entry_descend_fx_rise_n if monitoring_enabled else None, start_xyz_rpy=descend_start, target_xyz_rpy=descend_target)
    session.last_descend_output_at = None
    arm.start_servo_pose(2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0)  # type: ignore[attr-defined]
    session.continuous_servo_stopped = False
    stopped_by_fx = False
    try:
        move_continuous_with_recovery(session, sdk, move_continuous_linear_step, "post_entry_descend", descend_start, descend_target, linear_speed_mps=args.post_entry_descend_speed_mm_s / 1000.0, linear_accel_mps2=args.post_entry_linear_accel_mm_s2 / 1000.0, initial_speed_mps=0.0, force_monitor=partial(monitor_descend_raw_fx, session=session, descend_fx_base=descend_fx_base) if monitoring_enabled else None)
    except PostEntryFxRiseReached:
        stopped_by_fx = True
        print("进洞后下降：raw Fx 上升达到阈值，已停止。")
    finally:
        arm.stop_servo_pose(2)  # type: ignore[attr-defined]
        wait_for_right_task_stopped(arm, log_file, confirmed_event="right_post_entry_descend_stop_confirmed", timeout_event="right_post_entry_descend_stop_wait_timeout", cancel_if_still_running_after_s=0.5)
        session.continuous_servo_stopped = True
    log_event("post_entry_descend_completed", max_distance_mm=args.post_entry_descend_mm, stopped_by_raw_fx_rise=stopped_by_fx)


def run_second_hole_leg(session: PlacementSession, sdk: object, label: str, axis: int, distance_mm: float, speed_mm_s: float) -> None:
    """Run one measured-pose second-hole leg without force monitoring."""
    assert session.arm is not None and session.log_file is not None and session.log_event is not None
    args, arm, log_file, log_event = (
        session.args,
        session.arm,
        session.log_file,
        session.log_event,
    )
    actual = current_pose(arm, 2)
    start = actual[:3] + rotvec_to_rpy(actual[3:])
    target = list(start)
    target[axis] += distance_mm / 1000.0
    axis_name, direction = "XYZ"[axis], "+" if distance_mm > 0.0 else "−"
    print(f"第二洞路径 {label}：沿世界 {axis_name}{direction} {abs(distance_mm):.1f}mm，{speed_mm_s:.1f}mm/s。")
    log_event("second_hole_leg_start", label=label, axis=axis_name, distance_mm=distance_mm, speed_mm_s=speed_mm_s, start_xyz_rpy=start, target_xyz_rpy=target, force_monitoring_enabled=False)
    arm.start_servo_pose(2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0)  # type: ignore[attr-defined]
    session.continuous_servo_stopped = False
    try:
        is_descend = label == "descend_into_second_hole"
        move_continuous_with_recovery(session, sdk, move_continuous_linear_step, f"second_hole_{label}", start, target, linear_speed_mps=speed_mm_s / 1000.0, linear_accel_mps2=args.second_hole_linear_accel_mm_s2 / 1000.0, initial_speed_mps=0.0, near_target_mm=args.second_hole_descend_near_target_mm if is_descend else args.second_hole_near_target_mm, recovery_speed_mps=args.second_hole_descend_recovery_speed_mm_s / 1000.0 if is_descend else None)
    finally:
        arm.stop_servo_pose(2)  # type: ignore[attr-defined]
        wait_for_right_task_stopped(arm, log_file, confirmed_event=f"right_second_hole_{label}_stop_confirmed", timeout_event=f"right_second_hole_{label}_stop_wait_timeout", cancel_if_still_running_after_s=0.5)
        session.continuous_servo_stopped = True
    completed = current_pose(arm, 2)
    log_event("second_hole_leg_completed", label=label, actual_end_xyz_rpy=completed[:3] + rotvec_to_rpy(completed[3:]))


def move_from_first_hole_to_second_hole(session: PlacementSession, sdk: object) -> None:
    """Move from confirmed first hole to the taught second-hole offset."""
    assert session.arm is not None and session.log_file is not None and session.log_event is not None
    args, arm, log_file, log_event = session.args, session.arm, session.log_file, session.log_event
    if session.servo_started:
        if not session.continuous_servo_stopped:
            arm.stop_servo_pose(2)  # type: ignore[attr-defined]
            session.continuous_servo_stopped = True
        wait_for_right_task_stopped(arm, log_file, confirmed_event="right_pre_second_hole_stop_confirmed", timeout_event="right_pre_second_hole_stop_wait_timeout", cancel_if_still_running_after_s=0.5)
    print("第一洞确认成功；执行后续路径（不读取力传感器）。")
    if args.first_hole_settle_down_mm > 0.0:
        run_second_hole_leg(
            session, sdk, "settle_down_after_first_hole", 2,
            -args.first_hole_settle_down_mm, args.second_hole_descend_speed_mm_s,
        )
        print(f"第一洞后下探完成；停留 {args.second_hole_pause_after_first_down_s:.1f} 秒。")
        log_event("second_hole_pause", label="after_first_hole_down", duration_s=args.second_hole_pause_after_first_down_s)
        time.sleep(args.second_hole_pause_after_first_down_s)
    else:
        log_event("first_hole_settle_down_skipped", reason="configured_zero_distance")
        print("第一洞确认后不下压，直接抬升。")
    run_second_hole_leg(session, sdk, "lift_from_first_hole", 2, args.first_hole_lift_mm, args.second_hole_lateral_speed_mm_s)
    run_second_hole_leg(session, sdk, "shift_to_second_hole", 1, args.second_hole_y_offset_mm, args.second_hole_lateral_speed_mm_s)
    run_second_hole_leg(session, sdk, "descend_into_second_hole", 2, -args.second_hole_descend_mm, args.second_hole_descend_speed_mm_s)
    print("第二洞下降完成；停留 0.5 秒。")
    log_event("second_hole_pause", label="after_second_hole_down", duration_s=args.second_hole_pause_after_second_down_s)
    time.sleep(args.second_hole_pause_after_second_down_s)
    run_second_hole_leg(session, sdk, "lift_after_second_hole", 2, args.second_hole_lift_mm, args.second_hole_lateral_speed_mm_s)
    final_axis = "xyz".index(args.second_hole_final_axis)
    final_sign = 1.0 if args.second_hole_final_direction == "positive" else -1.0
    run_second_hole_leg(session, sdk, f"shift_{args.second_hole_final_axis}_{args.second_hole_final_direction}_after_second_hole", final_axis, final_sign * args.second_hole_final_distance_mm, args.second_hole_final_speed_mm_s)
    print("第二洞后续路径结束。")
