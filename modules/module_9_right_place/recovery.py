"""Safe recovery actions between failed placement-search attempts."""

from __future__ import annotations

import os

from cli_and_logging import RETRY_LIFT_MM, RETRY_LIFT_SPEED_MM_S
from move_by_joints import current_pose, ensure_ready, import_sdk, rotvec_to_rpy
from motion_primitives import move_in_active_session
from right_arm_state import wait_for_right_task_stopped
from tool.left_motion_right_servo_lock import right_servo_guard


def lift_after_exhausted_search(
    *, arm_host: str | None, session_log_event: object | None = None,
) -> None:
    """Safely clear the fixture before the next attempt re-plans to the taught pose."""
    if session_log_event is not None:
        session_log_event(  # type: ignore[operator]
            "retry_lift_recovery_connecting",
            distance_mm=RETRY_LIFT_MM,
            speed_mm_s=RETRY_LIFT_SPEED_MM_S,
            axis="z",
            direction="positive",
        )
    sdk = import_sdk()
    arm = None
    servo_guard = None
    servo_started = False
    try:
        arm = sdk.Arm(host=arm_host or os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
        ensure_ready(arm, 2)
        start = current_pose(arm, 2)
        start_rpy = start[:3] + rotvec_to_rpy(start[3:])
        target_rpy = list(start_rpy)
        target_rpy[2] += RETRY_LIFT_MM / 1000.0
        if session_log_event is not None:
            session_log_event(  # type: ignore[operator]
                "retry_lift_motion_start",
                distance_mm=RETRY_LIFT_MM,
                speed_mm_s=RETRY_LIFT_SPEED_MM_S,
                start_xyz_rpy=start_rpy,
                target_xyz_rpy=target_rpy,
            )
        print(
            "本轮未进洞：沿世界 Z+ LIN 抬升 "
            f"{RETRY_LIFT_MM:.1f}mm，随后重新规划回放置初始点。"
        )
        servo_guard = right_servo_guard()
        servo_guard.__enter__()
        arm.start_servo_pose(  # type: ignore[attr-defined]
            2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
            input_frequency_hz=50.0,
        )
        servo_started = True
        move_in_active_session(
            arm,
            target_rpy,
            linear_speed_mps=RETRY_LIFT_SPEED_MM_S / 1000.0,
            linear_accel_mps2=0.02,
        )
        print("30mm 抬升完成。")
        if session_log_event is not None:
            session_log_event(  # type: ignore[operator]
                "retry_lift_motion_completed", distance_mm=RETRY_LIFT_MM,
            )
    finally:
        try:
            if servo_started and arm is not None:
                arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                wait_for_right_task_stopped(
                    arm,
                    None,
                    confirmed_event="retry_lift_stop_confirmed",
                    timeout_event="retry_lift_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                    event_logger=session_log_event,
                )
        finally:
            if servo_guard is not None:
                servo_guard.__exit__(None, None, None)
            if arm is not None:
                arm.close()
