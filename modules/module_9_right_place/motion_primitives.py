"""Reusable right-arm joint and ServoPose motion primitives."""

from __future__ import annotations

import math
import time
from pathlib import Path

from loop_config import load_config
from move_by_joints import (
    current_pose,
    minimum_scurve_duration,
    orientation_error_rad,
    position_error_m,
    quat_to_rotvec,
    rotvec_to_quat,
    rotvec_to_rpy,
    slerp,
    smoothstep5,
)
from pose_utils import command_for
from cli_and_logging import write_control_snapshot
from placement_session import PlacementSession
from right_arm_state import current_right_joints_deg, wait_for_right_joint_target
from servo_worker import push_servo_pose_with_timeout
from tool.planner_runtime import PlannerRuntime


class TargetNotReached(RuntimeError):
    """A commanded servo target remained outside its endpoint tolerance."""


def move_to_saved_joints_servo(arm: object, target_deg: list[float]) -> None:
    """Continuously interpolate the recorded seven joint angles through servo_j."""
    start_deg = current_right_joints_deg(arm)
    largest_delta = max(abs(end - start) for start, end in zip(start_deg, target_deg))
    print(
        "关节 servo 回起点："
        f"当前={[round(value, 3) for value in start_deg]}，"
        f"目标={[round(value, 3) for value in target_deg]}，"
        f"最大关节差={largest_delta:.2f}°。"
    )
    if largest_delta <= 0.5:
        return
    hz, max_speed_deg_s = 50.0, 3.0
    duration = max(1.0 / hz, 1.875 * largest_delta / max_speed_deg_s)
    steps = max(1, math.ceil(duration * hz))
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        alpha = smoothstep5(index / float(steps))
        target = [start + alpha * (end - start) for start, end in zip(start_deg, target_deg)]
        arm.servo_j(2, target)  # type: ignore[attr-defined]
        deadline += 1.0 / hz
        time.sleep(max(0.0, deadline - time.monotonic()))
    settle_deadline = time.monotonic() + 8.0
    final = start_deg
    while time.monotonic() < settle_deadline:
        arm.servo_j(2, target_deg)  # type: ignore[attr-defined]
        final = current_right_joints_deg(arm)
        if max(abs(end - actual) for end, actual in zip(target_deg, final)) <= 0.5:
            print("关节 servo 已到保存起点。")
            return
        time.sleep(1.0 / hz)
    raise TargetNotReached(
        "关节 servo 未在超时内到位："
        f"最大关节误差={max(abs(end - actual) for end, actual in zip(target_deg, final)):.2f}°。"
    )


def move_to_saved_joints_planned(
    arm: object, config_path: Path, target_deg: list[float], *, planner_name: str,
) -> None:
    """Plan against the configured scene (including the table), then execute once."""
    config = load_config(config_path, live=True)
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        print(
            f"Planner {planner_name.upper()} 回起点：先使用当前规划场景做右臂碰撞检查和绕障规划，"
            f"目标={[round(value, 3) for value in target_deg]}。"
        )
        planner.move_right_joints(target_deg, planner=planner_name)
        print("Planner move_j 已下发，等待右臂物理到固定关节起点。")
        wait_for_right_joint_target(arm, target_deg, "固定关节起点")
    finally:
        planner.close()


def move_in_active_session(
    arm: object,
    target_rpy: list[float],
    *,
    initial: bool = False,
    position_tolerance_m: float | None = None,
    linear_speed_mps: float | None = None,
    linear_accel_mps2: float | None = None,
    settle_timeout_s: float | None = None,
    diagnostic: object | None = None,
) -> None:
    """Move within the caller-owned continuous servo-pose session."""
    target = command_for(target_rpy)
    start = current_pose(arm, 2)
    distance = position_error_m(target, start)
    angle = orientation_error_rad(target, start)
    if diagnostic is not None:
        diagnostic("servo_move_begin")  # type: ignore[operator]
    position_tolerance = position_tolerance_m if position_tolerance_m is not None else (0.0005 if initial else 0.0003)
    orientation_tolerance = 0.02 if initial else 0.01
    if distance <= position_tolerance and angle <= orientation_tolerance:
        return
    hz = 50.0
    period = 1.0 / hz
    speed = 0.01 if initial else (linear_speed_mps if linear_speed_mps is not None else 0.003)
    accel = 0.02 if initial else (linear_accel_mps2 if linear_accel_mps2 is not None else 0.01)
    linear_duration = minimum_scurve_duration(
        distance, speed, accel
    )
    angular_duration = minimum_scurve_duration(
        angle, 0.10 if initial else 0.05, 0.20 if initial else 0.10
    )
    steps = max(1, math.ceil(max(linear_duration, angular_duration, period) * hz))
    q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        alpha = smoothstep5(index / float(steps))
        waypoint = [start[axis] + alpha * (target[axis] - start[axis]) for axis in range(3)]
        waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
        push_servo_pose_with_timeout(arm, waypoint)
        if diagnostic is not None and (index == 1 or index % 10 == 0 or index == steps):
            diagnostic(f"servo_push_waypoint_{index}_of_{steps}")  # type: ignore[operator]
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))
    settle_deadline = time.monotonic() + (
        settle_timeout_s if settle_timeout_s is not None else (8.0 if initial else 4.0)
    )
    final_pose = start
    settle_push_count = 0
    while time.monotonic() < settle_deadline:
        settle_push_count += 1
        push_servo_pose_with_timeout(arm, target)
        final_pose = current_pose(arm, 2)
        if (
            position_error_m(target, final_pose) <= position_tolerance
            and orientation_error_rad(target, final_pose) <= orientation_tolerance
        ):
            if diagnostic is not None:
                diagnostic("servo_target_reached")  # type: ignore[operator]
            return
        if diagnostic is not None and settle_push_count % 10 == 0:
            diagnostic(f"servo_settle_push_{settle_push_count}")  # type: ignore[operator]
        time.sleep(period)
    if diagnostic is not None:
        diagnostic("servo_target_timeout")  # type: ignore[operator]
    raise TargetNotReached(
        "末端未在超时内到位："
        f"位置误差={position_error_m(target, final_pose) * 1000.0:.1f}mm，"
        f"姿态误差={math.degrees(orientation_error_rad(target, final_pose)):.2f}°。"
    )


def move_continuous_linear_step(
    arm: object,
    command_start_rpy: list[float],
    target_rpy: list[float],
    *,
    linear_speed_mps: float,
    linear_accel_mps2: float,
    initial_speed_mps: float,
    force_monitor: object | None = None,
    push_diagnostic: object | None = None,
    decelerate_to_stop: bool = False,
) -> float:
    """Stream one search increment at constant command velocity.

    This intentionally performs no endpoint settle/readback.  Consecutive
    calls share their command endpoint, so the controller receives a single
    linear stream instead of a sequence of one-millimetre S-curves.  The
    caller samples only after the stream has crossed each 1-mm checkpoint.
    Acceleration is carried across checkpoints, so a low configured
    acceleration produces one smooth ramp rather than 1-mm stop/start ramps.
    ``decelerate_to_stop`` is for a standalone approach leg: it uses a
    trapezoidal (or triangular) profile and reaches zero command velocity at
    the endpoint.
    """
    start = command_for(command_start_rpy)
    target = command_for(target_rpy)
    distance = position_error_m(target, start)
    if distance <= 0.0:
        return initial_speed_mps
    hz = 50.0
    period = 1.0 / hz
    speed0 = min(max(initial_speed_mps, 0.0), linear_speed_mps)
    if decelerate_to_stop:
        # Each side of a full-speed ramp consumes v²/(2a).  If the leg is
        # shorter, use a triangular profile whose peak exactly fits distance.
        full_ramp_distance = linear_speed_mps * linear_speed_mps / linear_accel_mps2
        speed1 = (
            linear_speed_mps
            if distance >= full_ramp_distance
            else math.sqrt(distance * linear_accel_mps2)
        )
        accel_time = speed1 / linear_accel_mps2
        accel_distance = speed1 * accel_time / 2.0
        cruise_distance = max(0.0, distance - 2.0 * accel_distance)
        cruise_time = cruise_distance / speed1 if speed1 > 0.0 else 0.0
        duration = 2.0 * accel_time + cruise_time
    else:
        speed1 = min(linear_speed_mps, math.sqrt(speed0 * speed0 + 2.0 * linear_accel_mps2 * distance))
        accel_time = (speed1 - speed0) / linear_accel_mps2
        accel_distance = (speed0 + speed1) * accel_time / 2.0
        cruise_distance = max(0.0, distance - accel_distance)
        duration = accel_time + cruise_distance / linear_speed_mps
    steps = max(1, math.ceil(duration * hz))
    q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        elapsed = min(index * duration / steps, duration)
        if decelerate_to_stop:
            if elapsed <= accel_time:
                travelled = linear_accel_mps2 * elapsed * elapsed / 2.0
            elif elapsed <= accel_time + cruise_time:
                travelled = accel_distance + speed1 * (elapsed - accel_time)
            else:
                decel_elapsed = elapsed - accel_time - cruise_time
                travelled = (
                    accel_distance + cruise_distance
                    + speed1 * decel_elapsed
                    - linear_accel_mps2 * decel_elapsed * decel_elapsed / 2.0
                )
        elif elapsed <= accel_time:
            travelled = speed0 * elapsed + linear_accel_mps2 * elapsed * elapsed / 2.0
        else:
            travelled = accel_distance + linear_speed_mps * (elapsed - accel_time)
        alpha = min(1.0, travelled / distance)
        waypoint = [start[axis] + alpha * (target[axis] - start[axis]) for axis in range(3)]
        waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
        if push_diagnostic is not None:
            push_diagnostic(  # type: ignore[operator]
                "servo_push_before", waypoint_index=index, waypoint_count=steps,
                waypoint_xyz_rotvec=waypoint,
            )
        try:
            push_servo_pose_with_timeout(arm, waypoint)
        except Exception as exc:
            if push_diagnostic is not None:
                push_diagnostic(  # type: ignore[operator]
                    "servo_push_failed", waypoint_index=index, waypoint_count=steps,
                    waypoint_xyz_rotvec=waypoint,
                    exception_type=type(exc).__name__, exception=str(exc),
                )
            raise
        if force_monitor is not None:
            force_monitor(alpha)  # type: ignore[operator]
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))
    return 0.0 if decelerate_to_stop else speed1


def move_wiggle_quick(
    arm: object,
    target_rpy: list[float],
    *,
    linear_speed_mps: float,
    linear_accel_mps2: float,
    endpoint_tolerance_m: float,
    endpoint_wait_s: float,
    diagnostic: object | None = None,
) -> None:
    """Stream one fast search leg, confirming its endpoint before sampling.

    The servo-pose session is never restarted.  Unlike the standard leg, this
    waits only a short time for a 0.5-mm endpoint confirmation.  If the arm is
    still outside that envelope, it falls back to the original 0.3-mm / 4-s
    confirmation before force is read; a quick run therefore never samples an
    unconfirmed endpoint merely to save time.
    """
    try:
        move_in_active_session(
            arm, target_rpy,
            position_tolerance_m=endpoint_tolerance_m,
            linear_speed_mps=linear_speed_mps,
            linear_accel_mps2=linear_accel_mps2,
            settle_timeout_s=endpoint_wait_s,
            diagnostic=diagnostic,
        )
    except TargetNotReached:
        print("快速运动段未在短确认窗口到位；退回完整到位确认后再采样。")
        move_in_active_session(
            arm, target_rpy,
            linear_speed_mps=linear_speed_mps,
            linear_accel_mps2=linear_accel_mps2,
            diagnostic=diagnostic,
        )



def pre_search_approach_leg(
    session: PlacementSession, sdk: object, move_with_recovery: object,
    label: str, axis: int, distance_mm: float, speed_mm_s: float,
) -> None:
    actual = current_pose(session.arm, 2)
    start = actual[:3] + rotvec_to_rpy(actual[3:])
    target = list(start)
    target[axis] += distance_mm / 1000.0
    axis_name = "XYZ"[axis]
    direction = "+" if distance_mm > 0.0 else "−"
    print(
        f"搜索前接近 {label}：沿世界 {axis_name}{direction} "
        f"{abs(distance_mm):.1f}mm，"
        f"{speed_mm_s:.1f}mm/s。"
    )
    session.log_event(
        "pre_search_approach_leg_start", label=label, axis=axis_name,
        distance_mm=distance_mm,
        speed_mm_s=speed_mm_s,
        start_xyz_rpy=start, target_xyz_rpy=target,
        force_monitoring_enabled=False,
    )
    def push_diagnostic(event: str, **fields: object) -> None:
        """Persist every approach push and the exact failed waypoint."""
        session.log_event(
            f"pre_search_{event}", label=label, axis=axis_name,
            **fields,
        )
        if event == "servo_push_failed":
            try:
                write_control_snapshot(
                    session.arm, session.log_file,
                    f"pre_search_{label}_servo_push_failed",
                )
            except Exception as snapshot_exc:
                session.log_event(
                    "pre_search_failure_snapshot_failed", label=label,
                    exception_type=type(snapshot_exc).__name__,
                    exception=str(snapshot_exc),
                )

    move_with_recovery(
        session, sdk, move_continuous_linear_step,
        f"pre_search_{label}", start, target,
        linear_speed_mps=speed_mm_s / 1000.0,
        linear_accel_mps2=session.args.pre_search_leg_accel_mm_s2 / 1000.0, initial_speed_mps=0.0,
        push_diagnostic=push_diagnostic,
        decelerate_to_stop=True,
        confirm_physical_endpoint=True,
    )
    finished = current_pose(session.arm, 2)
    session.log_event(
        "pre_search_approach_leg_completed", label=label,
        actual_end_xyz_rpy=finished[:3] + rotvec_to_rpy(finished[3:]),
    )
