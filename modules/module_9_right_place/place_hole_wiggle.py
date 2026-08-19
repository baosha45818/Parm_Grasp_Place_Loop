#!/usr/bin/env python3
"""Continuous 15-mm/s straight-line place search with 50-Hz force monitoring.

The taught start is a fixed right-arm joint vector.  After that joint move,
the actual SDK end-effector pose is used as the world-frame centre.  By
default, a cycle backs 1 mm in world X and samples at that centre point; no
lateral motion is performed.  Force is checked at every centre endpoint.
Hole entry is declared when a previously high Fx/Ty load abruptly unloads.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
import time
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUSH_T_DIR = Path(__file__).resolve().parent / "vendor/push_t"
if str(PUSH_T_DIR) not in sys.path:
    sys.path.insert(0, str(PUSH_T_DIR))

from move_by_joints import (  # noqa: E402
    current_pose,
    ensure_ready,
    import_sdk,
    rotvec_to_rpy,
)
from tools.left_motion_right_servo_lock import right_servo_guard  # noqa: E402
from cli_and_logging import (  # noqa: E402
    WORKPIECE_CONFIGS,
    make_event_logger,
    parse_args,
    profile_argv,
    read_profile,
    write_control_snapshot,
)
from servo_worker import (  # noqa: E402
    ServoArmProxy,
    move_continuous_with_recovery,
)
from placement_session import PlacementSession  # noqa: E402
from motion_primitives import (  # noqa: E402
    pre_search_approach_leg,
    move_continuous_linear_step,
    move_in_active_session,
    move_to_saved_joints_planned,
    move_to_saved_joints_servo,
)
from recovery import lift_after_exhausted_search  # noqa: E402
from right_arm_state import (  # noqa: E402
    wait_for_right_task_stopped,
)
from wrench_utils import mean, read_wrench  # noqa: E402
from force_monitoring import monitor_continuous_wrench, sample  # noqa: E402
from insertion_detection import (
    release_event_should_stop,
)  # noqa: E402
from post_insertion import (  # noqa: E402
    move_from_first_hole_to_second_hole,
    post_entry_relief_if_needed,
)

class PlacementContactReached(RuntimeError):
    """A validated insertion-release signature stopped the search successfully."""


SEARCH_DISTANCE_EXHAUSTED_EXIT_CODE = 42
MAX_SERVO_RECOVERY_ATTEMPTS = 20


def placement_main(argv: list[str] | None = None) -> int:
    run_started = time.monotonic()
    args = parse_args(argv)
    # Recorded UI joint values are the sole taught start definition.
    saved_joints_deg = list(args.initial_right_joints_deg)
    if not args.live:
        print(
            "dry-run：不会连接硬件；固定右臂起始关节角="
            f"{saved_joints_deg}，初始回位方式={args.initial_motion}"
        )
        print(
            f"单段连续直线：沿世界 {args.search_axis.upper()}"
            f"{'+' if args.search_direction == 'positive' else '−'} 从 0 连续移动至 "
            f"{args.max_back_mm:.1f}mm；速度 {args.fast_linear_speed_mm_s:.1f}mm/s；"
            f"以 {args.continuous_force_monitor_hz:.1f}Hz（每 "
            f"{1000.0 / args.continuous_force_monitor_hz:.0f}ms）监测力/扭矩。"
        )
        return 0

    output_path = args.output or Path("logs") / f"hole_a_wiggle_{datetime.now():%Y%m%d_%H%M%S}.log"
    output_path = output_path.expanduser().resolve().with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"运行日志将实时写入：{output_path}", flush=True)
    # planner_runtime opens its own SDK Arm client for planned trajectories
    # and obtains the host only from PARM_ARM_HOST.  Keep it aligned with the
    # per-workpiece --arm-host used by the rest of this placement attempt.
    os.environ["PARM_ARM_HOST"] = args.arm_host
    sdk = import_sdk()
    arm = sensor = None
    servo_started = False
    continuous_servo_stopped = False
    servo_guard = None
    search_started_at: float | None = None
    log_file = output_path.open("w", encoding="utf-8")

    log_event = make_event_logger(log_file, started_at=run_started)
    log_event("arm_host_synced_for_planner", arm_host=args.arm_host)
    session = PlacementSession(args=args, arm=arm, sensor=sensor, log_file=log_file, log_event=log_event)

    log_event(
        "run_started",
        initial_motion=args.initial_motion,
        search_axis=args.search_axis,
        search_direction=args.search_direction,
        max_back_mm=args.max_back_mm,
        quick_endpoint_tolerance_mm=args.quick_endpoint_tolerance_mm,
        quick_endpoint_wait_s=args.quick_endpoint_wait_s,
    )
    try:
        raw_arm = sdk.Arm(host=args.arm_host)
        ensure_ready(raw_arm, 2)
        arm = ServoArmProxy(raw_arm, args.arm_host)
        session.arm = arm
        write_control_snapshot(arm, log_file, "script_connected")
        sensor = sdk.ForceSensor(host=args.force_host, hand=2)
        if args.reset_right_task_before_servo:
            print(
                "清理右臂残留任务后再规划回起点。",
                flush=True,
            )
            arm.cancel_task(2)  # type: ignore[attr-defined]
            wait_for_right_task_stopped(
                arm, log_file,
                confirmed_event="right_cancel_confirmed",
                timeout_event="right_cancel_wait_timeout",
            )
        print("正在移动到保存的起始位姿……")
        if args.initial_motion == "planner-movej":
            move_to_saved_joints_planned(
                arm, args.planner_config, saved_joints_deg,
                planner_name=args.initial_joint_planner,
            )
        else:
            move_to_saved_joints_servo(arm, saved_joints_deg)
        # The taught joint pose is a safe landmark.  Start the legacy Hole-A
        # search only after the requested world-frame approach from that pose.

        servo_guard = right_servo_guard()
        servo_guard.__enter__()
        try:
            log_event("pre_search_servo_session_starting")
            arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                input_frequency_hz=50.0,
            )
            log_event("pre_search_servo_session_started")
            approach_legs = args.pre_search_legs or [
                {"name": "initial_descend", "axis": "z", "direction": "negative", "distance_mm": args.pre_search_initial_descend_mm, "speed_mm_s": args.pre_search_approach_speed_mm_s},
                {"name": "forward", "axis": "x", "direction": "positive", "distance_mm": args.pre_search_forward_mm, "speed_mm_s": args.pre_search_approach_speed_mm_s},
                {"name": "descend", "axis": "z", "direction": "negative", "distance_mm": args.pre_search_descend_mm, "speed_mm_s": args.pre_search_approach_speed_mm_s},
            ]
            for leg in approach_legs:
                direction_sign = 1.0 if leg["direction"] == "positive" else -1.0
                pre_search_approach_leg(
                    session, sdk, move_continuous_with_recovery,
                    leg["name"], "xyz".index(leg["axis"]),
                    direction_sign * float(leg["distance_mm"]), float(leg["speed_mm_s"]),
                )
        except Exception as exc:
            log_event(
                "pre_search_servo_session_error",
                exception_type=type(exc).__name__, exception=str(exc),
            )
            try:
                write_control_snapshot(arm, log_file, "pre_search_servo_session_error")
            except Exception as snapshot_exc:
                log_event(
                    "pre_search_failure_snapshot_failed",
                    exception_type=type(snapshot_exc).__name__, exception=str(snapshot_exc),
                )
            raise
        finally:
            primary_error_active = sys.exc_info()[0] is not None
            try:
                arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                log_event("pre_search_servo_session_stop_sent")
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_pre_search_approach_stop_confirmed",
                    timeout_event="right_pre_search_approach_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                    event_logger=log_event,
                )
            except Exception as stop_exc:
                log_event(
                    "pre_search_servo_session_stop_error",
                    exception_type=type(stop_exc).__name__, exception=str(stop_exc),
                )
                if not primary_error_active:
                    raise
            servo_guard.__exit__(None, None, None)
            servo_guard = None
        # The taught pose is the recovery landmark, not the force-baseline
        # pose.  Every retry returns here, then presses the configured extra
        # distance deeper than the preceding attempt before rebuilding the
        # baseline at that new search start.
        initial_pose = current_pose(arm, 2)
        press_start = initial_pose[:3] + rotvec_to_rpy(initial_pose[3:])
        press_depth_mm = (
            args.pre_search_press_mm
            + (args.search_attempt - 1) * args.retry_search_press_increment_mm
        )
        press_increment_mm = (
            args.pre_search_press_mm
            if args.search_attempt == 1 else args.retry_search_press_increment_mm
        )
        press_target = list(press_start)
        press_axis_index = "xyz".index(args.pre_search_press_axis)
        press_direction_sign = 1.0 if args.pre_search_press_direction == "positive" else -1.0
        press_target[press_axis_index] += press_direction_sign * press_depth_mm / 1000.0
        if press_depth_mm > 0.0:
            print(
                f"已到放置初始位姿；沿世界 {args.pre_search_press_axis.upper()}"
                f"{'+' if press_direction_sign > 0.0 else '−'} LIN 移动 "
                f"{press_depth_mm:.3f}mm（第 {args.search_attempt} 轮累计深度）后再采集力基线。"
            )
        else:
            print("已到搜索位；配置为不预压，直接采集力基线。")
        log_event(
            "pre_search_press_start",
            attempt=args.search_attempt,
            distance_mm=press_depth_mm,
            incremental_depth_mm=press_increment_mm,
            axis=args.pre_search_press_axis,
            direction=args.pre_search_press_direction,
            start_xyz_rpy=press_start,
            target_xyz_rpy=press_target,
        )
        if press_depth_mm > 0.0:
            servo_guard = right_servo_guard()
            servo_guard.__enter__()
            try:
                arm.start_servo_pose(  # type: ignore[attr-defined]
                    2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                    input_frequency_hz=50.0,
                )
                servo_started = True
                continuous_servo_stopped = False
                move_in_active_session(
                    arm,
                    press_target,
                    linear_speed_mps=args.pre_search_press_speed_mm_s / 1000.0,
                    linear_accel_mps2=args.pre_search_press_accel_mm_s2 / 1000.0,
                )
            finally:
                if servo_started and not continuous_servo_stopped:
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    wait_for_right_task_stopped(
                        arm, log_file,
                        confirmed_event="right_pre_search_press_stop_confirmed",
                        timeout_event="right_pre_search_press_stop_wait_timeout",
                        cancel_if_still_running_after_s=0.5,
                    )
                    continuous_servo_stopped = True
                servo_started = False
                servo_guard.__exit__(None, None, None)
                servo_guard = None
        log_event(
            "pre_search_press_completed",
            attempt=args.search_attempt,
            distance_mm=press_depth_mm,
            incremental_depth_mm=press_increment_mm,
        )
        # Contact limits must be relative to the actual red-dot start state,
        # not the state before this script moved the arm to that point.
        print("已到力采集位置；请勿接触工件，正在稳定并采集右腕力基线……")
        time.sleep(args.baseline_settle_s)
        baseline_samples = []
        for index in range(args.baseline_samples):
            baseline_samples.append(read_wrench(sensor, args.sample_timeout))
            if index + 1 < args.baseline_samples:
                time.sleep(args.baseline_interval_s)
        force_base = mean([force for force, _ in baseline_samples])
        torque_base = mean([torque for _, torque in baseline_samples])
        session.sensor = sensor
        session.force_base = force_base
        session.torque_base = torque_base
        print(f"起始位姿力基线 F={force_base} N，T={torque_base} N*m")
        log_event(
            "force_baseline", force_n=force_base, torque_nm=torque_base,
            settle_s=args.baseline_settle_s, samples=args.baseline_samples,
            interval_s=args.baseline_interval_s,
        )

        # Read the servo centre only after the loaded arm has settled during
        # baseline collection.  Reading it before this delay leaves a stale
        # target when the held workpiece causes sub-millimetre elastic settling.
        initial_actual = current_pose(arm, 2)
        centre = initial_actual[:3] + rotvec_to_rpy(initial_actual[3:])
        print(
            "力基线采集完成；以后拉中心使用稳定后的实际末端 xyz+rpy="
            f"{[round(value, 6) for value in centre]}。"
        )
        search_axis_index = 0 if args.search_axis == "x" else 1
        search_direction_sign = 1.0 if args.search_direction == "positive" else -1.0
        search_motion_name = (
            f"world_{args.search_axis}_{args.search_direction}_search"
        )
        session.search_motion_name = search_motion_name
        session.arm = arm
        session.sensor = sensor
        session.servo_started = servo_started
        session.continuous_servo_stopped = continuous_servo_stopped
        session.search_axis_index = search_axis_index
        session.search_direction_sign = search_direction_sign
        start_search_coordinate = centre[search_axis_index]
        log_event("search_centre_ready", pose_xyz_rpy=centre)


        # Keep the single log open until the ServoPose session has stopped.
        # The SDK subscriber receives force frames at 100 Hz.  During a
        # single continuous move, inspect its latest cached frame at the
        # 50-Hz ServoPose cadence. Logging is independently rate-limited to
        # 10 Hz so it cannot obscure the control output.
        previous_continuous_wrench: tuple[list[float], list[float]] | None = None
        # A release must start from a *local high-contact* frame.  Do not
        # let a long, mildly changing contact plateau become the release
        # window; lock the most recent local peak once a real descent
        # starts and count only the following descending frames.
        continuous_contact_peak: tuple[list[float], list[float], float] | None = None
        continuous_release_frames = 0
        # Workpieces whose contact direction is not aligned with SDK Fx/Ty
        # need a sign-independent release test.  It is disabled by default
        # and enabled only by a reviewed workpiece profile.
        continuous_resultant_peak: tuple[float, float, float] | None = None
        continuous_resultant_release_frames = 0
        session.release_events_seen = 0
        fx_reversal_confirmed_frames = 0
        fx_reversal_peak: tuple[float, float] | None = None
        last_continuous_monitor_output_at = 0.0
        continuous_monitor_tick = 0
        continuous_monitor_stride = (
            max(1, round(50.0 / args.continuous_force_monitor_hz))
            if args.continuous_force_monitor_hz > 0.0
            else 0
        )
        session.previous_continuous_wrench = previous_continuous_wrench
        session.continuous_contact_peak = continuous_contact_peak
        session.continuous_release_frames = continuous_release_frames
        session.continuous_resultant_peak = continuous_resultant_peak
        session.continuous_resultant_release_frames = continuous_resultant_release_frames
        session.fx_reversal_confirmed_frames = fx_reversal_confirmed_frames
        session.fx_reversal_peak = fx_reversal_peak
        session.last_continuous_monitor_output_at = last_continuous_monitor_output_at
        session.continuous_monitor_tick = continuous_monitor_tick
        session.continuous_monitor_stride = continuous_monitor_stride
        session.continuous_servo_stopped = continuous_servo_stopped

        sample(session, PlacementContactReached, "initial", 0, 0.0, centre)
        # Arm immediately before the single continuous search segment.
        servo_guard = right_servo_guard()
        servo_guard.__enter__()
        arm.start_servo_pose(  # type: ignore[attr-defined]
            2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0
        )
        servo_started = True
        # The preparatory Z-down LIN used its own ServoPose session and
        # stopped it before baseline collection.  This is a new search
        # session, so final cleanup must stop and confirm *this* task.
        continuous_servo_stopped = False
        # The pure-search clock excludes joint return, force-baseline
        # collection, task-reset handling, and any inter-arm lock wait.
        search_started_at = time.monotonic()
        single_target = list(centre)
        single_target[search_axis_index] = (
            start_search_coordinate
            + search_direction_sign * args.max_back_mm / 1000.0
        )
        log_event(
            "motion_start",
            motion=search_motion_name,
            mode="single_continuous_segment",
            start_back_mm=0.0,
            end_back_mm=round(args.max_back_mm, 3),
            linear_speed_mm_s=round(args.fast_linear_speed_mm_s, 3),
            linear_accel_mm_s2=round(args.fast_linear_accel_mm_s2, 3),
            insertion_decision="searching",
        )
        print(
            f"[motion_profile] 单段连续 {args.search_axis.upper()}"
            f"{'+' if args.search_direction == 'positive' else '−'} 搜索："
            f"0→{args.max_back_mm:.1f}mm，"
            f"{args.fast_linear_speed_mm_s:.1f}mm/s；力监测 "
            f"{args.continuous_force_monitor_hz:.1f}Hz。"
        )
        move_continuous_with_recovery(
            session, sdk, move_continuous_linear_step,
            "first_hole_single_continuous_search",
            centre,
            single_target,
            linear_speed_mps=args.fast_linear_speed_mm_s / 1000.0,
            linear_accel_mps2=args.fast_linear_accel_mm_s2 / 1000.0,
            initial_speed_mps=0.0,
            force_monitor=lambda alpha: monitor_continuous_wrench(
                session,
                PlacementContactReached,
                lambda signature, **details: release_event_should_stop(
                    session, signature, **details,
                ),
                alpha * args.max_back_mm,
            ),
        )
        log_event(
            "insertion_decision",
            result="not_entered_max_search_distance_reached",
            back_mm=round(args.max_back_mm, 3),
        )
        if args.return_search_exhausted:
            print("本轮搜索达到最大距离，交由工件配置入口执行抬升、回初始点并重试。")
            return SEARCH_DISTANCE_EXHAUSTED_EXIT_CODE
        raise RuntimeError(
            f"在最大搜索距离 {args.max_back_mm:.1f}mm 内未检测到进洞特征。"
        )
    except PlacementContactReached:
        log_event("insertion_decision", result="entered_by_validated_release")
        session.servo_started = servo_started
        session.continuous_servo_stopped = continuous_servo_stopped
        if args.post_entry_enabled:
            post_entry_relief_if_needed(session, sdk)
        if args.second_hole_enabled:
            move_from_first_hole_to_second_hole(session, sdk)
        else:
            log_event("second_hole_skipped", reason="disabled_by_workpiece_profile")
            print("第一洞确认成功；该工件未启用第二洞路径。")
        continuous_servo_stopped = session.continuous_servo_stopped
        return 0
    except Exception as exc:
        log_event("run_error", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        search_ended_at = time.monotonic()
        pure_search_elapsed_s = (
            search_ended_at - search_started_at
            if search_started_at is not None
            else None
        )
        stop_servo_error: Exception | None = None
        if servo_started and arm is not None and not continuous_servo_stopped:
            try:
                try:
                    write_control_snapshot(arm, log_file, "before_stop_servo_pose")
                except Exception as exc:
                    print(f"警告：停止前控制状态记录失败：{exc}", file=sys.stderr)
                arm.stop_servo_pose(2)
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_servo_stop_confirmed",
                    timeout_event="right_servo_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                )
                try:
                    write_control_snapshot(arm, log_file, "after_stop_servo_pose")
                except Exception as exc:
                    print(f"警告：停止后控制状态记录失败：{exc}", file=sys.stderr)
            except Exception as exc:
                # A placement must never be reported as successful unless the
                # active ServoPose session was confirmed stopped.
                stop_servo_error = exc
        if servo_guard is not None:
            servo_guard.__exit__(None, None, None)
        if sensor is not None:
            sensor.close()
        if arm is not None:
            arm.close()
        log_event(
            "run_finished",
            elapsed_s=round(time.monotonic() - run_started, 3),
            pure_search_elapsed_s=(round(pure_search_elapsed_s, 3) if pure_search_elapsed_s is not None else None),
        )
        log_file.close()
        print(f"运行日志：{output_path}")
        if pure_search_elapsed_s is not None:
            print(f"纯搜索耗时（ServoPose 启动至搜索结束）：{pure_search_elapsed_s:.2f} 秒")
        if stop_servo_error is not None:
            raise RuntimeError("停止右臂 ServoPose 失败，拒绝报告放置成功") from stop_servo_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="以工件序号运行内置的 Hole-A 连续搜索逻辑。"
    )
    parser.add_argument("--workpiece", type=int, choices=sorted(WORKPIECE_CONFIGS), required=True)
    parser.add_argument("--live", action="store_true", help="允许连接真实硬件")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--arm-host", help="右臂控制器地址；省略时使用执行器默认值")
    parser.add_argument(
        "--reset-right-task-before-servo",
        action="store_true",
        help="开始 ServoPose 前发送 cancel_task(2) 清理右臂残留任务",
    )
    parser.add_argument("--print-command", action="store_true", help="仅打印内置执行参数，不执行")
    wrapper_args = parser.parse_args()
    if wrapper_args.live != wrapper_args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    config_path = WORKPIECE_CONFIGS[wrapper_args.workpiece]
    try:
        workpiece, profile = read_profile(config_path)
        if workpiece != wrapper_args.workpiece:
            raise ValueError(f"工件号与配置不一致：{config_path}")
    except ValueError as exc:
        parser.error(str(exc))
    if wrapper_args.print_command:
        argv = profile_argv(
            profile, workpiece, live=wrapper_args.live,
            arm_host=wrapper_args.arm_host,
            reset_right_task_before_servo=wrapper_args.reset_right_task_before_servo,
            attempt=1,
        )
        print(f"执行工件 {workpiece} 配置：{config_path}")
        print("第 1 轮内置执行参数：", shlex.join(argv))
        return 0
    session_log_path = Path(__file__).resolve().parent / "logs" / (
        f"hole_a_workpiece_{workpiece}_retry_session_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    )
    session_log_path.parent.mkdir(parents=True, exist_ok=True)
    with session_log_path.open("w", encoding="utf-8") as session_log:
        session_log_event = make_event_logger(session_log)

        session_log_event(
            "retry_session_started",
            workpiece=workpiece,
            max_attempts=profile["retry"]["max_attempts"],
            first_search_press_mm=profile["pre_search"]["press"]["first_attempt_mm"],
            retry_search_press_increment_mm=profile["pre_search"]["press"]["retry_increment_mm"],
            retry_lift_mm=profile["retry"]["lift_mm"],
        )
        print(f"重试会话日志将实时写入：{session_log_path}")
        try:
            max_attempts = profile["retry"]["max_attempts"]
            for attempt in range(1, max_attempts + 1):
                argv = profile_argv(
                    profile, workpiece, live=wrapper_args.live,
                    arm_host=wrapper_args.arm_host,
                    reset_right_task_before_servo=wrapper_args.reset_right_task_before_servo,
                    attempt=attempt,
                )
                session_log_event(
                    "attempt_started", attempt=attempt,
                    press_depth_mm=(
                        profile["pre_search"]["press"]["first_attempt_mm"]
                        + (attempt - 1) * profile["pre_search"]["press"]["retry_increment_mm"]
                    ),
                    command=argv,
                )
                print(f"执行工件 {workpiece} 配置：{config_path}（第 {attempt}/{max_attempts} 轮）")
                print("内置执行参数：", shlex.join(argv))
                result = placement_main(argv)
                session_log_event("attempt_finished", attempt=attempt, exit_code=result)
                if result == 0:
                    session_log_event("retry_session_succeeded", attempt=attempt)
                    return 0
                if result != SEARCH_DISTANCE_EXHAUSTED_EXIT_CODE:
                    return result
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"已完成 {max_attempts} 轮搜索，仍未检测到进洞特征。"
                    )
                session_log_event("retry_lift_recovery_requested", attempt=attempt)
                lift_after_exhausted_search(
                    arm_host=wrapper_args.arm_host,
                    session_log_event=session_log_event,
                    lift_mm=profile["retry"]["lift_mm"],
                    speed_mm_s=profile["retry"]["speed_mm_s"],
                    accel_mm_s2=profile["retry"]["accel_mm_s2"],
                )
        except Exception as exc:
            session_log_event(
                "retry_session_error",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        finally:
            session_log_event("retry_session_finished")
    raise AssertionError("搜索重试循环意外结束")


if __name__ == "__main__":
    script_started = time.monotonic()
    try:
        raise SystemExit(main())
    finally:
        print(f"本次执行总耗时：{time.monotonic() - script_started:.2f} 秒", flush=True)
