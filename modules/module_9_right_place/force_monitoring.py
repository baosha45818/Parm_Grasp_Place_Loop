"""Force-sensor sampling helpers for a placement session."""

from __future__ import annotations

import math
import time

from move_by_joints import current_pose, rotvec_to_rpy
from placement_session import PlacementSession
from wrench_utils import norm, read_wrench, vector


def sample(session: PlacementSession, contact_reached: type[Exception], event: str, cycle: int, back_mm: float, target: list[float]) -> tuple[list[float], list[float]]:
    force, torque = read_wrench(session.sensor, session.args.sample_timeout)
    dforce = [value - base for value, base in zip(force, session.force_base)]
    dtorque = [value - base for value, base in zip(torque, session.torque_base)]
    actual = current_pose(session.arm, 2)
    actual_rpy = rotvec_to_rpy(actual[3:])
    force_norm, torque_norm = norm(dforce), norm(dtorque)
    motion = {
        "initial": "initial",
        "back_centre": session.search_motion_name,
        "wiggle_y_plus": "left_y_positive",
        "wiggle_y_minus": "right_y_negative",
        "wiggle_return_centre": "return_to_centre",
    }.get(event, event)
    session.log_event(
        "force_sample",
        motion=motion,
        cycle=cycle,
        back_mm=round(back_mm, 3),
        target_xyz_rpy=target,
        actual_xyz_rpy=actual[:3] + actual_rpy,
        force_n=force,
        torque_nm=torque,
        delta_force_n=dforce,
        delta_torque_nm=dtorque,
        delta_force_norm_n=round(force_norm, 6),
        delta_torque_norm_nm=round(torque_norm, 6),
        insertion_decision="pending",
    )
    print(
        f"[{event}] back={back_mm:.1f}mm dF={dforce} |dF|={force_norm:.2f}N "
        f"dT={dtorque} |dT|={torque_norm:.2f}N*m"
    )
    if force_norm > session.args.max_force_delta_n or torque_norm > session.args.max_torque_delta_nm:
        if event != "initial":
            session.log_event(
                "insertion_decision",
                result="entered_by_contact_limit",
                motion=motion,
                cycle=cycle,
                back_mm=round(back_mm, 3),
                delta_force_norm_n=round(force_norm, 6),
                delta_torque_norm_nm=round(torque_norm, 6),
            )
            print(
                "SUCCESS：插孔搜索阶段达到相对力/扭矩接触限；"
                "按已进入孔位处理，停止后续 ServoPose 推送。"
                f" event={event} back={back_mm:.1f}mm "
                f"|dF|={force_norm:.2f}N（限 {session.args.max_force_delta_n:.2f}），"
                f"|dT|={torque_norm:.2f}N*m（限 {session.args.max_torque_delta_nm:.2f}）。"
            )
            raise contact_reached()
        raise RuntimeError(
            "安全停止：初始位姿相对力/扭矩超限。"
            f"|dF|={force_norm:.2f}N（限 {session.args.max_force_delta_n:.2f}），"
            f"|dT|={torque_norm:.2f}N*m（限 {session.args.max_torque_delta_nm:.2f}）。"
        )
    return dforce, dtorque



def monitor_continuous_wrench(
    session: PlacementSession,
    contact_reached: type[Exception],
    release_should_stop: object,
    back_mm: float,
) -> None:
    if session.continuous_monitor_stride == 0:
        return
    session.continuous_monitor_tick += 1
    if session.continuous_monitor_tick % session.continuous_monitor_stride:
        return
    frame = session.sensor.get_state()  # type: ignore[attr-defined]
    if frame is None:
        return
    force = vector(frame, "filtered_force")
    torque = vector(frame, "filtered_torque")
    dforce = [value - base for value, base in zip(force, session.force_base)]
    dtorque = [value - base for value, base in zip(torque, session.torque_base)]
    force_norm, torque_norm = norm(dforce), norm(dtorque)
    now = time.monotonic()
    # Persist every 50-Hz frame for post-run verification of the
    # consecutive-release rule.  Keep console output at 10 Hz.
    session.log_event(
        "continuous_force_monitor_tick",
        back_mm=round(back_mm, 3),
        monitor_hz=session.args.continuous_force_monitor_hz,
        delta_force_n=dforce,
        delta_torque_nm=dtorque,
        delta_force_norm_n=round(force_norm, 6),
        delta_torque_norm_nm=round(torque_norm, 6),
    )
    output_period_s = (
        1.0 / session.args.continuous_force_output_hz
        if session.args.continuous_force_output_hz > 0.0 else math.inf
    )
    if now - session.last_continuous_monitor_output_at >= output_period_s:
        session.last_continuous_monitor_output_at = now
        session.log_event(
            "continuous_force_monitor",
            back_mm=round(back_mm, 3),
            monitor_hz=session.args.continuous_force_monitor_hz,
            delta_force_n=dforce,
            delta_torque_nm=dtorque,
            delta_force_norm_n=round(force_norm, 6),
            delta_torque_norm_nm=round(torque_norm, 6),
        )
        print(
            f"[force_100ms] back≈{back_mm:.2f}mm |dF|={force_norm:.2f}N "
            f"|dT|={torque_norm:.2f}N*m"
        )
    if force_norm > session.args.max_force_delta_n or torque_norm > session.args.max_torque_delta_nm:
        # Do not wait for logging or a control-state RPC before
        # ending the active stream: this path protects the part
        # after a high-rate force trigger.
        session.arm.stop_servo_pose(2)  # type: ignore[attr-defined]
        session.continuous_servo_stopped = True
        session.log_event(
            "insertion_decision",
            result="entered_by_contact_limit_continuous_monitor",
            back_mm=round(back_mm, 3),
            monitor_hz=session.args.continuous_force_monitor_hz,
            delta_force_norm_n=round(force_norm, 6),
            delta_torque_norm_nm=round(torque_norm, 6),
        )
        print(
            "SUCCESS：连续力监测达到相对力/扭矩接触限；"
            "已立即停止后续 ServoPose 推送。"
            f" back≈{back_mm:.2f}mm |dF|={force_norm:.2f}N，|dT|={torque_norm:.2f}N*m"
        )
        raise contact_reached()
    fx_reversal_enabled = session.args.entry_fx_reversal_confirm_frames > 0
    if fx_reversal_enabled:
        if (
            session.fx_reversal_peak is not None
            and back_mm - session.fx_reversal_peak[1] > session.args.entry_fx_reversal_peak_window_mm
        ):
            session.fx_reversal_peak = None
        if dforce[0] >= session.args.entry_fx_reversal_pre_n:
            if session.fx_reversal_peak is None:
                session.fx_reversal_peak = (dforce[0], back_mm)
            else:
                # Retain the strongest positive Fx in this contact
                # lobe, but measure the window from its final
                # positive frame so a gradual sign reversal is not
                # discarded as a stale peak.
                session.fx_reversal_peak = (
                    max(session.fx_reversal_peak[0], dforce[0]), back_mm,
                )
        if session.fx_reversal_confirmed_frames == 0:
            fx_reversal_candidate = (
                back_mm >= session.args.min_entry_back_mm
                and session.fx_reversal_peak is not None
                and dforce[0] <= session.args.entry_fx_reversal_current_n
                and session.fx_reversal_peak[0] - dforce[0] >= session.args.entry_fx_reversal_drop_n
            )
            if fx_reversal_candidate:
                session.fx_reversal_confirmed_frames = 1
        elif dforce[0] <= session.args.entry_fx_reversal_current_n:
            session.fx_reversal_confirmed_frames += 1
        else:
            session.fx_reversal_confirmed_frames = 0
        if session.fx_reversal_confirmed_frames >= session.args.entry_fx_reversal_confirm_frames:
            session.arm.stop_servo_pose(2)  # type: ignore[attr-defined]
            session.continuous_servo_stopped = True
            session.log_event(
                "insertion_decision",
                result="entered_by_fx_negative_reversal_continuous_monitor",
                back_mm=round(back_mm, 3),
                confirm_frames=session.fx_reversal_confirmed_frames,
                peak_delta_fx_n=session.fx_reversal_peak[0] if session.fx_reversal_peak else None,
                peak_back_mm=round(session.fx_reversal_peak[1], 3) if session.fx_reversal_peak else None,
                delta_fx_n=dforce[0],
                fx_drop_n=round((session.fx_reversal_peak[0] - dforce[0]) if session.fx_reversal_peak else 0.0, 6),
            )
            print(
                "SUCCESS：连续力监测检测到并确认 Fx 快速翻转为负载；"
                "立即停止后续 ServoPose 推送。"
                f" back≈{back_mm:.2f}mm frames={session.fx_reversal_confirmed_frames} "
                f"dFx_peak={session.fx_reversal_peak[0] if session.fx_reversal_peak else 0.0:.2f}"
                f"→{dforce[0]:.2f}N"
            )
            raise contact_reached()
    resultant_enabled = session.args.entry_resultant_release_frames >= 2 and all((
        session.args.entry_resultant_pre_force_n > 0.0,
        session.args.entry_resultant_pre_torque_nm > 0.0,
        session.args.entry_resultant_force_n > 0.0,
        session.args.entry_resultant_torque_nm > 0.0,
        session.args.entry_resultant_force_drop_n > 0.0,
        session.args.entry_resultant_torque_drop_nm > 0.0,
    ))
    if resultant_enabled and back_mm >= session.args.min_entry_back_mm:
        if session.continuous_resultant_peak is None:
            if (force_norm >= session.args.entry_resultant_pre_force_n
                    and torque_norm >= session.args.entry_resultant_pre_torque_nm):
                session.continuous_resultant_peak = (force_norm, torque_norm, back_mm)
                session.continuous_resultant_release_frames = 0
        else:
            peak_force_norm, peak_torque_norm, peak_back_mm = session.continuous_resultant_peak
            if force_norm >= peak_force_norm and torque_norm >= peak_torque_norm:
                session.continuous_resultant_peak = (force_norm, torque_norm, back_mm)
                session.continuous_resultant_release_frames = 0
            elif (session.previous_continuous_wrench is not None
                  # Keep a valid unload streak through normal
                  # session.sensor-scale rebound; use the same tolerances
                  # as the Fx/Ty continuous-release detector.
                  and force_norm <= (
                      norm(session.previous_continuous_wrench[0])
                      + session.args.continuous_release_fx_rise_tolerance_n
                  )
                  and torque_norm <= (
                      norm(session.previous_continuous_wrench[1])
                      + session.args.continuous_release_ty_rise_tolerance_nm
                  )):
                session.continuous_resultant_release_frames += 1
            else:
                session.continuous_resultant_peak = None
                session.continuous_resultant_release_frames = 0
        if session.continuous_resultant_peak is not None:
            peak_force_norm, peak_torque_norm, peak_back_mm = session.continuous_resultant_peak
            resultant_entry = (
                session.continuous_resultant_release_frames >= session.args.entry_resultant_release_frames
                and force_norm <= session.args.entry_resultant_force_n
                and torque_norm <= session.args.entry_resultant_torque_nm
                and peak_force_norm - force_norm >= session.args.entry_resultant_force_drop_n
                and peak_torque_norm - torque_norm >= session.args.entry_resultant_torque_drop_nm
            )
            if resultant_entry:
                if not release_should_stop(
                    "resultant_release",
                    back_mm=round(back_mm, 3),
                    release_frames=session.continuous_resultant_release_frames,
                    release_start_back_mm=round(peak_back_mm, 3),
                    release_start_force_norm_n=round(peak_force_norm, 6),
                    release_start_torque_norm_nm=round(peak_torque_norm, 6),
                    delta_force_norm_n=round(force_norm, 6),
                    delta_torque_norm_nm=round(torque_norm, 6),
                ):
                    session.continuous_resultant_peak = None
                    session.continuous_resultant_release_frames = 0
                    session.previous_continuous_wrench = (dforce, dtorque)
                    return
                session.arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                session.continuous_servo_stopped = True
                session.log_event(
                    "insertion_decision",
                    result="entered_by_resultant_release_continuous_monitor",
                    back_mm=round(back_mm, 3),
                    release_frames=session.continuous_resultant_release_frames,
                    release_start_back_mm=round(peak_back_mm, 3),
                    release_start_force_norm_n=round(peak_force_norm, 6),
                    release_start_torque_norm_nm=round(peak_torque_norm, 6),
                    delta_force_norm_n=round(force_norm, 6),
                    delta_torque_norm_nm=round(torque_norm, 6),
                )
                print(
                    "SUCCESS：检测到合力/合扭矩连续卸载，已停止后续 ServoPose 推送。"
                    f" back≈{back_mm:.2f}mm frames={session.continuous_resultant_release_frames} "
                    f"|dF|={peak_force_norm:.2f}→{force_norm:.2f}N "
                    f"|dT|={peak_torque_norm:.2f}→{torque_norm:.2f}N*m"
                )
                raise contact_reached()
    if session.previous_continuous_wrench is not None and back_mm >= session.args.min_entry_back_mm:
        previous_force, previous_torque = session.previous_continuous_wrench
        strong_entry = (
            previous_force[0] >= session.args.entry_pre_fx_n
            and previous_torque[1] >= session.args.entry_pre_ty_nm
            and previous_force[0] - dforce[0] >= session.args.entry_fx_drop_n
            and (
                previous_torque[1] - dtorque[1] >= session.args.entry_ty_drop_nm
                or abs(dtorque[1]) <= session.args.entry_ty_abs_nm
            )
        )
        relaxed_entry = (
            previous_force[0] >= session.args.entry_relaxed_pre_fx_n
            and previous_torque[1] >= session.args.entry_relaxed_pre_ty_nm
            and previous_force[0] - dforce[0] >= session.args.entry_relaxed_fx_drop_n
            and (
                previous_torque[1] - dtorque[1] >= session.args.entry_relaxed_ty_drop_nm
                or abs(dtorque[1]) <= session.args.entry_ty_abs_nm
            )
        )
        if strong_entry or relaxed_entry:
            if not release_should_stop(
                "fx_ty_single_frame_release",
                back_mm=round(back_mm, 3),
                delta_force_n=dforce,
                delta_torque_nm=dtorque,
                previous_delta_force_n=previous_force,
                previous_delta_torque_nm=previous_torque,
            ):
                session.previous_continuous_wrench = (dforce, dtorque)
                return
            session.arm.stop_servo_pose(2)  # type: ignore[attr-defined]
            session.continuous_servo_stopped = True
            session.log_event(
                "insertion_decision",
                result=("entered_by_strong_fx_ty_release_continuous_monitor"
                        if strong_entry else "entered_by_relaxed_fx_ty_release_continuous_monitor"),
                back_mm=round(back_mm, 3),
                monitor_hz=session.args.continuous_force_monitor_hz,
                previous_delta_force_n=previous_force,
                previous_delta_torque_nm=previous_torque,
                delta_force_n=dforce,
                delta_torque_nm=dtorque,
            )
            print(
                "SUCCESS：连续力监测检测到进洞的 Fx/Ty 同步突变卸载；"
                "立即停止后续 ServoPose 推送。"
                f" back≈{back_mm:.2f}mm dFx={previous_force[0]:.2f}→{dforce[0]:.2f}N "
                f"dTy={previous_torque[1]:.2f}→{dtorque[1]:.2f}N*m"
            )
            raise contact_reached()
    contact_level = (
        dforce[0] >= session.args.entry_relaxed_pre_fx_n
        and dtorque[1] >= session.args.entry_relaxed_pre_ty_nm
    )
    if session.continuous_contact_peak is None:
        if contact_level:
            session.continuous_contact_peak = (dforce, dtorque, back_mm)
            session.continuous_release_frames = 0
    else:
        peak_force, peak_torque, peak_back_mm = session.continuous_contact_peak
        near_peak = (
            dforce[0] >= peak_force[0] - session.args.continuous_release_fx_rise_tolerance_n
            and dtorque[1] >= peak_torque[1] - session.args.continuous_release_ty_rise_tolerance_nm
        )
        if near_peak and contact_level:
            # Refresh the reference while contact remains at its
            # local maximum, so release is measured from the last
            # loaded frame rather than an old approach frame.
            session.continuous_contact_peak = (dforce, dtorque, back_mm)
            session.continuous_release_frames = 0
        elif session.previous_continuous_wrench is not None:
            previous_force, previous_torque = session.previous_continuous_wrench
            descending = (
                dforce[0] <= previous_force[0] + session.args.continuous_release_fx_rise_tolerance_n
                and dtorque[1] <= previous_torque[1] + session.args.continuous_release_ty_rise_tolerance_nm
            )
            if descending:
                session.continuous_release_frames += 1
            elif contact_level:
                session.continuous_contact_peak = (dforce, dtorque, back_mm)
                session.continuous_release_frames = 0
            else:
                session.continuous_contact_peak = None
                session.continuous_release_frames = 0
    session.log_event(
        "continuous_release_state",
        back_mm=round(back_mm, 3),
        release_frames=session.continuous_release_frames,
        peak_back_mm=(round(session.continuous_contact_peak[2], 3)
                      if session.continuous_contact_peak is not None else None),
        peak_delta_force_n=(session.continuous_contact_peak[0]
                            if session.continuous_contact_peak is not None else None),
        peak_delta_torque_nm=(session.continuous_contact_peak[1]
                             if session.continuous_contact_peak is not None else None),
    )
    if (
        session.continuous_contact_peak is not None
        and session.continuous_release_frames >= session.args.continuous_release_frames
        and back_mm >= session.args.min_entry_back_mm
    ):
        release_start_force, release_start_torque, release_start_back_mm = session.continuous_contact_peak
        strong_entry = (
            release_start_force[0] >= session.args.entry_pre_fx_n
            and release_start_torque[1] >= session.args.entry_pre_ty_nm
            and release_start_force[0] - dforce[0] >= session.args.entry_fx_drop_n
            and (
                release_start_torque[1] - dtorque[1] >= session.args.entry_ty_drop_nm
                or abs(dtorque[1]) <= session.args.entry_ty_abs_nm
            )
        )
        relaxed_entry = (
            release_start_force[0] >= session.args.entry_relaxed_pre_fx_n
            and release_start_torque[1] >= session.args.entry_relaxed_pre_ty_nm
            and release_start_force[0] - dforce[0] >= session.args.entry_relaxed_fx_drop_n
            and (
                release_start_torque[1] - dtorque[1] >= session.args.entry_relaxed_ty_drop_nm
                or abs(dtorque[1]) <= session.args.entry_ty_abs_nm
            )
        )
        if strong_entry or relaxed_entry:
            if not release_should_stop(
                "fx_ty_multi_frame_release",
                back_mm=round(back_mm, 3),
                release_frames=session.continuous_release_frames,
                release_start_back_mm=round(release_start_back_mm, 3),
                release_start_delta_force_n=release_start_force,
                release_start_delta_torque_nm=release_start_torque,
                delta_force_n=dforce,
                delta_torque_nm=dtorque,
            ):
                session.continuous_contact_peak = None
                session.continuous_release_frames = 0
                session.previous_continuous_wrench = (dforce, dtorque)
                return
            session.arm.stop_servo_pose(2)  # type: ignore[attr-defined]
            session.continuous_servo_stopped = True
            session.log_event(
                "insertion_decision",
                result=("entered_by_strong_fx_ty_release_continuous_streak"
                        if strong_entry else "entered_by_relaxed_fx_ty_release_continuous_streak"),
                back_mm=round(back_mm, 3),
                monitor_hz=session.args.continuous_force_monitor_hz,
                release_frames=session.continuous_release_frames,
                release_start_back_mm=round(release_start_back_mm, 3),
                release_start_delta_force_n=release_start_force,
                release_start_delta_torque_nm=release_start_torque,
                delta_force_n=dforce,
                delta_torque_nm=dtorque,
            )
            print(
                "SUCCESS：连续力监测检测到多帧 Fx/Ty 同步卸载；"
                "立即停止后续 ServoPose 推送。"
                f" back≈{back_mm:.2f}mm frames={session.continuous_release_frames} "
                f"dFx={release_start_force[0]:.2f}→{dforce[0]:.2f}N "
                f"dTy={release_start_torque[1]:.2f}→{dtorque[1]:.2f}N*m"
            )
            raise contact_reached()
    session.previous_continuous_wrench = (dforce, dtorque)
