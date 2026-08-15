"""Insertion-event decisions shared by continuous and wiggle search."""

from __future__ import annotations

from placement_session import PlacementSession
from wrench_utils import median, norm
import time


def wiggle_entered(session: PlacementSession, back_mm: float, centre_force: list[float], centre_torque: list[float], force: list[float], torque: list[float]) -> bool:
    args = session.args
    return (back_mm >= args.min_entry_back_mm and centre_force[0] >= args.wiggle_entry_pre_fx_n and centre_torque[1] >= args.wiggle_entry_pre_ty_nm and force[0] < 0.0 and torque[1] < 0.0 and centre_force[0] - force[0] >= args.wiggle_entry_fx_drop_n and centre_torque[1] - torque[1] >= args.wiggle_entry_ty_drop_nm)


def report_wiggle_entry(session: PlacementSession, event: str, cycle: int, back_mm: float, centre_force: list[float], centre_torque: list[float], force: list[float], torque: list[float]) -> None:
    session.log_event("insertion_decision", result="entered_by_wiggle_fx_ty_sign_reversal", motion=event, cycle=cycle, back_mm=round(back_mm, 3), delta_force_n=force, delta_torque_nm=torque)  # type: ignore[operator]
    print(f"SUCCESS：检测到摆动进洞的 Fx/Ty 正负突变； event={event} back={back_mm:.1f}mm dFx={centre_force[0]:.2f}→{force[0]:.2f}N (drop={centre_force[0] - force[0]:.2f}N) dTy={centre_torque[1]:.2f}→{torque[1]:.2f}N*m (drop={centre_torque[1] - torque[1]:.2f}N*m)")


def release_event_should_stop(session: PlacementSession, signature: str, **details: object) -> bool:
    """Count independently validated unloads; a profile may require repeats."""
    session.release_events_seen += 1
    should_stop = session.release_events_seen >= session.args.release_events_required
    session.log_event(  # type: ignore[operator]
        "insertion_release_event",
        signature=signature,
        occurrence=session.release_events_seen,
        required_occurrences=session.args.release_events_required,
        stop=should_stop,
        **details,
    )
    if not should_stop:
        print(
            f"检测到第 {session.release_events_seen}/{session.args.release_events_required} 次合格卸载（{signature}）；"
            "继续连续搜索，等待下一次卸载。"
        )
    return should_stop



def wiggle_release_entered(
    session: PlacementSession, contact_reached: type[Exception], sample_fn: object,
    cycle: int, back_mm: float, centre_force: list[float], centre_torque: list[float],
    event: str, force: list[float], torque: list[float], target: list[float],
) -> bool:
    """Confirm a centre-contact to lateral-search unload in place."""
    if back_mm < session.args.min_entry_back_mm:
        return False
    centre_force_norm, centre_torque_norm = norm(centre_force), norm(centre_torque)
    force_norm, torque_norm = norm(force), norm(torque)
    if not (centre_force_norm >= session.args.entry_release_pre_force_n and centre_torque_norm >= session.args.entry_release_pre_torque_nm and force_norm <= session.args.entry_release_force_n and torque_norm <= session.args.entry_release_torque_nm and centre_force_norm - force_norm >= session.args.entry_release_force_drop_n and centre_torque_norm - torque_norm >= session.args.entry_release_torque_drop_nm):
        return False
    confirm_force_norms: list[float] = []
    confirm_torque_norms: list[float] = []
    for confirm_index in range(1, session.args.entry_release_confirm_samples + 1):
        time.sleep(session.args.entry_release_confirm_interval_s)
        force_row, torque_row = sample_fn(session, contact_reached, f"entry_release_confirm_{event}_{confirm_index}", cycle, back_mm, target)  # type: ignore[operator]
        confirm_force_norms.append(norm(force_row))
        confirm_torque_norms.append(norm(torque_row))
    if median(confirm_force_norms) > session.args.entry_release_force_n or median(confirm_torque_norms) > session.args.entry_release_torque_nm:
        return False
    session.log_event("insertion_decision", result="entered_by_wiggle_release_confirmed", motion=event, cycle=cycle, back_mm=round(back_mm, 3))  # type: ignore[operator]
    return True
