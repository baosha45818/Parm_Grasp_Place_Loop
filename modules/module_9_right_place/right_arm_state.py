"""Right-arm controller-state reads and task completion checks."""

from __future__ import annotations

import math
import time

from cli_and_logging import write_control_snapshot


def wait_for_right_task_stopped(
    arm: object,
    output_file: object | None,
    *,
    timeout_s: float = 3.0,
    confirmed_event: str = "right_task_stopped_confirmed",
    timeout_event: str = "right_task_stop_wait_timeout",
    cancel_if_still_running_after_s: float | None = None,
    event_logger: object | None = None,
) -> None:
    """Wait until the controller has actually retired the active right-arm task.

    ServoPose stop is asynchronous on the controller.  A short grace period
    is allowed first; callers starting a new ServoPose may request a
    cancel_task fallback so the old stream cannot overlap the new one.
    """
    deadline = time.monotonic() + timeout_s
    cancel_deadline = (
        time.monotonic() + cancel_if_still_running_after_s
        if cancel_if_still_running_after_s is not None else None
    )
    cancel_sent = False
    last_task_state = "unavailable"

    def record(event: str, task_state: str) -> None:
        if output_file is not None:
            write_control_snapshot(arm, output_file, event)
        if event_logger is not None:
            event_logger(event, task_state=task_state)  # type: ignore[operator]

    while time.monotonic() < deadline:
        state = arm.get_state() or {}  # type: ignore[attr-defined]
        task_state = str(
            (((state.get("state") or {}).get("right") or {}).get("task_state") or "")
        ).strip().lower()
        last_task_state = task_state or "unavailable"
        if task_state in {"completed", "cancelled", "idle", "stopped"}:
            record(confirmed_event, task_state)
            return
        if (
            cancel_deadline is not None
            and not cancel_sent
            and time.monotonic() >= cancel_deadline
        ):
            arm.cancel_task(2)  # type: ignore[attr-defined]
            cancel_sent = True
            record(confirmed_event + "_cancel_sent", task_state)
        time.sleep(0.05)
    record(timeout_event, last_task_state)
    raise RuntimeError(
        f"右臂任务未在 {timeout_s:.1f}s 内进入已停止状态；"
        f"最后 task_state={last_task_state!r}"
    )


def current_right_joints_deg(arm: object) -> list[float]:
    state = arm.get_state()  # type: ignore[attr-defined]
    values = ((state or {}).get("state") or {}).get("right", {}).get("position")
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise RuntimeError(f"右臂关节状态不可用: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"右臂关节状态含非有限值: {values!r}")
    return result


def wait_for_right_joint_target(
    arm: object, target_deg: list[float], label: str, *, timeout_s: float = 90.0,
) -> None:
    """Wait until the planner task has ended and the physical arm is settled."""
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    last_velocity_deg_s = float("inf")
    last_task_state = "unknown"
    stable_samples = 0
    required_stable_samples = 20  # 1 second at the 20 Hz polling rate below.
    while time.monotonic() < deadline:
        state = arm.get_state()  # type: ignore[attr-defined]
        per_arm = ((state or {}).get("state") or {}).get("right") or {}
        actual = per_arm.get("position")
        velocity = per_arm.get("velocity")
        if not isinstance(actual, (list, tuple)) or len(actual) != 7:
            stable_samples = 0
            time.sleep(0.05)
            continue
        actual_deg = [float(value) for value in actual]
        # Compare wrapped angles so equivalent values around +/-180 deg do not fail.
        errors_deg = [
            abs((actual - target + 180.0) % 360.0 - 180.0)
            for actual, target in zip(actual_deg, target_deg)
        ]
        last_error_deg = max(errors_deg)
        if isinstance(velocity, (list, tuple)) and len(velocity) == 7:
            last_velocity_deg_s = max(abs(float(value)) for value in velocity)
        else:
            last_velocity_deg_s = float("inf")
        last_task_state = str(per_arm.get("task_state", "")).strip().lower()
        settled = (
            last_error_deg <= 0.5
            and last_velocity_deg_s <= 0.05
            and last_task_state != "running"
        )
        stable_samples = stable_samples + 1 if settled else 0
        if stable_samples >= required_stable_samples:
            print(
                f"{label}已物理到位并稳定1秒：最大关节误差={last_error_deg:.3f}°，"
                f"最大关节速度={last_velocity_deg_s:.4f}°/s，"
                f"task_state={last_task_state!r}。"
            )
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"{label}未在 {timeout_s:.1f}s 内物理到位并稳定："
        f"最大关节误差={last_error_deg:.3f}°，"
        f"最大关节速度={last_velocity_deg_s:.4f}°/s，"
        f"task_state={last_task_state!r}。"
    )
