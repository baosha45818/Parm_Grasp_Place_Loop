"""Isolated ServoPose worker process and parent-side control proxy."""

from __future__ import annotations

import math
import multiprocessing as mp
import time

from move_by_joints import (
    current_pose,
    ensure_ready,
    orientation_error_rad,
    position_error_m,
    rotvec_to_rpy,
)
from placement_session import PlacementSession
from pose_utils import command_for
from right_arm_state import wait_for_right_task_stopped


MAX_SERVO_RECOVERY_ATTEMPTS = 20


class ServoWorkerTimeout(RuntimeError):
    """The isolated public-SDK ServoPose worker did not acknowledge in time."""


def _servo_worker_main(host: str, commands: object, replies: object) -> None:
    """Own one public SDK Arm client for one ServoPose session at a time."""
    arm = None
    try:
        from parm_sdk import Arm

        arm = Arm(host=host)
        while True:
            message = commands.recv()  # type: ignore[attr-defined]
            request_id = message["id"]
            operation = message["operation"]
            try:
                if operation == "start":
                    arm.start_servo_pose(**message["kwargs"])
                elif operation == "push":
                    arm.push_servo_pose(2, message["waypoint"])
                elif operation == "stop":
                    arm.stop_servo_pose(2)
                elif operation == "close":
                    replies.send({"id": request_id, "ok": True})  # type: ignore[attr-defined]
                    return
                else:
                    raise ValueError(f"unknown servo worker operation: {operation!r}")
                replies.send({"id": request_id, "ok": True})  # type: ignore[attr-defined]
            except Exception as exc:
                replies.send({  # type: ignore[attr-defined]
                    "id": request_id, "ok": False,
                    "exception_type": type(exc).__name__, "exception": str(exc),
                })
    except Exception as exc:
        # A start request waiting in the parent will time out if startup fails
        # before the first command can be consumed.
        try:
            replies.send({  # type: ignore[attr-defined]
                "id": None, "ok": False,
                "exception_type": type(exc).__name__, "exception": str(exc),
            })
        except Exception:
            pass
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass


class ServoWorker:
    """Parent-side RPC wrapper with a killable boundary around public SDK calls."""

    def __init__(self, host: str) -> None:
        self._context = mp.get_context("spawn")
        # A Pipe gives the parent a real ``poll(timeout)`` operation.  The
        # queue feeder thread used previously could itself wedge after the SDK
        # worker stalled, leaving the parent blocked before its watchdog ran.
        self._command_recv, self._command_send = self._context.Pipe(duplex=False)
        self._reply_recv, self._reply_send = self._context.Pipe(duplex=False)
        self._next_id = 0
        self._process = self._context.Process(
            target=_servo_worker_main,
            args=(host, self._command_recv, self._reply_send),
            daemon=True,
        )
        self._process.start()
        # The parent only sends commands and receives replies.  Closing its
        # duplicate child ends makes worker death visible as EOF/BrokenPipe.
        self._command_recv.close()
        self._reply_send.close()

    def _call(self, operation: str, *, timeout_s: float, **payload: object) -> None:
        if not self._process.is_alive():
            raise ServoWorkerTimeout(f"ServoPose worker exited before {operation}")
        self._next_id += 1
        request_id = self._next_id
        try:
            self._command_send.send({"id": request_id, "operation": operation, **payload})
            if not self._reply_recv.poll(timeout_s):
                raise TimeoutError
            reply = self._reply_recv.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.terminate()
            raise ServoWorkerTimeout(
                f"ServoPose worker connection failed during {operation}: {exc}"
            ) from exc
        except TimeoutError:
            self.terminate()
            raise ServoWorkerTimeout(
                f"ServoPose worker {operation} exceeded {timeout_s:.2f}s deadline"
            ) from None
        if reply.get("id") != request_id:
            self.terminate()
            raise ServoWorkerTimeout(
                f"ServoPose worker returned an unexpected reply during {operation}: {reply!r}"
            )
        if reply.get("ok"):
            return
        raise RuntimeError(
            f"ServoPose worker {operation} failed: "
            f"{reply.get('exception_type')}: {reply.get('exception')}"
        )

    def start(self, **kwargs: object) -> None:
        self._call("start", timeout_s=5.0, kwargs=kwargs)

    def push(self, waypoint: list[float]) -> None:
        self._call("push", timeout_s=0.5, waypoint=waypoint)

    def stop(self) -> None:
        self._call("stop", timeout_s=3.0)

    def terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1.0)
        for connection in (
            self._command_send, self._reply_recv,
        ):
            try:
                connection.close()
            except OSError:
                pass

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._call("close", timeout_s=0.5)
            except Exception:
                pass
        self.terminate()


class ServoArmProxy:
    """Delegate normal Arm API calls while isolating public ServoPose calls."""

    def __init__(self, arm: object, host: str) -> None:
        self._arm = arm
        self._host = host
        self._servo_worker: ServoWorker | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self._arm, name)

    def start_servo_pose(self, arm_id: int, **kwargs: object) -> None:
        if arm_id != 2:
            raise ValueError("此放置流程的隔离 ServoPose 仅允许右臂 arm_id=2")
        self.dispose_servo_worker()
        worker = ServoWorker(self._host)
        try:
            worker.start(arm_id=2, **kwargs)
        except Exception:
            worker.close()
            raise
        self._servo_worker = worker

    def push_servo_pose(self, arm_id: int, waypoint: list[float]) -> None:
        if arm_id != 2 or self._servo_worker is None:
            raise RuntimeError("ServoPose push requires an active isolated right-arm session")
        self._servo_worker.push(waypoint)

    def stop_servo_pose(self, arm_id: int) -> None:
        if arm_id != 2:
            raise ValueError("此放置流程的隔离 ServoPose 仅允许右臂 arm_id=2")
        worker = self._servo_worker
        self._servo_worker = None
        if worker is None:
            return
        try:
            worker.stop()
        finally:
            worker.close()

    def dispose_servo_worker(self) -> None:
        worker = self._servo_worker
        self._servo_worker = None
        if worker is not None:
            worker.close()

    def close(self) -> None:
        self.dispose_servo_worker()
        self._arm.close()


def push_servo_pose_with_timeout(
    arm: object, waypoint: list[float], *, timeout_s: float = 0.5,
) -> None:
    """Validate and send one ServoPose frame through the public SDK API."""
    if len(waypoint) != 6 or not all(math.isfinite(value) for value in waypoint):
        raise ValueError(f"ServoPose target must contain six finite values: {waypoint!r}")
    # ``timeout_s`` is retained in the signature so callers keep one common
    # interface; the SDK owns its command transport and ServoPose semantics.
    _ = timeout_s
    arm.push_servo_pose(2, waypoint)


def is_servo_connection_loss(session: PlacementSession, exc: Exception) -> bool:
    """Only recover the transient command-port loss, never motion/contact errors."""
    message = str(exc)
    return (
        isinstance(exc, ServoWorkerTimeout)
        or
        ("tcp://" in message and ":5600 unavailable" in message)
        or message == "timeout waiting for response"
        or message.startswith("socket error:")
    )


def reconnect_servo_arm(session: PlacementSession, sdk: object, label: str, recovery_index: int) -> None:
    """Replace the ZMQ client after its command port disappeared."""
    assert session.arm is not None
    # A killed worker only drops its client connection; it does not prove
    # the controller retired the previous ServoPose task.  Issue an
    # explicit public-SDK stop from a fresh worker and observe task end
    # before a recovery session is ever allowed to start.
    if isinstance(session.arm, ServoArmProxy):
        session.arm.dispose_servo_worker()
    stop_worker = ServoWorker(session.args.arm_host)
    try:
        session.log_event(
            "servo_recovery_stop_stale_session_sent", label=label,
            recovery_index=recovery_index,
        )
        stop_worker.stop()
    except Exception as stop_exc:
        session.log_event(
            "servo_recovery_stop_stale_session_error", label=label,
            recovery_index=recovery_index,
            exception_type=type(stop_exc).__name__, exception=str(stop_exc),
        )
        raise RuntimeError(
            f"{label} 恢复前无法停止旧 ServoPose 会话：{stop_exc}"
        ) from stop_exc
    finally:
        stop_worker.close()
    wait_for_right_task_stopped(
        session.arm, session.log_file,
        confirmed_event="servo_recovery_stale_session_stopped",
        timeout_event="servo_recovery_stale_session_stop_timeout",
        cancel_if_still_running_after_s=0.5,
        event_logger=session.log_event,
    )
    deadline = time.monotonic() + 10.0
    last_error: Exception | None = None
    try:
        session.arm.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    while time.monotonic() < deadline:
        candidate = None
        try:
            candidate = sdk.Arm(host=session.args.arm_host)
            ensure_ready(candidate, 2)
            # Preserve the proxy object's identity: placement_main still
            # holds that same proxy in its local ``arm`` variable.
            if isinstance(session.arm, ServoArmProxy):
                session.arm._arm = candidate
                session.arm._host = session.args.arm_host
            else:
                session.arm = ServoArmProxy(candidate, session.args.arm_host)
            session.log_event(
                "servo_reconnected", label=label,
                recovery_index=recovery_index,
            )
            return
        except Exception as exc:
            last_error = exc
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            time.sleep(0.25)
    raise RuntimeError(
        f"{label} 的 ServoPose 控制端在 10 秒内未恢复：{last_error}"
    )


def hold_servo_target_until_settled(
    session: PlacementSession,
    label: str, target_rpy: list[float], *, timeout_s: float = 15.0,
    position_tolerance_m: float = 0.001,
    orientation_tolerance_rad: float = math.radians(1.0),
    stable_samples_required: int = 10,
) -> None:
    """Keep streaming an endpoint until the physical TCP is stably there."""
    assert session.arm is not None
    target = command_for(target_rpy)
    deadline = time.monotonic() + timeout_s
    stable_samples = 0
    last_position_error_m = float("inf")
    last_orientation_error_rad = float("inf")
    while time.monotonic() < deadline:
        push_servo_pose_with_timeout(session.arm, target)
        actual = current_pose(session.arm, 2)
        last_position_error_m = position_error_m(target, actual)
        last_orientation_error_rad = orientation_error_rad(target, actual)
        settled = (
            last_position_error_m <= position_tolerance_m
            and last_orientation_error_rad <= orientation_tolerance_rad
        )
        stable_samples = stable_samples + 1 if settled else 0
        if stable_samples >= stable_samples_required:
            session.log_event(
                "servo_endpoint_physically_confirmed", label=label,
                position_error_mm=round(last_position_error_m * 1000.0, 3),
                orientation_error_deg=round(math.degrees(last_orientation_error_rad), 3),
                stable_samples=stable_samples,
            )
            return
        time.sleep(0.02)
    # Imported lazily to avoid a module-import cycle: motion primitives
    # use the ServoPose send helper defined above.
    from motion_primitives import TargetNotReached

    raise TargetNotReached(
        f"{label} 未在 {timeout_s:.1f}s 内物理到位："
        f"位置误差={last_position_error_m * 1000.0:.1f}mm，"
        f"姿态误差={math.degrees(last_orientation_error_rad):.2f}°。"
    )


def move_continuous_with_recovery(
    session: PlacementSession, sdk: object, move_linear: object,
    label: str,
    start_rpy: list[float],
    target_rpy: list[float],
    *,
    linear_speed_mps: float,
    linear_accel_mps2: float,
    initial_speed_mps: float,
    force_monitor: object | None = None,
    push_diagnostic: object | None = None,
    near_target_mm: float = 1.0,
    recovery_speed_mps: float | None = None,
    decelerate_to_stop: bool = False,
    confirm_physical_endpoint: bool = False,
) -> float:
    """Finish one absolute world-frame leg across transient 5600 disconnects.

    A resumed stream always begins from the newly measured TCP pose, never
    from the last command waypoint.  This prevents a stale command from
    producing a jump after the controller reconnects.
    """
    assert session.arm is not None
    segment_start = list(start_rpy)
    speed0 = initial_speed_mps
    full_distance_m = position_error_m(
        command_for(target_rpy), command_for(start_rpy),
    )
    for recovery_index in range(MAX_SERVO_RECOVERY_ATTEMPTS + 1):
        try:
            remaining_before_m = position_error_m(
                command_for(target_rpy), command_for(segment_start),
            )
            completed_fraction = (
                min(1.0, max(0.0, 1.0 - remaining_before_m / full_distance_m))
                if full_distance_m > 1e-9 else 1.0
            )
            segment_force_monitor = None
            if force_monitor is not None:
                segment_force_monitor = lambda alpha: force_monitor(  # type: ignore[operator]
                    completed_fraction + alpha * (1.0 - completed_fraction)
                )
            final_speed = move_linear(
                session.arm, segment_start, target_rpy,
                linear_speed_mps=(
                    recovery_speed_mps
                    if recovery_index and recovery_speed_mps is not None
                    else linear_speed_mps
                ),
                linear_accel_mps2=linear_accel_mps2,
                initial_speed_mps=speed0,
                force_monitor=segment_force_monitor,
                push_diagnostic=push_diagnostic,
                decelerate_to_stop=decelerate_to_stop,
            )
            if confirm_physical_endpoint:
                hold_servo_target_until_settled(session, label, target_rpy)
            return final_speed
        except Exception as exc:
            if (
                not is_servo_connection_loss(session, exc)
                or recovery_index >= MAX_SERVO_RECOVERY_ATTEMPTS
            ):
                raise
            session.log_event(
                "servo_disconnect", label=label,
                recovery_index=recovery_index + 1,
                exception_type=type(exc).__name__, exception=str(exc),
                last_command_start_xyz_rpy=segment_start,
                target_xyz_rpy=target_rpy,
            )
            print(
                f"{label} ServoPose 连接短暂断开；读取实际位姿后恢复剩余路径 "
                f"({recovery_index + 1}/{MAX_SERVO_RECOVERY_ATTEMPTS})。",
                flush=True,
            )
            reconnect_servo_arm(session, sdk, label, recovery_index + 1)
            actual = current_pose(session.arm, 2)
            actual_rpy = actual[:3] + rotvec_to_rpy(actual[3:])
            remaining_mm = position_error_m(command_for(target_rpy), actual) * 1000.0
            session.log_event(
                "servo_recovery_pose", label=label,
                recovery_index=recovery_index + 1,
                actual_xyz_rpy=actual_rpy, target_xyz_rpy=target_rpy,
                remaining_mm=round(remaining_mm, 3),
            )
            session.arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                input_frequency_hz=50.0,
            )
            session.servo_started = True
            session.continuous_servo_stopped = False
            if remaining_mm <= near_target_mm:
                session.log_event(
                    "servo_resume_skipped_near_target", label=label,
                    recovery_index=recovery_index + 1,
                    remaining_mm=round(remaining_mm, 3),
                    threshold_mm=near_target_mm,
                )
                if confirm_physical_endpoint:
                    hold_servo_target_until_settled(session, label, target_rpy)
                return 0.0
            segment_start = actual_rpy
            speed0 = 0.0
            session.log_event(
                "servo_resume", label=label, recovery_index=recovery_index + 1,
                restart_xyz_rpy=segment_start, target_xyz_rpy=target_rpy,
                remaining_mm=round(remaining_mm, 3),
                speed_mm_s=round((recovery_speed_mps or linear_speed_mps) * 1000.0, 3),
            )
    raise AssertionError("unreachable")
