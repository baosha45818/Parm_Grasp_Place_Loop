"""跨进程记录模块执行结果、耗时、任务成功率和异常恢复动作。"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
from time import time
from typing import Any

from loop_types import ModuleResult, ResultStatus


METRICS_DIR_ENV = "PARM_MODULE_METRICS_DIR"
STATUS_FILE_ENV = "PARM_MODULE_STATUS_FILE"
RUN_ID_ENV = "PARM_LOOP_RUN_ID"
REPORT_SCOPE_ENV = "PARM_MODULE_REPORT_SCOPE"
LEFT_PIPELINE = "module_1_to_3_left_pipeline"
RIGHT_FINISH = "module_10_right_idle"
SPECIAL_TASK_OUTCOMES = {
    "INSUFFICIENT_POINT_CLOUD_DISCARDED": {
        "reason_code": "INSUFFICIENT_POINT_CLOUD",
        "target_module": "module_1",
        "action": "左臂已回 graspup 并松爪；清空本轮状态，从 Module 1 开始新任务",
    },
    "HELD_OBJECT_DROPPED": {
        "reason_code": "HELD_OBJECT_DROPPED",
        "target_module": "module_1",
        "action": "清空本轮工件/缓冲区状态，从 Module 1 重新抓取",
    },
    "PLANNING_TIMEOUT_RECOVERED": {
        "reason_code": "PLANNING_TIMEOUT",
        "target_module": "module_1",
        "action": "规划超时恢复已回 graspup 并松爪；从 Module 1 开始新任务",
    },
}


def _normalize_module_name(module: str, message: str) -> str:
    """把旧调度器按脚本文件名生成的标签转换成业务模块名。"""
    if module == "main" and "Module 5-2" in message:
        return "module_5_2_handoff_prepare"
    return {
        "run_head_judge_then_pick": "module_4_to_5_reverse_pipeline",
        "workpiece_pipeline": "module_4_7_8_1_right_pipeline",
        "place_hole_a_wiggle": "module_9_right_place",
    }.get(module, module)


def _utc_text(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _empty_state() -> dict[str, Any]:
    return {
        "next_event_id": 1,
        "next_attempt_id": 1,
        "active_modules": {},
        "tasks": {
            "next_id": 1,
            "records": {},
            "current_by_left_pid": {},
            "ready": [],
        },
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": os.environ.get(RUN_ID_ENV, ""),
        "started_at": _utc_text(time()),
        "updated_at": _utc_text(time()),
        "event_count": 0,
        "modules": {},
        "task_execution": {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "recovered_failures": 0,
            "aborted": 0,
            "in_progress": 0,
            "success_rate": 0.0,
            "success_rate_percent": 0.0,
            "average_success_duration_seconds": 0.0,
            "total_success_duration_seconds": 0.0,
        },
    }


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return fallback
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return value if isinstance(value, dict) else fallback


def _module_recovery(
    status: ResultStatus,
    reason_code: str,
    message: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """把现有状态码转成结构化异常及回退策略。"""
    combined = f"{reason_code} {message}"
    if reason_code == "NO_GRASP_PREDICTED":
        return (
            {
                "type": "recoverable_perception_result",
                "reason_code": reason_code,
                "message": message,
            },
            {
                "level": "PARTIAL_ROLLBACK",
                "target_module": "module_1",
                "action": "保持机械臂安全位，不发送额外回程运动，重新执行抓取预测",
            },
        )
    if reason_code == "HELD_OBJECT_DROPPED":
        return (
            {
                "type": "workpiece_state_error",
                "reason_code": reason_code,
                "message": message,
            },
            {
                "level": "TASK_RESET",
                "target_module": "module_1",
                "action": "清空缓冲区和本轮工件状态，从 Module 1 重新开始",
            },
        )
    if reason_code == "NO_UNIQUE_REVERSE_GRASP":
        return (
            {
                "type": "recoverable_perception_result",
                "reason_code": reason_code,
                "message": message,
            },
            {
                "level": "PARTIAL_ROLLBACK",
                "target_module": "module_4_1",
                "action": "左臂保持安全位，重新采集头部 RGB-D 并执行反抓预测",
            },
        )
    if status == ResultStatus.RETRY and (
        reason_code == "EXIT_20" or "回退到 Module 5-1" in combined
    ):
        return (
            {
                "type": "recoverable_no_grasp",
                "reason_code": reason_code,
                "message": message,
            },
            {
                "level": "PARTIAL_ROLLBACK",
                "target_module": "module_5_1",
                "action": "保留前序检测和放置结果，在反抓安全位重新执行 Module 5-1",
            },
        )
    if status in {ResultStatus.FAULT, ResultStatus.ABORTED}:
        return (
            {
                "type": "module_failure",
                "reason_code": reason_code or status.value,
                "message": message,
            },
            {
                "level": "SAFE_STOP",
                "target_module": "initial_state",
                "action": (
                    "停止两个 worker 并 cancel_task；保留现场，"
                    "排除故障后由下一次启动准备流程复位到初始位"
                ),
            },
        )
    return None, None


def _module_stats(
    summary: dict[str, Any],
    module: str,
) -> dict[str, Any]:
    modules = summary.setdefault("modules", {})
    return modules.setdefault(
        module,
        {
            "attempts": 0,
            "successes": 0,
            "retries": 0,
            "faults": 0,
            "aborted": 0,
            "in_progress": 0,
            "success_rate": 0.0,
            "success_rate_percent": 0.0,
            "total_duration_seconds": 0.0,
            "average_duration_seconds": 0.0,
            "last_duration_seconds": None,
            "last_status": "",
        },
    )


def _close_task(
    state: dict[str, Any],
    summary: dict[str, Any],
    task_id: str,
    *,
    outcome: str,
    now: float,
    reason_code: str,
    recovery: dict[str, Any] | None,
) -> dict[str, Any] | None:
    tasks_state = state["tasks"]
    record = tasks_state["records"].pop(task_id, None)
    if record is None:
        return None
    current = tasks_state["current_by_left_pid"]
    for pid, current_id in tuple(current.items()):
        if current_id == task_id:
            current.pop(pid, None)
    tasks_state["ready"] = [
        ready_id for ready_id in tasks_state["ready"] if ready_id != task_id
    ]

    task_summary = summary["task_execution"]
    duration = max(0.0, now - float(record["started_at"]))
    if outcome == "SUCCESS":
        task_summary["successes"] += 1
        task_summary["total_success_duration_seconds"] += duration
        task_summary["average_success_duration_seconds"] = (
            task_summary["total_success_duration_seconds"]
            / task_summary["successes"]
        )
    elif outcome == "ABORTED":
        task_summary["aborted"] += 1
    else:
        task_summary["failures"] += 1
        if recovery is not None:
            task_summary["recovered_failures"] += 1
    return {
        "task_id": task_id,
        "outcome": outcome,
        "reason_code": reason_code,
        "duration_seconds": duration,
        "started_at": _utc_text(float(record["started_at"])),
        "finished_at": _utc_text(now),
        "recovery": recovery,
    }


def _update_task_metrics(
    state: dict[str, Any],
    summary: dict[str, Any],
    *,
    module: str,
    status: ResultStatus,
    message: str,
    reason_code: str,
    pid: int,
    now: float,
    recovery: dict[str, Any] | None,
) -> dict[str, Any] | None:
    tasks_state = state["tasks"]
    task_summary = summary["task_execution"]
    pid_key = str(pid)

    if module == LEFT_PIPELINE and status == ResultStatus.WAIT:
        number = int(tasks_state["next_id"])
        tasks_state["next_id"] = number + 1
        run_id = os.environ.get(RUN_ID_ENV, "run")
        task_id = f"{run_id}:{number:04d}"
        tasks_state["records"][task_id] = {
            "started_at": now,
            "left_pid": pid,
            "state": "LEFT_PIPELINE",
        }
        tasks_state["current_by_left_pid"][pid_key] = task_id
        task_summary["attempts"] += 1
        task_event: dict[str, Any] | None = {
            "task_id": task_id,
            "outcome": "STARTED",
            "started_at": _utc_text(now),
        }
    elif module == LEFT_PIPELINE and status != ResultStatus.WAIT:
        task_id = tasks_state["current_by_left_pid"].get(pid_key)
        task_event = None
        if task_id is not None:
            if status == ResultStatus.SUCCESS:
                special = next(
                    (
                        value
                        for marker, value in SPECIAL_TASK_OUTCOMES.items()
                        if marker in message
                    ),
                    None,
                )
                if special is None:
                    tasks_state["records"][task_id]["state"] = "RIGHT_PENDING"
                    tasks_state["ready"].append(task_id)
                    tasks_state["current_by_left_pid"].pop(pid_key, None)
                    task_event = {
                        "task_id": task_id,
                        "outcome": "LEFT_PIPELINE_COMPLETE",
                        "awaiting": RIGHT_FINISH,
                    }
                else:
                    special_recovery = {
                        "level": "TASK_RESET",
                        "target_module": special["target_module"],
                        "action": special["action"],
                    }
                    task_event = _close_task(
                        state,
                        summary,
                        task_id,
                        outcome="RECOVERED_FAILURE",
                        now=now,
                        reason_code=special["reason_code"],
                        recovery=special_recovery,
                    )
            else:
                task_event = _close_task(
                    state,
                    summary,
                    task_id,
                    outcome=(
                        "ABORTED"
                        if status == ResultStatus.ABORTED
                        else "RECOVERED_FAILURE"
                        if status == ResultStatus.RETRY
                        else "FAILURE"
                    ),
                    now=now,
                    reason_code=reason_code or status.value,
                    recovery=recovery,
                )
    elif module == RIGHT_FINISH and status in {
        ResultStatus.SUCCESS,
        ResultStatus.FAULT,
        ResultStatus.ABORTED,
    }:
        task_event = None
        if tasks_state["ready"]:
            task_id = tasks_state["ready"][0]
            task_event = _close_task(
                state,
                summary,
                task_id,
                outcome=status.value,
                now=now,
                reason_code=reason_code or status.value,
                recovery=recovery,
            )
    elif module == "modular_loop" and status in {
        ResultStatus.SUCCESS,
        ResultStatus.FAULT,
        ResultStatus.ABORTED,
    }:
        closed_tasks = []
        for task_id in tuple(tasks_state["records"]):
            closed = _close_task(
                state,
                summary,
                task_id,
                outcome=(
                    "FAILURE"
                    if status == ResultStatus.FAULT
                    else "ABORTED"
                ),
                now=now,
                reason_code=reason_code or "LOOP_ENDED_WITH_ACTIVE_TASK",
                recovery=recovery,
            )
            if closed is not None:
                closed_tasks.append(closed)
        task_event = (
            {
                "outcome": "ACTIVE_TASKS_CLOSED",
                "tasks": closed_tasks,
            }
            if closed_tasks
            else None
        )
    else:
        task_event = None

    task_summary["in_progress"] = len(tasks_state["records"])
    attempts = int(task_summary["attempts"])
    rate = task_summary["successes"] / attempts if attempts else 0.0
    task_summary["success_rate"] = rate
    task_summary["success_rate_percent"] = rate * 100.0
    return task_event


def _persist_metrics(
    *,
    module: str,
    status: ResultStatus,
    message: str,
    reason_code: str,
    now: float,
    exception: dict[str, Any] | None,
    recovery: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics_text = os.environ.get(METRICS_DIR_ENV)
    if not metrics_text:
        return {
            "attempt_id": 0,
            "started_at": now,
            "duration_seconds": 0.0,
            "task_event": None,
        }

    metrics_dir = Path(metrics_text)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    lock_path = metrics_dir / ".metrics.lock"
    state_path = metrics_dir / ".metrics_state.json"
    summary_path = metrics_dir / "summary.json"
    events_path = metrics_dir / "events.jsonl"
    pid = os.getpid()
    reporter_scope = os.environ.get(REPORT_SCOPE_ENV, str(pid))
    active_key = f"{reporter_scope}:{module}"

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = _read_json(state_path, _empty_state())
        summary = _read_json(summary_path, _empty_summary())
        active_modules = state.setdefault("active_modules", {})
        stats = _module_stats(summary, module)

        if status == ResultStatus.WAIT:
            attempt_id = int(state["next_attempt_id"])
            state["next_attempt_id"] = attempt_id + 1
            started_at = now
            active_modules[active_key] = {
                "attempt_id": attempt_id,
                "started_at": started_at,
                "module": module,
            }
            stats["attempts"] += 1
            stats["in_progress"] += 1
            duration = 0.0
        else:
            active = active_modules.pop(active_key, None)
            if active is None:
                attempt_id = int(state["next_attempt_id"])
                state["next_attempt_id"] = attempt_id + 1
                started_at = now
                stats["attempts"] += 1
            else:
                attempt_id = int(active["attempt_id"])
                started_at = float(active["started_at"])
                stats["in_progress"] = max(0, int(stats["in_progress"]) - 1)
            duration = max(0.0, now - started_at)
            counter = {
                ResultStatus.SUCCESS: "successes",
                ResultStatus.RETRY: "retries",
                ResultStatus.FAULT: "faults",
                ResultStatus.ABORTED: "aborted",
            }.get(status)
            if counter is not None:
                stats[counter] += 1
            stats["total_duration_seconds"] += duration
            terminal_count = (
                stats["successes"]
                + stats["retries"]
                + stats["faults"]
                + stats["aborted"]
            )
            stats["average_duration_seconds"] = (
                stats["total_duration_seconds"] / terminal_count
                if terminal_count
                else 0.0
            )
            stats["last_duration_seconds"] = duration

        stats["last_status"] = status.value
        rate = stats["successes"] / stats["attempts"] if stats["attempts"] else 0.0
        stats["success_rate"] = rate
        stats["success_rate_percent"] = rate * 100.0
        closed_active_modules = []
        if module == "modular_loop" and status in {
            ResultStatus.SUCCESS,
            ResultStatus.FAULT,
            ResultStatus.ABORTED,
        }:
            for stale_key, stale in tuple(active_modules.items()):
                stale_module = str(
                    stale.get("module") or stale_key.rsplit(":", 1)[-1]
                )
                stale_started_at = float(stale["started_at"])
                stale_duration = max(0.0, now - stale_started_at)
                stale_stats = _module_stats(summary, stale_module)
                stale_stats["in_progress"] = max(
                    0, int(stale_stats["in_progress"]) - 1
                )
                stale_stats["aborted"] += 1
                stale_stats["total_duration_seconds"] += stale_duration
                stale_terminal_count = (
                    stale_stats["successes"]
                    + stale_stats["retries"]
                    + stale_stats["faults"]
                    + stale_stats["aborted"]
                )
                stale_stats["average_duration_seconds"] = (
                    stale_stats["total_duration_seconds"]
                    / stale_terminal_count
                )
                stale_stats["last_duration_seconds"] = stale_duration
                stale_stats["last_status"] = ResultStatus.ABORTED.value
                stale_rate = (
                    stale_stats["successes"] / stale_stats["attempts"]
                    if stale_stats["attempts"]
                    else 0.0
                )
                stale_stats["success_rate"] = stale_rate
                stale_stats["success_rate_percent"] = stale_rate * 100.0
                closed_active_modules.append(
                    {
                        "module": stale_module,
                        "attempt_id": int(stale["attempt_id"]),
                        "status": ResultStatus.ABORTED.value,
                        "duration_seconds": stale_duration,
                        "reason_code": "LOOP_ENDED_WITH_ACTIVE_MODULE",
                    }
                )
                active_modules.pop(stale_key, None)
        task_event = _update_task_metrics(
            state,
            summary,
            module=module,
            status=status,
            message=message,
            reason_code=reason_code,
            pid=pid,
            now=now,
            recovery=recovery,
        )

        event_id = int(state["next_event_id"])
        state["next_event_id"] = event_id + 1
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "run_id": os.environ.get(RUN_ID_ENV, ""),
            "timestamp": _utc_text(now),
            "pid": pid,
            "reporter_scope": reporter_scope,
            "module": module,
            "attempt_id": attempt_id,
            "status": status.value,
            "reason_code": reason_code,
            "message": message,
            "started_at": _utc_text(started_at),
            "finished_at": _utc_text(now),
            "duration_seconds": duration,
            "exception": exception,
            "recovery": recovery,
            "task_event": task_event,
            "closed_active_modules": closed_active_modules,
        }
        with events_path.open("a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            events_file.flush()
            os.fsync(events_file.fileno())

        summary["updated_at"] = _utc_text(now)
        summary["event_count"] = event_id
        summary["last_event"] = event
        _atomic_write_json(state_path, state)
        _atomic_write_json(summary_path, summary)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return {
        "attempt_id": attempt_id,
        "started_at": started_at,
        "duration_seconds": duration,
        "task_event": task_event,
        "module_success_rate": stats["success_rate"],
        "task_success_rate": summary["task_execution"]["success_rate"],
    }


def _report(
    module: str,
    status: ResultStatus,
    message: str,
    *,
    reason_code: str,
    exception: dict[str, Any] | None = None,
    recovery: dict[str, Any] | None = None,
) -> ModuleResult:
    module = _normalize_module_name(module, message)
    now = time()
    inferred_exception, inferred_recovery = _module_recovery(
        status, reason_code, message
    )
    exception = exception or inferred_exception
    recovery = recovery or inferred_recovery
    metrics = _persist_metrics(
        module=module,
        status=status,
        message=message,
        reason_code=reason_code,
        now=now,
        exception=exception,
        recovery=recovery,
    )
    result = ModuleResult(
        status=status,
        previous_module=module,
        reason_code=reason_code,
        message=message,
        attempt_id=int(metrics["attempt_id"]),
        started_at=float(metrics["started_at"]),
        finished_at=now,
    )
    duration = float(metrics["duration_seconds"])
    print(
        f"[module-state] module={module} status={status.value} "
        f"reason={reason_code or '-'} elapsed={duration:.3f}s message={message}",
        flush=True,
    )
    if recovery is not None:
        print(
            "[recovery] "
            f"level={recovery.get('level')} "
            f"target={recovery.get('target_module')} "
            f"action={recovery.get('action')}",
            flush=True,
        )
    if status != ResultStatus.WAIT and METRICS_DIR_ENV in os.environ:
        print(
            f"[metrics] module_success_rate="
            f"{float(metrics.get('module_success_rate', 0.0)) * 100.0:.2f}% "
            f"task_success_rate="
            f"{float(metrics.get('task_success_rate', 0.0)) * 100.0:.2f}%",
            flush=True,
        )

    target_text = os.environ.get(STATUS_FILE_ENV)
    if target_text:
        _atomic_write_json(
            Path(target_text),
            {
                "module": result.previous_module,
                "status": result.status.value,
                "reason_code": result.reason_code,
                "message": result.message,
                "attempt_id": result.attempt_id,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "duration_seconds": duration,
                "exception": exception,
                "recovery": recovery,
            },
        )
    return result


def report(
    module: str,
    status: ResultStatus,
    message: str,
    *,
    reason_code: str = "",
) -> ModuleResult:
    """打印状态并持久化事件、耗时、模块成功率和整任务成功率。"""
    return _report(
        module,
        status,
        message,
        reason_code=reason_code,
    )


def report_exception(
    module: str,
    exc: BaseException,
    *,
    action: str,
    target_module: str,
    recovery_level: str,
    reason_code: str = "UNHANDLED_EXCEPTION",
    traceback_text: str = "",
) -> ModuleResult:
    """记录未处理异常及已经采取/明确要求的恢复动作。"""
    return _report(
        module,
        ResultStatus.FAULT,
        f"{type(exc).__name__}: {exc}",
        reason_code=reason_code,
        exception={
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text,
        },
        recovery={
            "level": recovery_level,
            "target_module": target_module,
            "action": action,
        },
    )
