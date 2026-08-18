"""运行一个模块并把异构 JSON/退出码规范化为 ModuleResult。"""
from __future__ import annotations

from pathlib import Path
import json
import os
import signal
import subprocess
import threading
import time
from contracts import ModuleResult, Status
from locks import acquire
from module_registry import ModuleSpec


def _normalize(name: str, path: Path, returncode: int, run_id: str) -> ModuleResult:
    raw = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    if name == "module_4_buffer_state":
        if returncode not in {0, 20, 21}:
            return ModuleResult(
                name, Status.FAULT, "FAILED", run_id, outputs=raw,
                source_result=str(path),
                error={"exit_code": returncode, "reason": "Module 4 非判定性退出"},
            )
        decision = raw.get("decision")
        if decision not in {"FRONT", "BACK", "UNCERTAIN"}:
            return ModuleResult(
                name, Status.FAULT, "INVALID_RESULT", run_id, outputs=raw,
                source_result=str(path),
                error={"exit_code": returncode, "reason": "Module 4 未写出有效 decision"},
            )
        expected_returncode = {"FRONT": 0, "BACK": 20, "UNCERTAIN": 21}[decision]
        if returncode != expected_returncode:
            return ModuleResult(
                name, Status.FAULT, "RESULT_EXIT_MISMATCH", run_id, outputs=raw,
                source_result=str(path),
                error={"exit_code": returncode, "expected_exit_code": expected_returncode},
            )
        status = Status.SUCCESS if decision in {"FRONT", "BACK"} else Status.RETRY
        return ModuleResult(name, status, decision, run_id, outputs=raw, source_result=str(path))
    # Module 3 松爪位规划失败时会安全回 graspup，但工件仍可能在左夹爪中；
    # 这是需要人工确认的放置失败，不能发布 Buffer 已就绪。
    if name in {"module_3_buffer", "module_3_middle"}:
        outcome = str(raw.get("outcome", "SUCCESS"))
        if outcome == "release_motion_failed_object_retained":
            return ModuleResult(
                name,
                Status.OPERATOR_REQUIRED,
                outcome,
                run_id,
                outputs=raw,
                source_result=str(path),
            )
    if returncode == 0:
        return ModuleResult(name, Status.SUCCESS, str(raw.get("outcome", "SUCCESS")), run_id, outputs=raw, source_result=str(path))
    if returncode == 20 and name in {"module_5_front", "module_5_reverse"}:
        return ModuleResult(name, Status.RETRY, "NO_GRASP", run_id, outputs=raw, source_result=str(path))
    if returncode == 21:
        return ModuleResult(name, Status.OPERATOR_REQUIRED, "OPERATOR_REQUIRED", run_id, outputs=raw, source_result=str(path))
    return ModuleResult(name, Status.FAULT, "FAILED", run_id, outputs=raw, source_result=str(path), error={"exit_code": returncode, "recoverable": False})


def _terminate_process_group(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(
    spec: ModuleSpec,
    context: dict[str, str],
    *,
    live: bool,
    runtime: Path,
    run_id: str,
    stop_event: threading.Event | None = None,
    attempt: str = "1",
) -> ModuleResult:
    result_path = Path(context["result"])
    command = spec.command(context, live)
    stdout_path = runtime / "logs" / f"{spec.name}_{attempt}.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    with acquire(runtime / "locks", spec.resources), stdout_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            if stop_event is not None and stop_event.wait(0.2):
                _terminate_process_group(process)
                result = ModuleResult(
                    spec.name, Status.CANCELLED, "STOP_REQUESTED", run_id,
                    source_result=str(result_path),
                )
                result.outputs.update({"command": command, "log": str(stdout_path)})
                result.write(runtime / "results" / f"{spec.name}_{attempt}.json")
                return result
            time.sleep(0.01)
        returncode = process.returncode
    normalized_input = result_path if spec.result_file_is_contract else Path("/nonexistent")
    result = _normalize(spec.name, normalized_input, returncode, run_id)
    result.outputs.setdefault("command", command)
    result.outputs.setdefault("log", str(stdout_path))
    result.outputs.setdefault("raw_result_file", str(result_path) if spec.result_file_is_contract else None)
    result.outputs.setdefault("result_origin", "module" if spec.result_file_is_contract else "controller_synthesized")
    normalized_path = runtime / "results" / f"{spec.name}_{attempt}.json"
    if not spec.result_file_is_contract:
        result.source_result = str(normalized_path)
    result.write(normalized_path)
    return result
