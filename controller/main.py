#!/usr/bin/env python3
"""新总控入口：只调度模块、资源锁、状态机、并行和人工恢复。"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import select
import sys
import threading

from contracts import ModuleResult, Status, now_id
from deployment import apply as apply_deployment
from module_registry import ModuleSpec, registry
from module_runner import run
from runtime_state import RuntimeState
from safety import cancel_both_arms
from scheduler_rules import decide


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RightTask:
    """一个已离开左手的工件，由右臂串行消费的任务。"""

    branch: str  # FRONT: M4.2→M7→M8.1→M9→M10; BACK: M9→M10
    start_index: int = 0
    launch_id: int = 0

    @property
    def modules(self) -> tuple[str, ...]:
        if self.branch == "FRONT":
            return ("module_8_right_front", "module_9_right_place", "module_10_right_idle")
        return ("module_9_right_place", "module_10_right_idle")


@dataclass(frozen=True)
class RightTaskFailure:
    task: RightTask
    index: int
    result: ModuleResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("xiangxin", "gzhu"), required=True)
    parser.add_argument("--workpiece", choices=("1", "2"), required=True)
    parser.add_argument("--live", action="store_true", help="允许模块传入真实运动授权参数")
    parser.add_argument("--dry-run", action="store_true", help="只演练新总控状态机，不启动任何模块")
    parser.add_argument("--simulate-face", choices=("FRONT", "BACK", "UNCERTAIN"), default="FRONT")
    parser.add_argument(
        "--max-buffer-observation-attempts", type=int, default=4,
        help="每个 Buffer 工件允许的 Module 4 完整观察次数；耗尽后回 Module 1",
    )
    parser.add_argument("--left-cycle-limit", type=int, default=0, help="左臂成功送出工件次数；0 表示持续运行")
    parser.add_argument("--planner-host", default=None, help="省略时由 --mode 决定")
    parser.add_argument("--arm-host", default=None, help="省略时由 --mode 决定")
    parser.add_argument("--gripper-host", default=None, help="省略时由 --mode 决定")
    parser.add_argument("--force-host", default=None, help="省略时由 --mode 决定")
    parser.add_argument("--head-host", default=None, help="省略时由 --mode 决定")
    parser.add_argument("--prepare-speed", type=float, default=0.3)
    return parser.parse_args()


def artifact(runtime: Path, name: str, attempt: str | int) -> Path:
    path = runtime / "artifacts" / f"{name}_{attempt}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def context_for(name: str, state: RuntimeState, args: argparse.Namespace, result: Path) -> dict[str, str]:
    common = {
        "mode": args.mode, "workpiece": args.workpiece, "result": str(result),
        "planner_host": args.planner_host, "arm_host": args.arm_host,
        "gripper_host": args.gripper_host, "force_host": args.force_host,
        "head_host": args.head_host,
        "prepare_speed": str(args.prepare_speed),
    }
    if name in {"module_3_buffer", "module_3_middle"}:
        common["handoff"] = state.artifacts["module_1_left_grasp" if name.endswith("buffer") else "module_5_front"]
    if name in {"module_5_front", "module_5_reverse"}:
        common["buffer_result"] = state.artifacts["module_4_buffer_state"]
    if name == "module_8_handoff":
        pointer = Path(state.artifacts.get("module_5_reverse_handoff", state.artifacts["module_5_reverse"]))
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        common["handoff_result"] = str(payload.get("result") or pointer)
        common["handoff_ready"] = str(result.with_name(f"{result.stem}_ready.json"))
    return common


def simulated(name: str, state: RuntimeState, result_path: Path, args: argparse.Namespace) -> ModuleResult:
    outputs: dict[str, object] = {}
    outcome = "SUCCESS"
    if name == "module_4_buffer_state":
        outcome = args.simulate_face
        outputs = {"decision": outcome, "workpiece_id": args.workpiece}
        status = Status.SUCCESS if outcome in {"FRONT", "BACK"} else Status.RETRY
    else:
        status = Status.SUCCESS
        if name == "module_1_left_grasp":
            outputs = {"weight_check_enabled": True, "weight_check": {"classification": "SINGLE"}, "handoff_pose": "graspuptop", "module": name}
        elif name == "module_5_front":
            outcome = "front_regrasp_handoff"
            outputs = {"module": "module_5_1_front_pick_to_graspuptop", "outcome": outcome, "handoff_pose": "graspuptop", "classification": "FRONT"}
    result = ModuleResult(name, status, outcome, state.run_id, outputs=outputs)
    result.write(result_path)
    result.source_result = str(result_path)
    return result


def wait_for_operator(module: str, state: RuntimeState, stop_event: threading.Event) -> str:
    print(f"\n[人工处理] 模块 {module} 不能安全自动恢复。")
    print(f"当前状态：Buffer={state.buffer}, Middle={state.middle}, 左臂={state.left_arm}, 右臂={state.right_arm}")
    print("人工处理完成后输入 y 并回车重新开始；输入 q 安全退出：", end="", flush=True)
    while not stop_event.is_set():
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not readable:
            continue
        answer = sys.stdin.readline().strip().lower()
        if not answer:
            raise EOFError("人工处理终端输入已关闭")
        if answer in {"y", "yes"}:
            if module in {"module_5_reverse", "module_5_reverse_handoff"} and state.left_arm == "HOLDS_WORKPIECE":
                # 反抓已确认完成时，仅允许恢复交接准备；绝不回 M4 重抓。
                return "module_5_reverse_handoff"
            # Buffer 的判断/重抓问题永远从观察位开始，不能直接复用旧 TF。
            if module in {"module_4_buffer_state", "module_5_front", "module_5_reverse"}:
                return "controller_observation_pose"
            if module == "module_8_handoff":
                # M8.2 异常后 handoff 状态会标为 UNKNOWN。操作者输入 y 表示
                # 已把工件与双臂恢复为「左手在交接准备位、右手可接取」的现场，
                # 才允许用全新的 M8.2 模块调用重新开始。
                state.handoff = "LEFT_READY"
                state.left_arm = "AT_HANDOFF"
                state.right_arm = "IDLE"
                return module
            return module
        if answer in {"q", "quit"}:
            raise KeyboardInterrupt
    raise KeyboardInterrupt


def run_one(
    name: str, *, state: RuntimeState, runtime: Path, args: argparse.Namespace,
    specs: dict[str, ModuleSpec], stop_event: threading.Event,
) -> ModuleResult:
    attempt = state.attempts.get(name, 0) + 1
    state.attempts[name] = attempt
    result_path = artifact(runtime, name, attempt)
    state.last_module = name
    state.save(runtime / "state.json")
    if args.dry_run:
        result = simulated(name, state, result_path, args)
    else:
        result = run(
            specs[name], context_for(name, state, args, result_path), live=True,
            runtime=runtime, run_id=state.run_id, stop_event=stop_event,
            attempt=str(attempt),
        )
    if result.source_result:
        state.artifacts[name] = result.source_result
    state.save(runtime / "state.json")
    print(f"[controller] {name}: {result.status}/{result.outcome}", flush=True)
    if result.status == Status.CANCELLED:
        raise KeyboardInterrupt
    return result


def run_right_task(
    task: RightTask, *, state: RuntimeState, runtime: Path, args: argparse.Namespace,
    specs: dict[str, ModuleSpec], stop_event: threading.Event,
) -> ModuleResult | RightTaskFailure:
    """右侧独立串联；它与下一件的左侧链并行，但只允许一个任务占用 Middle。"""
    # 右侧不修改左侧 worker 的 RuntimeState；共享的 Buffer/Middle 状态只由
    # main 线程提交，避免并发保存 state.json 时覆盖彼此的状态转换。
    right_state = RuntimeState(run_id=state.run_id, workpiece=state.workpiece)
    for index in range(task.start_index, len(task.modules)):
        name = task.modules[index]
        result_path = artifact(runtime, name, f"right_{task.launch_id}_{index}")
        if args.dry_run:
            result = simulated(name, right_state, result_path, args)
        else:
            result = run(
                specs[name], context_for(name, right_state, args, result_path), live=True,
                runtime=runtime, run_id=right_state.run_id, stop_event=stop_event,
                attempt=f"right_{task.launch_id}_{index}",
            )
        print(f"[controller/right] {name}: {result.status}/{result.outcome}", flush=True)
        if result.status == Status.CANCELLED:
            return RightTaskFailure(task, index, result)
        if result.status != Status.SUCCESS:
            return RightTaskFailure(task, index, result)
    return result


def main() -> int:
    args = parse_args()
    apply_deployment(args)
    if args.live == args.dry_run:
        raise SystemExit("必须且只能选择 --live 或 --dry-run")
    if args.max_buffer_observation_attempts < 1 or args.left_cycle_limit < 0:
        raise SystemExit("--max-buffer-observation-attempts 必须至少为 1，--left-cycle-limit 必须不小于 0")
    if not 0.0 < args.prepare_speed <= 1.0:
        raise SystemExit("--prepare-speed 必须在 (0, 1] 内")

    run_id = now_id()
    runtime = HERE / "runtime" / run_id
    state = RuntimeState(run_id=run_id, workpiece=args.workpiece)
    specs = registry()
    stop_event = threading.Event()
    def request_stop(_sig: int, _frame: object) -> None:
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sighup = signal.signal(signal.SIGHUP, request_stop)
    # dry-run 不传 limit 时只演练一件，避免命令无止境输出；真实运行默认持续。
    cycle_limit = 1 if args.dry_run and args.left_cycle_limit == 0 else args.left_cycle_limit
    next_module = "controller_prepare"
    dispatched = 0
    right_future: Future[ModuleResult | RightTaskFailure] | None = None
    pending_right_failure: RightTaskFailure | None = None
    right_launches = 0
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="right-consumer")

    def collect_right(*, wait: bool) -> RightTaskFailure | None:
        nonlocal right_future, pending_right_failure
        if pending_right_failure is not None:
            return pending_right_failure
        if right_future is None or (not wait and not right_future.done()):
            return None
        outcome = right_future.result()
        right_future = None
        if isinstance(outcome, RightTaskFailure):
            if outcome.result.status == Status.CANCELLED:
                raise KeyboardInterrupt
            state.right_arm = "FAULT"
            state.save(runtime / "state.json")
            pending_right_failure = outcome
            return pending_right_failure
        state.right_arm, state.middle = "IDLE", "EMPTY"
        state.save(runtime / "state.json")
        print("[controller] 右臂已消费上一件工件并回 Idle。", flush=True)
        return None

    def start_right(task: RightTask) -> None:
        nonlocal right_future, right_launches
        if right_future is not None:
            raise RuntimeError("Middle 尚有右臂消费任务，禁止发布下一件")
        right_launches += 1
        task = RightTask(task.branch, task.start_index, right_launches)
        state.right_arm = "BUSY"
        state.save(runtime / "state.json")
        right_future = executor.submit(
            run_right_task, task, state=state, runtime=runtime, args=args,
            specs=specs, stop_event=stop_event,
        )
        print(f"[controller] 已把 {task.branch} 工件交给右臂后台消费；左臂可开始下一件。", flush=True)

    def recover_pending_right() -> None:
        """右臂失败必须被成功重跑，绝不能只打印后丢失。"""
        nonlocal pending_right_failure
        if pending_right_failure is None:
            return
        failure = pending_right_failure
        pending_right_failure = None
        wait_for_operator(failure.result.module, state, stop_event)
        start_right(RightTask(failure.task.branch, failure.index))

    def wait_right_clear() -> None:
        """在占用 middle/交接区或退出前，确保右臂任务已真正成功完成。"""
        while right_future is not None or pending_right_failure is not None:
            failure = collect_right(wait=True)
            if failure is not None:
                recover_pending_right()

    try:
        print(f"[controller] run_id={run_id}; mode={args.mode}; workpiece={args.workpiece}; {'演练' if args.dry_run else '真实执行'}")
        while next_module:
            # 右臂异常不会中断正在运行的左侧模块；左模块结束后再一起进入人工恢复。
            if collect_right(wait=False) is not None:
                recover_pending_right()

            # Middle/交接区只有一个物理工位。左臂在发布下一件之前等待右臂
            # 消费完成；在此之前 M1~M5 可与右臂并行。
            if next_module in {"module_3_middle", "module_8_handoff"}:
                wait_right_clear()

            # 交接区不使用文件锁；由单一总控的持久化状态机互斥。M8.2 只可
            # 消费 M5.2 已明确发布的 LEFT_READY，启动前先落盘 TRANSFERRING，
            # 便于停止或异常后准确呈现现场，而不会把交接工件误记为 Buffer 工件。
            if next_module == "module_8_handoff":
                if state.handoff != "LEFT_READY":
                    print(
                        "[controller] Module 8.2 未启动：交接状态必须为 LEFT_READY，"
                        f"当前为 {state.handoff}。",
                        flush=True,
                    )
                    next_module = wait_for_operator("module_8_handoff", state, stop_event)
                    continue
                state.handoff = "TRANSFERRING"
                state.save(runtime / "state.json")

            result = run_one(next_module, state=state, runtime=runtime, args=args, specs=specs, stop_event=stop_event)
            decision = decide(
                state, result,
                max_buffer_observation_attempts=args.max_buffer_observation_attempts,
            )
            # 规则层会更新 Buffer/交接位/持件状态；尤其是反抓成功后交接失败时，
            # 必须在进入人工等待前落盘，不能让重启后误以为左手空闲。
            state.save(runtime / "state.json")
            print(f"[controller] {next_module} -> {decision.next_module or 'STOP'}", flush=True)

            # 两条分支在这里把工件从左侧交给右侧；右侧开始后台工作，左臂立刻
            # 可以从 Module 1 准备下一件。Buffer 单容量由 RuntimeState 和
            # scheduler_rules 的状态转换保护，不使用短生命周期的文件锁。
            if result.status == Status.SUCCESS and next_module == "module_3_middle":
                start_right(RightTask("FRONT"))
                dispatched += 1
                next_module = None if cycle_limit and dispatched >= cycle_limit else "module_1_left_grasp"
                continue
            if result.status == Status.SUCCESS and next_module == "module_8_handoff":
                state.buffer = "EMPTY"
                start_right(RightTask("BACK"))
                dispatched += 1
                next_module = None if cycle_limit and dispatched >= cycle_limit else "module_1_left_grasp"
                continue

            if decision.operator_required:
                # 人工输入之前，允许已经启动的右臂任务完整结束并回 Idle。
                wait_right_clear()
                next_module = wait_for_operator(next_module, state, stop_event)
            elif decision.finished:
                next_module = None
            else:
                next_module = decision.next_module

        wait_right_clear()
        print("[controller] 已完成本次限定循环，双臂均已回到可开始下一轮的状态。")
        return 0
    except (KeyboardInterrupt, EOFError):
        stop_event.set()
        cancel_both_arms(args.arm_host, "总控收到停止请求")
        print("\n[controller] 已安全退出。", file=sys.stderr)
        return 130
    finally:
        stop_event.set()
        executor.shutdown(wait=True, cancel_futures=False)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGHUP, previous_sighup)


if __name__ == "__main__":
    raise SystemExit(main())
