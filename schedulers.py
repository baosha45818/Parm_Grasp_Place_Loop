"""早期线程调度骨架；当前实机请使用 tests/loop/simple_dual_arm_loop.py。"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from loop_config import LoopConfig
from loop_types import (
    BufferState,
    LeftArmState,
    ModuleResult,
    ResultStatus,
    RightArmState,
    SystemState,
)
from modules.module_6_left_idle.implementation import module_6_left_idle
from modules.module_7_right_align.implementation import module_7_manual_align
from modules.module_8_right_grasp.module_8_1_buffer_grasp.implementation import (
    module_8_1_buffer_grasp,
)
from modules.module_9_right_place.implementation import module_9_right_place
from modules.module_10_right_idle.implementation import module_10_right_idle
from tool.planner_runtime import PlannerRuntime


def module_2_inspect(*_args, **_kwargs):
    """早期骨架不再维护旧 Module 2；实机入口使用 modules/modular_dual_arm_loop.py。"""
    raise RuntimeError("旧 schedulers.py 不支持 Module 2；请使用模块化总控入口")


@dataclass
class RuntimeContext:
    config: LoopConfig
    planner: PlannerRuntime
    max_cycles: int
    left: LeftArmState = field(default_factory=LeftArmState)
    right: RightArmState = field(default_factory=RightArmState)
    buffer: BufferState = field(default_factory=BufferState)
    system: SystemState = field(default_factory=SystemState)
    left_results: queue.Queue[ModuleResult] = field(
        default_factory=lambda: queue.Queue(maxsize=1)
    )
    right_results: queue.Queue[ModuleResult] = field(
        default_factory=lambda: queue.Queue(maxsize=2)
    )
    stop_event: threading.Event = field(default_factory=threading.Event)


def route_left_result(ctx: RuntimeContext, item: ModuleResult) -> None:
    ctx.left_results.put(item)


def route_right_result(ctx: RuntimeContext, item: ModuleResult) -> None:
    ctx.right_results.put(item)
    if item.previous_module == "8-2":
        ctx.left_results.put(item)


def consume_own_result(
    result_queue: queue.Queue[ModuleResult], expected_module: str
) -> ModuleResult:
    deferred: list[ModuleResult] = []
    while True:
        item = result_queue.get()
        if item.previous_module == expected_module:
            for pending in deferred:
                result_queue.put(pending)
            return item
        deferred.append(item)


def reserve_buffer(ctx: RuntimeContext, operator: str) -> bool:
    with ctx.buffer.lock:
        if ctx.buffer.operator != "NONE":
            return False
        ctx.buffer.operator = operator
        return True


def left_scheduler(ctx: RuntimeContext) -> None:
    """拒绝运行已废弃的左臂骨架。

    真实 Module 1-1 必须经由模块化 shell 入口运行，不能在此伪造
    ``holding=SINGLE`` 后进入后续模块。
    """
    try:
        raise RuntimeError(
            "schedulers.py 的左臂骨架已废弃，未接入真实 GraspNet；"
            "实机请运行 tests/loop/simple_dual_arm_loop.py 或 "
            "modules/modular_dual_arm_loop.py"
        )
    except Exception as exc:
        ctx.left.last_error = str(exc)
        ctx.system.last_fault = f"left scheduler: {exc}"
        ctx.system.run_mode = "FAULT"
        ctx.stop_event.set()


def right_scheduler(ctx: RuntimeContext) -> None:
    """运行编号已更新的右臂占位骨架；Module 8-2 尚未模块化。"""
    completed_cycles = 0
    try:
        route_right_result(ctx, module_10_right_idle(ctx.right, ctx.planner))
        consume_own_result(ctx.right_results, "10")
        while not ctx.stop_event.is_set() and completed_cycles < ctx.max_cycles:
            with ctx.buffer.lock:
                full = ctx.buffer.full
                needs_alignment = ctx.buffer.needs_alignment
                available = ctx.buffer.operator == "NONE"
            if full and available and reserve_buffer(ctx, "RIGHT"):
                if needs_alignment:
                    route_right_result(
                        ctx, module_7_manual_align(ctx.buffer, ctx.right)
                    )
                    consume_own_result(ctx.right_results, "7")
                    if not reserve_buffer(ctx, "RIGHT"):
                        raise RuntimeError("Module 7 后无法重新取得 Buffer")
                route_right_result(
                    ctx,
                    module_8_1_buffer_grasp(ctx.buffer, ctx.right, ctx.planner),
                )
                consume_own_result(ctx.right_results, "8-1")
                item = module_9_right_place(ctx.right, ctx.planner)
                route_right_result(ctx, item)
                consume_own_result(ctx.right_results, "9")
                if item.status is not ResultStatus.SUCCESS:
                    raise RuntimeError(item.message)
                route_right_result(ctx, module_10_right_idle(ctx.right, ctx.planner))
                consume_own_result(ctx.right_results, "10")
                completed_cycles += 1
                continue
            time.sleep(ctx.config.idle_poll_seconds)
    except Exception as exc:
        ctx.right.last_error = str(exc)
        ctx.system.last_fault = f"right scheduler: {exc}"
        ctx.system.run_mode = "FAULT"
        ctx.stop_event.set()
