"""纯规则：模块结果和状态决定下一个模块，不执行任何机器人代码。"""
from __future__ import annotations

from dataclasses import dataclass
from contracts import ModuleResult, Status
from runtime_state import RuntimeState


@dataclass(frozen=True)
class Decision:
    next_module: str | None
    operator_required: bool = False
    finished: bool = False


def decide(state: RuntimeState, result: ModuleResult, *, max_buffer_observation_attempts: int) -> Decision:
    if result.status in {Status.FAULT, Status.OPERATOR_REQUIRED, Status.CANCELLED}:
        return Decision(None, operator_required=result.status != Status.CANCELLED)
    if result.status == Status.RETRY:
        if result.module in {"module_5_front", "module_5_reverse"}:
            count = state.retries.get(result.module, 0) + 1
            state.retries[result.module] = count
            # 每次 Buffer 重抓均必须从 Module 4.1 重新观察并取得新 TF。
            return Decision("controller_observation_pose" if count < 5 else None, operator_required=count >= 5)
        if result.module == "module_4_buffer_state":
            # 0 个工件、多个工件、正反/ICP 不可靠统一是一次「Buffer 观察失败」。
            # 达上限后不再尝试无 TF 的 Module 5，而是认定本次 Buffer 工件已无效，
            # 清空 Buffer 并从 Module 1 重新抓取下一件。
            count = state.retries.get("buffer_observation", 0) + 1
            state.retries["buffer_observation"] = count
            if count < max_buffer_observation_attempts:
                return Decision("controller_observation_pose")
            state.buffer = "EMPTY"
            state.left_arm = "IDLE"
            state.orientation = None
            state.buffer_reset_reason = (
                f"Module 4 连续 {count} 次观察未得到可信工件/ICP，"
                "按掉件或无效 Buffer 工件处理"
            )
            state.artifacts.pop("module_4_buffer_state", None)
            return Decision("module_1_left_grasp")
        return Decision(result.module)
    name, outcome = result.module, result.outcome
    if name == "controller_prepare" and result.status == Status.SUCCESS:
        state.left_arm, state.right_arm = "IDLE", "IDLE"
        return Decision("module_1_left_grasp")
    if name == "module_1_left_grasp" and result.status == Status.SUCCESS:
        # 新工件进入 Buffer 生命周期；两个重试预算均从零开始，绝不继承上一件。
        state.retries.clear()
        state.buffer_reset_reason = None
        state.buffer, state.left_arm = "LEFT_PLACING", "HOLDS_WORKPIECE"
        return Decision("module_3_buffer")
    if name == "module_3_buffer" and result.status == Status.SUCCESS:
        state.buffer, state.left_arm = "OCCUPIED", "IDLE"
        return Decision("controller_observation_pose")
    if name == "controller_observation_pose" and result.status == Status.SUCCESS:
        return Decision("module_4_buffer_state")
    if name == "module_4_buffer_state":
        if outcome == "FRONT":
            state.orientation, state.buffer = "FRONT", "LEFT_REGRASPING"
            return Decision("module_5_front")
        if outcome == "BACK":
            state.orientation, state.buffer = "BACK", "LEFT_REGRASPING"
            return Decision("module_5_reverse")
        # UNCERTAIN 已在上方 Status.RETRY 分支处理；这里仅保留防御性兜底。
        return Decision("controller_observation_pose")
    if name == "module_5_front" and result.status == Status.SUCCESS:
        state.left_arm = "HOLDS_WORKPIECE"
        return Decision("module_3_middle")
    if name == "module_3_middle" and result.status == Status.SUCCESS:
        state.buffer, state.middle, state.left_arm = "EMPTY", "OCCUPIED", "IDLE"
        return Decision("module_8_right_front")
    if name == "module_8_right_front" and result.status == Status.SUCCESS:
        state.right_arm = "HOLDS_WORKPIECE"
        return Decision("module_9_right_place")
    if name == "module_5_reverse" and result.status == Status.SUCCESS:
        state.buffer, state.left_arm = "HANDOFF_READY", "AT_HANDOFF"
        return Decision("module_8_handoff")
    if name == "module_8_handoff" and result.status == Status.SUCCESS:
        state.buffer, state.left_arm, state.right_arm = "EMPTY", "IDLE", "HOLDS_WORKPIECE"
        return Decision("module_9_right_place")
    if name == "module_9_right_place" and result.status == Status.SUCCESS:
        return Decision("module_10_right_idle")
    if name == "module_10_right_idle" and result.status == Status.SUCCESS:
        state.buffer, state.right_arm = "EMPTY", "IDLE"
        return Decision(None, finished=True)
    return Decision(None, operator_required=True)
