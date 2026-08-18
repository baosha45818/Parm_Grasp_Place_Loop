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
        if (
            result.module in {"module_5_reverse", "module_5_reverse_handoff"}
            and result.outputs.get("left_gripper_holds_object") is True
        ):
            # 反抓已成功、仅交接准备失败：工件不再位于 Buffer，左手仍夹持。
            # 禁止回 M4/M5 重新观察或重抓，否则会碰撞或掉件。
            state.buffer = "EMPTY"
            state.handoff = "EMPTY"
            state.left_arm = "HOLDS_WORKPIECE"
            state.orientation = "BACK"
            return Decision(None, operator_required=True)
        # M8.2 异常时无法仅由退出码确认工件仍在哪只手；保留现场并交人工确认，
        # 禁止把不明交接状态伪装成 Buffer 或空交接位。
        if result.module == "module_8_handoff":
            state.handoff = "UNKNOWN"
        return Decision(None, operator_required=result.status != Status.CANCELLED)
    if result.status == Status.RETRY:
        if (
            result.module in {"module_3_buffer", "module_3_middle"}
            and result.outcome == "planning_failed_recovered_at_graspup"
        ):
            # M3 的失败恢复已经在 graspup 安全丢弃并松开夹爪。当前工件不能
            # 再进入 M4/M5；由 M1 从输入区重新观察、重新抓取。
            # Buffer 放置失败可与右臂消费上一件 Middle 工件并行，绝不能清空
            # 该上一件的 Middle 状态。Middle 放置失败则尚未发布新件，两工位
            # 都应为空。
            state.buffer = "EMPTY"
            if result.module == "module_3_middle":
                state.middle = "EMPTY"
            state.handoff = "EMPTY"
            state.left_arm = "IDLE"
            state.orientation = None
            state.buffer_reset_reason = "Module 3 放置规划失败，已在 graspup 安全松爪并重新抓取"
            state.artifacts.pop("module_4_buffer_state", None)
            return Decision("module_1_left_grasp")
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
        state.handoff = "EMPTY"
        return Decision("module_1_left_grasp")
    if name == "module_1_left_grasp" and result.status == Status.SUCCESS:
        # M1 的成功 handoff 已经是 ``graspuptop``。在这里先校验单容量
        # Buffer 确实为空，绝不能用新工件的 LEFT_PLACING 状态覆盖上一件的
        # OCCUPIED / LEFT_REGRASPING 状态。校验失败时保持左手在 M1 的
        # graspuptop 持件位，由总控转人工处理，而不是调用 M3 再次落件。
        if state.buffer != "EMPTY":
            state.left_arm = "HOLDS_WORKPIECE"
            return Decision(None, operator_required=True)
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
    if name in {"module_5_reverse", "module_5_reverse_handoff"} and result.status == Status.SUCCESS:
        # 工件已由左手从 Buffer 取走；它现在占用的是交接位而非 Buffer。
        state.buffer, state.handoff, state.left_arm = "EMPTY", "LEFT_READY", "AT_HANDOFF"
        return Decision("module_8_handoff")
    if name == "module_8_handoff" and result.status == Status.SUCCESS:
        state.handoff, state.left_arm, state.right_arm = "EMPTY", "IDLE", "HOLDS_WORKPIECE"
        return Decision("module_9_right_place")
    if name == "module_9_right_place" and result.status == Status.SUCCESS:
        return Decision("module_10_right_idle")
    if name == "module_10_right_idle" and result.status == Status.SUCCESS:
        state.buffer, state.right_arm = "EMPTY", "IDLE"
        return Decision(None, finished=True)
    return Decision(None, operator_required=True)
