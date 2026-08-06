"""模块 8-1 占位实现：模拟右手从 Buffer 抓取并更新状态。"""

from time import time

from loop_types import BufferState, Holding, ResultStatus, RightArmState, make_result
from tool.planner_runtime import PlannerRuntime


def module_8_1_buffer_grasp(
    buffer: BufferState, right: RightArmState, planner: PlannerRuntime
):
    started = time()
    right.current_module = "8-1"
    planner.move_cartesian("right", "right_buffer_grasp", planner="rrtstar")
    print("[8-1][占位] 关闭右夹爪；初版假设抓取成功")
    right.holding = Holding.SINGLE
    right.grasp_stable = True
    right.grasp_source = "BUFFER"
    right.motion_state = "BUFFER_GRASPED"
    planner.move_cartesian("right", "right_buffer_retreat", planner="rrtstar")
    with buffer.lock:
        buffer.full = False
        buffer.needs_alignment = False
        buffer.operator = "NONE"
    return make_result(ResultStatus.SUCCESS, "8-1", started, "right grasped from buffer")
