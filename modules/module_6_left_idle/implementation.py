"""模块 6 占位实现：将左臂移动到左侧安全 Idle 位姿。"""

from time import time

from loop_types import LeftArmState, ResultStatus, make_result
from tool.planner_runtime import PlannerRuntime


def module_6_left_idle(left: LeftArmState, planner: PlannerRuntime):
    started = time()
    left.current_module = "6"
    planner.move_cartesian("left", "left_idle", planner="rrtstar")
    left.motion_state = "IDLE"
    return make_result(ResultStatus.SUCCESS, "6", started, "left idle")
