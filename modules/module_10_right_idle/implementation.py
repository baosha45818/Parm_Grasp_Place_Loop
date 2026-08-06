"""模块 10 占位实现：将右臂移动到安全 Idle 位姿。"""

from time import time

from loop_types import ResultStatus, RightArmState, make_result
from tool.planner_runtime import PlannerRuntime


def module_10_right_idle(right: RightArmState, planner: PlannerRuntime):
    started = time()
    right.current_module = "10"
    planner.move_cartesian("right", "right_idle", planner="rrtstar")
    right.motion_state = "IDLE"
    return make_result(ResultStatus.SUCCESS, "10", started, "right idle")
