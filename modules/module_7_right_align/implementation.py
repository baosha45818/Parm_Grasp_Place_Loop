"""模块 7 临时实现：等待人工摆正 Buffer 工件并输入 y 确认。"""

from time import time

from loop_types import BufferState, ResultStatus, RightArmState, make_result


def module_7_manual_align(buffer: BufferState, right: RightArmState):
    started = time()
    right.current_module = "7"
    while True:
        answer = input("[7] 请人工摆正 Buffer 工件，完成后输入 y 并回车: ").strip().lower()
        if answer == "y":
            break
        print("[7] 尚未确认摆正，只接受 y。")
    with buffer.lock:
        buffer.needs_alignment = False
        buffer.operator = "NONE"
    right.motion_state = "BUFFER_ALIGNED_MANUALLY"
    return make_result(ResultStatus.SUCCESS, "7", started, "manual alignment confirmed")
