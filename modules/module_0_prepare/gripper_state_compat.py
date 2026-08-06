#!/usr/bin/env python3
"""兼容新旧夹爪驱动发布状态的就绪性判断。"""

from __future__ import annotations

from typing import Any


# 旧驱动用 initialized 标记回零完成；新左夹爪在动作终态发布 arrived，
# 即使 init 已成功也不会发布 ready。错误和掉线状态绝不能放行。
_USABLE_TERMINAL_STATES = frozenset({"ready", "arrived", "idle", "gripping"})


def is_gripper_usable(state: Any) -> bool:
    """返回当前状态是否足以安全发送下一条夹爪命令。"""
    if not isinstance(state, dict):
        return False
    if state.get("initialized") is True:
        return True
    return str(state.get("motion_state", "")).strip().lower() in _USABLE_TERMINAL_STATES
