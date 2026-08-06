#!/usr/bin/env python3
"""Module 8-2 gripper access compatible with both driver state schemas."""

from __future__ import annotations

import time
from typing import Any

from modules.module_0_prepare.gripper_state_compat import is_gripper_usable


STATUS_PORT = 7702


def _state_for_hand(gripper: Any, hand: int) -> dict[str, Any] | None:
    state = gripper.get_state().get(hand)
    return state if isinstance(state, dict) else None


def _wait_for_usable_state(
    gripper: Any, hand: int, timeout_s: float = 3.0
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_s
    state = _state_for_hand(gripper, hand)
    while not is_gripper_usable(state) and time.monotonic() < deadline:
        time.sleep(0.05)
        state = _state_for_hand(gripper, hand)
    return state


def _sync_response_arrived(response: object, hand: int) -> bool:
    """Read the new driver's completion state from a synchronous reply."""
    if not isinstance(response, dict):
        return False
    payload = response.get("data") or response
    devices = payload.get("devices") if isinstance(payload, dict) else None
    device_id = "left_gripper" if hand == 1 else "right_gripper"
    device = devices.get(device_id) if isinstance(devices, dict) else None
    return isinstance(device, dict) and device.get("device_state") == "arrived"


def _wait_for_sync_terminal_state(
    gripper: Any,
    hand: int,
    response: object,
    expected_states: set[str],
    timeout_s: float = 3.0,
) -> tuple[dict[str, Any] | None, bool]:
    """Accept the sync reply when PUB status is one frame behind the command."""
    response_arrived = _sync_response_arrived(response, hand)
    deadline = time.monotonic() + timeout_s
    state = _state_for_hand(gripper, hand)
    while time.monotonic() < deadline:
        motion_state = str((state or {}).get("motion_state", "")).strip().lower()
        if motion_state in expected_states:
            return state, True
        if motion_state in {"error", "dropped"}:
            raise RuntimeError(f"夹爪动作异常: {state}")
        # ``mode=sync`` 返回 arrived 后，驱动已确认动作完成；PUB 可能稍后才刷新。
        if response_arrived:
            return state, True
        time.sleep(0.05)
        state = _state_for_hand(gripper, hand)
    return state, response_arrived


def read_gripper_state(host: str, hand: int, label: str) -> dict[str, Any]:
    """Read a usable state without requiring the legacy initialized flag."""
    from parm_sdk import Gripper

    with Gripper(host=host, status_port=STATUS_PORT) as gripper:
        state = _wait_for_usable_state(gripper, hand)
    if not is_gripper_usable(state):
        raise RuntimeError(
            f"{label} gripper hand={hand} is unavailable; state={state}"
        )
    return state


def move_gripper(
    host: str,
    hand: int,
    position: float,
    force: float,
    label: str,
) -> dict[str, Any]:
    """Execute a synchronous command using the new/old shared state contract."""
    from parm_sdk import Gripper

    with Gripper(host=host, status_port=STATUS_PORT) as gripper:
        initial_state = _wait_for_usable_state(gripper, hand)
        if not is_gripper_usable(initial_state):
            raise RuntimeError(
                f"{label} gripper hand={hand} is unavailable; "
                f"state={initial_state}"
            )
        response = gripper.move(
            hand,
            position,
            force=force,
            mode="sync",
            preempt=False,
            timeout=60.0,
        )
        expected_states = {"arrived", "gripping"} if position <= 5.0 else {"arrived"}
        final_state, completed = _wait_for_sync_terminal_state(
            gripper, hand, response, expected_states
        )

    if response.get("status") != "success":
        raise RuntimeError(f"{label} gripper command failed: {response}")
    if not completed:
        raise RuntimeError(
            f"{label} gripper command returned but has unexpected final "
            f"state: {final_state}"
        )
    return response


def release_left_gripper(
    host: str,
    position: float,
    force: float,
    label: str = "left",
) -> dict[str, Any]:
    """Preempt an active left-hand grip and verify that the jaw really opens."""
    from parm_sdk import Gripper

    hand = 1
    print(
        "[Module 8-2 左夹爪释放] 使用async+preempt=True抢占夹持任务，"
        f"打开目标={position:.1f}%。",
        flush=True,
    )
    with Gripper(host=host, status_port=STATUS_PORT) as gripper:
        initial_state = _wait_for_usable_state(gripper, hand)
        if not is_gripper_usable(initial_state):
            raise RuntimeError(
                f"{label} gripper hand=1 is unavailable; state={initial_state}"
            )
        initial_position = float(initial_state.get("position", -1.0))
        response = gripper.move(
            hand,
            position,
            force=force,
            mode="async",
            preempt=True,
            timeout=60.0,
        )
        if response.get("status") not in {"accepted", "success"}:
            raise RuntimeError(f"left gripper release command failed: {response}")

        minimum_open_position = max(0.0, float(position) - 5.0)
        movement_threshold = min(
            minimum_open_position,
            initial_position + 1.0,
        )
        deadline = time.monotonic() + 60.0
        final_state: dict[str, Any] | None = None
        movement_observed = False
        while time.monotonic() < deadline:
            final_state = _state_for_hand(gripper, hand)
            if final_state is not None:
                motion_state = str(
                    final_state.get("motion_state", "")
                ).strip().lower()
                actual_position = float(final_state.get("position", -1.0))
                if actual_position >= movement_threshold:
                    movement_observed = True
                if (
                    movement_observed
                    and motion_state == "arrived"
                    and actual_position >= minimum_open_position
                ):
                    print(
                        "[Module 8-2 左夹爪释放] 已确认打开："
                        f"position={actual_position:.1f}%, state={motion_state}。",
                        flush=True,
                    )
                    return response
            time.sleep(0.1)

    raise RuntimeError(
        "Module 8-2左夹爪释放失败：抢占命令后未确认完全打开；"
        f"初始position={initial_position:.1f}%，"
        f"是否观察到运动={movement_observed}，"
        f"要求state=arrived且position>={minimum_open_position:.1f}%，"
        f"最终状态={final_state}。禁止右臂带件撤离"
    )
