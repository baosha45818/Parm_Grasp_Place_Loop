#!/usr/bin/env python3
"""Read-only Module 0 readiness checks for the Planner and hardware drivers.

This module intentionally never calls an SDK method that enables, moves,
initializes, cancels, or otherwise changes robot hardware state.
"""

from __future__ import annotations

from importlib import metadata
import math
import socket
import time
from typing import Any
from urllib.parse import urlparse

from modules.module_0_prepare.gripper_state_compat import is_gripper_usable


DEFAULT_REQUIRED_FRAMES = frozenset(
    {
        "body_base",
        "left_eef",
        "left_hand_transfer_end",
        "right_hand_transfer_end",
    }
)
LEFT_ONLY_REQUIRED_FRAMES = frozenset(
    {
        "body_base",
        "left_eef",
        "left_hand_transfer_end",
    }
)
# ``PlannerRuntime`` and Module 1 GraspNet now share the ordinary Planner.
# Keep the GraspNet tuple explicit because the preflight report and regression
# tests use it to verify the exact service contract consumed by Module 1.
PLANNER_SDK_PORTS = (5604, 5609, 5610)
GRASPNET_PLANNER_PORTS = PLANNER_SDK_PORTS


def _wait_for_state(reader, label: str, timeout_s: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        state = reader()
        if isinstance(state, dict):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{label} 未在 {timeout_s:.1f}s 内提供状态帧")
        time.sleep(0.05)


def _validate_arm_state(
    state: dict[str, Any], arm_names: tuple[str, ...] = ("left", "right")
) -> dict[str, Any]:
    estop_reason = state.get("estop_reason")
    # 以 system_state 判定是否真的进入急停。新驱动在已经恢复 RUNNING 后
    # 仍可能短暂保留上一条 HEARTBEAT_TIMEOUT；单看原因字段会误挡住运行。
    system_state = str(state.get("system_state", "")).strip().lower()
    normalized_estop_reason = (
        str(estop_reason).strip().lower()
        if estop_reason is not None
        else ""
    )
    if system_state in {"estop", "error"}:
        raise RuntimeError(f"机械臂急停锁存: {estop_reason}")
    if normalized_estop_reason not in {"", "none", "null"}:
        print(
            "[module-0] 警告：机械臂运行态保留了历史 estop_reason="
            f"{estop_reason!r}；system_state={state.get('system_state')!r}，继续预检。",
            flush=True,
        )
    arms = state.get("state")
    if not isinstance(arms, dict):
        raise RuntimeError(f"机械臂状态缺少 state: {state}")
    summary: dict[str, Any] = {
        "system_state": state.get("system_state"),
        "estop_reason": estop_reason,
    }
    for name in arm_names:
        arm = arms.get(name)
        positions = arm.get("position") if isinstance(arm, dict) else None
        if (
            not isinstance(positions, list)
            or len(positions) != 7
            or not all(math.isfinite(float(value)) for value in positions)
        ):
            raise RuntimeError(f"{name} 臂没有有效的 7 维关节状态: {arm}")
        summary[name] = {
            "enabled": arm.get("enabled"),
            "task_state": arm.get("task_state"),
        }
    return summary


def _validate_gripper_state(
    state: dict[Any, Any], hands: tuple[tuple[int, str], ...] = ((1, "left"), (2, "right"))
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for hand, label in hands:
        item = state.get(hand)
        if not is_gripper_usable(item):
            raise RuntimeError(f"{label} 夹爪不可用或未初始化: {item}")
        summary[label] = {
            "initialized": item.get("initialized"),
            "motion_state": item.get("motion_state"),
            "position": item.get("position"),
        }
    return summary


def _read_force_sensor(host: str, hand: int) -> dict[str, Any]:
    from parm_sdk import ForceSensor

    with ForceSensor(host=host, hand=hand) as sensor:
        state = _wait_for_state(sensor.get_state, f"hand={hand} 力传感器")
    force = state.get("filtered_force")
    torque = state.get("filtered_torque")
    for field, values in (("filtered_force", force), ("filtered_torque", torque)):
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise RuntimeError(f"hand={hand} 力传感器 {field} 无效: {state}")
    return {"force_n": [float(value) for value in force], "torque_nm": [float(value) for value in torque]}


def _check_tcp_endpoint(host: str, port: int, label: str) -> None:
    with socket.create_connection((host, port), timeout=2.0):
        pass


def _check_execute_endpoint(execute_url: str) -> None:
    parsed = urlparse(execute_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
        raise RuntimeError(f"执行端点必须是带端口的 http URL: {execute_url}")
    # TCP connect only: do not issue HTTP requests to the motion endpoint.
    _check_tcp_endpoint(parsed.hostname, parsed.port, "Planner execute")


def _sdk_version() -> str:
    try:
        return metadata.version("parm-sdk")
    except metadata.PackageNotFoundError:
        return "unknown"


def preflight_robot_infrastructure(
    *,
    planner_host: str,
    arm_host: str,
    gripper_host: str,
    force_sensor_host: str | None = None,
    execute_url: str = "http://127.0.0.1:5667/middle/waypoint/execute",
    required_frames: frozenset[str] | None = None,
    left_only: bool = False,
) -> dict[str, Any]:
    """Validate all loop-critical endpoints without changing hardware state."""
    from parm_sdk import Arm, Gripper, Planner

    force_host = force_sensor_host or gripper_host
    active_arm_names = ("left",) if left_only else ("left", "right")
    active_hands = ((1, "left"),) if left_only else ((1, "left"), (2, "right"))
    if required_frames is None:
        required_frames = (
            LEFT_ONLY_REQUIRED_FRAMES if left_only else DEFAULT_REQUIRED_FRAMES
        )
    with Planner(host=planner_host) as planner:
        frame_ids = set(planner.list_frame_ids(timeout=5.0))
    missing_frames = sorted(required_frames.difference(frame_ids))
    if missing_frames:
        raise RuntimeError(f"Planner TF 缺少必需坐标系: {missing_frames}")
    for port in GRASPNET_PLANNER_PORTS:
        _check_tcp_endpoint(planner_host, port, f"GraspNet Planner port {port}")
    _check_execute_endpoint(execute_url)

    with Arm(host=arm_host) as arm:
        arm_summary = _validate_arm_state(
            _wait_for_state(arm.get_state, "机械臂"), active_arm_names
        )
    with Gripper(host=gripper_host) as gripper:
        gripper_summary = _validate_gripper_state(
            _wait_for_state(gripper.get_state, "夹爪"), active_hands
        )

    force_summary = {
        label: _read_force_sensor(force_host, hand)
        for hand, label in active_hands
    }

    report = {
        "sdk_version": _sdk_version(),
        "planner": {
            "host": planner_host,
            "sdk_ports": PLANNER_SDK_PORTS,
            "graspnet_ports": GRASPNET_PLANNER_PORTS,
            "frame_count": len(frame_ids),
            "required_frames": sorted(required_frames),
            "execute_url": execute_url,
        },
        "arm": {"host": arm_host, **arm_summary},
        "gripper": {"host": gripper_host, **gripper_summary},
        "force_sensors": {
            "host": force_host,
            **force_summary,
        },
    }
    print(
        "[module-0] Planner/硬件只读预检通过: "
        f"SDK={report['sdk_version']}, TF={report['planner']['frame_count']} frames, "
        f"scope={'left-only' if left_only else 'dual-arm'}",
        flush=True,
    )
    return report
