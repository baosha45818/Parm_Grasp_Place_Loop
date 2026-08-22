#!/usr/bin/env python3
"""Module 4 前置动作：使左臂和头部到 Buffer 标定观察位。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEFT_OBSERVATION_DEG = [14.281, -29.886, -85.633, 102.504, 37.378, 56.016, 9.241]
BUFFER_HEAD_PITCH_DEG = 38.5
BUFFER_HEAD_YAW_DEG = 0.0
# gzhu 云台反馈的 pitch 与 SDK 命令符号相反；标定使用命令角 +38.5°。
BUFFER_HEAD_FEEDBACK_PITCH_DEG = -38.5


def _joint_error(arm_host: str, target: list[float]) -> float:
    import numpy as np
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        position = ((((arm.get_state() or {}).get("state") or {}).get("left") or {}).get("position") or [])
        actual = np.asarray(position, dtype=float)
    if actual.shape != (7,):
        raise RuntimeError("无法读取 left 臂 7 维关节状态")
    delta = (actual - np.asarray(target, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(delta)))


def _move_left_to_observation(*, arm_host: str, planner_host: str, speed: float) -> None:
    from loop_config import LoopConfig
    from tools.planner_runtime import PlannerRuntime

    config = LoopConfig(
        planner_host=planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=speed,
        max_acceleration=speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        if _joint_error(arm_host, LEFT_OBSERVATION_DEG) <= 0.5:
            print("[module-4] 左臂 Buffer 观察位已到位，跳过零长度规划", flush=True)
            return
        planner.move_left_joints(LEFT_OBSERVATION_DEG, planner="rrtstar")
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            error = _joint_error(arm_host, LEFT_OBSERVATION_DEG)
            if error <= 0.5:
                print(f"[module-4] 左臂 Buffer 观察位到位，最大关节误差 {error:.3f}deg", flush=True)
                return
            time.sleep(0.1)
        raise RuntimeError("左臂 Buffer 观察位 90 秒内未到位")
    finally:
        planner.close()


def _move_head(host: str) -> None:
    from parm_sdk import Head

    with Head(host=host) as head:
        head.ping()
        state = head.get_state() or {}
        if not (state.get("pitch") or {}).get("enabled") or not (state.get("yaw") or {}).get("enabled"):
            head.init()
            time.sleep(0.3)
        head.move_to(pitch=BUFFER_HEAD_PITCH_DEG, yaw=BUFFER_HEAD_YAW_DEG)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state = head.get_state() or {}
            try:
                pitch = float(state["pitch"]["position"])
                yaw = float(state["yaw"]["position"])
            except (KeyError, TypeError, ValueError):
                time.sleep(0.05)
                continue
            if (abs(pitch - BUFFER_HEAD_FEEDBACK_PITCH_DEG) <= 0.5
                    and abs(yaw - BUFFER_HEAD_YAW_DEG) <= 0.5):
                time.sleep(1.0)
                return
            time.sleep(0.05)
    raise RuntimeError("头部云台未在 15 秒内到达 Buffer 观察位")


def _write(path: Path, outcome: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"outcome": outcome}, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", required=True)
    parser.add_argument("--head-host", required=True)
    parser.add_argument("--speed", type=float, default=0.3)
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed 必须在 (0, 1] 内")
    _move_left_to_observation(
        arm_host=args.arm_host, planner_host=args.planner_host, speed=args.speed,
    )
    _move_head(args.head_host)
    _write(Path(args.result_file), "success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
