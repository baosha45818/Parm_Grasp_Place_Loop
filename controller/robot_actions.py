#!/usr/bin/env python3
"""总控自有的安全动作：启动归位和 Buffer 观察位。

这里不包含任何工件业务判断；业务抓取/放置仍由 modules 完成。把这类动作放在
controller 内，是为了让 Ctrl+C、资源锁和统一结果协议能覆盖它们。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEFT_GRASPUP_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]
# Buffer 重拍前的桌面撤离/观察位；沿用已标定的旧链路数值。
LEFT_OBSERVATION_DEG = [14.281, -29.886, -85.633, 102.504, 37.378, 56.016, 9.241]
# gzhu 头部 D435i 标定（gzhu_reverse_observation.json）对应的固定观察角。
# M4.1 发布的 ICP TF 必须与该实际云台姿态一致。
BUFFER_HEAD_PITCH_DEG = 38.5
BUFFER_HEAD_YAW_DEG = 0.0
# gzhu 云台的 SDK 命令坐标与 get_state() 反馈坐标 pitch 符号相反。
# 标定文件使用命令角 +38.5°，实际反馈到位值为 -38.5°。
BUFFER_HEAD_FEEDBACK_PITCH_DEG = -38.5


def _right_idle() -> list[float]:
    path = ROOT / "modules/module_10_right_idle/right_idle_pose.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = ((payload.get("arms") or {}).get("right") or {}).get("joint_positions_deg")
    if not isinstance(values, list) or len(values) != 7:
        raise RuntimeError(f"右臂 Idle 配置无效: {path}")
    return [float(value) for value in values]


def _joint_error(arm_host: str, arm_id: int, target: list[float]) -> float:
    import numpy as np
    from parm_sdk import Arm

    key = "left" if arm_id == 1 else "right"
    with Arm(host=arm_host) as arm:
        actual = np.asarray((((arm.get_state() or {}).get("state") or {}).get(key) or {}).get("position") or [], dtype=float)
    if actual.shape != (7,):
        raise RuntimeError(f"无法读取 {key} 臂 7 维关节状态")
    delta = (actual - np.asarray(target, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(delta)))


def _wait_joint(arm_host: str, arm_id: int, target: list[float], label: str) -> None:
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        error = _joint_error(arm_host, arm_id, target)
        if error <= 0.5:
            print(f"[controller] {label} 到位，最大关节误差 {error:.3f}deg", flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError(f"{label} 90 秒内未到位")


def _move_joint(*, arm_host: str, planner_host: str, side: str, target: list[float], speed: float, label: str) -> None:
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
    arm_id = 1 if side == "left" else 2
    frame = "left_hand_transfer_end" if side == "left" else "right_hand_transfer_end"
    try:
        planner.open(required_frames={"body_base", frame})
        if _joint_error(arm_host, arm_id, target) <= 0.5:
            print(f"[controller] {label} 已到位，跳过零长度规划", flush=True)
            return
        if side == "left":
            planner.move_left_joints(target, planner="rrtstar")
        else:
            planner.move_right_joints(target, planner="rrtstar")
        _wait_joint(arm_host, arm_id, target, label)
    finally:
        planner.close()


def _open_grippers(host: str) -> None:
    from parm_sdk import Gripper

    with Gripper(host=host, status_port=7702) as gripper:
        for hand, label in ((1, "左"), (2, "右")):
            response = gripper.move(hand, 100.0, force=100.0, speed=80.0 if hand == 1 else None, mode="sync", preempt=False, timeout=60.0)
            state = gripper.get_state().get(hand) or {}
            devices = ((response.get("data") or response).get("devices") or {}) if isinstance(response, dict) else {}
            device = devices.get("left_gripper" if hand == 1 else "right_gripper") or {}
            response_position = float((device.get("state_interfaces") or {}).get("gripper/position", 0.0))
            arrived = state.get("motion_state") == "arrived" or device.get("device_state") == "arrived"
            position = float(state.get("position", response_position))
            if not arrived or position < 95.0:
                raise RuntimeError(f"{label}夹爪打开未确认到位: {state}")
            print(f"[controller] {label}夹爪已打开：position={position:.1f}%", flush=True)


def _preflight(args: argparse.Namespace) -> None:
    from modules.module_0_prepare.ensure_sam2_service import ensure_sam2_service
    from modules.module_0_prepare.preflight_robot_infrastructure import preflight_robot_infrastructure

    # Module 1 在自身进程中直接用 RealSense 驱动打开 D405；总控启动阶段
    # 不占用、不探测任何相机，避免预检与实际采集争抢设备。
    ensure_sam2_service()
    preflight_robot_infrastructure(
        planner_host=args.planner_host,
        arm_host=args.arm_host,
        gripper_host=args.gripper_host,
        force_sensor_host=args.force_host,
    )


def _move_head(host: str) -> None:
    """移动到与 Module 4.1 gzhu 标定严格一致的 Buffer 观察角。"""
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
    parser.add_argument("--action", choices=("prepare", "observation"), required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--gripper-host", default="192.168.3.11")
    parser.add_argument("--force-host", default="192.168.3.22")
    parser.add_argument("--head-host", default="192.168.3.22")
    parser.add_argument("--speed", type=float, default=0.3)
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed 必须在 (0, 1] 内")
    if args.action == "prepare":
        _preflight(args)
        _open_grippers(args.gripper_host)
        # 两臂目标独立，启动阶段并行执行，且各自通过控制器资源锁保护。
        errors: list[BaseException] = []
        def move(side: str, target: list[float], label: str) -> None:
            try:
                _move_joint(arm_host=args.arm_host, planner_host=args.planner_host, side=side, target=target, speed=args.speed, label=label)
            except BaseException as exc:  # 汇总后统一由 controller 取消两臂。
                errors.append(exc)
        left = threading.Thread(target=move, args=("left", LEFT_GRASPUP_DEG, "左臂 graspup"))
        right = threading.Thread(target=move, args=("right", _right_idle(), "右臂 Idle"))
        left.start(); right.start(); left.join(); right.join()
        if errors:
            raise RuntimeError(f"启动归位失败: {errors[0]}") from errors[0]
    else:
        _move_joint(arm_host=args.arm_host, planner_host=args.planner_host, side="left", target=LEFT_OBSERVATION_DEG, speed=args.speed, label="左臂 Buffer 观察位")
        _move_head(args.head_host)
    _write(Path(args.result_file), "success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
