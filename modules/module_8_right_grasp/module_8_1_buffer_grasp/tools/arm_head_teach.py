#!/usr/bin/env python3
"""PARM 手臂 Free 示教与云台键盘控制。

按键无需回车。D 启动 Free 示教（踏板按住后可拖动），S 保存当前姿态，
空格切换所选机械臂的夹爪开/闭；Y 进入云台模式，云台模式中 I/K 控制俯仰、
J/L 控制偏航。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
import termios
import time
import tty
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR / "saved_poses"
PITCH_MIN, PITCH_MAX = -45.0, 45.0
YAW_MIN, YAW_MAX = -60.0, 60.0


def import_sdk() -> Any:
    """导入已安装 SDK；未安装时使用同机器上的 SDK wheel。"""
    try:
        return importlib.import_module("parm_sdk")
    except ImportError:
        wheel_override = os.environ.get("PARM_SDK_WHEEL", "").strip()
        candidates = [Path(wheel_override)] if wheel_override else []
        candidates.extend(sorted(
            Path("/home/dainanjun/Public/parm_sdk_1.0.0").glob("parm_sdk-*.whl"),
            reverse=True,
        ))
        for wheel in candidates:
            if wheel.is_file():
                sys.path.insert(0, str(wheel))
                try:
                    return importlib.import_module("parm_sdk")
                except ImportError:
                    continue
    raise ImportError(
        "无法导入 parm_sdk。请安装 SDK，或设置 PARM_SDK_WHEEL 指向 SDK wheel。"
    )


def arm_keys(arm_id: int) -> list[tuple[int, str]]:
    if arm_id == 0:
        return [(1, "left"), (2, "right")]
    return [(arm_id, "left" if arm_id == 1 else "right")]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quat_to_rotvec(quaternion: list[float]) -> list[float]:
    """Convert an xyzw quaternion to an axis-angle rotation vector."""
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("规划器返回了零四元数。")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    sine = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sine < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sine, qw)
    return [qx / sine * angle, qy / sine * angle, qz / sine * angle]


def planner_eef_pose(planner: Any, arm_id: int) -> tuple[str, dict[str, list[float]]]:
    """Read the planner-model TCP pose in body_base for one arm."""
    key = "left" if arm_id == 1 else "right"
    ik_frame = f"{key}_hand_transfer_base"
    response = planner.lookup_transform("body_base", ik_frame)
    data = response.get("data") or {}
    translation, rotation = data.get("translation"), data.get("rotation")
    if not isinstance(translation, dict) or not isinstance(rotation, dict):
        raise RuntimeError(f"规划器未返回 body_base→{ik_frame} 的有效 TF。")
    xyz = [float(translation[axis]) for axis in ("x", "y", "z")]
    quat = [float(rotation[axis]) for axis in ("x", "y", "z", "w")]
    if not all(math.isfinite(value) for value in xyz + quat):
        raise RuntimeError(f"规划器返回 body_base→{ik_frame} 的非有限值。")
    return ik_frame, {"xyz": xyz, "rotvec": quat_to_rotvec(quat)}


def save_pose(arm: Any, planner: Any, arm_id: int) -> Path:
    """保存规划器 body_base 中的 TCP 位姿及驱动原始位姿。"""
    state = arm.get_state()
    if state is None:
        raise RuntimeError("尚未收到机械臂状态，无法保存；请稍后再按 S。")

    arms: dict[str, dict[str, Any]] = {}
    raw_arms = state.get("state") or {}
    for selected_id, key in arm_keys(arm_id):
        per_arm = raw_arms.get(key)
        if per_arm is None:
            raise RuntimeError(f"arm_id={selected_id}（{key}）状态不可用，未保存。")
        ik_frame, planner_pose = planner_eef_pose(planner, selected_id)
        arms[key] = {
            "arm_id": selected_id,
            "joint_positions_deg": per_arm.get("position"),
            "eef_pose": planner_pose,
            "eef_pose_ik_frame": ik_frame,
            "driver_eef_pose": per_arm.get("eef_pose"),
        }

    timestamp = datetime.now(timezone.utc)
    record = {
        "saved_at_utc": timestamp.isoformat(),
        "selected_arm_id": arm_id,
        "joint_angle_unit": "deg",
        "eef_pose_note": "planner TCP pose: body_base -> *_hand_transfer_base; xyz in m, rotvec in rad",
        "eef_pose_reference_frame": "body_base",
        "arms": arms,
    }
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = SAVE_DIR / f"arm_pose_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    filename.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return filename


def read_head_angles(head: Any) -> tuple[float, float] | None:
    state = head.get_state()
    if state is None:
        return None
    try:
        return float(state["pitch"]["position"]), float(state["yaw"]["position"])
    except (KeyError, TypeError, ValueError):
        return None


def print_head_angles(angles: tuple[float, float]) -> None:
    """打印最近一次已确认下发的完整云台姿态。"""
    pitch, yaw = angles
    print(f"当前云台: pitch={pitch:.1f}°, yaw={yaw:.1f}°")


def start_free_teach(arm: Any, arm_id: int) -> None:
    # SDK 的示教接口仅支持单臂；选择 0 时分别给左右臂下发命令。
    for selected_id, _ in arm_keys(arm_id):
        arm.stop_teach(selected_id)
    for selected_id, name in arm_keys(arm_id):
        arm.start_teach(selected_id, free=True)
        print(f"{name} 臂已进入 Free 示教。")
    print("按住踏板约 0.5 秒后可拖动；松开踏板后请确认手臂稳定，再按 S 保存。")


def stop_teach(arm: Any, arm_id: int) -> None:
    for selected_id, name in arm_keys(arm_id):
        arm.stop_teach(selected_id)
        print(f"{name} 臂示教已停止。")


def toggle_grippers(gripper: Any, arm_id: int, target_open: bool | None) -> bool | None:
    """切换所选夹爪开/闭，并返回本次下发后的目标状态。

    首次调用依据状态确定反向目标；之后使用最近一次目标，避免夹爪尚在运动时
    连按空格却重复下发同一方向的命令。
    """
    selected = arm_keys(arm_id)
    if target_open is None:
        try:
            states = gripper.get_state()
        except Exception as exc:
            print(f"读取夹爪状态失败: {exc}")
            return None
        positions = [
            float(state["position"])
            for selected_id, _ in selected
            if (state := states.get(selected_id)) is not None
            and state.get("position") is not None
        ]
        if not positions:
            print("夹爪状态尚未就绪，无法判断开/闭状态。")
            return None
        # 任一所选夹爪未打开时，先执行打开；双臂状态不一致时可恢复一致。
        target_open = any(position < 50.0 for position in positions)
    else:
        target_open = not target_open

    position = 100.0 if target_open else 0.0
    action = "打开" if target_open else "闭合"
    moved: list[str] = []
    for selected_id, name in selected:
        try:
            # 异步且抢占旧指令，使按键控制不会阻塞，且连按可立即反向。
            gripper.move(selected_id, position, mode="async", preempt=True)
            moved.append(name)
        except Exception as exc:
            print(f"{name} 夹爪{action}命令失败: {exc}")
    if not moved:
        return None
    print(f"已下发{'、'.join(moved)}夹爪{action}命令。")
    return target_open


def clear_error_and_recover(arm: Any) -> None:
    """按 SDK 规定显式解除 ERROR，并在恢复后重新使能机械臂。"""
    state = arm.get_state()
    if state is None:
        print("[E] 尚未收到机械臂状态，无法确认是否处于安全故障模式。")
        return

    system_state = str(state.get("system_state", "")).strip().lower()
    if system_state == "estop":
        print(
            "[E] 当前为 ESTOP，不能由脚本解除；请先在机器人侧释放急停。"
            f"estop_reason={state.get('estop_reason')!r}"
        )
        return
    if system_state != "error":
        print(f"[E] 当前 system_state={system_state!r}，不是 ERROR，无需 clear_error。")
        return

    print("[E] 正在 clear_error()；请确认现场障碍、碰撞等故障已排除。")
    try:
        arm.clear_error()
    except Exception as exc:
        print(f"[E] clear_error() 失败: {exc}")
        return

    deadline = time.monotonic() + 5.0
    last_state = system_state
    while time.monotonic() < deadline:
        time.sleep(0.05)
        state = arm.get_state()
        if state is None:
            continue
        last_state = str(state.get("system_state", "")).strip().lower()
        if last_state == "estop":
            print(
                "[E] 恢复过程中进入 ESTOP；请先在机器人侧释放急停。"
                f"estop_reason={state.get('estop_reason')!r}"
            )
            return
        if last_state == "running":
            try:
                # enable() 是整机接口；只重新使能，不会让任何手臂产生运动。
                arm.enable()
                print("[E] ERROR 已清除，系统已恢复 RUNNING 并重新使能。")
            except Exception as exc:
                print(f"[E] 已恢复 RUNNING，但重新使能失败: {exc}")
            return
    print(f"[E] clear_error() 后等待 RUNNING 超时，当前 system_state={last_state!r}。")


def main() -> int:
    parser = argparse.ArgumentParser(description="PARM 手臂 Free 示教 + 云台键盘控制")
    parser.add_argument("--arm-id", type=int, choices=(0, 1, 2), required=True,
                        help="0=双臂，1=左臂，2=右臂")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    parser.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "localhost"))
    parser.add_argument("--head-host", default=os.environ.get("PARM_HEAD_HOST", "localhost"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
    parser.add_argument("--pitch-step", type=float, default=0.5, help="I/K 每次俯仰步长（度，默认 0.5）")
    parser.add_argument("--yaw-step", type=float, default=0.5, help="J/L 每次偏航步长（度，默认 0.5）")
    args = parser.parse_args()

    if args.pitch_step <= 0 or args.yaw_step <= 0:
        parser.error("云台步长必须大于 0")

    sdk = import_sdk()
    arm = sdk.Arm(host=args.arm_host)
    head = None
    gripper = None
    gripper_target_open: bool | None = None
    # 云台 7706 的状态发布在部分设备上会混入错误的 pitch 值。这个变量是
    # 最近一次成功下发的完整姿态：既作为相对控制基准，也作为控制台显示值，
    # 不受错误状态帧影响。进入云台模式时用当时的状态帧作初始值。
    head_target: tuple[float, float] | None = None
    mode = "arm"
    stdin_fd = sys.stdin.fileno()
    old_terminal = termios.tcgetattr(stdin_fd)

    print(f"机械臂目标: {args.arm_id}（0=双臂，1=左，2=右），arm host={args.arm_host}")
    print("手臂模式: [D] Free 示教（踩踏板拖动）  [S] 保存关节角度和末端位姿  [空格] 夹爪开/闭")
    print("[Y] 进入/返回云台模式  [X] 停止示教  [E] 解除 ERROR 安全故障  [Q] 退出")
    print("云台模式: [I] 抬头  [K] 低头  [J] 向左  [L] 向右（每次打印实时 pitch/yaw）")
    print("终端需保持焦点，直接按键，无需回车。")

    try:
        tty.setcbreak(stdin_fd)
        while True:
            key = sys.stdin.read(1).lower()
            if key in ("\n", "\r"):
                continue
            if key == "q":
                break
            if key == "y":
                if mode == "arm":
                    if head is None:
                        head = sdk.Head(host=args.head_host)
                    mode = "head"
                    print("已进入云台模式。I/K/J/L 控制；Y 返回手臂模式。")
                    head_target = read_head_angles(head)
                    if head_target is None:
                        print("云台状态尚未就绪。")
                    else:
                        print_head_angles(head_target)
                else:
                    mode = "arm"
                    print("已返回手臂模式。")
                continue
            if key == "e":
                clear_error_and_recover(arm)
                continue
            if mode == "arm":
                if key == "d":
                    start_free_teach(arm, args.arm_id)
                elif key == "s":
                    planner = None
                    try:
                        planner = sdk.Planner(host=args.planner_host)
                        path = save_pose(arm, planner, args.arm_id)
                        print(f"已保存规划器 body_base 下的末端位姿: {path}")
                    finally:
                        if planner is not None:
                            planner.close()
                elif key == "x":
                    stop_teach(arm, args.arm_id)
                elif key == " ":
                    if gripper is None:
                        gripper = sdk.Gripper(host=args.gripper_host)
                    gripper_target_open = toggle_grippers(
                        gripper, args.arm_id, gripper_target_open
                    )
                else:
                    print("手臂模式可用按键: D / S / 空格 / E / Y / X / Q")
                continue

            if key not in {"i", "j", "k", "l"}:
                print("云台模式可用按键: I / J / K / L / E / Y / Q")
                continue
            assert head is not None
            if head_target is None:
                head_target = read_head_angles(head)
            if head_target is None:
                print("云台状态尚未就绪，无法进行相对控制。")
                continue
            pitch, yaw = head_target
            if key == "i":
                pitch = clamp(pitch + args.pitch_step, PITCH_MIN, PITCH_MAX)
            elif key == "k":
                pitch = clamp(pitch - args.pitch_step, PITCH_MIN, PITCH_MAX)
            elif key == "j":
                yaw = clamp(yaw + args.yaw_step, YAW_MIN, YAW_MAX)
            else:  # l
                yaw = clamp(yaw - args.yaw_step, YAW_MIN, YAW_MAX)
            head.move_to(pitch=pitch, yaw=yaw)
            head_target = (pitch, yaw)
            print_head_angles(head_target)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出。")
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_terminal)
        try:
            stop_teach(arm, args.arm_id)
        except Exception as exc:
            print(f"退出时停止示教失败: {exc}", file=sys.stderr)
        if head is not None:
            head.close()
        if gripper is not None:
            gripper.close()
        arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
