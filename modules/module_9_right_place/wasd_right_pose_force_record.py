#!/usr/bin/env python3
"""Move the right arm to its fixed joint start, jog it with WASD, and log wrench data.

The pose supplied on the command line and the CSV pose columns use the SDK
world frame: ``x y z roll pitch yaw`` (metres, radians).  Keyboard jogging is
also expressed in that frame: W/S are +/-X and A/D are +/-Y.  The orientation
and Z coordinate remain fixed while jogging.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import sys
import termios
import time
import tty


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUSH_T_DIR = Path(__file__).resolve().parent / "vendor/push_t"
if str(PUSH_T_DIR) not in sys.path:
    sys.path.insert(0, str(PUSH_T_DIR))

from move_by_joints import (  # noqa: E402
    current_pose,
    ensure_ready,
    import_sdk,
    move_l,
    rotvec_to_rpy,
    rpy_to_rotvec,
)
from loop_config import load_config  # noqa: E402
from tool.planner_runtime import PlannerRuntime  # noqa: E402


DEFAULT_PLANNER_CONFIG = ROOT / "config.hole_a_first_live.json"
INITIAL_RIGHT_JOINTS_DEG = [
    -4.7626,
    44.446600000000004,
    -113.02600000000001,
    -51.498400000000004,
    77.4371,
    -40.534,
    29.5669
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="右臂回固定关节起点后，以 WASD 微调并记录实际位姿和右腕力传感器。"
    )
    parser.add_argument(
        "--target", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="可选：回固定关节起点后，再移动到该世界系 xyz（m）+ RPY（rad）",
    )
    parser.add_argument(
        "--initial-motion", choices=("planner-movej", "cartesian-servo"), default="planner-movej",
        help="进入 WASD 起点的方式；默认规划器避碰 move_j",
    )
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--live", action="store_true", help="连接并控制真实硬件；省略时只打印计划")
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="与 --live 一起使用，确认周围安全且允许真实右臂运动",
    )
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--force-host", default="127.0.0.1")
    parser.add_argument("--step-mm", type=float, default=1.0, help="每次 WASD 的世界系位移，默认 1 mm")
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument("--sample-timeout", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path,
        help="CSV 输出路径；默认 logs/right_wasd_pose_force_<时间>.csv",
    )
    args = parser.parse_args()
    if args.live != args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    if not math.isfinite(args.step_mm) or not 0 < args.step_mm <= 5.0:
        parser.error("--step-mm 必须在 (0, 5] mm 内")
    if args.baseline_samples < 1:
        parser.error("--baseline-samples 必须至少为 1")
    if not math.isfinite(args.sample_timeout) or args.sample_timeout <= 0:
        parser.error("--sample-timeout 必须为正数")
    if args.target is not None and not all(math.isfinite(value) for value in args.target):
        parser.error("--target 必须全为有限数")
    return args


def initial_joints_for_config(config_path: Path, *, live: bool) -> list[float]:
    """Match run_hole_a_insertion's preinsert-joint selection exactly."""
    config = load_config(config_path, live=live)
    insertion = config.hole_a_insertion
    if insertion is not None and insertion.preinsert_joints_deg is not None:
        return list(insertion.preinsert_joints_deg)
    return list(INITIAL_RIGHT_JOINTS_DEG)


def move_to_initial_joints_planned(config_path: Path) -> list[float]:
    config = load_config(config_path, live=True)
    insertion = config.hole_a_insertion
    target_joints = (
        list(insertion.preinsert_joints_deg)
        if insertion is not None and insertion.preinsert_joints_deg is not None
        else list(INITIAL_RIGHT_JOINTS_DEG)
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        print(
            "Planner move_j 到 WASD 初始位姿："
            f"目标={[round(value, 3) for value in target_joints]}。"
        )
        planner.move_right_joints(target_joints)
    finally:
        planner.close()
    return target_joints


def vector(frame: dict, key: str) -> list[float]:
    values = frame.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError(f"力传感器的 {key} 不是三轴数据: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"力传感器的 {key} 含非有限值: {values!r}")
    return result


def mean(samples: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in samples) / len(samples) for index in range(3)]


def read_wrench(sensor: object, timeout: float) -> tuple[list[float], list[float]]:
    frame = sensor.wait_first(timeout=timeout)  # type: ignore[attr-defined]
    return vector(frame, "filtered_force"), vector(frame, "filtered_torque")


def write_sample(
    writer: csv.writer, output_file, started: float, event: str, target_rpy: list[float],
    actual_rotvec: list[float], force: list[float], torque: list[float],
    force_base: list[float], torque_base: list[float],
) -> None:
    actual_rpy = rotvec_to_rpy(actual_rotvec[3:])
    delta_force = [value - base for value, base in zip(force, force_base)]
    delta_torque = [value - base for value, base in zip(torque, torque_base)]
    writer.writerow([
        f"{time.monotonic() - started:.6f}", f"{time.time():.6f}", event,
        *[f"{value:.9f}" for value in target_rpy],
        *[f"{value:.9f}" for value in actual_rotvec[:3] + actual_rpy],
        *[f"{value:.9f}" for value in force + torque + delta_force + delta_torque],
    ])
    output_file.flush()


def main() -> int:
    args = parse_args()
    if not args.live:
        initial_joints = initial_joints_for_config(args.planner_config, live=False)
        print("dry-run：不会连接硬件。")
        print(
            "初始关节角（deg，和 run_hole_a_insertion 的 preinsert 相同）: "
            f"{initial_joints}；方式={args.initial_motion}"
        )
        if args.target is None:
            print("初始 WASD 零点：规划器到达固定关节角后读取控制器实际末端位姿。")
        else:
            print(f"额外笛卡尔目标 world xyz+rpy: {list(args.target)}")
        print(f"W/S = +/-X，A/D = +/-Y，每次 {args.step_mm:.3f} mm。")
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError("WASD 控制需要在有焦点的交互式终端中运行。")

    output_path = args.output or Path("logs") / f"right_wasd_pose_force_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sdk = import_sdk()
    arm = sensor = None
    stdin_fd = sys.stdin.fileno()
    previous_terminal = termios.tcgetattr(stdin_fd)
    try:
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, 2)
        sensor = sdk.ForceSensor(host=args.force_host, hand=2)
        if args.initial_motion == "planner-movej":
            move_to_initial_joints_planned(args.planner_config)
        if args.target is None:
            actual_start = current_pose(arm, 2)
            target_rpy = actual_start[:3] + rotvec_to_rpy(actual_start[3:])
            print(f"已到固定关节起点；WASD 零点 xyz+rpy={target_rpy}")
        else:
            target_rpy = list(args.target)
            target_command = target_rpy[:3] + rpy_to_rotvec(target_rpy[3:])
            print("正在以低速移动到 --target 指定初始位姿……")
            move_l(arm, 2, target_command, 0.01, 0.10, 0.02, 0.20, 50.0, 0.0005, 0.02, 8.0)
        print("采集静态力基线，请勿触碰末端……")
        force_base, torque_base = [], []
        for _ in range(args.baseline_samples):
            force, torque = read_wrench(sensor, args.sample_timeout)
            force_base.append(force)
            torque_base.append(torque)
        force_base, torque_base = mean(force_base), mean(torque_base)
        print(f"力基线 F={force_base} N, T={torque_base} N*m")
        print(f"已到初始点。W/S=+/-X，A/D=+/-Y，Q=结束。每次 {args.step_mm:.3f} mm")
        print(f"CSV: {output_path}")

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "elapsed_s", "unix_time_s", "event",
                "target_x_m", "target_y_m", "target_z_m", "target_roll_rad", "target_pitch_rad", "target_yaw_rad",
                "actual_x_m", "actual_y_m", "actual_z_m", "actual_roll_rad", "actual_pitch_rad", "actual_yaw_rad",
                "fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm",
                "dfx_n", "dfy_n", "dfz_n", "dtx_nm", "dty_nm", "dtz_nm",
            ])
            started = time.monotonic()
            force, torque = read_wrench(sensor, args.sample_timeout)
            write_sample(writer, output_file, started, "initial", target_rpy, current_pose(arm, 2), force, torque, force_base, torque_base)
            tty.setcbreak(stdin_fd)
            while True:
                key = sys.stdin.read(1).lower()
                if key == "q" or key == "\x03":
                    break
                changes = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1)}
                if key not in changes:
                    continue
                axis, direction = changes[key]
                target_rpy[axis] += direction * args.step_mm / 1000.0
                target_command = target_rpy[:3] + rpy_to_rotvec(target_rpy[3:])
                move_l(arm, 2, target_command, 0.003, 0.05, 0.01, 0.10, 50.0, 0.0003, 0.01, 4.0)
                force, torque = read_wrench(sensor, args.sample_timeout)
                actual = current_pose(arm, 2)
                write_sample(writer, output_file, started, key, target_rpy, actual, force, torque, force_base, torque_base)
                print(f"[{key.upper()}] target xyz={target_rpy[:3]} actual xyz={actual[:3]} F={force} T={torque}")
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, previous_terminal)
        if sensor is not None:
            sensor.close()
        if arm is not None:
            arm.close()
    print(f"记录已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
