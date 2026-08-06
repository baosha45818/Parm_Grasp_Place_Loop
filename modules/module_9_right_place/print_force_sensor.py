#!/usr/bin/env python3
"""Print right-wrist force/torque values and their baseline-relative deltas."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="打印腕部六轴力传感器原始值及去基线后的差值"
    )
    parser.add_argument("--host", default="127.0.0.1", help="传感器服务地址")
    parser.add_argument("--arm-host", default="192.168.88.1", help="机械臂 SDK 地址")
    parser.add_argument(
        "--hand", type=int, choices=(1, 2), default=2,
        help="1=左手，2=右手；默认右手",
    )
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV 保存路径；省略时保存到 logs/force_sensor_<时间>.csv",
    )
    return parser.parse_args()


def vector(frame: dict, key: str) -> list[float]:
    values = frame.get(key)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 3
    ):
        raise ValueError(f"传感器帧中的 {key!r} 不是三轴数据: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"传感器帧中的 {key!r} 含非有限值: {values!r}")
    return result


def mean(samples: list[list[float]]) -> list[float]:
    return [sum(sample[axis] for sample in samples) / len(samples) for axis in range(3)]


def current_pose(arm: object, hand: int) -> list[float]:
    """Return the SDK pose as [x, y, z, roll, pitch, yaw]."""
    state = arm.get_state()  # type: ignore[attr-defined]
    arm_name = "left" if hand == 1 else "right"
    per_arm = ((state or {}).get("state") or {}).get(arm_name)
    if not isinstance(per_arm, dict):
        raise RuntimeError(f"机械臂状态中没有 {arm_name} 数据")
    raw_pose = per_arm.get("eef_pose")
    if isinstance(raw_pose, dict):
        xyz = raw_pose.get("xyz")
        rpy = raw_pose.get("rpy")
        raw_pose = [*(xyz or []), *(rpy or [])]
    if (
        not isinstance(raw_pose, Sequence)
        or isinstance(raw_pose, (str, bytes))
        or len(raw_pose) != 6
    ):
        raise ValueError(f"无效的 eef_pose: {raw_pose!r}")
    pose = [float(value) for value in raw_pose]
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"eef_pose 含非有限值: {raw_pose!r}")
    return pose


def main() -> int:
    args = parse_args()
    if args.baseline_samples < 1:
        raise ValueError("--baseline-samples 必须至少为 1")
    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0:
        raise ValueError("--rate-hz 必须是正数")

    try:
        from parm_sdk import Arm, ForceSensor
    except ImportError as exc:
        print(f"无法导入 parm_sdk: {exc}")
        return 2

    print("保持机械臂和工件静止，正在采集基线……")
    period = 1.0 / args.rate_hz
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("logs") / f"force_sensor_{timestamp}.csv"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arm = None
    try:
        arm = Arm(host=args.arm_host)
        with (
            output_path.open("w", newline="", encoding="utf-8") as output_file,
            ForceSensor(host=args.host, hand=args.hand) as sensor,
        ):
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "elapsed_s", "unix_time_s",
                    "fx_n", "fy_n", "fz_n",
                    "tx_nm", "ty_nm", "tz_nm",
                    "dfx_n", "dfy_n", "dfz_n",
                    "dtx_nm", "dty_nm", "dtz_nm",
                    "eef_x_m", "eef_y_m", "eef_z_m",
                    "eef_roll_rad", "eef_pitch_rad", "eef_yaw_rad",
                ]
            )
            output_file.flush()
            force_samples: list[list[float]] = []
            torque_samples: list[list[float]] = []
            for _ in range(args.baseline_samples):
                frame = sensor.wait_first(timeout=args.timeout)
                force_samples.append(vector(frame, "filtered_force"))
                torque_samples.append(vector(frame, "filtered_torque"))
                time.sleep(period)

            force_baseline = mean(force_samples)
            torque_baseline = mean(torque_samples)
            print(
                "基线 "
                f"force=[{force_baseline[0]:+.3f}, {force_baseline[1]:+.3f}, {force_baseline[2]:+.3f}] N  "
                f"torque=[{torque_baseline[0]:+.3f}, {torque_baseline[1]:+.3f}, {torque_baseline[2]:+.3f}] N·m"
            )
            print("现在可沿两个方向轻推；重点看 dFz 的正负。按 Ctrl+C 结束。")
            print(f"CSV 正在保存到: {output_path}")
            started_at = time.monotonic()

            while True:
                frame = sensor.wait_first(timeout=args.timeout)
                force = vector(frame, "filtered_force")
                torque = vector(frame, "filtered_torque")
                delta_force = [value - base for value, base in zip(force, force_baseline)]
                delta_torque = [value - base for value, base in zip(torque, torque_baseline)]
                pose = current_pose(arm, args.hand)
                writer.writerow(
                    [
                        f"{time.monotonic() - started_at:.6f}",
                        f"{time.time():.6f}",
                        *[f"{value:.9f}" for value in force],
                        *[f"{value:.9f}" for value in torque],
                        *[f"{value:.9f}" for value in delta_force],
                        *[f"{value:.9f}" for value in delta_torque],
                        *[f"{value:.9f}" for value in pose],
                    ]
                )
                # 每一帧都落盘，保证 Ctrl+C 或异常停止时已采集数据仍然完整。
                output_file.flush()
                print(
                    f"F=[{force[0]:+7.3f} {force[1]:+7.3f} {force[2]:+7.3f}] N  "
                    f"dF=[{delta_force[0]:+7.3f} {delta_force[1]:+7.3f} {delta_force[2]:+7.3f}] N  "
                    f"T=[{torque[0]:+7.3f} {torque[1]:+7.3f} {torque[2]:+7.3f}] N·m  "
                    f"dT=[{delta_torque[0]:+7.3f} {delta_torque[1]:+7.3f} {delta_torque[2]:+7.3f}] N·m  "
                    f"pose_xyz_rpy=[{pose[0]:+.6f} {pose[1]:+.6f} {pose[2]:+.6f} "
                    f"{pose[3]:+.6f} {pose[4]:+.6f} {pose[5]:+.6f}]",
                    flush=True,
                )
                time.sleep(period)
    except KeyboardInterrupt:
        print(f"\n打印和记录已停止。CSV 已保存到: {output_path}")
        return 0
    except Exception as exc:
        print(f"采集失败: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
