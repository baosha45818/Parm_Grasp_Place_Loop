#!/usr/bin/env python3
"""独立测试右臂 servo_l：持续沿 SDK world +Z 上升指定距离并记录双臂状态。

本文件不调用 Parm_Grasp_Place_Loop 的任何运动代码。servo_l 的 delta 是速度
比例而不是位移，因此脚本按固定频率持续发送，并根据右臂实时末端 z 闭环停止。
可在运行期间另外启动左臂运动，用 JSONL 日志检查右臂调用是否被阻塞。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time


SDK_WHEEL = next(iter(sorted((Path(__file__).resolve().parents[1] / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True)), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--distance-m", type=float, default=0.05,
                        help="沿 SDK world +Z 的目标上升距离，默认 0.05 m")
    parser.add_argument("--frequency-hz", type=float, default=20.0,
                        help="servo_l 发送频率，默认 20 Hz，必须高于 5 Hz watchdog 下限")
    parser.add_argument("--velocity-ratio", type=float, default=0.10,
                        help="servo_l 的 +Z 速度比例，范围 (0, 1]，默认 0.10")
    parser.add_argument("--timeout-s", type=float, default=30.0,
                        help="总测试超时，默认 30 s")
    parser.add_argument("--execute", action="store_true",
                        help="确认连接真实机械臂并允许运动；不传时只打印配置")
    parser.add_argument("--yes", action="store_true",
                        help="跳过终端 y 确认；仍必须同时传 --execute")
    parser.add_argument("--log", type=Path,
                        help="JSONL 日志路径；默认写到 loop/logs")
    args = parser.parse_args()
    if not math.isfinite(args.distance_m) or not (0.0 < args.distance_m <= 0.10):
        parser.error("--distance-m 必须在 (0, 0.10] m")
    if not math.isfinite(args.frequency_hz) or not (5.0 < args.frequency_hz <= 100.0):
        parser.error("--frequency-hz 必须在 (5, 100] Hz")
    if not math.isfinite(args.velocity_ratio) or not (0.0 < args.velocity_ratio <= 1.0):
        parser.error("--velocity-ratio 必须在 (0, 1]")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        parser.error("--timeout-s 必须为正数")
    if args.yes and not args.execute:
        parser.error("--yes 必须与 --execute 同时使用")
    return args


def arm_snapshot(state: dict, arm_key: str) -> dict:
    per_arm = (state.get("state") or {}).get(arm_key) or {}
    velocity = per_arm.get("velocity")
    velocity_values = (
        [float(value) for value in velocity]
        if isinstance(velocity, (list, tuple)) and len(velocity) == 7
        else None
    )
    return {
        "task_state": per_arm.get("task_state"),
        "enabled": per_arm.get("enabled"),
        "position_deg": per_arm.get("position"),
        "max_abs_velocity_deg_s": (
            max(abs(value) for value in velocity_values)
            if velocity_values is not None
            else None
        ),
        "eef_pose": per_arm.get("eef_pose"),
    }


def right_xyz(state: dict) -> list[float]:
    pose = ((state.get("state") or {}).get("right") or {}).get("eef_pose") or {}
    xyz = pose.get("xyz")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise RuntimeError(f"右臂 eef_pose.xyz 不可用: {pose}")
    result = [float(value) for value in xyz]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"右臂 eef_pose.xyz 含非有限值: {xyz}")
    return result


def main() -> int:
    args = parse_args()
    log_path = args.log or (
        Path(__file__).resolve().parents[1] / "logs" /
        f"right_servo_l_up_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    log_path = log_path.expanduser().resolve()
    print("=== 右臂 servo_l 向上测试 ===")
    print(f"arm_host={args.arm_host} arm_id=2")
    print(
        f"目标：SDK world +Z 上升 {args.distance_m:.3f} m，"
        f"frequency={args.frequency_hz:.1f} Hz，velocity_ratio={args.velocity_ratio:.3f}"
    )
    print(f"日志：{log_path}")
    if not args.execute:
        print("dry-run：未传 --execute，不连接机械臂。")
        return 0
    if not SDK_WHEEL.is_file():
        raise FileNotFoundError(f"新版 SDK wheel 不存在: {SDK_WHEEL}")
    if not args.yes:
        answer = input(
            "确认右臂上方至少有 10 cm 安全空间、急停可用。输入 y 开始："
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消，未发送运动指令。")
            return 0

    sys.path.insert(0, str(SDK_WHEEL))
    from parm_sdk import Arm

    log_path.parent.mkdir(parents=True, exist_ok=True)
    arm = Arm(host=args.arm_host)
    started_motion = False
    try:
        state = arm.get_state() or {}
        right = (state.get("state") or {}).get("right") or {}
        if str(state.get("system_state", "")).lower() != "running":
            raise RuntimeError(f"机械臂 system_state 不是 RUNNING: {state.get('system_state')}")
        if right.get("enabled") is not True:
            raise RuntimeError("右臂未使能，拒绝运动")
        start_xyz = right_xyz(state)
        target_z = start_xyz[2] + args.distance_m
        print(f"起点 xyz={start_xyz}")
        print(f"目标 z={target_z:.6f} m（SDK world 坐标系）")

        arm.cancel_task(2)
        time.sleep(0.1)
        period = 1.0 / args.frequency_hz
        deadline = time.monotonic() + args.timeout_s
        next_tick = time.monotonic()
        sequence = 0
        maximum_call_s = 0.0
        with log_path.open("w", encoding="utf-8") as output:
            while time.monotonic() < deadline:
                before_call = time.monotonic()
                arm.servo_l(
                    2,
                    [0.0, 0.0, args.velocity_ratio, 0.0, 0.0, 0.0],
                    frame="world",
                )
                started_motion = True
                call_s = time.monotonic() - before_call
                maximum_call_s = max(maximum_call_s, call_s)
                sequence += 1

                state = arm.get_state() or {}
                xyz = right_xyz(state)
                risen_m = xyz[2] - start_xyz[2]
                record = {
                    "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                    "sequence": sequence,
                    "servo_l_call_s": call_s,
                    "requested_frequency_hz": args.frequency_hz,
                    "requested_delta": [0.0, 0.0, args.velocity_ratio, 0.0, 0.0, 0.0],
                    "start_xyz": start_xyz,
                    "target_z": target_z,
                    "actual_xyz": xyz,
                    "risen_m": risen_m,
                    "left": arm_snapshot(state, "left"),
                    "right": arm_snapshot(state, "right"),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

                if call_s > max(0.2, 2.0 * period):
                    print(
                        f"[阻塞] 第 {sequence} 次 servo_l 调用耗时 {call_s:.3f}s；"
                        f"left={record['left']['task_state']} "
                        f"right={record['right']['task_state']}",
                        flush=True,
                    )
                elif sequence == 1 or sequence % max(1, round(args.frequency_hz)) == 0:
                    print(
                        f"[进度] 上升 {risen_m * 100.0:.2f}cm / "
                        f"{args.distance_m * 100.0:.2f}cm，"
                        f"调用耗时 {call_s * 1000.0:.1f}ms",
                        flush=True,
                    )

                if risen_m >= args.distance_m:
                    arm.cancel_task(2)
                    started_motion = False
                    print(
                        f"[完成] 实际上升 {risen_m * 100.0:.2f}cm；"
                        f"最大单次调用耗时 {maximum_call_s:.3f}s"
                    )
                    return 0
                if risen_m < -0.005:
                    raise RuntimeError(
                        "右臂沿 +Z 指令后反向下降超过 5 mm，已停止；请检查 world 坐标方向"
                    )
                if math.dist(xyz[:2], start_xyz[:2]) > 0.02:
                    raise RuntimeError("右臂水平偏移超过 2 cm，已停止")

                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
        raise TimeoutError(
            f"{args.timeout_s:.1f}s 内未上升到目标；最大单次 servo_l 调用耗时 "
            f"{maximum_call_s:.3f}s，详见 {log_path}"
        )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止右臂 servo_l。")
        return 130
    finally:
        if started_motion:
            try:
                arm.cancel_task(2)
            except Exception as exc:
                print(f"警告：cancel_task(2) 失败：{exc}", file=sys.stderr)
        arm.close()
        print(f"日志已保存：{log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
