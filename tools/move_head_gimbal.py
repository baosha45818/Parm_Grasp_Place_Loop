#!/usr/bin/env python3
"""把头部云台移动到模式标定的观察角，并验证实时反馈。"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    """解析通用云台动作参数；调用方必须明确授权执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", required=True)
    parser.add_argument("--pitch", type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--feedback-pitch-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    return parser.parse_args()


def wait_until_arrived(head, args: argparse.Namespace) -> None:
    """等待云台到达目标；兼容两代 driver 的 pitch 反馈符号。"""
    deadline = time.monotonic() + args.timeout_s
    last_state = None
    while time.monotonic() < deadline:
        state = head.get_state() or {}
        last_state = state
        try:
            pitch = args.feedback_pitch_sign * float(state["pitch"]["position"])
            yaw = float(state["yaw"]["position"])
        except (KeyError, TypeError, ValueError):
            time.sleep(0.05)
            continue
        if abs(pitch - args.pitch) <= 0.5 and abs(yaw - args.yaw) <= 0.5:
            if args.settle_s:
                time.sleep(args.settle_s)
            print(f"[head] 已到观察位: pitch={pitch:.2f}deg, yaw={yaw:.2f}deg")
            return
        time.sleep(0.05)
    raise RuntimeError(f"云台未在超时内到达观察位；目标=({args.pitch}, {args.yaw})，最后状态={last_state}")


def main() -> int:
    """检查授权和报警后，执行一次全自动云台定位。"""
    args = parse_args()
    if not args.execute:
        raise RuntimeError("真实云台动作必须显式传入 --execute")
    if args.settle_s < 0.0 or args.timeout_s <= 0.0:
        raise ValueError("--settle-s 必须非负，--timeout-s 必须为正")
    from parm_sdk import Head

    with Head(host=args.host) as head:
        head.ping()
        state = head.get_state() or {}
        pitch_state = state.get("pitch") or {}
        yaw_state = state.get("yaw") or {}
        if pitch_state.get("alarm_message") or yaw_state.get("alarm_message"):
            raise RuntimeError(f"云台存在报警，拒绝运动: {state}")
        if not pitch_state.get("enabled") or not yaw_state.get("enabled"):
            head.init()
            time.sleep(0.3)
        head.move_to(pitch=args.pitch, yaw=args.yaw)
        wait_until_arrived(head, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
