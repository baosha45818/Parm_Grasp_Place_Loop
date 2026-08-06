#!/usr/bin/env python3
"""通用云台动作工具：将头部相机移动到调用方指定的观察角。"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--pitch", type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("真实云台动作必须显式传入 --execute")
    if args.settle_s < 0.0 or args.timeout_s <= 0.0:
        raise ValueError("--settle-s 必须非负，--timeout-s 必须为正")
    from parm_sdk import Head

    with Head(host=args.host) as head:
        head.ping()
        initial = head.get_state() or {}
        pitch_state = initial.get("pitch") or {}
        yaw_state = initial.get("yaw") or {}
        if pitch_state.get("alarm_message") or yaw_state.get("alarm_message"):
            raise RuntimeError(f"云台存在报警，拒绝运动: {initial}")
        if not pitch_state.get("enabled") or not yaw_state.get("enabled"):
            head.init()
            time.sleep(0.3)
        head.move_to(pitch=args.pitch, yaw=args.yaw)
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            state = head.get_state() or {}
            try:
                # 当前 SDK 的 pitch 状态符号与 move_to 指令相反。
                pitch = -float(state["pitch"]["position"])
                yaw = float(state["yaw"]["position"])
            except (KeyError, TypeError, ValueError):
                time.sleep(0.05)
                continue
            if abs(pitch - args.pitch) <= 0.5 and abs(yaw - args.yaw) <= 0.5:
                if args.settle_s:
                    time.sleep(args.settle_s)
                print(f"[head] 已到观察位: pitch={pitch:.2f}deg, yaw={yaw:.2f}deg")
                return 0
            time.sleep(0.05)
    raise RuntimeError("云台未在超时内到达指定观察位")


if __name__ == "__main__":
    raise SystemExit(main())
