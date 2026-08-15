#!/usr/bin/env python3
"""只读记录 PARM 安全状态变化，不发送任何机械臂命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/arm_safety_state.jsonl"),
        help="JSONL 状态变化日志路径",
    )
    return parser.parse_args()


def snapshot(state: dict) -> dict:
    arms = state.get("state") or {}
    result = {
        "system_state": state.get("system_state"),
        "estop_reason": state.get("estop_reason"),
    }
    for name in ("left", "right"):
        arm = arms.get(name) or {}
        result[name] = {
            "enabled": arm.get("enabled"),
            "task_state": arm.get("task_state"),
            "error": arm.get("error"),
        }
    return result


def main() -> int:
    args = parse_args()
    if args.interval_s <= 0:
        raise ValueError("--interval-s 必须大于 0")

    from parm_sdk import Arm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    print(f"[watch] 只读监听 {args.arm_host}，日志: {args.output.resolve()}", flush=True)
    print("[watch] Ctrl+C 停止；本脚本不会发送 cancel_task、恢复或运动命令。", flush=True)

    with Arm(host=args.arm_host) as arm, args.output.open("a", encoding="utf-8") as log_file:
        try:
            while True:
                raw = arm.get_state()
                if isinstance(raw, dict):
                    current = snapshot(raw)
                    if current != previous:
                        record = {
                            "observed_at_unix_s": time.time(),
                            "state": current,
                        }
                        line = json.dumps(record, ensure_ascii=False)
                        print(line, flush=True)
                        log_file.write(line + "\n")
                        log_file.flush()
                        previous = current
                time.sleep(args.interval_s)
        except KeyboardInterrupt:
            print("[watch] 已停止。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
