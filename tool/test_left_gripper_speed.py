#!/usr/bin/env python3
"""测试新左夹爪的开度、力度和速度；仅在传入 --execute 时发送命令。"""

from __future__ import annotations

import argparse
import json
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="夹爪驱动主机")
    parser.add_argument("--cmd-port", type=int, default=7701)
    parser.add_argument("--status-port", type=int, default=7702)
    parser.add_argument(
        "--position",
        type=float,
        required=True,
        help="目标开度百分比：0=全闭，100=全开",
    )
    parser.add_argument(
        "--force",
        type=float,
        default=100.0,
        help="左夹爪力度百分比，范围 20~100（默认 100）",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=20.0,
        help="左夹爪速度，范围 1~100（默认 20）",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true", help="显式允许发送夹爪命令")
    return parser.parse_args()


def sync_terminal_state(response: dict[str, Any]) -> dict[str, Any] | None:
    """从新驱动 set_targets 的同步响应提取 left_gripper 终态。"""
    payload = response.get("data") or response
    device = (payload.get("devices") or {}).get("left_gripper")
    if not isinstance(device, dict):
        return None
    interfaces = device.get("state_interfaces") or {}
    return {
        "device_state": device.get("device_state"),
        "position": interfaces.get("gripper/position"),
        "force": interfaces.get("gripper/force"),
        "speed": interfaces.get("gripper/speed"),
    }


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("这是实机夹爪测试；请显式添加 --execute")

    from parm_sdk import Gripper

    print("=== 新左夹爪速度/力度测试 ===")
    print(
        f"host={args.host}:{args.cmd_port}, device=left_gripper, hand=1, "
        f"position={args.position:.1f}%, force={args.force:.1f}%, speed={args.speed:.1f}",
        flush=True,
    )
    with Gripper(
        host=args.host,
        cmd_port=args.cmd_port,
        status_port=args.status_port,
    ) as gripper:
        print("命令前 PUB 状态:", gripper.get_state().get(1), flush=True)
        response = gripper.move(
            1,
            args.position,
            force=args.force,
            speed=args.speed,
            mode="sync",
            preempt=False,
            timeout=args.timeout,
        )
        print("同步响应终态:", json.dumps(sync_terminal_state(response), ensure_ascii=False), flush=True)
        print("命令后 PUB 状态:", gripper.get_state().get(1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
