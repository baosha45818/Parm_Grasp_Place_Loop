#!/usr/bin/env python3
"""读取左夹爪当前开度，并在完全打开与完全闭合之间切换一次。"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys


LOOP_ROOT = Path(__file__).resolve().parents[1]
NEW_SDK_WHEEL = next(
    iter(sorted((LOOP_ROOT / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True)),
    None,
)
SDK_PYTHON = LOOP_ROOT / ".venv/bin/python"


def _ensure_compatible_python() -> None:
    """pyzmq 含 Python 3.10 扩展，其他解释器需切换到现有 SDK 环境。"""
    if Path(sys.executable).resolve() == SDK_PYTHON.resolve():
        return
    if not SDK_PYTHON.is_file():
        raise RuntimeError(f"找不到兼容 PARM SDK 的 Python: {SDK_PYTHON}")
    os.execv(str(SDK_PYTHON), [str(SDK_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def _configure_sdk_import() -> None:
    """固定使用新版 SDK，并补充其 pyzmq 等运行依赖。"""
    if not NEW_SDK_WHEEL.is_file():
        raise RuntimeError(f"找不到新版 PARM SDK wheel: {NEW_SDK_WHEEL}")
    sys.path.insert(0, str(NEW_SDK_WHEEL))


def parse_args() -> argparse.Namespace:
    """读取夹爪切换脚本的运行参数。"""
    parser = argparse.ArgumentParser(description="根据当前状态切换 PARM 左夹爪开/关")
    parser.add_argument("--host", default="127.0.0.1", help="夹爪服务主机，默认本机")
    parser.add_argument("--hand", type=int, default=1, choices=(1,), help="固定为左手 1")
    parser.add_argument("--open-position", type=float, default=100.0, help="打开目标开度")
    parser.add_argument("--close-position", type=float, default=0.0, help="闭合目标开度")
    parser.add_argument(
        "--toggle-threshold",
        type=float,
        default=50.0,
        help="当前开度大于等于此值时视为打开并执行闭合",
    )
    parser.add_argument("--force", type=float, default=100.0, help="夹持力度百分比，默认最大 100%%")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="夹爪速度百分比；AG 夹爪不支持时不要传",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="同步动作超时秒数")
    return parser.parse_args()


def main() -> None:
    """读取开度后执行一次同步开关切换。"""
    _ensure_compatible_python()
    args = parse_args()
    _validate_args(args)
    _configure_sdk_import()
    from parm_sdk import Gripper

    with Gripper(host=args.host) as gripper:
        hand_state = gripper.get_state().get(args.hand)
        if hand_state is None:
            raise RuntimeError(f"左夹爪 hand={args.hand} 没有状态帧或未连接")
        if hand_state.get("initialized") is not True:
            raise RuntimeError("左夹爪尚未初始化；请先用独立初始化流程完成回零")

        current = float(hand_state.get("position"))
        if not math.isfinite(current):
            raise RuntimeError(f"左夹爪当前开度无效: {hand_state.get('position')!r}")
        if current >= args.toggle_threshold:
            target = args.close_position
            action = "闭合"
        else:
            target = args.open_position
            action = "打开"

        print(
            f"左夹爪当前开度={current:.1f}%，阈值={args.toggle_threshold:.1f}%，"
            f"本次执行：{action}到 {target:.1f}%"
        )
        move_args: dict[str, float | str | bool] = {
            "force": args.force,
            "mode": "sync",
            "preempt": True,
            "timeout": args.timeout,
        }
        if args.speed is not None:
            move_args["speed"] = args.speed
        response = gripper.move(args.hand, target, **move_args)
        final_position = (response.get("data") or {}).get("position")
        print(f"左夹爪{action}完成，最终开度={final_position}%")


def _validate_args(args: argparse.Namespace) -> None:
    """在发送夹爪指令前校验所有百分比和超时参数。"""
    for name in ("open_position", "close_position", "toggle_threshold"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须在 [0, 100] 范围内")
    if not math.isfinite(args.force) or not 20.0 <= args.force <= 100.0:
        raise ValueError("--force 必须在 [20, 100] 范围内")
    if args.speed is not None and (
        not math.isfinite(args.speed) or not 1.0 <= args.speed <= 100.0
    ):
        raise ValueError("--speed 必须在 [1, 100] 范围内")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout 必须为正数")


if __name__ == "__main__":
    main()
