#!/usr/bin/env python3
"""确认两只 Key-1 踏板的左右映射（只读；--apply 才改驱动配置）。

两只踏板的 VID/PID/序列号相同，换电脑或换 USB 口后只能通过“实际踩一次”
确定哪只应映射到左臂 arm 1、哪只应映射到右臂 arm 2。

运行：
    ./.venv/bin/python tools/confirm_foot_pedals.py

确认无误后，如需自动更新 parm_sdk_1.0.0 的驱动配置：
    ./.venv/bin/python tools/confirm_foot_pedals.py --apply
"""

from __future__ import annotations

import argparse
from datetime import datetime
import glob
import os
from pathlib import Path
import re
import select
import shutil
import struct
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER_SCRIPT = (
    PROJECT_ROOT.parent / "parm_sdk_1.0.0" / "driver_servers" / "run_activate_teach.sh"
)
VENDOR_ID = "2704"
PRODUCT_ID = "2018"
EVENT = struct.Struct("@llHHI")  # Linux input_event on the project x86_64 hosts.
EV_KEY = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="确认后自动更新 run_activate_teach.sh，并保留 .bak 备份")
    parser.add_argument("--timeout", type=float, default=20.0, help="每只踏板的等待秒数，默认 20")
    parser.add_argument("--driver-script", type=Path, default=DEFAULT_DRIVER_SCRIPT)
    return parser.parse_args()


def properties(device: str) -> dict[str, str]:
    completed = subprocess.run(
        ["udevadm", "info", "--query=property", "--name", device],
        text=True,
        capture_output=True,
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def by_path(event_device: str) -> str:
    expected = os.path.realpath(event_device)
    matches = [
        path for path in glob.glob("/dev/input/by-path/*event*")
        if os.path.realpath(path) == expected
    ]
    return matches[0] if matches else event_device


def physical_pedal_id(device: str) -> str:
    """合并同一只踏板暴露出来的 keyboard/consumer-control 多个 event 节点。"""
    value = properties(device).get("ID_PATH", "")
    # 例：pci-...-usb-0:5.4:1.0 与 ...:1.1 属于同一个 USB 踏板。
    if value:
        return re.sub(r":\d+\.\d+$", "", value)
    return os.path.realpath(f"/sys/class/input/{Path(device).name}/device")


def find_key1_pedals() -> dict[str, list[str]]:
    pedals: dict[str, list[str]] = {}
    for device in sorted(glob.glob("/dev/input/event*")):
        info = properties(device)
        if info.get("ID_VENDOR_ID") == VENDOR_ID and info.get("ID_MODEL_ID") == PRODUCT_ID:
            pedals.setdefault(physical_pedal_id(device), []).append(device)
    return pedals


def wait_for_press(candidates: dict[str, list[str]], label: str, timeout_s: float) -> tuple[str, str]:
    print(f"\n请在 {timeout_s:.0f} 秒内只踩一下【{label}】。", flush=True)
    handles = {
        os.open(device, os.O_RDONLY | os.O_NONBLOCK): (pedal_id, device)
        for pedal_id, devices in candidates.items()
        for device in devices
    }
    buffers = {fd: b"" for fd in handles}
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready, _, _ = select.select(list(handles), [], [], min(0.5, deadline - time.monotonic()))
            for fd in ready:
                buffers[fd] += os.read(fd, EVENT.size * 16)
                while len(buffers[fd]) >= EVENT.size:
                    raw, buffers[fd] = buffers[fd][:EVENT.size], buffers[fd][EVENT.size:]
                    _, _, event_type, _, value = EVENT.unpack(raw)
                    if event_type == EV_KEY and value == 1:
                        pedal_id, device = handles[fd]
                        print(f"[INFO] 已识别 {label}: {device} -> {by_path(device)}", flush=True)
                        return pedal_id, device
    finally:
        for fd in handles:
            os.close(fd)
    raise TimeoutError(f"等待 {label} 超时；请检查踏板连接、udev 规则和权限")


def update_driver_script(path: Path, left: str, right: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"找不到踏板驱动配置：{path}")
    text = path.read_text(encoding="utf-8")
    # 仅替换每一行的设备值。必须以空白切断旧路径；不能用错误转义的字符类，
    # 否则会在路径中的字母 s 处截断，并把旧路径尾部拼到新路径后。
    updated = re.sub(
        r"(?m)^(\s*--pedal arm=1:device=)\S+",
        lambda match: f"{match.group(1)}{left}",
        text,
    )
    updated = re.sub(
        r"(?m)^(\s*--pedal arm=2:device=)\S+",
        lambda match: f"{match.group(1)}{right}",
        updated,
    )
    if updated == text:
        raise RuntimeError("未找到 arm=1 / arm=2 的踏板配置行，未修改文件")
    backup = path.with_name(path.name + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    return backup


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        # 读取 /dev/input/event* 常需 root 或 input 组权限；不使用 sudo 密码保存。
        os.execvp("sudo", ["sudo", sys.executable, *sys.argv])

    if shutil.which("udevadm") is None:
        print("[FAIL] 未找到 udevadm；请安装/启用 systemd-udev。", file=sys.stderr)
        return 2
    pedals = find_key1_pedals()
    if len(pedals) != 2:
        print(f"[FAIL] 找到 {len(pedals)} 个 Key-1 物理踏板（期望 2）：{pedals}", file=sys.stderr)
        print("请检查 USB 连接；设备特征应为 VID=2704、PID=2018。", file=sys.stderr)
        return 2

    print("[INFO] 找到两只 Key-1 踏板：", flush=True)
    for pedal_id, devices in pedals.items():
        print(f"  {pedal_id}", flush=True)
        for device in devices:
            print(f"    {device}  ->  {by_path(device)}", flush=True)
    left_pedal, left_event = wait_for_press(pedals, "左踏板（arm 1）", args.timeout)
    right_candidates = {key: value for key, value in pedals.items() if key != left_pedal}
    right_pedal, right_event = wait_for_press(right_candidates, "右踏板（arm 2）", args.timeout)
    left, right = by_path(left_event), by_path(right_event)
    print("\n[PASS] 踏板映射确认完成：", flush=True)
    print(f"  arm 1 / 左踏板: {left}", flush=True)
    print(f"  arm 2 / 右踏板: {right}", flush=True)
    if args.apply:
        backup = update_driver_script(args.driver_script.resolve(), left, right)
        print(f"[PASS] 已更新：{args.driver_script.resolve()}\n       备份：{backup}", flush=True)
    else:
        print("[INFO] 未修改配置。确认后加 --apply 可自动写入驱动配置。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
