#!/usr/bin/env python3
"""把 Module 5.2 的反抓和左手交接准备封装为一个控制器调度单元。"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-check-result", required=True)
    p.add_argument("--workpiece", choices=("1", "2"), required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--planner-host", default="127.0.0.1")
    p.add_argument("--arm-host", required=True)
    p.add_argument("--gripper-host", required=True)
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    if not args.live:
        return 2
    reverse = [str(PYTHON), str(ROOT / "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/main.py"), "--buffer-check-result", args.buffer_check_result, "--planner-host", args.planner_host, "--arm-host", args.arm_host, "--gripper-host", args.gripper_host, "--execute"]
    # 20 是 Module 5 约定的「未抓稳，可重新观察后再抓」；不能压成 1，
    # 否则控制器会把本可恢复的失败错误地升级为故障。
    reverse_result = subprocess.run(reverse, cwd=ROOT, check=False)
    if reverse_result.returncode != 0:
        return reverse_result.returncode
    handoff = [str(PYTHON), str(ROOT / "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/handoff/main.py"), "--workpiece", args.workpiece, "--handoff-result-file", args.result_file]
    return subprocess.run(handoff, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
