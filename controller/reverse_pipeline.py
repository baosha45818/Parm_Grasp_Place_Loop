#!/usr/bin/env python3
"""把 Module 5.2 的反抓和左手交接准备封装为一个控制器调度单元。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def publish_object_held_failure(destination: str, *, stage: str, returncode: int) -> None:
    """记录反抓已完成但交接准备失败的真实现场，供总控安全恢复。"""
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "outcome": "handoff_prepare_failed_object_held",
        "stage": stage,
        "left_gripper_holds_object": True,
        "safe_state": "LEFT_HOLDS_WORKPIECE",
        "handoff_returncode": returncode,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-check-result", required=True)
    p.add_argument("--workpiece", choices=("1", "2"), required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--planner-host", default="127.0.0.1")
    p.add_argument("--arm-host", required=True)
    p.add_argument("--gripper-host", required=True)
    p.add_argument("--head-host", required=True)
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
    handoff = [
        str(PYTHON), str(ROOT / "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/handoff/main.py"),
        "--workpiece", args.workpiece, "--handoff-result-file", args.result_file,
        "--planner-host", args.planner_host, "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host, "--head-host", args.head_host,
        "--driver-host", args.head_host,
    ]
    handoff_result = subprocess.run(handoff, cwd=ROOT, check=False)
    if handoff_result.returncode == 0:
        return 0
    # 此处反抓阶段已经成功。即使后续观察复检返回 20，也绝不能把它当成
    # "可重抓"：左手的实际持件状态不明，必须固定为持件并由人工确认后仅重试交接。
    publish_object_held_failure(
        args.result_file,
        stage="reverse_handoff_prepare",
        returncode=handoff_result.returncode,
    )
    return 21


if __name__ == "__main__":
    raise SystemExit(main())
