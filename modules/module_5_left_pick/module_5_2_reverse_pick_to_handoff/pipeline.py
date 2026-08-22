#!/usr/bin/env python3
"""执行 Module 5.2：反抓后进行左手交接准备。

这是反面工件的组合模块入口。反抓成功后若交接准备失败，会明确写入
“左手仍持件”的安全状态；总控据此只恢复交接，而不会再次反抓。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv/bin/python"
MODULE = "modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.main"
HANDOFF_MAIN = ROOT / "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/handoff/main.py"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-check-result", required=True)
    parser.add_argument("--workpiece", choices=("1", "2"), required=True)
    parser.add_argument("--mode", choices=("gzhu", "xiangxin"), required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", required=True)
    parser.add_argument("--gripper-host", required=True)
    parser.add_argument("--head-host", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        return 2

    reverse = [
        str(PYTHON), "-m", MODULE,
        "--buffer-check-result", args.buffer_check_result, "--mode", args.mode,
        "--planner-host", args.planner_host, "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host, "--execute",
    ]
    # 20 是 Module 5 约定的“未抓稳，可重新观察后再抓”；必须原样交给总控。
    reverse_result = subprocess.run(reverse, cwd=ROOT, check=False)
    if reverse_result.returncode != 0:
        return reverse_result.returncode

    handoff = [
        str(PYTHON), str(HANDOFF_MAIN),
        "--workpiece", args.workpiece, "--mode", args.mode,
        "--handoff-result-file", args.result_file,
        "--planner-host", args.planner_host, "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host, "--head-host", args.head_host,
        "--driver-host", args.head_host,
    ]
    handoff_result = subprocess.run(handoff, cwd=ROOT, check=False)
    if handoff_result.returncode == 0:
        return 0

    # 反抓已经成功；不能再把后续失败解释成可安全重抓。
    publish_object_held_failure(
        args.result_file,
        stage="reverse_handoff_prepare",
        returncode=handoff_result.returncode,
    )
    return 21


if __name__ == "__main__":
    raise SystemExit(main())
