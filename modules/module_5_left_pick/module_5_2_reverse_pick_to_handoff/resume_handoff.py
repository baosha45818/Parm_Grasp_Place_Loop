#!/usr/bin/env python3
"""只恢复 Module 5.2 的左手交接准备，绝不再次执行反抓。"""
from __future__ import annotations

import argparse
import subprocess

from .pipeline import HANDOFF_MAIN, PYTHON, ROOT, publish_object_held_failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", choices=("1", "2"), required=True)
    parser.add_argument("--mode", choices=("gzhu", "xiangxin"), required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", required=True)
    parser.add_argument("--gripper-host", required=True)
    parser.add_argument("--head-host", required=True)
    args = parser.parse_args()

    command = [
        str(PYTHON), str(HANDOFF_MAIN),
        "--workpiece", args.workpiece, "--mode", args.mode,
        "--handoff-result-file", args.result_file,
        "--planner-host", args.planner_host, "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host, "--head-host", args.head_host,
        "--driver-host", args.head_host,
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode == 0:
        return 0
    publish_object_held_failure(
        args.result_file,
        stage="reverse_handoff_prepare_resume",
        returncode=completed.returncode,
    )
    return 21


if __name__ == "__main__":
    raise SystemExit(main())
