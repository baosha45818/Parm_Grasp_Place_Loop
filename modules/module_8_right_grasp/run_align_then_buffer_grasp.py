#!/usr/bin/env python3
"""Buffer 分支入口：Module 7 检查/摆正，Module 8-1 抓取，必要时回到 Module 7。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
RIGHT_ALIGN = ROOT / "modules/module_7_right_align/push_to_q1_pipeline.py"
RIGHT_GRASP = (
    ROOT
    / "modules/module_8_right_grasp/module_8_1_buffer_grasp/"
    "push_to_q1_grasp_pipeline.py"
)
# Module 8-1 uses this exit code only when another Module-7 reorientation is
# safe to try. Other non-zero exits remain fatal.
MODULE_7_RETRY_EXIT_CODE = 23
DEFAULT_MAX_ALIGN_GRASP_CYCLES = 3

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_types import ResultStatus
from modules.module_status import (
    report as report_module_status,
    report_exception as report_module_exception,
)
from tool.workpiece_timer import module_timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument(
        "--motion-speed-scale",
        type=float,
        default=2.5 * float(os.environ.get("PARM_RIGHT_SPEED_SCALE", "1.0")),
    )
    parser.add_argument(
        "--max-align-grasp-cycles",
        type=int,
        default=DEFAULT_MAX_ALIGN_GRASP_CYCLES,
        help="maximum Module-7 -> Module-8-1 recovery cycles (default: 3)",
    )
    return parser.parse_args()


def run_stage(
    label: str,
    script: Path,
    args: argparse.Namespace,
    *extra: str,
    forward_execute: bool = True,
    recoverable_exit_codes: frozenset[int] = frozenset(),
) -> int:
    module = {
        "module-7": "module_7_right_align",
        "module-8-1": "module_8_1_buffer_grasp",
    }.get(label, label.replace("-", "_"))
    command = [str(PYTHON), str(script)]
    if script in {RIGHT_ALIGN, RIGHT_GRASP}:
        command.extend(["--motion-speed-scale", f"{args.motion_speed_scale:g}"])
    if args.execute and forward_execute:
        command.append("--execute")
    command.extend(extra)
    environment = os.environ.copy()
    environment["PARM_PLANNER_HOST"] = args.planner_host
    environment["PARM_ARM_HOST"] = args.arm_host
    environment["PARM_GRIPPER_HOST"] = args.gripper_host
    report_module_status(module, ResultStatus.WAIT, f"开始：{label}")
    print(f"[{label}] " + " ".join(command), flush=True)
    try:
        with module_timer(module, detail=label):
            subprocess.run(
                command,
                cwd=script.parent,
                env=environment,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        if exc.returncode in recoverable_exit_codes:
            report_module_status(
                module,
                ResultStatus.WAIT,
                f"{label}请求 Module 7 重新摆正（exit={exc.returncode}）",
                reason_code="REORIENTATION_REQUIRED",
            )
            return exc.returncode
        report_module_exception(
            module,
            exc,
            action="停止右臂当前子链并返回总控，由总控 cancel 两臂任务",
            target_module="initial_state",
            recovery_level="SAFE_STOP",
            reason_code=f"EXIT_{exc.returncode}",
        )
        raise
    except KeyboardInterrupt:
        report_module_status(
            module,
            ResultStatus.ABORTED,
            f"{label}收到中断，停止当前模块",
            reason_code="INTERRUPTED",
        )
        raise
    except BaseException as exc:
        report_module_exception(
            module,
            exc,
            action="停止右臂当前子链并返回总控，由总控 cancel 两臂任务",
            target_module="initial_state",
            recovery_level="SAFE_STOP",
        )
        raise
    else:
        report_module_status(module, ResultStatus.SUCCESS, f"完成：{label}")
        return 0


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("右臂模块化实机流程必须显式传入 --execute")
    if args.max_align_grasp_cycles < 1:
        raise ValueError("--max-align-grasp-cycles 必须至少为 1")

    # Module 7 owns the Module-4-2 observation plus any necessary
    # reorientation, so it is always the first stage after buffer placement.
    for cycle in range(1, args.max_align_grasp_cycles + 1):
        print(
            f"[buffer] Module 7 -> 8-1 cycle {cycle}/{args.max_align_grasp_cycles}",
            flush=True,
        )
        run_stage("module-7", RIGHT_ALIGN, args)
        grasp_exit = run_stage(
            "module-8-1",
            RIGHT_GRASP,
            args,
            "--skip-initial-move",
            recoverable_exit_codes=frozenset({MODULE_7_RETRY_EXIT_CODE}),
        )
        if grasp_exit == 0:
            return 0
        if cycle == args.max_align_grasp_cycles:
            raise RuntimeError(
                "Module 8-1 连续请求重新摆正，已达到 "
                f"--max-align-grasp-cycles={args.max_align_grasp_cycles}"
            )
        print("[module-8-1] 请求 Module 7 重新检查并摆正后再抓取。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
