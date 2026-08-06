#!/usr/bin/env python3
"""Buffer 分支入口：Module 4 判断后按需调用 Module 7，再由 Module 8-1 抓取。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
HEAD_MOTION = ROOT / "tool/move_head_gimbal.py"
RIGHT_ALIGN = ROOT / "modules/module_7_right_align/push_to_q1_pipeline.py"
RIGHT_GRASP = (
    ROOT
    / "modules/module_8_right_grasp/module_8_1_buffer_grasp/"
    "push_to_q1_grasp_pipeline.py"
)
BUFFER_JUDGE = (
    ROOT
    / "modules/module_4_head_judge/module_4_2_pose_estimation/buffer_judge.py"
)

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
    return parser.parse_args()


def run_stage(
    label: str,
    script: Path,
    args: argparse.Namespace,
    *extra: str,
    forward_execute: bool = True,
) -> None:
    module = {
        "module-7-observation-pose": "module_7_observation_pose",
        "head-buffer-observation": "head_buffer_observation",
        "module-4-2-buffer-judge": "module_4_2_buffer_judge",
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


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("右臂模块化实机流程必须显式传入 --execute")
    if not BUFFER_JUDGE.is_file():
        raise FileNotFoundError(f"Module 4-2 Buffer Judge 不存在：{BUFFER_JUDGE}")
    # 保持原流程的相机可见性：先让右臂到 3.json，再由 Module 4 观察。
    run_stage("module-7-observation-pose", RIGHT_ALIGN, args, "--prepare-observation-only")
    run_stage(
        "head-buffer-observation",
        HEAD_MOTION,
        args,
        "--host", "127.0.0.1", "--pitch", "38.5", "--yaw", "0.0",
    )
    with tempfile.TemporaryDirectory(prefix="parm_module_4_buffer_") as temporary_dir:
        decision_file = Path(temporary_dir) / "buffer_decision.json"
        run_stage(
            "module-4-2-buffer-judge",
            BUFFER_JUDGE,
            args,
            "--decision-file",
            str(decision_file),
            forward_execute=False,
        )
        decision = json.loads(decision_file.read_text(encoding="utf-8"))
        pose_file = str(decision["pose_file"])
        needs_align = decision.get("needs_align") is True
        print(
            "[module-4-2] "
            f"Q{decision['quadrant']}, yaw={float(decision['yaw_deg']):.2f}deg, "
            f"needs_align={needs_align}",
            flush=True,
        )
        if needs_align:
            print("[module-7] Module 4 判定不可直接抓取，执行摆正。", flush=True)
            run_stage(
                "module-7",
                RIGHT_ALIGN,
                args,
                "--skip-initial-move",
                "--initial-pose-file",
                pose_file,
            )
            # Module 7 已在 3.json 重新观察到可抓取 Q1；Module 8-1 再取一帧，
            # 作为真实抓取前的最终位姿确认。
            run_stage("module-8-1", RIGHT_GRASP, args, "--skip-initial-move")
        else:
            print("[module-8-1] Module 4 判定已是可抓取 Q1，跳过 Module 7。", flush=True)
            run_stage(
                "module-8-1",
                RIGHT_GRASP,
                args,
                "--skip-initial-move",
                "--initial-pose-file",
                pose_file,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
