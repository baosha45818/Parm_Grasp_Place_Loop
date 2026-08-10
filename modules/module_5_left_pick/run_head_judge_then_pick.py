#!/usr/bin/env python3
"""模块 4→5 适配入口：确认桌面反面工件后执行左臂反抓。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
HEAD_MOTION = ROOT / "tool/move_head_gimbal.py"
RGBD_CAPTURE = ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/implementation/capture_head_rgbd.py"
BODY_HEAD = ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/config/body_head_calibration.json"
REVERSE_PREDICTION = (
    ROOT
    / "modules/module_4_head_judge/module_4_1_reverse_grasp_prediction/main.py"
)
LEFT_PICK_EXECUTION = (
    ROOT
    / "modules/module_5_left_pick/module_5_1_reverse_grasp/execute_prediction.py"
)
OUTPUT_ROOT = ROOT / "output/module_5_left_pick/reverse_grasp"
WORKPIECE_2_PIPELINE = (
    ROOT
    / "modules/module_5_left_pick/module_5_1_reverse_grasp/"
    "workpiece_2_pipeline/run_full_reverse_grasp.sh"
)
NO_GRASP_EXIT_CODE = 20

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tool") not in sys.path:
    sys.path.insert(0, str(ROOT / "tool"))

from loop_types import ResultStatus
from modules.module_status import (
    report as report_module_status,
    report_exception as report_module_exception,
)
from tool.workpiece_timer import module_timer
from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile


def run_stage(
    command: list[str],
    *,
    module: str,
    cwd: Path,
    recoverable: bool = False,
    recoverable_return_codes: set[int] | None = None,
) -> None:
    """执行一个适配器子模块，并让跨进程统计拿到准确起止时间。"""
    report_module_status(module, ResultStatus.WAIT, f"开始：{module}")
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        if exc.returncode in (recoverable_return_codes or set()):
            report_module_status(
                module,
                ResultStatus.RETRY,
                "左夹爪未确认抓持；回反抓预备位后重新观察、预测并抓取",
                reason_code="LEFT_REVERSE_GRASP_RETRY",
            )
        elif recoverable:
            report_module_status(
                module,
                ResultStatus.RETRY,
                f"{module} 未得到唯一有效候选，保持安全位后重拍",
                reason_code="NO_UNIQUE_REVERSE_GRASP",
            )
        else:
            report_module_exception(
                module,
                exc,
                action="终止当前反抓子链并交由总控选择局部回退或安全停机",
                target_module="module_5_1",
                recovery_level="PARENT_DECISION",
                reason_code=f"EXIT_{exc.returncode}",
            )
        raise
    except KeyboardInterrupt:
        report_module_status(
            module,
            ResultStatus.ABORTED,
            f"{module}收到中断",
            reason_code="INTERRUPTED",
        )
        raise
    except BaseException as exc:
        report_module_exception(
            module,
            exc,
            action="终止当前反抓子链并交由总控安全停机",
            target_module="initial_state",
            recovery_level="SAFE_STOP",
            traceback_text="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )
        raise
    else:
        report_module_status(module, ResultStatus.SUCCESS, f"完成：{module}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument(
        "--head-host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--prediction-retry-seconds",
        type=float,
        default=2.0,
        help="反抓候选不唯一或非BACK时，下一次头部重拍前的等待时间",
    )
    parser.add_argument(
        "--max-prediction-attempts",
        type=int,
        default=0,
        help="反抓预测最大次数；0表示持续重试直到得到唯一BACK候选",
    )
    parser.add_argument(
        "--max-reverse-grasp-attempts",
        type=int,
        default=None,
        help="闭夹爪后的反抓尝试上限；默认读取工件位置配置，当前为5次",
    )
    return parser.parse_args()


def reopen_and_return_to_reverse_ready(args: argparse.Namespace, workpiece: str) -> None:
    """失败后先张开新左夹爪，再回反抓预备位，下一轮才重新观察。"""
    from loop_config import LoopConfig
    from parm_sdk import Arm, Gripper
    from planner_runtime import PlannerRuntime

    ready = load_workpiece_handoff_profile(workpiece)["left_reverse_grasp"]
    target = list(ready["ready_joints_deg"])
    print("[reverse-flow] 本次未夹到工件：同步张开新左夹爪。", flush=True)
    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        response = gripper.move(1, 100.0, force=100.0, speed=80.0, mode="sync", preempt=False, timeout=60.0)
    payload = response.get("data") or response
    device = (payload.get("devices") or {}).get("left_gripper") or {}
    device_state = str(device.get("device_state", "")).strip().lower()
    if response.get("status") != "success" or device_state not in {"arrived", "gripping"}:
        raise RuntimeError(f"左夹爪失败后张开失败: {response}")

    runtime = PlannerRuntime(LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base", left_group="left_arm", right_group="right_arm",
        left_tool_frame="left_eef", right_tool_frame="right_hand_transfer_end",
        max_velocity=0.2, max_acceleration=0.2, idle_poll_seconds=0.2,
        live=True, poses={},
    ))
    try:
        runtime.open(required_frames={"body_base", "left_eef"})
        print(f"[reverse-flow] Planner RRTStar 回左臂反抓预备位: {target}", flush=True)
        runtime.move_left_joints(target, planner="rrtstar")
        deadline = time.monotonic() + float(ready["arrival_timeout_s"])
        tolerance = float(ready["arrival_tolerance_deg"])
        with Arm(host=args.arm_host) as arm:
            while time.monotonic() < deadline:
                state = arm.get_state() or {}
                actual = ((state.get("state") or {}).get("left") or {}).get("position") or []
                if len(actual) == 7:
                    error = (np.asarray(actual, dtype=float) - target + 180.0) % 360.0 - 180.0
                    if float(np.max(np.abs(error))) <= tolerance:
                        print("[reverse-flow] 左臂已回反抓预备位；下一轮将重新头部观察、预测并抓取。", flush=True)
                        return
                time.sleep(0.05)
        raise RuntimeError("未抓到工件后左臂未能回到反抓预备位")
    finally:
        runtime.close()


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("模块 5 实机反抓必须显式传入 --execute")
    if args.prediction_retry_seconds < 0.0:
        raise ValueError("--prediction-retry-seconds 不能小于 0")
    if args.max_prediction_attempts < 0:
        raise ValueError("--max-prediction-attempts 不能小于 0")

    # 工件2的反抓预测和执行已经是独立的 Module 4-1 -> Module 5-1
    # 流水线。只在总控显式选择工件2时切换；工件1继续执行下方原逻辑。
    workpiece = os.environ.get("PARM_LOOP_WORKPIECE", "1")
    reverse_config = load_workpiece_handoff_profile(workpiece)["left_reverse_grasp"]
    max_grasp_attempts = (
        int(reverse_config["max_grasp_attempts"])
        if args.max_reverse_grasp_attempts is None
        else int(args.max_reverse_grasp_attempts)
    )
    if max_grasp_attempts < 1:
        raise ValueError("--max-reverse-grasp-attempts 必须不小于 1")
    if workpiece == "2":
        if not WORKPIECE_2_PIPELINE.is_file():
            raise FileNotFoundError(
                f"工件2反抓流水线入口不存在: {WORKPIECE_2_PIPELINE}"
            )
        print(
            "[reverse-flow] workpiece=2：执行 Module 4-1 固定反面预测与 "
            f"Module 5-1 凸耳反抓流水线；最多尝试{max_grasp_attempts}次。",
            flush=True,
        )
        for grasp_attempt in range(1, max_grasp_attempts + 1):
            run_name = (
                f"loop_workpiece2_reverse_{time.strftime('%Y%m%d_%H%M%S')}_"
                f"{time.time_ns() % 1_000_000_000:09d}"
            )
            command = ["bash", str(WORKPIECE_2_PIPELINE), "--run-name", run_name, "--execute"]
            print(f"[reverse-flow] 工件2反抓尝试 {grasp_attempt}/{max_grasp_attempts}", flush=True)
            with module_timer("module_4_1_reverse_grasp_prediction"):
                report_module_status(
                    "module_4_1_reverse_grasp_prediction",
                    ResultStatus.WAIT,
                    "工件2头部观察、SAM2、固定反面ICP与凸耳抓取点预测",
                )
                try:
                    subprocess.run(command, cwd=ROOT, check=True)
                except subprocess.CalledProcessError as exc:
                    if exc.returncode != NO_GRASP_EXIT_CODE:
                        report_module_exception(
                            "module_4_1_reverse_grasp_prediction",
                            exc,
                            action="工件2反抓流水线失败，保持现场并交由总控重试或安全停机",
                            target_module="module_5_1_reverse_grasp",
                            recovery_level="PARENT_DECISION",
                            reason_code=f"WORKPIECE_2_PIPELINE_EXIT_{exc.returncode}",
                        )
                        raise
                    report_module_status(
                        "module_4_1_reverse_grasp_prediction",
                        ResultStatus.RETRY,
                        "左夹爪未夹到单件或抬升后重量未确认SINGLE；回预备位重新观察",
                        reason_code="LEFT_REVERSE_GRASP_RETRY",
                    )
                    reopen_and_return_to_reverse_ready(args, workpiece)
                    if grasp_attempt == max_grasp_attempts:
                        raise RuntimeError(
                            f"工件2反抓已尝试{max_grasp_attempts}次，仍未确认夹到单件"
                        ) from exc
                    continue
                else:
                    report_module_status(
                        "module_4_1_reverse_grasp_prediction",
                        ResultStatus.SUCCESS,
                        "工件2预测与 Module 5-1 抓取抬升、重量确认均完成",
                    )
                    return 0
    if workpiece != "1":
        raise ValueError(f"PARM_LOOP_WORKPIECE 只能是 1 或 2，当前为 {workpiece!r}")

    from modules.module_4_head_judge import (
        ObjectOrientation,
        inspect_tabletop_orientation,
    )

    run_dir = OUTPUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[reverse-flow] 本次输出目录: {run_dir}", flush=True)
    calibration = json.loads(BODY_HEAD.read_text(encoding="utf-8"))
    pose = calibration["head_pose_deg"]
    head_command = [
        str(PYTHON), str(HEAD_MOTION), "--execute", "--host", args.head_host,
        "--pitch", str(pose["pitch"]), "--yaw", str(pose["yaw"]),
    ]
    print("[reverse-flow] " + " ".join(head_command), flush=True)
    run_stage(
        head_command,
        module="head_reverse_observation_pose",
        cwd=ROOT,
    )

    attempt = 0
    while True:
        attempt += 1
        attempt_dir = run_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        prediction_file = attempt_dir / "module_4_1_prediction.json"
        capture_command = [
            str(PYTHON), str(RGBD_CAPTURE), "--output-dir", str(attempt_dir),
        ]
        print(
            f"[reverse-flow] 反抓预测尝试 {attempt}: " + " ".join(capture_command),
            flush=True,
        )
        with module_timer("module_4_1_reverse_grasp_prediction"):
            run_stage(
                capture_command,
                module="module_4_1_rgbd_capture",
                cwd=RGBD_CAPTURE.parent,
            )
        capture_candidates = sorted(
            path for path in attempt_dir.glob("head_rgbd_*")
            if (path / "metadata.json").is_file()
        )
        if len(capture_candidates) != 1:
            raise RuntimeError(
                "头部 RGB-D 采集完成后未找到唯一有效目录: "
                f"root={attempt_dir}, candidates={capture_candidates}"
            )
        capture_dir = capture_candidates[0]
        print(f"[reverse-flow] 实际 RGB-D 目录: {capture_dir}", flush=True)
        prediction_command = [
            str(PYTHON), str(REVERSE_PREDICTION), "--execute",
            "--prediction-file", str(prediction_file),
            "--capture-dir", str(capture_dir),
        ]
        print("[module-4-1] " + " ".join(prediction_command), flush=True)
        try:
            with module_timer("module_4_1_reverse_grasp_prediction"):
                run_stage(
                    prediction_command,
                    module="module_4_1_reverse_grasp_prediction",
                    cwd=REVERSE_PREDICTION.parent,
                    recoverable=True,
                )
        except subprocess.CalledProcessError:
            grasp_json = capture_dir / "sam2_masks/grasp_poses.json"
            candidate_count = None
            if grasp_json.is_file():
                candidate_count = len(json.loads(grasp_json.read_text(encoding="utf-8")))
            if candidate_count is None:
                raise
            print(
                f"[reverse-flow] 反抓候选数={candidate_count}，需要唯一候选；"
                "保持左臂安全位后重拍。",
                flush=True,
            )
        else:
            prediction = json.loads(prediction_file.read_text(encoding="utf-8"))
            capture_dir = Path(prediction["capture_dir"])
            print("[module-4] 复用 Module 4-1 采集结果判断桌面工件正反状态", flush=True)
            report_module_status(
                "module_4_head_judge",
                ResultStatus.WAIT,
                "开始桌面工件正反状态判断",
            )
            try:
                with module_timer("module_4_head_judge"):
                    result = inspect_tabletop_orientation(capture_dir)
            except BaseException as exc:
                report_module_exception(
                    "module_4_head_judge",
                    exc,
                    action="保持左臂安全位，停止本次判断并交由上层处理",
                    target_module="module_4_1",
                    recovery_level="PARTIAL_ROLLBACK",
                    traceback_text="".join(
                        traceback.format_exception(
                            type(exc), exc, exc.__traceback__
                        )
                    ),
                )
                raise
            else:
                report_module_status(
                    "module_4_head_judge",
                    ResultStatus.SUCCESS,
                    f"方向判断完成：{result.orientation.value}",
                )
            print(
                f"[module-4] orientation={result.orientation.value}, "
                f"objects={len(result.objects)}, report={result.report_path}",
                flush=True,
            )
            if result.orientation is ObjectOrientation.BACK:
                break
            print(
                f"[reverse-flow] 当前方向={result.orientation.value}，需要BACK；"
                "保持左臂安全位后重拍。",
                flush=True,
            )

        if (
            args.max_prediction_attempts
            and attempt >= args.max_prediction_attempts
        ):
            raise RuntimeError(
                "反抓预测已达到最大重试次数，仍未得到唯一BACK候选："
                f"attempts={attempt}"
            )
        if args.prediction_retry_seconds > 0.0:
            time.sleep(args.prediction_retry_seconds)

    command = [
        str(PYTHON), str(LEFT_PICK_EXECUTION), "--prediction-file", str(prediction_file),
        "--execute", "--planner-host", args.planner_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
    ]
    print("[module-5-1] " + " ".join(command), flush=True)
    try:
        with module_timer("module_5_1_reverse_grasp"):
            run_stage(
                command,
                module="module_5_1_reverse_grasp",
                cwd=LEFT_PICK_EXECUTION.parent,
                recoverable_return_codes={NO_GRASP_EXIT_CODE},
            )
    except subprocess.CalledProcessError as exc:
        if exc.returncode != NO_GRASP_EXIT_CODE:
            raise
        reopen_and_return_to_reverse_ready(args, workpiece)
        current_attempt = int(os.environ.get("PARM_REVERSE_GRASP_ATTEMPT", "1"))
        if current_attempt >= max_grasp_attempts:
            raise RuntimeError(
                f"工件1反抓已尝试{max_grasp_attempts}次，仍未确认夹到单件"
            ) from exc
        next_attempt = current_attempt + 1
        print(
            f"[reverse-flow] 工件1反抓尝试 {current_attempt}/{max_grasp_attempts}失败；"
            f"开始第{next_attempt}次重新观察、预测和抓取。",
            flush=True,
        )
        os.environ["PARM_REVERSE_GRASP_ATTEMPT"] = str(next_attempt)
        return main()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
