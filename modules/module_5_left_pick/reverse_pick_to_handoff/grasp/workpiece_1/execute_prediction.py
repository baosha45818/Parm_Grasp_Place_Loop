#!/usr/bin/env python3
"""模块 5-1：消费 Module 4 的 BACK 判断，生成抓取位姿并执行左臂反抓。

该文件不控制云台、不采集图像、不运行 SAM2/ICP。Module 4-1 负责一次发布
正反和物体 ICP 位姿；抓取点/TCP 预测以及抓取执行属于本模块。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
LOOP_ROOT = MODULE_DIR.parents[2]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))

import numpy as np

from tools.execution_timing import ExecutionTiming
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.workpiece_1.grasp_prediction import predict as predict_grasp


IMPLEMENTATION_DIR = MODULE_DIR / "implementation"
NO_GRASP_EXIT_CODE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--grasp-config", type=Path, required=True)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"))
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1 * float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0")),
    )
    parser.add_argument("--approach-angles-deg", nargs="+", type=float, default=[90.0, 15.0, -15.0])
    # 工件 1 反面抓取：默认相对预测目标沿 body_base Z 方向下探 5 mm。
    parser.add_argument("--body-z-down-m", type=float, default=0.005)
    return parser.parse_args()


def load_prediction(prediction_file: Path) -> tuple[dict, Path, Path, Path]:
    payload = json.loads(prediction_file.read_text(encoding="utf-8"))
    event = payload.get("event")
    if event == "module_4_orientation_judged":
        orientation = payload.get("orientation")
        object_count = int(payload.get("object_count", 0))
    elif event == "module_4_1_buffer_state_estimated":
        orientation = payload.get("decision")
        object_count = len(payload.get("objects") or [])
        if str(payload.get("workpiece_id")) != "1":
            raise RuntimeError("工件1执行器不能消费其他工件的 Module 4-1 结果")
    else:
        raise RuntimeError("prediction-file 不是 Module 4-1 的正反+位姿结果")
    if orientation != "BACK" or object_count != 1:
        raise RuntimeError("Module 5-1 只接受 Module 4 判断出的唯一 BACK 工件")
    capture_dir = Path(payload["capture_dir"]).resolve()
    diagnostics_path = Path(payload["orientation_diagnostics"]).resolve()
    body_head_value = payload.get("body_head_calibration")
    if not body_head_value:
        raise RuntimeError("Module 4-1 结果缺少 body_head_calibration")
    body_head = Path(body_head_value).resolve()
    if not (capture_dir / "metadata.json").is_file() or not diagnostics_path.is_file() or not body_head.is_file():
        raise RuntimeError("Module 4 判断引用的 RGB-D 或 orientation_diagnostics.json 不存在")
    return payload, capture_dir, diagnostics_path, body_head


def move_left_to_ready(capture_dir: Path, planner_host: str, arm_host: str, speed: float) -> None:
    """在执行抓取前到固定预备位，并由 Arm 状态反馈确认到位。"""
    sys.path.insert(0, str(LOOP_ROOT))
    from loop_config import LoopConfig
    from parm_sdk import Arm
    from tools.planner_runtime import PlannerRuntime

    from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.profiles import load_reverse_ready_profile

    ready = load_reverse_ready_profile("1")
    target = list(ready["ready_joints_deg"])
    runtime = PlannerRuntime(
        LoopConfig(
            planner_host=planner_host,
            execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
            base_frame="body_base",
            left_group="left_arm",
            right_group="right_arm",
            left_tool_frame="left_eef",
            right_tool_frame="right_hand_transfer_end",
            max_velocity=speed,
            max_acceleration=speed,
            idle_poll_seconds=0.2,
            live=True,
            poses={},
        )
    )
    event = {"motion": "planner_rrtstar", "target_joint_angles_deg": target, "started_at": time.time()}
    try:
        runtime.open(required_frames={"body_base", "left_eef"})
        print(f"[module-5-1] Planner RRTStar 到反抓预备位 (left deg): {target}")
        runtime.move_left_joints(target, planner="rrtstar")
        deadline = time.monotonic() + float(ready["arrival_timeout_s"])
        tolerance = float(ready["arrival_tolerance_deg"])
        last_error = None
        with Arm(host=arm_host) as arm:
            while time.monotonic() < deadline:
                state = arm.get_state() or {}
                actual = ((state.get("state") or {}).get("left") or {}).get("position") or []
                if len(actual) == 7:
                    error = (np.asarray(actual, dtype=float) - target + 180.0) % 360.0 - 180.0
                    last_error = float(np.max(np.abs(error)))
                    if last_error <= tolerance:
                        event.update(status="reached", max_joint_error_deg=last_error)
                        print(f"[module-5-1] 反抓预备位已到位，最大关节误差={last_error:.3f}deg")
                        return
                time.sleep(0.05)
        raise RuntimeError(f"左臂反抓预备位未到达；最大关节误差={last_error}")
    except Exception as exc:
        event.update(status="failed", error=str(exc))
        raise
    finally:
        event["finished_at"] = time.time()
        (capture_dir / "motion_before_reverse_grasp.json").write_text(
            json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        runtime.close()


def execute_grasp(args: argparse.Namespace, grasp_json: Path, body_head: Path) -> None:
    command = [
        str(Path(sys.executable)),
        str(IMPLEMENTATION_DIR / "execute_confirmed_grasp.py"),
        "--grasp-json", str(grasp_json),
        "--body-head", str(body_head),
        "--planner-host", args.planner_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
        "--speed", str(args.speed),
        "--approach-angles-deg", *(str(value) for value in args.approach_angles_deg),
        "--body-z-down-m", str(args.body_z_down_m),
        "--execute",
    ]
    print("[module-5-1] 执行已确认反抓: " + " ".join(command))
    import subprocess
    try:
        subprocess.run(command, cwd=IMPLEMENTATION_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        # ``execute_confirmed_grasp.py`` uses 20 for the expected, recoverable
        # case where the left gripper is fully closed after the lift.  Preserve
        # that code so the parent can return to reverse-ready and retry.
        if exc.returncode == NO_GRASP_EXIT_CODE:
            raise SystemExit(NO_GRASP_EXIT_CODE) from exc
        raise


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("Module 5-1 真实反抓必须显式传入 --execute")
    _, capture_dir, diagnostics_path, body_head = load_prediction(args.prediction_file)
    timing = ExecutionTiming()
    timing_path = capture_dir / "reverse_grasp_timing.json"
    outcome = "FAILURE"
    try:
        timing.start("grasp_pose_prediction")
        grasp_json = predict_grasp(capture_dir, diagnostics_path, args.grasp_config.resolve())
        timing.finish("grasp_pose_prediction")
        if len(json.loads(grasp_json.read_text(encoding="utf-8"))) != 1:
            raise RuntimeError("Module 5-1 抓取位姿预测未得到唯一候选")

        timing.start("move_to_reverse_ready")
        move_left_to_ready(capture_dir, args.planner_host, args.arm_host, args.speed)
        timing.finish("move_to_reverse_ready")

        # The child process owns Planner preflight, pregrasp, descent, gripper
        # close and lift validation.  This parent-level stage intentionally
        # covers the complete real grasp transaction.
        timing.start("grasp_plan_execute_and_hold_check")
        execute_grasp(args, grasp_json, body_head)
        timing.finish("grasp_plan_execute_and_hold_check")
        outcome = "SUCCESS"
        return 0
    finally:
        timing.write(timing_path, outcome)
        print(f"REVERSE_GRASP_TIMING={timing_path}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
