#!/usr/bin/env python3
"""Module 5 单独测试入口：自动重抓或人工已持件后的预放置。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import termios
import time
import tty


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.deployment import apply as apply_deployment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--face", choices=("front", "back"),
        help="提供时进入人工抓取模式，并指定人工放入左夹爪的工件朝向；省略时自动从 Module 4 判断正反",
    )
    parser.add_argument("--workpiece", choices=("1", "2"), default="1")
    parser.add_argument("--mode", choices=("gzhu", "xiangxin"), default="gzhu")
    parser.add_argument("--planner-host", default=None)
    parser.add_argument("--arm-host", default=None)
    parser.add_argument("--gripper-host", default=None)
    parser.add_argument("--force-host", default=None)
    parser.add_argument("--head-host", default=None)
    return parser.parse_args()


def result_file(face: str) -> Path:
    destination = ROOT / "output/module_5_test" / time.strftime("%Y%m%d_%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    return destination / f"{face}_handoff.json"


def run(command: list[str]) -> int:
    print("[module-5-test] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def finish_front_to_place_ready(args: argparse.Namespace, handoff_file: Path) -> int:
    """正面：graspuptop → Middle → M8.1 右手持件放置准备位。"""
    middle_result = handoff_file.with_name("front_middle_place.json")
    code = run([
        str(ROOT / "modules/module_3_left_place/run.sh"),
        "--workpiece", args.workpiece,
        "--handoff-file", str(handoff_file),
        "--placement-kind", "middle",
        "--result-file", str(middle_result),
        "--planner-host", args.planner_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
        "--execute", "--confirm-real-place",
    ])
    if code != 0:
        return code
    right_ready = handoff_file.with_name("front_right_place_ready.json")
    return run([
        str(PYTHON),
        str(ROOT / "modules/module_8_right_grasp/module_8_1_middle_grasp/workpiece_pipeline.py"),
        "--workpiece", args.workpiece,
        "--result-file", str(right_ready),
        "--planner-host", args.planner_host,
        "--execute",
    ])


def finish_back_to_place_ready(args: argparse.Namespace, handoff_file: Path) -> int:
    """反面：读取 M5.2 同轮 ICP → M8.2 交接并到右臂放置准备位。"""
    import json

    payload = json.loads(handoff_file.read_text(encoding="utf-8"))
    if payload.get("event") != "module_5_2_left_handoff_ready":
        raise RuntimeError(f"M5.2 未发布有效交接握手：{handoff_file}")
    icp_result = Path(str(payload.get("result") or "")).expanduser().resolve()
    if not icp_result.is_file():
        raise RuntimeError(f"M5.2 握手引用的 ICP 结果不存在：{icp_result}")
    ready_marker = handoff_file.with_name("back_right_place_ready.json")
    return run([
        str(PYTHON),
        str(ROOT / "modules/module_8_right_grasp/module_8_2_left_handoff/main.py"),
        "--result", str(icp_result),
        "--workpiece", args.workpiece,
        "--mode", args.mode,
        "--right-place-ready-file", str(ready_marker),
        "--planner-host", args.planner_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
    ])


def read_key(prompt: str) -> str:
    """像示教工具一样读取一个按键，不要求回车。"""
    if not sys.stdin.isatty():
        raise RuntimeError("人工模式需要在交互式终端运行")
    print(prompt, end="", flush=True)
    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original)
    print(flush=True)
    return key


def _left_gripper_holds(state: dict) -> bool:
    try:
        position = float(state.get("position"))
    except (TypeError, ValueError):
        return False
    motion_state = str(state.get("motion_state", "")).strip().lower()
    return motion_state in {"gripping", "arrived"} and position > 20.0


def manual_gripper_control(gripper_host: str) -> dict:
    """空格反复切换左夹爪；只有 y 且反馈持件时才结束交互。"""
    from parm_sdk import Gripper

    target_open = True
    with Gripper(host=gripper_host, status_port=7702) as gripper:
        while True:
            key = read_key(
                "空格：切换左夹爪开/闭；y：确认持件并继续；q：退出："
            ).lower()
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "q":
                raise RuntimeError("操作者退出人工抓取")
            if key == " ":
                target_open = not target_open
                position = 100.0 if target_open else 0.0
                action = "打开" if target_open else "闭合"
                response = gripper.move(
                    1, position, force=60.0, speed=80.0,
                    mode="async", preempt=True, timeout=60.0,
                )
                if not isinstance(response, dict) or response.get("status") not in {
                    "accepted", "success",
                }:
                    raise RuntimeError(f"左夹爪{action}命令失败：{response}")
                print(f"[module-5-test] 已下发左夹爪{action}命令。", flush=True)
                continue
            if key == "y":
                state = gripper.get_state().get(1) or {}
                if _left_gripper_holds(state):
                    print(f"[module-5-test] 左夹爪已确认持件：{state}", flush=True)
                    return state
                print(
                    "[module-5-test] 当前尚未确认持件，请继续用空格调整；"
                    f"状态={state}",
                    flush=True,
                )


def manual_front(args: argparse.Namespace, arm_host: str, gripper_host: str) -> int:
    """人工已夹稳 FRONT：只到 graspuptop，不重抓、不松爪。"""
    from loop_config import LoopConfig
    from modules.module_5_left_pick.module_5_1_front_pick_to_graspuptop.main import (
        _wait_for_hold,
        _wait_for_left_joints,
        _write_result,
    )
    from modules.module_5_left_pick.module_5_1_front_pick_to_graspuptop.placement_parameters import (
        GRASPUP_ABOVE_JOINTS_DEG,
    )
    from tools.planner_runtime import PlannerRuntime

    held, position, motion = _wait_for_hold(gripper_host)
    if not held:
        raise RuntimeError(
            "人工模式进入交接前必须确认左夹爪已稳定持件；"
            f"当前状态 motion_state={motion!r}, position={position}"
        )
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
        runtime.set_next_plan_context("manual_front_held", "left_graspuptop")
        runtime.move_left_joints(GRASPUP_ABOVE_JOINTS_DEG, planner="rrtstar")
        _wait_for_left_joints(arm_host, GRASPUP_ABOVE_JOINTS_DEG, label="manual-held graspuptop")
    finally:
        runtime.close()
    _write_result(args.result_file, {
        "schema_version": 1, "module": "module_5_test_handoff",
        "outcome": "manual_front_held_at_graspuptop", "classification": "FRONT",
        "workpiece_id": args.workpiece, "handoff_pose": "graspuptop",
        "left_gripper_holds_object": True,
        "gripper_at_start": {"motion_state": motion, "position_percent": position},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"[module-5-test] 人工 FRONT 持件已到 graspuptop：{args.result_file}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    # 与总控共用同一套 mode -> 地址/标定解析，同时导出 PARM_* 环境变量。
    # PlannerRuntime 的 SDK 执行层读取这些变量，因此不能只把地址传给子命令。
    apply_deployment(args)
    from modules.module_0_prepare.main import prepare_robot

    print(
        "[module-5-test] 启动准备：双臂先回初始位并确认到位，再打开双夹爪。",
        flush=True,
    )
    prepare_robot(
        planner_host=args.planner_host,
        arm_host=args.arm_host,
        gripper_host=args.gripper_host,
        force_host=args.force_host,
        speed=0.3,
        open_grippers_before_motion=False,
    )
    arm_host = args.arm_host
    gripper_host = args.gripper_host
    head_host = args.head_host
    selected_face = args.face
    args.result_file = result_file(selected_face or "auto")
    if args.face is not None:
        from loop_config import LoopConfig
        from tools.planner_runtime import PlannerRuntime
        from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile

        ready = load_workpiece_handoff_profile(args.workpiece, mode=args.mode)["left_reverse_grasp"]["ready_joints_deg"]
        runtime = PlannerRuntime(LoopConfig(
            planner_host=args.planner_host, execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
            base_frame="body_base", left_group="left_arm", right_group="right_arm",
            left_tool_frame="left_hand_transfer_end", right_tool_frame="right_hand_transfer_end",
            max_velocity=0.2, max_acceleration=0.2, idle_poll_seconds=0.2, live=True, poses={},
        ))
        try:
            runtime.open(required_frames={"body_base", "left_hand_transfer_end"})
            runtime.move_left_joints(ready, planner="rrtstar")
        finally:
            runtime.close()
        print("左臂已到反抓预备位，请把工件放入左夹爪。", flush=True)
        manual_gripper_control(gripper_host)
        if args.face == "front":
            code = manual_front(args, arm_host, gripper_host)
            return code if code != 0 else finish_front_to_place_ready(
                args, args.result_file
            )
        code = run([
            str(PYTHON), "-m",
            "modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.resume_handoff",
            "--workpiece", args.workpiece, "--mode", args.mode,
            "--result-file", str(args.result_file), "--planner-host", args.planner_host,
            "--arm-host", arm_host, "--gripper-host", gripper_host, "--head-host", head_host,
        ])
        return code if code != 0 else finish_back_to_place_ready(
            args, args.result_file
        )
    # 自动模式从 Module 4 的观察位与 Buffer Check 开始，由实际判定结果决定
    # 正/反抓分支；不允许手工指定相反的朝向覆盖视觉结果。
    observation = args.result_file.with_name("module_4_observation.json")
    checked = args.result_file.with_name("module_4_buffer_check.json")
    if run([
        str(PYTHON), str(ROOT / "modules/module_4_head_judge/observation_pose.py"),
        "--result-file", str(observation), "--planner-host", args.planner_host,
        "--arm-host", arm_host, "--head-host", head_host,
    ]) != 0:
        return 1
    check_code = run([
        str(PYTHON), str(ROOT / "modules/module_4_head_judge/module_4_1_buffer_state_estimation/check_buffer_face.py"),
        "--workpiece", args.workpiece, "--mode", args.mode,
        "--result-file", str(checked), "--max-uncertain-retries", "0",
    ])
    if check_code not in {0, 20}:
        return check_code
    import json
    decision = str(json.loads(checked.read_text(encoding="utf-8")).get("decision", "")).lower()
    if decision not in {"front", "back"}:
        raise RuntimeError(f"Module 4 未得到唯一正/反工件：{checked}")
    selected_face = decision
    args.result_file = args.result_file.with_name(f"{selected_face}_handoff.json")
    if selected_face == "front":
        code = run([
            str(PYTHON), str(ROOT / "modules/module_5_left_pick/module_5_1_front_pick_to_graspuptop/main.py"),
            "--buffer-check-result", str(checked), "--result-file", str(args.result_file),
            "--planner-host", args.planner_host, "--arm-host", arm_host,
            "--gripper-host", gripper_host, "--execute",
        ])
        return code if code != 0 else finish_front_to_place_ready(
            args, args.result_file
        )
    code = run([
        str(PYTHON), "-m", "modules.module_5_left_pick.module_5_2_reverse_pick_to_handoff.pipeline",
        "--buffer-check-result", str(checked), "--workpiece", args.workpiece,
        "--mode", args.mode, "--result-file", str(args.result_file),
        "--planner-host", args.planner_host, "--arm-host", arm_host,
        "--gripper-host", gripper_host, "--head-host", head_host, "--live",
    ])
    return code if code != 0 else finish_back_to_place_ready(
        args, args.result_file
    )


if __name__ == "__main__":
    raise SystemExit(main())
