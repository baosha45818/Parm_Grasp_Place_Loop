#!/usr/bin/env python3
"""Reuse the first-workpiece Module-9 force-servo placement for workpiece 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tests.reversed_head_observation.handoff_attachment import request_attachment_owner
from loop_config import LoopConfig
from tool.planner_runtime import PlannerRuntime
from tests.loop.right_place_and_idle import wait_for_right_joint_target


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / ".venv/bin/python"
MODULE9 = ROOT / "modules/module_9_right_place/place_hole_a_wiggle.py"
FINISH = ROOT / "tests/loop/right_place_and_idle.py"
DEFAULT_CONFIG = HERE / "first_workpiece_place_config.json"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("strategy") != "reuse_first_workpiece_module9_force_servo":
        raise RuntimeError(f"未知放置策略: {config.get('strategy')!r}")
    if config.get("operator_authorized_for_workpiece_2") is not True:
        raise RuntimeError("第一类工件放置配置尚未获准用于第二类工件")
    for name in ("right_place_ready_joints_deg", "module9_initial_right_joints_deg"):
        values = config.get(name)
        if not isinstance(values, list) or len(values) != 7:
            raise RuntimeError(f"{name} 必须包含7个关节角")
    for name in ("module9_planner_config", "right_idle_pose"):
        if not Path(config[name]).is_file():
            raise FileNotFoundError(f"{name} 不存在: {config[name]}")
    return config


def run(label: str, command: list[str]) -> None:
    print(f"\n[沿用第一类放置/{label}] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--force-host", default="192.168.3.22")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--attachment-state", type=Path, default=HERE / "handoff_attachment_state.json")
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-reuse-first-workpiece-placement", action="store_true")
    parser.add_argument(
        "--switch-attachment",
        action="store_true",
        help="仅在附着碰撞发布器实际在线时解除MoveIt附着；默认不使用",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not args.execute or not args.confirm_reuse_first_workpiece_placement:
        raise RuntimeError(
            "真实执行须同时传入 --execute 和 --confirm-reuse-first-workpiece-placement"
        )
    print(json.dumps({
        "warning": "按操作者要求，第二类工件沿用第一类工件的Module 9力控放置参数",
        "right_place_ready_joints_deg": config["right_place_ready_joints_deg"],
        "module9_initial_right_joints_deg": config["module9_initial_right_joints_deg"],
        "module9_planner_config": config["module9_planner_config"],
    }, ensure_ascii=False, indent=2), flush=True)
    if input("确认第二类工件可沿用第一类孔位、抓持姿态和力阈值，输入 y 启动力控放置：").strip().lower() != "y":
        print("已取消；右手继续持件。")
        return 0

    ready = [float(value) for value in config["right_place_ready_joints_deg"]]
    runtime = PlannerRuntime(LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=args.speed,
        max_acceleration=args.speed,
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    ))
    try:
        runtime.open(required_frames={"body_base", "right_hand_transfer_end"})
        print(f"[沿用第一类放置] 规划到 right_place_ready: {ready}", flush=True)
        runtime.move_right_joints(ready)
        wait_for_right_joint_target(
            args.arm_host, ready, 0.5, 90.0, "第一类工件 right_place_ready"
        )
    finally:
        runtime.close()

    run("Module 9 力控 ServoPose", [
        str(PYTHON), str(MODULE9), "--live", "--confirm-live",
        "--planner-config", str(config["module9_planner_config"]),
        "--arm-host", args.arm_host, "--force-host", args.force_host,
        "--reset-right-task-before-servo",
    ])

    # The force-servo module returns only after insertion success. The object
    # is now supported by the fixture, so remove it from the robot attachment
    # before opening the gripper in the standard first-workpiece finish step.
    if args.switch_attachment:
        print(request_attachment_owner(args.attachment_state, "none", timeout_s=5.0))
    else:
        print("[沿用第一类放置] 未启用附着碰撞发布器，跳过attachment owner握手。")
    run("Module 10 松夹、抬升、回Idle", [
        str(PYTHON), str(FINISH), "--execute", "--require-confirmation",
        "--planner-host", args.planner_host, "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
        "--idle-pose", str(config["right_idle_pose"]),
        "--lift-m", str(config.get("post_release_lift_m", 0.05)),
        "--speed", str(args.speed),
    ])
    print("[完成] 已沿用第一类工件配置完成Module 9放置和Module 10收尾。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
