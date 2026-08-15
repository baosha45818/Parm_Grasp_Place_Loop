#!/usr/bin/env python3
"""Execute workpiece-2 reverse grasp, handoff, and calibrated right placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / ".venv/bin/python"
PLUCKY = Path("/home/dainanjun/Public/plucky_hjt_parm")
ALIGNMENT = (
    ROOT / "calibration/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch2_yaw26p5_kinematic.json"
)
OUTPUT_DIR = ROOT / "output/reversed_head_pose_d435_workpiece_2"
MOTION_STARTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--already-left-held",
        action="store_true",
        help=(
            "左夹爪已经在本轮抓稳第二类工件：跳过桌面反抓感知/抓取，"
            "直接从实时状态进入头部观察位"
        ),
    )
    parser.add_argument("--confirm-workpiece-2-reverse-grasp-geometry", action="store_true")
    parser.add_argument("--confirm-workpiece-2-handoff-geometry", action="store_true")
    parser.add_argument("--confirm-workpiece-2-place-calibration", action="store_true")
    parser.add_argument("--restart-planner-for-workpiece-2-collision", action="store_true")
    parser.add_argument("--handoff-center-m", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--place-config", type=Path,
        default=HERE / "first_workpiece_place_config.json",
        help="第一类工件Module 9/10放置参数映射",
    )
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--sam2-url", default="unix:///tmp/sam2-service.sock")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--speed", type=float, default=0.05)
    return parser.parse_args()


def run(label: str, command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n[完整流程/{label}] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def newest_result(previous: set[Path]) -> Path:
    candidates = [path for path in OUTPUT_DIR.glob("*/result.json") if path not in previous]
    if not candidates:
        raise RuntimeError(f"本轮没有在 {OUTPUT_DIR} 生成新的 result.json")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def restart_planner() -> None:
    from tests.reversed_head_observation.handoff_attachment import write_attachment_request

    state_path = HERE / "handoff_attachment_state.json"
    request_id = write_attachment_request(state_path, "left")
    command = [
        "docker", "compose",
        "-f", "docker-compose.yml",
        "-f", "docker-compose.tables.yml",
        "-f", "docker-compose.handoff.yml",
        "-f", str(HERE / "docker-compose.handoff.yml"),
        "up", "-d", "--build", "--force-recreate",
    ]
    run("重启工件2碰撞规划场景", command, cwd=PLUCKY)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 5604), timeout=1.0):
                break
        except OSError:
            time.sleep(1.0)
    else:
        raise RuntimeError("工件2碰撞规划场景重启后 5604 服务未就绪")

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("applied_owner") == "left"
                and state.get("applied_request_id") == request_id
            ):
                print(f"[完整流程] attached collision 已确认归左手: {state}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise RuntimeError("附着碰撞发布器未确认工件2归左手，拒绝继续运动")


def cancel_both_arms(arm_host: str) -> None:
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(f"[停止] 无法导入 parm_sdk: {exc}", flush=True)
        return
    for arm_id, label in ((1, "左臂"), (2, "右臂")):
        try:
            with Arm(host=arm_host, status_port=None) as arm:
                arm.cancel_task(arm_id)
            print(f"[停止] 已向{label}发送 cancel_task({arm_id})", flush=True)
        except Exception as exc:
            print(f"[停止] {label} cancel_task 失败: {exc}", flush=True)


def main() -> int:
    global MOTION_STARTED
    args = parse_args()
    if not args.execute:
        raise RuntimeError("完整流程会真实运动；必须显式传入 --execute")
    required_confirmations = {
        "--confirm-workpiece-2-handoff-geometry": args.confirm_workpiece_2_handoff_geometry,
        "--confirm-workpiece-2-place-calibration": args.confirm_workpiece_2_place_calibration,
    }
    if not args.already_left_held:
        required_confirmations[
            "--confirm-workpiece-2-reverse-grasp-geometry"
        ] = args.confirm_workpiece_2_reverse_grasp_geometry
    missing = [name for name, enabled in required_confirmations.items() if not enabled]
    if missing:
        raise RuntimeError("完整实体流程缺少显式确认参数: " + ", ".join(missing))
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed 必须在 (0,1] 内")
    if not args.alignment.is_file():
        raise FileNotFoundError(f"观察位外参不存在: {args.alignment}")

    # Validate the reused first-workpiece placement before the first motion.
    from tests.reversed_head_observation_2.execute_first_workpiece_place import load_config as load_place_config
    load_place_config(args.place_config)

    if args.already_left_held:
        print(
            "\n[完整流程/1 左手反抓] 已显式确认左夹爪在本轮抓稳工件；"
            "跳过桌面反抓感知和抓取，不复用旧采集结果。",
            flush=True,
        )
    else:
        reverse_command = [
            str(PYTHON), "-m", "tests.reversed_head_observation_2.execute_reverse_grasp",
            "--execute", "--confirm-workpiece-2-reverse-grasp-geometry",
            "--planner-host", args.planner_host, "--arm-host", args.arm_host,
            "--speed", str(args.speed),
        ]
        if args.capture_dir:
            reverse_command.extend(("--capture-dir", str(args.capture_dir)))
        MOTION_STARTED = True
        run("1 左手反抓", reverse_command)

    MOTION_STARTED = True
    run("2 左手到头部观察位", [
        str(PYTHON), "-m", "tests.reversed_head_observation_2.execute_plan",
        "--planner-host", args.planner_host, "--driver-host", args.driver_host,
        "--arm-host", args.arm_host, "--speed", str(args.speed),
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = set(OUTPUT_DIR.glob("*/result.json"))
    run("3 工件2 D435/SAM2/ICP", [
        str(PYTHON), "-m", "tests.reversed_head_observation_2.estimate_pose_head_d435",
        "--output-frame", "body_base", "--alignment", str(args.alignment),
        "--output-dir", str(OUTPUT_DIR), "--mask-mode", "interactive",
        "--no-auto-gripper-prompt",
        "--sam2-url", args.sam2_url, "--planner-host", args.planner_host,
        "--driver-host", args.driver_host,
    ])
    result = newest_result(previous)
    print(f"[完整流程] 本轮工件2 ICP: {result}", flush=True)

    motion_config = HERE / "handoff_config.json"
    if args.restart_planner_for_workpiece_2_collision:
        motion_config = HERE / "handoff_config.ready.json"
        run("4 生成附着碰撞配置", [
            str(PYTHON), "-m", "tests.reversed_head_observation_2.configure_handoff",
            "set-target", "--icp-result", str(result), "--center-m",
            *(str(value) for value in args.handoff_center_m), "--overwrite",
        ])
        restart_planner()
    else:
        print(
            "\n[完整流程/4] 按第一类工件流程使用已验证预设点；"
            "不重启Docker、不启用attached collision。",
            flush=True,
        )

    run("5 左手送交接位", [
        str(PYTHON), "-m", "tests.reversed_head_observation_2.move_object_center",
        "--result", str(result), "--config", str(motion_config),
        "--target-center", *(str(value) for value in args.handoff_center_m),
        "--reuse-grasp-relation", "--execute", "--planner-host", args.planner_host,
    ])

    handoff_command = [
        str(PYTHON), "-m", "tests.reversed_head_observation_2.execute_handoff",
        "--result", str(result), "--config", str(motion_config),
        "--expected-center-m", *(str(value) for value in args.handoff_center_m),
        "--planner-host", args.planner_host, "--gripper-host", args.gripper_host,
        "--arm-host", args.arm_host, "--execute", "--require-confirmation",
        "--confirm-workpiece-2-grasp-geometry",
    ]
    if args.restart_planner_for_workpiece_2_collision:
        handoff_command.append("--switch-attachment")
    run("6 右手接取", handoff_command)

    place_command = [
        str(PYTHON), "-m", "tests.reversed_head_observation_2.execute_first_workpiece_place",
        "--config", str(args.place_config), "--planner-host", args.planner_host,
        "--arm-host", args.arm_host, "--force-host", args.driver_host,
        "--gripper-host", args.gripper_host,
        "--attachment-state", str(HERE / "handoff_attachment_state.json"),
        "--speed", str(args.speed), "--execute",
        "--confirm-reuse-first-workpiece-placement",
    ]
    if args.restart_planner_for_workpiece_2_collision:
        place_command.append("--switch-attachment")
    run("7 右手放置", place_command)
    print("[完整流程完成] 左手反抓、右手交接和最终放置全部完成。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        if MOTION_STARTED:
            cancel_both_arms(parse_args().arm_host)
        raise SystemExit(130)
    except Exception:
        try:
            if MOTION_STARTED:
                cancel_both_arms(parse_args().arm_host)
        finally:
            raise
