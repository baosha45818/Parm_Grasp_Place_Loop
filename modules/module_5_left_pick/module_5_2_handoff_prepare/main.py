#!/usr/bin/env python3
"""Module 5-2：左手持件观察、ICP定位并移动到右手交接位。

执行前提是 Module 5-1 已完成左手反抓，且工件相对左夹爪保持不动。本模块
只负责左侧准备工作，不移动右臂、不闭合右夹爪，也不释放左夹爪。

成功后会输出本轮 ``result.json``。传入 ``--handoff-result-file`` 时，还会
原子写入一个供 Module 8-2 或兼容总控读取的 JSON 握手文件。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules.workpiece_2_handoff_profile import (
    load_workpiece_2_handoff_profile,
)
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(
        str(PYTHON),
        [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

DEFAULT_ALIGNMENT = (
    ROOT / "calibration/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch2_yaw26p5_kinematic.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "output/reversed_head_pose_d435"
DEFAULT_AUTO_MASK_CONFIG = (
    ROOT
    / "tests/reversed_head_observation/head_d435_auto_mask_config.json"
)
WORKPIECE_2_CONFIG_DIR = Path(__file__).resolve().parent / "workpiece_2_config"
WORKPIECE_2_POSE_CONFIG = WORKPIECE_2_CONFIG_DIR / "head_d435_pose_config.json"
WORKPIECE_2_AUTO_MASK_CONFIG = (
    WORKPIECE_2_CONFIG_DIR / "head_d435_auto_mask_config.json"
)
WORKPIECE_2_HANDOFF_CONFIG = WORKPIECE_2_CONFIG_DIR / "handoff_config.json"
WORKPIECE_2_OUTPUT_DIR = ROOT / "output/module_5_2_workpiece_2_head_d435"
HEAD_MOTION = ROOT / "tool/move_head_gimbal.py"
HANDOFF_HEAD_PITCH_DEG = 2.0
HANDOFF_HEAD_YAW_DEG = 26.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment",
        type=Path,
        default=DEFAULT_ALIGNMENT,
        help="头部D435在pitch=2°、yaw=26.5°时的body_base外参",
    )
    parser.add_argument("--sam2-url", default=os.environ.get("PARM_SAM2_URL", "unix:///tmp/sam2-service.sock"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "interactive"),
        default=None,
        help=(
            "SAM2分割方式；默认auto使用当前RGB-D自动标记；"
            "interactive使用全手工ROI/正负点"
        ),
    )
    parser.add_argument(
        "--auto-mask-config",
        type=Path,
        default=DEFAULT_AUTO_MASK_CONFIG,
        help="固定观察位自动ROI和SAM2正负点配置",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument(
        "--workpiece",
        choices=("1", "2"),
        default=os.environ.get("PARM_LOOP_WORKPIECE", "1"),
        help="工件编号；总控默认从 PARM_LOOP_WORKPIECE 传入，缺省为原工件1流程",
    )
    parser.add_argument(
        "--handoff-result-file",
        type=Path,
        help="可选：写入包含本轮result.json路径的Module 5-2握手JSON",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=4.0 * float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0")),
        help="相对原流程速度倍率，默认4.0",
    )
    parser.add_argument(
        "--require-confirmation",
        action="store_true",
        help="恢复各运动阶段人工确认；默认自动确认",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n[module-5-2] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def cancel_left_arm(arm_host: str) -> None:
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(
            f"[stop] 无法导入parm_sdk，未能停止左臂：{exc}",
            flush=True,
        )
        return
    arm = None
    try:
        arm = Arm(host=arm_host, status_port=None)
        arm.cancel_task(1)
        print("[stop] left arm cancel_task(1) 已发送", flush=True)
    except Exception as exc:
        print(f"[stop] left arm cancel_task失败：{exc}", flush=True)
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass


def newest_result(output_dir: Path, previous: set[Path]) -> Path:
    results = [
        path
        for path in output_dir.glob("*/result.json")
        if path not in previous
    ]
    if not results:
        raise RuntimeError(
            f"D435位姿估计没有在{output_dir}生成新的result.json"
        )
    return max(results, key=lambda path: path.stat().st_mtime).resolve()


def publish_handoff_result(
    destination: Path,
    result: Path,
    *,
    workpiece: str,
    profile_id: str | None = None,
    handoff_center_body_m: list[float] | None = None,
) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "module_5_2_left_handoff_ready",
        "workpiece": workpiece,
        "result": str(result),
        "output_frame": "body_base",
        "left_gripper_holds_object": True,
        "created_at": time.time(),
    }
    if profile_id is not None:
        payload["profile_id"] = profile_id
    if handoff_center_body_m is not None:
        payload["handoff_center_body_m"] = handoff_center_body_m
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"[module-5-2] 已写入交接握手：{destination}", flush=True)


def main() -> int:
    args = parse_args()
    if args.workpiece == "2":
        if args.mask_mode is None:
            args.mask_mode = "auto"
        return main_workpiece_2(args)
    if args.mask_mode is None:
        args.mask_mode = "auto"
    # 两类工件共用已经验证的较高观察/交接运动区域；工件模型、SAM2
    # 提示和ICP参数仍保持工件1自己的配置。
    profile = load_workpiece_2_handoff_profile()
    observation = profile["left_observation"]
    handoff_center = list(profile["handoff"]["center_body_m"])
    from modules.module_0_prepare.ensure_sam2_service import require_sam2_service
    require_sam2_service()
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    if not HEAD_MOTION.is_file():
        raise FileNotFoundError(f"云台动作工具不存在：{HEAD_MOTION}")
    alignment = args.alignment.expanduser().resolve()
    if not alignment.is_file():
        raise FileNotFoundError(f"D435外参不存在：{alignment}")
    output_dir = args.output_dir.expanduser().resolve()
    auto_mask_config = args.auto_mask_config.expanduser().resolve()
    if args.mask_mode == "auto" and not auto_mask_config.is_file():
        raise FileNotFoundError(
            f"自动分割配置不存在：{auto_mask_config}"
        )

    # 1. 左手经共用高位准备点到共用观察位。
    run([
        sys.executable,
        "-m",
        "tests.reversed_head_observation.execute_plan",
        "--planner-host",
        args.planner_host,
        "--driver-host",
        args.driver_host,
        "--arm-host",
        args.arm_host,
        "--speed",
        str(0.05 * args.speed_multiplier),
        "--middle-joints-deg",
        *(str(value) for value in observation["prepare_joints_deg"]),
        "--observation-joints-deg",
        *(str(value) for value in observation["observation_joints_deg"]),
        "--skip-head",
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])

    # 2. 反手交接专用流程：云台到标定角度，再运行 D435、SAM2 和 ICP。
    previous = set(output_dir.glob("*/result.json"))
    run([
        str(PYTHON), str(HEAD_MOTION), "--execute", "--host", args.driver_host,
        "--pitch", str(HANDOFF_HEAD_PITCH_DEG),
        "--yaw", str(HANDOFF_HEAD_YAW_DEG),
    ])
    run([
        sys.executable,
        "-m",
        "tests.reversed_head_observation.estimate_pose_head_d435",
        "--alignment",
        str(alignment),
        "--output-frame",
        "body_base",
        "--sam2-url",
        args.sam2_url,
        "--mask-mode",
        args.mask_mode,
        "--no-auto-gripper-prompt",
        "--auto-mask-config",
        str(auto_mask_config),
        "--output-dir",
        str(output_dir),
        "--planner-host",
        args.planner_host,
        "--driver-host",
        args.driver_host,
        "--expected-head-pitch",
        str(HANDOFF_HEAD_PITCH_DEG),
        "--expected-head-yaw",
        str(HANDOFF_HEAD_YAW_DEG),
    ])
    result = newest_result(output_dir, previous)
    print(
        f"[module-5-2] 反手交接观察完成；SAM2分割模式={args.mask_mode}；"
        f"自动配置={auto_mask_config if args.mask_mode == 'auto' else '未使用'}",
        flush=True,
    )
    print(f"[module-5-2] 本轮D435结果：{result}", flush=True)

    # 3. 复用拍摄时左TCP到工件的刚性关系，将工件送到交接位。
    run([
        sys.executable,
        "-m",
        "tests.reversed_head_observation.move_object_center",
        "--result",
        str(result),
        "--target-center",
        *(str(value) for value in handoff_center),
        "--fixed-handoff-orientation",
        "--reuse-grasp-relation",
        "--execute",
        "--planner-host",
        args.planner_host,
        "--speed-multiplier",
        str(args.speed_multiplier),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])

    if args.handoff_result_file:
        publish_handoff_result(
            args.handoff_result_file,
            result,
            workpiece="1",
            handoff_center_body_m=handoff_center,
        )
    print(
        "[module-5-2] 左手已持工件1到共用高位交接中心="
        f"{handoff_center}；左夹爪保持夹紧，右臂尚未运动。",
        flush=True,
    )
    print(f"MODULE_5_2_RESULT={result}", flush=True)
    return 0


def main_workpiece_2(args: argparse.Namespace) -> int:
    """工件2左手交接准备：只观察和送交接位，绝不驱动右臂。"""
    from modules.module_0_prepare.ensure_sam2_service import require_sam2_service

    profile = load_workpiece_2_handoff_profile()
    observation = profile["left_observation"]
    handoff = profile["handoff"]
    handoff_center = list(handoff["center_body_m"])
    head_pose = observation["head_pose_deg"]
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    for path, label in (
        (HEAD_MOTION, "云台动作工具"),
        (WORKPIECE_2_POSE_CONFIG, "工件2 D435位姿配置"),
        (WORKPIECE_2_AUTO_MASK_CONFIG, "工件2自动分割配置"),
        (WORKPIECE_2_HANDOFF_CONFIG, "工件2交接配置"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")

    require_sam2_service()
    output_dir = WORKPIECE_2_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = set(output_dir.glob("*/result.json"))

    # 工件2复用已验证的左臂头部观察位；该动作不涉及右臂。
    run([
        sys.executable,
        "-m",
        "tests.reversed_head_observation.execute_plan",
        "--planner-host", args.planner_host,
        "--driver-host", args.driver_host,
        "--arm-host", args.arm_host,
        "--speed", str(0.05 * args.speed_multiplier),
        "--middle-joints-deg",
        *(str(value) for value in observation["prepare_joints_deg"]),
        "--observation-joints-deg",
        *(str(value) for value in observation["observation_joints_deg"]),
        "--skip-head",
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])
    run([
        str(PYTHON), str(HEAD_MOTION), "--execute", "--host", args.driver_host,
        "--pitch", str(head_pose["pitch"]),
        "--yaw", str(head_pose["yaw"]),
    ])
    run([
        sys.executable,
        "-m",
        "tests.reversed_head_observation.estimate_pose_head_d435",
        "--config", str(WORKPIECE_2_POSE_CONFIG),
        "--alignment", str(observation["alignment"]),
        "--output-frame", "body_base",
        "--sam2-url", args.sam2_url,
        "--mask-mode", args.mask_mode,
        "--no-auto-gripper-prompt",
        "--auto-mask-config", str(WORKPIECE_2_AUTO_MASK_CONFIG),
        "--output-dir", str(output_dir),
        "--planner-host", args.planner_host,
        "--driver-host", args.driver_host,
        "--expected-head-pitch", str(head_pose["pitch"]),
        "--expected-head-yaw", str(head_pose["yaw"]),
    ])
    result = newest_result(output_dir, previous)
    run([
        sys.executable,
        "-m",
        "tests.reversed_head_observation.move_object_center",
        "--result", str(result),
        "--config", str(WORKPIECE_2_HANDOFF_CONFIG),
        "--target-center", *(str(value) for value in handoff_center),
        "--cartesian-planner", "rrtstar",
        "--fixed-handoff-orientation",
        "--reuse-grasp-relation",
        "--execute",
        "--planner-host", args.planner_host,
        "--speed-multiplier", str(args.speed_multiplier),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])
    if args.handoff_result_file:
        publish_handoff_result(
            args.handoff_result_file,
            result,
            workpiece="2",
            profile_id=str(profile["profile_id"]),
            handoff_center_body_m=handoff_center,
        )
    print(
        "[module-5-2/workpiece2] 左手已将工件2送到"
        f"body_base交接中心={handoff_center}；"
        "左夹爪保持夹紧，右臂尚未运动。",
        flush=True,
    )
    print(f"MODULE_5_2_RESULT={result}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stop] 收到Ctrl+C，停止左臂当前任务。", flush=True)
        cancel_left_arm(parse_args().arm_host)
        raise SystemExit(130)
