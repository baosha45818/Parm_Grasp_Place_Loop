#!/usr/bin/env python3
"""兼容总控：依次运行 Module 5-2 左侧准备和 Module 8-2 右侧交接。

业务实现已拆分到 ``modules/module_5_left_pick/module_5_2_handoff_prepare``
和 ``modules/module_8_right_grasp/module_8_2_left_handoff``。保留本入口是为了
兼容现有双臂总控和原独立执行命令。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_8_right_grasp.module_8_2_left_handoff.main import (
    V2_RIGHT_ABOVE_MODULE_9_JOINTS_DEG,
)


PYTHON = Path("/home/dainanjun/Public/data_collector/.venv/bin/python3")
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
DEFAULT_ALIGNMENT = Path(
    "/home/dainanjun/zhuyiming/calibration/output/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch15_yaw35_kinematic.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "output/reversed_head_pose_d435"
DEFAULT_AUTO_MASK_CONFIG = (
    ROOT
    / "tests/reversed_head_observation/head_d435_auto_mask_config.json"
)
MODULE_5_2 = (
    ROOT
    / "modules/module_5_left_pick/module_5_2_handoff_prepare/main.py"
)
MODULE_8_2 = (
    ROOT
    / "modules/module_8_right_grasp/module_8_2_left_handoff/main.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--sam2-url", default="unix:///tmp/sam2-service.sock")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "interactive"),
        default="auto",
        help="默认自动分割；interactive恢复全手工ROI/正负点",
    )
    parser.add_argument(
        "--auto-mask-config",
        type=Path,
        default=DEFAULT_AUTO_MASK_CONFIG,
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument(
        "--right-place-ready-file",
        type=Path,
        help="可选：右臂到Module 9上方准备位后写入总控同步标记",
    )
    parser.add_argument("--speed-multiplier", type=float, default=4.0)
    parser.add_argument("--require-confirmation", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n[loop-v2] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def cancel_both_arms(arm_host: str) -> None:
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(
            f"[stop] 无法导入parm_sdk，未能发送cancel_task()：{exc}",
            flush=True,
        )
        return
    for arm_id, label in ((1, "left"), (2, "right")):
        arm = None
        try:
            arm = Arm(host=arm_host, status_port=None)
            arm.cancel_task(arm_id)
            print(
                f"[stop] {label} arm cancel_task({arm_id}) 已发送",
                flush=True,
            )
        except Exception as exc:
            print(f"[stop] {label} arm cancel_task失败：{exc}", flush=True)
        finally:
            if arm is not None:
                try:
                    arm.close()
                except Exception:
                    pass


def read_left_handoff_result(state_file: Path) -> Path:
    if not state_file.is_file():
        raise RuntimeError("Module 5-2 未生成交接握手JSON")
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    if payload.get("event") != "module_5_2_left_handoff_ready":
        raise RuntimeError(f"Module 5-2 交接握手内容无效：{payload}")
    if payload.get("left_gripper_holds_object") is not True:
        raise RuntimeError(f"Module 5-2 未确认左手仍夹持工件：{payload}")
    result_value = payload.get("result")
    if not isinstance(result_value, str) or not result_value:
        raise RuntimeError(f"Module 5-2 交接握手缺少result路径：{payload}")
    result = Path(result_value).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Module 5-2 result.json不存在：{result}")
    return result


def main() -> int:
    args = parse_args()
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    if not MODULE_5_2.is_file() or not MODULE_8_2.is_file():
        raise FileNotFoundError(
            f"拆分模块入口缺失：Module 5-2={MODULE_5_2}, "
            f"Module 8-2={MODULE_8_2}"
        )

    with tempfile.TemporaryDirectory(prefix="parm-handoff-v2-") as temp_dir:
        state_file = Path(temp_dir) / "module_5_2_result.json"
        run([
            str(PYTHON),
            str(MODULE_5_2),
            "--alignment",
            str(args.alignment),
            "--sam2-url",
            args.sam2_url,
            "--output-dir",
            str(args.output_dir),
            "--mask-mode",
            args.mask_mode,
            "--auto-mask-config",
            str(args.auto_mask_config),
            "--planner-host",
            args.planner_host,
            "--driver-host",
            args.driver_host,
            "--arm-host",
            args.arm_host,
            "--handoff-result-file",
            str(state_file),
            "--speed-multiplier",
            str(args.speed_multiplier),
            *(
                ["--require-confirmation"]
                if args.require_confirmation
                else []
            ),
        ])
        result = read_left_handoff_result(state_file)
        print(f"[loop-v2] Module 5-2交付结果：{result}", flush=True)

        run([
            str(PYTHON),
            str(MODULE_8_2),
            "--result",
            str(result),
            "--planner-host",
            args.planner_host,
            "--gripper-host",
            args.gripper_host,
            "--arm-host",
            args.arm_host,
            "--speed-multiplier",
            str(args.speed_multiplier),
            *(
                [
                    "--right-place-ready-file",
                    str(args.right_place_ready_file),
                ]
                if args.right_place_ready_file
                else []
            ),
            *(
                ["--require-confirmation"]
                if args.require_confirmation
                else []
            ),
        ])

    print(
        "[loop-v2] Module 5-2左侧准备与Module 8-2右侧交接均已完成。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stop] 收到Ctrl+C，直接停止左右臂任务。", flush=True)
        cancel_both_arms(parse_args().arm_host)
        raise SystemExit(130)
