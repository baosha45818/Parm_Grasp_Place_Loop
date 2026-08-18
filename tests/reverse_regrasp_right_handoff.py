#!/usr/bin/env python3
"""受保护地串联：左手反抓 -> 头部观察/ICP -> 左交接 -> 右手接取。

每个会造成真实运动、夹爪动作或人工分割的阶段，仍由底层执行器单独确认。
本入口不会绕过任何已有的安全检查或标定质量门限。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PYTHON = ROOT / ".venv/bin/python"
REGRASP_DIR = Path("/home/dainanjun/shurenbin/desktop_grasp")
POSE_OUTPUT_DIR = ROOT / "output/reversed_head_pose_d435"
ALIGNMENT = Path(
    "/home/dainanjun/zhuyiming/calibration/output/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch2_yaw19_kinematic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="允许执行完整的物理流程；各危险阶段仍须在终端输入 y 确认",
    )
    parser.add_argument("--capture-dir", type=Path, help="复用反抓阶段的桌面 RGB-D 采集")
    parser.add_argument(
        "--handoff-config", type=Path, required=True,
        help="已完成示教/标定的交接配置（不能使用含 null 的模板）",
    )
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--observation-speed", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    return parser.parse_args()


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("\n[loop]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def newest_result(output_dir: Path, previous: set[Path]) -> Path:
    candidates = [path for path in output_dir.glob("*/result.json") if path not in previous]
    if not candidates:
        raise RuntimeError(f"位姿估计没有在 {output_dir} 生成新的 result.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise SystemExit("本入口会串联真实抓取和交接；请在现场确认后显式传入 --execute。")
    if not RUNTIME_PYTHON.is_file():
        raise RuntimeError(f"运行环境不存在：{RUNTIME_PYTHON}")
    if not args.handoff_config.is_file():
        raise RuntimeError(f"交接配置不存在：{args.handoff_config}")
    if not args.alignment.is_file():
        raise RuntimeError(
            f"观察位的 body_base 标定不存在：{args.alignment}；"
            "不能为交接猜测工件坐标。"
        )

    # 1. 左手从桌面抓取反面工件。底层执行器会在下降、抓取和抬升前询问操作者。
    regrasp = [str(RUNTIME_PYTHON), "main.py", "--execute"]
    if args.capture_dir:
        regrasp.extend(("--capture-dir", str(args.capture_dir)))
    run(regrasp, cwd=REGRASP_DIR)

    # 2. 携件到已标定的头部观察位，并将云台转至 2°/19°。
    run([
        str(RUNTIME_PYTHON), "-m", "tests.reversed_head_observation.execute_plan",
        "--planner-host", args.planner_host, "--driver-host", args.driver_host,
        "--speed", str(args.observation_speed),
    ])

    # 3. 在观察位采集 D435i、分割并 ICP；交接必须使用 body_base 下的新鲜结果。
    previous = set(POSE_OUTPUT_DIR.glob("*/result.json"))
    run([
        str(RUNTIME_PYTHON), "-m", "tests.reversed_head_observation.estimate_pose_head_d435",
        "--output-dir", str(POSE_OUTPUT_DIR), "--output-frame", "body_base",
        "--alignment", str(args.alignment), "--expected-head-pitch", "2",
        "--expected-head-yaw", "19", "--planner-host", args.planner_host,
        "--driver-host", args.driver_host,
    ])
    icp_result = newest_result(POSE_OUTPUT_DIR, previous)
    print(f"[loop] 使用本轮 ICP 结果：{icp_result}", flush=True)

    # 4. 左手送交接位、右手接取、切换碰撞附着体并释放左手。
    run([
        str(RUNTIME_PYTHON), "-m", "tests.reversed_head_observation.execute_handoff",
        "--icp-result", str(icp_result), "--config", str(args.handoff_config),
        "--planner-host", args.planner_host, "--gripper-host", args.gripper_host,
        "--gripper-force", str(args.gripper_force), "--execute",
    ])
    print("[loop] 左手反抓与右手交接流程完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
