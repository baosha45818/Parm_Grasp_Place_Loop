#!/usr/bin/env python3
"""串联左手反抓后的观察、交接位移动，以及右手抓取工件尾部。

执行前提：左手已经完成反抓，且从观察、拍摄到交接期间一直夹稳工件。默认
以原速度的四倍自动连续执行；传入 ``--require-confirmation`` 可恢复逐步确认。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIGNMENT = Path(
    "/home/dainanjun/zhuyiming/calibration/output/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch15_yaw35_kinematic.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "output/reversed_head_pose_d435"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment", type=Path, default=DEFAULT_ALIGNMENT,
        help="头部 D435 在 pitch=15°、yaw=35° 时的 body_base 外参",
    )
    parser.add_argument("--sam2-url", default="unix:///tmp/sam2-service.sock")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--handoff-config", type=Path,
        help=("可选：使用已示教的 handoff_config.ready.json 旧式交接；"
              "未提供时按 ICP 刚性关系执行右手尾部抓取"),
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument(
        "--right-place-ready-file",
        type=Path,
        help="可选：右臂到放置准备位后写入的总控同步标记文件",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=4.0,
        help="相对原流程速度倍率，默认4.0",
    )
    parser.add_argument(
        "--require-confirmation",
        action="store_true",
        help="恢复各运动阶段的人工确认；默认全流程自动确认",
    )
    parser.add_argument(
        "--cartesian-tail-handoff",
        action="store_true",
        help="使用末端位姿IK尾部交接；默认使用现场手工示教的右臂关节抓取点",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n[loop] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def cancel_both_arms(arm_host: str) -> None:
    """Keep standalone Ctrl+C safe too; the total loop already performs the same stop."""
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(f"[stop] 无法导入 parm_sdk，未能发送 cancel_task(): {exc}", flush=True)
        return
    for arm_id, label in ((1, "left"), (2, "right")):
        arm = None
        try:
            arm = Arm(host=arm_host, status_port=None)
            arm.cancel_task(arm_id)
            print(f"[stop] {label} arm cancel_task({arm_id}) 已发送", flush=True)
        except Exception as exc:
            print(f"[stop] {label} arm cancel_task 失败: {exc}", flush=True)
        finally:
            if arm is not None:
                try:
                    arm.close()
                except Exception:
                    pass


def newest_result(output_dir: Path, previous: set[Path]) -> Path:
    results = [path for path in output_dir.glob("*/result.json") if path not in previous]
    if not results:
        raise RuntimeError(f"D435 位姿估计没有在 {output_dir} 生成新的 result.json")
    return max(results, key=lambda path: path.stat().st_mtime)


def main() -> int:
    args = parse_args()
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    if not args.alignment.is_file():
        raise FileNotFoundError(f"D435 外参不存在：{args.alignment}")
    if args.handoff_config and not args.handoff_config.is_file():
        raise FileNotFoundError(f"交接配置不存在：{args.handoff_config}")
    if args.right_place_ready_file and args.cartesian_tail_handoff:
        raise ValueError(
            "--right-place-ready-file 目前仅支持默认的示教关节尾部交接流程"
        )

    # 左手已反抓后，移到已标定观察位并让云台到 pitch=15°、yaw=35°。
    run([
        sys.executable, "-m", "tests.reversed_head_observation.execute_plan",
        "--planner-host", args.planner_host, "--driver-host", args.driver_host,
        "--arm-host", args.arm_host,
        "--speed", str(0.05 * args.speed_multiplier),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])

    previous = set(args.output_dir.glob("*/result.json"))
    run([
        sys.executable, "-m", "tests.reversed_head_observation.estimate_pose_head_d435",
        "--output-frame", "body_base", "--alignment", str(args.alignment),
        "--sam2-url", args.sam2_url, "--output-dir", str(args.output_dir),
        "--planner-host", args.planner_host, "--driver-host", args.driver_host,
    ])
    result = newest_result(args.output_dir, previous)
    print(f"[loop] 本轮 D435 结果：{result}", flush=True)

    if args.handoff_config:
        # execute_handoff 必须从拍摄时的左 TCP 开始校验刚性关系，因此不能先运行
        # move_object_center。配置须含已示教的左右 TCP-工件关系和预抓取偏移。
        run([
            sys.executable, "-m", "tests.reversed_head_observation.execute_handoff",
            "--icp-result", str(result), "--config", str(args.handoff_config),
            "--planner-host", args.planner_host, "--gripper-host", args.gripper_host,
            "--execute",
        ])
        print("[loop] 左手反抓后的右手交接完成。")
        return 0

    # 使用拍摄时记录的刚性 TCP-工件关系将左手持件送至交接区。
    run([
        sys.executable, "-m", "tests.reversed_head_observation.move_object_center",
        "--result", str(result), "--reuse-grasp-relation", "--execute",
        "--planner-host", args.planner_host,
        "--speed-multiplier", str(args.speed_multiplier),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])

    # 左手移动后 ICP 的绝对 XYZ 已过时，但 TCP-工件关系仍然有效。交接器先用
    # 实时左 TCP 重建当前工件姿态并校验固定交接位，再移动右手。
    handoff_module = (
        "tests.reversed_head_observation.execute_tail_handoff"
        if args.cartesian_tail_handoff
        else "tests.reversed_head_observation.execute_taught_tail_handoff"
    )
    run([
        sys.executable,
        "-m",
        handoff_module,
        "--result",
        str(result),
        "--planner-host",
        args.planner_host,
        "--arm-host",
        args.arm_host,
        "--gripper-host",
        args.gripper_host,
        "--speed-multiplier",
        str(args.speed_multiplier),
        *(
            ["--right-place-ready-file", str(args.right_place_ready_file)]
            if args.right_place_ready_file
            else []
        ),
        *(
            ["--require-confirmation"]
            if args.require_confirmation
            else []
        ),
        "--execute",
    ])
    print("[loop] 右手已接取工件尾部，左手已释放并回到中间初始位。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stop] 收到 Ctrl+C，直接停止左右臂任务。", flush=True)
        cancel_both_arms(parse_args().arm_host)
        raise SystemExit(130)
