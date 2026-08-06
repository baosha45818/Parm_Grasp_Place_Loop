#!/usr/bin/env python3
"""Move the right arm to a saved TCP pose via a point 2 cm above it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_config import load_config
from loop_types import ObjectPose
from tool.planner_runtime import PlannerRuntime, TaskBusyError


DEFAULT_CONFIG = ROOT / "config.hole_a_first_live.json"
DEFAULT_SAVED_POSE = Path(__file__).with_name("arm_pose_20260722_122102_703666.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="右臂先到保存 TCP 位姿上方，再沿基座 Z 方向直线下降到保存位姿。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--saved-pose",
        type=Path,
        default=DEFAULT_SAVED_POSE,
        help="保存的右臂位姿 JSON（默认：module_9_right_place/arm_pose_20260722_122102_703666.json）",
    )
    parser.add_argument(
        "--height-m",
        type=float,
        default=0.02,
        help="上方安全点相对保存位姿的 +Z 高度（米，默认 0.02）",
    )
    parser.add_argument("--live", action="store_true", help="实际执行；省略时仅 dry-run")
    parser.add_argument(
        "--soft-stop-existing-task",
        action="store_true",
        help=(
            "仅当右臂执行接口报告已有任务时，明确授权软停止该任务、重新规划并重试；"
            "请先确认现场安全"
        ),
    )
    args = parser.parse_args()

    if not math.isfinite(args.height_m) or args.height_m <= 0.0:
        parser.error("--height-m 必须是正的有限数值")

    target = _load_right_pose(args.saved_pose)
    above = ObjectPose(
        frame_id="body_base",
        position=[target.position[0], target.position[1], target.position[2] + args.height_m],
        orientation=list(target.orientation),
    )
    print(f"[saved-pose] source={args.saved_pose}")
    print(f"[saved-pose] target xyz={target.position}, quaternion={target.orientation}")
    print(f"[saved-pose] above  xyz={above.position}, height={args.height_m * 1000.0:.1f} mm")

    config = load_config(args.config, live=args.live)
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        print("[saved-pose] phase 1/2: RRTStar -> above")
        _move_with_busy_recovery(
            planner, above, "rrtstar", args.soft_stop_existing_task
        )
        print("[saved-pose] phase 2/2: LIN -> saved pose")
        _move_with_busy_recovery(
            planner, target, "lin", args.soft_stop_existing_task
        )
    finally:
        planner.close()
    print("[saved-pose] completed")
    return 0


def _load_right_pose(path: Path) -> ObjectPose:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        right = payload["arms"]["right"]
        pose = right["eef_pose"]
        xyz = [float(value) for value in pose["xyz"]]
        rotvec = [float(value) for value in pose["rotvec"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取右臂保存位姿 {path}: {exc}") from exc
    if len(xyz) != 3 or len(rotvec) != 3 or not all(math.isfinite(value) for value in xyz + rotvec):
        raise ValueError("保存位姿必须包含 3 个有限 xyz 和 3 个有限 rotvec 数值")
    return ObjectPose(
        frame_id="body_base",
        position=xyz,
        orientation=_rotvec_to_quaternion(rotvec),
    )


def _move_with_busy_recovery(
    planner: PlannerRuntime,
    target: ObjectPose,
    motion_planner: str,
    soft_stop_existing_task: bool,
) -> None:
    try:
        planner.move_cartesian_pose("right", target, planner=motion_planner)
    except TaskBusyError as exc:
        if not soft_stop_existing_task:
            raise RuntimeError(
                "右臂已有任务正在运行；确认现场安全并等待任务结束后重试。"
                "若确认需要停止该任务，可用 --soft-stop-existing-task 重新运行。"
            ) from exc
        print("[saved-pose] 右臂已有任务，按明确授权软停止，等待后重新规划。")
        planner.soft_stop_arm("right")
        time.sleep(0.5)
        planner.move_cartesian_pose("right", target, planner=motion_planner)


def _rotvec_to_quaternion(rotvec: list[float]) -> list[float]:
    angle = math.sqrt(sum(value * value for value in rotvec))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    scale = math.sin(angle / 2.0) / angle
    return [rotvec[0] * scale, rotvec[1] * scale, rotvec[2] * scale, math.cos(angle / 2.0)]


if __name__ == "__main__":
    raise SystemExit(main())
