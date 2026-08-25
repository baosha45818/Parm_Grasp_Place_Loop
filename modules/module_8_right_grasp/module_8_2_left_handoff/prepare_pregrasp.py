#!/usr/bin/env python3
"""在左臂确认到达高位交接位后，把右臂送到安全预抓取位。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile
from modules.module_8_right_grasp.module_8_2_left_handoff.trajectory_cache import (
    compatibility,
    load as load_cache,
    save as save_cache,
)
from modules.module_8_right_grasp.module_8_2_left_handoff.gripper_compat import (
    move_gripper,
    read_gripper_state,
)
from tools.handoff_observation import execute_taught_tail_handoff as handoff
from tools.handoff_observation.preflight_handoff import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gzhu", "xiangxin"), required=True)
    parser.add_argument("--planner-host", required=True)
    parser.add_argument("--arm-host", required=True)
    parser.add_argument("--gripper-host", required=True)
    parser.add_argument("--marker-file", type=Path, required=True)
    parser.add_argument("--speed-multiplier", type=float, default=4.0)
    return parser.parse_args()


def publish_marker(destination: Path, *, endpoint: list[float], source: str) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "event": "right_pregrasp_ready",
        "workpiece": "1",
        "right_joints_deg": endpoint,
        "source": source,
        "created_at": time.time(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"[right-pregrasp] 已写入右臂预抓取到位标记：{destination}", flush=True)


def main() -> int:
    args = parse_args()
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    profile = load_workpiece_handoff_profile("1", mode=args.mode)
    grasp = list(profile["right_handoff"]["grasp_joints_deg"])
    cache_compatibility = compatibility(
        mode=args.mode, profile_id=str(profile["profile_id"]), grasp_joints_deg=grasp,
    )
    cached, status = load_cache(workpiece="1", compatibility_data=cache_compatibility)
    config = json.loads((ROOT / "tools/handoff_observation/handoff_config.json").read_text(encoding="utf-8"))
    config["max_velocity"] = min(1.0, float(config.get("max_velocity", 0.05)) * args.speed_multiplier)
    config["max_acceleration"] = min(1.0, float(config.get("max_acceleration", 0.05)) * args.speed_multiplier)
    planner = build_runtime(config, args.planner_host, execute_url="http://127.0.0.1:5667/middle/waypoint/execute")
    try:
        planner.open(required_frames={"body_base", config["right_tool_frame"]})
        right_state = read_gripper_state(args.gripper_host, 2, "right")
        right_opening = float(right_state.get("position", -1.0))
        if right_opening < 95.0:
            print(move_gripper(args.gripper_host, 2, 100.0, 60.0, "right"), flush=True)
        if cached is not None:
            response = cached["response"]
            try:
                handoff.validate_trajectory_start_matches_current_state(
                    response, arm_host=args.arm_host, arm_id=2,
                    label="parallel cached idle-to-pregrasp", tolerance_deg=1.0,
                )
            except RuntimeError as exc:
                # 旧缓存可能是在非 Idle 状态下生成。此时保持并行流程，
                # 但禁止从错误首点直接回放，立即退回在线 RRTstar。
                print(
                    f"[right-pregrasp] 缓存起点与当前右臂不匹配（{exc}）；"
                    "改用在线 RRTstar。",
                    flush=True,
                )
                cached = None
                status = "start_mismatch"
            else:
                source = "trajectory_cache"
                print("[right-pregrasp] 命中 Idle→预抓取历史 RRTstar 轨迹，开始回放。", flush=True)
        if cached is None:
            nominal_object = handoff.taught_handoff_object_transform(
                list(profile["handoff"]["center_body_m"]), 0.0,
            )
            taught_tcp = handoff.taught_right_tcp_from_joints(grasp)
            object_to_right_tcp = handoff.validate_transform(
                np.linalg.inv(nominal_object) @ taught_tcp,
                "parallel_nominal_object_to_right_tcp",
            )
            nominal_grasp = handoff.validate_transform(
                nominal_object @ object_to_right_tcp, "parallel_nominal_right_grasp",
            )
            offset, _ = handoff.pregrasp_offset_body(
                nominal_grasp, nominal_object, 0.05, "object-minus-x",
            )
            nominal_pregrasp = nominal_grasp.copy()
            nominal_pregrasp[:3, 3] += offset
            target, _ = handoff.solve_nearby_right_joints(nominal_pregrasp, grasp)
            response = planner.plan_joint_trajectory("right", target, planner="rrtstar")
            if response is None:
                raise RuntimeError("Planner returned no parallel right pregrasp plan")
            handoff.validate_arm_joint_plan(
                response, target, arm_id=2, label="parallel right pregrasp", tolerance_deg=2.0,
            )
            handoff.validate_trajectory_start_matches_current_state(
                response, arm_host=args.arm_host, arm_id=2,
                label="parallel online idle-to-pregrasp", tolerance_deg=2.0,
            )
            source = f"online_rrtstar_cache_miss:{status}"
            print(
                f"[right-pregrasp] 缓存不可用（{status}），"
                "按名义高位交接姿态在线 RRTstar。",
                flush=True,
            )
        trajectory = (response.get("data") or {}).get("trajectory") or []
        if not trajectory:
            raise RuntimeError("right pregrasp trajectory is empty")
        endpoint = [float(value) for value in trajectory[-1]]
        planner.execute_planned_result(response)
        handoff.wait_for_joint_target(
            args.arm_host, 2, endpoint, 0.5, "parallel right pregrasp", timeout_s=90.0,
        )
        if source.startswith("online_rrtstar"):
            save_cache(workpiece="1", compatibility_data=cache_compatibility, response=response)
        publish_marker(args.marker_file, endpoint=endpoint, source=source)
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
