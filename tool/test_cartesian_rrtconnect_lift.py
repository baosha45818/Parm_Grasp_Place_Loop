#!/usr/bin/env python3
"""单独测试：直接发送完整 cart+ompl 请求，让当前末端沿 body_base +Z 上移。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


SDK_WHEEL = next(iter(sorted((Path(__file__).resolve().parents[1] / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True)), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="直接用完整 cart+ompl/RRTConnect 请求规划当前末端沿 body_base +Z 上移。"
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--group", default="left_arm")
    parser.add_argument("--ik-frame", default="left_hand_transfer_end")
    parser.add_argument("--frame-id", default="body_base")
    parser.add_argument("--arm-id", type=int, default=1)
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.frame_id != "body_base":
        parser.error("--frame-id 必须为 body_base")
    if args.arm_id not in {1, 2}:
        parser.error("--arm-id 必须为 1（左臂）或 2（右臂）")
    expected_arm_id = {
        "left_arm": 1,
        "right_arm": 2,
    }.get(args.group)
    if expected_arm_id is None:
        parser.error("--group 必须为 left_arm 或 right_arm")
    if args.arm_id != expected_arm_id:
        parser.error(
            f"--group {args.group} 必须配合 --arm-id {expected_arm_id}，"
            f"不能使用 --arm-id {args.arm_id}"
        )
    if not 0.0 < args.lift_m <= 0.20:
        parser.error("--lift-m 必须在 (0, 0.20] m 内")
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed 必须在 (0, 1] 内")
    if not SDK_WHEEL.is_file():
        raise FileNotFoundError(f"找不到新版 SDK wheel: {SDK_WHEEL}")

    sys.path.insert(0, str(SDK_WHEEL))
    from parm_sdk import Planner

    with Planner(host=args.planner_host) as planner:
        frames = set(planner.list_frame_ids())
        missing = sorted({args.frame_id, args.ik_frame}.difference(frames))
        if missing:
            raise RuntimeError(f"规划器 TF 缺少坐标系: {missing}")

        current_response = planner.lookup_transform(args.frame_id, args.ik_frame)
        current_data = current_response.get("data") or current_response
        translation = current_data.get("translation") or {}
        rotation = current_data.get("rotation") or {}
        try:
            current_xyz = [float(translation[axis]) for axis in "xyz"]
            current_xyzw = [float(rotation[axis]) for axis in "xyzw"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"规划器返回的当前 TF 格式无效: {current_response}") from exc
        if not all(math.isfinite(value) for value in current_xyz + current_xyzw):
            raise RuntimeError(f"规划器返回的当前 TF 含非有限数: {current_response}")

        target_xyz = [current_xyz[0], current_xyz[1], current_xyz[2] + args.lift_m]
        print(f"group = {args.group}")
        print(f"frame_id = {args.frame_id}")
        print(f"ik_frame = {args.ik_frame}")
        print(f"arm_id = {args.arm_id}")
        print(f"max_vel = {args.speed}")
        print(f"max_acc = {args.speed}")
        print(f"current xyz (m) = {current_xyz}")
        print(f"current quaternion xyzw = {current_xyzw}")
        print(f"target xyz (m) = {target_xyz}")
        print(f"target quaternion xyzw = {current_xyzw}")

        # SDK 1.0.0 公开方法拒绝 cart+ompl；此处直接发送与其余 Planner
        # 请求相同的 ZMQ wire payload。RRTConnect profile 服务端固定把
        # planner=ompl 路由到 RRTConnectkConfigDefault。
        payload = {
            "waypoints": [{
                "name": "rrtconnect_current_pose_z_lift",
                "type": "cart",
                "group": args.group,
                "planner": "ompl",
                "max_velocity_scaling_factor": args.speed,
                "max_acceleration_scaling_factor": args.speed,
                "frame_id": args.frame_id,
                "ik_frame": args.ik_frame,
                "position": target_xyz,
                "orientation": current_xyzw,
            }],
            "topp": {
                "max_vel_factor": args.speed,
                "max_acc_factor": args.speed,
                "final_phase_ratio": 0.05,
                "final_phase_oversample": 2.0,
            },
        }
        print("\n提交给 RRTConnect 规划器的完整 waypoint:")
        print(json.dumps(payload["waypoints"][0], ensure_ascii=False, indent=2))
        result = planner._trajectory.send_payload(payload, timeout=30.0)

    data = result.get("data") or {}
    trajectory = data.get("trajectory") or []
    if not trajectory:
        raise RuntimeError(f"RRTConnect 未返回可执行轨迹: {result}")
    returned_arm_id = int(data.get("arm_id", -1))
    if returned_arm_id != args.arm_id:
        raise RuntimeError(
            f"规划器返回 arm_id={returned_arm_id}，预期 --arm-id={args.arm_id}"
        )
    sample_frequency = float(data.get("sample_frequency", 10.0))
    print(
        f"\nRRTConnect 规划成功：{len(trajectory)} 个关节路点，"
        f"sample_frequency={sample_frequency} Hz"
    )
    if not args.execute:
        print("未传 --execute：只完成规划，未向机械臂发送运动指令。")
        return

    from parm_sdk import Arm

    with Arm(host=args.arm_host) as arm:
        arm.follow_trajectory(
            arm_id=args.arm_id,
            trajectory=[[float(value) for value in point] for point in trajectory],
            sample_frequency=sample_frequency,
        )
    print("RRTConnect 上移轨迹已执行完成。")


if __name__ == "__main__":
    main()
