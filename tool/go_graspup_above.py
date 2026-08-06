#!/usr/bin/env python3
"""左臂先到固定 graspup，再沿 body_base +Z 规划并上移指定高度。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


SDK_WHEEL = next(
    iter(sorted((Path(__file__).resolve().parents[1] / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True)),
    None,
)
GRASPUP_JOINTS_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="左臂 RRTStar 到 graspup，再沿 body_base +Z 上移。"
    )
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument(
        "--lift-planner",
        choices=("rrtstar", "rrtconnect"),
        default="rrtstar",
        help="graspup 上方目标的规划算法；默认 rrtstar。",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed 必须在 (0, 1] 内")
    if not 0.0 < args.lift_m <= 0.20:
        parser.error("--lift-m 必须在 (0, 0.20] m 内")
    if not SDK_WHEEL.is_file():
        raise FileNotFoundError(f"找不到新版 SDK wheel: {SDK_WHEEL}")

    sys.path.insert(0, str(SDK_WHEEL))
    from parm_sdk import Arm, Planner

    arm = None
    planner = None
    try:
        arm = Arm(host=args.arm_host)
        arm_state = arm.get_state() or {}
        if str(arm_state.get("system_state", "")).strip().upper() != "RUNNING":
            raise RuntimeError(
                f"机械臂未处于 RUNNING，当前={arm_state.get('system_state')!r}"
            )
        if not bool(((arm_state.get("state") or {}).get("left") or {}).get("enabled")):
            raise RuntimeError("左臂未使能，拒绝发送 graspup 轨迹")
        planner = Planner(host=args.planner_host)
        frames = set(planner.list_frame_ids())
        required_frames = {"body_base", "left_hand_transfer_end"}
        missing = sorted(required_frames.difference(frames))
        if missing:
            raise RuntimeError(f"规划器 TF 缺少坐标系: {missing}")

        joint_payload = {
            "waypoints": [{
                "name": "go_graspup_joint_target",
                "type": "joint",
                "group": "left_arm",
                "planner": "rrtstar",
                "max_velocity_scaling_factor": args.speed,
                "max_acceleration_scaling_factor": args.speed,
                "joint_names": [f"left_arm_joint{index}" for index in range(1, 8)],
                "joint_values": GRASPUP_JOINTS_DEG,
            }],
            "topp": {
                "max_vel_factor": args.speed,
                "max_acc_factor": args.speed,
                "final_phase_ratio": 0.05,
                "final_phase_oversample": 2.0,
            },
        }
        print("=== graspup 固定关节目标 ===")
        print(f"joint_values (deg) = {GRASPUP_JOINTS_DEG}")
        joint_result = planner._trajectory.send_payload(joint_payload, timeout=30.0)
        joint_data = joint_result.get("data") or {}
        joint_trajectory = joint_data.get("trajectory") or []
        joint_frequency = float(joint_data.get("sample_frequency", 10.0))
        if int(joint_data.get("arm_id", -1)) != 1 or not joint_trajectory:
            raise RuntimeError(f"graspup 关节 RRTStar 未返回左臂轨迹: {joint_result}")
        print(
            f"graspup RRTStar 规划成功：{len(joint_trajectory)} 个关节路点，"
            f"sample_frequency={joint_frequency} Hz"
        )
        if not args.execute:
            print("未传 --execute：未向机械臂发送运动指令。")
            return

        arm.follow_trajectory(1, joint_trajectory, sample_frequency=joint_frequency)
        print("graspup 轨迹已下发，等待左臂实际到位...")
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = ((state.get("state") or {}).get("left") or {}).get("position") or []
            if len(actual) == 7:
                error_deg = max(
                    abs((float(actual[index]) - GRASPUP_JOINTS_DEG[index] + 180.0) % 360.0 - 180.0)
                    for index in range(7)
                )
                if error_deg <= 0.5:
                    print(f"左臂已到 graspup，最大关节误差={error_deg:.3f} deg")
                    break
            time.sleep(0.05)
        else:
            raise RuntimeError("左臂未在 90 秒内到达 graspup")

        current_response = planner.lookup_transform("body_base", "left_hand_transfer_end")
        current_data = current_response.get("data") or current_response
        translation = current_data.get("translation") or {}
        rotation = current_data.get("rotation") or {}
        graspup_xyz = [float(translation[axis]) for axis in "xyz"]
        graspup_xyzw = [float(rotation[axis]) for axis in "xyzw"]
        if not all(math.isfinite(value) for value in graspup_xyz + graspup_xyzw):
            raise RuntimeError(f"graspup TF 含无效数值: {current_response}")
        target_xyz = [graspup_xyz[0], graspup_xyz[1], graspup_xyz[2] + args.lift_m]
        print("\n=== graspup 上方目标 ===")
        print("frame_id = body_base")
        print("ik_frame = left_hand_transfer_end")
        print(f"graspup xyz (m) = {graspup_xyz}")
        print(f"target  xyz (m) = {target_xyz}")
        print(f"quaternion xyzw = {graspup_xyzw}")
        print(f"lift = body_base +Z {args.lift_m:.3f} m")

        if args.lift_planner == "rrtstar":
            lift_payload = {
                "waypoints": [{
                    "name": "graspup_body_base_z_lift",
                    "type": "cart",
                    "group": "left_arm",
                    "planner": "rrtstar",
                    "max_velocity_scaling_factor": args.speed,
                    "max_acceleration_scaling_factor": args.speed,
                    "frame_id": "body_base",
                    "ik_frame": "left_hand_transfer_end",
                    "position": target_xyz,
                    "orientation": graspup_xyzw,
                }],
                "topp": {"max_vel_factor": args.speed, "max_acc_factor": args.speed},
            }
            lift_result = planner._trajectory.send_payload(lift_payload, timeout=30.0)
        else:
            lift_payload = {
                "waypoints": [{
                    "name": "graspup_body_base_z_lift",
                    "type": "cart",
                    "group": "left_arm",
                    "planner": "ompl",
                    "max_velocity_scaling_factor": args.speed,
                    "max_acceleration_scaling_factor": args.speed,
                    "frame_id": "body_base",
                    "ik_frame": "left_hand_transfer_end",
                    "position": target_xyz,
                    "orientation": graspup_xyzw,
                }],
                "topp": {
                    "max_vel_factor": args.speed,
                    "max_acc_factor": args.speed,
                    "final_phase_ratio": 0.05,
                    "final_phase_oversample": 2.0,
                },
            }
            lift_result = planner._trajectory.send_payload(lift_payload, timeout=30.0)

        lift_data = lift_result.get("data") or {}
        lift_trajectory = lift_data.get("trajectory") or []
        lift_frequency = float(lift_data.get("sample_frequency", 10.0))
        if int(lift_data.get("arm_id", -1)) != 1 or not lift_trajectory:
            raise RuntimeError(f"graspup 上方规划未返回左臂轨迹: {lift_result}")
        print(
            f"{args.lift_planner.upper()} 上移规划成功：{len(lift_trajectory)} 个关节路点，"
            f"sample_frequency={lift_frequency} Hz"
        )
        arm.follow_trajectory(1, lift_trajectory, sample_frequency=lift_frequency)
        print("graspup 上方轨迹已下发。")
    except KeyboardInterrupt:
        if arm is not None:
            try:
                arm.cancel_task(1)
                print("收到 Ctrl+C：已调用 left arm cancel_task(1)。")
            except Exception as exc:
                print(f"收到 Ctrl+C，但 cancel_task(1) 失败：{exc}", file=sys.stderr)
        raise
    finally:
        if planner is not None:
            planner.close()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    main()
