#!/usr/bin/env python3
"""右臂独立慢速往返测试：直接使用 PARM Planner 和 Arm SDK。"""

from __future__ import annotations

import argparse
import math
import time

from scipy.spatial.transform import Rotation


ARM_ID = 2
ARM_KEY = "right"
GROUP = "right_arm"
TOOL_FRAME = "right_hand_transfer_end"
BASE_FRAME = "body_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument(
        "--motion-mode", choices=("planner", "servo-pose"), default="planner",
        help="planner=body_base -Y；servo-pose=SDK world -Y",
    )
    parser.add_argument("--distance-m", type=float, default=0.10)
    parser.add_argument("--planner-speed", type=float, default=0.02)
    parser.add_argument("--servo-speed-mps", type=float, default=0.01)
    parser.add_argument("--servo-hz", type=float, default=50.0)
    parser.add_argument("--cycles", type=int, default=0, help="0=持续循环直到 Ctrl+C")
    parser.add_argument("--keep-endpoint", action="store_true")
    parser.add_argument("--auto-confirm", action="store_true")
    args = parser.parse_args()
    if not 0.005 <= args.distance_m <= 0.15:
        parser.error("--distance-m 必须在 0.005 到 0.15 m 之间")
    if not 0.01 <= args.planner_speed <= 1.0:
        parser.error("--planner-speed 必须在 0.01 到 1.0 之间")
    if args.servo_speed_mps <= 0.0 or not 10.0 <= args.servo_hz <= 100.0:
        parser.error("ServoPose 速度必须为正，频率必须在 10 到 100 Hz 之间")
    if args.cycles < 0 or (args.keep_endpoint and args.cycles != 1):
        parser.error("--cycles 必须非负；--keep-endpoint 只能配合 --cycles 1")
    return args


def planner_pose(response: dict) -> tuple[list[float], list[float]]:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [float(translation[axis]) for axis in "xyz"]
    xyzw = [float(rotation[axis]) for axis in "xyzw"]
    if not all(math.isfinite(value) for value in xyz + xyzw):
        raise RuntimeError(f"规划器 TF 含非有限值: {response}")
    return xyz, xyzw


def servo_pose(arm: object) -> list[float]:
    state = arm.get_state() or {}  # type: ignore[attr-defined]
    raw = ((state.get("state") or {}).get(ARM_KEY) or {}).get("eef_pose")
    if not isinstance(raw, dict):
        raise RuntimeError(f"右臂 eef_pose 不可用: {raw!r}")
    xyz = raw.get("xyz")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise RuntimeError(f"右臂 eef_pose.xyz 不可用: {raw!r}")
    rotvec = raw.get("rotvec")
    if not isinstance(rotvec, (list, tuple)) or len(rotvec) != 3:
        rpy = raw.get("rpy")
        if not isinstance(rpy, (list, tuple)) or len(rpy) != 3:
            raise RuntimeError(f"右臂 eef_pose 旋转不可用: {raw!r}")
        rotvec = Rotation.from_euler("xyz", rpy).as_rotvec().tolist()
    return [float(value) for value in xyz] + [float(value) for value in rotvec]


def run_planner(args: argparse.Namespace) -> None:
    from parm_sdk import Arm, Planner

    with Planner(host=args.planner_host) as planner:
        frames = set(planner.list_frame_ids())
        missing = {BASE_FRAME, TOOL_FRAME}.difference(frames)
        if missing:
            raise RuntimeError(f"规划器 TF 缺少坐标系: {sorted(missing)}")
        origin_xyz, origin_xyzw = planner_pose(planner.lookup_transform(BASE_FRAME, TOOL_FRAME))
        target_xyz = [origin_xyz[0], origin_xyz[1] - args.distance_m, origin_xyz[2]]
        print(f"origin body_base xyz={origin_xyz}")
        print(f"target body_base xyz={target_xyz}")
        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            for label, xyz in (("-Y", target_xyz), ("回程", origin_xyz)):
                if label == "回程" and args.keep_endpoint:
                    break
                print(f"[cycle {cycle}] 规划右臂 {label} LIN。", flush=True)
                result = planner.plan_cart_trajectory(
                    group=GROUP, planner="lin", frame_id=BASE_FRAME, ik_frame=TOOL_FRAME,
                    position=xyz, orientation=origin_xyzw,
                    max_vel=args.planner_speed, max_acc=args.planner_speed,
                )
                data = result.get("data") or {}
                trajectory = data.get("trajectory") or []
                frequency = float(data.get("sample_frequency", 10.0))
                if int(data.get("arm_id", -1)) != ARM_ID or not trajectory:
                    raise RuntimeError(f"右臂规划器未返回可执行轨迹: {result}")
                with Arm(host=args.arm_host) as arm:
                    arm.follow_trajectory(ARM_ID, trajectory, frequency)
                print(f"[cycle {cycle}] 右臂 {label} 完成。", flush=True)


def run_servo_pose(args: argparse.Namespace) -> None:
    from parm_sdk import Arm

    arm = None
    started = False
    try:
        arm = Arm(host=args.arm_host)
        origin = servo_pose(arm)
        target = origin.copy()
        target[1] -= args.distance_m
        print(f"origin SDK world xyz={origin[:3]}")
        print(f"target SDK world xyz={target[:3]}")
        arm.start_servo_pose(ARM_ID, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                             input_frequency_hz=args.servo_hz)
        started = True
        period = 1.0 / args.servo_hz
        steps = max(2, math.ceil(args.distance_m / args.servo_speed_mps * args.servo_hz))
        deadline = time.monotonic()
        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            for label, start, end in (("-Y", origin, target), ("回程", target, origin)):
                if label == "回程" and args.keep_endpoint:
                    break
                print(f"[cycle {cycle}] 右臂 ServoPose {label}。", flush=True)
                for index in range(1, steps + 1):
                    alpha = index / float(steps)
                    arm.push_servo_pose(ARM_ID, [
                        start[axis] + alpha * (end[axis] - start[axis]) for axis in range(6)
                    ])
                    deadline += period
                    time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        if started and arm is not None:
            arm.stop_servo_pose(ARM_ID)
        if arm is not None:
            arm.close()


def main() -> None:
    args = parse_args()
    if not args.auto_confirm:
        input(f"右臂将以 {args.motion_mode} 模式慢速往返。确认安全后按回车：")
    try:
        if args.motion_mode == "planner":
            run_planner(args)
        else:
            run_servo_pose(args)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，发送 cancel_task(2)。", flush=True)
        from parm_sdk import Arm
        with Arm(host=args.arm_host) as arm:
            arm.cancel_task(ARM_ID)
        raise


if __name__ == "__main__":
    main()
