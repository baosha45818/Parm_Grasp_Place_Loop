#!/usr/bin/env python3
"""Joint-angle movement CLI plus shared pose helpers for move_by_eef_pose."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


def import_sdk() -> Any:
    try:
        return importlib.import_module("parm_sdk")
    except ImportError:
        override = os.environ.get("PARM_SDK_WHEEL", "").strip()
        candidates = [Path(override)] if override else []
        candidates.extend(sorted(
            Path("/home/dainanjun/Public/parm_sdk_1.0.0").glob("parm_sdk-*.whl"),
            reverse=True,
        ))
        for wheel in candidates:
            if wheel.is_file():
                sys.path.insert(0, str(wheel))
                try:
                    return importlib.import_module("parm_sdk")
                except ImportError:
                    continue
    raise ImportError("无法导入 parm_sdk；请安装 SDK 或设置 PARM_SDK_WHEEL。")


def finite(values: Any, count: int, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} 必须恰好包含 {count} 个数。")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 必须全为有限数。")
    return result


def rpy_to_rotvec(rpy: list[float]) -> list[float]:
    """RPY (rad, ZYX composition) -> axis-angle rotation vector (rad)."""
    roll, pitch, yaw = finite(rpy, 3, "rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    cos_theta = max(-1.0, min(1.0, (rotation[0][0] + rotation[1][1] + rotation[2][2] - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-10:
        return [0.0, 0.0, 0.0]
    sine = math.sin(theta)
    if abs(sine) < 1e-8:  # near 180 degrees
        axis = [
            math.sqrt(max(0.0, (rotation[0][0] + 1.0) / 2.0)),
            math.sqrt(max(0.0, (rotation[1][1] + 1.0) / 2.0)),
            math.sqrt(max(0.0, (rotation[2][2] + 1.0) / 2.0)),
        ]
        if rotation[2][1] - rotation[1][2] < 0:
            axis[0] = -axis[0]
        if rotation[0][2] - rotation[2][0] < 0:
            axis[1] = -axis[1]
        if rotation[1][0] - rotation[0][1] < 0:
            axis[2] = -axis[2]
    else:
        axis = [
            (rotation[2][1] - rotation[1][2]) / (2.0 * sine),
            (rotation[0][2] - rotation[2][0]) / (2.0 * sine),
            # The third skew component is R[1,0] - R[0,1].  This must
            # match data_collector/_rotmat_to_rotvec exactly.
            (rotation[1][0] - rotation[0][1]) / (2.0 * sine),
        ]
    norm = math.sqrt(sum(component * component for component in axis))
    if norm < 1e-10:
        raise ValueError("无法将 rpy 转换为旋转向量。")
    return [component / norm * theta for component in axis]


def rotvec_to_quat(rotvec: list[float]) -> list[float]:
    x, y, z = finite(rotvec, 3, "rotation vector")
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    s = math.sin(angle / 2.0) / angle
    return [x * s, y * s, z * s, math.cos(angle / 2.0)]


def quat_multiply(first: list[float], second: list[float]) -> list[float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + az * bw + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def rotate_vector(quat: list[float], vector: list[float]) -> list[float]:
    qx, qy, qz, qw = quat
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def compose_pose(first: list[float], second: list[float]) -> list[float]:
    """Compose two [xyz, rotvec] poses."""
    first_quat, second_quat = rotvec_to_quat(first[3:]), rotvec_to_quat(second[3:])
    offset = rotate_vector(first_quat, second[:3])
    return [first[0] + offset[0], first[1] + offset[1], first[2] + offset[2]] + quat_to_rotvec(
        quat_multiply(first_quat, second_quat)
    )


def quat_to_rotvec(quat: list[float]) -> list[float]:
    qx, qy, qz, qw = finite(quat, 4, "quaternion")
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("四元数不能为零。")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    sine = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sine < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sine, qw)
    return [qx / sine * angle, qy / sine * angle, qz / sine * angle]


def rotvec_to_rpy(rotvec: list[float]) -> list[float]:
    """Axis-angle rotation vector -> RPY (rad, ZYX composition)."""
    qx, qy, qz, qw = rotvec_to_quat(rotvec)
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r10 = 2.0 * (qx * qy + qz * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    # At pitch=±90° yaw and roll are coupled.  Selecting roll=0 is a valid
    # canonical representation; normal operating poses should avoid this
    # singularity for planar-yaw motion.
    if abs(math.cos(pitch)) < 1e-8:
        return [0.0, pitch, math.atan2(-2.0 * (qx * qy - qz * qw), r00)]
    return [math.atan2(r21, r22), pitch, math.atan2(r10, r00)]


def shortest_angle_delta(start: float, end: float) -> float:
    """Signed shortest angular difference in [-pi, pi)."""
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def slerp(q0: list[float], q1: list[float], alpha: float) -> list[float]:
    """Shortest-path quaternion interpolation, matching collector move_to_pose."""
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1, dot = [-item for item in q1], -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        result = [a + alpha * (b - a) for a, b in zip(q0, q1)]
        norm = math.sqrt(sum(item * item for item in result))
        return [item / norm for item in result]
    theta = math.acos(dot)
    sine = math.sin(theta)
    return [
        math.sin((1.0 - alpha) * theta) / sine * a
        + math.sin(alpha * theta) / sine * b
        for a, b in zip(q0, q1)
    ]


def smoothstep5(alpha: float) -> float:
    """Quintic timing: zero velocity and acceleration at both ends."""
    alpha = max(0.0, min(1.0, alpha))
    return alpha ** 3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def minimum_scurve_duration(distance: float, max_speed: float, max_acceleration: float) -> float:
    if distance <= 1e-12:
        return 0.0
    # h(u)=10u³-15u⁴+6u⁵: max h'=1.875, max |h''|=10/sqrt(3).
    return max(
        1.875 * distance / max_speed,
        math.sqrt((10.0 / math.sqrt(3.0)) * distance / max_acceleration),
    )


def position_error_m(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def orientation_error_rad(first: list[float], second: list[float]) -> float:
    q0, q1 = rotvec_to_quat(first[3:]), rotvec_to_quat(second[3:])
    dot = max(-1.0, min(1.0, abs(sum(a * b for a, b in zip(q0, q1)))))
    return 2.0 * math.acos(dot)


def pose_to_command(raw_pose: Any) -> list[float]:
    """Convert saved eef_pose to internal [xyz, axis-angle rotvec]."""
    if isinstance(raw_pose, dict):
        xyz = finite(raw_pose.get("xyz"), 3, "eef_pose.xyz")
        if "rpy" in raw_pose:
            return xyz + rpy_to_rotvec(raw_pose["rpy"])
        for key in ("rotvec", "axis_angle", "rxryrz", "rotation_vector"):
            if key in raw_pose:
                return xyz + finite(raw_pose[key], 3, f"eef_pose.{key}")
        raise ValueError("eef_pose 缺少 rpy 或旋转向量字段。")
    # 与采集器一致：扁平 6 元末端位姿按 SDK servo-pose 的旋转向量解释。
    return finite(raw_pose, 6, "eef_pose")


def load_pose_file(
    path: Path, arm_id: int, target: str, prefer_driver_pose: bool = False,
) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = "left" if arm_id == 1 else "right"
    saved_arm = (data.get("arms") or {}).get(key)
    if not isinstance(saved_arm, dict):
        raise ValueError(f"文件中没有 {key} 臂数据，不能用于 arm_id={arm_id}。")
    if target == "joint":
        return finite(saved_arm.get("joint_positions_deg"), 7, "joint_positions_deg")
    if prefer_driver_pose and "driver_eef_pose" in saved_arm:
        return pose_to_command(saved_arm["driver_eef_pose"])
    return pose_to_command(saved_arm.get("eef_pose"))


def arm_state(arm: Any, arm_id: int) -> dict[str, Any]:
    """Return one fresh per-arm status frame, as the collector does."""
    state = arm.get_state()
    key = "left" if arm_id == 1 else "right"
    per_arm = ((state or {}).get("state") or {}).get(key)
    if not isinstance(per_arm, dict):
        raise RuntimeError("当前机械臂状态不可用，无法开始笛卡尔移动。")
    return per_arm


def current_pose(arm: Any, arm_id: int) -> list[float]:
    per_arm = arm_state(arm, arm_id)
    return pose_to_command(per_arm.get("eef_pose"))


def ensure_ready(arm: Any, arm_id: int, recover_timeout_s: float = 10.0) -> None:
    """Match the collector's recover-before-motion sequence."""
    state = arm.get_state()
    if state is None:
        raise RuntimeError("尚未收到机械臂状态。")
    system_state = str(state.get("system_state", "")).strip().lower()
    if system_state == "error":
        print("检测到 ERROR，正在 clear_error() 并等待恢复...")
        arm.clear_error()
        deadline = time.monotonic() + recover_timeout_s
        while time.monotonic() < deadline:
            state = arm.get_state()
            system_state = str((state or {}).get("system_state", "")).strip().lower()
            if system_state == "running":
                break
            if system_state == "estop":
                raise RuntimeError(f"clear_error() 后进入 ESTOP：{state.get('estop_reason')!r}")
            time.sleep(0.05)
        else:
            raise RuntimeError(f"clear_error() 后未能恢复 RUNNING，当前={system_state!r}")
    if system_state == "estop":
        raise RuntimeError(f"系统处于 ESTOP：{state.get('estop_reason')!r}")
    if system_state != "running":
        raise RuntimeError(f"系统尚未就绪，当前 system_state={system_state!r}")
    key = "left" if arm_id == 1 else "right"
    if not bool((state.get("state") or {}).get(key, {}).get("enabled", False)):
        print("目标手臂未使能，正在 enable()...")
        arm.enable()
        deadline = time.monotonic() + recover_timeout_s
        while time.monotonic() < deadline:
            state = arm.get_state()
            per_arm = ((state or {}).get("state") or {}).get(key, {})
            if bool(per_arm.get("enabled", False)):
                return
            time.sleep(0.05)
        raise RuntimeError("enable() 后目标手臂仍未使能。")


def move_l(
    arm: Any,
    arm_id: int,
    target: list[float],
    linear_speed_mps: float,
    angular_speed_radps: float,
    max_linear_accel_mps2: float,
    max_angular_accel_radps2: float,
    hz: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    settle_timeout_s: float,
    planar_xy_yaw: bool = False,
    tcp: list[float] | None = None,
) -> None:
    """Stream an absolute Cartesian target, optionally for a calibrated TCP."""
    tcp_rpy = finite(tcp if tcp is not None else [0.0] * 6, 6, "servo tcp")
    tcp_pose = tcp_rpy[:3] + rpy_to_rotvec(tcp_rpy[3:])
    start = compose_pose(current_pose(arm, arm_id), tcp_pose)
    start_rpy: list[float] | None = None
    target_rpy: list[float] | None = None
    if planar_xy_yaw:
        start_rpy = rotvec_to_rpy(start[3:])
        target_rpy = rotvec_to_rpy(target[3:])
        # Lock the out-of-plane coordinates to their actual values at motion
        # start; only target X/Y and yaw are accepted from the requested pose.
        target = [target[0], target[1], start[2]] + rpy_to_rotvec(
            [start_rpy[0], start_rpy[1], target_rpy[2]]
        )
    distance = position_error_m(target, start)
    q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
    angle = orientation_error_rad(target, start)
    print(f"当前末端 [xyz,rotvec]: {[round(value, 6) for value in start]}")
    print(f"目标末端 [xyz,rotvec]: {[round(value, 6) for value in target]}")
    if distance <= position_tolerance_m and angle <= orientation_tolerance_rad:
        print("当前末端已在目标容差内；无需移动。")
        return
    period = 1.0 / hz
    linear_duration = minimum_scurve_duration(distance, linear_speed_mps, max_linear_accel_mps2)
    angular_duration = minimum_scurve_duration(angle, angular_speed_radps, max_angular_accel_radps2)
    duration = max(linear_duration, angular_duration, period)
    steps = max(1, math.ceil(duration * hz))
    print(
        f"[move_to_pose] distance={distance:.4f}m rotation={math.degrees(angle):.2f}deg "
        f"duration={duration:.2f}s (linear={linear_duration:.2f}s angular={angular_duration:.2f}s) "
        f"steps={steps} @ {hz:.1f}Hz"
    )
    arm.start_servo_pose(
        arm_id, pose_type="absolute", frame="world", tcp=tcp_rpy,
        input_frequency_hz=hz,
    )
    try:
        deadline = time.monotonic()
        for index in range(1, steps + 1):
            alpha = smoothstep5(index / float(steps))
            waypoint = [start[axis] + alpha * (target[axis] - start[axis]) for axis in range(3)]
            if planar_xy_yaw:
                assert start_rpy is not None and target_rpy is not None
                yaw = start_rpy[2] + alpha * shortest_angle_delta(start_rpy[2], target_rpy[2])
                waypoint += rpy_to_rotvec([start_rpy[0], start_rpy[1], yaw])
            else:
                waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
            arm.push_servo_pose(arm_id, waypoint)
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))

        # Keep pushing the final target and verify it from the status stream.
        settle_deadline = time.monotonic() + settle_timeout_s
        final_pose = start
        while time.monotonic() < settle_deadline:
            arm.push_servo_pose(arm_id, target)
            final_pose = compose_pose(current_pose(arm, arm_id), tcp_pose)
            pos_error = position_error_m(target, final_pose)
            rot_error = orientation_error_rad(target, final_pose)
            if pos_error <= position_tolerance_m and rot_error <= orientation_tolerance_rad:
                print(f"[move_to_pose] reached: pos_error={pos_error * 1000.0:.2f}mm rot_error={math.degrees(rot_error):.2f}deg")
                return
            time.sleep(period)
        pos_error = position_error_m(target, final_pose)
        rot_error = orientation_error_rad(target, final_pose)
        moved = position_error_m(final_pose, start)
        raise RuntimeError(
            "末端位姿未在超时内到位："
            f"位置误差={pos_error * 1000.0:.1f} mm，旋转误差={math.degrees(rot_error):.2f}°，"
            f"实际位移={moved * 1000.0:.1f} mm。"
        )
    finally:
        try:
            arm.stop_servo_pose(arm_id)
        except Exception:
            pass


def _legacy_main() -> int:
    parser = argparse.ArgumentParser(description="PARM 单臂 move_j / 末端位姿移动")
    parser.add_argument("--arm-id", type=int, choices=(1, 2), required=True, help="1=左臂，2=右臂")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--joint-positions-deg", nargs=7, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
    source.add_argument("--eef-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                        help="xyz（米）+ RPY（弧度）")
    source.add_argument("--pose-file", type=Path, help="saved_poses 中的 JSON 文件")
    parser.add_argument("--file-target", choices=("joint", "pose"), default="joint", help="使用 --pose-file 时选取的字段")
    parser.add_argument("--joint-speed", type=float, default=0.2, help="move_j 速度比例 [0,1]")
    parser.add_argument("--linear-speed", type=float, default=0.05, help="move_l 最大线速度 m/s")
    parser.add_argument("--angular-speed", type=float, default=0.30, help="move_l 最大角速度 rad/s")
    parser.add_argument("--max-linear-accel", type=float, default=0.10, help="move_l 最大线加速度 m/s²")
    parser.add_argument("--max-angular-accel", type=float, default=0.60, help="move_l 最大角加速度 rad/s²")
    parser.add_argument("--servo-hz", type=float, default=50.0, choices=(10.0, 20.0, 25.0, 50.0, 100.0), help="servo-pose 推送频率 Hz（与采集器一致，默认 50）")
    parser.add_argument("--position-tolerance-m", type=float, default=0.0005, help="末端位置到位容差 m（默认 0.0005）")
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.035, help="末端旋转到位容差 rad（默认 0.035）")
    parser.add_argument("--settle-timeout", type=float, default=8.0, help="末端到位等待秒数（默认 8）")
    parser.add_argument("--execute", action="store_true", help="实际下发运动；省略时仅打印目标")
    args = parser.parse_args()

    if not 0.0 <= args.joint_speed <= 1.0:
        parser.error("--joint-speed 必须在 [0, 1] 内")
    if min(args.linear_speed, args.angular_speed, args.max_linear_accel, args.max_angular_accel,
           args.servo_hz, args.position_tolerance_m, args.orientation_tolerance_rad,
           args.settle_timeout) <= 0:
        parser.error("move_l 的速度、加速度、频率、容差和超时必须大于 0")

    is_joint = args.joint_positions_deg is not None or (args.pose_file is not None and args.file_target == "joint")
    if args.pose_file is not None:
            target = load_pose_file(
                args.pose_file, args.arm_id, args.file_target, prefer_driver_pose=True,
            )
    elif is_joint:
        target = finite(args.joint_positions_deg, 7, "joint_positions_deg")
    else:
        raw_pose = finite(args.eef_pose, 6, "eef_pose")
        target = raw_pose[:3] + rpy_to_rotvec(raw_pose[3:])

    kind = "move_j 关节角（度）" if is_joint else "末端位姿 xyz(m)+rotvec(rad)"
    print(f"目标 arm_id={args.arm_id}，{kind}: {target}")
    if not args.execute:
        print("这是预览，未发送运动命令。确认无碰撞风险后添加 --execute。")
        return 0

    arm = None
    try:
        arm = import_sdk().Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        if is_joint:
            arm.move_j(args.arm_id, target, speed=args.joint_speed)
            print("move_j 已完成。")
        else:
            move_l(
                arm, args.arm_id, target, args.linear_speed, args.angular_speed,
                args.max_linear_accel, args.max_angular_accel, args.servo_hz,
                args.position_tolerance_m, args.orientation_tolerance_rad,
                args.settle_timeout,
            )
            print("末端位姿移动指令已完成。")
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if arm is not None:
            arm.close()
    return 0


def main() -> int:
    """CLI entry point: move one arm only by seven joint angles."""
    parser = argparse.ArgumentParser(description="PARM 单臂基于关节角度的 move_j")
    parser.add_argument("--arm-id", type=int, choices=(1, 2), required=True, help="1=左臂，2=右臂")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--joint-positions-deg", nargs=7, type=float,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
    source.add_argument("--pose-file", type=Path, help="读取保存文件中的 joint_positions_deg")
    parser.add_argument("--joint-speed", type=float, default=0.2, help="move_j 速度比例 [0,1]")
    parser.add_argument("--execute", action="store_true", help="实际下发运动；省略时仅打印目标")
    args = parser.parse_args()

    if not 0.0 <= args.joint_speed <= 1.0:
        parser.error("--joint-speed 必须在 [0, 1] 内")
    target = (
        load_pose_file(args.pose_file, args.arm_id, "joint")
        if args.pose_file is not None
        else finite(args.joint_positions_deg, 7, "joint_positions_deg")
    )
    print(f"目标 arm_id={args.arm_id}，move_j 关节角（度）: {target}")
    if not args.execute:
        print("这是预览，未发送运动命令。确认无碰撞风险后添加 --execute。")
        return 0

    arm = None
    try:
        arm = import_sdk().Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        arm.move_j(args.arm_id, target, speed=args.joint_speed)
        print("move_j 已完成。")
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if arm is not None:
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
