#!/usr/bin/env python3
"""只读核对 ARM driver 与规划器的实时七关节状态。

关节同步判定直接比较 ``Arm.get_state().state.<arm>.position`` 与规划器
Docker 容器内 ROS2 ``/joint_states``，不使用任何 SDK→Planner bridge。
本工具不发送运动、示教、夹爪或使能命令。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


LEFT_JOINTS = [f"left_arm_joint{index}" for index in range(1, 8)]
RIGHT_JOINTS = [f"right_arm_joint{index}" for index in range(1, 8)]
FRAME_CANDIDATES = {
    "left": [
        "left_arm_end_link",
        "left_hand_transfer_base",
        "left_hand_transfer_end",
        "left_eef",
    ],
    "right": [
        "right_arm_end_link",
        "right_hand_transfer_base",
        "right_hand_transfer_end",
        "right_eef",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1")
    )
    parser.add_argument(
        "--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--sdk-joint-unit",
        choices=("deg", "rad"),
        default=os.environ.get("PARM_JOINT_UNIT", "deg"),
        help="ARM.get_state().state.*.position 的单位；学校现场默认 deg。",
    )
    parser.add_argument("--samples", type=int, default=1, help="连续只读采样次数，默认 1。")
    parser.add_argument("--interval", type=float, default=0.3, help="相邻样本间隔秒数，默认 0.3。")
    parser.add_argument("--joint-tolerance-deg", type=float, default=0.2)
    parser.add_argument(
        "--include-tf-diagnostics",
        action="store_true",
        help="额外读取 body_base 下末端 TF 并记录相对量；不影响关节同步判定。",
    )
    parser.add_argument(
        "--planner-container",
        default="",
        help="可选：运行规划器 ROS 节点的 Docker 容器名；默认自动发现。",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="JSON 输出目录；默认 output/fk_consistency。",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.interval < 0:
        parser.error("--interval must be >= 0")
    if args.joint_tolerance_deg <= 0:
        parser.error("--joint-tolerance-deg must be > 0")
    return args


def run(command: list[str], *, timeout: float) -> str:
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=timeout
    )
    return completed.stdout


def find_planner_container(explicit_name: str) -> str:
    """返回唯一的 Planner 主容器；也允许命令行显式指定。"""
    if explicit_name:
        return explicit_name
    output = run(
        [
            "docker",
            "ps",
            "--filter",
            "label=com.docker.compose.service=plucky_hjt_parm",
            "--format",
            "{{.Names}}",
        ],
        timeout=5.0,
    )
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if len(names) != 1:
        raise RuntimeError(
            "无法唯一确定规划器容器。请启动一个 plucky_hjt_parm 容器，"
            "或通过 --planner-container 明确指定。当前匹配: " + repr(names)
        )
    return names[0]


def planner_joint_state(container: str) -> dict[str, float]:
    """从规划器容器的 ROS2 ``/joint_states`` 读取一帧关节角（弧度）。"""
    command = (
        "source /opt/ros/jazzy/setup.bash && source /app/install/setup.bash && "
        "timeout 5s ros2 topic echo /joint_states --once --csv"
    )
    output = run(
        ["docker", "exec", container, "bash", "-lc", command], timeout=8.0
    ).strip()
    row = next(csv.reader([output]), None)
    if row is None or len(row) < 5:
        raise RuntimeError(f"无法解析规划器 /joint_states CSV: {output!r}")

    # CSV 顺序：sec,nanosec,frame_id, joint_names..., positions...。
    position_start = None
    for index in range(3, len(row)):
        try:
            float(row[index])
        except ValueError:
            continue
        position_start = index
        break
    if position_start is None:
        raise RuntimeError(f"规划器 /joint_states 中没有 position: {output!r}")
    names = row[3:position_start]
    values = row[position_start : position_start + len(names)]
    if not names or len(values) != len(names):
        raise RuntimeError(f"规划器 /joint_states 名称/位置长度异常: {output!r}")
    return {name: float(value) for name, value in zip(names, values)}


def matrix_from_rpy(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def matrix_from_quaternion(xyzw: list[float]) -> list[list[float]]:
    x, y, z, w = xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def rpy_from_matrix(matrix: list[list[float]]) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(pitch)) < 1e-8:
        return [math.atan2(-matrix[0][1], matrix[1][1]), pitch, 0.0]
    return [
        math.atan2(matrix[2][1], matrix[2][2]),
        pitch,
        math.atan2(matrix[1][0], matrix[0][0]),
    ]


def relative_transform(
    sdk_xyz: list[float],
    sdk_rpy: list[float],
    planner_xyz: list[float],
    planner_xyzw: list[float],
) -> dict[str, Any]:
    """记录 SDK EEF 与 Planner 末端 TF 的相对量，仅作诊断而非同步判据。"""
    r_sdk = matrix_from_rpy(sdk_rpy)
    r_planner = matrix_from_quaternion(planner_xyzw)
    r_sdk_inverse = transpose(r_sdk)
    translation_body = [planner_xyz[i] - sdk_xyz[i] for i in range(3)]
    translation_sdk = mat_vec(r_sdk_inverse, translation_body)
    rotation_sdk = multiply(r_sdk_inverse, r_planner)
    relative_rpy = rpy_from_matrix(rotation_sdk)
    angle_rad = math.acos(
        max(
            -1.0,
            min(1.0, (sum(rotation_sdk[i][i] for i in range(3)) - 1.0) / 2.0),
        )
    )
    return {
        "translation_m_in_sdk_eef": translation_sdk,
        "translation_norm_mm": math.sqrt(sum(value * value for value in translation_sdk))
        * 1000.0,
        "rpy_rad_in_sdk_eef": relative_rpy,
        "rpy_deg_in_sdk_eef": [math.degrees(value) for value in relative_rpy],
        "rotation_angle_deg": math.degrees(angle_rad),
    }


def lookup_transform(planner: Any, source_frame: str, target_frame: str) -> dict[str, Any]:
    response = planner.lookup_transform(source_frame, target_frame)
    data = (response or {}).get("data") or response or {}
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    try:
        xyz = [float(translation[axis]) for axis in ("x", "y", "z")]
        xyzw = [float(rotation[axis]) for axis in ("x", "y", "z", "w")]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"TF {source_frame}->{target_frame} 返回格式异常: {response!r}") from error
    return {"xyz_m": xyz, "quaternion_xyzw": xyzw}


def sdk_pose(arm_state: dict[str, Any], arm_name: str) -> tuple[list[float], list[float]]:
    pose = arm_state.get("eef_pose")
    if not isinstance(pose, dict):
        raise RuntimeError(f"SDK {arm_name}.eef_pose 不存在或不是 dict: {pose!r}")
    try:
        xyz = [float(value) for value in pose["xyz"]]
        rpy = [float(value) for value in pose["rpy"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"SDK {arm_name}.eef_pose 必须包含 xyz/rpy: {pose!r}") from error
    if len(xyz) != 3 or len(rpy) != 3:
        raise RuntimeError(f"SDK {arm_name}.eef_pose xyz/rpy 长度应为 3: {pose!r}")
    return xyz, rpy


def arm_report(
    *,
    arm_name: str,
    arm_state: dict[str, Any],
    planner_joints_rad: dict[str, float],
    planner: Any | None,
    frame_ids: set[str],
    sdk_joint_unit: str,
    tolerance_deg: float,
) -> dict[str, Any]:
    """比较一条手臂的七个实时关节角，并保留可选末端 TF 诊断。"""
    joint_names = LEFT_JOINTS if arm_name == "left" else RIGHT_JOINTS
    sdk_values = [float(value) for value in arm_state.get("position") or []]
    if len(sdk_values) != 7:
        raise RuntimeError(f"SDK {arm_name}.position 应为 7 关节，实际为 {sdk_values!r}")
    sdk_deg = (
        sdk_values
        if sdk_joint_unit == "deg"
        else [math.degrees(value) for value in sdk_values]
    )
    missing = [name for name in joint_names if name not in planner_joints_rad]
    if missing:
        raise RuntimeError(f"规划器 /joint_states 缺少 {arm_name} 关节: {missing}")
    planner_deg = [math.degrees(planner_joints_rad[name]) for name in joint_names]
    joint_delta_deg = [planner_deg[index] - sdk_deg[index] for index in range(7)]
    max_joint_error = max(abs(value) for value in joint_delta_deg)

    frames: dict[str, Any] = {}
    sdk_eef_pose = None
    if planner is not None:
        sdk_xyz, sdk_rpy = sdk_pose(arm_state, arm_name)
        sdk_eef_pose = {
            "xyz_m": sdk_xyz,
            "rpy_rad": sdk_rpy,
            "rpy_deg": [math.degrees(value) for value in sdk_rpy],
        }
        for frame in FRAME_CANDIDATES[arm_name]:
            if frame not in frame_ids:
                continue
            planner_pose = lookup_transform(planner, "body_base", frame)
            frames[frame] = {
                "planner_T_body_base_frame": planner_pose,
                "relative_T_sdk_eef_frame": relative_transform(
                    sdk_xyz,
                    sdk_rpy,
                    planner_pose["xyz_m"],
                    planner_pose["quaternion_xyzw"],
                ),
            }
    return {
        "sdk_joint_deg": sdk_deg,
        "planner_joint_deg": planner_deg,
        "planner_minus_sdk_joint_deg": joint_delta_deg,
        "max_abs_joint_error_deg": max_joint_error,
        "joint_state_consistent": max_joint_error <= tolerance_deg,
        "sdk_eef_pose": sdk_eef_pose,
        "frames": frames,
    }


def print_arm_report(arm_name: str, report: dict[str, Any], tolerance_deg: float) -> None:
    """在终端输出关节同步结论和仅供定位的末端 TF 差。"""
    chinese_name = "左臂" if arm_name == "left" else "右臂"
    joint_status = "一致" if report["joint_state_consistent"] else "不一致"

    print(f"{chinese_name}：")
    print(
        f"  关节角：{joint_status}，最大差 {report['max_abs_joint_error_deg']:.4f}° "
        f"（判定阈值 {tolerance_deg:.3f}°）"
    )
    if report["frames"]:
        end_frame = f"{arm_name}_arm_end_link"
        if end_frame not in report["frames"]:
            end_frame = next(iter(report["frames"]))
        relative = report["frames"][end_frame]["relative_T_sdk_eef_frame"]
        translation_mm = [
            value * 1000.0 for value in relative["translation_m_in_sdk_eef"]
        ]
        print(
            f"  比较末端：SDK EEF 与规划器 {end_frame}"
            "（仅诊断，不参与关节同步判定）"
        )
        print(
            "  末端位置差："
            f"[{translation_mm[0]:.3f}, {translation_mm[1]:.3f}, {translation_mm[2]:.3f}] mm，"
            f"总长度 {relative['translation_norm_mm']:.3f} mm"
        )
        print(f"  末端旋转差：{relative['rotation_angle_deg']:.4f}°")


def main() -> int:
    args = parse_args()
    from parm_sdk import Arm, Planner

    container = find_planner_container(args.planner_container)
    arm = Arm(host=args.arm_host)
    planner = Planner(host=args.planner_host) if args.include_tf_diagnostics else None
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "output" / "fk_consistency"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    try:
        frame_ids: set[str] = set()
        if planner is not None:
            frame_ids = set(planner.list_frame_ids())
            if "body_base" not in frame_ids:
                raise RuntimeError("规划器 TF 缺少 body_base，无法读取末端诊断信息")
        print("=== PARM 控制器与规划器关节同步检查（只读，不会运动）===")
        print(f"控制器：{args.arm_host}；规划器：{args.planner_host}；容器：{container}")
        print(f"采样次数：{args.samples}；关节判定阈值：{args.joint_tolerance_deg:.3f}°")
        for index in range(args.samples):
            state = arm.get_state()
            if not isinstance(state, dict):
                raise RuntimeError(f"ARM.get_state() 返回异常: {state!r}")
            state_arms = state.get("state") or {}
            planner_joints = planner_joint_state(container)
            result = {
                "timestamp_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "sdk_state_timestamp": state.get("timestamp"),
                "left": arm_report(
                    arm_name="left",
                    arm_state=state_arms.get("left") or {},
                    planner_joints_rad=planner_joints,
                    planner=planner,
                    frame_ids=frame_ids,
                    sdk_joint_unit=args.sdk_joint_unit,
                    tolerance_deg=args.joint_tolerance_deg,
                ),
                "right": arm_report(
                    arm_name="right",
                    arm_state=state_arms.get("right") or {},
                    planner_joints_rad=planner_joints,
                    planner=planner,
                    frame_ids=frame_ids,
                    sdk_joint_unit=args.sdk_joint_unit,
                    tolerance_deg=args.joint_tolerance_deg,
                ),
            }
            samples.append(result)
            print(f"\n第 {index + 1}/{args.samples} 次采样：{result['timestamp_local']}")
            print_arm_report("left", result["left"], args.joint_tolerance_deg)
            print_arm_report("right", result["right"], args.joint_tolerance_deg)
            if index + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        try:
            if planner is not None:
                planner.close()
        except Exception:
            pass
        try:
            arm.close()
        except Exception:
            pass

    output_path = output_dir / f"fk_consistency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(
        json.dumps(
            {
                "read_only": True,
                "joint_sync_method": (
                    "Arm.get_state().state.<arm>.position (SDK degree) versus "
                    "Planner container ROS2 /joint_states (radian converted to degree)"
                ),
                "arm_host": args.arm_host,
                "planner_host": args.planner_host,
                "planner_container": container,
                "sdk_joint_unit": args.sdk_joint_unit,
                "joint_tolerance_deg": args.joint_tolerance_deg,
                "include_tf_diagnostics": args.include_tf_diagnostics,
                "frame_candidates": FRAME_CANDIDATES,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n完整原始数据已保存：{output_path}")
    print("提示：关节同步只看每条手臂的“最大差”。末端差仅供定位 frame/TCP 定义问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
