#!/usr/bin/env python3
"""按目标末端位姿（基坐标系）规划并移动一条 PARM 手臂。

目标姿态以 ``X Y Z ROLL PITCH YAW`` 给出：位置单位为米，RPY 单位为
弧度，旋转顺序为 ZYX（yaw-pitch-roll）。脚本不提供关节空间移动入口，
避免将两种控制方式混用。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

LOOP_ROOT = Path(__file__).resolve().parents[3]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))

from tools.planner_task_retry import retry_task_creation
from move_by_joints import (
    ensure_ready,
    finite,
    import_sdk,
    load_pose_file,
    quat_to_rotvec,
    rotvec_to_quat,
    rotvec_to_rpy,
    rpy_to_rotvec,
)


def quat_multiply(first: list[float], second: list[float]) -> list[float]:
    """Compose xyzw quaternions: the returned transform is first ∘ second."""
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def rotate_vector(quat: list[float], vector: list[float]) -> list[float]:
    """Rotate a 3-vector by an xyzw quaternion."""
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
    """Compose [xyz, rotvec] transforms: returned = first ∘ second."""
    first_quat, second_quat = rotvec_to_quat(first[3:]), rotvec_to_quat(second[3:])
    offset = rotate_vector(first_quat, second[:3])
    return [
        first[0] + offset[0], first[1] + offset[1], first[2] + offset[2],
    ] + quat_to_rotvec(quat_multiply(first_quat, second_quat))


def inverse_pose(pose: list[float]) -> list[float]:
    """Invert a [xyz, rotvec] transform."""
    quat = rotvec_to_quat(pose[3:])
    inverse_quat = [-quat[0], -quat[1], -quat[2], quat[3]]
    inverse_xyz = rotate_vector(inverse_quat, [-pose[0], -pose[1], -pose[2]])
    return inverse_xyz + quat_to_rotvec(inverse_quat)


def pose_from_lookup_transform(data: object) -> list[float]:
    """Convert Planner.lookup_transform() data to [xyz, rotvec]."""
    if not isinstance(data, dict):
        raise RuntimeError("规划器返回的 TF 数据无效。")
    translation, rotation = data.get("translation"), data.get("rotation")
    if not isinstance(translation, dict) or not isinstance(rotation, dict):
        raise RuntimeError("规划器返回的 TF 缺少 translation 或 rotation。")
    xyz = finite([translation.get("x"), translation.get("y"), translation.get("z")], 3, "planner TF translation")
    quat = finite([rotation.get("x"), rotation.get("y"), rotation.get("z"), rotation.get("w")], 4, "planner TF rotation")
    return xyz + quat_to_rotvec(quat)


def planner_relative_pose(
    planner: object, frame_id: str, parent_frame: str, child_frame: str,
) -> list[float]:
    """Read ``T_parent_child`` from the planner's live TF tree."""
    body_parent = pose_from_lookup_transform(
        planner.lookup_transform(frame_id, parent_frame).get("data")
    )
    body_child = pose_from_lookup_transform(
        planner.lookup_transform(frame_id, child_frame).get("data")
    )
    return compose_pose(inverse_pose(body_parent), body_child)


def pose_file_reference_frame(path: Path) -> str | None:
    """Return the explicit eef_pose reference frame recorded by teach v2."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    frame = payload.get("eef_pose_reference_frame")
    return frame if isinstance(frame, str) and frame else None


def load_workpiece_base_xy_yaw(path: Path, index: int) -> list[float]:
    """Load one workpiece's robot-base centre and tail-to-head yaw.

    ``mask_workpiece_pose.py`` writes one entry per connected component in
    descending mask-area order.  The returned target deliberately has zero
    z/roll/pitch because this source is only valid with --planar-xy-yaw;
    The planned target then replaces those values with the live EEF pose.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"工件位姿 JSON 无法解析: {path}: {exc}") from exc
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list):
        raise ValueError(f"工件位姿 JSON 缺少 workpieces 数组: {path}")
    if index < 0 or index >= len(workpieces):
        raise ValueError(f"--workpiece-index={index} 超出范围；文件中共有 {len(workpieces)} 个工件")
    entry = workpieces[index]
    if not isinstance(entry, dict) or not isinstance(entry.get("robot_base_xy_yaw"), dict):
        raise ValueError(
            f"workpieces[{index}] 缺少 robot_base_xy_yaw；请使用新版 mask_workpiece_pose.py 重新生成 {path}"
        )
    pose = entry["robot_base_xy_yaw"]
    x, y, yaw = finite([pose.get("x_m"), pose.get("y_m"), pose.get("yaw_rad")], 3, "robot_base_xy_yaw")
    return [x, y, 0.0] + rpy_to_rotvec([0.0, 0.0, yaw])


def main() -> int:
    parser = argparse.ArgumentParser(description="PARM 单臂基于末端位姿的绝对移动")
    parser.add_argument("--arm-id", type=int, choices=(1, 2), required=True, help="1=左臂，2=右臂")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument(
        "--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "127.0.0.1"),
        help="parm-planner 服务地址（默认：PARM_PLANNER_HOST 或 127.0.0.1）",
    )
    parser.add_argument(
        "--frame-id", default=os.environ.get("PARM_PLANNER_FRAME", "body_base"),
        help="规划目标的参考坐标系（本机规划模型默认：body_base）",
    )
    parser.add_argument(
        "--ik-frame", default=None,
        help="末端 IK 坐标系（默认：左臂 left_hand_transfer_base，右臂 right_hand_transfer_base）",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--target", nargs=6, type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="基坐标系下 xyz（m）+ RPY（rad，ZYX 组合）",
    )
    source.add_argument(
        "--pose-file", type=Path,
        help="保存的位姿 JSON，例如 saved_poses/arm_pose_*.json；读取其中所选手臂的 eef_pose",
    )
    source.add_argument(
        "--workpiece-pose-file", type=Path,
        help="mask_workpiece_pose.py 输出的 JSON；读取 workpieces[].robot_base_xy_yaw 作为目标",
    )
    parser.add_argument(
        "--workpiece-index",
        type=int,
        default=0,
        help="--workpiece-pose-file 中的工件索引（按 mask 面积从大到小，默认：0）",
    )
    parser.add_argument("--max-vel", type=float, default=0.1, help="规划最大速度比例 (0, 1]")
    parser.add_argument("--max-acc", type=float, default=0.1, help="规划最大加速度比例 (0, 1]")
    parser.add_argument(
        "--planner", choices=("lin", "ptp"), default="lin",
        help="Planner 运动类型：lin=笛卡尔直线，ptp=关节空间点到点（默认：lin）",
    )
    parser.add_argument(
        "--planar-xy-yaw", action="store_true",
        help="将规划目标的 z、roll、pitch 锁定为当前值；只采用 x/y 和 yaw",
    )
    parser.add_argument(
        "--start-sync-retries", type=int, default=3,
        help="轨迹首点与执行器当前关节状态轻微不一致时，重新规划的次数（默认：3）",
    )
    parser.add_argument("--execute", action="store_true", help="实际下发运动；省略时仅预览")
    args = parser.parse_args()

    if not 0.0 < args.max_vel <= 1.0 or not 0.0 < args.max_acc <= 1.0:
        parser.error("--max-vel 和 --max-acc 必须在 (0, 1] 内")
    if args.start_sync_retries < 0:
        parser.error("--start-sync-retries 不能小于 0")
    if args.workpiece_pose_file is not None and not args.planar_xy_yaw:
        parser.error("--workpiece-pose-file 必须与 --planar-xy-yaw 一起使用，因为工件 JSON 只提供 x/y/yaw")

    group = "left_arm" if args.arm_id == 1 else "right_arm"
    default_ik_frame = "left_hand_transfer_base" if args.arm_id == 1 else "right_hand_transfer_base"
    ik_frame = args.ik_frame or default_ik_frame
    tcp_name = ik_frame
    tcp_offset = [0.0] * 6
    target_position_correction_frame = args.frame_id
    target_position_correction = [0.0] * 3
    if args.pose_file is not None:
        target = load_pose_file(args.pose_file, args.arm_id, "pose")
        target_reference_frame = pose_file_reference_frame(args.pose_file) or args.frame_id
        source_description = f"保存位姿文件 {args.pose_file}"
    elif args.workpiece_pose_file is not None:
        target = load_workpiece_base_xy_yaw(args.workpiece_pose_file, args.workpiece_index)
        # mask_workpiece_pose.py uses T_base_head_camera; in this system the
        # SDK calibration base is the Planner body_base.
        target_reference_frame = args.frame_id
        source_description = (
            f"工件位姿文件 {args.workpiece_pose_file} 的 workpieces[{args.workpiece_index}].robot_base_xy_yaw"
        )
    else:
        raw_target = finite(args.target, 6, "target")
        target = raw_target[:3] + rpy_to_rotvec(raw_target[3:])
        target_reference_frame = args.frame_id
        source_description = "命令行目标"
    if any(target_position_correction):
        if target_reference_frame != target_position_correction_frame:
            parser.error(
                f"虚拟 TCP 的目标位置校正定义在 {target_position_correction_frame!r}，"
                f"但目标参考系为 {target_reference_frame!r}。"
            )
        target[:3] = [
            target[index] + target_position_correction[index]
            for index in range(3)
        ]
        print(
            f"已应用 {target_position_correction_frame} 下的目标位置校正 (m): "
            f"{target_position_correction}"
        )
    print(f"目标来源: {source_description}")
    print(f"目标 arm_id={args.arm_id}，TCP={tcp_name} [xyz(m), rotvec(rad)]: {target}")
    if not args.execute:
        print("这是预览，未发送运动命令。确认路径无碰撞风险后添加 --execute。")
        return 0

    arm = None
    planner = None
    try:
        sdk = import_sdk()
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=args.planner_host)
        if target_reference_frame != args.frame_id:
            raise RuntimeError(
                f"目标参考系 {target_reference_frame!r} 与规划参考系 {args.frame_id!r} 不一致。"
            )
        if args.planar_xy_yaw:
            # Use Planner's state and TF here, rather than the driver's
            # anonymous EEF pose.  The constrained endpoint and the planned
            # trajectory are then expressed in one consistent frame/model.
            ik_start = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
            start = compose_pose(ik_start, tcp_offset)
            start_rpy = rotvec_to_rpy(start[3:])
            target_rpy = rotvec_to_rpy(target[3:])
            target = [target[0], target[1], start[2]] + rpy_to_rotvec(
                [start_rpy[0], start_rpy[1], target_rpy[2]]
            )
            print(f"Planner 当前 {args.frame_id}→{tcp_name} [xyz(m), rotvec(rad)]: {start}")
            print(f"平面约束后的目标 [xyz(m), rotvec(rad)]: {target}")
        planner_target = compose_pose(target, inverse_pose(tcp_offset))
        print(f"目标已在 {args.frame_id} 中；已从 TCP={tcp_name} 换算到 IK frame。")
        print(f"规划器目标 {args.frame_id}→{ik_frame} [xyz(m), rotvec(rad)]: {planner_target}")
        for attempt in range(args.start_sync_retries + 1):
            # The planner and arm server are independent services.  Re-plan
            # immediately before execution so the first waypoint follows the
            # newest planner-side joint state.
            result = retry_task_creation(
                lambda: planner.plan_cart_trajectory(
                    group=group,
                    planner=args.planner,
                    frame_id=args.frame_id,
                    ik_frame=ik_frame,
                    position=planner_target[:3],
                    orientation=rotvec_to_quat(planner_target[3:]),
                    max_vel=args.max_vel,
                    max_acc=args.max_acc,
                ),
                label=f"{arm_label} {args.planner.upper()} 笛卡尔规划",
            )
            data = result.get("data") or {}
            trajectory = data.get("trajectory")
            planned_arm_id = data.get("arm_id")
            sample_frequency = data.get("sample_frequency", 10.0)
            if planned_arm_id != args.arm_id:
                raise RuntimeError(f"规划结果 arm_id={planned_arm_id!r}，与请求的 arm_id={args.arm_id} 不一致。")
            if not isinstance(trajectory, list) or not trajectory:
                raise RuntimeError("规划服务未返回有效 trajectory。")
            if not isinstance(sample_frequency, (int, float)) or sample_frequency <= 0:
                raise RuntimeError(f"规划服务返回的 sample_frequency 无效：{sample_frequency!r}")
            print(f"规划完成：{len(trajectory)} 个关节路点，采样频率 {sample_frequency} Hz。")
            try:
                arm.follow_trajectory(
                    arm_id=planned_arm_id,
                    trajectory=trajectory,
                    sample_frequency=float(sample_frequency),
                )
                break
            except Exception as exc:
                is_start_mismatch = "DiscreteTask::validate: first point mismatch" in str(exc)
                if not is_start_mismatch or attempt >= args.start_sync_retries:
                    raise
                print(
                    f"轨迹首点与执行器实时关节状态轻微不一致；"
                    f"正在重新规划并重试（{attempt + 1}/{args.start_sync_retries}）..."
                )
                time.sleep(0.2)
        print("末端位姿移动已完成。")
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if planner is not None:
            planner.close()
        if arm is not None:
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
