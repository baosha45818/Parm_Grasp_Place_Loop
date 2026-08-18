"""Module 3 的命令行、handoff、位姿和设备公共函数。

主流程保留在 ``standalone_left_place.py``，本文件不编排任何业务阶段，便于独立
查看和测试输入校验、关节到位、夹爪以及规划器配置。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from loop_config import LoopConfig
from loop_types import ObjectPose
from modules.module_3_left_place.placement_parameters import (
    GRASPUP_JOINTS_DEG,
    GRASPUP_LIFT_M,
)
from modules.stage_result import write_stage_result


ROOT = Path(__file__).resolve().parents[2]
LEFT_GRIPPER_HAND = 1


def parse_args() -> argparse.Namespace:
    """解析独立运行参数，并落实真实运动的双参数授权门禁。"""
    parser = argparse.ArgumentParser(
        description="Module 3：把左臂在 graspuptop 已持有的工件放至固定位置。"
    )
    parser.add_argument("--workpiece", choices=("1", "2"), default="1")
    handoff_group = parser.add_mutually_exclusive_group(required=True)
    handoff_group.add_argument("--handoff-file", type=Path, help="上游模块写出的 handoff JSON")
    handoff_group.add_argument(
        "--weight-handoff-file", dest="handoff_file", type=Path,
        help="兼容旧参数；等价于 --handoff-file",
    )
    parser.add_argument("--placement-kind", choices=("buffer", "middle"), default="buffer")
    parser.add_argument(
        "--result-file", type=Path,
        default=ROOT / "output/module_3_left_place" / f"buffer_place_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"))
    parser.add_argument("--gripper-hand", type=int, default=LEFT_GRIPPER_HAND, choices=(LEFT_GRIPPER_HAND,))
    parser.add_argument("--execute-url", default="http://127.0.0.1:5667/middle/waypoint/execute")
    left_scale = float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0"))
    parser.add_argument("--preset-speed", type=float, default=0.2 * left_scale)
    parser.add_argument("--motion-speed", type=float, default=0.1 * left_scale)
    parser.add_argument("--post-graspup-planner", choices=("rrtconnect", "rrtstar"), default="rrtstar")
    parser.add_argument("--offline-start-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--placement-planning-inflight", "--placement-planning-workers", dest="placement_planning_inflight", type=int, default=8)
    parser.add_argument("--graspup-lift-m", type=float, default=GRASPUP_LIFT_M)
    parser.add_argument("--pre-place-middle-joints-deg", nargs=7, type=float)
    parser.add_argument("--placement-ready-file", type=Path)
    parser.add_argument("--placement-permit-file", type=Path)
    parser.add_argument("--placement-permit-poll-s", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-real-place", action="store_true")
    args = parser.parse_args()
    if args.execute != args.confirm_real_place:
        parser.error("physical motion requires both --execute and --confirm-real-place")
    args.skip_motion = not args.execute
    return args


def matrix_pose(matrix: np.ndarray) -> ObjectPose:
    """把 4×4 齐次矩阵转换为 Planner 使用的 ``ObjectPose``。"""
    return ObjectPose(
        frame_id="body_base",
        position=np.asarray(matrix[:3, 3], dtype=float).tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def planner_matrix(response: dict) -> np.ndarray:
    """把 Planner TF 响应转换为 4×4 齐次矩阵并校验字段完整性。"""
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [translation.get(axis) for axis in "xyz"]
    xyzw = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in xyz + xyzw):
        raise RuntimeError(f"规划器 TF 格式无效: {response}")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    result[:3, 3] = xyz
    return result


def read_left_joints_deg(arm_host: str) -> np.ndarray:
    """从 Arm SDK 读取左臂七关节角，单位为度。"""
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
    joints = np.asarray(((state.get("state") or {}).get("left") or {}).get("position") or [], dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"无法读取有效的左臂七关节状态: {state}")
    return joints


def wait_for_left_joint_target(
    arm_host: str, target_joints_deg: list[float], *, label: str,
    tolerance_deg: float = 0.5, timeout_s: float = 90.0,
) -> None:
    """轮询实际关节角，直到左臂到达指定目标或超时。"""
    from parm_sdk import Arm

    target = np.asarray(target_joints_deg, dtype=float)
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = np.asarray(((state.get("state") or {}).get("left") or {}).get("position") or [], dtype=float)
            if actual.shape == (7,):
                delta = (actual - target + 180.0) % 360.0 - 180.0
                last_error_deg = float(np.max(np.abs(delta)))
                if last_error_deg <= tolerance_deg:
                    print(f"[{label}] 左臂实际到位，最大关节误差={last_error_deg:.3f}deg", flush=True)
                    return
            time.sleep(0.05)
    raise RuntimeError(f"左臂未在 {timeout_s:.1f}s 内到达{label}；最后最大关节误差={last_error_deg:.3f}deg")


def wait_for_left_graspup(arm_host: str) -> None:
    """等待左臂到达固定 ``graspup`` 安全关节位。"""
    wait_for_left_joint_target(arm_host, GRASPUP_JOINTS_DEG, label="graspup")


def validate_z_candidates(name: str, values: tuple[float, ...], *, allow_empty: bool = False) -> list[float]:
    """校验严格递增的 Z 候选；松爪候选可显式为空。"""
    candidates = np.asarray(values, dtype=float)
    if allow_empty and candidates.size == 0:
        return []
    if candidates.size == 0 or not np.all(np.isfinite(candidates)) or np.any(np.diff(candidates) <= 0.0):
        raise ValueError(f"{name} 必须是严格递增的有限数值")
    return candidates.tolist()


def open_left_gripper(args: argparse.Namespace) -> float | None:
    """确认左夹爪可用后完全打开，并返回驱动报告的最终开度。"""
    from parm_sdk import Gripper
    from modules.module_0_prepare.gripper_state_compat import is_gripper_usable

    with Gripper(host=args.gripper_host, status_port=7702) as gripper:
        hand_state = gripper.get_state().get(LEFT_GRIPPER_HAND)
        if hand_state is None:
            raise RuntimeError("左夹爪 hand=1 没有同步状态或未连接")
        if not is_gripper_usable(hand_state):
            raise RuntimeError("左夹爪尚未初始化，禁止发送打开指令")
        response = gripper.move(LEFT_GRIPPER_HAND, 100.0, force=100.0, speed=80.0, mode="sync", preempt=False, timeout=60.0)
    payload = response.get("data") or response
    interfaces = ((payload.get("devices") or {}).get("left_gripper") or {}).get("state_interfaces") or {}
    position = interfaces.get("gripper/position", payload.get("position"))
    return None if position is None else float(position)


def publish_ready(args: argparse.Namespace, payload: dict) -> None:
    """如配置了外部协调文件，则原子写出当前阶段状态。"""
    if args.placement_ready_file is not None:
        write_stage_result(args.placement_ready_file, payload)


def wait_permit(args: argparse.Namespace) -> None:
    """仅在显式配置时等待外部 Buffer 执行许可。"""
    if args.skip_motion or args.placement_permit_file is None:
        return
    print(f"[module3] 等待 Buffer 许可: {args.placement_permit_file}", flush=True)
    while not args.placement_permit_file.is_file():
        time.sleep(float(args.placement_permit_poll_s))
    content = args.placement_permit_file.read_text(encoding="utf-8").strip().lower()
    if content not in {"allow", "y", "yes", "1"}:
        raise RuntimeError(f"放置许可内容无效: {content!r}")


def validate_runtime_args(args: argparse.Namespace) -> None:
    """校验会影响运动安全、并发数和外部门禁的运行参数。"""
    if (args.placement_ready_file is None) != (args.placement_permit_file is None):
        raise ValueError("placement ready/permit 文件必须同时提供")
    if not np.isfinite(args.graspup_lift_m) or not np.isclose(args.graspup_lift_m, GRASPUP_LIFT_M, atol=1e-9):
        raise ValueError("--graspup-lift-m 当前必须为固定值 0.10m")
    if args.pre_place_middle_joints_deg is not None and not np.all(np.isfinite(args.pre_place_middle_joints_deg)):
        raise ValueError("--pre-place-middle-joints-deg 必须包含七个有限角度")
    for name, value in (("--preset-speed", args.preset_speed), ("--motion-speed", args.motion_speed)):
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} 必须是 (0, 1] 内的有限数值")
    if not np.isfinite(args.offline_start_tolerance_deg) or args.offline_start_tolerance_deg <= 0.0:
        raise ValueError("--offline-start-tolerance-deg 必须是大于0的有限数值")
    if not 1 <= args.placement_planning_inflight <= 16:
        raise ValueError("--placement-planning-inflight 必须在[1, 16]内")
    if not np.isfinite(args.placement_permit_poll_s) or args.placement_permit_poll_s <= 0.0:
        raise ValueError("--placement-permit-poll-s 必须是大于0的有限数值")


def load_handoff(path: Path, placement_kind: str) -> dict:
    """按放置类型校验上游 handoff，禁止将两类持件状态混用。"""
    if not path.is_file():
        raise FileNotFoundError(f"Module 3 上游 handoff 不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if placement_kind == "buffer":
        if payload.get("weight_check_enabled") is not True:
            raise RuntimeError("Module 3(buffer) 只接受启用重量检查的 Module 1 handoff")
        weight_check = payload.get("weight_check")
        if not isinstance(weight_check, dict):
            raise RuntimeError("Module 1 handoff 缺少 weight_check")
        if weight_check.get("classification") != "SINGLE":
            raise RuntimeError(f"Module 3(buffer) 只接受 Module 1 验证过的 SINGLE 工件，实际为 {weight_check.get('classification')!r}")
        if payload.get("handoff_pose") != "graspuptop":
            raise RuntimeError("Module 1 handoff 不在 graspuptop，拒绝 Buffer 放置")
        return payload
    if placement_kind == "middle":
        if payload.get("module") != "module_5_1_front_pick_to_graspuptop":
            raise RuntimeError("Module 3(middle) 只接受 Module 5.1 handoff")
        if payload.get("outcome") != "front_regrasp_handoff":
            raise RuntimeError("Module 5.1 handoff outcome 不正确，拒绝桌中放置")
        if payload.get("handoff_pose") != "graspuptop":
            raise RuntimeError("Module 5.1 handoff 不在 graspuptop，拒绝桌中放置")
        if payload.get("classification") != "FRONT":
            raise RuntimeError("Module 3(middle) 只接受 FRONT 工件")
        return payload
    raise AssertionError(f"未处理的 placement_kind: {placement_kind!r}")


def build_planner_config(args: argparse.Namespace) -> LoopConfig:
    """集中构造 Module 3 的 Planner/执行配置，避免主流程散落参数。"""
    return LoopConfig(
        planner_host=args.planner_host, execute_url=args.execute_url,
        base_frame="body_base", left_group="left_arm", right_group="right_arm",
        left_tool_frame="left_eef", right_tool_frame="right_hand_transfer_end",
        max_velocity=args.motion_speed, max_acceleration=args.motion_speed,
        idle_poll_seconds=0.2, live=True, poses={},
    )
