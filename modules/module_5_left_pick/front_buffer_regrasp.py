#!/usr/bin/env python3
"""用 Buffer Check 的正面 ICP 位姿重新抓取桌面工件。

只接受已经通过 Buffer Check 质量门的唯一 FRONT 结果。以 ICP 原点作为
抓取参考；预抓高度和 LIN 下移距离独立配置，默认闭爪点为 ICP 上方 3 cm。
对 ICP 工件轴尝试两个夹爪姿态候选：``(+Z,+Y,+X)=(+X,+Y,-Z)`` 及
``(-X,-Y,-Z)``；均可达时选择左臂关节位移较小者。闭爪后 LIN 上提 3 cm。
成功后输出
Module 3 可直接消费的 ``T_tool_object`` / ``T_body_base_object``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tool")]

from loop_config import LoopConfig
from loop_types import ObjectPose
from planner_runtime import PlannerRuntime
from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from modules.stage_result import write_stage_result


DEFAULT_PREGRASP_HEIGHT_M = 0.05
DEFAULT_DESCENT_DISTANCE_M = 0.02
POST_GRASP_LIFT_M = 0.03
# 新左夹爪在桌面反抓到工件后应保留至少 35% 的开度；过小开度表示
# 夹爪基本闭合，不能把它当作可靠持件。
HELD_POSITION_MIN_PERCENT = 20.0
NO_GRASP_EXIT_CODE = 20
MAX_ICP_ROTATION_PROJECTION_ERROR = 0.01
BODY_HEAD = (
    ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/config/"
    "body_head_calibration.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-check-result", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument(
        "--pregrasp-height-m", type=float, default=DEFAULT_PREGRASP_HEIGHT_M,
        help="预抓位相对 ICP 原点的 body_base +Z 高度，默认 0.05m",
    )
    parser.add_argument(
        "--descent-distance-m", type=float, default=DEFAULT_DESCENT_DISTANCE_M,
        help="从预抓位沿 body_base -Z 的 LIN 下移距离，默认 0.02m",
    )
    return parser.parse_args()


def _pose(matrix: np.ndarray) -> ObjectPose:
    return ObjectPose(
        frame_id="body_base",
        position=matrix[:3, 3].tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def _matrix_from_tf(response: object, label: str) -> np.ndarray:
    payload = (response or {}).get("data") or response or {}
    translation = payload.get("translation") or {}
    rotation = payload.get("rotation") or {}
    try:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            [float(rotation[axis]) for axis in "xyzw"]
        ).as_matrix()
        matrix[:3, 3] = [float(translation[axis]) for axis in "xyz"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Planner 未返回有效 {label} TF: {response!r}") from exc
    return matrix


def _nearest_proper_rotation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a finite ICP 3x3 block to SO(3), preserving its nearest axes."""
    raw = np.asarray(matrix, dtype=float)
    u, _, vt = np.linalg.svd(raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation, float(np.linalg.norm(raw - rotation))


def _front_icp_pose(result_path: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("decision") != "FRONT" or payload.get("continue_allowed") is not True:
        raise RuntimeError(f"Buffer Check 未放行唯一 FRONT 工件: {payload}")
    objects = payload.get("objects") or []
    if len(objects) != 1 or objects[0].get("orientation") != "FRONT":
        raise RuntimeError(f"Buffer Check 必须返回唯一 FRONT 工件: {payload}")
    diagnostic = objects[0].get("diagnostics") or {}
    try:
        T_camera_object = np.asarray(
            diagnostic["positive"]["T_camera_object"], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Buffer Check 缺少正面 ICP 位姿；请确认 sam2_icp_grasp.py 已输出 "
            "diagnostics.positive.T_camera_object"
        ) from exc
    if T_camera_object.shape != (4, 4) or not np.all(np.isfinite(T_camera_object)):
        raise RuntimeError("Buffer Check 的 T_camera_object 不是有效 4x4 有限矩阵")
    rotation, correction_error = _nearest_proper_rotation(T_camera_object[:3, :3])
    if correction_error > MAX_ICP_ROTATION_PROJECTION_ERROR:
        raise RuntimeError(
            "ICP 工件旋转矩阵偏离正交矩阵过大，拒绝据此抓取："
            f"projection_error={correction_error:.6f}"
        )
    T_camera_object[:3, :3] = rotation
    return T_camera_object, correction_error


def _eef_rotations_from_object_axes(
    T_camera_object: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Return the two requested right-handed EEF frames from the ICP frame."""
    R_object = T_camera_object[:3, :3]
    if not np.allclose(R_object.T @ R_object, np.eye(3), atol=1e-4):
        raise RuntimeError("ICP 工件旋转矩阵非正交，拒绝据此设置夹爪姿态")
    candidates = [
        (
            "eef(+X,+Y,+Z)=object(-Z,+Y,+X)",
            # [EEF +X, EEF +Y, EEF +Z] = [Object -Z, Object +Y, Object +X]
            np.column_stack((-R_object[:, 2], R_object[:, 1], R_object[:, 0])),
        ),
        (
            "eef(+X,+Y,+Z)=object(-Z,-Y,-X)",
            # [EEF +X, EEF +Y, EEF +Z] = [Object -Z, Object -Y, Object -X]
            np.column_stack((-R_object[:, 2], -R_object[:, 1], -R_object[:, 0])),
        ),
    ]
    if not all(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4) for _, rotation in candidates):
        raise RuntimeError("由 ICP 轴映射生成的夹爪旋转矩阵不是右手坐标系")
    return candidates


def _current_left_joints_deg(arm_host: str) -> np.ndarray:
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
    values = ((state.get("state") or {}).get("left") or {}).get("position") or []
    joints = np.asarray(values, dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"未获得有效左臂当前关节角: {values}")
    return joints


def _joint_displacement_norm_deg(start: np.ndarray, end: list[float]) -> float:
    delta = (np.asarray(end, dtype=float) - start + 180.0) % 360.0 - 180.0
    return float(np.linalg.norm(delta))


def _vertical_regrasp_targets(
    T_body_icp: np.ndarray, *, pregrasp_height_m: float, descent_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pregrasp, close-grasp, and post-close lift targets on body +Z."""
    pregrasp = T_body_icp.copy()
    pregrasp[2, 3] += pregrasp_height_m
    grasp = pregrasp.copy()
    grasp[2, 3] -= descent_distance_m
    lift = grasp.copy()
    lift[2, 3] += POST_GRASP_LIFT_M
    return pregrasp, grasp, lift


def _write_result(path: Path, payload: dict) -> None:
    """Publish the Module 2/3 handoff atomically for the loop consumer."""
    write_stage_result(path, payload)


def _wait_for_hold(gripper_host: str, timeout_s: float = 3.0) -> tuple[bool, float | None, str]:
    """Wait for the subscribed driver state after the blocking post-grasp lift.

    The new gripper's synchronous command acknowledgement can precede its next
    status PUB frame.  This is one terminal-state wait, not repeated grasp
    verification windows.
    """
    from parm_sdk import Gripper

    with Gripper(host=gripper_host, status_port=7702) as gripper:
        deadline = time.monotonic() + timeout_s
        state: dict = {}
        while True:
            state = gripper.get_state().get(1) or {}
            motion_state = str(state.get("motion_state", "")).strip().lower()
            try:
                position = float(state.get("position"))
            except (TypeError, ValueError):
                position = None
            if motion_state == "gripping" and position is not None and position > HELD_POSITION_MIN_PERCENT:
                return True, position, motion_state
            if motion_state in {"error", "dropped"} or time.monotonic() >= deadline:
                return False, position, motion_state
            time.sleep(0.05)


def _open_left_gripper_or_raise(gripper_host: str, *, context: str) -> None:
    """Open the gripper synchronously and verify an actual open terminal state."""
    from parm_sdk import Gripper

    with Gripper(host=gripper_host, status_port=7702) as gripper:
        response = gripper.move(
            1, 100.0, force=100.0, speed=80.0, mode="sync", preempt=False, timeout=60.0,
        )
    payload = (response.get("data") or response) if isinstance(response, dict) else {}
    device = (payload.get("devices") or {}).get("left_gripper") or {}
    response_arrived = device.get("device_state") == "arrived"
    if not isinstance(response, dict) or response.get("status") != "success":
        raise RuntimeError(
            f"{context}：左夹爪张开命令失败: response={response}"
        )
    # The synchronous command response can arrive one PUB frame before the
    # position update.  Require an actual >=95% open state before motion is
    # allowed to continue, rather than trusting the command acknowledgement.
    deadline = time.monotonic() + 8.0
    state: dict = {}
    while time.monotonic() < deadline:
        with Gripper(host=gripper_host, status_port=7702) as gripper:
            state = gripper.get_state().get(1) or {}
        motion_state = str(state.get("motion_state", "")).lower()
        try:
            position = float(state.get("position"))
        except (TypeError, ValueError):
            position = float("nan")
        if motion_state in {"error", "dropped"}:
            break
        if np.isfinite(position) and position >= 95.0 and (
            response_arrived or motion_state == "arrived"
        ):
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"{context}：左夹爪未确认张开到位: response={response}, state={state}"
    )


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("正面恢复抓取必须显式传入 --execute")
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed 必须在 (0, 1] 内")
    if not 0.0 < args.pregrasp_height_m <= 0.20:
        raise ValueError("--pregrasp-height-m 必须在 (0, 0.20] m 内")
    if not 0.0 < args.descent_distance_m <= args.pregrasp_height_m:
        raise ValueError(
            "--descent-distance-m 必须在 (0, --pregrasp-height-m] m 内"
        )
    T_camera_object, icp_rotation_projection_error = _front_icp_pose(
        args.buffer_check_result
    )
    calibration = json.loads(BODY_HEAD.read_text(encoding="utf-8"))
    T_body_camera = np.asarray(calibration["T"], dtype=float)
    if T_body_camera.shape != (4, 4):
        raise RuntimeError("body-head 标定不是有效 4x4 矩阵")
    T_body_object_at_table = T_body_camera @ T_camera_object

    config = LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base", left_group="left_arm", right_group="right_arm",
        left_tool_frame="left_eef", right_tool_frame="right_hand_transfer_end",
        max_velocity=args.speed, max_acceleration=args.speed, idle_poll_seconds=0.2,
        live=True, poses={},
    )
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={"body_base", "left_eef", "left_hand_transfer_end"})
        # All three trajectories must be feasible before closing the gripper.
        # This makes a reachability failure a normal re-observe/retry outcome;
        # after closure, a failed lift must remain a hard safety fault instead.
        def preclose_plan(label: str, target: np.ndarray, planner_name: str,
                          start_joints_deg: list[float] | None = None) -> dict | None:
            try:
                return planner.plan_cartesian_pose(
                    "left", _pose(target), planner=planner_name,
                    start_joints_deg=start_joints_deg,
                )
            except Exception as exc:
                print(
                    f"NO_GRASP: {label} 规划不可达；将重新 ICP 后重试：{exc}",
                    flush=True,
                )
                return None

        current_joints = _current_left_joints_deg(args.arm_host)
        feasible_candidates = []
        for label, R_camera_eef in _eef_rotations_from_object_axes(T_camera_object):
            T_camera_eef = np.eye(4)
            T_camera_eef[:3, :3] = R_camera_eef
            T_camera_eef[:3, 3] = T_camera_object[:3, 3]
            candidate_icp = T_body_camera @ T_camera_eef
            candidate_pregrasp, candidate_grasp, candidate_lift = (
                _vertical_regrasp_targets(
                    candidate_icp,
                    pregrasp_height_m=args.pregrasp_height_m,
                    descent_distance_m=args.descent_distance_m,
                )
            )
            pregrasp_plan = preclose_plan(
                f"{label} / ICP 原点 Z+{args.pregrasp_height_m * 100:.1f}cm 预抓位",
                candidate_pregrasp, "rrtstar",
            )
            trajectory = (pregrasp_plan or {}).get("data", {}).get("trajectory") or []
            if not trajectory or len(trajectory[-1]) != 7:
                print(f"NO_GRASP: {label} 预抓规划无有效末端关节路点。", flush=True)
                continue
            descent_plan = preclose_plan(
                f"{label} / LIN 下移{args.descent_distance_m * 100:.1f}cm",
                candidate_grasp, "lin", trajectory[-1]
            )
            descent_trajectory = (descent_plan or {}).get("data", {}).get("trajectory") or []
            if not descent_trajectory or len(descent_trajectory[-1]) != 7:
                print(f"NO_GRASP: {label} 下探规划无有效末端关节路点。", flush=True)
                continue
            lift_plan = preclose_plan(
                f"{label} / Z+3cm LIN 上提", candidate_lift, "lin", descent_trajectory[-1]
            )
            if lift_plan is None:
                continue
            displacement = _joint_displacement_norm_deg(current_joints, trajectory[-1])
            feasible_candidates.append((
                displacement, label, candidate_icp, candidate_grasp, candidate_pregrasp,
                candidate_lift, pregrasp_plan, descent_plan, lift_plan,
            ))
        if not feasible_candidates:
            return NO_GRASP_EXIT_CODE
        (
            displacement, selected_label, T_body_icp, T_body_eef, T_body_pregrasp, T_body_lift,
            pregrasp_plan, descent_plan, lift_plan,
        ) = min(feasible_candidates, key=lambda item: item[0])
        print(
            "[front-regrasp] 姿态候选规划完成："
            f"可达={len(feasible_candidates)}/2，选择={selected_label}，"
            f"预抓关节位移范数={displacement:.3f}deg",
            flush=True,
        )

        from parm_sdk import Gripper
        with Gripper(host=args.gripper_host, status_port=7702) as gripper:
            state = gripper.get_state().get(1)
        if not is_gripper_usable(state):
            raise RuntimeError("左夹爪不可用，拒绝开始正面恢复抓取")
        _open_left_gripper_or_raise(args.gripper_host, context="正面恢复抓取开始前")
        print(
            "[front-regrasp] ICP 参考预抓/上方闭爪："
            f"pregrasp_xyz={T_body_pregrasp[:3, 3].round(6).tolist()}, "
            f"grasp_xyz={T_body_eef[:3, 3].round(6).tolist()}, "
            f"pregrasp_height_m={args.pregrasp_height_m:.3f}, "
            f"descent_distance_m={args.descent_distance_m:.3f}", flush=True,
        )
        planner.execute_planned_result(pregrasp_plan)
        planner.execute_planned_result(descent_plan)
        with Gripper(host=args.gripper_host, status_port=7702) as gripper:
            response = gripper.move(1, 0.0, force=100.0, speed=80.0, mode="sync", preempt=False, timeout=60.0)
        if response.get("status") != "success":
            raise RuntimeError(f"左夹爪闭合命令失败: {response}")
        planner.execute_planned_result(lift_plan)
        held, position, motion_state = _wait_for_hold(args.gripper_host)
        if not held:
            _open_left_gripper_or_raise(args.gripper_host, context="正面恢复抓取未持件后")
            print(
                "NO_GRASP: 正面恢复抓取上提3cm后未确认持件；已打开左夹爪，"
                f"motion_state={motion_state!r}, position={position}",
                flush=True,
            )
            return NO_GRASP_EXIT_CODE

        T_body_eef_after_lift = _matrix_from_tf(
            planner.lookup_transform("body_base", "left_eef"), "left_eef"
        )
        T_body_transfer_after_lift = _matrix_from_tf(
            planner.lookup_transform("body_base", "left_hand_transfer_end"),
            "left_hand_transfer_end",
        )
        T_eef_object = np.linalg.inv(T_body_eef) @ T_body_object_at_table
        T_body_object_held = T_body_eef_after_lift @ T_eef_object
        T_transfer_object = np.linalg.inv(T_body_transfer_after_lift) @ T_body_object_held
        _write_result(args.result_file, {
            # Module 3 consumes the stable ``left_inspect`` result contract.
            # This recovery has produced the same two required transforms after
            # a fresh ICP-guided grasp; retain its real provenance separately.
            "stage": "left_inspect",
            "source_stage": "front_buffer_regrasp",
            "outcome": "success",
            "classification": "FRONT",
            "T_tool_object": T_transfer_object.tolist(),
            "T_body_base_object": T_body_object_held.tolist(),
            # Keep the standard Module 2 -> Module 3 contract.  These are
            # captured after the successful regrasp/lift, when the object is
            # rigidly held by the actual left EEF.
            "T_body_left_eef_at_inspect": T_body_eef_after_lift.tolist(),
            "T_left_eef_object": T_eef_object.tolist(),
            "icp_object_pose_body_base": T_body_object_at_table.tolist(),
            "icp_rotation_projection_error": icp_rotation_projection_error,
            "pregrasp_height_m": args.pregrasp_height_m,
            "descent_distance_m": args.descent_distance_m,
            "grasp_height_above_icp_m": (
                args.pregrasp_height_m - args.descent_distance_m
            ),
            "icp_target_body_base": T_body_icp.tolist(),
            "grasp_target_body_base": T_body_eef.tolist(),
            "pregrasp_target_body_base": T_body_pregrasp.tolist(),
            "post_grasp_lift_m": POST_GRASP_LIFT_M,
            "gripper": {"motion_state": motion_state, "position_percent": position},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[front-regrasp] 成功；Module 3 输入已写入 {args.result_file}", flush=True)
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
