#!/usr/bin/env python3
"""Use a mask pose JSON to turn one workpiece toward base-XY quadrant I.

The script first moves the right fingertip TCP in XY to the reverse extension
of the intended push/grasp line while retaining its reference grasp pose.  It
then turns yaw at that on-line point, and finally advances along the line.  Q1
scans the rear-line endpoints for one trajectory that combines the first two
actions while holding the saved grasp-reference z/roll/pitch; the last segment
passes its inside tail grasp point and advances before closing.
For Q2--Q4 it pushes through JSON ``push_entry_point`` by the configured
distance.

Example (preview first):
    python3 push_workpiece_to_q1.py --arm-id 2 \
      --workpiece-pose-file /path/to/15_mask_pose.json --workpiece-index 0

    python3 push_workpiece_to_q1.py --arm-id 2 \
      --workpiece-pose-file /path/to/15_mask_pose.json --workpiece-index 0 \
      --planner lin --tcp right-fingertip-center --execute

    # Move the fingertip TCP to the workpiece centre and align its yaw with
    # the tail-to-head (red-arrow) direction, without operating the gripper.
    python3 push_workpiece_to_q1.py --arm-id 2 \
      --workpiece-pose-file /path/to/15_mask_pose.json --workpiece-index 0 \
      --planner lin --tcp right-fingertip-center --test --execute
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from move_by_eef_pose import (
    RIGHT_FINGERTIP_CENTER_CONFIG,
    compose_pose,
    inverse_pose,
    load_virtual_tcp_offset,
    pose_from_lookup_transform,
)
from move_by_joints import (
    ensure_ready,
    finite,
    import_sdk,
    rotvec_to_quat,
    rotvec_to_rpy,
    rpy_to_rotvec,
)
from right_trajectory_recorder import capture_tool_pose, record_plan


SCRIPT_DIR = Path(__file__).resolve().parent


# Temporary body_base compensation measured from the overlay.  Command the
# right-fingertip TCP 15 mm in +X and 20 mm in -Y so the physical closed
# two-fingertip centre reaches the requested workpiece point.
TEST_CLOSED_FINGERTIP_CENTER_COMPENSATION_XY_M = [0.010, -0.018]
# TEST_CLOSED_FINGERTIP_CENTER_COMPENSATION_XY_M = [0.00, -0.00]


# Q2--Q4 push only: the contact needs to be made by the left edge of the
# closed fingertip pair rather than its centre.  This is expressed in
# body_base XY and intentionally does not affect --test or Q1 grasp alignment.
NON_TEST_CONTACT_EDGE_COMPENSATION_XY_M = [0.020, -0.010]

# When a Q3 workpiece leans toward Q4, the required push contact is made by
# the opposite fingertip edge relative to the other Q2--Q4 push cases.
Q3_TOWARD_Q4_CONTACT_EDGE_COMPENSATION_XY_M = [-0.010, -0.010]

# Q2 workpieces leaning toward Q3 use the same opposite contact edge.
Q2_TOWARD_Q3_CONTACT_EDGE_COMPENSATION_XY_M = [-0.015, -0.005]

# For Q3 toward Q2, the shifted contact point is the new purple point used
# to construct every blue-line waypoint.
Q3_TOWARD_Q2_CONTACT_EDGE_COMPENSATION_XY_M = [-0.020, -0.015]

# A LIN planner interpolates the transfer-base frame, not the virtual TCP.
# Splitting a TCP-pivot turn into small chords keeps the fingertip centre
# effectively fixed while allowing the transfer-base X/Y coordinates to move.
DEFAULT_YAW_PIVOT_STEP_DEG = 2.0
DEFAULT_LINE_ENTRY_CLEARANCE_M = 0.010
DEFAULT_LINE_MAX_REAR_DISTANCE_M = 0.100
DEFAULT_PUSH_Z_TOLERANCE_M = 0.0005
DEFAULT_Q1_Z_TOLERANCE_M = 0.0001
DEFAULT_Q1_PITCH_TOLERANCE_RAD = math.radians(0.1)
DEFAULT_Q1_POST_GRASP_LIFT_M = (0.050, 0.040, 0.030, 0.020, 0.010)
Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE = 20


class TrajectoryPlanningError(RuntimeError):
    """A planner-side failure, distinct from an execution/follow failure."""


class NarrowQ1GraspError(RuntimeError):
    """The fingers stopped on the workpiece's narrow end, not a safe grasp."""


class Q1DirectPathUnavailableError(RuntimeError):
    """All direct-Q1 approach candidates are unplannable; reorient the part."""


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to [0, 360)."""
    return angle_deg % 360.0


def push_yaw_policy(workpiece_yaw_deg: float, quadrant: int) -> tuple[float, str, bool]:
    """Return target yaw, human-readable policy, and whether this is a grasp.

    Within Q2--Q4, the 45-degree midpoint selects the adjacent quadrant the
    workpiece is leaning toward. Acute branches retain the 75-degree policy.
    The only obtuse branches are Q2=[135,180), Q3=[180,225), and
    Q4=[270,315). Q2/Q4 use a 110-degree offset. Q3 toward Q2 uses a
    145-degree offset, so its push line forms the requested obtuse angle.
    """
    if quadrant == 1:
        return normalize_angle_deg(workpiece_yaw_deg), "q1_grasp_align", True
    if quadrant == 2:
        if 135.0 <= workpiece_yaw_deg < 180.0:
            return normalize_angle_deg(workpiece_yaw_deg - 110.0), "q2_toward_q3_obtuse_minus_110", False
        return normalize_angle_deg(workpiece_yaw_deg - 75.0), "q2_toward_q1_acute_minus_75", False
    if quadrant == 3:
        if 180.0 <= workpiece_yaw_deg < 225.0:
            return normalize_angle_deg(workpiece_yaw_deg - 145.0), "q3_toward_q2_obtuse_minus_145", False
        return normalize_angle_deg(workpiece_yaw_deg + 75.0), "q3_toward_q4_acute_plus_75", False
    if quadrant == 4:
        if 270.0 <= workpiece_yaw_deg < 315.0:
            return normalize_angle_deg(workpiece_yaw_deg + 110.0), "q4_toward_q3_obtuse_plus_110", False
        return normalize_angle_deg(workpiece_yaw_deg + 75.0), "q4_toward_q1_acute_plus_75", False
    raise ValueError(f"head_direction_base_quadrant 必须在 1--4，实际为 {quadrant!r}")


def load_workpiece(path: Path, index: int) -> tuple[float, int, list[float], list[float]]:
    """Load base yaw/quadrant, centre point, and push-entry point from pose JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"工件位姿 JSON 无法读取: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"工件位姿 JSON 无法解析: {path}: {exc}") from exc
    if payload.get("coordinate_frame") not in (None, "body_base"):
        raise ValueError(f"工件 JSON 坐标系不是 body_base: {payload.get('coordinate_frame')!r}")
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list) or not 0 <= index < len(workpieces):
        count = len(workpieces) if isinstance(workpieces, list) else 0
        raise ValueError(f"--workpiece-index={index} 超出范围；文件中共有 {count} 个工件")
    entry = workpieces[index]
    if not isinstance(entry, dict):
        raise ValueError(f"workpieces[{index}] 不是对象")
    push_point = entry.get("push_entry_point")
    if not isinstance(push_point, dict):
        raise ValueError("JSON 缺少 push_entry_point；请用更新后的 mask_workpiece_pose.py 重新生成")
    yaw_deg = finite([entry.get("head_direction_base_angle_deg")], 1, "head_direction_base_angle_deg")[0]
    quadrant_raw = entry.get("head_direction_base_quadrant")
    if not isinstance(quadrant_raw, int):
        raise ValueError("JSON 缺少整数 head_direction_base_quadrant")
    center_xy = finite(entry.get("center_base_xy_m"), 2, "center_base_xy_m")
    point_xy = finite(push_point.get("base_xy_m"), 2, "push_entry_point.base_xy_m")
    return normalize_angle_deg(yaw_deg), quadrant_raw, center_xy, point_xy


def load_reference_transfer_base_pose(path: Path, frame_id: str) -> list[float]:
    """Read the saved body_base -> right_hand_transfer_base pose."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"初始高度位姿文件无法读取: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"初始高度位姿文件无法解析: {path}: {exc}") from exc
    if payload.get("eef_pose_reference_frame") not in (None, frame_id):
        raise ValueError(
            f"初始高度位姿参考系不是 {frame_id!r}: {payload.get('eef_pose_reference_frame')!r}"
        )
    try:
        pose = payload["arms"]["right"]["eef_pose"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"初始高度位姿缺少 arms.right.eef_pose: {path}") from exc
    xyz = finite(pose.get("xyz"), 3, "初始高度 eef_pose.xyz")
    rotvec = finite(pose.get("rotvec"), 3, "初始高度 eef_pose.rotvec")
    return xyz + rotvec


def load_reference_tcp_pose(path: Path, tcp_offset: list[float], frame_id: str) -> list[float]:
    """Read the saved transfer-base pose and convert it to the closed TCP pose."""
    return compose_pose(load_reference_transfer_base_pose(path, frame_id), tcp_offset)


def set_gripper_state(
    gripper: object,
    *,
    arm_id: int,
    target_open: bool,
    timeout_s: float,
    tolerance: float,
) -> None:
    """Command the gripper and wait until its reported position reaches it."""
    target_position = 100.0 if target_open else 0.0
    action = "打开" if target_open else "闭合"
    gripper.move(arm_id, target_position, mode="async", preempt=True)
    print(f"已下发右夹爪{action}命令，等待到位...")
    deadline = time.monotonic() + timeout_s
    last_position: float | None = None
    while time.monotonic() < deadline:
        states = gripper.get_state()
        state = states.get(arm_id) if isinstance(states, dict) else None
        if isinstance(state, dict) and state.get("position") is not None:
            last_position = float(state["position"])
            if abs(last_position - target_position) <= tolerance:
                print(f"右夹爪已{action}到位（position={last_position:.1f}）。")
                return
        time.sleep(0.05)
    observed = "无有效状态" if last_position is None else f"position={last_position:.1f}"
    raise RuntimeError(f"右夹爪{action}等待超时（{observed}）。机械臂不会继续运动。")


def close_gripper_for_q1_grasp(
    gripper: object,
    *,
    arm_id: int,
    timeout_s: float,
    fully_closed_max_position: float,
    valid_grasp_min_position: float,
    empty_grasp_max_abs_force: float,
) -> None:
    """Close on the Q1 workpiece and classify empty/narrow/valid grasps.

    Position 0 means fully closed and 100 means fully open.  A successful
    grasp must stop at a sufficiently wide opening.  The calibrated ranges
    are: empty when fully closed *and* force is low; narrow/unsafe when the
    fingers stop at an intermediate opening or fully close under high force;
    and valid >= ``valid_grasp_min_position``.
    """
    initial_position: float | None = None
    try:
        before_state = gripper.get_state().get(arm_id)
        if isinstance(before_state, dict) and before_state.get("position") is not None:
            initial_position = float(before_state["position"])
    except Exception:
        # The close command and later confirmation still provide the useful
        # failure mode when a pre-command status frame is unavailable.
        pass
    gripper.move(arm_id, 0.0, mode="async", preempt=True)
    print("已下发右夹爪 Q1 闭合抓取命令，等待夹爪接触工件并停止...")
    deadline = time.monotonic() + timeout_s
    last_position: float | None = None
    last_motion_state: str | None = None
    last_force: float | None = None
    observed_closing = False
    while time.monotonic() < deadline:
        states = gripper.get_state()
        state = states.get(arm_id) if isinstance(states, dict) else None
        if isinstance(state, dict) and state.get("position") is not None:
            last_position = float(state["position"])
            raw_force = state.get("force")
            if isinstance(raw_force, (int, float)):
                last_force = float(raw_force)
            raw_motion_state = state.get("motion_state")
            last_motion_state = str(raw_motion_state).lower() if raw_motion_state is not None else None
            # A just-submitted async command can initially expose the previous
            # open-state frame: position=100, motion_state=arrived.  It is not
            # a grasp result.  Require evidence that this close command began.
            if last_motion_state == "moving" or (
                initial_position is not None and last_position < initial_position - 1.0
            ):
                observed_closing = True
            # This gripper reports `gripping` when its fingers are blocked by
            # a workpiece; `arrived` is used when it reaches an unobstructed
            # commanded opening.  Both are terminal states, but only a
            # non-zero opening represents a successful Q1 grasp.
            if last_motion_state in {"arrived", "gripping"} and observed_closing:
                if last_position <= fully_closed_max_position:
                    if last_force is not None and abs(last_force) > empty_grasp_max_abs_force:
                        raise NarrowQ1GraspError(
                            "Q1 抓取失败：夹爪虽已闭合但检测到高夹持力，判定为抓到工件窄处"
                            f"（position={last_position:.1f} <= {fully_closed_max_position:.1f}，"
                            f"force={last_force:.1f}，空抓最大绝对力={empty_grasp_max_abs_force:.1f}）。"
                        )
                    raise RuntimeError(
                        "Q1 抓取失败：夹爪已完全闭合"
                        f"（position={last_position:.1f} <= {fully_closed_max_position:.1f}，"
                        f"force={last_force!r}），判定为空抓。"
                    )
                if last_position < valid_grasp_min_position:
                    raise NarrowQ1GraspError(
                        "Q1 抓取失败：夹爪仅抓到工件窄处，不能安全搬运"
                        f"（position={last_position:.1f}，空抓上限="
                        f"{fully_closed_max_position:.1f}，force={last_force!r}，"
                        f"正确抓取下限={valid_grasp_min_position:.1f}），"
                        "判定为窄抓。"
                    )
                print(
                    "Q1 抓取成功：夹爪在工件正确位置停止"
                    f"（position={last_position:.1f} >= {valid_grasp_min_position:.1f}, "
                    f"motion_state={last_motion_state}）。"
                )
                return
            if last_motion_state in {"error", "fault", "failed"}:
                raise RuntimeError(f"Q1 闭爪失败：gripper motion_state={raw_motion_state!r}")
        time.sleep(0.05)
    observed = "无有效状态" if last_position is None else (
        f"position={last_position:.1f}, motion_state={last_motion_state!r}, observed_closing={observed_closing}"
    )
    raise RuntimeError(f"Q1 闭爪抓取等待超时（{observed}）；未确认夹住工件。")


def plan_and_follow(
    arm: object,
    planner: object,
    *,
    arm_id: int,
    group: str,
    frame_id: str,
    ik_frame: str,
    tcp_offset: list[float],
    target: list[float],
    target_is_tcp: bool,
    planner_kind: str,
    max_vel: float,
    max_acc: float,
    retries: int,
    stage: str,
) -> None:
    """Plan one TCP or IK-frame target and execute it with fresh retries."""
    if target_is_tcp:
        planner_target = compose_pose(target, inverse_pose(tcp_offset))
        target_kind = "TCP"
    else:
        planner_target = target
        target_kind = "IK frame"
    print(f"{stage} {target_kind} target [xyz(m), rotvec(rad)]: {[round(value, 6) for value in target]}")
    record_target = {
        "frame_id": frame_id,
        "ik_frame": ik_frame,
        "position_m": planner_target[:3],
        "orientation_xyzw": rotvec_to_quat(planner_target[3:]),
        "tcp_target_xyz_rotvec": target if target_is_tcp else None,
        "max_vel": max_vel,
        "max_acc": max_acc,
    }
    for attempt in range(retries + 1):
        start_tool_pose = capture_tool_pose(planner, frame_id, ik_frame)
        try:
            result = planner.plan_cart_trajectory(
                group=group,
                planner=planner_kind,
                frame_id=frame_id,
                ik_frame=ik_frame,
                position=planner_target[:3],
                orientation=rotvec_to_quat(planner_target[3:]),
                max_vel=max_vel,
                max_acc=max_acc,
            )
        except Exception as exc:
            record_plan(
                module="8-1", stage=f"{stage}:rpc:{attempt + 1}", status="failure",
                planner=planner_kind, request_type="cartesian", target=record_target, error=exc,
                start_tool_pose=start_tool_pose,
            )
            raise TrajectoryPlanningError(f"轨迹规划失败: {exc}") from exc
        data = result.get("data") or {}
        trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
        frequency = data.get("sample_frequency", 10.0)
        if planned_arm_id != arm_id:
            error = TrajectoryPlanningError(f"{stage} 规划结果 arm_id={planned_arm_id!r}，与请求不一致")
            record_plan(module="8-1", stage=f"{stage}:rpc:{attempt + 1}", status="failure",
                        planner=planner_kind, request_type="cartesian", target=record_target, response=result,
                        error=error, start_tool_pose=start_tool_pose)
            raise error
        if not isinstance(trajectory, list) or not trajectory:
            error = TrajectoryPlanningError(f"{stage} 规划服务未返回有效 trajectory")
            record_plan(module="8-1", stage=f"{stage}:rpc:{attempt + 1}", status="failure",
                        planner=planner_kind, request_type="cartesian", target=record_target, response=result,
                        error=error, start_tool_pose=start_tool_pose)
            raise error
        if not isinstance(frequency, (int, float)) or frequency <= 0:
            error = TrajectoryPlanningError(f"{stage} 规划服务返回的 sample_frequency 无效: {frequency!r}")
            record_plan(module="8-1", stage=f"{stage}:rpc:{attempt + 1}", status="failure",
                        planner=planner_kind, request_type="cartesian", target=record_target, response=result,
                        error=error, start_tool_pose=start_tool_pose)
            raise error
        record_plan(module="8-1", stage=f"{stage}:rpc:{attempt + 1}", status="success",
                    planner=planner_kind, request_type="cartesian", target=record_target, response=result,
                    start_tool_pose=start_tool_pose)
        print(f"{stage} 规划完成：{len(trajectory)} 个关节路点，采样频率 {frequency} Hz。")
        try:
            arm.follow_trajectory(arm_id=planned_arm_id, trajectory=trajectory, sample_frequency=float(frequency))
            return
        except Exception as exc:
            if "DiscreteTask::validate: first point mismatch" not in str(exc) or attempt >= retries:
                raise
            print(f"{stage} 起点不同步，重新规划（{attempt + 1}/{retries}）...")
            time.sleep(0.2)


def preplan_tcp_trajectory(
    planner: object,
    *,
    arm_id: int,
    group: str,
    frame_id: str,
    ik_frame: str,
    tcp_offset: list[float],
    target: list[float],
    planner_kind: str,
    max_vel: float,
    max_acc: float,
    timeout_s: float = 5.0,
) -> tuple[list, float]:
    """Plan one TCP target and retain that exact trajectory for execution."""
    planner_target = compose_pose(target, inverse_pose(tcp_offset))
    record_target = {
        "frame_id": frame_id, "ik_frame": ik_frame,
        "position_m": planner_target[:3], "orientation_xyzw": rotvec_to_quat(planner_target[3:]),
        "tcp_target_xyz_rotvec": target, "max_vel": max_vel, "max_acc": max_acc, "timeout_s": timeout_s,
    }
    start_tool_pose = capture_tool_pose(planner, frame_id, ik_frame)
    try:
        result = planner.plan_cart_trajectory(
            group=group,
            planner=planner_kind,
            frame_id=frame_id,
            ik_frame=ik_frame,
            position=planner_target[:3],
            orientation=rotvec_to_quat(planner_target[3:]),
            max_vel=max_vel,
            max_acc=max_acc,
            timeout=timeout_s,
        )
    except Exception as exc:
        record_plan(module="8-1", stage="preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, error=exc, start_tool_pose=start_tool_pose)
        raise TrajectoryPlanningError(f"轨迹规划失败: {exc}") from exc
    data = result.get("data") or {}
    trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
    frequency = data.get("sample_frequency", 10.0)
    if planned_arm_id != arm_id:
        error = TrajectoryPlanningError(f"规划结果 arm_id={planned_arm_id!r}，与请求不一致")
        record_plan(module="8-1", stage="preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        raise error
    if not isinstance(trajectory, list) or not trajectory:
        error = TrajectoryPlanningError("规划服务未返回有效 trajectory")
        record_plan(module="8-1", stage="preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        raise error
    if not isinstance(frequency, (int, float)) or frequency <= 0:
        error = TrajectoryPlanningError(f"规划服务返回的 sample_frequency 无效: {frequency!r}")
        record_plan(module="8-1", stage="preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        raise error
    record_plan(module="8-1", stage="preplan_tcp", status="success", planner=planner_kind,
                request_type="cartesian", target=record_target, response=result, start_tool_pose=start_tool_pose)
    return trajectory, float(frequency)


def can_plan_target(
    planner: object,
    *,
    arm_id: int,
    group: str,
    frame_id: str,
    ik_frame: str,
    tcp_offset: list[float],
    target: list[float],
    target_is_tcp: bool,
    planner_kind: str,
    max_vel: float,
    max_acc: float,
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """Check a Cartesian target without sending any arm motion command."""
    planner_target = compose_pose(target, inverse_pose(tcp_offset)) if target_is_tcp else target
    record_target = {
        "frame_id": frame_id, "ik_frame": ik_frame,
        "position_m": planner_target[:3], "orientation_xyzw": rotvec_to_quat(planner_target[3:]),
        "tcp_target_xyz_rotvec": target if target_is_tcp else None,
        "max_vel": max_vel, "max_acc": max_acc, "timeout_s": timeout_s,
    }
    start_tool_pose = capture_tool_pose(planner, frame_id, ik_frame)
    try:
        result = planner.plan_cart_trajectory(
            group=group,
            planner=planner_kind,
            frame_id=frame_id,
            ik_frame=ik_frame,
            position=planner_target[:3],
            orientation=rotvec_to_quat(planner_target[3:]),
            max_vel=max_vel,
            max_acc=max_acc,
            timeout=timeout_s,
        )
    except Exception as exc:
        record_plan(module="8-1", stage="can_plan_target", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, error=exc, start_tool_pose=start_tool_pose)
        return False, str(exc)
    data = result.get("data") or {}
    trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
    if planned_arm_id != arm_id:
        error = f"arm_id={planned_arm_id!r}"
        record_plan(module="8-1", stage="can_plan_target", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        return False, error
    if not isinstance(trajectory, list) or not trajectory:
        error = "规划器未返回有效 trajectory"
        record_plan(module="8-1", stage="can_plan_target", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        return False, error
    record_plan(module="8-1", stage="can_plan_target", status="success", planner=planner_kind,
                request_type="cartesian", target=record_target, response=result, start_tool_pose=start_tool_pose)
    return True, f"{len(trajectory)} 个关节路点"


def verify_and_recover_stage2_tcp(
    arm: object,
    planner: object,
    *,
    arm_id: int,
    group: str,
    frame_id: str,
    ik_frame: str,
    tcp_offset: list[float],
    target: list[float],
    planner_kind: str,
    max_vel: float,
    max_acc: float,
    start_sync_retries: int,
    z_tolerance_m: float,
    pitch_tolerance_rad: float,
    correction_attempts: int,
    stage: str,
    continue_after_exhausted: bool = False,
) -> list[float]:
    """Close the z/pitch loop after one stage-2 yaw segment.

    ``follow_trajectory`` accepting a trajectory only proves that it was sent;
    it does not prove that the physical arm reached the virtual-TCP endpoint.
    Keep the measured XY fixed while retrying the requested z/pitch/yaw.
    """
    expected_pitch = rotvec_to_rpy(target[3:])[1]
    for attempt in range(correction_attempts + 1):
        # Planner TF is updated asynchronously after trajectory execution.
        time.sleep(0.2)
        current_ik = pose_from_lookup_transform(planner.lookup_transform(frame_id, ik_frame).get("data"))
        current_tcp = compose_pose(current_ik, tcp_offset)
        current_pitch = rotvec_to_rpy(current_tcp[3:])[1]
        z_error_m = current_tcp[2] - target[2]
        pitch_error_rad = current_pitch - expected_pitch
        if abs(z_error_m) <= z_tolerance_m and abs(pitch_error_rad) <= pitch_tolerance_rad:
            if attempt:
                print(
                    f"{stage} 闭环恢复成功（{attempt}/{correction_attempts}）："
                    f"z 偏差={z_error_m * 1000.0:.3f} mm，"
                    f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°。"
                )
            return current_tcp
        if attempt >= correction_attempts:
            if continue_after_exhausted:
                print(
                    f"警告：{stage} 已执行 {correction_attempts} 次恢复，"
                    f"仍有 z 偏差={z_error_m * 1000.0:.3f} mm、"
                    f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°；按当前实际姿态继续。"
                )
                return current_tcp
            raise RuntimeError(
                f"{stage} 闭环恢复失败：z 偏差={z_error_m * 1000.0:.3f} mm，"
                f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°；"
                f"已执行 {correction_attempts} 次恢复，拒绝继续转 yaw。"
            )
        print(
            f"{stage} 实际 TCP 未到位：z 偏差={z_error_m * 1000.0:.3f} mm，"
            f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°；"
            f"固定当前 XY，恢复目标 z/pitch/yaw（{attempt + 1}/{correction_attempts}）。"
        )
        recovery_target = current_tcp[:2] + [target[2]] + target[3:]
        try:
            plan_and_follow(
                arm, planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target=recovery_target, target_is_tcp=True, planner_kind=planner_kind,
                max_vel=max_vel, max_acc=max_acc, retries=start_sync_retries,
                stage=f"{stage}：TCP z/pitch 闭环恢复（{attempt + 1}/{correction_attempts}）",
            )
        except TrajectoryPlanningError as exc:
            raise RuntimeError(f"{stage} TCP z/pitch 闭环恢复规划失败；拒绝继续转 yaw：{exc}") from exc
    raise AssertionError("unreachable")


def recover_q1_transfer_base_z_pitch_with_planner(
    arm: object,
    planner: object,
    *,
    arm_id: int,
    group: str,
    frame_id: str,
    ik_frame: str,
    reference_transfer_base: list[float],
    planner_kind: str,
    max_vel: float,
    max_acc: float,
    start_sync_retries: int,
    z_tolerance_m: float,
    pitch_tolerance_rad: float,
    correction_attempts: int,
    stage: str,
) -> list[float]:
    """Recover world z/pitch against the live transfer-base start snapshot.

    The snapshot is taken after reaching pose 3, rather than read from
    3.json.  Do not use its tilted local Z: horizontal motion would otherwise
    appear as a false local-z error and command the wrist downward.
    """
    expected_pitch = rotvec_to_rpy(reference_transfer_base[3:])[1]
    current_ik: list[float] | None = None
    for attempt in range(correction_attempts + 1):
        time.sleep(0.2)
        current_ik = pose_from_lookup_transform(planner.lookup_transform(frame_id, ik_frame).get("data"))
        current_pitch = rotvec_to_rpy(current_ik[3:])[1]
        z_error_m = current_ik[2] - reference_transfer_base[2]
        pitch_error_rad = current_pitch - expected_pitch
        if abs(z_error_m) <= z_tolerance_m and abs(pitch_error_rad) <= pitch_tolerance_rad:
            if attempt:
                print(
                    f"{stage} planner LIN 恢复成功（{attempt}/{correction_attempts}）："
                    f"z 偏差={z_error_m * 1000.0:.3f} mm，"
                    f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°。"
                )
            return current_ik
        if attempt >= correction_attempts:
            print(
                f"警告：{stage} planner LIN 已执行 {correction_attempts} 次，"
                f"仍有 z 偏差={z_error_m * 1000.0:.3f} mm、"
                f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°；按当前实际姿态继续。"
            )
            return current_ik

        current_rpy = rotvec_to_rpy(current_ik[3:])
        recovery_target = current_ik[:2] + [reference_transfer_base[2]] + rpy_to_rotvec(
            [current_rpy[0], expected_pitch, current_rpy[2]]
        )
        print(
            f"{stage} transfer-base 相对实时起点的 body_base z/pitch 未到位：z 偏差={z_error_m * 1000.0:.3f} mm，"
            f"pitch 偏差={math.degrees(pitch_error_rad):.3f}°；"
            f"固定当前 transfer-base XY/roll/yaw，通过 planner LIN 恢复 z/pitch（{attempt + 1}/{correction_attempts}）。"
        )
        try:
            plan_and_follow(
                arm, planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
                tcp_offset=[0.0] * 6, target=recovery_target, target_is_tcp=False, planner_kind=planner_kind,
                max_vel=max_vel, max_acc=max_acc, retries=start_sync_retries,
                stage=f"{stage}：transfer-base 实时起点 z/pitch planner LIN 恢复（{attempt + 1}/{correction_attempts})",
            )
        except TrajectoryPlanningError as exc:
            raise RuntimeError(f"{stage} transfer-base z/pitch planner LIN 恢复规划失败：{exc}") from exc
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", type=int, choices=(2,), required=True, help="目前仅支持右臂 2")
    parser.add_argument("--workpiece-pose-file", type=Path, required=True)
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    parser.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "localhost"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
    parser.add_argument("--frame-id", default=os.environ.get("PARM_PLANNER_FRAME", "body_base"))
    parser.add_argument("--planner", choices=("lin",), default="lin")
    parser.add_argument(
        "--yaw-pivot-step-deg", type=float, default=DEFAULT_YAW_PIVOT_STEP_DEG,
        help="阶段 1 每个 LIN 小段的最大 yaw（度；默认 2）",
    )
    parser.add_argument(
        "--line-entry-clearance-m", type=float, default=DEFAULT_LINE_ENTRY_CLEARANCE_M,
        help="对齐蓝线时距紫色切入点的最小后方余量（默认 0.010 m）",
    )
    parser.add_argument(
        "--line-max-rear-distance-m", type=float, default=DEFAULT_LINE_MAX_REAR_DISTANCE_M,
        help="对齐蓝线时距紫色切入点的最大后方距离（默认 0.100 m）",
    )
    parser.add_argument(
        "--line-scan-count", type=int, default=15,
        help="阶段 1 在蓝线安全窗口内预规划的候选点数（默认 9）",
    )
    parser.add_argument(
        "--line-yaw-recovery-rear-distance-m", type=float, default=0.200,
        help="阶段 2 yaw 失败时，沿蓝线向后重定位的最大距离（默认 0.200 m；绝不越过紫点）",
    )
    parser.add_argument("--tcp", choices=("right-fingertip-center",), default="right-fingertip-center")
    parser.add_argument(
        "--planar-xy-yaw", action="store_true",
        help="兼容 move_by_eef_pose.py；本脚本始终锁定当前 z/roll/pitch",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="仅将 fingertip TCP 移到工件中心，并使 yaw 与工件红箭头（tail-to-head）方向一致；不操作夹爪、不推送",
    )
    parser.add_argument("--push-distance-m", type=float, default=0.090, help="Q2--Q4 经过切入点后的推送距离（默认 0.090 m）")
    parser.add_argument("--q1-grasp-advance-m", type=float, default=0.013, help="Q1 到达抓取点后额外推进距离（默认 0.013 m）")
    parser.add_argument(
        "--push-z-reference-pose-file", type=Path,
        default=SCRIPT_DIR / "saved_poses" / "5.json",
        help="阶段 3 推送高度基准的保存位姿文件（默认 saved_poses/5.json）",
    )
    parser.add_argument(
        "--q1-grasp-reference-pose-file", type=Path,
        default=SCRIPT_DIR / "saved_poses" / "3.json",
        help=(
            "兼容旧命令的 3.json 路径参数；实际 Q1 执行会在到达该姿态后读取实时 "
            "body_base→transfer_base 快照作为 z/pitch 基准，不修改此文件"
        ),
    )
    parser.add_argument(
        "--q1-z-tolerance-m", type=float, default=DEFAULT_Q1_Z_TOLERANCE_M,
        help="Q1 对齐及抓取前允许的 TCP z 偏差（默认 0.0001 m）",
    )
    parser.add_argument(
        "--push-z-tolerance-m", type=float, default=DEFAULT_PUSH_Z_TOLERANCE_M,
        help="阶段 3 前允许的 TCP z 偏差（默认 0.0005 m）",
    )
    parser.add_argument(
        "--push-z-correction-attempts", type=int, default=3,
        help="阶段 3 前 TCP z 的最多闭环修正次数（默认 3；仍有偏差也继续推送）",
    )
    parser.add_argument(
        "--approach-distance-m", type=float, default=0.070,
        help="Q2--Q4 推送前，在切入点反向延长线上的预接触距离（默认 0.070 m）",
    )
    parser.add_argument("--max-vel", type=float, default=0.1)
    parser.add_argument("--max-acc", type=float, default=0.1)
    parser.add_argument("--start-sync-retries", type=int, default=3)
    parser.add_argument("--gripper-timeout-s", type=float, default=5.0, help="等待夹爪到位的超时（默认 5 秒）")
    parser.add_argument("--gripper-position-tolerance", type=float, default=5.0, help="夹爪到位位置容差（默认 5，范围 0--100）")
    parser.add_argument(
        "--q1-fully-closed-max-position", type=float, default=0.5,
        help=(
            "Q1 最终闭爪时判为空抓的最大开度；position <= 此值表示空载完全闭合 "
            "（默认 0.5；当前空载实测 position=0.0，抓住窄处的 1.4 不应判空抓）"
        ),
    )
    parser.add_argument(
        "--q1-valid-grasp-min-position", type=float, default=15.0,
        help=(
            "Q1 正确抓取的最小开度；空抓上限与此值之间判为抓到窄处并拒绝搬运 "
            "（默认 15；当前正确抓取实测 position=23.8，窄抓实测 1.4）"
        ),
    )
    parser.add_argument(
        "--q1-empty-grasp-max-abs-force", type=float, default=20.0,
        help=(
            "Q1 夹爪完全闭合时仍判为空抓的最大绝对夹持力；超过此值说明抓到极窄工件，"
            "执行窄抓 +Z 复抓（默认 20；空抓实测约 4，窄抓实测约 60）"
        ),
    )
    parser.add_argument(
        "--q1-narrow-grasp-lift-m", type=float, default=0.0025,
        help="Q1 窄抓时松爪后、原地重新抓取前 TCP 仅沿 +Z 抬高的距离（默认 0.0025 m）",
    )
    parser.add_argument(
        "--q1-narrow-grasp-retries", type=int, default=1,
        help="Q1 窄抓后的原地 +Z 抬高复抓次数（默认 1）",
    )
    parser.add_argument(
        "--q1-post-grasp-lift-m", nargs="+", type=float, default=DEFAULT_Q1_POST_GRASP_LIFT_M,
        help="Q1 抓住后仅沿 body_base +Z 上抬的候选高度（m，按优先顺序；默认 0.05 0.04 0.03 0.02 0.01）",
    )
    parser.add_argument(
        "--q1-pitch-tolerance-rad", type=float, default=DEFAULT_Q1_PITCH_TOLERANCE_RAD,
        help="Q1 对齐及抓取前允许的初始 pitch 偏差（rad，默认 0.1 度）",
    )
    parser.add_argument(
        "--stage2-pitch-tolerance-rad", type=float, default=0.035,
        help="阶段 2 每个 yaw 小段的 TCP pitch 闭环容差（rad，默认 0.035）",
    )
    parser.add_argument(
        "--stage2-correction-attempts", type=int, default=8,
        help="阶段 2 每个 yaw 小段未到位时的 TCP z/pitch 闭环恢复次数（默认 8）",
    )
    parser.add_argument("--execute", action="store_true", help="实际下发运动；省略时仅预览")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.workpiece_pose_file.is_file():
        print(f"错误: 工件位姿文件不存在: {args.workpiece_pose_file}", file=sys.stderr)
        return 1
    if args.push_distance_m <= 0.0 or args.approach_distance_m <= 0.0 or args.q1_grasp_advance_m <= 0.0:
        print("错误: --push-distance-m、--approach-distance-m 和 --q1-grasp-advance-m 必须大于 0", file=sys.stderr)
        return 1
    if args.gripper_timeout_s <= 0.0 or not 0.0 <= args.gripper_position_tolerance <= 100.0:
        print("错误: 夹爪超时必须大于 0，position 容差必须在 [0, 100]", file=sys.stderr)
        return 1
    if not 0.0 <= args.q1_fully_closed_max_position < 100.0:
        print("错误: --q1-fully-closed-max-position 必须在 [0, 100) 内", file=sys.stderr)
        return 1
    if not args.q1_fully_closed_max_position < args.q1_valid_grasp_min_position <= 100.0:
        print(
            "错误: --q1-valid-grasp-min-position 必须大于空抓上限且不大于 100",
            file=sys.stderr,
        )
        return 1
    if args.q1_empty_grasp_max_abs_force < 0.0:
        print("错误: --q1-empty-grasp-max-abs-force 不能小于 0", file=sys.stderr)
        return 1
    if args.q1_narrow_grasp_lift_m <= 0.0 or args.q1_narrow_grasp_retries < 0:
        print("错误: Q1 窄抓上抬距离必须大于 0，复抓次数不能小于 0", file=sys.stderr)
        return 1
    if not args.q1_post_grasp_lift_m or any(height <= 0.0 for height in args.q1_post_grasp_lift_m):
        print("错误: --q1-post-grasp-lift-m 必须至少给出一个正数高度", file=sys.stderr)
        return 1
    if args.q1_z_tolerance_m < 0.0 or args.q1_pitch_tolerance_rad < 0.0:
        print("错误: --q1-z-tolerance-m 和 --q1-pitch-tolerance-rad 不能小于 0", file=sys.stderr)
        return 1
    if args.stage2_pitch_tolerance_rad < 0.0:
        print("错误: --stage2-pitch-tolerance-rad 不能小于 0", file=sys.stderr)
        return 1
    if args.stage2_correction_attempts < 1:
        print("错误: --stage2-correction-attempts 必须至少为 1", file=sys.stderr)
        return 1
    if not 0.0 < args.max_vel <= 1.0 or not 0.0 < args.max_acc <= 1.0:
        print("错误: --max-vel 和 --max-acc 必须在 (0, 1] 内", file=sys.stderr)
        return 1
    if not 0.0 < args.yaw_pivot_step_deg <= 15.0:
        print("错误: --yaw-pivot-step-deg 必须在 (0, 15] 内", file=sys.stderr)
        return 1
    if args.line_entry_clearance_m < 0.0:
        print("错误: --line-entry-clearance-m 不能小于 0", file=sys.stderr)
        return 1
    if args.line_max_rear_distance_m < args.line_entry_clearance_m:
        print("错误: --line-max-rear-distance-m 必须不小于 --line-entry-clearance-m", file=sys.stderr)
        return 1
    if args.line_scan_count < 2:
        print("错误: --line-scan-count 必须至少为 2", file=sys.stderr)
        return 1
    if args.line_yaw_recovery_rear_distance_m <= 0.0:
        print("错误: --line-yaw-recovery-rear-distance-m 必须大于 0", file=sys.stderr)
        return 1
    if args.push_z_tolerance_m < 0.0:
        print("错误: --push-z-tolerance-m 不能小于 0", file=sys.stderr)
        return 1
    if args.push_z_correction_attempts < 1:
        print("错误: --push-z-correction-attempts 必须至少为 1", file=sys.stderr)
        return 1
    try:
        workpiece_yaw_deg, quadrant, center_xy, entry_xy = load_workpiece(args.workpiece_pose_file, args.workpiece_index)
        target_yaw_deg, policy, is_grasp = push_yaw_policy(workpiece_yaw_deg, quadrant)
        _, tcp_offset, correction_frame, correction = load_virtual_tcp_offset(
            RIGHT_FINGERTIP_CENTER_CONFIG, "right_hand_transfer_base"
        )
        push_reference_tcp_pose = None if args.test else load_reference_tcp_pose(
            args.push_z_reference_pose_file, tcp_offset, args.frame_id
        )
        # Q1's reference is deliberately captured live after the pipeline has
        # reached pose 3.  Do not use the saved 3.json transform as a motion
        # reference: it is only the taught joint destination that gets us to
        # the starting posture.
        q1_grasp_reference_tcp_pose: list[float] | None = None
        q1_grasp_reference_transfer_base_pose: list[float] | None = None
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    push_reference_tcp_z = None if push_reference_tcp_pose is None else push_reference_tcp_pose[2]
    push_reference_tcp_rpy = (
        None if push_reference_tcp_pose is None else rotvec_to_rpy(push_reference_tcp_pose[3:])
    )
    q1_grasp_reference_tcp_rpy: list[float] | None = None
    if correction_frame != args.frame_id:
        print(
            f"错误: TCP 目标位置校正定义在 {correction_frame!r}，规划坐标系为 {args.frame_id!r}",
            file=sys.stderr,
        )
        return 1
    center_xy = [center_xy[0] + correction[0], center_xy[1] + correction[1]]
    entry_xy = [entry_xy[0] + correction[0], entry_xy[1] + correction[1]]
    # The workpiece points are defined for the closed two-fingertip centre,
    # while the current virtual TCP resolves to the right fingertip.  Apply
    # this to both test-centre and normal push/grasp targets.
    center_xy = [
        center_xy[axis] + TEST_CLOSED_FINGERTIP_CENTER_COMPENSATION_XY_M[axis]
        for axis in range(2)
    ]
    entry_xy = [
        entry_xy[axis] + TEST_CLOSED_FINGERTIP_CENTER_COMPENSATION_XY_M[axis]
        for axis in range(2)
    ]
    if args.test:
        # The red arrow in the pose overlay is tail-to-head, exactly the yaw
        # stored in head_direction_base_angle_deg.  Do not apply the push
        # policy: this mode is intended solely to verify centre/yaw alignment.
        target_yaw_deg = workpiece_yaw_deg
        policy = "test_center_align_workpiece_yaw"
    elif not is_grasp:
        # Shift the contact point so the left edge of the closed fingertips,
        # rather than their centre, reaches the purple push-entry point.
        contact_edge_compensation = NON_TEST_CONTACT_EDGE_COMPENSATION_XY_M
        if quadrant == 2 and workpiece_yaw_deg >= 135.0:
            contact_edge_compensation = Q2_TOWARD_Q3_CONTACT_EDGE_COMPENSATION_XY_M
        elif quadrant == 3 and 180.0 <= workpiece_yaw_deg < 225.0:
            contact_edge_compensation = Q3_TOWARD_Q2_CONTACT_EDGE_COMPENSATION_XY_M
        elif quadrant == 3 and workpiece_yaw_deg >= 225.0:
            contact_edge_compensation = Q3_TOWARD_Q4_CONTACT_EDGE_COMPENSATION_XY_M
        entry_xy = [
            entry_xy[axis] + contact_edge_compensation[axis]
            for axis in range(2)
        ]
    print(f"工件 base yaw={workpiece_yaw_deg:.2f}° Q{quadrant}；策略={policy}")
    print(f"目标 fingertip yaw={target_yaw_deg:.2f}°")
    if args.test:
        print(f"测试模式：闭合两指尖中心对准绿点的 TCP 目标 xy(m)={center_xy}")
        print(
            "已加入闭合两指尖中心补偿（对全部模式生效）"
            f"xy(m)={TEST_CLOSED_FINGERTIP_CENTER_COMPENSATION_XY_M}"
        )
        print("测试模式：不操作夹爪，不执行推送；仅对齐工件中心与红箭头 yaw。")
    else:
        if is_grasp:
            print(f"push_entry_point（已应用 TCP 及闭合中心校正）xy(m)={entry_xy}")
        else:
            print(f"push_entry_point（已应用 TCP、闭合中心及接触边缘校正）xy(m)={entry_xy}")
            print(f"Q2--Q4 推送接触边缘校正 xy(m)={contact_edge_compensation}")
    push_direction: list[float] | None = None
    precontact_xy: list[float] | None = None
    push_end_xy: list[float] | None = None
    if args.test:
        pass
    elif is_grasp:
        yaw_rad = math.radians(target_yaw_deg)
        q1_direction = [math.cos(yaw_rad), math.sin(yaw_rad)]
        precontact_xy = [
            entry_xy[0] - args.approach_distance_m * q1_direction[0],
            entry_xy[1] - args.approach_distance_m * q1_direction[1],
        ]
        print(
            "Q1：开爪后在工件 yaw 的后方延长线预规划多个终点；选定一条后移+最终转向轨迹后一次执行，随后沿红箭头方向经过 tail_inside 并额外推进 "
            f"{args.q1_grasp_advance_m * 1000.0:.1f} mm，最后闭爪抓取。"
        )
        print(f"Q1 抓取线预接触点 xy(m)={precontact_xy}")
    else:
        yaw_rad = math.radians(target_yaw_deg)
        push_direction = [math.cos(yaw_rad), math.sin(yaw_rad)]
        precontact_xy = [
            entry_xy[0] - args.approach_distance_m * push_direction[0],
            entry_xy[1] - args.approach_distance_m * push_direction[1],
        ]
        push_end_xy = [
            entry_xy[0] + args.push_distance_m * push_direction[0],
            entry_xy[1] + args.push_distance_m * push_direction[1],
        ]
        print(
            f"Q{quadrant}：先闭爪并移动到目标 yaw 的后方延长线，再在该线转向；随后沿该线经过切入点并推 "
            f"{args.push_distance_m * 1000.0:.1f} mm。"
        )
        print(f"旧版固定 70 mm 预接触点（仅供参考）xy(m)={precontact_xy}")
        print(f"推送终点（紫点正向 {args.push_distance_m * 1000.0:.1f} mm）xy(m)={push_end_xy}")
    if not args.execute:
        if args.test:
            print("这是预览，未连接机械臂。添加 --execute 后将按当前 TCP 的 z/roll/pitch 移动，使闭合两指尖中心落在绿点并对齐 yaw。")
        elif is_grasp:
            print("这是预览，未连接机械臂。添加 --execute 后将沿 Q1 yaw 预接触、经过抓取点并额外推进后闭爪。")
        else:
            print("这是预览，未连接机械臂。添加 --execute 后将执行预接触和推送路径。")
        return 0

    arm = planner = gripper = None
    try:
        sdk = import_sdk()
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=args.planner_host)
        gripper = sdk.Gripper(host=args.gripper_host)
        group, ik_frame = "right_arm", "right_hand_transfer_base"
        if is_grasp and not args.test:
            # Freeze the *actual* transfer-base pose reached after the
            # pipeline moved to 3.json.  This is the requested
            # transfer_base_at_3 reference; subsequent Q1 z/pitch recovery
            # compares against this snapshot, never the saved JSON values.
            q1_grasp_reference_transfer_base_pose = pose_from_lookup_transform(
                planner.lookup_transform(args.frame_id, ik_frame).get("data")
            )
            q1_grasp_reference_tcp_pose = compose_pose(
                q1_grasp_reference_transfer_base_pose, tcp_offset
            )
            q1_grasp_reference_tcp_rpy = rotvec_to_rpy(q1_grasp_reference_tcp_pose[3:])
            snapshot_rpy = rotvec_to_rpy(q1_grasp_reference_transfer_base_pose[3:])
            print(
                "Q1 实时冻结 transfer_base_at_3 基准："
                f"transfer-base z={q1_grasp_reference_transfer_base_pose[2]:.6f} m，"
                f"pitch={math.degrees(snapshot_rpy[1]):.3f}°；"
                f"TCP z={q1_grasp_reference_tcp_pose[2]:.6f} m。"
            )
        if not args.test:
            set_gripper_state(
                gripper, arm_id=args.arm_id, target_open=is_grasp,
                timeout_s=args.gripper_timeout_s, tolerance=args.gripper_position_tolerance,
            )

        start_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
        start_tcp = compose_pose(start_ik, tcp_offset)
        start_tcp_rpy = rotvec_to_rpy(start_tcp[3:])
        start_yaw_deg = math.degrees(start_tcp_rpy[2])
        q1_reference_z: float | None = None
        q1_reference_roll: float | None = None
        q1_reference_pitch: float | None = None
        q1_reference_transfer_base: list[float] | None = None
        if args.test:
            # Use the same line-alignment workflow as a normal push: move to
            # a reachable point on the rear extension of the workpiece's
            # tail-to-head line, turn there, then advance to the centre.
            # Rotating at the initial pose can make the transfer-base IK
            # frame unreachable because a virtual TCP pivot moves that frame.
            line_direction = [math.cos(math.radians(target_yaw_deg)), math.sin(math.radians(target_yaw_deg))]
            line_projection_m = sum(
                (start_tcp[axis] - center_xy[axis]) * line_direction[axis]
                for axis in range(2)
            )
            line_parameter_m = min(
                max(line_projection_m, -args.line_max_rear_distance_m),
                -args.line_entry_clearance_m,
            )
            candidate_parameters = [line_parameter_m]
            for index in range(args.line_scan_count):
                rear_distance_m = args.line_max_rear_distance_m - (
                    (args.line_max_rear_distance_m - args.line_entry_clearance_m)
                    * index / (args.line_scan_count - 1)
                )
                candidate_parameter_m = -rear_distance_m
                if all(abs(candidate_parameter_m - known) > 1e-9 for known in candidate_parameters):
                    candidate_parameters.append(candidate_parameter_m)
            line_alignment_xy: list[float] | None = None
            scan_failures: list[str] = []
            print(
                f"阶段 1：固定当前 yaw={start_yaw_deg:.2f}°，扫描工件 tail-to-head 线后方安全窗口 "
                f"[{args.line_entry_clearance_m * 1000.0:.1f}, "
                f"{args.line_max_rear_distance_m * 1000.0:.1f}] mm；"
                f"投影={line_projection_m * 1000.0:.1f} mm，首选={line_parameter_m * 1000.0:.1f} mm。"
            )
            for scan_index, candidate_parameter_m in enumerate(candidate_parameters, start=1):
                candidate_xy = [
                    center_xy[axis] + candidate_parameter_m * line_direction[axis]
                    for axis in range(2)
                ]
                candidate_target = candidate_xy + [start_tcp[2]] + start_tcp[3:]
                feasible, detail = can_plan_target(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target=candidate_target, target_is_tcp=True, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc,
                )
                if feasible:
                    line_alignment_xy = candidate_xy
                    print(
                        f"阶段 1：红箭头线候选 {scan_index}/{len(candidate_parameters)} 可规划："
                        f"后方 {candidate_parameter_m * 1000.0:.1f} mm，xy(m)={candidate_xy}（{detail}）。"
                    )
                    break
                scan_failures.append(f"{candidate_parameter_m * 1000.0:.1f} mm: {detail}")
            if line_alignment_xy is None:
                raise RuntimeError(
                    "阶段 1：工件 tail-to-head 线后方安全窗口内没有可执行的 LIN 轨迹；"
                    + "；".join(scan_failures)
                )
            line_move_target = line_alignment_xy + [start_tcp[2]] + start_tcp[3:]
            plan_and_follow(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target=line_move_target, target_is_tcp=True, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                stage="阶段 1/3：保持当前 yaw 移动到工件红箭头线后方",
            )
            current_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
            current_tcp = compose_pose(current_ik, tcp_offset)
            current_rpy = rotvec_to_rpy(current_tcp[3:])
            current_yaw_deg = math.degrees(current_rpy[2])
            yaw_delta_deg = (target_yaw_deg - current_yaw_deg + 180.0) % 360.0 - 180.0
            yaw_steps = max(1, math.ceil(abs(yaw_delta_deg) / args.yaw_pivot_step_deg))
            print(
                f"阶段 2：固定红箭头线 xy(m)={line_alignment_xy}，转 yaw "
                f"{current_yaw_deg:.2f}° → {target_yaw_deg:.2f}°；分为 {yaw_steps} 段。"
            )
            for step in range(1, yaw_steps + 1):
                step_yaw_rad = math.radians(current_yaw_deg + yaw_delta_deg * step / yaw_steps)
                rotate_target = line_alignment_xy + [current_tcp[2]] + rpy_to_rotvec(
                    [current_rpy[0], current_rpy[1], step_yaw_rad]
                )
                plan_and_follow(
                    arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target=rotate_target, target_is_tcp=True, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                    stage=f"阶段 2/3：工件红箭头线原地转 yaw（{step}/{yaw_steps}）",
                )
        elif is_grasp:
            assert precontact_xy is not None
            # Q1 uses the complete taught grasp orientation from pose 3 as a
            # fixed reference.  Scan rear-line endpoints for a single LIN
            # trajectory that combines the XY retreat and final yaw alignment.
            assert (
                q1_grasp_reference_tcp_pose is not None
                and q1_grasp_reference_tcp_rpy is not None
                and q1_grasp_reference_transfer_base_pose is not None
            )
            q1_z = q1_grasp_reference_tcp_pose[2]
            q1_roll, q1_pitch, _q1_initial_yaw = q1_grasp_reference_tcp_rpy
            q1_reference_z = q1_z
            q1_reference_roll = q1_roll
            q1_reference_pitch = q1_pitch
            q1_reference_transfer_base = q1_grasp_reference_transfer_base_pose
            print(
                "Q1 抓取 z/roll/pitch 固定使用实时 transfer_base_at_3 快照："
                f"TCP z={q1_z:.6f} m，"
                f"roll={math.degrees(q1_roll):.3f}°，pitch={math.degrees(q1_pitch):.3f}°。"
            )
            line_direction = [math.cos(math.radians(target_yaw_deg)), math.sin(math.radians(target_yaw_deg))]
            line_projection_m = sum(
                (start_tcp[axis] - entry_xy[axis]) * line_direction[axis]
                for axis in range(2)
            )
            preferred_parameter_m = min(
                max(line_projection_m, -args.line_max_rear_distance_m),
                -args.line_entry_clearance_m,
            )
            line_parameters = [preferred_parameter_m]
            for index in range(args.line_scan_count):
                rear_distance_m = args.line_max_rear_distance_m - (
                    (args.line_max_rear_distance_m - args.line_entry_clearance_m)
                    * index / (args.line_scan_count - 1)
                )
                parameter_m = -rear_distance_m
                if all(abs(parameter_m - known) > 1e-9 for known in line_parameters):
                    line_parameters.append(parameter_m)
            q1_alignment_xy: list[float] | None = None
            q1_alignment_target: list[float] | None = None
            q1_alignment_trajectory: list | None = None
            q1_alignment_frequency: float | None = None
            scan_failures: list[str] = []
            print(
                f"阶段 1：Q1 开爪，预规划抓取线后方安全窗口 "
                f"[{args.line_entry_clearance_m * 1000.0:.1f}, {args.line_max_rear_distance_m * 1000.0:.1f}] mm；"
                "每个候选均同时完成 TCP XY 后移和最终 yaw 对齐；z/roll/pitch 固定为参考抓取姿态。"
            )
            for scan_index, parameter_m in enumerate(line_parameters, start=1):
                candidate_xy = [entry_xy[axis] + parameter_m * line_direction[axis] for axis in range(2)]
                candidate_target = candidate_xy + [q1_z] + rpy_to_rotvec(
                    [q1_roll, q1_pitch, math.radians(target_yaw_deg)]
                )
                print(
                    f"阶段 1：Q1 预规划候选 {scan_index}/{len(line_parameters)}："
                    f"后方 {parameter_m * 1000.0:.1f} mm，最终 yaw={target_yaw_deg:.2f}°...",
                    end="", flush=True,
                )
                try:
                    trajectory, frequency = preplan_tcp_trajectory(
                        planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target=candidate_target, planner_kind=args.planner,
                        max_vel=args.max_vel, max_acc=args.max_acc,
                    )
                except TrajectoryPlanningError as exc:
                    scan_failures.append(f"{parameter_m * 1000.0:.1f} mm: {exc}")
                    print(f"失败（{exc}）。")
                    continue
                q1_alignment_xy = candidate_xy
                q1_alignment_target = candidate_target
                q1_alignment_trajectory = trajectory
                q1_alignment_frequency = frequency
                print(f"可达，已保存 {len(trajectory)} 个关节路点。")
                break
            if (
                q1_alignment_xy is None or q1_alignment_target is None
                or q1_alignment_trajectory is None or q1_alignment_frequency is None
            ):
                raise Q1DirectPathUnavailableError(
                    "阶段 1：Q1 抓取线后方安全窗口内没有可同时完成后移和最终 yaw 对齐的轨迹；"
                    + "；".join(scan_failures)
                )
            print(
                "阶段 1/2：Q1 执行已保存的后移+最终 yaw 对齐轨迹；"
                f"终点 xy(m)={q1_alignment_xy}，姿态使用参考抓取姿态。"
            )
            try:
                arm.follow_trajectory(
                    arm_id=args.arm_id, trajectory=q1_alignment_trajectory,
                    sample_frequency=q1_alignment_frequency,
                )
            except Exception as exc:
                raise RuntimeError(f"Q1 已保存的后移+最终 yaw 对齐轨迹执行失败（不会改为分段重规划）: {exc}") from exc
            assert q1_reference_transfer_base is not None
            recover_q1_transfer_base_z_pitch_with_planner(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                reference_transfer_base=q1_reference_transfer_base, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, start_sync_retries=args.start_sync_retries,
                z_tolerance_m=args.q1_z_tolerance_m,
                pitch_tolerance_rad=args.q1_pitch_tolerance_rad,
                correction_attempts=args.stage2_correction_attempts,
                stage="Q1 后移+最终 yaw 对齐后：TCP z/pitch 校验",
            )
        else:
            assert precontact_xy is not None
            # Project the current TCP to the intended line, then clamp the
            # projection to a safe rear window.  This prevents a far-away
            # projection from pulling the transfer-base into its inner
            # unreachable/collision region, while keeping the initial move a
            # single direct LIN plan on the blue-line extension.
            line_direction = [math.cos(math.radians(target_yaw_deg)), math.sin(math.radians(target_yaw_deg))]
            line_projection_m = sum(
                (start_tcp[axis] - entry_xy[axis]) * line_direction[axis]
                for axis in range(2)
            )
            line_parameter_m = min(
                max(line_projection_m, -args.line_max_rear_distance_m),
                -args.line_entry_clearance_m,
            )
            print(
                f"阶段 1：固定 yaw={start_yaw_deg:.2f}°，扫描蓝线后方安全窗口 "
                f"[{args.line_entry_clearance_m * 1000.0:.1f}, "
                f"{args.line_max_rear_distance_m * 1000.0:.1f}] mm；"
                f"投影={line_projection_m * 1000.0:.1f} mm，首选={line_parameter_m * 1000.0:.1f} mm。"
            )
            candidate_parameters = [line_parameter_m]
            for index in range(args.line_scan_count):
                rear_distance_m = args.line_max_rear_distance_m - (
                    (args.line_max_rear_distance_m - args.line_entry_clearance_m)
                    * index / (args.line_scan_count - 1)
                )
                candidate_parameter_m = -rear_distance_m
                if all(abs(candidate_parameter_m - known) > 1e-9 for known in candidate_parameters):
                    candidate_parameters.append(candidate_parameter_m)
            line_alignment_xy: list[float] | None = None
            line_alignment_parameter_m: float | None = None
            scan_failures: list[str] = []
            for scan_index, candidate_parameter_m in enumerate(candidate_parameters, start=1):
                candidate_xy = [
                    entry_xy[axis] + candidate_parameter_m * line_direction[axis]
                    for axis in range(2)
                ]
                candidate_target = candidate_xy + [start_tcp[2]] + start_tcp[3:]
                feasible, detail = can_plan_target(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target=candidate_target, target_is_tcp=True, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc,
                )
                if feasible:
                    line_alignment_xy = candidate_xy
                    line_alignment_parameter_m = candidate_parameter_m
                    print(
                        f"阶段 1：蓝线候选 {scan_index}/{len(candidate_parameters)} 可规划："
                        f"后方 {candidate_parameter_m * 1000.0:.1f} mm，xy(m)={candidate_xy}（{detail}）。"
                    )
                    break
                failure = f"{candidate_parameter_m * 1000.0:.1f} mm: {detail}"
                scan_failures.append(failure)
                print(f"阶段 1：蓝线候选 {scan_index}/{len(candidate_parameters)} 不可规划（{failure}）。")
            if line_alignment_xy is None or line_alignment_parameter_m is None:
                raise RuntimeError(
                    "阶段 1：蓝线安全窗口内没有可执行的 LIN 轨迹；"
                    + "；".join(scan_failures)
                )
            move_target = line_alignment_xy + [start_tcp[2]] + start_tcp[3:]
            plan_and_follow(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target=move_target, target_is_tcp=True, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                stage="阶段 1/3：TCP 移动到蓝线后方延长线",
            )

            current_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
            current_tcp = compose_pose(current_ik, tcp_offset)
            fixed_rpy = rotvec_to_rpy(current_tcp[3:])
            current_yaw_deg = math.degrees(fixed_rpy[2])
            yaw_delta_deg = (target_yaw_deg - current_yaw_deg + 180.0) % 360.0 - 180.0
            yaw_steps = max(1, math.ceil(abs(yaw_delta_deg) / args.yaw_pivot_step_deg))
            assert push_reference_tcp_z is not None and push_reference_tcp_rpy is not None
            stage2_z = push_reference_tcp_z
            stage2_roll, stage2_pitch = push_reference_tcp_rpy[:2]
            print(
                f"阶段 2：固定蓝线 xy(m)={line_alignment_xy}，转 yaw "
                f"{current_yaw_deg:.2f}° → {target_yaw_deg:.2f}°，"
                f"强制 TCP z={stage2_z:.6f} m、roll/pitch="
                f"{math.degrees(stage2_roll):.3f}°/{math.degrees(stage2_pitch):.3f}°；分为 {yaw_steps} 段。"
            )
            for step in range(1, yaw_steps + 1):
                step_yaw_rad = math.radians(current_yaw_deg + yaw_delta_deg * step / yaw_steps)
                rotate_target = line_alignment_xy + [stage2_z] + rpy_to_rotvec(
                    [stage2_roll, stage2_pitch, step_yaw_rad]
                )
                feasible, detail = can_plan_target(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target=rotate_target, target_is_tcp=True, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc,
                )
                if not feasible:
                    print(
                        f"阶段 2：当前蓝线点无法继续转 yaw（{detail}）；"
                        "沿蓝线重定位后重试，且不会越过紫点。"
                    )
                    # Try points on both sides of the current line point, but
                    # stay on the rear extension at least the configured
                    # clearance behind the purple point.
                    recovery_parameters = [
                        -args.line_yaw_recovery_rear_distance_m
                        + (args.line_yaw_recovery_rear_distance_m - args.line_entry_clearance_m)
                        * index / (args.line_scan_count - 1)
                        for index in range(args.line_scan_count)
                    ]
                    recovery_parameters.sort(key=lambda value: abs(value - line_alignment_parameter_m))
                    recovered = False
                    recovery_failures: list[str] = []
                    previous_yaw_rad = math.radians(
                        current_yaw_deg + yaw_delta_deg * (step - 1) / yaw_steps
                    )
                    previous_orientation = rpy_to_rotvec([stage2_roll, stage2_pitch, previous_yaw_rad])
                    for recovery_index, parameter_m in enumerate(recovery_parameters, start=1):
                        if abs(parameter_m - line_alignment_parameter_m) <= 1e-9:
                            continue
                        candidate_xy = [
                            entry_xy[axis] + parameter_m * line_direction[axis]
                            for axis in range(2)
                        ]
                        move_target = candidate_xy + [stage2_z] + previous_orientation
                        move_ok, move_detail = can_plan_target(
                            planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                            tcp_offset=tcp_offset, target=move_target, target_is_tcp=True, planner_kind=args.planner,
                            max_vel=args.max_vel, max_acc=args.max_acc,
                        )
                        yaw_target = candidate_xy + [stage2_z] + rpy_to_rotvec(
                            [stage2_roll, stage2_pitch, step_yaw_rad]
                        )
                        yaw_ok, yaw_detail = can_plan_target(
                            planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                            tcp_offset=tcp_offset, target=yaw_target, target_is_tcp=True, planner_kind=args.planner,
                            max_vel=args.max_vel, max_acc=args.max_acc,
                        )
                        if move_ok and yaw_ok:
                            line_alignment_xy, line_alignment_parameter_m = candidate_xy, parameter_m
                            print(
                                f"阶段 2：蓝线恢复候选 {recovery_index}/{len(recovery_parameters)} 可规划："
                                f"后方 {parameter_m * 1000.0:.1f} mm，xy(m)={candidate_xy}。"
                            )
                            plan_and_follow(
                                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                                tcp_offset=tcp_offset, target=move_target, target_is_tcp=True, planner_kind=args.planner,
                                max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                                stage="阶段 2：沿蓝线重定位以继续转 yaw",
                            )
                            rotate_target = yaw_target
                            recovered = True
                            break
                        recovery_failures.append(
                            f"{parameter_m * 1000.0:.1f} mm: 移动={move_detail}，转 yaw={yaw_detail}"
                        )
                    if not recovered:
                        fallback_ik = pose_from_lookup_transform(
                            planner.lookup_transform(args.frame_id, ik_frame).get("data")
                        )
                        fallback_tcp = compose_pose(fallback_ik, tcp_offset)
                        target_yaw_deg = normalize_angle_deg(math.degrees(rotvec_to_rpy(fallback_tcp[3:])[2]))
                        fallback_yaw_rad = math.radians(target_yaw_deg)
                        assert push_end_xy is not None
                        push_end_xy = [
                            entry_xy[axis] + args.push_distance_m * (
                                math.cos(fallback_yaw_rad) if axis == 0 else math.sin(fallback_yaw_rad)
                            )
                            for axis in range(2)
                        ]
                        print(
                            f"警告：阶段 2 无法继续转到原目标 yaw；保留当前实际 yaw={target_yaw_deg:.2f}°，"
                            "阶段 3 将按该 yaw 朝紫点方向推送。"
                        )
                        break
                try:
                    plan_and_follow(
                        arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target=rotate_target, target_is_tcp=True, planner_kind=args.planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                        stage=f"阶段 2/3：蓝线原地转 yaw（{step}/{yaw_steps}）",
                    )
                    verify_and_recover_stage2_tcp(
                        arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target=rotate_target, planner_kind=args.planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, start_sync_retries=args.start_sync_retries,
                        z_tolerance_m=args.push_z_tolerance_m,
                        pitch_tolerance_rad=args.stage2_pitch_tolerance_rad,
                        correction_attempts=args.stage2_correction_attempts,
                        stage=f"阶段 2/3：蓝线转 yaw（{step}/{yaw_steps}）",
                    )
                except TrajectoryPlanningError as exc:
                    # See the equivalent Q1 case above: after a recovery
                    # translation, re-evaluate from the actual pose and push
                    # with the yaw that has really been reached.
                    fallback_ik = pose_from_lookup_transform(
                        planner.lookup_transform(args.frame_id, ik_frame).get("data")
                    )
                    fallback_tcp = compose_pose(fallback_ik, tcp_offset)
                    target_yaw_deg = normalize_angle_deg(math.degrees(rotvec_to_rpy(fallback_tcp[3:])[2]))
                    fallback_yaw_rad = math.radians(target_yaw_deg)
                    assert push_end_xy is not None
                    push_end_xy = [
                        entry_xy[axis] + args.push_distance_m * (
                            math.cos(fallback_yaw_rad) if axis == 0 else math.sin(fallback_yaw_rad)
                        )
                        for axis in range(2)
                    ]
                    print(
                        f"警告：阶段 2 的实际 yaw 重规划仍失败（{exc}）；"
                        f"保留当前实际 yaw={target_yaw_deg:.2f}°，阶段 3 将按该 yaw 朝紫点方向推送。"
                    )
                    break

        current_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
        current_tcp = compose_pose(current_ik, tcp_offset)
        fixed_rpy = rotvec_to_rpy(current_tcp[3:])
        if args.test:
            push_roll_pitch = fixed_rpy[:2]
        elif is_grasp:
            assert q1_reference_roll is not None and q1_reference_pitch is not None
            push_roll_pitch = [q1_reference_roll, q1_reference_pitch]
        else:
            assert push_reference_tcp_rpy is not None
            push_roll_pitch = push_reference_tcp_rpy[:2]
        target_orientation = rpy_to_rotvec([push_roll_pitch[0], push_roll_pitch[1], math.radians(target_yaw_deg)])
        push_z = q1_reference_z if is_grasp else current_tcp[2]
        if is_grasp:
            assert q1_reference_z is not None and q1_reference_pitch is not None and q1_reference_transfer_base is not None
            reference_base_pitch = rotvec_to_rpy(q1_reference_transfer_base[3:])[1]
            current_base_pitch = rotvec_to_rpy(current_ik[3:])[1]
            q1_z_error_m = current_ik[2] - q1_reference_transfer_base[2]
            q1_pitch_error_rad = current_base_pitch - reference_base_pitch
            if abs(q1_z_error_m) > args.q1_z_tolerance_m or abs(q1_pitch_error_rad) > args.q1_pitch_tolerance_rad:
                print(
                    "Q1 阶段 2 后：transfer-base 相对实时起点偏离，"
                    f"z 偏差={q1_z_error_m * 1000.0:.3f} mm，"
                    f"pitch 偏差={math.degrees(q1_pitch_error_rad):.3f}°；"
                    "固定当前 transfer-base XY/roll/yaw，先恢复实时起点 z/pitch 后才抓取。"
                )
                current_ik = recover_q1_transfer_base_z_pitch_with_planner(
                    arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    reference_transfer_base=q1_reference_transfer_base, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc, start_sync_retries=args.start_sync_retries,
                    z_tolerance_m=args.q1_z_tolerance_m,
                    pitch_tolerance_rad=args.q1_pitch_tolerance_rad,
                    correction_attempts=args.stage2_correction_attempts,
                    stage="Q1 阶段 3 前：transfer-base 恢复实时起点 z/pitch",
                )
                current_tcp = compose_pose(current_ik, tcp_offset)
                fixed_rpy = rotvec_to_rpy(current_tcp[3:])
                remaining_base_pitch = rotvec_to_rpy(current_ik[3:])[1]
                remaining_z_error_m = current_ik[2] - q1_reference_transfer_base[2]
                remaining_pitch_error_rad = remaining_base_pitch - reference_base_pitch
                if abs(remaining_z_error_m) > args.q1_z_tolerance_m or abs(remaining_pitch_error_rad) > args.q1_pitch_tolerance_rad:
                    print(
                        "警告：Q1 阶段 2 后一次 transfer-base 实时起点恢复仍未达到严阈值；"
                        f"剩余 z 偏差={remaining_z_error_m * 1000.0:.3f} mm，"
                        f"pitch 偏差={math.degrees(remaining_pitch_error_rad):.3f}°；按当前状态继续抓取。"
                    )
            else:
                print(
                    "Q1 阶段 2 后：transfer-base z/pitch 已保持实时起点，无需修正；"
                    f"z 偏差={q1_z_error_m * 1000.0:.3f} mm，"
                    f"pitch 偏差={math.degrees(q1_pitch_error_rad):.3f}°。"
                )
            # Stage 3 must use the pre-Q1-grasp z and pitch, never a height
            # or pitch accidentally reached while rotating in stage 2. Roll
            # deliberately remains the current value.
            push_z = q1_reference_z
        if not args.test and not is_grasp:
            assert push_reference_tcp_z is not None and precontact_xy is not None
            z_error_m = current_tcp[2] - push_reference_tcp_z
            print(
                f"阶段 3 前 TCP z 检查：当前={current_tcp[2]:.6f} m，"
                f"基准={push_reference_tcp_z:.6f} m，偏差={z_error_m * 1000.0:.3f} mm。"
            )
            if abs(z_error_m) > args.push_z_tolerance_m:
                print(
                    f"阶段 3 前：TCP z 偏差超过 {args.push_z_tolerance_m * 1000.0:.3f} mm，"
                    f"最多闭环修正 {args.push_z_correction_attempts} 次；保持当前 xy/yaw/roll/pitch。"
                )
                for correction_attempt in range(1, args.push_z_correction_attempts + 1):
                    z_recovery_target = current_tcp[:2] + [push_reference_tcp_z] + target_orientation
                    plan_and_follow(
                        arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target=z_recovery_target, target_is_tcp=True, planner_kind=args.planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                        stage=f"阶段 3 前：恢复 TCP 初始 z 高度（{correction_attempt}/{args.push_z_correction_attempts}）",
                    )
                    # Planner TF updates asynchronously after trajectory follow.
                    time.sleep(0.2)
                    recovered_ik = pose_from_lookup_transform(
                        planner.lookup_transform(args.frame_id, ik_frame).get("data")
                    )
                    current_tcp = compose_pose(recovered_ik, tcp_offset)
                    recovered_error_m = current_tcp[2] - push_reference_tcp_z
                    print(
                        f"阶段 3 前 TCP z 复查（{correction_attempt}/{args.push_z_correction_attempts}）："
                        f"当前={current_tcp[2]:.6f} m，偏差={recovered_error_m * 1000.0:.3f} mm。"
                    )
                    if abs(recovered_error_m) <= args.push_z_tolerance_m:
                        break
                final_error_m = current_tcp[2] - push_reference_tcp_z
                if abs(final_error_m) > args.push_z_tolerance_m:
                    print(
                        f"警告：TCP z 经修正后偏差仍为 {final_error_m * 1000.0:.3f} mm；"
                        "按当前实际高度继续阶段 3 推送。"
                    )
            # Do not combine an unachieved z correction with the contact push.
            # Stage 3 keeps the actual height reached after the correction loop.
            push_z = current_tcp[2]
        if args.test:
            center_target = center_xy + [current_tcp[2]] + target_orientation
            plan_and_follow(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target=center_target, target_is_tcp=True, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                stage="阶段 3/3：沿红箭头线移动 fingertip TCP 到工件中心（绿点）",
            )
        elif is_grasp:
            yaw_rad = math.radians(target_yaw_deg)
            q1_direction = [math.cos(yaw_rad), math.sin(yaw_rad)]
            grasp_advance_xy = [
                entry_xy[0] + args.q1_grasp_advance_m * q1_direction[0],
                entry_xy[1] + args.q1_grasp_advance_m * q1_direction[1],
            ]
            grasp_advance_target = grasp_advance_xy + [push_z] + target_orientation
            plan_and_follow(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target=grasp_advance_target, target_is_tcp=True, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                stage=(
                    "阶段 2/2：沿 Q1 yaw 经过 tail_inside 并额外推进 "
                    f"{args.q1_grasp_advance_m * 1000.0:.1f} mm"
                ),
            )
            assert q1_reference_transfer_base is not None
            recover_q1_transfer_base_z_pitch_with_planner(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                reference_transfer_base=q1_reference_transfer_base, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, start_sync_retries=args.start_sync_retries,
                z_tolerance_m=args.q1_z_tolerance_m,
                pitch_tolerance_rad=args.q1_pitch_tolerance_rad,
                correction_attempts=args.stage2_correction_attempts,
                stage="Q1 最终抓取点：TCP z/pitch 校验",
            )
        else:
            # The TCP yaw is the required push-line direction in body_base.
            # Stages 1 and 2 have already reached and aligned at the reverse
            # extension, so this final LIN segment is the only contact motion.
            assert precontact_xy is not None and push_end_xy is not None
            prepush_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
            prepush_tcp = compose_pose(prepush_ik, tcp_offset)
            push_target = push_end_xy + [push_z] + target_orientation
            plan_and_follow(
                arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target=push_target, target_is_tcp=True, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                stage="阶段 3/3：沿推送线经过切入点并继续推送",
            )
        if is_grasp and not args.test:
            narrow_grasp_attempt = 0
            while True:
                try:
                    close_gripper_for_q1_grasp(
                        gripper, arm_id=args.arm_id, timeout_s=args.gripper_timeout_s,
                        fully_closed_max_position=args.q1_fully_closed_max_position,
                        valid_grasp_min_position=args.q1_valid_grasp_min_position,
                        empty_grasp_max_abs_force=args.q1_empty_grasp_max_abs_force,
                    )
                    break
                except NarrowQ1GraspError:
                    if narrow_grasp_attempt >= args.q1_narrow_grasp_retries:
                        raise
                    narrow_grasp_attempt += 1
                    print(
                        "Q1 窄抓恢复：先松开夹爪，保持 XY/姿态不变，仅将 TCP 沿 +Z 抬高 "
                        f"{args.q1_narrow_grasp_lift_m * 1000.0:.1f} mm 后原地复抓 "
                        f"（{narrow_grasp_attempt}/{args.q1_narrow_grasp_retries}）。"
                    )
                    set_gripper_state(
                        gripper, arm_id=args.arm_id, target_open=True,
                        timeout_s=args.gripper_timeout_s,
                        tolerance=args.gripper_position_tolerance,
                    )
                    narrow_grasp_ik = pose_from_lookup_transform(
                        planner.lookup_transform(args.frame_id, ik_frame).get("data")
                    )
                    narrow_grasp_tcp = compose_pose(narrow_grasp_ik, tcp_offset)
                    narrow_grasp_lift_target = [
                        narrow_grasp_tcp[0], narrow_grasp_tcp[1],
                        narrow_grasp_tcp[2] + args.q1_narrow_grasp_lift_m,
                        *narrow_grasp_tcp[3:],
                    ]
                    plan_and_follow(
                        arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target=narrow_grasp_lift_target, target_is_tcp=True,
                        planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc,
                        retries=args.start_sync_retries,
                        stage=(
                            "Q1 窄抓恢复：松爪后 TCP 原地 +Z 上抬 "
                            f"{args.q1_narrow_grasp_lift_m * 1000.0:.1f} mm"
                        ),
                    )
            # The part is now confirmed in the gripper.  Do not change XY or
            # orientation: lift strictly in body_base +Z.  Prefer 50 mm, then
            # fall back to 40/30/20/10 mm only when the larger LIN motion is not
            # plannable from the actual post-grasp pose.
            post_grasp_ik = pose_from_lookup_transform(
                planner.lookup_transform(args.frame_id, ik_frame).get("data")
            )
            post_grasp_tcp = compose_pose(post_grasp_ik, tcp_offset)
            lift_failures: list[str] = []
            for lift_m in args.q1_post_grasp_lift_m:
                lift_target = [
                    post_grasp_tcp[0], post_grasp_tcp[1], post_grasp_tcp[2] + lift_m,
                    *post_grasp_tcp[3:],
                ]
                feasible, detail = can_plan_target(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target=lift_target, target_is_tcp=True, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc,
                )
                if not feasible:
                    lift_failures.append(f"{lift_m * 1000.0:.0f} mm: {detail}")
                    print(f"Q1 抓后仅 +Z 上抬 {lift_m * 1000.0:.0f} mm 不可规划，尝试更小高度。")
                    continue
                print(f"Q1 抓取确认，保持 XY/姿态不变，仅沿 body_base +Z 上抬 {lift_m * 1000.0:.0f} mm。")
                plan_and_follow(
                    arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target=lift_target, target_is_tcp=True, planner_kind=args.planner,
                    max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries,
                    stage=f"Q1 抓后仅 +Z 上抬 {lift_m * 1000.0:.0f} mm",
                )
                break
            else:
                raise RuntimeError(
                    "Q1 已夹住工件，但 50/40/30/20/10 mm 的仅 +Z 抬升均不可规划；"
                    "未移动机械臂。原因：" + "；".join(lift_failures)
                )
        print("动作完成。")
    except Q1DirectPathUnavailableError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        print(
            f"Q1 直接抓取路径不可规划；以退出码 {Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE} 请求管线先调整工件位姿。",
            file=sys.stderr,
        )
        return Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        if planner is not None:
            planner.close()
        if gripper is not None:
            gripper.close()
        if arm is not None:
            arm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
