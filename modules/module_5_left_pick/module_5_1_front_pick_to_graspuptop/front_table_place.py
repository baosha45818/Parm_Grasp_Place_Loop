"""旧 Module 3 FRONT 分支：把左夹爪中的正面工件放到桌面中间。"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from loop_config import LoopConfig
from loop_types import ObjectPose
from planner_runtime import PlannerRuntime
from placement_config import PlacementParameters, load_workpiece_parameters
from placement_parameters import (
    DISCARD_BELOW_GRASPUP_M,
    GRASPUP_ABOVE_JOINTS_DEG,
    GRASPUP_ABOVE_PLANNING_TIMEOUT_S,
    GRASPUP_JOINTS_DEG,
)
# 旧总控传给 Module 3 的正面重抓撤离中间位。
FRONT_REGRASP_MIDDLE_JOINTS_DEG = [
    14.281, -29.886, -85.633, 102.504, 37.378, 56.016, 9.241,
]
OFFLINE_START_TOLERANCE_DEG = 1.0
PLANNING_INFLIGHT = 8


def _pose(matrix: np.ndarray) -> ObjectPose:
    return ObjectPose(
        frame_id="body_base",
        position=np.asarray(matrix[:3, 3], dtype=float).tolist(),
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


def _recover_after_planning_timeout(
    *,
    runtime: PlannerRuntime,
    arm_host: str,
    open_gripper: Callable[[str], None],
) -> dict:
    """复现旧 Module 3：回 graspup、下降5cm丢弃、松爪并回 graspup。"""
    runtime.set_next_plan_context("left_graspup_top", "left_graspup")
    runtime.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
    _wait_for_left_joints(arm_host, GRASPUP_JOINTS_DEG, label="规划超时回graspup")

    T_body_transfer = _matrix_from_tf(
        runtime.lookup_transform("body_base", "left_hand_transfer_end"),
        "graspup left_hand_transfer_end",
    )
    discard = T_body_transfer.copy()
    discard[2, 3] -= DISCARD_BELOW_GRASPUP_M
    runtime.set_next_plan_context("left_graspup", "left_timeout_discard_position")
    runtime.move_cartesian_pose("left", _pose(discard), planner="lin")
    open_gripper("桌中放置规划超时丢弃位")

    runtime.set_next_plan_context("left_timeout_discard_position", "left_graspup")
    runtime.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
    _wait_for_left_joints(arm_host, GRASPUP_JOINTS_DEG, label="规划超时恢复graspup")
    return {
        "outcome": "planning_timeout_recovered_at_graspup",
        "planning_timeout_s": GRASPUP_ABOVE_PLANNING_TIMEOUT_S,
        "discard_below_graspup_m": DISCARD_BELOW_GRASPUP_M,
        "released": True,
        "released_at_verified_target": False,
        "returned_to_graspup": True,
    }


def _read_left_joints_deg(arm_host: str) -> np.ndarray:
    from parm_sdk import Arm

    with Arm(host=arm_host) as arm:
        state = arm.get_state() or {}
    joints = np.asarray(
        ((state.get("state") or {}).get("left") or {}).get("position") or [],
        dtype=float,
    )
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"无法读取有效左臂关节角: {state}")
    return joints


def _wait_for_left_joints(
    arm_host: str,
    target_deg: list[float],
    *,
    label: str,
    tolerance_deg: float = 0.5,
    timeout_s: float = 90.0,
) -> None:
    target = np.asarray(target_deg, dtype=float)
    deadline = time.monotonic() + timeout_s
    last_error = float("inf")
    while time.monotonic() < deadline:
        actual = _read_left_joints_deg(arm_host)
        delta = (actual - target + 180.0) % 360.0 - 180.0
        last_error = float(np.max(np.abs(delta)))
        if last_error <= tolerance_deg:
            print(f"[{label}] 到位，最大关节误差={last_error:.3f}deg", flush=True)
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"左臂未在 {timeout_s:.1f}s 内到达{label}；最大误差={last_error:.3f}deg"
    )


def _object_target(
    profile: PlacementParameters,
    *,
    z_m: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """复现旧配置的 local_rpy FRONT 工件目标。"""
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler(
        "XYZ", [roll_deg, pitch_deg, yaw_deg], degrees=True
    ).as_matrix()
    target[:3, 3] = [
        profile.FRONT_TARGET_OBJECT_XYZ_M[0],
        profile.FRONT_TARGET_OBJECT_XYZ_M[1],
        z_m,
    ]
    return target


def build_front_candidates(
    workpiece: str,
    T_transfer_object: np.ndarray,
    *,
    parameters: PlacementParameters | None = None,
) -> list[dict]:
    """按旧 Module 3 的 roll/pitch/yaw 优先级生成 FRONT 候选。"""
    profile = parameters or load_workpiece_parameters(str(workpiece))
    relation = np.asarray(T_transfer_object, dtype=float)
    if relation.shape != (4, 4) or not np.all(np.isfinite(relation)):
        raise ValueError("T_transfer_object 必须是有效4x4矩阵")
    object_to_transfer = np.linalg.inv(relation)
    raw = []
    for z_index, z_m in enumerate(profile.FRONT_TARGET_Z_CANDIDATES_M):
        for yaw_index, yaw_deg in enumerate(profile.FRONT_YAW_CANDIDATES_DEG):
            for roll_index, roll_deg in enumerate(profile.FRONT_ROLL_CANDIDATES_DEG):
                for pitch_index, pitch_deg in enumerate(profile.FRONT_PITCH_CANDIDATES_DEG):
                    raw.append((
                        roll_index + pitch_index,
                        roll_index,
                        pitch_index,
                        yaw_index,
                        z_index,
                        float(z_m),
                        float(roll_deg),
                        float(pitch_deg),
                        float(yaw_deg),
                    ))
    raw.sort(key=lambda item: item[:5])
    candidates = []
    for priority, item in enumerate(raw):
        _, roll_index, pitch_index, yaw_index, z_index, z_m, roll, pitch, yaw = item
        target_object = _object_target(
            profile, z_m=z_m, roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw
        )
        target_transfer = target_object @ object_to_transfer
        candidates.append({
            "priority_index": priority,
            "z_index": z_index,
            "yaw_index": yaw_index,
            "roll_index": roll_index,
            "pitch_index": pitch_index,
            "object_z_m": z_m,
            "yaw_deg": yaw,
            "roll_deg": roll,
            "pitch_deg": pitch,
            "target_object": target_object,
            "target_transfer": target_transfer,
        })
    return candidates


def place_front_workpiece_at_table_center(
    *,
    workpiece: str,
    T_transfer_object: np.ndarray,
    planner_host: str,
    arm_host: str,
    speed: float,
    require_holding: Callable[[str], None],
    open_gripper: Callable[[str], None],
    parameters: PlacementParameters | None = None,
) -> dict:
    """从正面重抓上提位撤离，经 graspup_top 放到旧桌中 FRONT 目标。"""
    profile = parameters or load_workpiece_parameters(str(workpiece))
    config = LoopConfig(
        planner_host=planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(speed),
        max_acceleration=float(speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    runtime = PlannerRuntime(config)
    planning_attempts: list[dict] = []
    selected: dict | None = None
    selected_plan: dict | None = None

    try:
        runtime.open(required_frames={"body_base", "left_hand_transfer_end"})

        require_holding("正面重抓后移动到桌中放置中间位")
        runtime.set_next_plan_context("front_regrasp_lift", "front_place_middle")
        runtime.move_left_joints(FRONT_REGRASP_MIDDLE_JOINTS_DEG, planner="rrtstar")
        _wait_for_left_joints(
            arm_host, FRONT_REGRASP_MIDDLE_JOINTS_DEG, label="正面放置中间位"
        )

        require_holding("从中间位移动到graspup_top")
        runtime.set_next_plan_context("front_place_middle", "left_graspup_top")
        runtime.move_left_joints(GRASPUP_ABOVE_JOINTS_DEG, planner="rrtstar")
        _wait_for_left_joints(arm_host, GRASPUP_ABOVE_JOINTS_DEG, label="graspup_top")

        candidates = build_front_candidates(
            workpiece,
            T_transfer_object,
            parameters=profile,
        )
        print(
            f"[FRONT桌中放置] 工件{workpiece}候选={len(candidates)}，"
            f"目标XY={list(profile.FRONT_TARGET_OBJECT_XYZ_M[:2])}",
            flush=True,
        )

        def plan_candidate(candidate: dict) -> dict:
            target = candidate["target_transfer"]
            if float(target[2, 2]) >= 0.0:
                raise RuntimeError(
                    "left_hand_transfer_end +Z在body_base Z上的投影必须为负"
                )
            planner = PlannerRuntime(config)
            try:
                planner.open(required_frames={"body_base", "left_hand_transfer_end"})
                result = planner.plan_cartesian_pose(
                    "left",
                    _pose(target),
                    planner="rrtstar",
                    start_joints_deg=GRASPUP_ABOVE_JOINTS_DEG,
                    rrtstar_goal_tolerance=True,
                    use_left_eef_collision_box=True,
                )
                if result is None:
                    raise RuntimeError("离线规划返回空结果")
                trajectory = (result.get("data") or {}).get("trajectory") or []
                if not trajectory:
                    raise RuntimeError("离线规划没有返回轨迹")
                return result
            finally:
                planner.close()

        next_index = 0
        in_flight = {}
        planning_timed_out = False
        executor = ThreadPoolExecutor(
            max_workers=PLANNING_INFLIGHT,
            thread_name_prefix="module5-front-table-place",
        )
        deadline = time.monotonic() + GRASPUP_ABOVE_PLANNING_TIMEOUT_S
        try:
            while selected is None and (in_flight or next_index < len(candidates)):
                while len(in_flight) < PLANNING_INFLIGHT and next_index < len(candidates):
                    candidate = candidates[next_index]
                    future = executor.submit(plan_candidate, candidate)
                    in_flight[future] = candidate
                    next_index += 1
                if not in_flight:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    planning_timed_out = True
                    break
                completed, _ = wait(
                    tuple(in_flight),
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    planning_timed_out = True
                    break
                future = next(iter(completed))
                candidate = in_flight.pop(future)
                record = {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"target_object", "target_transfer"}
                }
                try:
                    result = future.result()
                except Exception as exc:
                    record.update({"planned": False, "error": f"{type(exc).__name__}: {exc}"})
                    planning_attempts.append(record)
                    continue
                record.update({
                    "planned": True,
                    "target_object_pose": _pose(candidate["target_object"]).__dict__,
                    "target_tool_pose": _pose(candidate["target_transfer"]).__dict__,
                })
                planning_attempts.append(record)
                selected = candidate
                selected_plan = result
        finally:
            if planning_timed_out:
                for future in in_flight:
                    future.cancel()
                # 规划RPC各自使用独立连接；超时后不能等待仍在飞的请求，必须立即
                # 让主连接执行旧Module3的丢弃恢复。
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        if planning_timed_out and selected is None:
            print(
                "[规划超时] 到达graspup_top后"
                f"{GRASPUP_ABOVE_PLANNING_TIMEOUT_S:.1f}s仍无可行FRONT候选；"
                "执行旧Module3丢弃恢复",
                flush=True,
            )
            recovery = _recover_after_planning_timeout(
                runtime=runtime,
                arm_host=arm_host,
                open_gripper=open_gripper,
            )
            return {
                "place_module": "legacy-module-3-1-front",
                "placement_branch": "BRANCH_1_FRONT",
                "workpiece_id": str(workpiece),
                "planning_attempts": planning_attempts,
                **recovery,
            }

        if selected is None or selected_plan is None:
            raise RuntimeError("旧Module3 FRONT桌中放置候选全部规划失败")

        trajectory = (selected_plan.get("data") or {}).get("trajectory") or []
        offline_start = np.asarray(trajectory[0], dtype=float)
        actual_start = _read_left_joints_deg(arm_host)
        delta = (actual_start - offline_start + 180.0) % 360.0 - 180.0
        start_error = float(np.max(np.abs(delta)))
        planning_mode = "offline"
        if start_error > OFFLINE_START_TOLERANCE_DEG:
            runtime.set_next_plan_context("left_graspup_top", "front_table_place")
            selected_plan = runtime.plan_cartesian_pose(
                "left",
                _pose(selected["target_transfer"]),
                planner="rrtstar",
                rrtstar_goal_tolerance=True,
                use_left_eef_collision_box=True,
            )
            if selected_plan is None:
                raise RuntimeError("离线起点不匹配且在线重规划失败")
            planning_mode = "online_replan_after_start_mismatch"

        require_holding("从graspup_top移动到桌中FRONT放置位")
        runtime.execute_planned_result(selected_plan)

        selected_object = selected["target_object"]
        selected_transfer = selected["target_transfer"]
        object_to_transfer = np.linalg.inv(np.asarray(T_transfer_object, dtype=float))
        active_release_z = [
            float(value)
            for value in profile.FRONT_RELEASE_Z_CANDIDATES_M
            if float(value) <= float(selected_object[2, 3]) + 1e-9
        ]
        release_succeeded = not bool(profile.FRONT_RELEASE_Z_CANDIDATES_M)
        release_attempts = []
        descended = False
        for release_z in active_release_z:
            release_object = selected_object.copy()
            release_object[2, 3] = release_z
            release_transfer = release_object @ object_to_transfer
            record = {"object_z_m": release_z, "succeeded": False}
            try:
                runtime.set_next_plan_context("front_table_place", "front_table_release")
                runtime.move_cartesian_pose("left", _pose(release_transfer), planner="lin")
                record["succeeded"] = True
                release_succeeded = True
                descended = not np.allclose(release_transfer, selected_transfer)
                release_attempts.append(record)
                break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                release_attempts.append(record)

        open_gripper("FRONT桌中放置目标")

        if release_succeeded and descended:
            runtime.set_next_plan_context("front_table_release", "front_table_place")
            runtime.move_cartesian_pose("left", _pose(selected_transfer), planner="lin")

        runtime.config = replace(
            runtime.config,
            max_velocity=float(speed),
            max_acceleration=float(speed),
        )
        runtime.set_next_plan_context("front_table_place", "left_graspup_top")
        runtime.move_left_joints(
            GRASPUP_ABOVE_JOINTS_DEG,
            planner="rrtstar",
            use_left_eef_collision_box=True,
        )
        _wait_for_left_joints(
            arm_host, GRASPUP_ABOVE_JOINTS_DEG, label="graspup_top回撤"
        )
        runtime.set_next_plan_context("left_graspup_top", "left_graspup")
        runtime.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
        _wait_for_left_joints(arm_host, GRASPUP_JOINTS_DEG, label="graspup")

        if not release_succeeded:
            raise RuntimeError("未到达任何经规划验证的FRONT松爪高度")

        return {
            "place_module": "legacy-module-3-1-front",
            "placement_branch": "BRANCH_1_FRONT",
            "workpiece_id": str(workpiece),
            "planning_mode": planning_mode,
            "offline_start_max_error_deg": start_error,
            "selected_candidate": {
                key: value
                for key, value in selected.items()
                if key not in {"target_object", "target_transfer"}
            },
            "planning_attempts": planning_attempts,
            "release_attempts": release_attempts,
            "released": True,
            "returned_to_graspup": True,
        }
    finally:
        runtime.close()
