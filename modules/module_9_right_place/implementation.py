"""模块 9：沿示教的三维路径将工件孔 A 套入焊台 peg。"""

from __future__ import annotations

import math
import os
from time import time
from typing import TYPE_CHECKING

from loop_config import HoleAInsertionConfig
from loop_types import Holding, ObjectPose, ResultStatus, RightArmState, clear_right_object, make_result
from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from tool.planner_runtime import PlannerRuntime

if TYPE_CHECKING:
    from parm_sdk import ForceSensor


RIGHT_HAND = 2
OPEN_POSITION = 100.0
GRIPPER_TIMEOUT_SECONDS = 30.0
WAYPOINT_NAMES = (
    "right_hole_a_preinsert",
    "right_hole_a_ramp_contact",
    "right_hole_a_ramp_peak",
    "right_hole_a_inserted",
)


def module_9_right_place(
    right: RightArmState,
    planner: PlannerRuntime,
    *,
    release_after_success: bool | None = None,
    move_to_preinsert: bool = True,
):
    """Run the taught 3-D hole-A route and validate its measured Fz signature.

    The four positions intentionally vary in X/Y/Z and orientation.  Starting
    at the taught preinsert pose, each adjacent pair is interpolated into short
    Cartesian LIN moves.  A successful insertion requires: contact torque Ty
    rise, an Fz negative peak on the ramp, and Fz recovery to a low magnitude
    in the final (peg-in-hole) segment.
    """
    started = time()
    right.current_module = "9"
    if right.holding is not Holding.SINGLE or not right.grasp_stable:
        return _fault(right, started, "right hand must hold one stable object")

    try:
        poses, config = _require_insertion_setup(planner, release_after_success)
        if move_to_preinsert:
            right.motion_state = "MOVING_TO_HOLE_A_PREINSERT"
        if move_to_preinsert and config is not None and config.preinsert_joints_deg is not None:
            planner.move_right_joints(config.preinsert_joints_deg)
        elif move_to_preinsert:
            planner.move_cartesian("right", WAYPOINT_NAMES[0], planner="rrtstar")
        else:
            print("[9] 已在 right_hole_a_preinsert：跳过自动到位，仅执行孔 A 插入段")

        if not planner.config.live:
            print("[9][dry-run] 孔 A：按 preinsert → contact → peak → inserted 三维小步路径运行")
        else:
            assert config is not None
            right.motion_state = "INSERTING_HOLE_A"
            _insert_hole_a(right, planner, poses, config)

        should_release = (
            config.release_after_success if release_after_success is None and config else release_after_success
        )
        if should_release:
            right.motion_state = "HOLE_A_INSERTED_RELEASING"
            _open_right_gripper(planner)
            clear_right_object(right)
            right.motion_state = "HOLE_A_INSERTED"
            planner.move_cartesian("right", "right_idle", planner="rrtstar")
        else:
            right.motion_state = "HOLE_A_INSERTED_HELD"
    except Exception as exc:
        return _fault(right, started, str(exc))

    right.last_error = ""
    message = "hole A inserted onto peg" if should_release else "hole A inserted; gripper remains closed"
    return make_result(ResultStatus.SUCCESS, "9", started, message)


def _require_insertion_setup(
    planner: PlannerRuntime, release_after_success: bool | None
) -> tuple[list[ObjectPose], HoleAInsertionConfig | None]:
    raw_poses = [planner.config.poses.get(name) for name in WAYPOINT_NAMES]
    if not planner.config.live:
        return [pose for pose in raw_poses if pose is not None], planner.config.hole_a_insertion
    missing = [name for name, pose in zip(WAYPOINT_NAMES, raw_poses) if pose is None]
    config = planner.config.hole_a_insertion
    should_release = config.release_after_success if release_after_success is None and config else release_after_success
    if should_release and planner.config.poses.get("right_idle") is None:
        missing.append("right_idle")
    if missing:
        raise RuntimeError(f"live hole-A insertion requires configured poses: {missing}")
    if config is None:
        raise RuntimeError("live hole-A insertion requires hole_a_insertion force settings")
    poses = [pose for pose in raw_poses if pose is not None]
    if len(poses) != len(WAYPOINT_NAMES):
        raise AssertionError("missing taught waypoint")
    for name, first, second in zip(WAYPOINT_NAMES, poses, poses[1:]):
        if _distance(first.position, second.position) == 0.0:
            raise ValueError(f"hole-A waypoint {name} duplicates the following waypoint")
    return poses, config


def _insert_hole_a(
    right: RightArmState,
    planner: PlannerRuntime,
    poses: list[ObjectPose],
    config: HoleAInsertionConfig,
) -> None:
    from parm_sdk import ForceSensor

    # Planner is local, but the Xianxin force driver runs on the head/FT controller.
    with ForceSensor(
        host=os.environ.get("PARM_FORCE_SENSOR_HOST", "192.168.3.22"),
        hand=RIGHT_HAND,
    ) as sensor:
        baseline_samples = [
            _read_wrench(sensor, config.sample_timeout_seconds)
            for _ in range(config.baseline_samples)
        ]
        baseline_force = _mean([force for force, _ in baseline_samples])
        baseline_torque = _mean([torque for _, torque in baseline_samples])
        baseline_ty = baseline_torque[1]
        print(f"[9] 右腕力基线 force={baseline_force}, torque={baseline_torque}")

        saw_contact = False
        saw_fz_peak = False
        min_fz = float("inf")
        for stage_index, (start, end) in enumerate(zip(poses, poses[1:]), start=1):
            start_name, end_name = WAYPOINT_NAMES[stage_index - 1], WAYPOINT_NAMES[stage_index]
            count = math.ceil(_distance(start.position, end.position) / config.step_size_m)
            print(f"[9] 孔 A 阶段 {stage_index}/3: {start_name} -> {end_name}, {count} 个 LIN 小步")
            for step_index in range(1, count + 1):
                target = _interpolate_pose(start, end, step_index / count)
                planner.move_cartesian_pose("right", target, planner="lin")
                force, torque = _read_wrench(sensor, config.sample_timeout_seconds)
                delta_force = [value - base for value, base in zip(force, baseline_force)]
                delta_torque = [value - base for value, base in zip(torque, baseline_torque)]
                if _norm(delta_force) > config.max_force_delta_n:
                    raise RuntimeError(f"hole-A insertion force limit exceeded: {delta_force}")
                if _norm(delta_torque) > config.max_torque_delta_nm:
                    raise RuntimeError(f"hole-A insertion torque limit exceeded: {delta_torque}")

                fz, ty = force[2], torque[1]
                min_fz = min(min_fz, fz)
                saw_contact = saw_contact or ty - baseline_ty >= config.ty_contact_rise_nm
                saw_fz_peak = saw_fz_peak or fz <= config.fz_peak_max_n
                print(
                    f"[9] {end_name} 步 {step_index}/{count}: Fz={fz:.3f}N, Ty={ty:.3f}N*m, "
                    f"min_Fz={min_fz:.3f}N, dF={delta_force}, dT={delta_torque}"
                )

                if stage_index == 3 and (
                    saw_contact
                    and saw_fz_peak
                    and fz - min_fz >= config.fz_recovery_n
                    and abs(fz) <= config.fz_final_abs_max_n
                ):
                    right.motion_state = "HOLE_A_INSERTION_CONFIRMED"
                    return
    raise RuntimeError(
        "hole-A inserted waypoint reached without Fz recovery signature "
        f"(contact={saw_contact}, fz_peak={saw_fz_peak}, min_fz={min_fz:.3f})"
    )


def _read_wrench(sensor: "ForceSensor", timeout_seconds: float) -> tuple[list[float], list[float]]:
    frame = sensor.wait_first(timeout=timeout_seconds)
    try:
        force = [float(value) for value in frame["filtered_force"]]
        torque = [float(value) for value in frame["filtered_torque"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid right force-sensor frame: {frame!r}") from exc
    if len(force) != 3 or len(torque) != 3 or not all(math.isfinite(value) for value in force + torque):
        raise RuntimeError(f"invalid right force-sensor wrench: {frame!r}")
    return force, torque


def _interpolate_pose(start: ObjectPose, end: ObjectPose, fraction: float) -> ObjectPose:
    return ObjectPose(
        frame_id="body_base",
        position=[left + fraction * (right - left) for left, right in zip(start.position, end.position)],
        orientation=_slerp_xyzw(start.orientation, end.orientation, fraction),
    )


def _slerp_xyzw(first: list[float], second: list[float], fraction: float) -> list[float]:
    first = _normalise_quaternion(first)
    second = _normalise_quaternion(second)
    dot = sum(left * right for left, right in zip(first, second))
    if dot < 0.0:
        second, dot = [-value for value in second], -dot
    if dot > 0.9995:
        return _normalise_quaternion(
            [left + fraction * (right - left) for left, right in zip(first, second)]
        )
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sine = math.sin(theta)
    first_weight = math.sin((1.0 - fraction) * theta) / sine
    second_weight = math.sin(fraction * theta) / sine
    return [first_weight * left + second_weight * right for left, right in zip(first, second)]


def _normalise_quaternion(values: list[float]) -> list[float]:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("hole-A waypoint orientation must be a finite xyzw quaternion")
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        raise ValueError("hole-A waypoint orientation must not be zero")
    return [value / magnitude for value in values]


def _open_right_gripper(planner: PlannerRuntime) -> None:
    if not planner.config.live:
        print("[9][dry-run] 孔 A 插入成功后打开右夹爪")
        return
    from parm_sdk import Gripper

    with Gripper(host=planner.config.planner_host) as gripper:
        hand_state = gripper.get_state().get(RIGHT_HAND)
        if not is_gripper_usable(hand_state):
            raise RuntimeError("right gripper is unavailable or not initialized")
        response = gripper.move(RIGHT_HAND, OPEN_POSITION, mode="sync", timeout=GRIPPER_TIMEOUT_SECONDS)
        if response.get("status") != "success":
            raise RuntimeError(f"right gripper open was not completed: {response}")


def _fault(right: RightArmState, started: float, message: str):
    right.motion_state = "HOLE_A_INSERTION_FAILED"
    right.last_error = message
    return make_result(ResultStatus.FAULT, "9", started, message)


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _mean(rows: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(3)]


def _dot(first: list[float], second: list[float]) -> float:
    return sum(left * right for left, right in zip(first, second))


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))
