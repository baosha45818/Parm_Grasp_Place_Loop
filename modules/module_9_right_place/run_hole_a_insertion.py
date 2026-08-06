#!/usr/bin/env python3
"""Run module 9's standalone hole-A insertion commissioning flow."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_config import load_config
from loop_types import Holding, RightArmState
from modules.module_9_right_place.implementation import module_9_right_place
from tool.planner_runtime import PlannerRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="仅运行右臂孔 A 四路点插入，不启动完整双臂 Loop")
    parser.add_argument("--config", type=Path, required=True, help="包含四个孔 A 路点的配置 JSON")
    parser.add_argument("--live", action="store_true", help="连接硬件；省略时只 dry-run")
    parser.add_argument(
        "--direct-cartesian-servo", action="store_true",
        help="使用 zhuyiming 已验证的 SDK start/push/stop_servo_pose 笛卡尔链，绕过 Planner 轨迹执行",
    )
    parser.add_argument(
        "--release-after-success",
        action="store_true",
        help="确认插入后才打开右夹爪并撤到 right_idle；首次联调不要加此选项",
    )
    parser.add_argument(
        "--already-at-preinsert",
        action="store_true",
        help="右臂已通过人工示教到 right_hole_a_preinsert；跳过当前到位轨迹，仅执行插入段",
    )
    parser.add_argument(
        "--return-to-preinsert-only",
        action="store_true",
        help="用碰撞规划轨迹将右臂移到 preinsert 后退出，不启动插入和 servo-pose",
    )
    parser.add_argument(
        "--driver-z-offset-m",
        type=float,
        default=0.0,
        help="仅 direct-cartesian-servo：给全部示教路点叠加的数值 Z 偏移（米，限 ±3 mm）",
    )
    args = parser.parse_args()

    config = load_config(args.config, live=args.live)
    if args.driver_z_offset_m and not args.direct_cartesian_servo:
        parser.error("--driver-z-offset-m requires --direct-cartesian-servo")
    if args.return_to_preinsert_only:
        return _return_to_preinsert_only(config, live=args.live)
    if args.direct_cartesian_servo:
        return _run_direct_cartesian_servo(config, args)
    planner = PlannerRuntime(config)
    right = RightArmState(holding=Holding.SINGLE, grasp_stable=True, grasp_source="MANUAL")
    try:
        # This commissioning tool moves only the right arm.  Do not require a
        # left-hand TF that may be absent from a right-arm-only planner model.
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        result = module_9_right_place(
            right,
            planner,
            release_after_success=args.release_after_success,
            move_to_preinsert=not args.already_at_preinsert,
        )
    finally:
        planner.close()
    print(f"[hole-a] status={result.status.value}, state={right.motion_state}, message={result.message}")
    if result.status.value != "SUCCESS":
        return 1
    return 0


def _return_to_preinsert_only(config, *, live: bool) -> int:
    insertion = config.hole_a_insertion
    if insertion is None or not insertion.driver_waypoints:
        raise ValueError("config requires hole_a_insertion.driver_waypoints")
    taught_target = [float(value) for value in insertion.driver_waypoints[0]]
    if len(taught_target) != 6 or not all(math.isfinite(value) for value in taught_target):
        raise ValueError("hole_a_insertion.driver_waypoints[0] must contain six finite values")
    if not live:
        print(f"[hole-a][dry-run] servo-pose -> right_hole_a_preinsert xyz+rpy: {taught_target}")
        return 0

    # Planner RRTStar and SDK move_j can unwrap a valid taught joint target
    # onto an invalid equivalent revolution (-113.369 deg -> +246.631 deg).
    # Return through the same verified Cartesian servo-pose route used by the
    # direct insertion flow, avoiding that erroneous joint-space unwrapping.
    push_t_dir = Path(__file__).resolve().parent / "vendor/push_t"
    if str(push_t_dir) not in sys.path:
        sys.path.insert(0, str(push_t_dir))
    from move_by_joints import ensure_ready, import_sdk, move_l, rpy_to_rotvec

    target = taught_target[:3] + rpy_to_rotvec(taught_target[3:])

    arm = None
    try:
        arm = import_sdk().Arm(host="192.168.88.1")
        ensure_ready(arm, 2)
        print(f"[hole-a] servo-pose -> right_hole_a_preinsert xyz+rpy: {taught_target}")
        move_l(
            arm,
            2,
            target,
            linear_speed_mps=0.02,
            angular_speed_radps=0.20,
            max_linear_accel_mps2=0.05,
            max_angular_accel_radps2=0.40,
            hz=50.0,
            position_tolerance_m=0.0005,
            orientation_tolerance_rad=math.radians(2.0),
            settle_timeout_s=8.0,
        )
    finally:
        if arm is not None:
            arm.close()
    print("[hole-a] 已通过 servo-pose 到达 right_hole_a_preinsert；未启动插入。")
    return 0


def _run_direct_cartesian_servo(config, args: argparse.Namespace) -> int:
    """Commission the taught path through zhuyiming's verified direct servo helper."""
    if not args.live:
        print("[hole-a] direct-cartesian-servo dry-run; add --live to command hardware")
        return 0
    if not args.already_at_preinsert:
        raise ValueError("direct Cartesian commissioning requires --already-at-preinsert")
    insertion = config.hole_a_insertion
    if insertion is None or insertion.driver_waypoints is None:
        raise ValueError("config requires hole_a_insertion.driver_waypoints")
    driver_z_offset_m = float(getattr(args, "driver_z_offset_m", 0.0))
    if not math.isfinite(driver_z_offset_m) or abs(driver_z_offset_m) > 0.003:
        raise ValueError("--driver-z-offset-m must be finite and within ±0.003 m")
    driver_waypoints = [list(waypoint) for waypoint in insertion.driver_waypoints]
    for waypoint in driver_waypoints:
        waypoint[2] += driver_z_offset_m
    push_t_dir = Path(__file__).resolve().parent / "vendor/push_t"
    sys.path.insert(0, str(push_t_dir))
    from move_by_joints import (
        current_pose,
        ensure_ready,
        import_sdk,
        minimum_scurve_duration,
        orientation_error_rad,
        position_error_m,
        quat_to_rotvec,
        rotvec_to_quat,
        rpy_to_rotvec,
        slerp,
        smoothstep5,
    )

    sdk = import_sdk()
    arm = sensor = None
    try:
        arm = sdk.Arm(host="192.168.88.1")
        ensure_ready(arm, 2)
        expected_start = driver_waypoints[0][:3] + rpy_to_rotvec(
            driver_waypoints[0][3:]
        )
        actual_start = current_pose(arm, 2)
        start_position_error = position_error_m(expected_start, actual_start)
        start_orientation_error = orientation_error_rad(expected_start, actual_start)
        if start_position_error > 0.003 or start_orientation_error > math.radians(3.0):
            raise RuntimeError(
                "拒绝启动直接插入：当前末端不在 right_hole_a_preinsert 附近；"
                f"位置误差={start_position_error * 1000.0:.1f}mm，"
                f"姿态误差={math.degrees(start_orientation_error):.1f}°。"
                "请先通过 Free 示教将右臂移到预插入点。"
            )
        sensor = sdk.ForceSensor(host=config.planner_host, hand=2)
        baseline_frames = [sensor.wait_first(timeout=insertion.sample_timeout_seconds) for _ in range(insertion.baseline_samples)]
        baseline_force = [
            sum(float(frame["filtered_force"][axis]) for frame in baseline_frames)
            / len(baseline_frames)
            for axis in range(3)
        ]
        baseline_torque = [
            sum(float(frame["filtered_torque"][axis]) for frame in baseline_frames)
            / len(baseline_frames)
            for axis in range(3)
        ]
        baseline_ty = baseline_torque[1]
        print(
            f"[hole-a][direct] force 基线={baseline_force}，"
            f"torque 基线={baseline_torque}；从 preinsert 开始。"
        )
        if driver_z_offset_m:
            print(
                "[hole-a][direct] 本次所有示教路点的数值 Z 偏移="
                f"{driver_z_offset_m * 1000.0:+.1f} mm。"
            )
        min_fz, saw_contact, saw_peak = float("inf"), False, False
        servo_hz = 50.0
        servo_period = 1.0 / servo_hz

        class ServoPoseStream:
            """Continuously resend the latest pose throughout one servo session."""

            def __init__(self, initial_target: list[float]) -> None:
                self._target = list(initial_target)
                self._target_generation = 0
                self._pushed_generation = -1
                self._condition = threading.Condition()
                self._stop = threading.Event()
                self._error: BaseException | None = None
                self._thread = threading.Thread(
                    target=self._run,
                    name="hole-a-servo-pose-stream",
                    daemon=True,
                )

            def start(self) -> None:
                self._thread.start()

            def set_target(self, target: list[float]) -> None:
                """Publish a new target and wait until the stream sends it once."""
                with self._condition:
                    self._raise_if_failed()
                    self._target = list(target)
                    self._target_generation += 1
                    generation = self._target_generation
                    deadline = time.monotonic() + 12.0
                    while self._pushed_generation < generation:
                        self._raise_if_failed()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            raise RuntimeError("servo-pose 发送线程未及时确认新目标。")
                        self._condition.wait(timeout=remaining)
                    self._raise_if_failed()

            def close(self) -> None:
                self._stop.set()
                self._thread.join(timeout=12.0)
                if self._thread.is_alive():
                    raise RuntimeError("servo-pose 发送线程未能停止。")
                with self._condition:
                    self._raise_if_failed()

            def _raise_if_failed(self) -> None:
                if self._error is not None:
                    raise RuntimeError("servo-pose 连续发送失败。") from self._error

            def _run(self) -> None:
                deadline = time.monotonic()
                try:
                    while not self._stop.is_set():
                        with self._condition:
                            target = list(self._target)
                            generation = self._target_generation
                        arm.push_servo_pose(2, target)
                        with self._condition:
                            self._pushed_generation = max(
                                self._pushed_generation, generation
                            )
                            self._condition.notify_all()
                        deadline += servo_period
                        delay = deadline - time.monotonic()
                        if delay > 0.0:
                            self._stop.wait(delay)
                        else:
                            # A slow SDK round trip must not cause a burst of
                            # stale setpoints when it recovers.
                            deadline = time.monotonic()
                except BaseException as exc:
                    with self._condition:
                        self._error = exc
                        self._condition.notify_all()

        def move_in_active_session(target: list[float]) -> None:
            """Move to one checkpoint while the caller-owned servo session stays active."""
            start = current_pose(arm, 2)
            distance = position_error_m(target, start)
            angle = orientation_error_rad(target, start)
            print(f"当前末端 [xyz,rotvec]: {[round(value, 6) for value in start]}")
            print(f"目标末端 [xyz,rotvec]: {[round(value, 6) for value in target]}")
            if distance <= 0.0003 and angle <= 0.01:
                stream.set_target(target)
                print("当前末端已在目标容差内；保持连续发送目标位置。")
                return

            linear_duration = minimum_scurve_duration(distance, 0.003, 0.01)
            angular_duration = minimum_scurve_duration(angle, 0.05, 0.1)
            duration = max(linear_duration, angular_duration, servo_period)
            servo_steps = max(1, math.ceil(duration * servo_hz))
            print(
                f"[move_to_pose] distance={distance:.4f}m "
                f"rotation={math.degrees(angle):.2f}deg duration={duration:.2f}s "
                f"(linear={linear_duration:.2f}s angular={angular_duration:.2f}s) "
                f"steps={servo_steps} @ {servo_hz:.1f}Hz"
            )
            q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
            for index in range(1, servo_steps + 1):
                alpha = smoothstep5(index / float(servo_steps))
                waypoint = [
                    start[axis] + alpha * (target[axis] - start[axis])
                    for axis in range(3)
                ]
                waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
                stream.set_target(waypoint)

            settle_deadline = time.monotonic() + 3.0
            final_pose = start
            while time.monotonic() < settle_deadline:
                stream.set_target(target)
                final_pose = current_pose(arm, 2)
                pos_error = position_error_m(target, final_pose)
                rot_error = orientation_error_rad(target, final_pose)
                if pos_error <= 0.0003 and rot_error <= 0.01:
                    print(
                        f"[move_to_pose] reached: pos_error={pos_error * 1000.0:.2f}mm "
                        f"rot_error={math.degrees(rot_error):.2f}deg"
                    )
                    return
            pos_error = position_error_m(target, final_pose)
            rot_error = orientation_error_rad(target, final_pose)
            raise RuntimeError(
                "末端位姿未在超时内到位："
                f"位置误差={pos_error * 1000.0:.1f} mm，"
                f"旋转误差={math.degrees(rot_error):.2f}°。"
            )

        initial_servo_target = current_pose(arm, 2)
        stream = ServoPoseStream(initial_servo_target)
        servo_body_error: BaseException | None = None
        arm.start_servo_pose(
            2,
            pose_type="absolute",
            frame="world",
            tcp=[0.0] * 6,
            input_frequency_hz=servo_hz,
        )
        stream.start()
        print("[hole-a][direct] servo-pose session 已启动；全路径连续发送位置。")
        try:
            for stage, (start, end) in enumerate(zip(driver_waypoints, driver_waypoints[1:]), start=1):
                distance = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
                steps = math.ceil(distance / insertion.step_size_m)
                for step in range(1, steps + 1):
                    alpha = step / steps
                    xyz_rpy = [start[i] + alpha * (end[i] - start[i]) for i in range(6)]
                    move_in_active_session(xyz_rpy[:3] + rpy_to_rotvec(xyz_rpy[3:]))
                    # ForceSensor caches its latest frame in a subscriber
                    # thread, so this read returns immediately after startup
                    # and does not interrupt the 50 Hz command stream.
                    frame = sensor.wait_first(timeout=insertion.sample_timeout_seconds)
                    fz, ty = float(frame["filtered_force"][2]), float(frame["filtered_torque"][1])
                    force = [float(value) for value in frame["filtered_force"]]
                    torque = [float(value) for value in frame["filtered_torque"]]
                    force_delta = [value - base for value, base in zip(force, baseline_force)]
                    torque_delta = [value - base for value, base in zip(torque, baseline_torque)]
                    if math.sqrt(sum(value * value for value in force_delta)) > insertion.max_force_delta_n:
                        raise RuntimeError(
                            "孔 A 直接插入超出力硬限："
                            f"delta_force={force_delta} N"
                        )
                    if math.sqrt(sum(value * value for value in torque_delta)) > insertion.max_torque_delta_nm:
                        raise RuntimeError(
                            "孔 A 直接插入超出扭矩硬限："
                            f"delta_torque={torque_delta} N*m"
                        )
                    min_fz = min(min_fz, fz)
                    saw_contact |= ty - baseline_ty >= insertion.ty_contact_rise_nm
                    saw_peak |= fz <= insertion.fz_peak_max_n
                    print(f"[hole-a][direct] stage={stage} step={step}/{steps} Fz={fz:.3f} Ty={ty:.3f} minFz={min_fz:.3f}")
                    if stage == 3 and saw_contact and saw_peak and fz - min_fz >= insertion.fz_recovery_n and abs(fz) <= insertion.fz_final_abs_max_n:
                        print("[hole-a][direct] SUCCESS: 孔 A 插入特征已确认；保持夹爪闭合。")
                        return 0
            print("[hole-a][direct] FAULT: 到终点未满足孔 A Fz 特征。")
            return 1
        except BaseException as exc:
            servo_body_error = exc
            raise
        finally:
            cleanup_errors: list[tuple[str, Exception]] = []
            try:
                stream.close()
            except Exception as stream_close_exc:
                cleanup_errors.append(("停止连续发送线程", stream_close_exc))
            try:
                arm.stop_servo_pose(2)
            except Exception as stop_exc:
                cleanup_errors.append(("停止 servo-pose session", stop_exc))
            if not cleanup_errors:
                print("[hole-a][direct] servo-pose session 已停止。")
            elif servo_body_error is None:
                action, cleanup_exc = cleanup_errors[0]
                raise RuntimeError(f"{action}失败。") from cleanup_exc
            else:
                details = "; ".join(
                    f"{action}: {cleanup_exc}"
                    for action, cleanup_exc in cleanup_errors
                )
                print(f"[hole-a][direct] WARNING: 异常退出后的清理失败: {details}")
    finally:
        if sensor is not None: sensor.close()
        if arm is not None: arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
