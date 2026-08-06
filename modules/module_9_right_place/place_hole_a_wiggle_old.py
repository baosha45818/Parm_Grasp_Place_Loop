#!/usr/bin/env python3
"""Place the right-hand workpiece by backing up 1 mm and wiggling in Y.

The taught start is a fixed right-arm joint vector.  After that joint move,
the actual SDK end-effector pose is used as the world-frame centre.  A cycle is: back 1 mm in world X,
then Y=centre+1 mm, Y=centre-1 mm, then back to the centre.  Thus its total
left/right span is 2 mm.  Force is checked after every short move.  Hole
entry is declared at the back-centre pose when a previously high Fx/Ty load
abruptly unloads in one 1 mm back-pull step.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUSH_T_DIR = Path(__file__).resolve().parent / "vendor/push_t"
if str(PUSH_T_DIR) not in sys.path:
    sys.path.insert(0, str(PUSH_T_DIR))

from move_by_joints import (  # noqa: E402
    current_pose,
    ensure_ready,
    import_sdk,
    minimum_scurve_duration,
    orientation_error_rad,
    position_error_m,
    quat_to_rotvec,
    rotvec_to_quat,
    rotvec_to_rpy,
    rpy_to_rotvec,
    slerp,
    smoothstep5,
)
from loop_config import load_config  # noqa: E402
from tool.planner_runtime import PlannerRuntime  # noqa: E402
from tool.left_motion_right_servo_lock import right_servo_guard  # noqa: E402


DEFAULT_PLANNER_CONFIG = ROOT / "config.hole_a_first_live.json"
INITIAL_RIGHT_JOINTS_DEG = [
    22.49,
    23.12,
    0.50,
    67.24,
    -62.29,
    -64.43,
    25.40
]


class PlacementContactReached(RuntimeError):
    """插孔搜索已发生用户定义的接触阈值，按放置成功结束 ServoPose。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="右臂回固定关节起点后，每后拉 1 mm 做一次左右 2 mm 摆动，以力特征确认孔已进入。"
    )
    parser.add_argument(
        "--initial-motion", choices=("planner-movej", "joint-servo"), default="planner-movej",
        help="回固定关节起点的方式；默认规划器避碰 move_j，joint-servo 仅用于已验证的近距离动作",
    )
    parser.add_argument(
        "--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG,
        help=f"规划器配置；默认 {DEFAULT_PLANNER_CONFIG}",
    )
    parser.add_argument("--live", action="store_true", help="连接真实硬件")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--force-host", default="127.0.0.1")
    parser.add_argument(
        "--reset-right-task-before-servo",
        action="store_true",
        help="启动 servo-pose 前仅调用 cancel_task(2) 清理右臂残留任务；总控串联时使用",
    )
    parser.add_argument(
        "--retry-lift-mm", type=float, default=0.1,
        help="后拉未到位时，沿 right_hand_transfer_end 局部 +Z 的单次恢复抬高量；默认 0.1 mm",
    )
    parser.add_argument(
        "--max-retry-lifts", type=int, default=10,
        help="同一后拉中心目标最多连续恢复上提次数；默认 10 次",
    )
    parser.add_argument("--back-step-mm", type=float, default=1.0, help="每周期世界 X 后拉距离，默认 1 mm")
    parser.add_argument(
        "--wiggle-half-amplitude-mm", type=float, default=1.0,
        help="世界 Y 相对中心的单侧摆幅，默认 1 mm（左右总摆幅 2 mm）",
    )
    parser.add_argument("--fast-linear-speed-mm-s", type=float, default=5.0, help="12 mm 前短段线速度，默认 5 mm/s")
    parser.add_argument("--fast-linear-accel-mm-s2", type=float, default=20.0, help="12 mm 前短段线加速度，默认 20 mm/s²")
    parser.add_argument("--slow-after-back-mm", type=float, default=12.0, help="累计后拉达到该距离后降速，默认 12 mm")
    parser.add_argument("--slow-linear-speed-mm-s", type=float, default=3.0, help="降速后短段线速度，默认 3 mm/s")
    parser.add_argument("--slow-linear-accel-mm-s2", type=float, default=10.0, help="降速后短段线加速度，默认 10 mm/s²")
    parser.add_argument("--max-back-mm", type=float, default=40.0, help="最大累计后拉距离，默认 40 mm")
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument("--sample-timeout", type=float, default=2.0)
    parser.add_argument(
        "--max-force-delta-n", type=float, default=20.0,
        help="相对力基线的合力硬限；默认 20 N，超过立即停止",
    )
    parser.add_argument("--max-torque-delta-nm", type=float, default=5.0, help="相对力基线的合扭矩硬限；默认 5 N*m")
    # Entry is a two-frame release: Fx must drop sharply from contact, while
    # Ty must return close to zero.  The Ty absolute-value guard rejects a
    # transition into an opposite-direction collision.
    parser.add_argument("--entry-pre-fx-n", type=float, default=10.0, help="进洞前一步所需最小 dFx")
    parser.add_argument("--entry-pre-ty-nm", type=float, default=2.5, help="进洞前一步所需最小 dTy")
    parser.add_argument("--entry-fx-drop-n", type=float, default=10.0, help="进洞单步所需最小 dFx 下降量")
    parser.add_argument("--entry-ty-drop-nm", type=float, default=2.5, help="进洞单步所需最小 dTy 下降量")
    parser.add_argument("--entry-ty-abs-nm", type=float, default=1.0, help="进洞当前步允许的最大 |dTy|")
    # A confirmed insertion can also unload from a moderate contact load,
    # rather than from the older >=10 N / >=2.5 N*m peak.  Keep this as a
    # separate condition so the original strong-jump rule is unchanged.
    parser.add_argument("--entry-relaxed-pre-fx-n", type=float, default=6.0, help="中等进洞条件的前一步最小 |dFx|")
    parser.add_argument("--entry-relaxed-pre-ty-nm", type=float, default=1.5, help="中等进洞条件的前一步最小 |dTy|")
    parser.add_argument("--entry-relaxed-fx-drop-n", type=float, default=4.0, help="中等进洞条件的最小 |dFx| 下降量")
    parser.add_argument("--entry-relaxed-ty-drop-nm", type=float, default=1.0, help="中等进洞条件的最小 |dTy| 下降量")
    # A smaller, simultaneous force/torque unload can also mean that the part
    # has entered the hole.  Unlike the large-jump rules above, require the
    # load to remain low while holding the current pose.  The previous load
    # may be the cycle centre and the unload may occur during a lateral wiggle.
    parser.add_argument("--entry-release-pre-force-n", type=float, default=2.0, help="卸载进洞前一步所需最小 |dF|")
    parser.add_argument("--entry-release-pre-torque-nm", type=float, default=0.4, help="卸载进洞前一步所需最小 |dT|")
    parser.add_argument("--entry-release-force-n", type=float, default=1.5, help="卸载进洞当前及确认采样允许的最大 |dF|")
    parser.add_argument("--entry-release-torque-nm", type=float, default=0.4, help="卸载进洞当前及确认采样允许的最大 |dT|")
    parser.add_argument("--entry-release-force-drop-n", type=float, default=2.0, help="卸载进洞所需最小 |dF| 下降量")
    parser.add_argument("--entry-release-torque-drop-nm", type=float, default=0.4, help="卸载进洞所需最小 |dT| 下降量")
    parser.add_argument("--entry-release-confirm-samples", type=int, default=3, help="卸载进洞原地确认采样次数，默认 3")
    parser.add_argument("--entry-release-confirm-interval-s", type=float, default=0.1, help="卸载进洞确认采样间隔，默认 0.1 秒")
    # In this taught setup, a positive Z-force rise at the back-centre pose is
    # also an insertion signature.  Keep it centre-only to avoid treating a
    # lateral-wiggle collision peak as a successful insertion.
    parser.add_argument("--entry-fz-rise-n", type=float, default=3.0, help="Z 力进洞条件所需最小正向 dFz")
    parser.add_argument("--entry-fz-rise-delta-n", type=float, default=1.5, help="Z 力进洞条件相对上一中心点所需最小 dFz 上升量")
    # Some insertions happen during the lateral search rather than the next
    # back-centre step.  Confirm this with a Fx/Ty sign reversal plus a large
    # drop, rather than requiring one fixed final magnitude.  Confirmed runs:
    # +7.73 -> -12.50 N / +1.90 -> -3.88 N*m, and
    # +9.63 -> -7.52 N / +2.28 -> -2.32 N*m.
    parser.add_argument("--wiggle-entry-pre-fx-n", type=float, default=5.0, help="摆动进洞前中心点所需最小 dFx")
    parser.add_argument("--wiggle-entry-pre-ty-nm", type=float, default=1.5, help="摆动进洞前中心点所需最小 dTy")
    parser.add_argument("--wiggle-entry-fx-drop-n", type=float, default=10.0, help="摆动进洞所需最小 dFx 下降量")
    parser.add_argument("--wiggle-entry-ty-drop-nm", type=float, default=3.0, help="摆动进洞所需最小 dTy 下降量")
    parser.add_argument("--min-entry-back-mm", type=float, default=8.0, help="至少后拉该距离后才允许判进洞")
    parser.add_argument("--output", type=Path, help="运行日志路径；默认 logs/hole_a_wiggle_<时间>.log")
    args = parser.parse_args()
    if args.live != args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    for name in ("retry_lift_mm", "back_step_mm", "wiggle_half_amplitude_mm", "fast_linear_speed_mm_s", "fast_linear_accel_mm_s2", "slow_after_back_mm", "slow_linear_speed_mm_s", "slow_linear_accel_mm_s2", "max_back_mm", "sample_timeout", "max_force_delta_n", "max_torque_delta_nm", "min_entry_back_mm", "entry_pre_fx_n", "entry_pre_ty_nm", "entry_fx_drop_n", "entry_ty_drop_nm", "entry_ty_abs_nm", "entry_relaxed_pre_fx_n", "entry_relaxed_pre_ty_nm", "entry_relaxed_fx_drop_n", "entry_relaxed_ty_drop_nm", "entry_release_pre_force_n", "entry_release_pre_torque_nm", "entry_release_force_n", "entry_release_torque_nm", "entry_release_force_drop_n", "entry_release_torque_drop_nm", "entry_release_confirm_interval_s", "entry_fz_rise_n", "entry_fz_rise_delta_n", "wiggle_entry_pre_fx_n", "wiggle_entry_pre_ty_nm", "wiggle_entry_fx_drop_n", "wiggle_entry_ty_drop_nm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    if args.back_step_mm > 2.0 or args.wiggle_half_amplitude_mm > 2.0:
        parser.error("为安全起见，单步后拉和单侧摆幅均不得超过 2 mm")
    if args.baseline_samples < 1:
        parser.error("--baseline-samples 必须至少为 1")
    if args.max_retry_lifts < 1:
        parser.error("--max-retry-lifts 必须至少为 1")
    if args.entry_release_confirm_samples < 1:
        parser.error("--entry-release-confirm-samples 必须至少为 1")
    return args


def vector(frame: dict, key: str) -> list[float]:
    values = frame.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError(f"无效力传感器帧 {key}: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"力传感器 {key} 存在非有限值")
    return result


def read_wrench(sensor: object, timeout: float) -> tuple[list[float], list[float]]:
    """Return the SDK subscriber's currently cached force frame."""
    frame = sensor.wait_first(timeout=timeout)  # type: ignore[attr-defined]
    return vector(frame, "filtered_force"), vector(frame, "filtered_torque")


def mean(rows: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(3)]


def norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def command_for(target_rpy: list[float]) -> list[float]:
    return target_rpy[:3] + rpy_to_rotvec(target_rpy[3:])


class TargetNotReached(RuntimeError):
    """A commanded servo target remained outside its endpoint tolerance."""


def write_control_snapshot(arm: object, output_file: object, event: str) -> None:
    """Record both arms so a right-servo failure can be correlated with left-arm tasks."""
    state = arm.get_state() or {}  # type: ignore[attr-defined]
    arms = state.get("state") or {}
    record: dict[str, object] = {
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "monotonic_s": time.monotonic(),
        "event": event,
        "system_state": state.get("system_state"),
    }
    for arm_key in ("left", "right"):
        per_arm = arms.get(arm_key) or {}
        velocity = per_arm.get("velocity")
        velocity_values = (
            [float(value) for value in velocity]
            if isinstance(velocity, (list, tuple)) and len(velocity) == 7
            else None
        )
        record[arm_key] = {
            "task_state": per_arm.get("task_state"),
            "enabled": per_arm.get("enabled"),
            "position_deg": per_arm.get("position"),
            "velocity_deg_s": velocity_values,
            "max_abs_velocity_deg_s": (
                max(abs(value) for value in velocity_values)
                if velocity_values is not None
                else None
            ),
            "eef_pose": per_arm.get("eef_pose"),
        }
    output_file.write("[CONTROL] " + json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[attr-defined]
    output_file.flush()  # type: ignore[attr-defined]


def wait_for_right_task_stopped(
    arm: object, output_file: object, *, timeout_s: float = 3.0,
) -> None:
    """Confirm cancel_task(2) has removed the old right-arm task before servo starts."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = arm.get_state() or {}  # type: ignore[attr-defined]
        task_state = str(
            (((state.get("state") or {}).get("right") or {}).get("task_state") or "")
        ).strip().lower()
        if task_state != "running":
            write_control_snapshot(arm, output_file, "right_cancel_confirmed")
            return
        time.sleep(0.05)
    write_control_snapshot(arm, output_file, "right_cancel_wait_timeout")
    raise RuntimeError("cancel_task(2) 后右臂 task_state 仍为 running")


def local_z_in_world(rpy: list[float]) -> list[float]:
    """Unit +Z of right_hand_transfer_end expressed in SDK world coordinates."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Third column of Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr]


def current_right_joints_deg(arm: object) -> list[float]:
    state = arm.get_state()  # type: ignore[attr-defined]
    values = ((state or {}).get("state") or {}).get("right", {}).get("position")
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise RuntimeError(f"右臂关节状态不可用: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"右臂关节状态含非有限值: {values!r}")
    return result


def wait_for_right_joint_target(
    arm: object, target_deg: list[float], label: str, *, timeout_s: float = 90.0,
) -> None:
    """Wait until the planner task has ended and the physical arm is settled."""
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    last_velocity_deg_s = float("inf")
    last_task_state = "unknown"
    stable_samples = 0
    required_stable_samples = 20  # 1 second at the 20 Hz polling rate below.
    while time.monotonic() < deadline:
        state = arm.get_state()  # type: ignore[attr-defined]
        per_arm = ((state or {}).get("state") or {}).get("right") or {}
        actual = per_arm.get("position")
        velocity = per_arm.get("velocity")
        if not isinstance(actual, (list, tuple)) or len(actual) != 7:
            stable_samples = 0
            time.sleep(0.05)
            continue
        actual_deg = [float(value) for value in actual]
        # Compare wrapped angles so equivalent values around +/-180 deg do not fail.
        errors_deg = [
            abs((actual - target + 180.0) % 360.0 - 180.0)
            for actual, target in zip(actual_deg, target_deg)
        ]
        last_error_deg = max(errors_deg)
        if isinstance(velocity, (list, tuple)) and len(velocity) == 7:
            last_velocity_deg_s = max(abs(float(value)) for value in velocity)
        else:
            last_velocity_deg_s = float("inf")
        last_task_state = str(per_arm.get("task_state", "")).strip().lower()
        settled = (
            last_error_deg <= 0.5
            and last_velocity_deg_s <= 0.05
            and last_task_state != "running"
        )
        stable_samples = stable_samples + 1 if settled else 0
        if stable_samples >= required_stable_samples:
            print(
                f"{label}已物理到位并稳定1秒：最大关节误差={last_error_deg:.3f}°，"
                f"最大关节速度={last_velocity_deg_s:.4f}°/s，"
                f"task_state={last_task_state!r}。"
            )
            return
        time.sleep(0.05)
    raise TargetNotReached(
        f"{label}未在 {timeout_s:.1f}s 内物理到位并稳定："
        f"最大关节误差={last_error_deg:.3f}°，"
        f"最大关节速度={last_velocity_deg_s:.4f}°/s，"
        f"task_state={last_task_state!r}。"
    )


def move_to_saved_joints_servo(arm: object, target_deg: list[float]) -> None:
    """Continuously interpolate the recorded seven joint angles through servo_j."""
    start_deg = current_right_joints_deg(arm)
    largest_delta = max(abs(end - start) for start, end in zip(start_deg, target_deg))
    print(
        "关节 servo 回起点："
        f"当前={[round(value, 3) for value in start_deg]}，"
        f"目标={[round(value, 3) for value in target_deg]}，"
        f"最大关节差={largest_delta:.2f}°。"
    )
    if largest_delta <= 0.5:
        return
    hz, max_speed_deg_s = 50.0, 3.0
    duration = max(1.0 / hz, 1.875 * largest_delta / max_speed_deg_s)
    steps = max(1, math.ceil(duration * hz))
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        alpha = smoothstep5(index / float(steps))
        target = [start + alpha * (end - start) for start, end in zip(start_deg, target_deg)]
        arm.servo_j(2, target)  # type: ignore[attr-defined]
        deadline += 1.0 / hz
        time.sleep(max(0.0, deadline - time.monotonic()))
    settle_deadline = time.monotonic() + 8.0
    final = start_deg
    while time.monotonic() < settle_deadline:
        arm.servo_j(2, target_deg)  # type: ignore[attr-defined]
        final = current_right_joints_deg(arm)
        if max(abs(end - actual) for end, actual in zip(target_deg, final)) <= 0.5:
            print("关节 servo 已到保存起点。")
            return
        time.sleep(1.0 / hz)
    raise TargetNotReached(
        "关节 servo 未在超时内到位："
        f"最大关节误差={max(abs(end - actual) for end, actual in zip(target_deg, final)):.2f}°。"
    )


def move_to_saved_joints_planned(
    arm: object, config_path: Path, target_deg: list[float],
) -> None:
    """Plan against the configured scene (including the table), then execute once."""
    config = load_config(config_path, live=True)
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        print(
            "Planner move_j 回起点：先使用当前规划场景做右臂碰撞检查和轨迹规划，"
            f"目标={[round(value, 3) for value in target_deg]}。"
        )
        planner.move_right_joints(target_deg)
        print("Planner move_j 已下发，等待右臂物理到固定关节起点。")
        wait_for_right_joint_target(arm, target_deg, "固定关节起点")
    finally:
        planner.close()


def move_in_active_session(
    arm: object,
    target_rpy: list[float],
    *,
    initial: bool = False,
    position_tolerance_m: float | None = None,
    linear_speed_mps: float | None = None,
    linear_accel_mps2: float | None = None,
    diagnostic: object | None = None,
) -> None:
    """Move within the caller-owned continuous servo-pose session."""
    target = command_for(target_rpy)
    start = current_pose(arm, 2)
    distance = position_error_m(target, start)
    angle = orientation_error_rad(target, start)
    if diagnostic is not None:
        diagnostic("servo_move_begin")  # type: ignore[operator]
    position_tolerance = position_tolerance_m if position_tolerance_m is not None else (0.0005 if initial else 0.0003)
    orientation_tolerance = 0.02 if initial else 0.01
    if distance <= position_tolerance and angle <= orientation_tolerance:
        return
    hz = 50.0
    period = 1.0 / hz
    speed = 0.01 if initial else (linear_speed_mps if linear_speed_mps is not None else 0.003)
    accel = 0.02 if initial else (linear_accel_mps2 if linear_accel_mps2 is not None else 0.01)
    linear_duration = minimum_scurve_duration(
        distance, speed, accel
    )
    angular_duration = minimum_scurve_duration(
        angle, 0.10 if initial else 0.05, 0.20 if initial else 0.10
    )
    steps = max(1, math.ceil(max(linear_duration, angular_duration, period) * hz))
    q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        alpha = smoothstep5(index / float(steps))
        waypoint = [start[axis] + alpha * (target[axis] - start[axis]) for axis in range(3)]
        waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
        arm.push_servo_pose(2, waypoint)  # type: ignore[attr-defined]
        if diagnostic is not None and (index == 1 or index % 10 == 0 or index == steps):
            diagnostic(f"servo_push_waypoint_{index}_of_{steps}")  # type: ignore[operator]
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))
    settle_deadline = time.monotonic() + (8.0 if initial else 4.0)
    final_pose = start
    settle_push_count = 0
    while time.monotonic() < settle_deadline:
        settle_push_count += 1
        arm.push_servo_pose(2, target)  # type: ignore[attr-defined]
        final_pose = current_pose(arm, 2)
        if (
            position_error_m(target, final_pose) <= position_tolerance
            and orientation_error_rad(target, final_pose) <= orientation_tolerance
        ):
            if diagnostic is not None:
                diagnostic("servo_target_reached")  # type: ignore[operator]
            return
        if diagnostic is not None and settle_push_count % 10 == 0:
            diagnostic(f"servo_settle_push_{settle_push_count}")  # type: ignore[operator]
        time.sleep(period)
    if diagnostic is not None:
        diagnostic("servo_target_timeout")  # type: ignore[operator]
    raise TargetNotReached(
        "末端未在超时内到位："
        f"位置误差={position_error_m(target, final_pose) * 1000.0:.1f}mm，"
        f"姿态误差={math.degrees(orientation_error_rad(target, final_pose)):.2f}°。"
    )


def main() -> int:
    run_started = time.monotonic()
    args = parse_args()
    # Recorded UI joint values are the sole taught start definition.
    saved_joints_deg = list(INITIAL_RIGHT_JOINTS_DEG)
    if not args.live:
        print(
            "dry-run：不会连接硬件；固定右臂起始关节角="
            f"{saved_joints_deg}，初始回位方式={args.initial_motion}"
        )
        print(
            f"每周期：X -= {args.back_step_mm:.3f}mm，Y=中心±{args.wiggle_half_amplitude_mm:.3f}mm，再回中心；"
            f"{args.slow_after_back_mm:.1f}mm 前 {args.fast_linear_speed_mm_s:.1f}mm/s，之后 "
            f"{args.slow_linear_speed_mm_s:.1f}mm/s；最大后拉 {args.max_back_mm:.1f}mm。"
        )
        return 0

    output_path = args.output or Path("logs") / f"hole_a_wiggle_{datetime.now():%Y%m%d_%H%M%S}.log"
    output_path = output_path.expanduser().resolve().with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"运行日志将实时写入：{output_path}", flush=True)
    sdk = import_sdk()
    arm = sensor = None
    servo_started = False
    servo_guard = None
    log_file = output_path.open("w", encoding="utf-8")

    def log_event(event: str, **fields: object) -> None:
        record = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.monotonic() - run_started, 3),
            "event": event,
            **fields,
        }
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

    log_event(
        "run_started",
        initial_motion=args.initial_motion,
        back_step_mm=args.back_step_mm,
        wiggle_half_amplitude_mm=args.wiggle_half_amplitude_mm,
        max_back_mm=args.max_back_mm,
    )
    try:
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, 2)
        write_control_snapshot(arm, log_file, "script_connected")
        sensor = sdk.ForceSensor(host=args.force_host, hand=2)
        print("正在移动到保存的起始位姿……")
        if args.initial_motion == "planner-movej":
            move_to_saved_joints_planned(arm, args.planner_config, saved_joints_deg)
        else:
            move_to_saved_joints_servo(arm, saved_joints_deg)
        # Contact limits must be relative to the actual red-dot start state,
        # not the state before this script moved the arm to that point.
        print("已到起始位姿；请勿接触工件，正在稳定并采集右腕力基线……")
        time.sleep(0.5)
        baseline_samples = []
        for _ in range(args.baseline_samples):
            baseline_samples.append(read_wrench(sensor, args.sample_timeout))
            time.sleep(0.05)
        force_base = mean([force for force, _ in baseline_samples])
        torque_base = mean([torque for _, torque in baseline_samples])
        print(f"起始位姿力基线 F={force_base} N，T={torque_base} N*m")
        log_event("force_baseline", force_n=force_base, torque_nm=torque_base)

        # Read the servo centre only after the loaded arm has settled during
        # baseline collection.  Reading it before this delay leaves a stale
        # target when the held workpiece causes sub-millimetre elastic settling.
        initial_actual = current_pose(arm, 2)
        centre = initial_actual[:3] + rotvec_to_rpy(initial_actual[3:])
        print(
            "力基线采集完成；以后拉中心使用稳定后的实际末端 xyz+rpy="
            f"{[round(value, 6) for value in centre]}。"
        )
        start_x = centre[0]
        log_event("search_centre_ready", pose_xyz_rpy=centre)

        # Keep the single log open until the ServoPose session has stopped.
        if True:
            previous_back_wrench: tuple[list[float], list[float]] | None = None
            def sample(event: str, cycle: int, back_mm: float, target: list[float]) -> tuple[list[float], list[float]]:
                force, torque = read_wrench(sensor, args.sample_timeout)
                dforce = [value - base for value, base in zip(force, force_base)]
                dtorque = [value - base for value, base in zip(torque, torque_base)]
                actual = current_pose(arm, 2)
                actual_rpy = rotvec_to_rpy(actual[3:])
                force_norm, torque_norm = norm(dforce), norm(dtorque)
                motion = {
                    "initial": "initial",
                    "back_centre": "backward_x_negative",
                    "wiggle_y_plus": "left_y_positive",
                    "wiggle_y_minus": "right_y_negative",
                    "wiggle_return_centre": "return_to_centre",
                }.get(event, event)
                log_event(
                    "force_sample",
                    motion=motion,
                    cycle=cycle,
                    back_mm=round(back_mm, 3),
                    target_xyz_rpy=target,
                    actual_xyz_rpy=actual[:3] + actual_rpy,
                    force_n=force,
                    torque_nm=torque,
                    delta_force_n=dforce,
                    delta_torque_nm=dtorque,
                    delta_force_norm_n=round(force_norm, 6),
                    delta_torque_norm_nm=round(torque_norm, 6),
                    insertion_decision="pending",
                )
                print(
                    f"[{event}] back={back_mm:.1f}mm dF={dforce} |dF|={force_norm:.2f}N "
                    f"dT={dtorque} |dT|={torque_norm:.2f}N*m"
                )
                if force_norm > args.max_force_delta_n or torque_norm > args.max_torque_delta_nm:
                    if event != "initial":
                        log_event(
                            "insertion_decision",
                            result="entered_by_contact_limit",
                            motion=motion,
                            cycle=cycle,
                            back_mm=round(back_mm, 3),
                            delta_force_norm_n=round(force_norm, 6),
                            delta_torque_norm_nm=round(torque_norm, 6),
                        )
                        print(
                            "SUCCESS：插孔搜索阶段达到相对力/扭矩接触限；"
                            "按已进入孔位处理，停止后续 ServoPose 摆动并进入松爪收尾。"
                            f" event={event} back={back_mm:.1f}mm "
                            f"|dF|={force_norm:.2f}N（限 {args.max_force_delta_n:.2f}），"
                            f"|dT|={torque_norm:.2f}N*m（限 {args.max_torque_delta_nm:.2f}）。"
                        )
                        raise PlacementContactReached()
                    raise RuntimeError(
                        "安全停止：相对力/扭矩超限；"
                        f"|dF|={force_norm:.2f}N（限 {args.max_force_delta_n:.2f}），"
                        f"|dT|={torque_norm:.2f}N*m（限 {args.max_torque_delta_nm:.2f}）。"
                    )
                return dforce, dtorque

            sample("initial", 0, 0.0, centre)
            write_control_snapshot(arm, log_file, "before_right_task_reset")
            if args.reset_right_task_before_servo:
                print(
                    "总控模式：Servo 启动前仅调用 cancel_task(2) 清理右臂残留任务。",
                    flush=True,
                )
                arm.cancel_task(2)  # type: ignore[attr-defined]
                wait_for_right_task_stopped(arm, log_file)
            # Arm immediately before the first non-zero command, then keep
            # feeding the same session through all back/wiggle segments.
            servo_guard = right_servo_guard()
            servo_guard.__enter__()
            arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0
            )
            servo_started = True
            write_control_snapshot(arm, log_file, "after_start_servo_pose")
            diagnostic = lambda event: write_control_snapshot(arm, log_file, event)
            cycles = math.ceil(args.max_back_mm / args.back_step_mm)
            for cycle in range(1, cycles + 1):
                back_mm = min(cycle * args.back_step_mm, args.max_back_mm)
                if back_mm >= args.slow_after_back_mm:
                    linear_speed_mps = args.slow_linear_speed_mm_s / 1000.0
                    linear_accel_mps2 = args.slow_linear_accel_mm_s2 / 1000.0
                    speed_label = "slow"
                else:
                    linear_speed_mps = args.fast_linear_speed_mm_s / 1000.0
                    linear_accel_mps2 = args.fast_linear_accel_mm_s2 / 1000.0
                    speed_label = "fast"
                print(
                    f"[motion_profile] back={back_mm:.1f}mm {speed_label} "
                    f"v={linear_speed_mps * 1000.0:.1f}mm/s a={linear_accel_mps2 * 1000.0:.1f}mm/s²"
                )
                log_event(
                    "motion_start",
                    motion="backward_x_negative",
                    cycle=cycle,
                    back_mm=round(back_mm, 3),
                    speed_profile=speed_label,
                    linear_speed_mm_s=round(linear_speed_mps * 1000.0, 3),
                    linear_accel_mm_s2=round(linear_accel_mps2 * 1000.0, 3),
                    insertion_decision="searching",
                )
                centre[0] = start_x - back_mm / 1000.0
                try:
                    move_in_active_session(
                        arm, centre,
                        linear_speed_mps=linear_speed_mps,
                        linear_accel_mps2=linear_accel_mps2,
                        diagnostic=diagnostic,
                    )
                    back_event = "back_centre"
                except TargetNotReached:
                    back_event = ""
                    for retry_index in range(1, args.max_retry_lifts + 1):
                        actual = current_pose(arm, 2)
                        actual_rpy = actual[:3] + rotvec_to_rpy(actual[3:])
                        lift_axis = local_z_in_world(actual_rpy[3:])
                        lift_target = list(actual_rpy)
                        for axis in range(3):
                            lift_target[axis] += lift_axis[axis] * args.retry_lift_mm / 1000.0
                        print(
                            "后拉目标未到位；沿 right_hand_transfer_end 局部 +Z 抬高 "
                            f"{args.retry_lift_mm:.3f}mm 后重试 "
                            f"({retry_index}/{args.max_retry_lifts})。"
                        )
                        # 0.1 mm is below the normal 0.3 mm endpoint tolerance;
                        # use 50 µm here so the commanded recovery is real.
                        try:
                            move_in_active_session(
                                arm, lift_target, position_tolerance_m=0.00005,
                                linear_speed_mps=linear_speed_mps,
                                linear_accel_mps2=linear_accel_mps2,
                                diagnostic=diagnostic,
                            )
                            planned_lift_axis = local_z_in_world(centre[3:])
                            for axis in range(3):
                                centre[axis] += planned_lift_axis[axis] * args.retry_lift_mm / 1000.0
                            move_in_active_session(
                                arm, centre, linear_speed_mps=linear_speed_mps,
                                linear_accel_mps2=linear_accel_mps2,
                                diagnostic=diagnostic,
                            )
                            back_event = f"back_centre_after_lift_retry_{retry_index}"
                            break
                        except TargetNotReached:
                            if retry_index == args.max_retry_lifts:
                                raise
                            print("恢复后仍未到位，继续下一次 0.1mm 恢复。")
                    if not back_event:
                        raise RuntimeError("后拉恢复循环未产生有效的中心目标")
                dforce, dtorque = sample(back_event, cycle, back_mm, centre)
                if previous_back_wrench is not None and back_mm >= args.min_entry_back_mm:
                    previous_force, previous_torque = previous_back_wrench
                    strong_entry = (
                        previous_force[0] >= args.entry_pre_fx_n
                        and previous_torque[1] >= args.entry_pre_ty_nm
                        and previous_force[0] - dforce[0] >= args.entry_fx_drop_n
                        and (
                            previous_torque[1] - dtorque[1] >= args.entry_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    # Use magnitudes here because the taught path can invert
                    # the sensor sign between otherwise equivalent setups.
                    relaxed_entry = (
                        abs(previous_force[0]) >= args.entry_relaxed_pre_fx_n
                        and abs(previous_torque[1]) >= args.entry_relaxed_pre_ty_nm
                        and abs(previous_force[0]) - abs(dforce[0])
                        >= args.entry_relaxed_fx_drop_n
                        and (
                            abs(previous_torque[1]) - abs(dtorque[1])
                            >= args.entry_relaxed_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    previous_force_norm = norm(previous_force)
                    previous_torque_norm = norm(previous_torque)
                    force_norm = norm(dforce)
                    torque_norm = norm(dtorque)
                    release_entry_candidate = (
                        previous_force_norm >= args.entry_release_pre_force_n
                        and previous_torque_norm >= args.entry_release_pre_torque_nm
                        and force_norm <= args.entry_release_force_n
                        and torque_norm <= args.entry_release_torque_nm
                        and previous_force_norm - force_norm >= args.entry_release_force_drop_n
                        and previous_torque_norm - torque_norm >= args.entry_release_torque_drop_nm
                    )
                    if release_entry_candidate:
                        print(
                            "候选进洞：检测到中心位姿接触载荷同步卸载；"
                            "保持当前位置进行原地确认。"
                            f" back={back_mm:.1f}mm "
                            f"|dF|={previous_force_norm:.2f}→{force_norm:.2f}N "
                            f"|dT|={previous_torque_norm:.2f}→{torque_norm:.2f}N*m"
                        )
                        release_confirmed = True
                        for confirm_index in range(1, args.entry_release_confirm_samples + 1):
                            time.sleep(args.entry_release_confirm_interval_s)
                            confirm_force, confirm_torque = sample(
                                f"entry_release_confirm_{confirm_index}", cycle, back_mm, centre,
                            )
                            confirm_force_norm = norm(confirm_force)
                            confirm_torque_norm = norm(confirm_torque)
                            if (
                                confirm_force_norm > args.entry_release_force_n
                                or confirm_torque_norm > args.entry_release_torque_nm
                            ):
                                release_confirmed = False
                                print(
                                    "卸载候选未确认：原地载荷回升；继续搜索。"
                                    f" confirm={confirm_index}/{args.entry_release_confirm_samples} "
                                    f"|dF|={confirm_force_norm:.2f}N "
                                    f"|dT|={confirm_torque_norm:.2f}N*m"
                                )
                                break
                        if release_confirmed:
                            log_event("insertion_decision", result="entered_by_centre_release_confirmed",
                                      cycle=cycle, back_mm=round(back_mm, 3))
                            print(
                                "SUCCESS：检测到并确认中心位姿接触载荷同步卸载；"
                                "停止在当前后拉中心位姿，不执行本轮左右摆动。"
                                f" back={back_mm:.1f}mm "
                                f"|dF|={previous_force_norm:.2f}→{force_norm:.2f}N "
                                f"|dT|={previous_torque_norm:.2f}→{torque_norm:.2f}N*m"
                            )
                            return 0
                    fz_entry_candidate = (
                        dforce[2] >= args.entry_fz_rise_n
                        and dforce[2] - previous_force[2] >= args.entry_fz_rise_delta_n
                    )
                    if fz_entry_candidate:
                        print(
                            "候选进洞：检测到后拉中心位姿的正向 Z 力突增；"
                            "保持当前位置进行原地确认。"
                            f" back={back_mm:.1f}mm "
                            f"dFz={previous_force[2]:.2f}→{dforce[2]:.2f}N"
                        )
                        fz_confirmed = True
                        for confirm_index in range(1, args.entry_release_confirm_samples + 1):
                            time.sleep(args.entry_release_confirm_interval_s)
                            confirm_force, _ = sample(
                                f"entry_fz_confirm_{confirm_index}", cycle, back_mm, centre,
                            )
                            if confirm_force[2] < args.entry_fz_rise_n:
                                fz_confirmed = False
                                print(
                                    "Z 力候选未确认：原地正向 dFz 回落；继续搜索。"
                                    f" confirm={confirm_index}/{args.entry_release_confirm_samples} "
                                    f"dFz={confirm_force[2]:.2f}N"
                                )
                                break
                        if fz_confirmed:
                            log_event("insertion_decision", result="entered_by_centre_fz_rise_confirmed",
                                      cycle=cycle, back_mm=round(back_mm, 3))
                            print(
                                "SUCCESS：检测到并确认后拉中心位姿的正向 Z 力突增；"
                                "停止在当前后拉中心位姿，不执行本轮左右摆动。"
                                f" back={back_mm:.1f}mm dFz={dforce[2]:.2f}N"
                            )
                            return 0
                    entered = strong_entry or relaxed_entry
                    if entered:
                        log_event("insertion_decision",
                                  result=("entered_by_strong_fx_ty_release" if strong_entry
                                          else "entered_by_relaxed_fx_ty_release"),
                                  cycle=cycle, back_mm=round(back_mm, 3))
                        print(
                            "SUCCESS：检测到进洞的 Fx/Ty 同步突变卸载"
                            f"（{'强突变' if strong_entry else '中等接触卸载'}）；"
                            "立即停止在后拉中心位姿，不执行本轮左右摆动。"
                            f" back={back_mm:.1f}mm "
                            f"dFx={previous_force[0]:.2f}→{dforce[0]:.2f}N "
                            f"(drop={previous_force[0] - dforce[0]:.2f}N) "
                            f"dTy={previous_torque[1]:.2f}→{dtorque[1]:.2f}N*m "
                            f"(drop={previous_torque[1] - dtorque[1]:.2f}N*m, "
                            f"|current|={abs(dtorque[1]):.2f}N*m)"
                        )
                        return 0
                previous_back_wrench = (dforce, dtorque)
                centre_force, centre_torque = dforce, dtorque

                def wiggle_entered(force: list[float], torque: list[float]) -> bool:
                    return (
                        back_mm >= args.min_entry_back_mm
                        and centre_force[0] >= args.wiggle_entry_pre_fx_n
                        and centre_torque[1] >= args.wiggle_entry_pre_ty_nm
                        and force[0] < 0.0
                        and torque[1] < 0.0
                        and centre_force[0] - force[0] >= args.wiggle_entry_fx_drop_n
                        and centre_torque[1] - torque[1] >= args.wiggle_entry_ty_drop_nm
                    )

                def report_wiggle_entry(event: str, force: list[float], torque: list[float]) -> None:
                    log_event("insertion_decision", result="entered_by_wiggle_fx_ty_sign_reversal",
                              motion=event, cycle=cycle, back_mm=round(back_mm, 3),
                              delta_force_n=force, delta_torque_nm=torque)
                    print(
                        "SUCCESS：检测到摆动进洞的 Fx/Ty 正负突变；"
                        "立即停止，不再执行后续摆动或后拉。"
                        f" event={event} back={back_mm:.1f}mm "
                        f"dFx={centre_force[0]:.2f}→{force[0]:.2f}N "
                        f"(drop={centre_force[0] - force[0]:.2f}N) "
                        f"dTy={centre_torque[1]:.2f}→{torque[1]:.2f}N*m "
                        f"(drop={centre_torque[1] - torque[1]:.2f}N*m)"
                    )

                def wiggle_release_entered(
                    event: str,
                    force: list[float],
                    torque: list[float],
                    target: list[float],
                ) -> bool:
                    """Confirm a centre-contact to lateral-search unload in place."""
                    if back_mm < args.min_entry_back_mm:
                        return False
                    centre_force_norm = norm(centre_force)
                    centre_torque_norm = norm(centre_torque)
                    force_norm = norm(force)
                    torque_norm = norm(torque)
                    if not (
                        centre_force_norm >= args.entry_release_pre_force_n
                        and centre_torque_norm >= args.entry_release_pre_torque_nm
                        and force_norm <= args.entry_release_force_n
                        and torque_norm <= args.entry_release_torque_nm
                        and centre_force_norm - force_norm >= args.entry_release_force_drop_n
                        and centre_torque_norm - torque_norm >= args.entry_release_torque_drop_nm
                    ):
                        return False
                    print(
                        "候选进洞：摆动后检测到中心接触载荷同步卸载；"
                        "保持当前位置进行原地确认。"
                        f" event={event} back={back_mm:.1f}mm "
                        f"|dF|={centre_force_norm:.2f}→{force_norm:.2f}N "
                        f"|dT|={centre_torque_norm:.2f}→{torque_norm:.2f}N*m"
                    )
                    for confirm_index in range(1, args.entry_release_confirm_samples + 1):
                        time.sleep(args.entry_release_confirm_interval_s)
                        confirm_force, confirm_torque = sample(
                            f"entry_release_confirm_{event}_{confirm_index}",
                            cycle,
                            back_mm,
                            target,
                        )
                        confirm_force_norm = norm(confirm_force)
                        confirm_torque_norm = norm(confirm_torque)
                        if (
                            confirm_force_norm > args.entry_release_force_n
                            or confirm_torque_norm > args.entry_release_torque_nm
                        ):
                            print(
                                "摆动卸载候选未确认：原地载荷回升；继续搜索。"
                                f" confirm={confirm_index}/{args.entry_release_confirm_samples} "
                                f"|dF|={confirm_force_norm:.2f}N "
                                f"|dT|={confirm_torque_norm:.2f}N*m"
                            )
                            return False
                    print(
                        "SUCCESS：检测到并确认摆动后的接触载荷同步卸载；"
                        "停止在当前位置，不再执行后续摆动或后拉。"
                        f" event={event} back={back_mm:.1f}mm"
                    )
                    log_event("insertion_decision", result="entered_by_wiggle_release_confirmed",
                              motion=event, cycle=cycle, back_mm=round(back_mm, 3))
                    return True

                left = list(centre)
                left[1] += args.wiggle_half_amplitude_mm / 1000.0
                log_event("motion_start", motion="left_y_positive", cycle=cycle,
                          back_mm=round(back_mm, 3), insertion_decision="searching")
                move_in_active_session(
                    arm, left,
                    linear_speed_mps=linear_speed_mps,
                    linear_accel_mps2=linear_accel_mps2,
                    diagnostic=diagnostic,
                )
                dforce, dtorque = sample("wiggle_y_plus", cycle, back_mm, left)
                if wiggle_release_entered("wiggle_y_plus", dforce, dtorque, left):
                    return 0
                if wiggle_entered(dforce, dtorque):
                    report_wiggle_entry("wiggle_y_plus", dforce, dtorque)
                    return 0

                right = list(centre)
                right[1] -= args.wiggle_half_amplitude_mm / 1000.0
                log_event("motion_start", motion="right_y_negative", cycle=cycle,
                          back_mm=round(back_mm, 3), insertion_decision="searching")
                move_in_active_session(
                    arm, right,
                    linear_speed_mps=linear_speed_mps,
                    linear_accel_mps2=linear_accel_mps2,
                    diagnostic=diagnostic,
                )
                dforce, dtorque = sample("wiggle_y_minus", cycle, back_mm, right)
                if wiggle_release_entered("wiggle_y_minus", dforce, dtorque, right):
                    return 0
                if wiggle_entered(dforce, dtorque):
                    report_wiggle_entry("wiggle_y_minus", dforce, dtorque)
                    return 0

                log_event("motion_start", motion="return_to_centre", cycle=cycle,
                          back_mm=round(back_mm, 3), insertion_decision="searching")
                move_in_active_session(
                    arm, centre,
                    linear_speed_mps=linear_speed_mps,
                    linear_accel_mps2=linear_accel_mps2,
                    diagnostic=diagnostic,
                )
                dforce, dtorque = sample("wiggle_return_centre", cycle, back_mm, centre)
                if wiggle_release_entered("wiggle_return_centre", dforce, dtorque, centre):
                    return 0
                if wiggle_entered(dforce, dtorque):
                    report_wiggle_entry("wiggle_return_centre", dforce, dtorque)
                    return 0
                log_event("insertion_decision", result="not_entered_continue_search",
                          cycle=cycle, back_mm=round(back_mm, 3))
        log_event("insertion_decision", result="not_entered_max_back_reached",
                  back_mm=round(args.max_back_mm, 3))
        raise RuntimeError(f"在最大后拉 {args.max_back_mm:.1f}mm 内未检测到进洞特征。")
    except PlacementContactReached:
        log_event("insertion_decision", result="entered_by_contact_limit")
        return 0
    except Exception as exc:
        log_event("run_error", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        stop_servo_error: Exception | None = None
        if servo_started and arm is not None:
            try:
                try:
                    write_control_snapshot(arm, log_file, "before_stop_servo_pose")
                except Exception as exc:
                    print(f"警告：停止前控制状态记录失败：{exc}", file=sys.stderr)
                arm.stop_servo_pose(2)
                try:
                    write_control_snapshot(arm, log_file, "after_stop_servo_pose")
                except Exception as exc:
                    print(f"警告：停止后控制状态记录失败：{exc}", file=sys.stderr)
            except Exception as exc:
                # A placement must never be reported as successful unless the
                # active ServoPose session was confirmed stopped.
                stop_servo_error = exc
        if servo_guard is not None:
            servo_guard.__exit__(None, None, None)
        if sensor is not None:
            sensor.close()
        if arm is not None:
            arm.close()
        log_event("run_finished", elapsed_s=round(time.monotonic() - run_started, 3))
        log_file.close()
        print(f"运行日志：{output_path}")
        if stop_servo_error is not None:
            raise RuntimeError("停止右臂 ServoPose 失败，拒绝报告放置成功") from stop_servo_error


if __name__ == "__main__":
    script_started = time.monotonic()
    try:
        raise SystemExit(main())
    finally:
        print(f"本次执行总耗时：{time.monotonic() - script_started:.2f} 秒", flush=True)
