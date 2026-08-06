#!/usr/bin/env python3
"""Hole-A wiggle insertion using Planner LIN for every Cartesian segment.

Unlike ``place_hole_a_wiggle.py``, this script sends no SDK Cartesian-servo
commands.  The initial joint return is Planner RRTStar and every 1 mm back/wiggle
segment is a collision-checked Planner LIN trajectory in ``body_base``.  The
lateral search is subdivided and force-checked between waypoints so a peg that
enters during a LIN segment is not carried through and back out of the hole.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_config import load_config  # noqa: E402
from loop_types import ObjectPose  # noqa: E402
from tool.planner_runtime import PlannerRuntime  # noqa: E402
from modules.module_9_right_place.place_hole_a_wiggle import (  # noqa: E402
    INITIAL_RIGHT_JOINTS_DEG,
    mean,
    norm,
    read_wrench,
    vector,
)


DEFAULT_PLANNER_CONFIG = ROOT / "config.hole_a_first_live.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="右臂以 Planner LIN 每后拉 1 mm、左右摆动 2 mm，按力特征停止。"
    )
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--live", action="store_true", help="连接规划器、力传感器并执行真实运动")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--force-host", default="127.0.0.1")
    parser.add_argument("--back-step-mm", type=float, default=1.0)
    parser.add_argument("--wiggle-half-amplitude-mm", type=float, default=1.0)
    parser.add_argument("--max-back-mm", type=float, default=20.0)
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument("--sample-timeout", type=float, default=2.0)
    parser.add_argument("--settle-ms", type=float, default=150.0)
    parser.add_argument("--force-average-samples", type=int, default=5)
    parser.add_argument("--force-sample-interval-ms", type=float, default=20.0)
    parser.add_argument(
        "--planner-lin-sample-step-mm",
        type=float,
        default=0.25,
        help="摆动 LIN 中每隔该距离停下采力；默认 0.25 mm，最大 1 mm",
    )
    parser.add_argument(
        "--entry-confirmations",
        type=int,
        default=1,
        help=(
            "进洞候选的原地力确认次数；默认 1，检测到卸载即停止。"
            "设为 2 或以上时，只在当前位置复采力，不会继续运动"
        ),
    )
    parser.add_argument("--confirmation-interval-ms", type=float, default=50.0)
    parser.add_argument("--arrival-position-tolerance-mm", type=float, default=0.4)
    parser.add_argument("--arrival-orientation-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--arrival-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-force-delta-n", type=float, default=20.0)
    parser.add_argument("--max-torque-delta-nm", type=float, default=5.0)
    # Planner LIN's force sign can differ from ServoPose's taught path, so use
    # force/torque magnitudes.  The sequencing intentionally matches the
    # working ServoPose version: compare this back-centre reading directly to
    # the preceding back-centre reading, then stop before this cycle wiggles.
    parser.add_argument("--entry-pre-fx-n", type=float, default=6.0)
    parser.add_argument("--entry-pre-ty-nm", type=float, default=1.2)
    parser.add_argument("--entry-fx-drop-n", type=float, default=1.8)
    parser.add_argument("--entry-ty-drop-nm", type=float, default=0.5)
    parser.add_argument("--entry-ty-abs-nm", type=float, default=1.0)
    # Direction-independent centre-to-wiggle contact unloading.
    parser.add_argument("--wiggle-entry-pre-fx-n", type=float, default=5.0)
    parser.add_argument("--wiggle-entry-pre-ty-nm", type=float, default=1.2)
    parser.add_argument("--wiggle-entry-fx-drop-n", type=float, default=2.0)
    parser.add_argument("--wiggle-entry-ty-drop-nm", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.live != args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    positive = (
        "back_step_mm", "wiggle_half_amplitude_mm", "max_back_mm", "sample_timeout",
        "settle_ms", "force_sample_interval_ms", "confirmation_interval_ms",
        "planner_lin_sample_step_mm",
        "arrival_position_tolerance_mm", "arrival_orientation_tolerance_deg",
        "arrival_timeout_s",
        "max_force_delta_n", "max_torque_delta_nm",
        "entry_pre_fx_n", "entry_pre_ty_nm", "entry_fx_drop_n", "entry_ty_drop_nm",
        "entry_ty_abs_nm", "wiggle_entry_pre_fx_n", "wiggle_entry_pre_ty_nm",
        "wiggle_entry_fx_drop_n", "wiggle_entry_ty_drop_nm",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    if args.baseline_samples < 1:
        parser.error("--baseline-samples 必须至少为 1")
    if not 100.0 <= args.settle_ms <= 200.0:
        parser.error("--settle-ms 必须在 100 到 200 ms 之间")
    if args.force_average_samples < 3:
        parser.error("--force-average-samples 必须至少为 3")
    if args.entry_confirmations < 1:
        parser.error("--entry-confirmations 必须至少为 1")
    if args.back_step_mm > 2.0 or args.wiggle_half_amplitude_mm > 2.0:
        parser.error("为安全起见，单步后拉和单侧摆幅均不得超过 2 mm")
    if args.planner_lin_sample_step_mm > 1.0:
        parser.error("--planner-lin-sample-step-mm 不得超过 1 mm")
    return args


def planner_pose(response: dict, frame_id: str) -> ObjectPose:
    """Parse Planner's body_base -> tool response for a LIN target."""
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    try:
        position = [float(translation[axis]) for axis in "xyz"]
        orientation = [float(rotation[axis]) for axis in "xyzw"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"规划器末端 TF 格式无效: {response}") from exc
    if not all(math.isfinite(value) for value in position + orientation):
        raise RuntimeError(f"规划器末端 TF 含非有限数: {response}")
    return ObjectPose(frame_id=frame_id, position=position, orientation=orientation)


def current_right_tool_pose(planner: PlannerRuntime) -> ObjectPose:
    response = planner.get_tool_pose("right")
    if response is None:
        raise RuntimeError("规划器没有返回右臂当前末端位姿")
    return planner_pose(response, planner.config.base_frame)


def pose_error(target: ObjectPose, actual: ObjectPose) -> tuple[float, float]:
    """Return Cartesian translation (mm) and quaternion angle (deg)."""
    translation_mm = 1000.0 * math.sqrt(
        sum(
            (target.position[index] - actual.position[index]) ** 2
            for index in range(3)
        )
    )
    target_norm = math.sqrt(sum(value * value for value in target.orientation))
    actual_norm = math.sqrt(sum(value * value for value in actual.orientation))
    if target_norm <= 0.0 or actual_norm <= 0.0:
        raise RuntimeError("规划器返回零长度四元数")
    dot = abs(
        sum(
            target.orientation[index] * actual.orientation[index]
            for index in range(4)
        )
        / (target_norm * actual_norm)
    )
    rotation_deg = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))
    return translation_mm, rotation_deg


def wait_for_pose_target(
    planner: PlannerRuntime, target: ObjectPose, args: argparse.Namespace, label: str
) -> tuple[float, float]:
    """Wait for the asynchronous trajectory executor to reach a LIN endpoint."""
    deadline = time.monotonic() + args.arrival_timeout_s
    last_translation_mm = math.inf
    last_rotation_deg = math.inf
    while time.monotonic() < deadline:
        actual = current_right_tool_pose(planner)
        last_translation_mm, last_rotation_deg = pose_error(target, actual)
        if (
            last_translation_mm <= args.arrival_position_tolerance_mm
            and last_rotation_deg <= args.arrival_orientation_tolerance_deg
        ):
            return last_translation_mm, last_rotation_deg
        time.sleep(0.02)
    raise RuntimeError(
        f"{label} 未真正到位：位置误差={last_translation_mm:.3f}mm"
        f"（限{args.arrival_position_tolerance_mm:.3f}mm），"
        f"姿态误差={last_rotation_deg:.3f}°"
        f"（限{args.arrival_orientation_tolerance_deg:.3f}°）"
    )


def contact_unloaded(
    reference_force: list[float],
    reference_torque: list[float],
    current_force: list[float],
    current_torque: list[float],
    *,
    pre_fx_n: float,
    pre_ty_nm: float,
    fx_drop_n: float,
    ty_drop_nm: float,
    current_ty_abs_nm: float | None = None,
) -> bool:
    """Detect unloading by magnitudes so either sensor sign is accepted."""
    reference_fx = abs(reference_force[0])
    reference_ty = abs(reference_torque[1])
    current_fx = abs(current_force[0])
    current_ty = abs(current_torque[1])
    torque_unloaded = reference_ty - current_ty >= ty_drop_nm
    if current_ty_abs_nm is not None:
        torque_unloaded = torque_unloaded or current_ty <= current_ty_abs_nm
    return (
        reference_fx >= pre_fx_n
        and reference_ty >= pre_ty_nm
        and reference_fx - current_fx >= fx_drop_n
        and torque_unloaded
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.planner_config, live=args.live)
    if not args.live:
        print("dry-run：不会连接硬件或执行轨迹。")
        print(f"Planner RRTStar 起点关节角={INITIAL_RIGHT_JOINTS_DEG}")
        print(
            f"后拉/摆动均使用 Planner LIN：X -= {args.back_step_mm:.3f}mm，"
            f"Y=中心±{args.wiggle_half_amplitude_mm:.3f}mm，最大后拉 {args.max_back_mm:.1f}mm。"
        )
        return 0

    output_path = args.output or Path("logs") / f"hole_a_wiggle2_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from parm_sdk import ForceSensor

    planner = PlannerRuntime(config)
    sensor = None
    try:
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        print(
            "Planner move_j 回起点："
            f"目标={[round(value, 3) for value in INITIAL_RIGHT_JOINTS_DEG]}。"
        )
        planner.move_right_joints(list(INITIAL_RIGHT_JOINTS_DEG))
        centre = current_right_tool_pose(planner)
        print(
            "已到固定关节起点；Planner body_base 后拉中心 xyz="
            f"{[round(value, 6) for value in centre.position]}。"
        )
        sensor = ForceSensor(host=args.force_host, hand=2)
        print("已到起始位姿；请勿接触工件，正在稳定并采集右腕力基线……")
        time.sleep(0.5)
        baseline_samples = []
        for index in range(args.baseline_samples):
            baseline_samples.append(read_wrench(sensor, args.sample_timeout))
            if index + 1 < args.baseline_samples:
                time.sleep(args.force_sample_interval_ms / 1000.0)
        force_base = mean([force for force, _ in baseline_samples])
        torque_base = mean([torque for _, torque in baseline_samples])
        print(f"起始位姿力基线 F={force_base} N，T={torque_base} N*m")

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "elapsed_s", "event", "cycle", "back_mm",
                "target_x_m", "target_y_m", "target_z_m", "target_qx", "target_qy", "target_qz", "target_qw",
                "actual_x_m", "actual_y_m", "actual_z_m", "actual_qx", "actual_qy", "actual_qz", "actual_qw",
                "fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm",
                "dfx_n", "dfy_n", "dfz_n", "dtx_nm", "dty_nm", "dtz_nm", "delta_force_norm_n", "delta_torque_norm_nm",
            ])
            started = time.monotonic()

            def sample(event: str, cycle: int, back_mm: float, target: ObjectPose) -> tuple[list[float], list[float]]:
                wrench_samples = []
                for index in range(args.force_average_samples):
                    raw_force, raw_torque = read_wrench(
                        sensor, args.sample_timeout
                    )
                    raw_dforce = [
                        value - base for value, base in zip(raw_force, force_base)
                    ]
                    raw_dtorque = [
                        value - base for value, base in zip(raw_torque, torque_base)
                    ]
                    raw_force_norm = norm(raw_dforce)
                    raw_torque_norm = norm(raw_dtorque)
                    if (
                        raw_force_norm > args.max_force_delta_n
                        or raw_torque_norm > args.max_torque_delta_nm
                    ):
                        raise RuntimeError(
                            "安全停止：单帧相对力/扭矩超限；"
                            f"|dF|={raw_force_norm:.2f}N"
                            f"（限 {args.max_force_delta_n:.2f}），"
                            f"|dT|={raw_torque_norm:.2f}N*m"
                            f"（限 {args.max_torque_delta_nm:.2f}）。"
                        )
                    wrench_samples.append((raw_force, raw_torque))
                    if index + 1 < args.force_average_samples:
                        time.sleep(args.force_sample_interval_ms / 1000.0)
                force = mean([row[0] for row in wrench_samples])
                torque = mean([row[1] for row in wrench_samples])
                dforce = [value - base for value, base in zip(force, force_base)]
                dtorque = [value - base for value, base in zip(torque, torque_base)]
                actual = current_right_tool_pose(planner)
                force_norm, torque_norm = norm(dforce), norm(dtorque)
                writer.writerow([
                    f"{time.monotonic() - started:.6f}", event, cycle, f"{back_mm:.3f}",
                    *[f"{value:.9f}" for value in target.position + target.orientation],
                    *[f"{value:.9f}" for value in actual.position + actual.orientation],
                    *[f"{value:.9f}" for value in force + torque + dforce + dtorque],
                    f"{force_norm:.9f}", f"{torque_norm:.9f}",
                ])
                output_file.flush()
                print(
                    f"[{event}] back={back_mm:.1f}mm "
                    f"{args.force_average_samples}帧平均 "
                    f"dF={dforce} |dF|={force_norm:.2f}N "
                    f"dT={dtorque} |dT|={torque_norm:.2f}N*m"
                )
                return dforce, dtorque

            def confirm_entry(
                event: str,
                cycle: int,
                back_mm: float,
                target: ObjectPose,
                predicate,
            ) -> bool:
                """Optionally re-sample at the same pose; never move while deciding."""
                if args.entry_confirmations == 1:
                    print(
                        f"[entry-confirmed] {event} 首帧卸载已满足；"
                        "按安全优先策略立即停止，不再执行下一段轨迹。"
                    )
                    return True
                for confirmation in range(2, args.entry_confirmations + 1):
                    time.sleep(args.confirmation_interval_ms / 1000.0)
                    dforce, dtorque = sample(
                        f"{event}_confirm_{confirmation}",
                        cycle,
                        back_mm,
                        target,
                    )
                    if not predicate(dforce, dtorque):
                        print(
                            f"[entry-rejected] {event} 第{confirmation}次"
                            "连续确认不满足卸载条件，继续搜索。"
                        )
                        return False
                print(
                    f"[entry-confirmed] {event} 连续"
                    f"{args.entry_confirmations}次平均读数均满足卸载条件。"
                )
                return True

            sample("initial", 0, 0.0, centre)
            start_x, start_y = centre.position[0], centre.position[1]
            previous_centre_wrench: tuple[list[float], list[float]] | None = None
            cycles = math.ceil(args.max_back_mm / args.back_step_mm)

            def lin_move(target: ObjectPose, label: str) -> None:
                print(f"[planner-lin] {label} -> xyz={[round(value, 6) for value in target.position]}")
                planner.move_cartesian_pose("right", target, planner="lin")
                translation_mm, rotation_deg = wait_for_pose_target(
                    planner, target, args, label
                )
                print(
                    f"[planner-lin] {label} 已到位："
                    f"{translation_mm:.3f}mm/{rotation_deg:.3f}°；"
                    f"稳定 {args.settle_ms:.0f}ms 后采力。"
                )
                time.sleep(args.settle_ms / 1000.0)

            for cycle in range(1, cycles + 1):
                back_mm = min(cycle * args.back_step_mm, args.max_back_mm)
                centre = ObjectPose(
                    frame_id=config.base_frame,
                    position=[start_x - back_mm / 1000.0, start_y, centre.position[2]],
                    orientation=list(centre.orientation),
                )
                lin_move(centre, "back_centre")
                dforce, dtorque = sample("back_centre", cycle, back_mm, centre)
                if previous_centre_wrench is not None:
                    reference_force, reference_torque = previous_centre_wrench

                    def centre_entry(
                        force: list[float], torque: list[float]
                    ) -> bool:
                        return contact_unloaded(
                            reference_force,
                            reference_torque,
                            force,
                            torque,
                            pre_fx_n=args.entry_pre_fx_n,
                            pre_ty_nm=args.entry_pre_ty_nm,
                            fx_drop_n=args.entry_fx_drop_n,
                            ty_drop_nm=args.entry_ty_drop_nm,
                            current_ty_abs_nm=args.entry_ty_abs_nm,
                        )

                    if centre_entry(dforce, dtorque):
                        if confirm_entry(
                            "back_centre",
                            cycle,
                            back_mm,
                            centre,
                            centre_entry,
                        ):
                            print(
                                "SUCCESS：相邻后拉中心检测到"
                                "与方向无关的|dFx|/|dTy|同步卸载；"
                                f"back={back_mm:.1f}mm，停止。"
                            )
                            return 0
                        raise RuntimeError(
                            "检测到中心卸载候选，但连续确认未通过；"
                            "为避免把可能已进洞的工件带出，保持当前位置并停止，"
                            "不再执行任何摆动。"
                        )
                # Always retain the immediately preceding centre frame.  Do
                # not gate this on the current frame still being "contact":
                # entering the hole is precisely the transition that removes
                # that contact load.
                previous_centre_wrench = (list(dforce), list(dtorque))
                centre_force, centre_torque = dforce, dtorque

                def wiggle_entry(force: list[float], torque: list[float]) -> bool:
                    return (
                        contact_unloaded(
                            centre_force,
                            centre_torque,
                            force,
                            torque,
                            pre_fx_n=args.wiggle_entry_pre_fx_n,
                            pre_ty_nm=args.wiggle_entry_pre_ty_nm,
                            fx_drop_n=args.wiggle_entry_fx_drop_n,
                            ty_drop_nm=args.wiggle_entry_ty_drop_nm,
                        )
                    )

                def move_wiggle_leg_and_watch(
                    source: ObjectPose,
                    destination: ObjectPose,
                    event: str,
                ) -> bool:
                    """Move one lateral leg in short LIN pieces and stop on entry.

                    Planner execution does not expose a force callback during a
                    single trajectory.  Sampling only after a 2 mm Y+ -> Y-
                    path can therefore miss a short insertion completely.  A
                    short, checked LIN piece bounds any travel after insertion
                    to ``planner_lin_sample_step_mm``.
                    """
                    distance_m = math.sqrt(
                        sum(
                            (destination.position[axis] - source.position[axis]) ** 2
                            for axis in range(3)
                        )
                    )
                    steps = max(
                        1,
                        math.ceil(
                            distance_m
                            / (args.planner_lin_sample_step_mm / 1000.0)
                        ),
                    )
                    for index in range(1, steps + 1):
                        ratio = index / steps
                        waypoint = ObjectPose(
                            config.base_frame,
                            [
                                source.position[axis]
                                + ratio
                                * (destination.position[axis] - source.position[axis])
                                for axis in range(3)
                            ],
                            [
                                source.orientation[axis]
                                + ratio
                                * (destination.orientation[axis] - source.orientation[axis])
                                for axis in range(4)
                            ],
                        )
                        lin_move(waypoint, f"{event}_{index}of{steps}")
                        leg_force, leg_torque = sample(
                            f"{event}_{index}of{steps}", cycle, back_mm, waypoint
                        )
                        if wiggle_entry(leg_force, leg_torque):
                            if confirm_entry(
                                event,
                                cycle,
                                back_mm,
                                waypoint,
                                wiggle_entry,
                            ):
                                print(
                                    "SUCCESS：摆动微段检测到与方向无关的"
                                    "|dFx|/|dTy|卸载；"
                                    f"event={event}，back={back_mm:.1f}mm，停止。"
                                )
                                return True
                            raise RuntimeError(
                                "摆动微段出现卸载候选但连续确认未通过；"
                                "为避免带出，保持当前位置停止。"
                            )
                    return False

                left = ObjectPose(config.base_frame, [centre.position[0], start_y + args.wiggle_half_amplitude_mm / 1000.0, centre.position[2]], list(centre.orientation))
                if move_wiggle_leg_and_watch(centre, left, "wiggle_y_plus"):
                    return 0

                right = ObjectPose(config.base_frame, [centre.position[0], start_y - args.wiggle_half_amplitude_mm / 1000.0, centre.position[2]], list(centre.orientation))
                # Do not traverse Y+ directly to Y- as one 2 mm trajectory:
                # return to centre first, then search the other side.
                if move_wiggle_leg_and_watch(left, centre, "wiggle_y_plus_return"):
                    return 0
                if move_wiggle_leg_and_watch(centre, right, "wiggle_y_minus"):
                    return 0
                if move_wiggle_leg_and_watch(right, centre, "wiggle_return_centre"):
                    return 0
        raise RuntimeError(f"在最大后拉 {args.max_back_mm:.1f}mm 内未检测到进洞特征。")
    finally:
        if sensor is not None:
            sensor.close()
        planner.close()
        print(f"CSV 记录：{output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
