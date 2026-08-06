#!/usr/bin/env python3
"""验证右臂 ServoPose 与左臂规划器轨迹是否能无锁并发执行。

本脚本独立于总控。右臂在 SDK ``world`` 坐标系沿 +Y 缓慢推送 10 cm，左臂在
规划器 ``body_base`` 坐标系沿 +Z 缓慢规划并执行 10 cm。默认依次执行两轮：

1. 右臂先启动 ServoPose，再启动左臂规划器轨迹；
2. 左臂先开始规划器轨迹，再启动右臂 ServoPose。

每轮结束后，右臂在仍有效的 ServoPose 会话内回到起点，左臂在 ServoPose
停止后通过规划器回到起点，避免两轮测试累积位移。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import threading
import time

from scipy.spatial.transform import Rotation


LOOP_ROOT = Path(__file__).resolve().parents[2]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))

from loop_config import LoopConfig
from loop_types import ObjectPose
from tool.planner_runtime import PlannerRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--execute-url", default="http://127.0.0.1:5667/middle/waypoint/execute")
    parser.add_argument(
        "--mode",
        choices=("right-first", "left-first", "both"),
        default="both",
        help="默认依次执行两个顺序测试",
    )
    parser.add_argument("--right-distance-m", type=float, default=0.10,
                        help="右臂 world +Y 位移，默认 0.10 m")
    parser.add_argument("--left-distance-m", type=float, default=0.10,
                        help="左臂 body_base +Z 位移，默认 0.10 m")
    parser.add_argument("--left-speed", type=float, default=0.02,
                        help="左臂规划器速度比例，默认 0.02，使右臂可在其运动中启动")
    parser.add_argument("--servo-hz", type=float, default=50.0)
    parser.add_argument("--right-speed-mps", type=float, default=0.01,
                        help="右臂 ServoPose 直线推送速度，默认 0.01 m/s")
    parser.add_argument("--left-first-delay-s", type=float, default=2.0,
                        help="左臂执行启动后等待多久再启动右臂 ServoPose，默认 2 秒")
    parser.add_argument("--right-first-lead-s", type=float, default=2.0,
                        help="右臂开始非零 +Y 推送后，等待多久再启动左臂，默认 2 秒")
    parser.add_argument("--hold-timeout-s", type=float, default=60.0,
                        help="右臂到达 +Y 目标后最多保持 ServoPose 的时间")
    parser.add_argument("--keep-endpoints", action="store_true",
                        help="不自动回到每轮开始位，仅用于已确认安全的检查")
    parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.005 <= args.right_distance_m <= 0.15:
        parser.error("--right-distance-m 必须在 0.005 到 0.15 m 之间")
    if not 0.005 <= args.left_distance_m <= 0.15:
        parser.error("--left-distance-m 必须在 0.005 到 0.15 m 之间")
    if not 0.01 <= args.left_speed <= 1.0:
        parser.error("--left-speed 必须在 0.01 到 1.0 之间")
    if not 10.0 <= args.servo_hz <= 100.0:
        parser.error("--servo-hz 必须在 10 到 100 Hz 之间")
    if (
        args.right_speed_mps <= 0.0
        or args.left_first_delay_s < 0.0
        or args.right_first_lead_s < 0.0
    ):
        parser.error("速度必须为正，延迟不能为负")
    return args


def confirm(args: argparse.Namespace, message: str) -> None:
    if args.auto_confirm:
        print(message + " [自动确认]", flush=True)
        return
    input(message)


def planner_config(args: argparse.Namespace) -> LoopConfig:
    return LoopConfig(
        planner_host=args.planner_host,
        execute_url=args.execute_url,
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.left_speed),
        max_acceleration=float(args.left_speed),
        idle_poll_seconds=0.2,
        live=not args.dry_run,
        poses={},
    )


def pose_from_transform(response: dict) -> ObjectPose:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    position = [translation.get(axis) for axis in "xyz"]
    orientation = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in position + orientation):
        raise RuntimeError(f"规划器 TF 格式无效: {response}")
    return ObjectPose(
        frame_id="body_base",
        position=[float(value) for value in position],
        orientation=[float(value) for value in orientation],
    )


def right_pose_command(arm: object) -> list[float]:
    state = arm.get_state() or {}  # type: ignore[attr-defined]
    raw = ((state.get("state") or {}).get("right") or {}).get("eef_pose")
    if not isinstance(raw, dict):
        raise RuntimeError(f"无法读取右臂末端位姿: {raw!r}")
    xyz = raw.get("xyz")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise RuntimeError(f"右臂末端 XYZ 无效: {raw!r}")
    rotvec = None
    for key in ("rotvec", "axis_angle", "rxryrz", "rotation_vector"):
        value = raw.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            rotvec = [float(item) for item in value]
            break
    if rotvec is None:
        rpy = raw.get("rpy")
        if not isinstance(rpy, (list, tuple)) or len(rpy) != 3:
            raise RuntimeError(f"右臂末端旋转无效: {raw!r}")
        rotvec = Rotation.from_euler("xyz", rpy).as_rotvec().tolist()
    pose = [float(value) for value in xyz] + rotvec
    if not all(math.isfinite(value) for value in pose):
        raise RuntimeError(f"右臂末端位姿含非有限值: {pose}")
    return pose


def right_task_state(arm: object) -> str:
    """Return the current right-arm task state from one fresh SDK status frame."""
    state = arm.get_state() or {}  # type: ignore[attr-defined]
    right = ((state.get("state") or {}).get("right") or {})
    return str(right.get("task_state") or "").strip().lower()


def wait_right_task_stopped(arm: object, timeout_s: float = 3.0) -> None:
    """Match the proven placement flow: do not start ServoPose over an old task."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if right_task_state(arm) != "running":
            return
        time.sleep(0.05)
    raise RuntimeError("cancel_task(2) 后右臂 task_state 仍为 running，拒绝启动 ServoPose")


def cancel_both(arm_host: str) -> None:
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(f"[cleanup] 无法导入 parm_sdk: {exc}", flush=True)
        return
    for arm_id in (1, 2):
        arm = None
        try:
            arm = Arm(host=arm_host, status_port=None)
            arm.cancel_task(arm_id)
            print(f"[cleanup] 已发送 cancel_task({arm_id})", flush=True)
        except Exception as exc:
            print(f"[cleanup] cancel_task({arm_id}) 失败: {exc}", flush=True)
        finally:
            if arm is not None:
                arm.close()


def run_right_servo(
    args: argparse.Namespace,
    *,
    started: threading.Event,
    other_finished: threading.Event,
    errors: list[BaseException],
) -> None:
    """在一个右臂 ServoPose 会话内 +Y 运动、保持、可选回到起点。"""
    arm = None
    servo_started = False
    try:
        from parm_sdk import Arm

        arm = Arm(host=args.arm_host)
        state = arm.get_state() or {}
        right_state = ((state.get("state") or {}).get("right") or {})
        if not bool(right_state.get("enabled", False)):
            print("[right-servo] 右臂未使能，正在调用 enable()。", flush=True)
            arm.enable()
            time.sleep(0.3)
        origin = right_pose_command(arm)
        target = origin.copy()
        target[1] += float(args.right_distance_m)
        print(
            "[right-servo] SDK world: "
            f"origin={['%.6f' % value for value in origin[:3]]}, "
            f"target +Y={target[1]:.6f} m",
            flush=True,
        )
        arm.cancel_task(2)
        wait_right_task_stopped(arm)
        arm.start_servo_pose(
            2,
            pose_type="absolute",
            frame="world",
            tcp=[0.0] * 6,
            input_frequency_hz=float(args.servo_hz),
        )
        servo_started = True
        arm.push_servo_pose(2, origin)

        period = 1.0 / float(args.servo_hz)
        steps = max(2, math.ceil(
            float(args.right_distance_m) / float(args.right_speed_mps) * args.servo_hz
        ))
        deadline = time.monotonic()
        for index in range(1, steps + 1):
            alpha = index / float(steps)
            pose = [
                origin[axis] + alpha * (target[axis] - origin[axis])
                for axis in range(6)
            ]
            arm.push_servo_pose(2, pose)
            if index == 1:
                # Signal only after a non-zero Cartesian target was actually sent.
                started.set()
            if index == 1 or index % max(1, int(args.servo_hz)) == 0 or index == steps:
                actual = right_pose_command(arm)
                print(
                    f"[right-servo] push {index}/{steps}: actual world Y="
                    f"{actual[1]:.6f}m, deltaY={actual[1] - origin[1]:+.4f}m",
                    flush=True,
                )
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
        print(
            f"[right-servo] 已推送到 +Y {args.right_distance_m * 100.0:.1f}cm，"
            "保持 ServoPose 等待左臂结束。",
            flush=True,
        )
        hold_deadline = time.monotonic() + float(args.hold_timeout_s)
        while not other_finished.is_set() and time.monotonic() < hold_deadline:
            arm.push_servo_pose(2, target)
            time.sleep(period)
        if not other_finished.is_set():
            raise TimeoutError("右臂 ServoPose 等待左臂结束超时")

        actual_at_target = right_pose_command(arm)
        moved_y = actual_at_target[1] - origin[1]
        if abs(moved_y) < 0.01:
            raise RuntimeError(
                "右臂 ServoPose 已完成推送但实际 Y 位移不足 1cm；"
                f"origin_y={origin[1]:.6f}, actual_y={actual_at_target[1]:.6f}, "
                f"task_state={right_task_state(arm)!r}。"
            )

        if not args.keep_endpoints:
            for index in range(1, steps + 1):
                alpha = index / float(steps)
                pose = [
                    target[axis] + alpha * (origin[axis] - target[axis])
                    for axis in range(6)
                ]
                arm.push_servo_pose(2, pose)
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
            print("[right-servo] 已在 ServoPose 会话内回到本轮起点。", flush=True)
    except BaseException as exc:
        errors.append(exc)
    finally:
        if servo_started and arm is not None:
            try:
                arm.stop_servo_pose(2)
                print("[right-servo] stop_servo_pose(2) 已完成。", flush=True)
            except Exception as exc:
                errors.append(exc)
        if arm is not None:
            arm.close()


def plan_and_execute_left(
    args: argparse.Namespace,
    planner: PlannerRuntime,
    *,
    execution_started: threading.Event,
    finished: threading.Event,
    errors: list[BaseException],
) -> ObjectPose | None:
    """规划并执行左臂 body_base +Z，再把结果交给调用方决定是否回程。"""
    try:
        transform = planner.lookup_transform("body_base", "left_hand_transfer_end")
        if transform is None:
            return None
        origin = pose_from_transform(transform)
        target = ObjectPose(
            frame_id="body_base",
            position=[
                origin.position[0],
                origin.position[1],
                origin.position[2] + args.left_distance_m,
            ],
            orientation=origin.orientation.copy(),
        )
        print(
            "[left-planner] body_base: "
            f"origin={['%.6f' % value for value in origin.position]}, "
            f"target +Z={target.position[2]:.6f} m",
            flush=True,
        )
        result = planner.plan_cartesian_pose("left", target, planner="lin")
        print("[left-planner] 规划完成，开始执行左臂 LIN 轨迹。", flush=True)
        execution_started.set()
        if result is not None:
            planner.execute_planned_result(result)
        print(
            f"[left-planner] 左臂 +Z {args.left_distance_m * 100.0:.1f}cm 轨迹执行返回。",
            flush=True,
        )
        return origin
    except BaseException as exc:
        errors.append(exc)
        # Unblock the phase coordinator so a planning failure is reported as-is,
        # rather than being misreported as an execution-start timeout.
        execution_started.set()
        return None
    finally:
        finished.set()


def run_phase(args: argparse.Namespace, name: str) -> None:
    right_started = threading.Event()
    left_execution_started = threading.Event()
    left_finished = threading.Event()
    errors: list[BaseException] = []
    planner = PlannerRuntime(planner_config(args))
    right_thread = None
    left_thread = None
    left_origin: list[ObjectPose | None] = [None]

    def left_worker() -> None:
        left_origin[0] = plan_and_execute_left(
            args,
            planner,
            execution_started=left_execution_started,
            finished=left_finished,
            errors=errors,
        )

    try:
        planner.open(required_frames={"body_base", "left_hand_transfer_end"})
        if name == "right-first":
            right_thread = threading.Thread(
                target=run_right_servo,
                kwargs={
                    "args": args,
                    "started": right_started,
                    "other_finished": left_finished,
                    "errors": errors,
                },
                name="right-servo-first",
            )
            right_thread.start()
            if not right_started.wait(timeout=10.0):
                if errors:
                    raise RuntimeError(f"右臂 ServoPose 启动失败: {errors[0]}") from errors[0]
                raise TimeoutError("右臂 ServoPose 未在10秒内启动")
            print(
                f"[phase] 右臂已开始非零 +Y 推送；先独立运行 "
                f"{args.right_first_lead_s:.1f}s，再启动左臂规划器。",
                flush=True,
            )
            time.sleep(float(args.right_first_lead_s))
            left_thread = threading.Thread(target=left_worker, name="left-planner-second")
            left_thread.start()
            if not left_execution_started.wait(timeout=40.0):
                raise TimeoutError("左臂未在40秒内完成规划并开始执行")
            if errors:
                raise RuntimeError(f"左臂规划或执行启动失败: {errors[0]}") from errors[0]
            if not right_thread.is_alive():
                raise RuntimeError("右臂 ServoPose 在线程启动左臂轨迹前已退出；本轮没有实际并发")
        else:
            left_thread = threading.Thread(target=left_worker, name="left-planner-first")
            left_thread.start()
            if not left_execution_started.wait(timeout=40.0):
                raise TimeoutError("左臂未在40秒内完成规划并开始执行")
            if errors:
                raise RuntimeError(f"左臂规划或执行启动失败: {errors[0]}") from errors[0]
            time.sleep(float(args.left_first_delay_s))
            right_thread = threading.Thread(
                target=run_right_servo,
                kwargs={
                    "args": args,
                    "started": right_started,
                    "other_finished": left_finished,
                    "errors": errors,
                },
                name="right-servo-second",
            )
            right_thread.start()
            if not right_started.wait(timeout=10.0):
                if errors:
                    raise RuntimeError(f"右臂 ServoPose 启动失败: {errors[0]}") from errors[0]
                raise TimeoutError("右臂 ServoPose 未在10秒内启动")
            if left_finished.is_set():
                raise RuntimeError(
                    "左臂轨迹已在右臂 ServoPose 启动前结束；本轮没有实际并发"
                )

        left_thread.join()
        left_finished.set()
        right_thread.join()
        if errors:
            raise RuntimeError(f"{name} 并发测试失败: {errors[0]}") from errors[0]

        if not args.keep_endpoints and left_origin[0] is not None:
            print("[left-planner] 右臂 ServoPose 已停止，规划左臂回本轮起点。", flush=True)
            planner.move_cartesian_pose("left", left_origin[0], planner="lin")
            print("[left-planner] 左臂已下发回程。", flush=True)
        print(f"[phase] {name} 完成。", flush=True)
    finally:
        left_finished.set()
        if left_thread is not None and left_thread.is_alive():
            left_thread.join(timeout=2.0)
        if right_thread is not None and right_thread.is_alive():
            right_thread.join(timeout=5.0)
        planner.close()


def main() -> None:
    args = parse_args()
    # This test must bypass the production exclusion to measure actual coexistence.
    os.environ.pop("PARM_LEFT_RIGHT_SERVO_LOCK", None)
    os.environ["PARM_ARM_HOST"] = args.arm_host
    phases = (
        ("right-first", "测试 1/2：右臂先 ServoPose，随后左臂规划器执行"),
        ("left-first", "测试 2/2：左臂先规划器执行，随后右臂 ServoPose"),
    )
    print("=== 右臂 ServoPose 与左臂规划器并发测试 ===", flush=True)
    print("本脚本已禁用生产互斥锁；不要与 simple_dual_arm_loop.py 同时运行。", flush=True)
    print(
        f"右臂：SDK world +Y {args.right_distance_m:.3f}m；"
        f"左臂：planner body_base +Z {args.left_distance_m:.3f}m，"
        f"speed={args.left_speed:.3f}。",
        flush=True,
    )
    if args.dry_run:
        for phase, description in phases:
            if args.mode in {"both", phase}:
                print(f"[dry-run] {description}", flush=True)
        return
    try:
        for phase, description in phases:
            if args.mode not in {"both", phase}:
                continue
            confirm(args, f"{description}。确认两臂路径和周围空间安全后按回车开始：")
            run_phase(args, phase)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止 ServoPose 并取消两臂任务。", flush=True)
        cancel_both(args.arm_host)
        raise
    except BaseException:
        cancel_both(args.arm_host)
        raise


if __name__ == "__main__":
    main()
