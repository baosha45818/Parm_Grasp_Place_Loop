#!/usr/bin/env python3
"""Module 8-1: observe at pose 3 and directly grasp the adjusted Q1 workpiece.

Module 7 is responsible for any required reorientation. This module never
rotates the workpiece: after a failed or unplannable direct Q1 grasp, it stops
and asks Module 7 to adjust the workpiece again.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MODULE_DIR = Path(__file__).resolve().parent
LOOP_ROOT = MODULE_DIR.parents[2]
MODULE_4_2_ROOT = (
    MODULE_DIR.parents[1]
    / "module_4_head_judge/module_4_2_pose_estimation"
)
MODULE_4_2_SCRIPT = MODULE_4_2_ROOT / "run_pose_estimation.py"
MODULE_4_2_PYTHON = MODULE_4_2_ROOT / ".venv/bin/python"
MODULE_4_2_OUTPUT_DIR = LOOP_ROOT / "output/4_2"
MODULE_4_2_NO_WORKPIECE_EXIT_CODE = 2
PUSH_ROOT = MODULE_DIR / "vendor/push_t"
sys.path.insert(0, str(PUSH_ROOT))
from move_by_joints import ensure_ready, load_pose_file  # noqa: E402
from right_console_logger import console_log  # noqa: E402
from right_trajectory_recorder import record_plan, start_run  # noqa: E402

GRASP_SCRIPT = PUSH_ROOT / "push_workpiece_to_q1.py"
MOVE_SCRIPT = PUSH_ROOT / "move_by_joints.py"
POSE_DIR = PUSH_ROOT / "saved_poses"
Q1_GRASP_YAW_MIN_DEG = 15.0
Q1_GRASP_YAW_MAX_DEG = 75.0
RIGHT_ARM_JOINT_NAMES = [f"right_arm_joint{index}" for index in range(1, 8)]
INITIAL_JOINT_SPEED = 0.2
TRANSFER_MAX_VEL = 0.1
TRANSFER_MAX_ACC = 0.1
GRASP_MAX_VEL = 0.1
GRASP_MAX_ACC = 0.1
Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE = 20
Q1_EMPTY_GRASP_EXIT_CODE = 21
Q1_NARROW_GRASP_EXIT_CODE = 22
# The adapter treats this as a recoverable request to return to Module 7.
Q1_REORIENTATION_REQUIRED_EXIT_CODE = 23
MAX_DIRECT_GRASP_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", type=int, choices=(2,), default=2)
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument(
        "--max-attempts", type=int, default=MAX_DIRECT_GRASP_ATTEMPTS,
        help="maximum empty/narrow-grasp recovery attempts, 1--3 (default: 3)",
    )
    parser.add_argument("--head-host", default=os.environ.get("PARM_HEAD_HOST", "localhost"))
    parser.add_argument("--head-pitch", type=float, default=38.5)
    parser.add_argument("--head-yaw", type=float, default=0.0)
    parser.add_argument("--head-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--motion-speed-scale", type=float,
        default=2.5 * float(os.environ.get("PARM_RIGHT_SPEED_SCALE", "1.0")),
        help=("机械臂已规划轨迹的速度/加速度倍率；1 保持原速度，"
              "例如 1.5 将各阶段提高至原来的 150%%（默认：2.5）"),
    )
    parser.add_argument("--roi", nargs=4, type=float, default=(0.08, 0.20, 0.98, 0.9))
    parser.add_argument("--outlier-mad-scale", type=float, default=7.0)
    parser.add_argument("--initial-pose-file", type=Path)
    parser.add_argument("--skip-initial-move", action="store_true")
    parser.add_argument("--execute", action="store_true", help="allow real robot, head and gripper motion")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def import_sdk():
    try:
        return importlib.import_module("parm_sdk")
    except ImportError:
        wheel_override = os.environ.get("PARM_SDK_WHEEL", "").strip()
        candidates = [Path(wheel_override)] if wheel_override else []
        # Keep the project runtime SDK authoritative; the copied wheel lets
        # this module remain portable when that shared runtime is unavailable.
        candidates.extend(sorted((LOOP_ROOT / "runtime/parm_sdk").glob("parm_sdk-*.whl"), reverse=True))
        candidates.extend(sorted((PUSH_ROOT.parent / "parm_sdk").glob("parm_sdk-*.whl"), reverse=True))
        for wheel in candidates:
            if wheel.is_file():
                sys.path.insert(0, str(wheel))
                try:
                    return importlib.import_module("parm_sdk")
                except ImportError:
                    pass
    raise RuntimeError("无法导入 parm_sdk；请安装 SDK 或设置 PARM_SDK_WHEEL。")


def move_to_initial_pose3_by_joints(args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(MOVE_SCRIPT), "--arm-id", str(args.arm_id),
        "--pose-file", str(POSE_DIR / "3.json"),
        "--joint-speed", f"{INITIAL_JOINT_SPEED * args.motion_speed_scale:g}",
    ]
    if args.execute:
        command.append("--execute")
    run(command, cwd=PUSH_ROOT)


def current_right_joint_positions_deg(arm: object) -> list[float] | None:
    try:
        right = ((arm.get_state() or {}).get("state") or {}).get("right") or {}
        positions = right.get("position")
        if isinstance(positions, (list, tuple)) and len(positions) == 7:
            return [float(value) for value in positions]
    except Exception:
        pass
    return None


def plan_and_follow_saved_joints(arm: object, planner: object, target: list[float], args: argparse.Namespace, name: str) -> None:
    for retry in range(4):
        start_joints_deg = current_right_joint_positions_deg(arm)
        plan_target = {
            "joint_values_deg": target, "joint_names": RIGHT_ARM_JOINT_NAMES,
            "max_vel": TRANSFER_MAX_VEL * args.motion_speed_scale,
            "max_acc": TRANSFER_MAX_ACC * args.motion_speed_scale,
        }
        try:
            # RRTStar 不在 SDK joint helper 的本地白名单中，直接向规划器
            # 提交标准 joint waypoint，避免退回到 PTP 或触发客户端校验错误。
            result = planner._trajectory.send_payload({
                "waypoints": [{
                    "name": f"module_8_1_{name}_rrtstar",
                    "type": "joint",
                    "group": "right_arm",
                    "planner": "rrtstar",
                    "max_velocity_scaling_factor": plan_target["max_vel"],
                    "max_acceleration_scaling_factor": plan_target["max_acc"],
                    "joint_names": RIGHT_ARM_JOINT_NAMES,
                    "joint_values": target,
                }],
                "topp": {
                    "max_vel_factor": plan_target["max_vel"],
                    "max_acc_factor": plan_target["max_acc"],
                },
            }, timeout=30.0)
        except Exception as exc:
            record_plan(module="8-1", stage=f"saved_joint:{name}:rpc:{retry + 1}", status="failure",
                        planner="rrtstar", request_type="joint", target=plan_target, error=exc,
                        start_joints_deg=start_joints_deg)
            raise
        data = result.get("data") or {}
        trajectory, arm_id = data.get("trajectory"), data.get("arm_id")
        frequency = data.get("sample_frequency", 10.0)
        if arm_id != args.arm_id or not isinstance(trajectory, list) or not trajectory:
            error = RuntimeError(f"规划到 {name} 未返回 arm_id=2 的有效轨迹")
            record_plan(module="8-1", stage=f"saved_joint:{name}:rpc:{retry + 1}", status="failure",
                        planner="rrtstar", request_type="joint", target=plan_target, response=result,
                        error=error, start_joints_deg=start_joints_deg)
            raise error
        record_plan(module="8-1", stage=f"saved_joint:{name}:rpc:{retry + 1}", status="success",
                    planner="rrtstar", request_type="joint", target=plan_target, response=result,
                    start_joints_deg=start_joints_deg)
        try:
            arm.follow_trajectory(arm_id=arm_id, trajectory=trajectory, sample_frequency=float(frequency))
            return
        except Exception as exc:
            if "DiscreteTask::validate: first point mismatch" not in str(exc) or retry == 3:
                raise
            print(f"[planner] {name} 起点不同步，重新规划（{retry + 1}/3）")
            time.sleep(0.2)


def move_between_saved_poses_by_planner(pose_number: int, args: argparse.Namespace) -> None:
    transition_path, pose_path = POSE_DIR / "10.json", POSE_DIR / f"{pose_number}.json"
    transition = load_pose_file(transition_path, args.arm_id, "joint")
    target = load_pose_file(pose_path, args.arm_id, "joint")
    if not args.execute:
        print(f"[dry-run] 规划器将以关节 RRTStar 经 {transition_path.name} 过渡，再移动至 {pose_path.name}。")
        return
    sdk = import_sdk()
    arm = planner = None
    try:
        arm = sdk.Arm(host=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=os.environ.get("PARM_PLANNER_HOST", "localhost"))
        plan_and_follow_saved_joints(arm, planner, transition, args, transition_path.name)
        plan_and_follow_saved_joints(arm, planner, target, args, pose_path.name)
    finally:
        if planner is not None:
            planner.close()
        if arm is not None:
            arm.close()


def release_gripper_before_retry(args: argparse.Namespace, failure_kind: str) -> None:
    """Open before returning to pose 3 so a marginal grasp is never carried."""
    if not args.execute:
        print(f"[dry-run] {failure_kind}后将打开右夹爪，再回观察位。")
        return
    sdk = import_sdk()
    gripper = None
    try:
        gripper = sdk.Gripper(host=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
        gripper.move(args.arm_id, 100.0, mode="async", preempt=True)
        deadline = time.monotonic() + 5.0
        last_position: float | None = None
        while time.monotonic() < deadline:
            states = gripper.get_state()
            state = states.get(args.arm_id) if isinstance(states, dict) else None
            if isinstance(state, dict) and state.get("position") is not None:
                last_position = float(state["position"])
                motion_state = str(state.get("motion_state", "")).lower()
                if last_position >= 95.0 and motion_state in {"arrived", "gripping"}:
                    print(f"[recovery] {failure_kind}后右夹爪已打开（position={last_position:.1f}）。")
                    return
                if motion_state in {"error", "fault", "failed"}:
                    raise RuntimeError(f"{failure_kind}后右夹爪打开失败：motion_state={motion_state!r}")
            time.sleep(0.05)
        observed = "无有效状态" if last_position is None else f"position={last_position:.1f}"
        raise RuntimeError(f"{failure_kind}后右夹爪未能打开（{observed}）；拒绝回 3.json。")
    finally:
        if gripper is not None:
            gripper.close()


def selected_workpiece_state(payload: dict, index: int) -> tuple[int, float]:
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list) or not 0 <= index < len(workpieces):
        count = len(workpieces) if isinstance(workpieces, list) else 0
        raise RuntimeError(f"workpiece-index={index} 无效：pose 文件中有 {count} 个工件")
    quadrant = workpieces[index].get("head_direction_base_quadrant")
    if quadrant not in (1, 2, 3, 4):
        raise RuntimeError(f"无效的工件象限：{quadrant!r}")
    try:
        yaw_deg = float(workpieces[index]["head_direction_base_angle_deg"]) % 360.0
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("pose 文件缺少有效的 head_direction_base_angle_deg") from exc
    return quadrant, yaw_deg


def is_graspable_q1(quadrant: int, yaw_deg: float) -> bool:
    return quadrant == 1 and Q1_GRASP_YAW_MIN_DEG <= yaw_deg <= Q1_GRASP_YAW_MAX_DEG


def grasp_q1(pose: Path, args: argparse.Namespace) -> None:
    """Run the existing direct-Q1 grasp implementation with the fresh pose."""
    command = [
        sys.executable, str(GRASP_SCRIPT), "--arm-id", str(args.arm_id),
        "--workpiece-pose-file", str(pose), "--workpiece-index", str(args.workpiece_index),
        "--planar-xy-yaw", "--planner", "lin", "--tcp", "right-fingertip-center",
        "--q1-grasp-reference-pose-file", str(POSE_DIR / "3.json"),
        "--max-vel", f"{GRASP_MAX_VEL * args.motion_speed_scale:g}",
        "--max-acc", f"{GRASP_MAX_ACC * args.motion_speed_scale:g}",
    ]
    if args.execute:
        command.append("--execute")
    run(command, cwd=PUSH_ROOT)


def observe(args: argparse.Namespace) -> tuple[Path, int, float]:
    """Run Module 4-2 and read exactly the pose JSON created by this call."""
    if not MODULE_4_2_PYTHON.is_file():
        raise FileNotFoundError(f"Module 4-2 虚拟环境不存在：{MODULE_4_2_PYTHON}")
    before = set(MODULE_4_2_OUTPUT_DIR.glob("*_mask_pose.json"))
    command = [
        str(MODULE_4_2_PYTHON), str(MODULE_4_2_SCRIPT),
        "--head-host", args.head_host,
        "--head-pitch", str(args.head_pitch),
        "--head-yaw", str(args.head_yaw),
        "--head-settle-s", str(args.head_settle_s),
        "--roi", *map(str, args.roi),
        "--outlier-mad-scale", str(args.outlier_mad_scale),
    ]
    if not args.execute:
        command.append("--skip-head-move")
    try:
        run(command, cwd=MODULE_4_2_ROOT)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == MODULE_4_2_NO_WORKPIECE_EXIT_CODE:
            raise RuntimeError("桌面没有工件，右臂无法抓取。") from None
        raise
    created = sorted(set(MODULE_4_2_OUTPUT_DIR.glob("*_mask_pose.json")) - before)
    if len(created) != 1:
        raise RuntimeError(f"Module 4-2 应只生成一个新的 pose JSON，实际为：{created}")
    pose = created[0]
    payload = json.loads(pose.read_text(encoding="utf-8"))
    quadrant, yaw_deg = selected_workpiece_state(payload, args.workpiece_index)
    return pose, quadrant, yaw_deg


def main() -> int:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.max_attempts > MAX_DIRECT_GRASP_ATTEMPTS:
        raise ValueError(f"--max-attempts 不能大于 {MAX_DIRECT_GRASP_ATTEMPTS}")
    if not 0.0 < args.motion_speed_scale <= 5.0:
        raise ValueError("--motion-speed-scale 必须在 (0, 5] 内")
    record_dir = start_run("8-1")
    print(f"[right-planner-record] 规划记录目录：{record_dir}")
    print(
        "[motion] speed scale=" f"{args.motion_speed_scale:g}; "
        f"initial move_j={INITIAL_JOINT_SPEED * args.motion_speed_scale:g}, "
        f"transfer={TRANSFER_MAX_VEL * args.motion_speed_scale:g}/"
        f"{TRANSFER_MAX_ACC * args.motion_speed_scale:g}, "
        f"grasp={GRASP_MAX_VEL * args.motion_speed_scale:g}/"
        f"{GRASP_MAX_ACC * args.motion_speed_scale:g} (vel/acc)"
    )
    required = [
        MODULE_4_2_SCRIPT, MODULE_4_2_PYTHON, GRASP_SCRIPT, MOVE_SCRIPT,
        *(POSE_DIR / f"{number}.json" for number in (3, 10)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少所需文件：\n" + "\n".join(missing))

    if not args.skip_initial_move:
        move_to_initial_pose3_by_joints(args)
    for attempt in range(1, args.max_attempts + 1):
        if attempt == 1 and args.initial_pose_file is not None:
            pose = args.initial_pose_file.expanduser().resolve()
            if not pose.is_file():
                raise FileNotFoundError(f"初始 Module 4-2 pose 不存在：{pose}")
            payload = json.loads(pose.read_text(encoding="utf-8"))
            quadrant, yaw_deg = selected_workpiece_state(
                payload, args.workpiece_index
            )
        else:
            pose, quadrant, yaw_deg = observe(args)
        print(
            f"[decision] attempt={attempt}/{args.max_attempts}, "
            f"workpiece={args.workpiece_index}, Q{quadrant}, yaw={yaw_deg:.2f}°, pose={pose}"
        )
        if not is_graspable_q1(quadrant, yaw_deg):
            print("工件位姿已不满足直接抓取情况，请回到module_7重新调整")
            return Q1_REORIENTATION_REQUIRED_EXIT_CODE
        try:
            grasp_q1(pose, args)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == Q1_DIRECT_PATH_UNAVAILABLE_EXIT_CODE:
                print("无法抓取，需module_7再次调整位姿")
                return Q1_REORIENTATION_REQUIRED_EXIT_CODE
            if exc.returncode in {Q1_EMPTY_GRASP_EXIT_CODE, Q1_NARROW_GRASP_EXIT_CODE}:
                failure_kind = "Q1 空抓" if exc.returncode == Q1_EMPTY_GRASP_EXIT_CODE else "Q1 窄抓恢复失败"
                if attempt == args.max_attempts:
                    print("无法抓取，需module_7再次调整位姿")
                    return Q1_REORIENTATION_REQUIRED_EXIT_CODE
                print(
                    f"[retry] {failure_kind}（attempt={attempt}/{args.max_attempts}）；"
                    "松爪后经 10.json 回 3.json，重新观察并再抓一次。"
                )
                release_gripper_before_retry(args, failure_kind)
                move_between_saved_poses_by_planner(3, args)
                continue
            print("无法抓取，需module_7再次调整位姿")
            return Q1_REORIENTATION_REQUIRED_EXIT_CODE
        print("[done] Q1 抓取成功。")
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    with console_log("8_1"):
        try:
            raise SystemExit(main())
        except KeyboardInterrupt:
            print("\n已中断，未再下发后续动作。", file=sys.stderr)
            raise SystemExit(130)
        except Exception as exc:
            print(f"\n错误：{exc}", file=sys.stderr)
            raise SystemExit(1)
