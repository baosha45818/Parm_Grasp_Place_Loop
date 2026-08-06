#!/usr/bin/env python3
"""Observe one workpiece and reorient it to a directly graspable Q1 pose.

The pipeline always observes from arm pose 3 with the head at pitch 38.5 and
yaw 0.  A workpiece already in a directly graspable Q1 pose is left untouched.
Q2--Q4 (and hard-to-grasp Q1 edge ranges) are reoriented with
``reorient_q3_to_q1_old.py`` from pose 6.  All observations are made from
pose 3.  After each reorientation, the pipeline returns to pose 3 and uses a
fresh observation to decide whether further adjustment is required.
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


LOOP_ROOT = Path(__file__).resolve().parents[2]
PUSH_ROOT = Path(__file__).resolve().parent / "vendor/push_t"
sys.path.insert(0, str(PUSH_ROOT))
from move_by_joints import ensure_ready, load_pose_file  # noqa: E402
from right_console_logger import console_log  # noqa: E402
from right_trajectory_recorder import record_plan, start_run  # noqa: E402


MODULE_4_2_ROOT = (
    Path(__file__).resolve().parents[1]
    / "module_4_head_judge/module_4_2_pose_estimation"
)
MODULE_4_2_SCRIPT = MODULE_4_2_ROOT / "run_pose_estimation.py"
MODULE_4_2_PYTHON = MODULE_4_2_ROOT / ".venv/bin/python"
MODULE_4_2_OUTPUT_DIR = LOOP_ROOT / "output/4_2"
MODULE_4_2_NO_WORKPIECE_EXIT_CODE = 2
# Public Module-7 result: Module 4-2 observed an empty buffer/table.
MODULE_7_NO_WORKPIECE_EXIT_CODE = 24
MOVE_SCRIPT = PUSH_ROOT / "move_by_joints.py"
REORIENT_SCRIPT = PUSH_ROOT / "reorient_q3_to_q1_old.py"
POSE_DIR = PUSH_ROOT / "saved_poses"
Q1_GRASP_YAW_MIN_DEG = 15.0
Q1_GRASP_YAW_MAX_DEG = 75.0
Q1_REORIENT_TARGET_YAWS_DEG = (45.0, 40.0, 50.0, 35.0, 55.0, 30.0, 60.0, 25.0, 65.0, 20.0, 70.0, 15.0, 75.0)
RIGHT_ARM_JOINT_NAMES = [f"right_arm_joint{index}" for index in range(1, 8)]
INITIAL_JOINT_SPEED = 0.2
TRANSFER_MAX_VEL = 0.1
TRANSFER_MAX_ACC = 0.1
REORIENT_MAX_VEL = 0.08
REORIENT_MAX_ACC = 0.08


class NoWorkpieceError(RuntimeError):
    """Raised when Module 4-2 reports that the buffer is empty."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", type=int, choices=(2,), default=2)
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument(
        "--max-attempts", type=int, default=10,
        help="maximum reorientation/grasp attempts before stopping (default: 10)",
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
    parser.add_argument("--prepare-observation-only", action="store_true")
    parser.add_argument(
        "--place-tcp-extra-clearance-m", type=float, default=0.005,
        help="重定向后放下并松爪时额外抬高 TCP（米；默认 5 mm）",
    )
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
    """Use the taught joint posture only for the pipeline's initial setup."""
    command = [
        sys.executable, str(MOVE_SCRIPT), "--arm-id", str(args.arm_id),
        "--pose-file", str(POSE_DIR / "3.json"),
        "--joint-speed", f"{INITIAL_JOINT_SPEED * args.motion_speed_scale:g}",
    ]
    if args.execute:
        command.append("--execute")
    run(command, cwd=PUSH_ROOT)


def current_right_joint_positions_deg(arm: object) -> list[float] | None:
    """Best-effort actual joint start state for planner diagnostics only."""
    try:
        right = ((arm.get_state() or {}).get("state") or {}).get("right") or {}
        positions = right.get("position")
        if isinstance(positions, (list, tuple)) and len(positions) == 7:
            return [float(value) for value in positions]
    except Exception:
        pass
    return None


def plan_and_follow_saved_joints(arm: object, planner: object, target: list[float], args: argparse.Namespace, name: str) -> None:
    """Plan and execute a saved seven-joint target, retrying a stale start."""
    for retry in range(4):
        start_joints_deg = current_right_joint_positions_deg(arm)
        plan_target = {
            "joint_values_deg": target,
            "joint_names": RIGHT_ARM_JOINT_NAMES,
            "max_vel": TRANSFER_MAX_VEL * args.motion_speed_scale,
            "max_acc": TRANSFER_MAX_ACC * args.motion_speed_scale,
        }
        try:
            # SDK 的 plan_joint_trajectory 本地只接受 ptp/lin/ompl；RRTStar
            # 必须向 5604 直接发送标准 waypoint，不能只替换 planner 字符串。
            result = planner._trajectory.send_payload({
                "waypoints": [{
                    "name": f"module_7_{name}_rrtstar",
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
            record_plan(
                module="7", stage=f"saved_joint:{name}:rpc:{retry + 1}", status="failure",
                planner="rrtstar", request_type="joint", target=plan_target, error=exc,
                start_joints_deg=start_joints_deg,
            )
            raise
        data = result.get("data") or {}
        trajectory, arm_id = data.get("trajectory"), data.get("arm_id")
        frequency = data.get("sample_frequency", 10.0)
        if arm_id != args.arm_id or not isinstance(trajectory, list) or not trajectory:
            error = RuntimeError(f"规划到 {name} 未返回 arm_id=2 的有效轨迹")
            record_plan(
                module="7", stage=f"saved_joint:{name}:rpc:{retry + 1}", status="failure",
                planner="rrtstar", request_type="joint", target=plan_target, response=result, error=error,
                start_joints_deg=start_joints_deg,
            )
            raise error
        record_plan(
            module="7", stage=f"saved_joint:{name}:rpc:{retry + 1}", status="success",
            planner="rrtstar", request_type="joint", target=plan_target, response=result,
            start_joints_deg=start_joints_deg,
        )
        try:
            arm.follow_trajectory(arm_id=arm_id, trajectory=trajectory, sample_frequency=float(frequency))
            return
        except Exception as exc:
            if "DiscreteTask::validate: first point mismatch" not in str(exc) or retry == 3:
                raise
            print(f"[planner] {name} 起点不同步，重新规划（{retry + 1}/3）")
            time.sleep(0.2)


def move_between_saved_poses_by_planner(pose_number: int, args: argparse.Namespace) -> None:
    """Use 10.json and joint-space planning for every post-startup transition."""
    transition_path = POSE_DIR / "10.json"
    pose_path = POSE_DIR / f"{pose_number}.json"
    transition = load_pose_file(transition_path, args.arm_id, "joint")
    target = load_pose_file(pose_path, args.arm_id, "joint")
    if not args.execute:
        print(
            f"[dry-run] 规划器将以关节 RRTStar 经 {transition_path.name} 过渡，"
            f"再移动至 {pose_path.name}。"
        )
        return
    sdk = import_sdk()
    arm = planner = None
    try:
        arm = sdk.Arm(host=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=os.environ.get("PARM_PLANNER_HOST", "localhost"))
        print(f"[planner] 关节 RRTStar 规划经 {transition_path.name} 过渡，再移动至 {pose_path.name}")
        plan_and_follow_saved_joints(arm, planner, transition, args, transition_path.name)
        plan_and_follow_saved_joints(arm, planner, target, args, pose_path.name)
    finally:
        if planner is not None:
            planner.close()
        if arm is not None:
            arm.close()


def release_gripper_after_reorient_failure(args: argparse.Namespace) -> None:
    """Never carry a possibly grasped workpiece back to the camera pose."""
    if not args.execute:
        print("[dry-run] 重定向失败恢复前将打开右夹爪。")
        return
    sdk = import_sdk()
    gripper = None
    try:
        gripper = sdk.Gripper(host=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
        gripper.move(args.arm_id, 100.0, mode="async", preempt=True)
        print("[recovery] 重定向失败；已下发右夹爪打开命令，等待确认后才回 3.json。")
        deadline = time.monotonic() + 5.0
        last_position: float | None = None
        while time.monotonic() < deadline:
            states = gripper.get_state()
            state = states.get(args.arm_id) if isinstance(states, dict) else None
            if isinstance(state, dict) and state.get("position") is not None:
                last_position = float(state["position"])
                motion_state = str(state.get("motion_state", "")).lower()
                if last_position >= 95.0 and motion_state in {"arrived", "gripping"}:
                    print(f"[recovery] 右夹爪已打开（position={last_position:.1f}）。")
                    return
                if motion_state in {"error", "fault", "failed"}:
                    raise RuntimeError(f"重定向失败恢复时右夹爪打开失败：motion_state={motion_state!r}")
            time.sleep(0.05)
        observed = "无有效状态" if last_position is None else f"position={last_position:.1f}"
        raise RuntimeError(f"重定向失败恢复时右夹爪未能打开（{observed}）；拒绝携带工件返回 3.json。")
    finally:
        if gripper is not None:
            gripper.close()


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
        # A dry-run may still acquire and process an image, but it must never
        # command the head.  Module 4-2 owns the head-motion implementation.
        command.append("--skip-head-move")
    try:
        run(command, cwd=MODULE_4_2_ROOT)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == MODULE_4_2_NO_WORKPIECE_EXIT_CODE:
            raise NoWorkpieceError("桌面没有工件，右臂无法抓取。") from None
        raise
    created = sorted(set(MODULE_4_2_OUTPUT_DIR.glob("*_mask_pose.json")) - before)
    if len(created) != 1:
        raise RuntimeError(f"Module 4-2 应只生成一个新的 pose JSON，实际为：{created}")
    pose = created[0]
    quadrant, yaw_deg = selected_workpiece_state(json.loads(pose.read_text(encoding="utf-8")), args.workpiece_index)
    return pose, quadrant, yaw_deg


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
    """Only the central, easy-to-grasp part of Q1 may go directly to grasp."""
    return quadrant == 1 and Q1_GRASP_YAW_MIN_DEG <= yaw_deg <= Q1_GRASP_YAW_MAX_DEG


def reorient_to_q1(pose: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(REORIENT_SCRIPT), "--arm-id", str(args.arm_id),
        "--workpiece-pose-file", str(pose), "--workpiece-index", str(args.workpiece_index),
        "--workpiece-height-m", "0.010", "--grasp-tcp-extra-clearance-m", "0",
        "--place-tcp-extra-clearance-m", str(args.place_tcp_extra_clearance_m),
        "--placement-visibility-roi", *map(str, args.roi),
        "--grasp-xy-offset-m", "0", "0", "--planner", "rrtstar", "--turn-planner", "rrtstar",
        "--max-vel", f"{REORIENT_MAX_VEL * args.motion_speed_scale:g}",
        "--max-acc", f"{REORIENT_MAX_ACC * args.motion_speed_scale:g}",
    ]
    for yaw_deg in Q1_REORIENT_TARGET_YAWS_DEG:
        command.extend(["--q1-target-yaw-deg", str(yaw_deg)])
    if args.execute:
        command.append("--execute")
    run(command, cwd=PUSH_ROOT)


def main() -> int:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if not 0.0 < args.motion_speed_scale <= 5.0:
        raise ValueError("--motion-speed-scale 必须在 (0, 5] 内")
    record_dir = start_run("7")
    print(f"[right-planner-record] 规划记录目录：{record_dir}")
    print(
        "[motion] speed scale=" f"{args.motion_speed_scale:g}; "
        f"initial move_j={INITIAL_JOINT_SPEED * args.motion_speed_scale:g}, "
        f"transfer={TRANSFER_MAX_VEL * args.motion_speed_scale:g}/"
        f"{TRANSFER_MAX_ACC * args.motion_speed_scale:g}, "
        f"reorient={REORIENT_MAX_VEL * args.motion_speed_scale:g}/"
        f"{REORIENT_MAX_ACC * args.motion_speed_scale:g} (vel/acc)"
    )
    required = [MODULE_4_2_SCRIPT, MODULE_4_2_PYTHON, MOVE_SCRIPT, REORIENT_SCRIPT,
                *(POSE_DIR / f"{number}.json" for number in (3, 6, 10))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少所需文件：\n" + "\n".join(missing))

    # Every observation is made from pose 3.  Module 7 only decides whether
    # adjustment is needed and, when required, reorients the workpiece; it
    # never performs the following module's final Q1 grasp.
    if not args.skip_initial_move:
        move_to_initial_pose3_by_joints(args)
    if args.prepare_observation_only:
        print("[module-7] 右臂已到 Module 4-2 缓冲区观察位，未做摆正或抓取。")
        return 0
    if args.initial_pose_file is not None:
        pose = args.initial_pose_file.expanduser().resolve()
        if not pose.is_file():
            raise FileNotFoundError(f"初始 Module 4-2 pose 不存在：{pose}")
        quadrant, yaw_deg = selected_workpiece_state(
            json.loads(pose.read_text(encoding="utf-8")), args.workpiece_index
        )
    else:
        pose, quadrant, yaw_deg = observe(args)
    print(f"[decision] workpiece={args.workpiece_index}, Q{quadrant}, yaw={yaw_deg:.2f}°, pose={pose}")

    attempt = 0
    while attempt < args.max_attempts:
        if is_graspable_q1(quadrant, yaw_deg):
            if attempt == 0:
                print("[done] 该阶段不需要右臂调整，可直接抓")
            else:
                print("[done] 右臂调整完成，可进行下一步抓取")
            return 0

        # Q2--Q4 and the hard-to-grasp Q1 edge ranges are reoriented first.
        if quadrant == 1:
            print(
                f"[decision] Q1 yaw={yaw_deg:.2f}° 不在可抓取区间 "
                f"[{Q1_GRASP_YAW_MIN_DEG:.0f}, {Q1_GRASP_YAW_MAX_DEG:.0f}]°；先重定向。"
            )
        move_between_saved_poses_by_planner(6, args)
        attempt += 1
        try:
            reorient_to_q1(pose, args)
        except subprocess.CalledProcessError as exc:
            print(f"[retry] Q{quadrant} 转至可抓取 Q1 失败（exit={exc.returncode}）；先松爪，再回到 3.json 重新观察后重试。")
            release_gripper_after_reorient_failure(args)
            if attempt == args.max_attempts:
                raise RuntimeError(f"达到 --max-attempts={args.max_attempts}，工件仍未成功转至 Q1。") from exc
            move_between_saved_poses_by_planner(3, args)
            pose, quadrant, yaw_deg = observe(args)
            print(f"[decision] recovery observation: Q{quadrant}, yaw={yaw_deg:.2f}°, pose={pose}")
            continue

        # The real-world result of a successful command is still unknown.
        # Return to pose 3 and use fresh vision before deciding whether another
        # Module-7 adjustment is needed.
        move_between_saved_poses_by_planner(3, args)
        pose, quadrant, yaw_deg = observe(args)
        print(f"[decision] post-reorientation observation: Q{quadrant}, yaw={yaw_deg:.2f}°, pose={pose}")

    raise AssertionError("unreachable")


if __name__ == "__main__":
    with console_log("7"):
        try:
            raise SystemExit(main())
        except KeyboardInterrupt:
            print("\n已中断，未再下发后续动作。", file=sys.stderr)
            raise SystemExit(130)
        except NoWorkpieceError as exc:
            print(f"\n错误：{exc}", file=sys.stderr)
            raise SystemExit(MODULE_7_NO_WORKPIECE_EXIT_CODE)
        except Exception as exc:
            print(f"\n错误：{exc}", file=sys.stderr)
            raise SystemExit(1)
