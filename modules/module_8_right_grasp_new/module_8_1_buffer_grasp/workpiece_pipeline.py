#!/usr/bin/env python3
"""Observe, grasp, or reorient workpiece 1/2 in a closed loop.

Observations made while already at saved pose 3 happen directly at pose 3.
After a reorientation, vision starts at pose 10 during the 6 -> 10 -> 3
return, so the decision is ready when pose 3 is reached.  A directly
graspable workpiece is grasped immediately.  Three consecutive direct-grasp
failures force one reorientation using the latest observation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

from planning_log import PLAN_LOG_ENV, append_event


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[2]
RUNTIME_PYTHON = PROJECT_ROOT / ".venv/bin/python"
if not RUNTIME_PYTHON.is_file():
    raise RuntimeError(f"总控运行环境不存在: {RUNTIME_PYTHON}")
# Preserve the venv path rather than resolving symlinks: a venv and its base
# interpreter normally resolve to the same binary but do not share packages.
if Path(sys.executable).absolute() != RUNTIME_PYTHON.absolute():
    os.execv(str(RUNTIME_PYTHON), [str(RUNTIME_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

VISION_MODULE_DIR = PROJECT_ROOT / "modules/module_4_head_judge_new/module_4_2_pose_estimation"
ALIGN_MODULE_DIR = PROJECT_ROOT / "modules/module_7_right_align_new"
POSE_SCRIPT = VISION_MODULE_DIR / "tabletop_pointcloud_workpiece_pose.py"
CAPTURE_ROOT = PROJECT_ROOT / "output/module_4_2"
MOVE_SCRIPT = MODULE_ROOT / "move_by_joints.py"
POSE_DIR = MODULE_ROOT / "saved_poses"
GRASP_SCRIPT = MODULE_ROOT / "buffer_grasp.py"
REORIENT_SCRIPT = ALIGN_MODULE_DIR / "right_align.py"
DEFAULT_LOG_DIR = PROJECT_ROOT / "output/module_8_1"
Q1_MIN_DEG = 15.0
Q1_MAX_DEG = 75.0
CAPTURE_READY_TIMEOUT_S = 30.0

WORKPIECE_CONFIGS = {
    1: {
        "grasp": MODULE_ROOT / "configs/workpiece_1_grasp.json",
        "reorient": ALIGN_MODULE_DIR / "configs/workpiece_1_reorient.json",
    },
    2: {
        "grasp": MODULE_ROOT / "configs/workpiece_2_grasp.json",
        "reorient": ALIGN_MODULE_DIR / "configs/workpiece_2_reorient.json",
    },
}


class NoWorkpieceDetectedError(RuntimeError):
    """The observation completed but did not contain the selected item."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", type=int, choices=(1, 2), required=True,
                        help="选择工件模型及对应的抓取/调整 JSON")
    parser.add_argument("--arm-id", type=int, choices=(2,), default=2)
    parser.add_argument("--workpiece-index", type=int, default=0,
                        help="mask_pose.json 中要操作的工件下标")
    parser.add_argument("--max-grasp-failures", type=int, default=3,
                        help="连续直接抓取失败多少次后强制调整（默认 3）")
    parser.add_argument("--max-cycles", type=int, default=20,
                        help="观察/抓取/调整闭环的总周期上限（默认 20）")
    parser.add_argument("--observation-attempts", type=int, default=3,
                        help="在 10.json 拍照时未识别到工件的重拍次数（默认 3）")
    parser.add_argument("--roi", nargs=4, type=float,
                        default=(0.08, 0.20, 0.98, 0.97))
    parser.add_argument("--joint-speed", type=float, default=0.2)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help=f"运行日志目录（默认 {DEFAULT_LOG_DIR}）")
    parser.add_argument("--execute", action="store_true",
                        help="实际拍照并执行机械臂和夹爪动作；省略时仅检查并打印流程")
    return parser.parse_args()


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if str(MODULE_ROOT) not in entries:
        entries.insert(0, str(MODULE_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def format_command(command: list[str]) -> str:
    return shlex.join(command)


class TeeStream:
    """Mirror parent-process output to the terminal and the session log."""

    def __init__(self, terminal: object, log_handle: object, lock: threading.Lock) -> None:
        self.terminal, self.log_handle, self.lock = terminal, log_handle, lock

    def write(self, text: str) -> int:
        with self.lock:
            self.terminal.write(text)
            self.terminal.flush()
            self.log_handle.write(text)
            self.log_handle.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


class PipelineRunLog:
    """Own one human-readable log and one structured event stream per run."""

    def __init__(self, args: argparse.Namespace) -> None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"workpiece_{args.workpiece}_{stamp}_pid{os.getpid()}"
        self.text_path = args.log_dir / f"{stem}.log"
        self.event_path = args.log_dir / f"{stem}.jsonl"
        self.handle = self.text_path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()
        self.original_stdout, self.original_stderr = sys.stdout, sys.stderr

    def __enter__(self) -> "PipelineRunLog":
        os.environ[PLAN_LOG_ENV] = str(self.event_path)
        sys.stdout = TeeStream(self.original_stdout, self.handle, self.lock)
        sys.stderr = TeeStream(self.original_stderr, self.handle, self.lock)
        append_event({"event": "pipeline_start", "text_log": str(self.text_path)})
        print(f"[log] text={self.text_path}")
        print(f"[log] events={self.event_path}")
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        if exc is not None:
            print(f"[pipeline-error] {type(exc).__name__}: {exc}", file=sys.stderr)
        append_event({
            "event": "pipeline_end",
            "success": exc is None,
            "failure_reason": None if exc is None else f"{type(exc).__name__}: {exc}",
        })
        sys.stdout, sys.stderr = self.original_stdout, self.original_stderr
        os.environ.pop(PLAN_LOG_ENV, None)
        self.handle.close()

    def run(self, command: list[str], *, cwd: Path, check: bool, action: str) -> int:
        command_text = format_command(command)
        print(f"\n$ cd {cwd}\n$ {command_text}", flush=True)
        started = time.perf_counter()
        append_event({
            "event": "subprocess_start", "action": action,
            "cwd": str(cwd), "command": command,
        })
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=command_env(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
        except Exception as exc:
            append_event({
                "event": "subprocess_end", "action": action, "success": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            })
            raise
        append_event({
            "event": "subprocess_end", "action": action,
            "success": returncode == 0, "returncode": returncode,
            "failure_reason": None if returncode == 0 else f"process exited with status {returncode}",
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        })
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        return returncode


ACTIVE_LOG: PipelineRunLog | None = None


def run_checked(command: list[str], *, cwd: Path) -> None:
    if ACTIVE_LOG is None:
        subprocess.run(command, cwd=cwd, env=command_env(), check=True)
        return
    ACTIVE_LOG.run(command, cwd=cwd, check=True, action="checked_command")


def run_action(command: list[str], *, cwd: Path) -> int:
    """Run grasp/reorient while leaving its nonzero status to the controller."""
    if ACTIVE_LOG is None:
        return subprocess.run(command, cwd=cwd, env=command_env(), check=False).returncode
    return ACTIVE_LOG.run(command, cwd=cwd, check=False, action="manipulation_command")


def import_sdk() -> object:
    try:
        return importlib.import_module("parm_sdk")
    except ImportError:
        override = os.environ.get("PARM_SDK_WHEEL", "").strip()
        candidates = [Path(override)] if override else []
        candidates.extend(sorted(
            Path("/home/dainanjun/Public/parm_sdk_1.0.0").glob("parm_sdk-*.whl"),
            reverse=True,
        ))
        for wheel in candidates:
            if not wheel.is_file():
                continue
            sys.path.insert(0, str(wheel))
            try:
                return importlib.import_module("parm_sdk")
            except ImportError:
                continue
    raise RuntimeError("无法导入 parm_sdk；不能安全控制失败后的夹爪释放。")


def release_gripper(args: argparse.Namespace, reason: str) -> None:
    """Open the gripper before moving away from any failed manipulation."""
    if not args.execute:
        print(f"[dry-run] {reason} 后打开夹爪。")
        return
    gripper = None
    try:
        gripper = import_sdk().Gripper(
            host=os.environ.get("PARM_GRIPPER_HOST", "localhost")
        )
        gripper.move(args.arm_id, 100.0, mode="async", preempt=True)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            states = gripper.get_state()
            state = states.get(args.arm_id) if isinstance(states, dict) else None
            if (isinstance(state, dict) and state.get("position") is not None
                    and float(state["position"]) >= 95.0):
                print(f"[recovery] {reason} 后夹爪已打开。")
                return
            time.sleep(0.05)
        raise RuntimeError(f"{reason} 后夹爪未确认打开，停止后续机械臂运动。")
    finally:
        if gripper is not None:
            gripper.close()


def move_to_saved_pose(number: int, args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(MOVE_SCRIPT),
        "--arm-id", str(args.arm_id),
        "--pose-file", str(POSE_DIR / f"{number}.json"),
        "--joint-speed", str(args.joint_speed),
    ]
    if args.execute:
        command.append("--execute")
        run_checked(command, cwd=MODULE_ROOT)
    else:
        print(f"[dry-run] {format_command(command + ['--execute'])}")


def observe_once(args: argparse.Namespace) -> tuple[Path, int, float]:
    capture_root = CAPTURE_ROOT / f"workpiece_{args.workpiece}"
    before = {path.resolve() for path in capture_root.glob("*/mask_pose.json")}
    command = [
        sys.executable, str(POSE_SCRIPT),
        "--workpiece", str(args.workpiece),
        "--roi", *map(str, args.roi),
    ]
    run_checked(command, cwd=VISION_MODULE_DIR)
    after = {path.resolve() for path in capture_root.glob("*/mask_pose.json")}
    created = sorted(
        after - before,
        key=lambda path: int(path.parent.name) if path.parent.name.isdecimal() else -1,
    )
    if len(created) != 1:
        raise RuntimeError(
            "本次位姿计算应新建且仅新建一个 mask_pose.json，"
            f"实际新建文件为：{created}"
        )
    pose_file = created[0]
    try:
        payload = json.loads(pose_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取本次位姿文件 {pose_file}: {exc}") from exc
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list) or not 0 <= args.workpiece_index < len(workpieces):
        count = len(workpieces) if isinstance(workpieces, list) else 0
        raise NoWorkpieceDetectedError(
            f"workpiece-index={args.workpiece_index} 无效，本次识别到 {count} 个工件"
        )
    item = workpieces[args.workpiece_index]
    if not isinstance(item, dict):
        raise RuntimeError(f"workpieces[{args.workpiece_index}] 不是对象")
    quadrant = item.get("head_direction_base_quadrant")
    if quadrant not in (1, 2, 3, 4):
        raise RuntimeError(f"位姿文件包含无效象限：{quadrant!r}")
    try:
        yaw_deg = float(item["head_direction_base_angle_deg"]) % 360.0
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("位姿文件缺少有效的 head_direction_base_angle_deg") from exc
    return pose_file, int(quadrant), yaw_deg


def observe_at_10_while_returning_to_3(
    args: argparse.Namespace, *, already_at_10: bool = False,
) -> tuple[Path, int, float]:
    """Capture only after pose 10 is reached, then calculate during 10 -> 3.

    ``move_to_saved_pose`` waits for the motion command to finish.  In addition
    to starting the vision job after pose 10, wait until that job has saved its
    RGB frame before beginning 10 -> 3.  Thus the camera exposure is complete
    at pose 10; only the remaining pose calculation overlaps the return move.
    Any retry likewise makes a fresh 3 -> 10 trip before capturing.
    """
    for attempt in range(1, args.observation_attempts + 1):
        if not already_at_10:
            print("[motion] 到达 10.json 后开始拍照；视觉计算将与 10 -> 3 回程并行。")
            move_to_saved_pose(10, args)
        else:
            print("[motion] 已到达 10.json，开始拍照；视觉计算将与 10 -> 3 回程并行。")
        already_at_10 = False

        capture_root = CAPTURE_ROOT / f"workpiece_{args.workpiece}"
        previous_rgb = {path.resolve() for path in capture_root.glob("*/rgb.png")}
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="workpiece-observe") as executor:
            observation = executor.submit(observe_once, args)
            capture_deadline = time.monotonic() + CAPTURE_READY_TIMEOUT_S
            while True:
                current_rgb = {path.resolve() for path in capture_root.glob("*/rgb.png")}
                if current_rgb - previous_rgb:
                    print("[observe] 10.json 位姿下的 RGB-D 已采集，开始返回 3.json 并继续计算。")
                    break
                if observation.done():
                    # Propagate a camera/vision startup failure instead of
                    # masking it as a timeout.
                    observation.result()
                    raise RuntimeError("视觉任务结束但未找到本次 RGB-D 采集文件。")
                if time.monotonic() >= capture_deadline:
                    raise RuntimeError(
                        f"到达 10.json 后 {CAPTURE_READY_TIMEOUT_S:.0f} 秒内未完成 RGB-D 采集。"
                    )
                time.sleep(0.05)
            move_to_saved_pose(3, args)
            try:
                return observation.result()
            except NoWorkpieceDetectedError as exc:
                if attempt == args.observation_attempts:
                    raise RuntimeError(
                        f"连续 {args.observation_attempts} 次在 10.json 拍照均未识别到目标工件。"
                    ) from exc
                print(
                    f"[observe] 第 {attempt}/{args.observation_attempts} 次未识别到工件："
                    f"{exc}；将从 3.json 返回 10.json 后重拍。"
                )
    raise AssertionError("unreachable")


def observe_at_pose_3(args: argparse.Namespace) -> tuple[Path, int, float]:
    """Observe directly at pose 3; retries remain at pose 3 as well."""
    for attempt in range(1, args.observation_attempts + 1):
        try:
            return observe_once(args)
        except NoWorkpieceDetectedError as exc:
            if attempt == args.observation_attempts:
                raise RuntimeError(
                    f"连续 {args.observation_attempts} 次在 3.json 拍照均未识别到目标工件。"
                ) from exc
            print(
                f"[observe] 第 {attempt}/{args.observation_attempts} 次未识别到工件："
                f"{exc}；保持在 3.json 重拍。"
            )
    raise AssertionError("unreachable")


def directly_graspable(quadrant: int, yaw_deg: float) -> bool:
    return quadrant == 1 and Q1_MIN_DEG <= yaw_deg <= Q1_MAX_DEG


def grasp(pose_file: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable, str(GRASP_SCRIPT),
        "--workpiece", str(args.workpiece),
        "--arm-id", str(args.arm_id),
        "--workpiece-pose-file", str(pose_file),
        "--workpiece-index", str(args.workpiece_index),
        "--execute",
    ]
    return run_action(command, cwd=MODULE_ROOT)


def reorient(pose_file: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable, str(REORIENT_SCRIPT),
        "--workpiece", str(args.workpiece),
        "--arm-id", str(args.arm_id),
        "--workpiece-pose-file", str(pose_file),
        "--workpiece-index", str(args.workpiece_index),
        "--execute",
    ]
    return run_action(command, cwd=ALIGN_MODULE_DIR)


def validate(args: argparse.Namespace) -> None:
    if args.max_grasp_failures < 1:
        raise ValueError("--max-grasp-failures 必须至少为 1")
    if args.max_cycles < 1:
        raise ValueError("--max-cycles 必须至少为 1")
    if args.observation_attempts < 1:
        raise ValueError("--observation-attempts 必须至少为 1")
    if not 0.0 < args.joint_speed <= 1.0:
        raise ValueError("--joint-speed 必须位于 (0, 1]")
    required = [
        POSE_SCRIPT, MOVE_SCRIPT, GRASP_SCRIPT, REORIENT_SCRIPT,
        WORKPIECE_CONFIGS[args.workpiece]["grasp"],
        WORKPIECE_CONFIGS[args.workpiece]["reorient"],
        *(POSE_DIR / f"{number}.json" for number in (3, 6, 10)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少总控所需文件：\n" + "\n".join(missing))


def print_dry_run(args: argparse.Namespace) -> None:
    configs = WORKPIECE_CONFIGS[args.workpiece]
    print("预演模式：未拍照，也不会下发机械臂或夹爪动作。")
    print(f"工件: {args.workpiece}")
    print(f"观察模型: --workpiece {args.workpiece}")
    print(f"抓取配置: {configs['grasp']}")
    print(f"调整配置: {configs['reorient']}")
    print(f"直接抓取条件: Q1 且 yaw 位于 [{Q1_MIN_DEG:.0f}, {Q1_MAX_DEG:.0f}]°")
    print(f"连续抓取失败 {args.max_grasp_failures} 次后强制调整。")
    print("观察时序: 正常在 3.json 拍照；仅调整后的 6 -> 10 -> 3 回程在 10.json 拍照，并行计算。")
    move_to_saved_pose(3, args)
    print("[dry-run] 在 3.json 拍照并计算位姿。")


def main() -> int:
    global ACTIVE_LOG
    args = parse_args()
    with PipelineRunLog(args) as run_log:
        ACTIVE_LOG = run_log
        append_event({
            "event": "pipeline_arguments",
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        })
        try:
            validate(args)
            if not args.execute:
                print_dry_run(args)
                return 0

            consecutive_grasp_failures = 0
            force_reorient = False
            move_to_saved_pose(3, args)
            observation_at_pose_10 = False

            for cycle in range(1, args.max_cycles + 1):
                if observation_at_pose_10:
                    pose_file, quadrant, yaw_deg = observe_at_10_while_returning_to_3(
                        args, already_at_10=True,
                    )
                else:
                    pose_file, quadrant, yaw_deg = observe_at_pose_3(args)
                observation_at_pose_10 = False
                append_event({
                    "event": "observation", "cycle": cycle,
                    "pose_file": str(pose_file), "quadrant": quadrant, "yaw_deg": yaw_deg,
                })
                print(
                    f"[decision] 周期 {cycle}/{args.max_cycles}：工件 {args.workpiece}，"
                    f"Q{quadrant}，yaw={yaw_deg:.2f}°，pose={pose_file}，"
                    f"连续抓取失败={consecutive_grasp_failures}/{args.max_grasp_failures}"
                )

                # A fresh Q1 observation in the direct-grasp yaw window always
                # wins over any previous adjustment state.  ``force_reorient``
                # only affects a non-directly-graspable observation.
                if directly_graspable(quadrant, yaw_deg):
                    print("[decision] 位于可直接抓取的 Q1 15°--75° 区间，执行直接抓取。")
                    grasp_status = grasp(pose_file, args)
                    if grasp_status == 0:
                        print("[done] 直接抓取成功，总控结束。")
                        return 0

                    consecutive_grasp_failures += 1
                    print(
                        f"[retry] 直接抓取失败（exit={grasp_status}），连续失败 "
                        f"{consecutive_grasp_failures}/{args.max_grasp_failures}。"
                    )
                    release_gripper(args, "直接抓取失败")
                    move_to_saved_pose(3, args)
                    if consecutive_grasp_failures >= args.max_grasp_failures:
                        force_reorient = True
                        print("[decision] 已连续抓取失败达到上限；下一次观察后强制执行调整。")
                    continue

                reason = (
                    f"当前 Q{quadrant}、yaw={yaw_deg:.2f}° 不满足直接抓取条件"
                    if not force_reorient
                    else f"已连续直接抓取失败 {consecutive_grasp_failures} 次"
                )
                print(f"[decision] {reason}；按 3 -> 10 -> 6 执行调整准备。")
                move_to_saved_pose(10, args)
                move_to_saved_pose(6, args)
                reorient_status = reorient(pose_file, args)
                if reorient_status != 0:
                    print(f"[retry] 调整失败（exit={reorient_status}）；打开夹爪后返回观察位。")
                    release_gripper(args, "调整失败")
                else:
                    print("[motion] 调整完成；连续抓取失败计数清零。")
                    consecutive_grasp_failures = 0
                # The forced-adjustment request is consumed by this attempt.
                # Let the next fresh observation decide again; in particular,
                # a Q1 workpiece must go straight to direct grasp.
                force_reorient = False

                print("[motion] 按 6 -> 10 返回；到达 10.json 后拍照，并在 10 -> 3 回程中完成计算。")
                move_to_saved_pose(10, args)
                # Start the next observation only after 10 is actually reached;
                # its calculation overlaps the remaining return to pose 3.
                observation_at_pose_10 = True

            raise RuntimeError(
                f"达到 --max-cycles={args.max_cycles}，仍未成功抓取工件 {args.workpiece}。"
            )
        finally:
            ACTIVE_LOG = None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断；不会继续下发后续动作。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
