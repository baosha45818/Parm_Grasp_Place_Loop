#!/usr/bin/env python3
"""只通过 modules 业务入口运行双臂 Grasp-and-Place 大 Loop。

并发调度、共享状态、ServoPose 互斥和 Ctrl+C 停止链路复用已经实机验证的
simple_dual_arm_loop.py；本文件只替换业务模块入口，不修改旧总控。
"""

from __future__ import annotations

import importlib
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
LEGACY_LOOP_DIR = ROOT / "tests/loop"
LEGACY_LOOP = LEGACY_LOOP_DIR / "simple_dual_arm_loop.py"
PYTHON = ROOT / ".venv/bin/python"

MODULE_1 = (
    ROOT
    / "modules/module_1_left_grasp/module_1_3_icp_grasp/"
    "run_grasp_to_inspect_place.sh"
)
MODULE_2_3 = (
    ROOT
    / "modules/module_2_left_inspect_apart/run_inspect_then_place.py"
)
MODULE_4_5 = (
    ROOT
    / "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/main.py"
)
MODULE_5_2 = (
    ROOT
    / "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/handoff/main.py"
)
MODULE_7_8_1 = (
    ROOT
    / "modules/module_8_right_grasp/module_8_1_buffer_grasp/"
    "workpiece_pipeline.py"
)
MODULE_8_2 = (
    ROOT
    / "modules/module_8_right_grasp/module_8_2_left_handoff/main.py"
)
MODULE_9 = ROOT / "modules/module_9_right_place/place_hole_wiggle.py"
MODULE_10 = ROOT / "modules/module_10_right_idle/main.py"
RIGHT_IDLE_POSE = (
    ROOT / "modules/module_10_right_idle/right_idle_pose.json"
)
BASE_MOTION_SPEED = 0.1
DEFAULT_LEFT_SPEED = 0.4
DEFAULT_RIGHT_SPEED = 0.3

# multiprocessing spawn 会在子进程重新导入 target 所在模块；因此必须在模块
# 顶层就让旧调度器目录可被标准 import 找到，不能使用临时动态模块名。
for import_path in (ROOT, LEGACY_LOOP_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tool.runtime_log import RuntimeLogCapture
from loop_types import ResultStatus
from modules.module_status import (
    report as report_module_status,
    report_exception as report_module_exception,
)
from modules.module_0_prepare.ensure_camera_broker import (
    ensure_camera_broker,
    stop_camera_broker,
)
from modules.module_0_prepare.ensure_sam2_service import ensure_sam2_service, sam2_url
from modules.module_0_prepare.preflight_robot_infrastructure import (
    preflight_robot_infrastructure,
)
from xianxin_runtime import configure_environment


def load_scheduler():
    """以稳定模块名加载旧调度器，保证 spawn 子进程可反序列化 worker。"""
    return importlib.import_module("simple_dual_arm_loop")


def prepare_both_arms_in_parallel(scheduler, args) -> None:
    """模块化入口专用：左右臂归位并行执行，旧总控文件保持不变。"""
    left_only = bool(getattr(args, "left_only", False))
    report_module_status(
        "prepare_both_arms",
        ResultStatus.WAIT,
        (
            "单左臂准备：仅左夹爪打开、左臂回 graspup；右侧不发送命令"
            if left_only
            else "开始并行准备：左臂回 graspup、右臂回 Idle"
        ),
    )
    try:
        scheduler.open_both_grippers_for_prepare(args)
    except BaseException as exc:
        report_module_exception(
            "prepare_both_arms",
            exc,
            action="夹爪准备失败，拒绝启动 loop 并保留当前机械状态",
            target_module="initial_state",
            recovery_level="SAFE_STOP",
            reason_code="GRIPPER_PREPARE_FAILED",
            traceback_text="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )
        raise
    config = scheduler.LoopConfig(
        planner_host=args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        base_frame="body_base",
        left_group="left_arm",
        right_group="right_arm",
        left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end",
        max_velocity=float(args.prepare_speed),
        max_acceleration=float(args.prepare_speed),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    )
    # ``--prepare-speed`` is the modular left-arm speed.  Keep the right arm
    # at its established startup speed; the legacy scheduler otherwise shares
    # this argument between both arms.
    right_prepare_config = replace(
        config,
        max_velocity=float(args.right_speed),
        max_acceleration=float(args.right_speed),
    )
    failures: list[tuple[str, BaseException]] = []
    failure_lock = threading.Lock()

    def record_failure(label: str, exc: BaseException) -> None:
        with failure_lock:
            failures.append((label, exc))

    def prepare_left() -> None:
        report_module_status(
            "prepare_left_graspup",
            ResultStatus.WAIT,
            "开始左臂启动归位",
        )
        planner = None
        try:
            planner = scheduler.PlannerRuntime(config)
            planner.open(required_frames={"body_base", "left_hand_transfer_end"})
            error_deg = scheduler.current_joint_target_error_deg(
                args.arm_host, 1, scheduler.LEFT_GRASPUP_JOINTS_DEG
            )
            if error_deg <= 0.5:
                print(
                    "[prepare] 左臂已在 graspup，跳过零长度 RRTstar"
                    f"（误差 {error_deg:.3f}deg）。",
                    flush=True,
                )
                report_module_status(
                    "prepare_left_graspup",
                    ResultStatus.SUCCESS,
                    f"左臂已在 graspup；关节误差={error_deg:.3f}deg",
                )
                return
            print("[prepare] 左臂并行规划关节 RRTstar 到 graspup。", flush=True)
            planner.set_next_plan_context("left_startup_current_pose", "left_graspup")
            planner.move_left_joints(
                scheduler.LEFT_GRASPUP_JOINTS_DEG,
                planner="rrtstar",
            )
            scheduler.wait_for_joint_target(
                args.arm_host, 1, scheduler.LEFT_GRASPUP_JOINTS_DEG, "左臂 graspup"
            )
        except BaseException as exc:
            report_module_exception(
                "prepare_left_graspup",
                exc,
                action="拒绝启动任务；结束并行准备后 cancel 两臂任务，保留现场排查",
                target_module="initial_state",
                recovery_level="SAFE_STOP",
                reason_code="LEFT_PREPARE_FAILED",
                traceback_text="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            record_failure("左臂 graspup", exc)
        else:
            report_module_status(
                "prepare_left_graspup",
                ResultStatus.SUCCESS,
                "左臂已到 graspup",
            )
        finally:
            if planner is not None:
                planner.close()

    def prepare_right() -> None:
        report_module_status(
            "prepare_right_idle",
            ResultStatus.WAIT,
            "开始右臂启动归位",
        )
        planner = None
        try:
            planner = scheduler.PlannerRuntime(right_prepare_config)
            planner.open(required_frames={"body_base", "right_hand_transfer_end"})
            right_idle = scheduler.load_right_idle_joints(args.right_idle_pose)
            error_deg = scheduler.current_joint_target_error_deg(
                args.arm_host, 2, right_idle
            )
            if error_deg <= 0.5:
                print(
                    "[prepare] 右臂已在 Idle，跳过所有右臂启动运动"
                    f"（误差 {error_deg:.3f}deg）。",
                    flush=True,
                )
                report_module_status(
                    "prepare_right_idle",
                    ResultStatus.SUCCESS,
                    f"右臂已在 Idle；关节误差={error_deg:.3f}deg",
                )
                return
            print("[prepare] 右臂并行尝试关节 RRTstar 到 Idle。", flush=True)
            try:
                planner.set_next_plan_context("right_startup_current_pose", "right_idle")
                plan = planner.plan_joint_trajectory(
                    "right",
                    right_idle,
                    planner="rrtstar",
                )
            except Exception as direct_error:
                print(
                    "[prepare] 右臂直接到 Idle 规划失败，改走启动过渡位："
                    f"{direct_error}",
                    flush=True,
                )
                transition = scheduler.reverse_handoff_prepare_joints_deg()
                transition_error = scheduler.current_joint_target_error_deg(
                    args.arm_host, 2, transition
                )
                if transition_error > 0.5:
                    planner.set_next_plan_context(
                        "right_startup_current_pose", "right_startup_transition"
                    )
                    planner.move_right_joints(transition, planner="rrtstar")
                    scheduler.wait_for_joint_target(
                        args.arm_host, 2, transition, "右臂启动过渡位"
                    )
                planner.set_next_plan_context("right_startup_transition", "right_idle")
                planner.move_right_joints(right_idle, planner="rrtstar")
            else:
                if plan is None:
                    raise RuntimeError("实时模式下右臂直接到 Idle 的规划结果为空")
                planner.execute_planned_result(plan)
            scheduler.wait_for_joint_target(args.arm_host, 2, right_idle, "右臂 Idle")
        except BaseException as exc:
            report_module_exception(
                "prepare_right_idle",
                exc,
                action="拒绝启动任务；结束并行准备后 cancel 两臂任务，保留现场排查",
                target_module="initial_state",
                recovery_level="SAFE_STOP",
                reason_code="RIGHT_PREPARE_FAILED",
                traceback_text="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            record_failure("右臂 Idle", exc)
        else:
            report_module_status(
                "prepare_right_idle",
                ResultStatus.SUCCESS,
                "右臂已到 Idle",
            )
        finally:
            if planner is not None:
                planner.close()

    left_thread = threading.Thread(target=prepare_left, name="prepare-left")
    right_thread = None if left_only else threading.Thread(target=prepare_right, name="prepare-right")
    print(
        "[prepare] LEFT-ONLY：仅左臂归位，完全跳过右臂准备。"
        if left_only
        else "[prepare] 左臂 graspup 与右臂 Idle 将并行归位。",
        flush=True,
    )
    left_thread.start()
    if right_thread is not None:
        right_thread.start()
    left_thread.join()
    if right_thread is not None:
        right_thread.join()
    if failures:
        label, exc = failures[0]
        report_module_exception(
            "prepare_both_arms",
            exc,
            action="准备阶段失败，拒绝进入 loop；旧调度器将 cancel 两臂任务",
            target_module="initial_state",
            recovery_level="SAFE_STOP",
            reason_code="PARALLEL_PREPARE_FAILED",
        )
        raise RuntimeError(f"并行准备失败：{label}: {exc}") from exc
    print(
        "[prepare] 左臂已到启动位；右臂未操作，开始单左臂流程。"
        if left_only
        else "[prepare] 左右臂均已到启动位，开始 loop。",
        flush=True,
    )
    report_module_status(
        "prepare_both_arms",
        ResultStatus.SUCCESS,
        "左臂已到启动位；右侧未发送命令" if left_only else "左右臂均已到初始启动位",
    )


def configure_module_entries(scheduler) -> None:
    required = [
        PYTHON,
        MODULE_1,
        MODULE_2_3,
        MODULE_4_5,
        MODULE_5_2,
        MODULE_8_2,
        MODULE_10,
        RIGHT_IDLE_POSE,
    ]
    required.append(MODULE_7_8_1)
    required.append(MODULE_9)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "模块化总控缺少入口:\n" + "\n".join(str(path) for path in missing)
        )

    scheduler.PYTHON = PYTHON
    scheduler.GRASPNET_FLOW_WORKDIR = MODULE_1.parent
    scheduler.GRASPNET_FLOW = MODULE_1
    scheduler.LEFT_FLOW_SCRIPTS = {"apart": MODULE_2_3}
    scheduler.LEFT_PICK = MODULE_4_5
    scheduler.LEFT_HANDOFF_PREP = MODULE_5_2
    # 新的闭环小总控在自身内部按 --workpiece 切换 Module 4-2、7、8-1。
    # 旧模块目录和旧小总控保留，但不再由模块化大总控调用。
    scheduler.RIGHT_PUSH_TO_Q1 = MODULE_7_8_1
    scheduler.RIGHT_LEFT_HANDOFF = MODULE_8_2
    # Module 9 由统一入口按 --workpiece 读取对应 TOML profile。
    scheduler.RIGHT_PLACE_WIGGLE = MODULE_9
    scheduler.RIGHT_FINISH = MODULE_10


def inject_fixed_module_arguments() -> None:
    if "--left-flow" not in sys.argv:
        sys.argv.extend(["--left-flow", "apart"])
    if "--right-idle-pose" not in sys.argv:
        sys.argv.extend(["--right-idle-pose", str(RIGHT_IDLE_POSE)])
    _replace_cli_option(
        "--prepare-speed",
        # ``inject_fixed_module_arguments`` is also used by offline contract
        # checks before ``configure_motion_speeds`` publishes this variable.
        os.environ.get("PARM_LEFT_ARM_SPEED", f"{DEFAULT_LEFT_SPEED:g}"),
    )


def _replace_cli_option(flag: str, value: str) -> None:
    """Make the modular speed authoritative over legacy scheduler defaults."""
    for index, argument in enumerate(sys.argv):
        if argument == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{flag} 缺少数值")
            sys.argv[index + 1] = value
            return
        if argument.startswith(f"{flag}="):
            sys.argv[index] = f"{flag}={value}"
            return
    sys.argv.extend([flag, value])


def _consume_cli_float(flag: str, default: float) -> float:
    """Read and remove a modular-only numeric option before legacy parsing."""
    for index, argument in enumerate(sys.argv):
        if argument == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{flag} 缺少数值")
            value = float(sys.argv[index + 1])
            del sys.argv[index:index + 2]
            return value
        if argument.startswith(f"{flag}="):
            value = float(argument.split("=", 1)[1])
            del sys.argv[index]
            return value
    return default


def configure_motion_speeds() -> None:
    """Publish one normalized speed control for every left/right submodule."""
    left_speed = _consume_cli_float("--left-speed", DEFAULT_LEFT_SPEED)
    right_speed = _consume_cli_float("--right-speed", DEFAULT_RIGHT_SPEED)
    for name, speed in (("--left-speed", left_speed), ("--right-speed", right_speed)):
        if not 0.0 < speed <= 1.0:
            raise ValueError(f"{name} 必须在 (0, 1] 内，当前为 {speed}")
    os.environ["PARM_LEFT_ARM_SPEED"] = f"{left_speed:g}"
    os.environ["PARM_RIGHT_ARM_SPEED"] = f"{right_speed:g}"
    os.environ["PARM_LEFT_SPEED_SCALE"] = f"{left_speed / BASE_MOTION_SPEED:g}"
    os.environ["PARM_RIGHT_SPEED_SCALE"] = f"{right_speed / BASE_MOTION_SPEED:g}"
    _replace_cli_option("--right-speed", f"{right_speed:g}")


def extract_workpiece_argument() -> None:
    """Consume the modular-only workpiece selector before legacy argv parsing."""
    workpiece: str | None = None
    for index, argument in enumerate(sys.argv):
        if argument == "--workpiece":
            if index + 1 >= len(sys.argv):
                raise ValueError("--workpiece 缺少工件编号（1 或 2）")
            workpiece = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            break
        if argument.startswith("--workpiece="):
            workpiece = argument.split("=", 1)[1]
            del sys.argv[index]
            break
    if workpiece is None:
        if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
            # Help must remain available without satisfying production-only
            # selectors.  A temporary value is needed only so the entry map
            # can be assembled before the legacy ArgumentParser prints help.
            os.environ["PARM_LOOP_WORKPIECE"] = "1"
            return
        raise ValueError("必须显式指定 --workpiece 1 或 --workpiece 2")
    if workpiece not in {"1", "2"}:
        raise ValueError(f"--workpiece 只能是 1 或 2，当前为 {workpiece!r}")
    os.environ["PARM_LOOP_WORKPIECE"] = workpiece


def extract_inspect_prompt_mode_argument() -> None:
    """Consume the Module-2-only prompt selector before legacy argv parsing."""
    mode: str | None = None
    for index, argument in enumerate(sys.argv):
        if argument == "--inspect-prompt-mode":
            if index + 1 >= len(sys.argv):
                raise ValueError("--inspect-prompt-mode 缺少模式")
            mode = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            break
        if argument.startswith("--inspect-prompt-mode="):
            mode = argument.split("=", 1)[1]
            del sys.argv[index]
            break
    if mode is not None and mode not in {"auto", "manual", "auto-then-manual"}:
        raise ValueError(f"--inspect-prompt-mode 无效：{mode!r}")
    if mode is None:
        os.environ.pop("PARM_INSPECT_PROMPT_MODE", None)
    else:
        os.environ["PARM_INSPECT_PROMPT_MODE"] = mode


def extract_graspnet_confirmation_argument() -> None:
    """Consume the modular-only flag before the unchanged legacy scheduler parses argv."""
    if "--auto-confirm-real-grasp" in sys.argv:
        sys.argv.remove("--auto-confirm-real-grasp")
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "1"
    elif "--no-auto-confirm-real-grasp" in sys.argv:
        sys.argv.remove("--no-auto-confirm-real-grasp")
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "0"


def extract_graspnet_visualization_argument() -> None:
    """Keep Open3D visibility and real-grasp confirmation in one mode switch."""
    if "--vis" in sys.argv:
        sys.argv.remove("--vis")
        os.environ["GRASPNET_VISUALIZATION"] = "1"
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "0"
        print(
            "[modular-loop] GraspNet预测可视化已启用；"
            "关闭Open3D窗口后需在终端人工确认真实抓取。",
            flush=True,
        )
    elif "--no-vis" in sys.argv:
        sys.argv.remove("--no-vis")
        os.environ["GRASPNET_VISUALIZATION"] = "0"
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "1"
    else:
        # Default to the same unattended mode as ``--no-vis``.
        os.environ["GRASPNET_VISUALIZATION"] = "0"
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "1"


def main() -> None:
    configure_environment()
    # 必须先消费选择参数：后续 Module 5-2/8-2/9 的入口绑定依赖此环境值。
    # 工件类型不可由历史默认值猜测，必须由本次调用显式给出。
    # The managed production entry owns SAM2 for the complete Loop lifetime.
    # Business subprocesses inherit this flag and may recover the resident
    # process if its Unix socket disappears between cycles.
    os.environ["PARM_SAM2_AUTO_RECOVER"] = (
        "0"
        if os.environ.get("PARM_LOOP_REQUIRE_PRESTARTED_SERVICES") == "1"
        else "1"
    )
    extract_workpiece_argument()
    extract_inspect_prompt_mode_argument()
    # 先完成纯命令行解析。这样 ``--help`` 与无效参数不会启动相机/SAM2/
    # GraspNet 服务，更不会进入任何硬件预检。
    scheduler = load_scheduler()
    configure_module_entries(scheduler)
    extract_graspnet_confirmation_argument()
    extract_graspnet_visualization_argument()
    configure_motion_speeds()
    inject_fixed_module_arguments()
    readiness_args = scheduler.parse_args()
    # Current production flow uses traditional ICP Module 1's wrist D405 only.
    # Keep Femto disabled so the chest camera is neither initialized nor held.
    ensure_sam2_service()
    ensure_camera_broker(
        require_femto=False,
        expected_d405_serial=os.environ["PARM_D405_SERIAL"],
    )
    os.environ["PARM_SAM2_URL"] = sam2_url()
    print(
        "[module-0] 传统 ICP Module 1 使用 D405-only Camera Broker；"
        "胸部 Femto、Module 2 Inspect 与 GraspNet 服务均不启动。",
        flush=True,
    )
    # ``spawn`` 产生的左右 worker 会重新导入旧调度器；用环境标记让它们
    # 同样恢复 modules 下的业务入口，而不是回退到旧目录中的脚本。
    os.environ["PARM_MODULAR_LOOP_ENTRIES"] = "1"
    # spawn 子进程会重新导入旧调度器，不能继承上面的 Python 全局变量覆盖。
    # 用环境标志让其固定 Module 5 入口转发到模块化反抓链；直接运行旧入口不受影响。
    os.environ["PARM_MODULAR_REVERSE_PICK"] = "1"
    # The legacy scheduler owns the spawned workers.  An environment-backed
    # status file is therefore deliberately used instead of in-memory state.
    # Every child can report the same latest safe recovery point.
    status_file = os.environ.get("PARM_MODULE_STATUS_FILE")
    if not status_file:
        status_file = str(
            ROOT
            / "output"
            / "modular_dual_arm_loop"
            / "latest_module_status.json"
        )
        os.environ["PARM_MODULE_STATUS_FILE"] = status_file
    scheduler.prepare_both_arms = lambda args: prepare_both_arms_in_parallel(
        scheduler, args
    )
    # This precedes the legacy scheduler's prepare stage.  It reads only
    # planner/driver state and never sends a motion, gripper, or force command.
    preflight_robot_infrastructure(
        planner_host=readiness_args.planner_host,
        arm_host=readiness_args.arm_host,
        gripper_host=readiness_args.gripper_host,
        force_sensor_host=os.environ["PARM_FORCE_SENSOR_HOST"],
        left_only=bool(getattr(readiness_args, "left_only", False)),
    )
    print(
        "[modular-loop] 业务入口全部切换到 modules；Inspect=apart；"
        f"workpiece={os.environ['PARM_LOOP_WORKPIECE']}，"
        f"left-speed={os.environ['PARM_LEFT_ARM_SPEED']}，"
        f"right-speed={os.environ['PARM_RIGHT_ARM_SPEED']}",
        flush=True,
    )
    report_module_status(
        "modular_loop",
        ResultStatus.WAIT,
        f"模块化状态机已启动；状态文件={status_file}",
        reason_code="LOOP_STARTED",
    )
    try:
        scheduler.main()
    except BaseException as exc:
        report_module_exception(
            "modular_loop",
            exc,
            action=(
                "总控停止两个 worker 并 cancel 两臂任务；保留现场，"
                "排除故障后重新启动，由准备流程回到初始状态"
            ),
            target_module="initial_state",
            recovery_level="SAFE_STOP",
            reason_code="LOOP_UNHANDLED_EXCEPTION",
            traceback_text="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )
        raise
    else:
        report_module_status(
            "modular_loop",
            ResultStatus.SUCCESS,
            "模块化状态机正常结束",
            reason_code="LOOP_FINISHED",
        )


if __name__ == "__main__":
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        # Parse help before creating a run directory and before entering the
        # cleanup block.  A read-only help request must not stop a Camera
        # Broker owned by another running process.
        main()
    else:
        run_dir = (
            ROOT
            / "output/modular_dual_arm_loop"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        os.environ["PARM_LOOP_RUN_ID"] = run_dir.name
        os.environ["PARM_MODULE_METRICS_DIR"] = str(run_dir)
        os.environ["PARM_MODULE_STATUS_FILE"] = str(
            run_dir / "latest_module_status.json"
        )
        log_path = run_dir / "run.log"
        # Traditional ICP uses this to send its verbose reports to this
        # persistent run log while keeping the scheduler terminal concise.
        os.environ["PARM_LOOP_RUNTIME_LOG_FILE"] = str(log_path)
        with RuntimeLogCapture(log_path):
            print(f"[log] 本次完整运行日志: {log_path}", flush=True)
            print(f"[log] 结构化事件: {run_dir / 'events.jsonl'}", flush=True)
            print(f"[log] 耗时与成功率汇总: {run_dir / 'summary.json'}", flush=True)
            print(f"[log] command: {' '.join(sys.argv)}", flush=True)
            try:
                try:
                    main()
                except SystemExit:
                    raise
                except KeyboardInterrupt:
                    print("[log] 收到 Ctrl+C，运行中断。", flush=True)
                    raise SystemExit(130)
                except BaseException:
                    traceback.print_exc()
                    raise SystemExit(1)
            finally:
                # D405-only Broker 独占腕部相机；总控无论正常结束、异常或 Ctrl+C，
                # 都释放它，避免后续 GCS 出现设备占用错误。
                try:
                    stop_camera_broker()
                except BaseException as exc:
                    print(f"[module-0] WARNING: Camera Broker cleanup failed: {exc}", flush=True)
