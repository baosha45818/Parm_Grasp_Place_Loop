#!/usr/bin/env python3
"""Module 3：把左臂在 ``graspuptop`` 已持有的工件放至固定位置。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from modules.module_3_left_place.placement_config import load_workpiece_parameters
from modules.module_3_left_place.placement_support import (
    build_planner_config,
    load_handoff,
    matrix_pose,
    open_left_gripper,
    parse_args,
    planner_matrix,
    publish_ready,
    read_left_joints_deg,
    validate_runtime_args,
    validate_z_candidates,
    wait_for_left_graspup,
    wait_for_left_joint_target,
    wait_permit,
)
from modules.module_3_left_place.trajectory_cache import (
    load_trajectory_cache,
    save_trajectory_cache,
    trajectory_cache_path,
)
from modules.module_3_left_place.placement_parameters import (
    DISCARD_BELOW_GRASPUP_M,
    GRASPUP_ABOVE_JOINTS_DEG,
    GRASPUP_ABOVE_PLANNING_TIMEOUT_S,
    GRASPUP_JOINTS_DEG,
)
from modules.stage_result import write_stage_result
from tools.planner_runtime import PlannerRuntime


def main() -> int:
    """按“校验→到位→规划→放置→松爪→回程”的顺序执行 Module 3。"""
    # 阶段 1：解析参数、同步 SDK 主机并完成所有静态校验。
    args = parse_args()
    os.environ["PARM_ARM_HOST"] = str(args.arm_host)
    os.environ["PARM_GRIPPER_HOST"] = str(args.gripper_host)
    if args.skip_motion:
        print(
            "[module3] plan-only: robot/gripper motion is disabled; "
            "only Planner feasibility is evaluated",
            flush=True,
        )
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    validate_runtime_args(args)
    # Module 3 既承担 Buffer 放置也承担 Middle 放置；结果契约必须反映实际
    # 子路径，不能让 Middle 的成功结果伪装成 Buffer。
    stage = f"{args.placement_kind}_place"
    ready_stage = f"{args.placement_kind}_place_ready"
    module_id = f"3-{args.placement_kind}-place"
    classification = args.placement_kind.upper()
    placement_branch = f"{classification}_DIRECT"

    # 阶段 2：验证上游 handoff，读取当前放置类型的固定目标。
    upstream_handoff = load_handoff(args.handoff_file, args.placement_kind)

    parameters = load_workpiece_parameters(args.workpiece, args.placement_kind)
    target_xyz = np.asarray(parameters.target_left_eef_xyz_m, dtype=float)
    if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
        raise ValueError("放置 target_left_eef_xyz_m 必须包含三个有限数值")
    target_z_candidates = validate_z_candidates(
        f"[{args.placement_kind}].target_z_candidates_m",
        parameters.target_z_candidates_m,
    )
    release_z_candidates = validate_z_candidates(
        f"[{args.placement_kind}].release_z_candidates_m",
        parameters.release_z_candidates_m,
        allow_empty=True,
    )
    post_graspup_planner = {
        "rrtconnect": "ompl",
        "rrtstar": "rrtstar",
    }[args.post_graspup_planner]
    scheduler = {
        "mode": "sliding_window",
        "max_inflight": int(args.placement_planning_inflight),
        "planner": post_graspup_planner,
        "offline_start_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
        "policy": f"fixed_left_eef_{args.placement_kind}_target",
        "left_placement_frame": "left_eef",
    }
    # 缓存不能只按文件名判断：工件配置、固定起点、规划器类型或速度被修改时，
    # 旧轨迹必须自动失效并重新请求规划器。
    def route_compatibility(route: str) -> dict:
        return {
            "route": route,
            "workpiece": str(args.workpiece),
            "placement_kind": args.placement_kind,
            "target_left_eef_xyz_m": target_xyz.tolist(),
            "target_z_candidates_m": target_z_candidates,
            "graspup_top_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
            "outbound_planner": post_graspup_planner,
            "motion_speed": float(args.motion_speed),
            "return_speed": float(args.preset_speed),
        }

    outbound_route = f"graspuptop_to_{args.placement_kind}"
    return_route = f"{args.placement_kind}_to_graspuptop"
    trajectory_cache_events: list[dict] = []
    planning_attempts: list[dict] = []
    placement_trace_path = args.result_file.parent / "placement_planning_attempts.jsonl"
    os.environ["PARM_PLANNING_RECORD_DIR"] = str(
        ROOT / "output/left_place_trajectories" / time.strftime("%Y%m%d_%H%M%S")
    )

    def append_trace(record: dict) -> None:
        """追加一条带调度器上下文的规划诊断记录。"""
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "planning_scheduler": scheduler,
            **record,
        }
        with placement_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    planner_cfg = build_planner_config(args)
    planner = PlannerRuntime(planner_cfg)
    def write_terminal_outcome(outcome: str, **extra) -> None:
        """统一写出提前结束或恢复结束的终态结果。"""
        write_stage_result(
            args.result_file,
            {
                "stage": stage,
                "outcome": outcome,
                "module": "module_3_left_place",
                "workpiece": args.workpiece,
                "placement_kind": args.placement_kind,
                "place_module": module_id,
                "classification": classification,
                "placement_branch": placement_branch,
                "planning_scheduler": scheduler,
                **extra,
            },
        )
        publish_ready(
            args,
            {
                "stage": ready_stage,
                "outcome": outcome,
                "result_path": str(args.result_file),
            },
        )

    def return_to_graspup_after_buffer() -> None:
        """从当前固定放置位沿缓存/新规划主路径回 ``graspup_top`` 再回安全位。"""
        planner.config = replace(
            planner.config,
            max_velocity=float(args.preset_speed),
            max_acceleration=float(args.preset_speed),
        )
        cached, cache_status = load_trajectory_cache(
            workpiece=args.workpiece,
            route=return_route,
            compatibility=route_compatibility(return_route),
        )
        return_plan = None
        save_after_success = False
        event = {
            "route": return_route,
            "direction": "return",
            "cache_status": cache_status,
            "cache_path": str(trajectory_cache_path(args.workpiece, return_route)),
        }
        if cached is not None:
            cached_start = np.asarray(
                ((cached["response"].get("data") or {}).get("trajectory") or [[]])[0],
                dtype=float,
            )
            actual_start = read_left_joints_deg(args.arm_host)
            delta = (actual_start - cached_start + 180.0) % 360.0 - 180.0
            start_error = float(np.max(np.abs(delta)))
            event["cache_start_max_error_deg"] = start_error
            if start_error <= args.offline_start_tolerance_deg:
                return_plan = cached["response"]
                event["execution_source"] = "trajectory_cache"
                print(f"[轨迹缓存] 直接执行 {return_route}", flush=True)
            else:
                # 不从未知起点强行播放离线轨迹；这时重新规划并在成功后覆盖缓存。
                event["execution_source"] = "replan_after_cache_start_mismatch"
                event["cache_start_mismatch"] = True
                save_after_success = True
                print(
                    f"[轨迹缓存] {return_route} 首点误差 {start_error:.4f}deg，重新规划",
                    flush=True,
                )
        if return_plan is None:
            return_plan = planner.plan_joint_trajectory(
                "left", GRASPUP_ABOVE_JOINTS_DEG, planner=post_graspup_planner
            )
            if return_plan is None:
                raise RuntimeError("回 graspup_top 的规划器返回空结果")
            save_after_success = True
            event.setdefault("execution_source", "planner")
        planner.execute_planned_result(return_plan)
        wait_for_left_joint_target(
            args.arm_host, GRASPUP_ABOVE_JOINTS_DEG, label="graspup_top"
        )
        if save_after_success:
            saved_path = save_trajectory_cache(
                workpiece=args.workpiece,
                route=return_route,
                compatibility=route_compatibility(return_route),
                planned_response=return_plan,
            )
            event["cache_saved"] = str(saved_path)
        trajectory_cache_events.append(event)
        planner.move_left_joints(GRASPUP_JOINTS_DEG, planner=post_graspup_planner)
        wait_for_left_graspup(args.arm_host)

    approach_reached = threading.Event()
    approach_finished = threading.Event()
    plan_found = threading.Event()
    planning_finished = threading.Event()
    planning_timed_out = threading.Event()
    approach_errors: list[BaseException] = []
    timeout_errors: list[BaseException] = []
    approach_thread = None
    timeout_thread = None

    try:
        # 阶段 3：把左臂送到 graspup_top。是否有有效 Buffer 工件统一由
        # Module 4 在放置后的实际视觉结果判定，Module 3 不做运输中持件判断。
        planner.open(required_frames={"body_base", "left_eef"})
        if not args.skip_motion:
            def move_to_graspup_top() -> None:
                """真实模式下把持件左臂送到离线规划起点 ``graspup_top``。"""
                motion = PlannerRuntime(planner_cfg)
                try:
                    motion.open(required_frames={"body_base", "left_eef"})
                    motion.config = replace(
                        motion.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    if args.pre_place_middle_joints_deg is not None:
                        middle = [float(value) for value in args.pre_place_middle_joints_deg]
                        motion.set_next_plan_context("module1_handoff", "pre_place_middle")
                        motion.move_left_joints(middle, planner="rrtstar")
                        wait_for_left_joint_target(args.arm_host, middle, label="放置前中间位")
                        motion.set_next_plan_context("pre_place_middle", "left_graspup_top")
                        motion.move_left_joints(GRASPUP_ABOVE_JOINTS_DEG, planner="rrtstar")
                        wait_for_left_joint_target(
                            args.arm_host,
                            GRASPUP_ABOVE_JOINTS_DEG,
                            label="graspup_top",
                        )
                    else:
                        motion.set_next_plan_context("upstream_handoff", "left_graspup_top")
                        motion.move_left_joints(
                            GRASPUP_ABOVE_JOINTS_DEG, planner=post_graspup_planner
                        )
                        wait_for_left_joint_target(
                            args.arm_host, GRASPUP_ABOVE_JOINTS_DEG, label="graspuptop"
                        )
                    approach_reached.set()
                    print("[运动线程] 已到达 graspup_top", flush=True)
                except BaseException as exc:
                    approach_errors.append(exc)
                finally:
                    approach_finished.set()
                    motion.close()

            approach_thread = threading.Thread(
                target=move_to_graspup_top,
                name="module3-graspup-top-motion",
                daemon=False,
            )
            approach_thread.start()

            def recover_after_planning_failure() -> None:
                """到位后长期无可行解时执行安全丢弃并返回 ``graspup``。"""
                while not approach_finished.is_set():
                    if approach_reached.wait(timeout=0.05):
                        break
                if not approach_reached.is_set():
                    return
                deadline = time.monotonic() + GRASPUP_ABOVE_PLANNING_TIMEOUT_S
                while not plan_found.is_set() and not planning_finished.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    plan_found.wait(timeout=min(0.05, remaining))
                if plan_found.is_set():
                    return
                planning_timed_out.set()
                print("[规划失败恢复] 无可行 Buffer 轨迹，执行安全丢弃", flush=True)
                recovery = PlannerRuntime(planner_cfg)
                try:
                    recovery.open(required_frames={"body_base", "left_eef"})
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    recovery.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
                    wait_for_left_graspup(args.arm_host)
                    response = recovery.lookup_transform("body_base", "left_eef")
                    if response is None:
                        raise RuntimeError("规划器没有返回 graspup 末端位姿")
                    discard = planner_matrix(response)
                    discard[2, 3] -= DISCARD_BELOW_GRASPUP_M
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.motion_speed),
                        max_acceleration=float(args.motion_speed),
                    )
                    recovery.move_cartesian_pose(
                        "left", matrix_pose(discard), planner="lin"
                    )
                    open_left_gripper(args)
                    recovery.config = replace(
                        recovery.config,
                        max_velocity=float(args.preset_speed),
                        max_acceleration=float(args.preset_speed),
                    )
                    recovery.move_left_joints(GRASPUP_JOINTS_DEG, planner="rrtstar")
                    wait_for_left_graspup(args.arm_host)
                except BaseException as exc:
                    timeout_errors.append(exc)
                finally:
                    recovery.close()

            timeout_thread = threading.Thread(
                target=recover_after_planning_failure,
                name="module3-planning-failure-recovery",
                daemon=False,
            )
            timeout_thread.start()

        if not args.skip_motion:
            while not approach_reached.wait(timeout=0.05):
                if approach_finished.is_set():
                    if approach_errors:
                        raise RuntimeError(
                            f"到达 graspup_top 失败: {approach_errors[0]}"
                        ) from approach_errors[0]
                    raise RuntimeError("到达 graspup_top 前运动线程已结束")

        # 阶段 4：读取 graspup_top 末端旋转。Buffer 只改变 TCP 平移，
        # 不猜测工件相对夹爪的姿态。
        response = planner.lookup_transform("body_base", "left_eef")
        if response is None:
            raise RuntimeError("规划器没有返回 graspup_top 末端位姿")
        target_rotation = planner_matrix(response)[:3, :3].copy()
        scheduler["orientation_source"] = "body_base_T_left_eef_at_graspup_top"
        scheduler["tool_rpy_deg"] = Rotation.from_matrix(target_rotation).as_euler(
            "xyz", degrees=True
        ).tolist()
        print(
            f"[Buffer] 保持 graspup_top left_eef RPY="
            f"{np.round(scheduler['tool_rpy_deg'], 3).tolist()}",
            flush=True,
        )

        def plan_candidate(target_tool: np.ndarray) -> dict:
            """使用独立 Planner 连接，从固定关节起点规划一个 Buffer 候选。"""
            candidate_planner = PlannerRuntime(planner_cfg)
            try:
                candidate_planner.open(required_frames={"body_base", "left_eef"})
                result = candidate_planner.plan_cartesian_pose(
                    "left",
                    matrix_pose(target_tool),
                    planner=post_graspup_planner,
                    start_joints_deg=GRASPUP_ABOVE_JOINTS_DEG,
                    rrtstar_goal_tolerance=(post_graspup_planner == "rrtstar"),
                    use_left_eef_collision_box=False,
                )
                if result is None:
                    raise RuntimeError("离线规划返回空结果")
                return result
            finally:
                candidate_planner.close()

        # 阶段 5：按配置顺序并发预取 Z 候选；首个成功结果停止补发。
        selected = None
        selected_plan = None
        outbound_cache, outbound_cache_status = load_trajectory_cache(
            workpiece=args.workpiece,
            route=outbound_route,
            compatibility=route_compatibility(outbound_route),
        )
        outbound_cache_save_after_success = False
        outbound_cache_event = {
            "route": outbound_route,
            "direction": "outbound",
            "cache_status": outbound_cache_status,
            "cache_path": str(trajectory_cache_path(args.workpiece, outbound_route)),
        }
        if outbound_cache is not None:
            cached_candidate = (outbound_cache["payload"] or {}).get("selected_candidate")
            if isinstance(cached_candidate, dict) and isinstance(
                cached_candidate.get("target_tool_pose"), dict
            ):
                selected = dict(cached_candidate)
                selected_plan = outbound_cache["response"]
                selected["trajectory_cache"] = {
                    "used": True,
                    "path": str(outbound_cache["path"]),
                }
                cache_record = {
                    "record_type": "trajectory_cache",
                    "route": outbound_route,
                    "cache_status": "hit",
                    "planned": True,
                    "execution_source": "trajectory_cache",
                }
                planning_attempts.append(cache_record)
                append_trace(cache_record)
                plan_found.set()
                outbound_cache_event["execution_source"] = "trajectory_cache"
                print(f"[轨迹缓存] 直接执行 {outbound_route}", flush=True)
            else:
                # 旧版/手工生成的文件即使轨迹字段完整，也不能猜测放置目标。
                outbound_cache_status = "missing_selected_candidate"
                outbound_cache_event["cache_status"] = outbound_cache_status
                print(
                    f"[轨迹缓存] {outbound_route} 缺少选中目标，重新规划",
                    flush=True,
                )
        if selected is None:
            print(
                f"[离线规划] start_state={GRASPUP_ABOVE_JOINTS_DEG}; "
                f"最多{args.placement_planning_inflight}条在飞请求",
                flush=True,
            )
        submitted = 0
        completed = 0
        next_index = 0
        in_flight = {}
        try:
            with ThreadPoolExecutor(
                max_workers=int(args.placement_planning_inflight),
                thread_name_prefix="module3-buffer-plan",
            ) as executor:
                while in_flight or next_index < len(target_z_candidates):
                    while (
                        selected is None
                        and len(in_flight) < int(args.placement_planning_inflight)
                        and next_index < len(target_z_candidates)
                        and not planning_timed_out.is_set()
                    ):
                        z_index = next_index
                        target_z = float(target_z_candidates[z_index])
                        target_tool = np.eye(4)
                        target_tool[:3, :3] = target_rotation
                        target_tool[:3, 3] = [target_xyz[0], target_xyz[1], target_z]
                        attempt = {
                            "place_module": module_id,
                            "classification": classification,
                            "placement_branch": placement_branch,
                            "priority_index": z_index,
                            "z_index": z_index,
                            "target_left_eef_z_m": target_z,
                            "target_tool_xyz": target_tool[:3, 3].tolist(),
                            "target_kind": "fixed_left_eef",
                            "release_z_candidates_m": release_z_candidates,
                            "planning_start_joints_deg": GRASPUP_ABOVE_JOINTS_DEG,
                            "planning_submission_order": submitted + 1,
                            "planning_inflight_after_submit": len(in_flight) + 1,
                            "use_left_eef_collision_box": False,
                            "target_tool": target_tool,
                        }
                        if float(target_tool[2, 2]) >= 0.0:
                            rejected = {
                                key: value
                                for key, value in attempt.items()
                                if key != "target_tool"
                            }
                            rejected.update(
                                {
                                    "planned": False,
                                    "rejected_before_planning": True,
                                    "error": "left_eef +Z在body_base Z上的投影必须为负",
                                    "target_tool_positive_z_in_body_base": target_tool[
                                        :3, 2
                                    ].tolist(),
                                }
                            )
                            planning_attempts.append(rejected)
                            append_trace(rejected)
                            next_index += 1
                            continue
                        future = executor.submit(plan_candidate, target_tool)
                        in_flight[future] = attempt
                        submitted += 1
                        next_index += 1
                        print(
                            f"[提交 {z_index + 1}/{len(target_z_candidates)}] "
                            f"Buffer left_eef z={target_z:.3f}",
                            flush=True,
                        )
                    if not in_flight:
                        break
                    future = next(as_completed(in_flight))
                    attempt = in_flight.pop(future)
                    completed += 1
                    try:
                        plan = future.result()
                    except Exception as exc:
                        failed = {
                            key: value for key, value in attempt.items() if key != "target_tool"
                        }
                        failed.update(
                            {
                                "planned": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "planning_completion_order": completed,
                                "planning_inflight_after_complete": len(in_flight),
                            }
                        )
                        planning_attempts.append(failed)
                        append_trace(failed)
                        continue
                    successful = {
                        key: value for key, value in attempt.items() if key != "target_tool"
                    }
                    data = plan.get("data") or {}
                    trajectory = data.get("trajectory") or []
                    target_tool = attempt["target_tool"]
                    successful.update(
                        {
                            "planned": True,
                            "planning_completion_order": completed,
                            "planning_inflight_after_complete": len(in_flight),
                            "target_tool_pose": {
                                "position": target_tool[:3, 3].tolist(),
                                "orientation_xyzw": Rotation.from_matrix(
                                    target_tool[:3, :3]
                                ).as_quat().tolist(),
                            },
                            "planned_trajectory_joint_deg": trajectory,
                            "planned_start_joints_deg": (
                                trajectory[0] if trajectory else GRASPUP_ABOVE_JOINTS_DEG
                            ),
                            "planned_end_joints_deg": trajectory[-1] if trajectory else None,
                            "sample_frequency_hz": data.get("sample_frequency"),
                        }
                    )
                    planning_attempts.append(successful)
                    append_trace(successful)
                    if selected is None and not planning_timed_out.is_set():
                        selected = successful
                        selected_plan = plan
                        plan_found.set()
                        print(
                            f"[找到可行Buffer候选] left_eef z="
                            f"{successful['target_left_eef_z_m']:.3f}",
                            flush=True,
                        )
        finally:
            planning_finished.set()

        if approach_thread is not None:
            approach_thread.join()
        if timeout_thread is not None:
            timeout_thread.join()
        if approach_errors:
            raise RuntimeError(f"到达 graspup_top 失败: {approach_errors[0]}") from approach_errors[0]
        if timeout_errors:
            raise RuntimeError(f"规划失败恢复异常: {timeout_errors[0]}") from timeout_errors[0]
        if planning_timed_out.is_set():
            write_terminal_outcome(
                "planning_failed_recovered_at_graspup",
                completed_without_error=True,
                planning_attempts=planning_attempts,
                returned_to_graspup=True,
                released=True,
                recovery_action="discard_below_graspup_then_open_left_gripper",
            )
            return 0
        if selected is None or selected_plan is None:
            raise RuntimeError("全部 Buffer Z 候选均规划失败")

        # 阶段 6：真实执行前核对离线轨迹起点；不一致时在线重规划。
        outbound_from_cache = bool(
            isinstance(selected.get("trajectory_cache"), dict)
            and selected["trajectory_cache"].get("used") is True
        )
        if not outbound_from_cache:
            # 本次是由规划器刚刚算出的轨迹，只有真实执行并到位后才允许落盘。
            outbound_cache_save_after_success = True
        planning_mode = "trajectory_cache" if outbound_from_cache else "dry_run"
        if not args.skip_motion:
            trajectory = ((selected_plan.get("data") or {}).get("trajectory") or [])
            if not trajectory:
                raise RuntimeError("离线规划结果缺少可执行轨迹")
            offline_start = np.asarray(trajectory[0], dtype=float)
            if offline_start.shape != (7,) or not np.all(np.isfinite(offline_start)):
                raise RuntimeError("离线轨迹首点不是有效七关节角")
            actual_start = read_left_joints_deg(args.arm_host)
            delta = (actual_start - offline_start + 180.0) % 360.0 - 180.0
            start_error = float(np.max(np.abs(delta)))
            selected["offline_trajectory_first_point_deg"] = offline_start.tolist()
            selected["actual_execution_start_deg"] = actual_start.tolist()
            selected["offline_start_max_error_deg"] = start_error
            if start_error <= args.offline_start_tolerance_deg:
                planning_mode = "trajectory_cache" if outbound_from_cache else "offline"
            else:
                print(
                    f"[离线起点误差{start_error:.4f}deg] 在线重规划",
                    flush=True,
                )
                target_tool = np.eye(4)
                target_tool[:3, :3] = Rotation.from_quat(
                    selected["target_tool_pose"]["orientation_xyzw"]
                ).as_matrix()
                target_tool[:3, 3] = selected["target_tool_pose"]["position"]
                selected_plan = planner.plan_cartesian_pose(
                    "left",
                    matrix_pose(target_tool),
                    planner=post_graspup_planner,
                    rrtstar_goal_tolerance=(post_graspup_planner == "rrtstar"),
                    use_left_eef_collision_box=False,
                )
                if selected_plan is None:
                    raise RuntimeError("在线重规划结果为空")
                planning_mode = "online_replan_after_start_mismatch"
                outbound_cache_save_after_success = True
                outbound_cache_event["execution_source"] = planning_mode
                outbound_cache_event["cache_start_mismatch"] = True
                outbound_cache_event["cache_start_max_error_deg"] = start_error

        # 阶段 7：发布“轨迹已准备”状态，等待可选外部门禁，然后执行放置。
        publish_ready(
            args,
            {
                "stage": ready_stage,
                "outcome": "success",
                "result_path": str(args.result_file),
                "selected_candidate": selected,
            },
        )
        wait_permit(args)
        if not args.skip_motion:
            planner.execute_planned_result(selected_plan)
            executed_trajectory = ((selected_plan.get("data") or {}).get("trajectory") or [])
            if not executed_trajectory:
                raise RuntimeError("已执行的放置轨迹缺少终点关节角")
            wait_for_left_joint_target(
                args.arm_host,
                [float(value) for value in executed_trajectory[-1]],
                label=f"{args.placement_kind}主放置位",
            )
            if outbound_cache_save_after_success:
                saved_path = save_trajectory_cache(
                    workpiece=args.workpiece,
                    route=outbound_route,
                    compatibility=route_compatibility(outbound_route),
                    planned_response=selected_plan,
                    selected_candidate=selected,
                )
                outbound_cache_event["cache_saved"] = str(saved_path)
                outbound_cache_event.setdefault("execution_source", "planner")
            else:
                outbound_cache_event.setdefault("execution_source", "trajectory_cache")
        else:
            outbound_cache_event["execution_source"] = "plan_only_not_saved"
        trajectory_cache_events.append(outbound_cache_event)

        # 阶段 8：到达 Buffer 后处理可选松爪下降，并打开左夹爪。
        target_tool = np.eye(4)
        target_tool[:3, :3] = Rotation.from_quat(
            selected["target_tool_pose"]["orientation_xyzw"]
        ).as_matrix()
        target_tool[:3, 3] = selected["target_tool_pose"]["position"]
        target_tool_z = float(target_tool[2, 3])
        active_release_z = [
            float(value) for value in release_z_candidates if float(value) <= target_tool_z + 1e-9
        ]
        release_required = bool(release_z_candidates)
        if release_required and not active_release_z:
            raise RuntimeError("松爪 Z 候选中没有不高于已选 Buffer Z 的候选")
        release_attempts: list[dict] = []
        release_succeeded = not release_required
        selected_release_tool = target_tool if not release_required else None
        if not release_required:
            record = {
                "record_type": "release_descent",
                "skipped_by_empty_release_z_candidates": True,
                "succeeded": True,
                "error": None,
            }
            release_attempts.append(record)
            append_trace(record)
            print("[松爪位] release_z_candidates_m=[]；在Buffer目标直接松爪", flush=True)
        for release_z in active_release_z:
            release_tool = target_tool.copy()
            release_tool[2, 3] = release_z
            record = {
                "record_type": "release_descent",
                "release_z_m": release_z,
                "release_z_frame": "left_eef",
                "target_tool_xyz": release_tool[:3, 3].tolist(),
                "succeeded": False,
                "error": None,
            }
            try:
                if not args.skip_motion:
                    planner.move_cartesian_pose(
                        "left", matrix_pose(release_tool), planner="lin"
                    )
                record["succeeded"] = True
                release_succeeded = True
                selected_release_tool = release_tool
                release_attempts.append(record)
                append_trace(record)
                break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                release_attempts.append(record)
                append_trace(record)
                print(f"[松爪候选失败] z={release_z}: {record['error']}", flush=True)

        # 下降松爪位只是可选的更低放置高度。主放置轨迹已经成功到达并验证
        # ``target_tool``；若所有下降候选不可达，按部署约定直接在这个已验证
        # 的目标位松爪，绝不因额外 LIN 下降失败而把工件带回 graspup。
        # 保留每次下降失败记录，方便事后调整 release_z_candidates_m。
        release_descent_succeeded = release_succeeded
        release_fallback_at_target = False
        if release_required and not release_succeeded:
            release_fallback_at_target = True
            release_succeeded = True
            selected_release_tool = target_tool
            record = {
                "record_type": "release_at_verified_target_fallback",
                "target_tool_xyz": target_tool[:3, 3].tolist(),
                "succeeded": True,
                "error": None,
                "reason": "all_release_descent_candidates_failed",
            }
            release_attempts.append(record)
            append_trace(record)
            print(
                "[松爪回退] 所有下降候选不可达；已在验证到达的目标位直接松爪。",
                flush=True,
            )

        final_position = None
        # 阶段 9：确认松爪位有效后打开夹爪，再按 Buffer → graspup_top →
        # graspup 安全回程。
        if not args.skip_motion:
            final_position = open_left_gripper(args)
        # 只有真的下降过才需要先线性回到主目标位；回退直接松爪时本来就在目标位。
        if release_descent_succeeded and release_required and not args.skip_motion:
            planner.move_cartesian_pose("left", matrix_pose(target_tool), planner="lin")
        if not args.skip_motion:
            return_to_graspup_after_buffer()

        # 阶段 10：写出可供新总控消费的最终结果。
        write_stage_result(
            args.result_file,
            {
                "schema_version": 1,
                "stage": stage,
                "outcome": "success" if args.execute else "planned",
                "module": "module_3_left_place",
                "source_handoff": str(args.handoff_file.resolve()),
                "source_module": upstream_handoff.get("module", "module_1_left_grasp"),
                "workpiece": args.workpiece,
                "placement_kind": args.placement_kind,
                "place_module": module_id,
                "classification": classification,
                "placement_branch": placement_branch,
                "selected_candidate": selected,
                "planning_mode": planning_mode,
                "planning_scheduler": scheduler,
                "planning_attempts": planning_attempts,
                "trajectory_cache": trajectory_cache_events,
                "release_motion_succeeded": release_descent_succeeded,
                "release_fallback_at_target": release_fallback_at_target,
                "release_motion_attempts": release_attempts,
                "release_tool_pose": (
                    {
                        "frame_id": "body_base",
                        "position": selected_release_tool[:3, 3].tolist(),
                        "orientation_xyzw": Rotation.from_matrix(
                            selected_release_tool[:3, :3]
                        ).as_quat().tolist(),
                    }
                    if selected_release_tool is not None
                    else None
                ),
                "gripper_final_position_percent": final_position,
                "released": not args.skip_motion,
                "released_at_verified_target": not args.skip_motion,
                "returned_to_graspup": not args.skip_motion,
            },
        )
        print(
            (
                "[module3] Buffer放置完成，已返回 graspup"
                if args.execute
                else "[module3] Buffer纯规划完成；未发送机械臂或夹爪命令"
            ),
            flush=True,
        )
        return 0
    finally:
        if approach_thread is not None and approach_thread.is_alive():
            approach_thread.join()
        if timeout_thread is not None and timeout_thread.is_alive():
            timeout_thread.join()
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
