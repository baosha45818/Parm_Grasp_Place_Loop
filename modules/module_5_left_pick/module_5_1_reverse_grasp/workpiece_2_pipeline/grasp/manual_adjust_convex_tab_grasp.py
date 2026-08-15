#!/usr/bin/env python3
"""Move to the workpiece_2 grasp point, then manually jog before closing.

This is intentionally separate from execute_convex_tab_grasp.py: the ordinary
automatic grasp path remains unchanged.  All jog vectors are expressed in the
fresh ICP workpiece frame and have a fixed 5 mm magnitude.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from execute_convex_tab_grasp import (  # noqa: E402
    CONFIG, build_targets, check_head, debug_pose, load_json, make_required_grasp_frame,
    nearest_rotation, normalize, pose, validate_pose_quality,
    wait_left, within_here,
)
from modules.module_0_prepare.gripper_state_compat import is_gripper_usable
from modules.module_5_left_pick.reverse_grasp_weight_check import (
    POST_CLOSE_CHECK_LIFT_M,
    capture_open_gripper_baseline,
    verify_object_after_lift,
)


@contextmanager
def raw_keyboard():
    """Yield a terminal in cbreak mode and always restore it."""
    if not sys.stdin.isatty():
        raise RuntimeError("手动微调需要在真实终端中运行，stdin 不能被重定向")
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def read_key() -> str:
    first = os.read(sys.stdin.fileno(), 1)
    if first != b"\x1b":
        return first.decode("utf-8", errors="replace").lower()
    sequence = first
    # Arrow / Page key escape suffixes arrive immediately on a local terminal.
    while select.select([sys.stdin], [], [], .035)[0]:
        sequence += os.read(sys.stdin.fileno(), 1)
    return {
        b"\x1b[A": "up", b"\x1b[B": "down", b"\x1b[C": "right", b"\x1b[D": "left",
        b"\x1b[5~": "pgup", b"\x1b[6~": "pgdn",
    }.get(sequence, "escape")


def args_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="允许移动左臂、张开/闭合夹爪")
    parser.add_argument("--pregrasp-m", type=float, default=.060)
    parser.add_argument("--lift-m", type=float, default=POST_CLOSE_CHECK_LIFT_M,
                        help="固定为 0.03m：闭夹爪后上提3cm再做力传感器判断")
    parser.add_argument("--jog-m", type=float, default=.005, help="每次按键移动量，固定默认 5 mm")
    parser.add_argument("--speed", type=float, default=.10)
    parser.add_argument("--close-force", type=float, default=100.)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--head-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.3.11")
    return parser.parse_args()


def main() -> int:
    args = args_parse()
    if (args.pregrasp_m <= 0 or abs(args.lift_m - POST_CLOSE_CHECK_LIFT_M) > 1e-9
            or args.jog_m <= 0 or not (0 < args.speed <= 1)):
        raise ValueError(f"pregrasp/jog 必须为正，lift 必须固定为 {POST_CLOSE_CHECK_LIFT_M:.2f}m，speed 必须在 (0,1]")
    output = within_here(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
    prediction = load_json(args.prediction_json.expanduser().resolve())
    if prediction.get("event") != "workpiece_2_convex_tab_grasp_prediction":
        raise RuntimeError("prediction-json 不是本项目的凸起抓取点预测")
    pose_source = load_json(Path(prediction["source_pose"]))
    quality = validate_pose_quality(pose_source)
    T_camera_object = np.asarray(pose_source["T_camera_object"], dtype=float)
    object_rotation, rotation_error = nearest_rotation(T_camera_object[:3, :3])
    T_camera_object[:3, :3] = object_rotation
    raw_normal = np.asarray(pose_source.get("table_normal_camera", []), dtype=float)
    table_normal = normalize(raw_normal) if raw_normal.shape == (3,) else None
    T_camera_tcp = make_required_grasp_frame(T_camera_object, table_normal)
    T_camera_tcp[:3, 3] = np.asarray(prediction["grasp_contact_point_camera_m"], dtype=float)
    calibration = load_json(CONFIG / "body_head_calibration.json")
    body_head = np.asarray(calibration["T"], dtype=float)
    output.mkdir(parents=True)
    report = {
        "event": "workpiece_2_manual_adjust_grasp",
        "mode": "execute" if args.execute else "dry_run",
        "prediction_json": str(args.prediction_json.expanduser().resolve()),
        "pose_quality_gate": quality,
        "grasp_tcp_head_camera": debug_pose(T_camera_tcp),
        "planner_ik_frame": "left_eef",
        "planner_target_mode": "direct_grasp_center_no_legacy_tcp_or_bridge",
        "source_icp_rotation_orthogonality_error": rotation_error,
        "jog_step_m": args.jog_m,
        "jog_frame": "fresh ICP workpiece frame; arrows: +X/-X/+Y/-Y, PageUp/PageDown: +Z/-Z",
        "local_table_normal_down_offset_m": float(prediction.get("grasp_down_offset_m", 0.0)),
        "adjustments": [],
        "created_at": time.time(),
    }
    report_path = output / "manual_adjust_grasp_plan.json"
    try:
        if not quality["accepted"]:
            raise RuntimeError("execution blocked: ICP quality gate rejected pose " + json.dumps(quality, ensure_ascii=False))
        if not args.execute:
            report["status"] = "dry_run_ok_no_hardware_access"
            return 0
        check_head(args.head_host, calibration["head_pose_deg"])
        from loop_config import LoopConfig
        from planner_runtime import PlannerRuntime
        from parm_sdk import Gripper
        from modules.reverse_grasp_handoff_positions import load_workpiece_handoff_profile
        ready = list(load_workpiece_handoff_profile("2")["left_reverse_grasp"]["ready_joints_deg"])
        cfg = LoopConfig(planner_host=args.planner_host, execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
                         base_frame="body_base", left_group="left_arm", right_group="right_arm",
                         left_tool_frame="left_eef", right_tool_frame="right_hand_transfer_end",
                         max_velocity=args.speed, max_acceleration=args.speed, idle_poll_seconds=.2, live=True, poses={})
        runtime = PlannerRuntime(cfg)
        try:
            runtime.open(required_frames={"body_base", "left_eef"})
            runtime.move_left_joints(ready, planner="rrtstar")
            wait_left(args.arm_host, ready, "左臂预备位")
            with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                state = gripper.get_state().get(1)
                if not is_gripper_usable(state):
                    raise RuntimeError("左夹爪不可用或未初始化")
                gripper.move(1, 100., force=args.close_force, speed=80.0, mode="sync", preempt=False, timeout=60.)
            target, above = build_targets(T_camera_tcp, body_head, args.pregrasp_m)
            pre = runtime.plan_cartesian_pose("left", pose(above), planner="rrtstar", start_joints_deg=ready)
            pre_end = ((pre or {}).get("data") or {}).get("trajectory") or []
            if not pre_end or len(pre_end[-1]) != 7:
                raise RuntimeError("预抓取规划没有返回七关节终点")
            descent = runtime.plan_cartesian_pose("left", pose(target), planner="lin", start_joints_deg=pre_end[-1])
            descent_end = ((descent or {}).get("data") or {}).get("trajectory") or []
            if not descent_end or len(descent_end[-1]) != 7:
                raise RuntimeError("抓取位 LIN 规划没有返回七关节终点")
            runtime.execute_planned_result(pre)
            wait_left(args.arm_host, [float(x) for x in pre_end[-1]], "左臂预抓取位")
            runtime.execute_planned_result(descent)
            current_joints = [float(x) for x in descent_end[-1]]
            wait_left(args.arm_host, current_joints, "左臂张开夹爪抓取位")
            current_target = target.copy()
            cumulative_offset_object_m = np.zeros(3, dtype=float)
            # ICP object axes, expressed in body_base.  These axes retain their
            # meaning even when the part is moved or yawed on the tabletop.
            object_axes_body = body_head[:3, :3] @ object_rotation
            controls = {
                "up": ("+X", object_axes_body[:, 0], np.array([1., 0., 0.])),
                "down": ("-X", -object_axes_body[:, 0], np.array([-1., 0., 0.])),
                "right": ("+Y", object_axes_body[:, 1], np.array([0., 1., 0.])),
                "left": ("-Y", -object_axes_body[:, 1], np.array([0., -1., 0.])),
                "pgup": ("+Z", object_axes_body[:, 2], np.array([0., 0., 1.])),
                "pgdn": ("-Z", -object_axes_body[:, 2], np.array([0., 0., -1.])),
            }
            print("\n已到抓取点，夹爪保持张开。按键：↑ +X，↓ -X，→ +Y，← -Y，PgUp +Z，PgDn -Z；每次 5 mm。")
            print("按 c 闭合夹爪并抬升；按 q 取消（不闭合、不再移动）。")
            with raw_keyboard():
                while True:
                    key = read_key()
                    if key == "c":
                        report["offset_at_close_object_m"] = cumulative_offset_object_m.round(8).tolist()
                        report["offset_at_close_object_mm"] = (cumulative_offset_object_m * 1000.).round(3).tolist()
                        report["final_open_grasp_body_base"] = debug_pose(current_target)
                        break
                    if key == "q":
                        report["status"] = "operator_cancelled_at_open_grasp_pose"
                        report["final_open_grasp_body_base"] = debug_pose(current_target)
                        report["offset_at_cancel_object_m"] = cumulative_offset_object_m.round(8).tolist()
                        report["offset_at_cancel_object_mm"] = (cumulative_offset_object_m * 1000.).round(3).tolist()
                        print("已取消：夹爪仍张开，机械臂保持当前位置。")
                        return 0
                    if key not in controls:
                        continue
                    label, direction, object_delta_direction = controls[key]
                    candidate = current_target.copy()
                    candidate[:3, 3] += normalize(direction) * args.jog_m
                    event = {"key": key, "object_axis_direction": label, "delta_m": args.jog_m,
                             "target_body_base": debug_pose(candidate)}
                    try:
                        plan = runtime.plan_cartesian_pose("left", pose(candidate), planner="lin", start_joints_deg=current_joints)
                        end = ((plan or {}).get("data") or {}).get("trajectory") or []
                        if not end or len(end[-1]) != 7:
                            raise RuntimeError("微调 LIN 规划没有返回七关节终点")
                        runtime.execute_planned_result(plan)
                        current_joints = [float(x) for x in end[-1]]
                        wait_left(args.arm_host, current_joints, f"微调 {label}")
                        current_target = candidate
                        event["accepted"] = True
                        cumulative_offset_object_m += object_delta_direction * args.jog_m
                        event["cumulative_offset_object_m"] = cumulative_offset_object_m.round(8).tolist()
                        event["cumulative_offset_object_mm"] = (cumulative_offset_object_m * 1000.).round(3).tolist()
                        print(f"  {label}  {args.jog_m * 1000:.0f} mm")
                    except Exception as exc:
                        event["accepted"] = False
                        event["error"] = str(exc)
                        print(f"  {label} 无法规划，未移动：{exc}")
                    report["adjustments"].append(event)
            weight_baseline = capture_open_gripper_baseline()
            with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                response = gripper.move(1, 0., force=args.close_force, speed=80.0, mode="sync", preempt=False, timeout=60.)
            check_lift = current_target.copy()
            check_lift[2, 3] += POST_CLOSE_CHECK_LIFT_M
            lift_plan = runtime.plan_cartesian_pose("left", pose(check_lift), planner="lin", start_joints_deg=current_joints)
            lift_end = ((lift_plan or {}).get("data") or {}).get("trajectory") or []
            if not lift_end:
                raise RuntimeError("抬升规划没有返回终点")
            runtime.execute_planned_result(lift_plan)
            current_joints = [float(x) for x in lift_end[-1]]
            wait_left(args.arm_host, current_joints, "左臂上提3cm检查位")
            post_lift_force = verify_object_after_lift(weight_baseline)
            report["post_lift_force_check"] = post_lift_force
            if not post_lift_force["accepted"]:
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    gripper.move(1, 100.0, force=100.0, speed=80.0, mode="sync", preempt=False, timeout=60.0)
                raise RuntimeError("手动反抓上提3cm后力传感器未确认抓到工件；已重新张开左夹爪")
            report.update({"status": "manual_adjust_grasp_and_lift_complete", "gripper_close_response": response,
                           "final_grasp_body_base": debug_pose(current_target), "final_lift_body_base": debug_pose(check_lift)})
        finally:
            runtime.close()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"MANUAL_GRASP_REPORT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
