#!/usr/bin/env python3
"""Execute a left-arm grasp of the confirmed workpiece_2 convex tab.

No motion is sent unless ``--execute`` is explicitly supplied.  The Planner target
is the new gripper's direct open-jaw centre frame, ``left_eef`` (the same production
mode as Module 1-1); legacy flange-TCP offsets are intentionally not applied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
PROJECT = HERE.parents[4]
CONFIG = PIPELINE_ROOT / "config"
QUALITY_GATE = HERE / "execution_quality_gate.json"
sys.path[:0] = [str(PROJECT), str(PROJECT / "tool")]

from modules.module_0_prepare.gripper_state_compat import is_gripper_usable


NO_GRASP_EXIT_CODE = 20
LEFT_GRIPPER_HOLDING_POSITION_MAX = 85.0


def within_here(path: Path) -> Path:
    result = path.expanduser().resolve()
    try:
        result.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须在 {HERE}: {result}") from exc
    return result


def pose(T: np.ndarray):
    from loop_types import ObjectPose
    return ObjectPose(frame_id="body_base", position=T[:3, 3].tolist(),
                      orientation=Rotation.from_matrix(T[:3, :3]).as_quat().tolist())


def debug_pose(T: np.ndarray) -> dict:
    return {"translation_m": T[:3, 3].round(8).tolist(),
            "rotation_matrix": T[:3, :3].round(8).tolist(),
            "quaternion_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().round(8).tolist()}


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def nearest_rotation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Project the rounded ICP 3x3 block to the nearest proper rotation."""
    raw = np.asarray(matrix, dtype=float)
    u, _, vt = np.linalg.svd(raw)
    result = u @ vt
    if np.linalg.det(result) < 0.:
        u[:, -1] *= -1.
        result = u @ vt
    error = float(np.max(np.abs(raw.T @ raw - np.eye(3))))
    return result, error


def make_required_grasp_frame(T_camera_object: np.ndarray, table_normal_camera: np.ndarray | None) -> np.ndarray:
    """Return camera<-TCP with jaw sides fixed by the operator's X-axis request.

    Version7/GraspNet: TCP +X is approach down and TCP +Y is jaw opening.
    The signed jaw convention used here is +Y=left finger, -Y=right finger.
    """
    object_rotation, _ = nearest_rotation(T_camera_object[:3, :3])
    approach = (-normalize(table_normal_camera) if table_normal_camera is not None
                else object_rotation[:, 2])             # physical down -> table
    # Keep the requested +X finger side, while enforcing that jaws lie in the
    # table plane and remain perpendicular to the real down direction.
    left_finger = object_rotation[:, 0] - approach * np.dot(object_rotation[:, 0], approach)
    left_finger = normalize(left_finger)
    if abs(float(np.dot(approach, left_finger))) > 1e-4:
        raise RuntimeError("工件 +X 与 +Z 不正交，拒绝构造夹爪姿态")
    side = normalize(np.cross(approach, left_finger))
    left_finger = normalize(np.cross(side, approach))
    T = np.eye(4)
    T[:3, :3] = np.column_stack((approach, left_finger, side))
    if np.linalg.det(T[:3, :3]) < 0.0:
        raise RuntimeError("夹爪姿态不是右手坐标系")
    return T


def planner_tf_target_source(response: dict, source: str, target: str) -> np.ndarray:
    data = (response or {}).get("data") or {}
    xyz, quat = data.get("translation") or {}, data.get("rotation") or {}
    if data.get("source_frame") != source or data.get("target_frame") != target:
        raise RuntimeError(f"Planner TF {source}->{target} 不符合预期: {data}")
    encoded = np.eye(4)
    encoded[:3, :3] = Rotation.from_quat([quat[k] for k in "xyzw"]).as_matrix()
    encoded[:3, 3] = [xyz[k] for k in "xyz"]
    # Same deployed Planner serialization convention as Module 5.
    return np.linalg.inv(encoded)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pose_quality(pose_source: dict) -> dict:
    gate = load_json(QUALITY_GATE)
    icp = pose_source.get("icp") or {}
    if "match_fraction" in icp:
        # Fresh Module 4-1 schema: one selected global full-6DoF ICP result
        # contains all quality metrics directly in its icp object.
        match = icp.get("match_fraction")
        source_schema = "full_6d_icp"
    else:
        # Legacy schema: find the selected yaw trial's rendered depth score.
        yaw_index = int(icp.get("yaw_index", -1))
        trials = pose_source.get("yaw_trials") or []
        trial = next((item for item in trials if int(item.get("yaw_index", -2)) == yaw_index), None)
        match = None if trial is None else ((trial.get("render") or {}).get("match_fraction"))
        source_schema = "legacy_yaw_trials"
    fitness = float(icp.get("fitness", float("nan")))
    rmse = float(icp.get("rmse_m", float("nan")))
    result = {"source_schema": source_schema, "fitness": fitness, "icp_rmse_m": rmse, "depth_match_fraction": match,
              "minimum_fitness": float(gate["minimum_icp_fitness"]),
              "minimum_depth_match_fraction": float(gate["minimum_depth_match_fraction"]),
              "maximum_icp_rmse_m": float(gate["maximum_icp_rmse_m"]),
              "gate_file": str(QUALITY_GATE)}
    result["accepted"] = bool(np.isfinite(fitness) and match is not None
                              and fitness >= result["minimum_fitness"]
                              and float(match) >= result["minimum_depth_match_fraction"]
                              and np.isfinite(rmse) and rmse <= result["maximum_icp_rmse_m"])
    return result


def build_targets(T_camera_grasp_center: np.ndarray, body_head: np.ndarray,
                  pregrasp_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Build direct ``left_eef`` targets for the new Planner/gripper model.

    ``left_eef`` is the Planner's open-jaw grasp centre, exactly like Module
    1-1's ``--planner-direct-grasp-center`` mode.  The visual grasp centre is
    therefore already the desired IK-frame origin; applying the retired
    EEF->TCP or EEF->``left_arm_end_link`` transforms here would introduce an
    approximately centimetre-scale systematic error.
    """
    target = body_head @ T_camera_grasp_center
    above_center = T_camera_grasp_center.copy()
    above_center[:3, 3] -= T_camera_grasp_center[:3, 0] * pregrasp_m
    return target, body_head @ above_center


def wait_left(arm_host: str, joints: list[float], label: str, timeout: float = 90.) -> None:
    from parm_sdk import Arm
    target = np.asarray(joints, dtype=float)
    deadline = time.monotonic() + timeout
    with Arm(host=arm_host) as arm:
        while time.monotonic() < deadline:
            state = arm.get_state() or {}
            actual = np.asarray(((state.get("state") or {}).get("left") or {}).get("position") or [], dtype=float)
            if actual.shape == (7,):
                error = (actual - target + 180.) % 360. - 180.
                if float(np.max(np.abs(error))) <= .5:
                    return
            time.sleep(.05)
    raise RuntimeError(f"{label} 未在 {timeout:.0f}s 内到位")


def wait_gripper_terminal(gripper: object, hand: int, label: str, timeout: float = 15.) -> dict:
    """Wait for the driver's physical terminal state after a gripper command."""
    deadline = time.monotonic() + timeout
    last_state: dict | None = None
    while time.monotonic() < deadline:
        states = gripper.get_state()
        state = states.get(hand) if isinstance(states, dict) else None
        if isinstance(state, dict):
            last_state = state
            motion_state = str(state.get("motion_state", "")).strip().lower()
            if motion_state in {"error", "fault", "failed"}:
                raise RuntimeError(f"{label}失败: {state}")
            if motion_state in {"arrived", "gripping"}:
                return state
        time.sleep(.05)
    raise RuntimeError(f"{label}在 {timeout:.1f}s 内未到终态，最后状态: {last_state}")


def require_left_gripper_holding(state: dict) -> None:
    """A new left gripper holds one object only when its opening is below 85%."""
    position = state.get("position")
    if position is None:
        raise RuntimeError(f"左夹爪闭合状态未提供开度: {state}")
    opening = float(position)
    print(f"左夹爪闭合终态开度={opening:.1f}%", flush=True)
    if opening >= LEFT_GRIPPER_HOLDING_POSITION_MAX:
        print(
            f"NO_GRASP: 左夹爪开度={opening:.1f}% >= "
            f"{LEFT_GRIPPER_HOLDING_POSITION_MAX:.1f}%；不执行抬升。",
            flush=True,
        )
        raise SystemExit(NO_GRASP_EXIT_CODE)


def check_head(head_host: str, expected: dict) -> None:
    from parm_sdk import Head
    with Head(host=head_host) as head:
        state = head.get_state() or {}
    pitch = -float(state["pitch"]["position"])
    yaw = float(state["yaw"]["position"])
    if abs(pitch - float(expected["pitch"])) > .5 or abs(yaw - float(expected["yaw"])) > .5:
        raise RuntimeError(f"云台不在标定观察位：实际 pitch={pitch:.2f}, yaw={yaw:.2f}；期望 {expected}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="显式允许机器人、夹爪与 Planner 运动")
    p.add_argument("--preflight-only", action="store_true", help="仅做 Planner RRTStar+LIN 可达性检查，不运动")
    p.add_argument("--pregrasp-m", type=float, default=.060)
    p.add_argument("--lift-m", type=float, default=.060)
    p.add_argument("--approach-angles-deg", type=float, nargs="+", default=[90., 15., -15.],
                   help="相对桌面的接近角；90 为竖直，15/-15 为绕两指轴侧向接近")
    p.add_argument("--speed", type=float, default=.15)
    p.add_argument("--close-force", type=float, default=100.)
    p.add_argument("--planner-host", default="127.0.0.1")
    p.add_argument("--gripper-host", default="127.0.0.1")
    p.add_argument("--head-host", default="127.0.0.1")
    p.add_argument("--arm-host", default="192.168.88.1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and args.preflight_only:
        raise ValueError("--execute 与 --preflight-only 不能同时使用")
    if (args.pregrasp_m <= 0 or args.lift_m <= 0 or not (0 < args.speed <= 1)
            or not args.approach_angles_deg or any(not 0 < abs(x) <= 90 for x in args.approach_angles_deg)):
        raise ValueError("pregrasp/lift 必须为正，speed 必须在 (0,1]")
    output = within_here(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
    prediction = load_json(args.prediction_json.expanduser().resolve())
    if prediction.get("event") != "workpiece_2_convex_tab_grasp_prediction":
        raise RuntimeError("prediction-json 不是本项目的凸起抓取点预测")
    pose_source = load_json(Path(prediction["source_pose"]))
    quality = validate_pose_quality(pose_source)
    T_camera_object = np.asarray(pose_source["T_camera_object"], dtype=float)
    object_rotation, object_rotation_error = nearest_rotation(T_camera_object[:3, :3])
    T_camera_object[:3, :3] = object_rotation
    point_camera = np.asarray(prediction["grasp_contact_point_camera_m"], dtype=float)
    raw_table_normal = np.asarray(pose_source.get("table_normal_camera", []), dtype=float)
    table_normal = normalize(raw_table_normal) if raw_table_normal.shape == (3,) else None
    T_camera_tcp = make_required_grasp_frame(T_camera_object, table_normal)
    T_camera_tcp[:3, 3] = point_camera
    # With a table-normal approach, the jaw axis is object +X projected onto
    # the table plane. Compare against that required physical direction, not
    # raw 3-D +X (which may have a small vertical component).
    required_jaw = normalize(T_camera_object[:3, 0])
    if table_normal is not None:
        approach = -table_normal
        required_jaw = normalize(required_jaw - approach * np.dot(required_jaw, approach))
    jaw_alignment = float(np.dot(T_camera_tcp[:3, 1], required_jaw))
    if jaw_alignment < .999:
        raise RuntimeError(f"左指 +Y 未对齐工件 +X 的桌面投影: dot={jaw_alignment:.6f}")

    calibration = load_json(CONFIG / "body_head_calibration.json")
    body_head = np.asarray(calibration["T"], dtype=float)
    output.mkdir(parents=True)
    report = {
        "event": "workpiece_2_convex_tab_left_grasp_plan",
        "mode": "execute" if args.execute else ("preflight_only" if args.preflight_only else "dry_run"),
        "prediction_json": str(args.prediction_json.expanduser().resolve()),
        "source_pose": prediction["source_pose"],
        "grasp_frame_convention": "Module-1-1 direct grasp centre: Planner IK frame left_eef; +X approach-into-table, +Y jaw-opening; +Y side is left finger",
        "planner_ik_frame": "left_eef",
        "planner_target_mode": "direct_grasp_center_no_legacy_tcp_or_bridge",
        "finger_side_constraint": {"left_finger": "grasp point object +X", "right_finger": "grasp point object -X"},
        "left_finger_axis_alignment_to_object_x_table_projection": jaw_alignment,
        "grasp_tcp_head_camera": debug_pose(T_camera_tcp),
        "source_icp_rotation_orthogonality_error": object_rotation_error,
        "pose_quality_gate": quality,
        "grasp_contact_camera_m": point_camera.round(8).tolist(),
        "local_table_normal_down_offset_m": float(prediction.get("grasp_down_offset_m", 0.0)),
        "pregrasp_m": args.pregrasp_m,
        "lift_m": args.lift_m,
        "approach_angles_deg": args.approach_angles_deg,
        "created_at": time.time(),
    }
    report_path = output / "grasp_plan.json"
    try:
        if not quality["accepted"]:
            raise RuntimeError("execution blocked: Module-5-1 ICP quality gate rejected pose " + json.dumps(quality, ensure_ascii=False))
        if not args.execute and not args.preflight_only:
            report["status"] = "dry_run_ok_no_hardware_access"
            return 0
        check_head(args.head_host, calibration["head_pose_deg"])
        from loop_config import LoopConfig
        from planner_runtime import PlannerRuntime
        from parm_sdk import Gripper
        ready = [float(x) for x in load_json(CONFIG / "left_reverse_regrasp_ready_joints.json")["joint_angles_deg"]]
        report["preflight_start_joint_angles_deg"] = ready
        cfg = LoopConfig(planner_host=args.planner_host, execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
            base_frame="body_base", left_group="left_arm", right_group="right_arm",
            left_tool_frame="left_eef", right_tool_frame="right_hand_transfer_end",
            max_velocity=args.speed, max_acceleration=args.speed, idle_poll_seconds=.2, live=True, poses={})
        runtime = PlannerRuntime(cfg)
        try:
            runtime.open(required_frames={"body_base", "left_eef"})
            if args.execute:
                runtime.move_left_joints(ready, planner="rrtstar")
                wait_left(args.arm_host, ready, "左臂预备位")
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    state = gripper.get_state().get(1)
                    if not is_gripper_usable(state):
                        raise RuntimeError("左夹爪不可用或未初始化")
                    gripper.move(1, 100., force=args.close_force, mode="sync", preempt=False, timeout=60.)
            # Rotating around TCP +Y leaves the signed jaw direction untouched:
            # left finger remains object +X and right finger object -X.
            attempts, reachable = [], []
            for angle in sorted(dict.fromkeys(args.approach_angles_deg), key=lambda x: -abs(x)):
                tilt_deg = float(np.copysign(90. - abs(angle), angle))
                candidate_tcp = T_camera_tcp.copy()
                candidate_tcp[:3, :3] = T_camera_tcp[:3, :3] @ Rotation.from_rotvec([0., np.deg2rad(tilt_deg), 0.]).as_matrix()
                target, above = build_targets(candidate_tcp, body_head, args.pregrasp_m)
                item = {"table_approach_angle_deg": angle, "side_tilt_about_tcp_y_deg": tilt_deg,
                        "left_finger_axis_alignment": "TCP +Y unchanged = object +X",
                        "pregrasp_body_base": debug_pose(above), "grasp_body_base": debug_pose(target)}
                try:
                    # Preflight must use exactly the state used by real
                    # execution (the fixed reverse-grasp ready pose), not an
                    # arbitrary parked arm state at the time of the check.
                    pre = runtime.plan_cartesian_pose("left", pose(above), planner="rrtstar", start_joints_deg=ready)
                    endpoint = ((pre or {}).get("data") or {}).get("trajectory") or []
                    if not endpoint or len(endpoint[-1]) != 7:
                        raise RuntimeError("预抓取规划没有返回七关节终点")
                    descent = runtime.plan_cartesian_pose("left", pose(target), planner="lin", start_joints_deg=endpoint[-1])
                    item["reachable"] = True
                    item["pregrasp_joint_target_deg"] = [float(x) for x in endpoint[-1]]
                    reachable.append((angle, candidate_tcp, target, above, pre, descent,
                                      [float(x) for x in endpoint[-1]], item))
                except Exception as exc:
                    item["reachable"] = False
                    item["reason"] = str(exc)
                attempts.append(item)
            report["approach_preflight"] = attempts
            if not reachable:
                raise RuntimeError("所有保持左右指方向的接近角均不可达；检查 grasp_plan.json 的 approach_preflight")
            angle, selected_tcp, target, above, pre, descent, pre_endpoint, selected = reachable[0]
            report["selected_approach"] = selected
            report["preflight_status"] = "ptp_pregrasp_and_lin_descent_reachable"
            if args.preflight_only:
                report["status"] = "preflight_ok_no_motion"
                return 0
            runtime.execute_planned_result(pre)
            wait_left(args.arm_host, pre_endpoint, "左臂预抓取位")
            runtime.execute_planned_result(descent)
            descent_end = ((descent or {}).get("data") or {}).get("trajectory") or []
            if not descent_end:
                raise RuntimeError("LIN 下探没有返回终点")
            wait_left(args.arm_host, descent_end[-1], "左臂抓取位")
            with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                response = gripper.move(1, 0., force=args.close_force, mode="sync", preempt=False, timeout=60.)
                state = wait_gripper_terminal(gripper, 1, "左夹爪闭合")
                report["gripper_close_state"] = state
                require_left_gripper_holding(state)
            lift = target.copy(); lift[2, 3] += args.lift_m
            lift_plan = runtime.plan_cartesian_pose("left", pose(lift), planner="lin")
            runtime.execute_planned_result(lift_plan)
            lift_end = ((lift_plan or {}).get("data") or {}).get("trajectory") or []
            if not lift_end:
                raise RuntimeError("抬升规划没有返回终点")
            wait_left(args.arm_host, lift_end[-1], "左臂抬升位")
            report["status"] = "grasp_and_lift_complete"
            report["gripper_close_response"] = response
        finally:
            runtime.close()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"GRASP_REPORT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
