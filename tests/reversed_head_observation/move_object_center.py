#!/usr/bin/env python3
"""Move a left-held object to a bounded handoff pose in body_base.

The ICP result supplies the rigid T_tool_object relation.  The handoff keeps
the object's front (+Z) horizontal/up and points its tail (+X) toward the
robot's right arm (+Y in body_base, verified at the physical workstation),
with only small bounded redundancy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if __name__ == "__main__" and Path(sys.executable) != PYTHON:
    os.execv(
        str(PYTHON),
        [str(PYTHON), "-m", "tests.reversed_head_observation.move_object_center", *sys.argv[1:]],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from loop_config import LoopConfig
from tool.planner_runtime import PlannerRuntime
from tests.reversed_head_observation.handoff_geometry import (
    object_pose_to_transform,
    planner_tf_target_source,
    transform_debug,
    transform_error,
    transform_to_object_pose,
    validate_transform,
)
from tests.reversed_head_observation.preflight_handoff import trajectory_summary


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT = ROOT / "output/module_2_inspect/20260722_165216/result.json"
DEFAULT_CONFIG = HERE / "handoff_config.json"
DEFAULT_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"
DEFAULT_HANDOFF_CENTER_BODY_M = np.asarray([0.42, 0.20, 0.12], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将工件移到固定body_base交接区域，正面朝上且尾部朝右臂"
    )
    parser.add_argument("--result", "--icp-result", dest="result", type=Path,
                        default=DEFAULT_RESULT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--target-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
        help="body_base中的工件中心；默认读取result.placement.target_object_pose.position",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--execute-url", default=DEFAULT_EXECUTE_URL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--speed-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--cartesian-planner",
        choices=("ompl", "rrtstar"),
        default="rrtstar",
        help="左臂交接位姿规划器；rrtstar使用RRTStar，ompl使用RRTConnect",
    )
    parser.add_argument(
        "--exact-handoff-pose",
        action="store_true",
        help="只规划名义中心、正面+Z朝上、尾部+X严格朝body_base +Y的单一位姿",
    )
    parser.add_argument(
        "--fixed-handoff-orientation",
        action="store_true",
        help=(
            "工件+Z严格朝上、+X严格朝body_base +Y；禁止yaw/tilt冗余，"
            "但允许在安全交接区域内搜索中心位置"
        ),
    )
    parser.add_argument(
        "--verbose-prediction-json",
        action="store_true",
        help="调试时打印完整目标、候选和轨迹预测JSON；默认仅打印摘要",
    )
    parser.add_argument(
        "--object-center-tolerance-mm",
        type=float,
        default=6.0,
        help=(
            "执行后相对所选交接候选点的工件中心误差上限；默认6mm，"
            "略高于左TCP的4mm到位门限"
        ),
    )
    parser.add_argument(
        "--reuse-grasp-relation", action="store_true",
        help=("左手从ICP后已移动、但工件始终未松夹且未滑动时，使用实时TCP和"
              "记录的TCP-工件刚性关系重建当前工件位姿"),
    )
    return parser.parse_args()


def centre_target(
    T_base_object_current: np.ndarray,
    T_base_left_tcp_captured: np.ndarray,
    target_center_body_m: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tool/object relation, centre-only object target and left TCP target."""
    current = validate_transform(T_base_object_current, "T_base_object_current")
    left = validate_transform(T_base_left_tcp_captured, "T_base_left_tcp_captured")
    center = np.asarray(target_center_body_m, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("target centre must contain three finite values")
    T_left_tcp_object = np.linalg.inv(left) @ current
    target_object = current.copy()
    target_object[:3, 3] = center
    target_left = target_object @ np.linalg.inv(T_left_tcp_object)
    return T_left_tcp_object, target_object, target_left


def redundant_handoff_candidates(
    T_base_object_current: np.ndarray,
    T_left_tcp_object: np.ndarray,
    nominal_center: np.ndarray,
) -> list[dict[str, Any]]:
    """Build candidates with front up, tail right and <=20 mm position slack."""
    current = validate_transform(T_base_object_current, "T_base_object_current")
    relation = validate_transform(T_left_tcp_object, "T_left_tcp_object")
    center = np.asarray(nominal_center, dtype=float).reshape(3)

    # STL/object convention established by the pose-estimation visualization:
    # local +Z is the visible front-face normal, local +X points to the tail.
    # Physical workstation verification takes precedence over the misleading
    # arm-mount sign inferred from the URDF: the required tail direction toward
    # the operator-identified right arm is body_base +Y.
    nominal_rotation = np.asarray([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    exact_rotations = [
        ("front_up_tail_right", "z", 0.0),
        ("yaw_p5", "z", 5.0), ("yaw_m5", "z", -5.0),
        ("yaw_p10", "z", 10.0), ("yaw_m10", "z", -10.0),
        ("yaw_p15", "z", 15.0), ("yaw_m15", "z", -15.0),
        ("tilt_x_p5", "x", 5.0), ("tilt_x_m5", "x", -5.0),
        ("tilt_y_p5", "y", 5.0), ("tilt_y_m5", "y", -5.0),
    ]
    region_centers = [
        ("back10", [-0.01, 0.0, 0.0]),
        ("forward10", [0.01, 0.0, 0.0]),
        ("left10", [0.0, -0.01, 0.0]),
        ("right10", [0.0, 0.01, 0.0]),
        ("up10", [0.0, 0.0, 0.01]),
        ("back10_up10", [-0.01, 0.0, 0.01]),
        ("back15_up10", [-0.015, 0.0, 0.01]),
        ("left10_up10", [0.0, -0.01, 0.01]),
        ("right10_up10", [0.0, 0.01, 0.01]),
    ]
    region_rotations = [
        ("front_up_tail_right", "z", 0.0),
        ("yaw_p5", "z", 5.0), ("yaw_m5", "z", -5.0),
        ("yaw_p10", "z", 10.0), ("yaw_m10", "z", -10.0),
    ]

    candidates: list[dict[str, Any]] = []
    phases = [("exact", np.zeros(3), exact_rotations)] + [
        (name, np.asarray(offset, dtype=float), region_rotations)
        for name, offset in region_centers
    ]
    for center_name, offset, rotations in phases:
        for rotation_name, axis, degrees in rotations:
            target_object = current.copy()
            target_object[:3, 3] = center + offset
            target_object[:3, :3] = (
                Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
                @ nominal_rotation
            )
            front_error_deg, tail_error_deg = handoff_axis_errors(target_object)
            target_left = target_object @ np.linalg.inv(relation)
            candidates.append({
                "name": f"{center_name}/{rotation_name}",
                "center_offset_m": offset.tolist(),
                "orientation_delta": {
                    "relative_to": "front_up_tail_right",
                    "world_axis": axis,
                    "degrees": degrees,
                },
                "front_up_error_deg": front_error_deg,
                "tail_to_right_error_deg": tail_error_deg,
                "T_base_object_target": target_object,
                "T_base_left_tcp_target": target_left,
            })
    return candidates


def handoff_axis_errors(T_base_object: np.ndarray) -> tuple[float, float]:
    """Return front(+Z)-to-up and tail(+X)-to-right angular errors."""
    rotation = validate_transform(T_base_object, "T_base_object")[:3, :3]
    front_cos = float(np.clip(np.dot(rotation[:, 2], [0.0, 0.0, 1.0]), -1.0, 1.0))
    tail_cos = float(np.clip(np.dot(rotation[:, 0], [0.0, 1.0, 0.0]), -1.0, 1.0))
    return float(np.degrees(np.arccos(front_cos))), float(np.degrees(np.arccos(tail_cos)))


def load_result(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    result = json.loads(path.read_text(encoding="utf-8"))
    pose = result.get("object_pose") or {}
    if pose.get("frame_id") != "body_base":
        raise RuntimeError(
            f"工件坐标系是{pose.get('frame_id')!r}，必须是'body_base'才能规划。"
            "这份结果不能通过关闭检查来执行。请使用与当前云台角匹配的外参，"
            "并以--output-frame body_base重新估计。"
        )
    current_object = object_pose_to_transform(pose, "object_pose")
    placement = result.get("placement") or {}
    relation_value = placement.get("T_tool_object")
    if relation_value is not None:
        relation = validate_transform(relation_value, "placement.T_tool_object")
    else:
        snapshot = result.get("left_tcp_snapshot") or {}
        if snapshot.get("frame_id") != "body_base":
            raise RuntimeError(
                "body_base结果中既没有placement.T_tool_object，也没有同步的"
                "left_tcp_snapshot，无法确定左夹爪与工件的刚性关系"
            )
        captured_tcp = validate_transform(
            snapshot.get("T_base_left_tcp"), "left_tcp_snapshot.T_base_left_tcp"
        )
        relation = validate_transform(
            np.linalg.inv(captured_tcp) @ current_object, "T_left_tcp_object"
        )
    return result, current_object, relation


def result_target_center(result: dict[str, Any]) -> np.ndarray:
    placement = result.get("placement") or {}
    target_pose = placement.get("target_object_pose") or {}
    center = np.asarray(target_pose.get("position"), dtype=float)
    if center.shape == (3,) and np.all(np.isfinite(center)):
        return center
    return DEFAULT_HANDOFF_CENTER_BODY_M.copy()


def captured_left_tcp(
    T_base_object_current: np.ndarray, T_left_tcp_object: np.ndarray
) -> np.ndarray:
    return validate_transform(
        T_base_object_current @ np.linalg.inv(T_left_tcp_object),
        "T_base_left_tcp_at_capture",
    )


def read_left_tcp(planner: PlannerRuntime, frame: str) -> np.ndarray:
    response = planner.lookup_transform(frame, "body_base")
    if response is None:
        raise RuntimeError(f"规划器未返回{frame}的实时TF")
    return planner_tf_target_source(response, frame, "body_base")


def wait_for_left_tcp_target(
    planner: PlannerRuntime,
    frame: str,
    target: np.ndarray,
    *,
    timeout_s: float = 90.0,
    translation_tolerance_mm: float = 4.0,
    rotation_tolerance_deg: float = 2.0,
) -> tuple[float, float]:
    """Wait for real TCP feedback instead of accepting the async execute acknowledgement."""
    deadline = time.monotonic() + timeout_s
    last_translation_mm = float("inf")
    last_rotation_deg = float("inf")
    while time.monotonic() < deadline:
        live = read_left_tcp(planner, frame)
        translation_m, rotation_deg = transform_error(target, live)
        last_translation_mm = translation_m * 1000.0
        last_rotation_deg = rotation_deg
        if (
            last_translation_mm <= translation_tolerance_mm
            and last_rotation_deg <= rotation_tolerance_deg
        ):
            return last_translation_mm, last_rotation_deg
        time.sleep(0.05)
    raise RuntimeError(
        "左臂没有在交接位到位："
        f"{last_translation_mm:.3f}mm/{last_rotation_deg:.3f}deg"
    )


def build_runtime(config: dict[str, Any], args: argparse.Namespace) -> PlannerRuntime:
    return PlannerRuntime(LoopConfig(
        planner_host=args.planner_host,
        execute_url=args.execute_url,
        base_frame="body_base",
        left_group=str(config.get("left_group", "left_arm")),
        right_group=str(config.get("right_group", "right_arm")),
        left_tool_frame=str(config.get("left_tool_frame", "left_hand_transfer_end")),
        right_tool_frame=str(config.get("right_tool_frame", "right_hand_transfer_end")),
        max_velocity=min(
            1.0,
            float(config.get("max_velocity", 0.05)) * args.speed_multiplier,
        ),
        max_acceleration=min(
            1.0,
            float(config.get("max_acceleration", 0.05)) * args.speed_multiplier,
        ),
        idle_poll_seconds=0.2,
        live=True,
        poses={},
    ))


def main() -> int:
    args = parse_args()
    if args.exact_handoff_pose and args.fixed_handoff_orientation:
        raise ValueError(
            "--exact-handoff-pose 与 --fixed-handoff-orientation 不能同时使用"
        )
    if not np.isfinite(args.speed_multiplier) or args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    if (
        not np.isfinite(args.object_center_tolerance_mm)
        or args.object_center_tolerance_mm <= 0.0
    ):
        raise ValueError("--object-center-tolerance-mm must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result, current_object, recorded_relation = load_result(args.result)
    center = np.asarray(args.target_center, dtype=float) if args.target_center else result_target_center(result)

    capture_left = captured_left_tcp(current_object, recorded_relation)
    relation, target_object, target_left = centre_target(
        current_object, capture_left, center.tolist()
    )
    planner = build_runtime(config, args)
    try:
        left_frame = planner.config.left_tool_frame
        planner.open(required_frames={"body_base", left_frame})
        live_left = read_left_tcp(planner, left_frame)
        moved_m, moved_deg = transform_error(capture_left, live_left)
        max_mm = float(config.get("current_vs_capture_tcp_max_translation_mm", 3.0))
        max_deg = float(config.get("current_vs_capture_tcp_max_rotation_deg", 2.0))
        if moved_m * 1000.0 > max_mm or moved_deg > max_deg:
            if not args.reuse_grasp_relation:
                raise RuntimeError(
                    "ICP后左臂位置已改变。若工件一直未松夹且确认没有滑动，可加"
                    " --reuse-grasp-relation；否则请重新做位姿估计。"
                    f" 当前差异={moved_m * 1000.0:.3f} mm/{moved_deg:.3f}°"
                )
            current_object = validate_transform(
                live_left @ recorded_relation, "T_base_object_reconstructed_live"
            )
            capture_left = live_left
            relation = recorded_relation
            _, target_object, target_left = centre_target(
                current_object, live_left, center.tolist()
            )
            print(
                "[警告] 使用记录的TCP-工件刚性关系和实时左TCP重建当前工件位姿；"
                "仅在左夹爪从ICP后始终未松开且工件没有滑动时有效。",
                flush=True,
            )

        candidates = redundant_handoff_candidates(current_object, relation, center)
        if args.exact_handoff_pose:
            candidates = [candidates[0]]
        elif args.fixed_handoff_orientation:
            candidates = [
                candidate for candidate in candidates
                if abs(float(candidate["orientation_delta"]["degrees"])) < 1e-9
            ]
        nominal_candidate = candidates[0]
        nominal_target_object = nominal_candidate["T_base_object_target"]
        first_target_left = nominal_candidate["T_base_left_tcp_target"]
        preplan_report = {
            "result": "CENTER_MOVE_TARGET_BEFORE_PLANNING",
            "source_result": str(args.result.resolve()),
            "planner": args.cartesian_planner,
            "planning_group": planner.config.left_group,
            "ik_frame": left_frame,
            "reference_frame": "body_base",
            "current_object_pose": transform_debug(current_object),
            "current_tail_axis_body": current_object[:3, 0].tolist(),
            "nominal_target_object_pose": transform_debug(nominal_target_object),
            "nominal_target_tail_axis_body": nominal_target_object[:3, 0].tolist(),
            "handoff_redundancy": {
                "enabled": not args.exact_handoff_pose,
                "fixed_orientation": bool(args.fixed_handoff_orientation),
                "candidate_count": len(candidates),
                "exact_center_orientation_candidates": sum(
                    candidate["name"].startswith("exact/") for candidate in candidates
                ),
                "region_center_offsets_m": sorted({
                    tuple(candidate["center_offset_m"])
                    for candidate in candidates
                    if candidate["name"].split("/", 1)[0] != "exact"
                }),
                "no_negative_z_offset": True,
                "max_center_offset_m": max(
                    np.linalg.norm(candidate["center_offset_m"])
                    for candidate in candidates
                ),
                "object_axis_constraints": {
                    "front_axis": "+Z",
                    "front_target_body_axis": "+Z (up)",
                    "tail_axis": "+X",
                    "tail_target_body_axis": "+Y (physically verified right arm side)",
                    "right_arm_side_body_axis": [0.0, 1.0, 0.0],
                },
            },
            "captured_left_tcp": transform_debug(capture_left),
            "live_left_tcp": transform_debug(live_left),
            "live_vs_capture_error": {
                "translation_mm": moved_m * 1000.0,
                "rotation_deg": moved_deg,
            },
            "first_requested_left_tcp_pose": transform_debug(first_target_left),
        }
        if args.exact_handoff_pose:
            print(
                "[规划请求] 使用RRTConnect搜索单一精确交接位姿："
                "工件+Z朝上、+X严格朝body_base +Y；不允许yaw或位置冗余。"
                if args.cartesian_planner == "ompl"
                else
                "[规划请求] 使用RRTStar搜索单一精确交接位姿："
                "工件+Z朝上、+X严格朝body_base +Y；不允许yaw或位置冗余。"
                if args.cartesian_planner == "rrtstar"
                else
                "[规划请求] 搜索单一精确交接位姿；不允许yaw或位置冗余。"
            )
        elif args.fixed_handoff_orientation:
            print(
                f"[规划请求] 使用{args.cartesian_planner.upper()}搜索固定朝向交接区域："
                "工件+Z严格朝上、+X严格朝body_base +Y；"
                "禁止yaw/tilt，仅搜索中心位置。"
            )
        else:
            print("[规划请求] 将按以下交接区域候选顺序搜索左臂笛卡尔目标：")
        if args.verbose_prediction_json:
            print(json.dumps(preplan_report, indent=2, ensure_ascii=False), flush=True)
        plan = None
        selected = None
        failures = []
        for index, candidate in enumerate(candidates, start=1):
            target_left = candidate["T_base_left_tcp_target"]
            target_object = candidate["T_base_object_target"]
            if args.verbose_prediction_json:
                print(json.dumps({
                    "candidate": f"{index}/{len(candidates)}",
                    "name": candidate["name"],
                    "object_center_body_m": target_object[:3, 3].tolist(),
                    "center_offset_m": candidate["center_offset_m"],
                    "orientation_delta": candidate["orientation_delta"],
                    "front_up_error_deg": candidate["front_up_error_deg"],
                    "tail_to_right_error_deg": candidate["tail_to_right_error_deg"],
                    "target_tail_axis_body": target_object[:3, 0].tolist(),
                    "target_tail_alignment_with_right_arm": float(
                        np.dot(target_object[:3, 0], [0.0, 1.0, 0.0])
                    ),
                    "left_tcp_target": transform_debug(target_left),
                }, ensure_ascii=False), flush=True)
            try:
                candidate_plan = planner.plan_cartesian_pose(
                    "left", transform_to_object_pose(target_left),
                    planner=args.cartesian_planner,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"name": candidate["name"], "error": error})
                print(f"[候选失败] {candidate['name']}: {error}", flush=True)
                continue
            if candidate_plan is not None:
                plan = candidate_plan
                selected = candidate
                print(f"[候选成功] {candidate['name']}", flush=True)
                break
        if plan is None or selected is None:
            raise RuntimeError(
                f"全部{len(candidates)}个交接区域候选均规划失败；"
                f"最后错误={failures[-1] if failures else 'planner returned no result'}"
            )
        target_object = selected["T_base_object_target"]
        target_left = selected["T_base_left_tcp_target"]
        report = {
            "result": "CENTER_MOVE_READY" if args.execute else "CENTER_MOVE_PLANNED_NOT_EXECUTED",
            "source_result": str(args.result.resolve()),
            "handoff_frame": "body_base",
            "current_object_center_m": current_object[:3, 3].tolist(),
            "nominal_handoff_object_center_m": center.tolist(),
            "selected_handoff_object_center_m": target_object[:3, 3].tolist(),
            "selected_candidate": selected["name"],
            "selected_center_offset_m": selected["center_offset_m"],
            "selected_orientation_delta": selected["orientation_delta"],
            "front_up_error_deg": selected["front_up_error_deg"],
            "tail_to_right_error_deg": selected["tail_to_right_error_deg"],
            "current_tail_axis_body": current_object[:3, 0].tolist(),
            "target_tail_axis_body": target_object[:3, 0].tolist(),
            "right_arm_side_body_axis": [0.0, 1.0, 0.0],
            "target_tail_alignment_with_right_arm": float(
                np.dot(target_object[:3, 0], [0.0, 1.0, 0.0])
            ),
            "object_orientation_preserved": False,
            "left_tcp_target": transform_debug(target_left),
            "plan": trajectory_summary(plan, 1),
            "collision_model_note": "MoveIt当前只保证已配置场景；夹持工件需已加入attached collision才能被检查",
        }
        if args.verbose_prediction_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(
                f"[规划成功] planner={args.cartesian_planner}，"
                f"目标={selected['name']}，轨迹点数={report['plan']['trajectory_points']}，"
                f"工件中心={np.asarray(target_object[:3, 3]).round(4).tolist()}，"
                f"尾部方向={np.asarray(target_object[:3, 0]).round(4).tolist()}。",
                flush=True,
            )
        if not args.execute:
            print("[完成] 只规划，机器人未运动。确认轨迹后加 --execute。")
            return 0

        print(
            "[方向核对] 当前工件尾部(+X)在body_base中的方向="
            f"{np.asarray(current_object[:3, 0]).round(4).tolist()}；"
            "规划目标方向="
            f"{np.asarray(target_object[:3, 0]).round(4).tolist()}；"
            "现场确认的右臂方向为body_base +Y=[0,1,0]。",
            flush=True,
        )
        prompt = (
            "确认目标尾部明确朝向右臂、现场安全且急停可用，"
            "输入 y 执行左臂调整工件："
        )
        if args.auto_confirm:
            print(f"{prompt}[自动确认]")
            answer = "y"
        else:
            answer = input(prompt).strip().lower()
        if answer != "y":
            print("已取消；机器人未运动。")
            return 0
        planner.execute_planned_result(plan)
        arrived_mm, arrived_deg = wait_for_left_tcp_target(
            planner, left_frame, target_left
        )
        final_left = read_left_tcp(planner, left_frame)
        final_object = final_left @ relation
        _, orientation_error_deg = transform_error(target_object, final_object)
        selected_center = target_object[:3, 3]
        center_error_mm = float(
            np.linalg.norm(final_object[:3, 3] - selected_center) * 1000.0
        )
        tolerance_mm = float(args.object_center_tolerance_mm)
        if center_error_mm > tolerance_mm:
            raise RuntimeError(
                f"执行后推算的工件中心误差{center_error_mm:.3f} mm，超过{tolerance_mm:.3f} mm"
            )
        orientation_tolerance_deg = float(
            config.get("object_orientation_tolerance_deg", 2.0)
        )
        if orientation_error_deg > orientation_tolerance_deg:
            raise RuntimeError(
                f"执行后推算的工件姿态误差{orientation_error_deg:.3f}°，"
                f"超过{orientation_tolerance_deg:.3f}°"
            )
        final_front_error, final_tail_error = handoff_axis_errors(final_object)
        print(
            f"[完成] 工件中心已到body_base {final_object[:3, 3].tolist()}，"
            f"位置误差={center_error_mm:.3f} mm，姿态误差={orientation_error_deg:.3f}°，"
            f"正面水平误差={final_front_error:.3f}°，尾部朝右臂误差={final_tail_error:.3f}°；"
            f"左TCP到位误差={arrived_mm:.3f}mm/{arrived_deg:.3f}°；"
            "左手仍夹持，右手未运动。"
        )
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
