#!/usr/bin/env python3
"""Use the observation ICP relation to let the right gripper take the tail."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(__file__).resolve().parents[2] / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable) != PYTHON:
    os.execv(
        str(PYTHON),
        [
            str(PYTHON),
            "-m",
            "tests.reversed_head_observation.execute_tail_handoff",
            *sys.argv[1:],
        ],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tests.reversed_head_observation.check_plan import (
    START_JOINTS_DEG,
    validate_joint_vector,
)
from tests.reversed_head_observation.execute_handoff import (
    move_gripper,
    read_gripper_state,
    read_tcp,
    require_pose_close,
    validate_joint_return_plan,
)
from tests.reversed_head_observation.handoff_attachment import (
    request_attachment_owner,
)
from tests.reversed_head_observation.handoff_geometry import (
    compute_right_tail_grasp_targets,
    planner_tf_target_source,
    transform_debug,
    transform_to_object_pose,
    validate_transform,
)
from tests.reversed_head_observation.move_object_center import handoff_axis_errors
from tests.reversed_head_observation.preflight_handoff import (
    build_runtime,
    load_icp_object_transform,
    trajectory_summary,
    validate_icp_for_motion,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "handoff_config.json"
DEFAULT_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"

# Measured from the deployed right-hand URDF. right_hand_transfer_end is the
# palm frame; the nominal pinch centre is 108.9 mm along its local +Z.
DEFAULT_RIGHT_PINCH_POINT_TCP_M = [0.0, 0.0, 0.1089]

# STL bounds are X=[-48.936, 49.060] mm, Y=[-26.633, 26.613] mm,
# Z=[0, 21.300] mm. This point is 15 mm inboard from the +X tail end and at
# the mid-thickness plane, so the fingers clamp across local Y.
DEFAULT_TAIL_GRASP_POINT_OBJECT_M = [0.03406, -0.00001, 0.01065]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", "--icp-result", dest="result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--execute-url", default=DEFAULT_EXECUTE_URL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--right-planner-frame",
        default="right_hand_transfer_base",
        help="右臂规划器IK帧；当前部署使用 right_hand_transfer_base",
    )
    parser.add_argument(
        "--right-palm-frame",
        default="right_hand_transfer_end",
        help="右夹爪掌部几何帧",
    )
    parser.add_argument(
        "--tail-grasp-point-object-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_TAIL_GRASP_POINT_OBJECT_M,
    )
    parser.add_argument(
        "--right-pinch-point-tcp-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_RIGHT_PINCH_POINT_TCP_M,
    )
    parser.add_argument("--pregrasp-distance-m", type=float, default=0.08)
    parser.add_argument(
        "--expected-center-m", type=float, nargs=3, default=[0.42, 0.20, 0.12]
    )
    parser.add_argument("--center-tolerance-mm", type=float, default=25.0)
    parser.add_argument("--axis-tolerance-deg", type=float, default=10.0)
    parser.add_argument(
        "--diagnostic-use-nominal-handoff-pose",
        action="store_true",
        help="仅用于只规划：假设工件已位于中心、正面朝上",
    )
    parser.add_argument(
        "--diagnostic-tail-body-y-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="诊断假设中工件尾部(+X)朝body_base哪一侧；1=+Y，-1=-Y",
    )
    parser.add_argument("--right-close-position", type=float, default=0.0)
    parser.add_argument("--right-open-position", type=float, default=100.0)
    parser.add_argument("--left-open-position", type=float, default=100.0)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    parser.add_argument("--arrival-translation-tolerance-mm", type=float, default=4.0)
    parser.add_argument("--arrival-rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--attachment-state",
        type=Path,
        default=HERE / "handoff_attachment_state.json",
    )
    parser.add_argument(
        "--switch-attachment",
        action="store_true",
        help="仅当附着碰撞发布器在线且包含右手几何时启用",
    )
    parser.add_argument(
        "--left-return-joints-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=START_JOINTS_DEG,
    )
    parser.add_argument("--left-return-tolerance-deg", type=float, default=0.5)
    parser.add_argument(
        "--verbose-prediction-json",
        action="store_true",
        help="在控制台打印坐标系、逐候选位姿和规划报告JSON；默认不打印",
    )
    return parser.parse_args()


def confirmed(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def validate_args(args: argparse.Namespace) -> None:
    if args.execute and args.diagnostic_use_nominal_handoff_pose:
        raise ValueError("diagnostic nominal pose cannot be used with --execute")
    if not (
        0.0 <= args.right_close_position <= 100.0
        and 0.0 <= args.right_open_position <= 100.0
        and 0.0 <= args.left_open_position <= 100.0
        and 20.0 <= args.gripper_force <= 100.0
    ):
        raise ValueError("gripper positions must be 0..100 and force must be 20..100")
    if args.right_open_position - args.right_close_position < 10.0:
        raise ValueError("right open and close positions must differ by at least 10%")
    if args.center_tolerance_mm <= 0.0 or args.axis_tolerance_deg <= 0.0:
        raise ValueError("handoff pose tolerances must be positive")
    validate_joint_vector(args.left_return_joints_deg, "left return joints")


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    T_base_object_observed, result = load_icp_object_transform(args.result)
    T_base_left_tcp_captured = validate_icp_for_motion(result, config)
    T_left_tcp_object = validate_transform(
        np.linalg.inv(T_base_left_tcp_captured) @ T_base_object_observed,
        "T_left_tcp_object_from_observation",
    )

    runtime_config = dict(config)
    runtime_config["right_tool_frame"] = args.right_planner_frame
    planner = build_runtime(
        runtime_config, args.planner_host, execute_url=args.execute_url
    )
    try:
        left_frame = planner.config.left_tool_frame
        right_planner_frame = planner.config.right_tool_frame
        right_palm_frame = args.right_palm_frame
        planner.open(required_frames={
            "body_base", left_frame, right_planner_frame, right_palm_frame
        })

        fixed_response = planner.lookup_transform(
            right_palm_frame, right_planner_frame
        )
        if fixed_response is None:
            raise RuntimeError(
                f"Planner returned no TF for {right_palm_frame} -> "
                f"{right_planner_frame}"
            )
        T_right_planner_right_palm = planner_tf_target_source(
            fixed_response, right_palm_frame, right_planner_frame
        )
        T_right_palm_right_planner = np.linalg.inv(
            T_right_planner_right_palm
        )
        current_right_planner = read_tcp(planner, right_planner_frame)
        if args.verbose_prediction_json:
            print(json.dumps({
                "right_frame_bridge": {
                    "planner_ik_frame": right_planner_frame,
                    "palm_geometry_frame": right_palm_frame,
                    "T_right_planner_right_palm": transform_debug(
                        T_right_planner_right_palm
                    ),
                    "current_right_planner_pose": transform_debug(
                        current_right_planner
                    ),
                }
            }, indent=2, ensure_ascii=False))

        # The left arm has moved since ICP, but the grasp is rigid. Reconstruct
        # the current object pose from live left TCP instead of reusing stale XYZ.
        T_base_left_tcp_live = read_tcp(planner, left_frame)
        T_base_object_live = validate_transform(
            T_base_left_tcp_live @ T_left_tcp_object,
            "T_base_object_reconstructed_from_live_left_tcp",
        )
        object_pose_source = "live_left_tcp @ observed_T_left_tcp_object"
        if args.diagnostic_use_nominal_handoff_pose:
            tail_y_sign = float(args.diagnostic_tail_body_y_sign)
            T_base_object_live = np.eye(4)
            T_base_object_live[:3, :3] = np.asarray([
                [0.0, -tail_y_sign, 0.0],
                [tail_y_sign, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ])
            T_base_object_live[:3, 3] = np.asarray(
                args.expected_center_m, dtype=float
            )
            object_pose_source = "diagnostic_nominal_handoff_pose_not_live"
        center_error_mm = float(
            np.linalg.norm(
                T_base_object_live[:3, 3] - np.asarray(args.expected_center_m, dtype=float)
            )
            * 1000.0
        )
        front_error_deg, tail_error_deg = handoff_axis_errors(T_base_object_live)
        if args.diagnostic_use_nominal_handoff_pose:
            tail_cos = float(np.clip(
                np.dot(
                    T_base_object_live[:3, 0],
                    [0.0, float(args.diagnostic_tail_body_y_sign), 0.0],
                ),
                -1.0,
                1.0,
            ))
            tail_error_deg = float(np.degrees(np.arccos(tail_cos)))
        if center_error_mm > args.center_tolerance_mm:
            raise RuntimeError(
                f"重建工件中心与交接中心相差 {center_error_mm:.3f} mm，"
                f"超过 {args.center_tolerance_mm:.3f} mm"
            )
        if (
            front_error_deg > args.axis_tolerance_deg
            or tail_error_deg > args.axis_tolerance_deg
        ):
            raise RuntimeError(
                "工件没有满足正面朝上、尾部朝右："
                f"front={front_error_deg:.3f}°, tail={tail_error_deg:.3f}°"
            )

        selected = None
        staged_fallback = None
        failures: list[dict[str, str]] = []
        candidate_records: list[dict[str, object]] = []
        candidates = (
            ("from_tail", 0.0),
            ("from_tail", 180.0),
            ("top_down", 0.0),
            ("top_down", 180.0),
            ("tail_above_20", 0.0),
            ("tail_above_20", 180.0),
            ("tail_above_30", 0.0),
            ("tail_above_30", 180.0),
            ("tail_above_45", 0.0),
            ("tail_above_45", 180.0),
            ("tail_above_60", 0.0),
            ("tail_above_60", 180.0),
            ("tail_below_20", 0.0),
            ("tail_below_20", 180.0),
            ("tail_below_30", 0.0),
            ("tail_below_30", 180.0),
            ("tail_below_45", 0.0),
            ("tail_below_45", 180.0),
            ("tail_below_60", 0.0),
            ("tail_below_60", 180.0),
            ("slide_from_body_30", 0.0),
            ("slide_from_body_30", 180.0),
            ("slide_from_body_45", 0.0),
            ("slide_from_body_45", 180.0),
            ("slide_from_body_60", 0.0),
            ("slide_from_body_60", 180.0),
        )
        for candidate_index, (approach_mode, symmetry_deg) in enumerate(
            candidates, start=1
        ):
            targets = compute_right_tail_grasp_targets(
                T_base_object_live,
                list(args.tail_grasp_point_object_m),
                list(args.right_pinch_point_tcp_m),
                args.pregrasp_distance_m,
                jaw_symmetry_deg=symmetry_deg,
                approach_mode=approach_mode,
            )
            # Geometry targets are expressed for the palm frame. The deployed
            # right-arm IK endpoint is transfer_base, so convert before planning:
            # T_world_palm = T_world_planner @ T_planner_palm.
            T_base_right_planner_pregrasp = validate_transform(
                targets["T_base_right_tcp_pregrasp"]
                @ T_right_palm_right_planner,
                "T_base_right_planner_pregrasp",
            )
            T_base_right_planner_grasp = validate_transform(
                targets["T_base_right_tcp_grasp"]
                @ T_right_palm_right_planner,
                "T_base_right_planner_grasp",
            )
            candidate_record: dict[str, object] = {
                "jaw_symmetry_deg": symmetry_deg,
                "approach_mode": approach_mode,
                "pregrasp": transform_debug(
                    targets["T_base_right_tcp_pregrasp"]
                ),
                "grasp": transform_debug(targets["T_base_right_tcp_grasp"]),
                "grasp_point_base_m": targets["grasp_point_base_m"].tolist(),
                "planner_pregrasp": transform_debug(
                    T_base_right_planner_pregrasp
                ),
                "planner_grasp": transform_debug(T_base_right_planner_grasp),
            }
            candidate_records.append(candidate_record)
            if args.verbose_prediction_json:
                print(json.dumps(
                    {"right_tail_candidate": candidate_record},
                    indent=2,
                    ensure_ascii=False,
                ))
            try:
                plan = planner.plan_cartesian_sequence(
                    "right",
                    [
                        (
                            "rrtstar",
                            transform_to_object_pose(
                                T_base_right_planner_pregrasp
                            ),
                        ),
                        (
                            "lin",
                            transform_to_object_pose(T_base_right_planner_grasp),
                        ),
                    ],
                )
            except Exception as exc:
                pregrasp_diagnostic = "not_tested"
                grasp_diagnostic = "not_tested"
                pregrasp_only = None
                grasp_only = None
                try:
                    pregrasp_only = planner.plan_cartesian_pose(
                        "right",
                        transform_to_object_pose(
                            T_base_right_planner_pregrasp
                        ),
                        planner="rrtstar",
                    )
                    pregrasp_diagnostic = (
                        "reachable"
                        if pregrasp_only is not None
                        else "planner_returned_none"
                    )
                except Exception as pregrasp_exc:
                    pregrasp_diagnostic = (
                        f"{type(pregrasp_exc).__name__}: {pregrasp_exc}"
                    )
                try:
                    grasp_only = planner.plan_cartesian_pose(
                        "right",
                        transform_to_object_pose(T_base_right_planner_grasp),
                        planner="rrtstar",
                    )
                    grasp_diagnostic = (
                        "reachable"
                        if grasp_only is not None
                        else "planner_returned_none"
                    )
                except Exception as grasp_exc:
                    grasp_diagnostic = (
                        f"{type(grasp_exc).__name__}: {grasp_exc}"
                    )
                if (
                    pregrasp_only is not None
                    and grasp_only is not None
                    and staged_fallback is None
                ):
                    staged_fallback = (
                        approach_mode,
                        symmetry_deg,
                        targets,
                        T_base_right_planner_pregrasp,
                        T_base_right_planner_grasp,
                        pregrasp_only,
                    )
                failures.append({
                    "approach_mode": approach_mode,
                    "jaw_symmetry_deg": str(symmetry_deg),
                    "error": f"{type(exc).__name__}: {exc}",
                    "pregrasp_only": pregrasp_diagnostic,
                    "grasp_only": grasp_diagnostic,
                })
                candidate_record["planning_failure"] = failures[-1]
                print(
                    f"[右手候选 {candidate_index}/{len(candidates)} 失败] "
                    f"approach={approach_mode}, symmetry={symmetry_deg:.0f}°；"
                    "组合/预抓取/抓取单点均未通过规划"
                )
                continue
            if plan is not None:
                selected = (
                    approach_mode,
                    symmetry_deg,
                    targets,
                    T_base_right_planner_pregrasp,
                    T_base_right_planner_grasp,
                    plan,
                )
                break
        if selected is None and staged_fallback is not None:
            (
                approach_mode,
                symmetry_deg,
                targets,
                T_base_right_planner_pregrasp,
                T_base_right_planner_grasp,
                pregrasp_plan,
            ) = staged_fallback
            selected = (
                approach_mode,
                symmetry_deg,
                targets,
                T_base_right_planner_pregrasp,
                T_base_right_planner_grasp,
                None,
                pregrasp_plan,
            )
            print(
                f"[分阶段候选] approach={approach_mode}, "
                f"symmetry={symmetry_deg:.0f}°；预抓取可达，"
                "LIN抓取将在真实到位后重新规划。"
            )
        elif selected is not None:
            selected = (*selected, None)
        if selected is None:
            diagnostics_path = args.result.resolve().parent / (
                "right_handoff_planning_failures.json"
            )
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "source_result": str(args.result.resolve()),
                        "right_planner_frame": right_planner_frame,
                        "right_palm_frame": right_palm_frame,
                        "candidate_count": len(candidates),
                        "candidates": candidate_records,
                    },
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"右手全部 {len(candidates)} 个尾部抓取候选均规划失败；"
                "组合路径、预抓取单点和抓取单点均失败，"
                f"详细诊断已保存到 {diagnostics_path}"
            )

        (
            approach_mode,
            symmetry_deg,
            targets,
            T_base_right_planner_pregrasp,
            T_base_right_planner_grasp,
            right_plan,
            staged_pregrasp_plan,
        ) = selected
        staged_execution = right_plan is None
        body_side_slide = approach_mode.startswith("slide_from_body_")
        pregrasp_pinch_base = (
            targets["T_base_right_tcp_pregrasp"][:3, 3]
            + targets["T_base_right_tcp_pregrasp"][:3, :3]
            @ np.asarray(args.right_pinch_point_tcp_m, dtype=float)
        )
        pregrasp_pinch_object = (
            np.linalg.inv(T_base_object_live)
            @ np.r_[pregrasp_pinch_base, 1.0]
        )[:3]
        report = {
            "mode": "EXECUTE" if args.execute else "PLAN_ONLY",
            "source_result": str(args.result.resolve()),
            "object_pose_source": object_pose_source,
            "reconstructed_object": transform_debug(T_base_object_live),
            "center_error_mm": center_error_mm,
            "front_up_error_deg": front_error_deg,
            "tail_to_right_error_deg": tail_error_deg,
            "tail_grasp_point_object_m": list(args.tail_grasp_point_object_m),
            "right_pinch_point_tcp_m": list(args.right_pinch_point_tcp_m),
            "jaw_symmetry_deg": symmetry_deg,
            "approach_mode": approach_mode,
            "approach_path_class": (
                "OPEN_JAW_BODY_SIDE_SLIDE"
                if body_side_slide
                else "OUTSIDE_TAIL_OR_TOP_APPROACH"
            ),
            "pregrasp_pinch_center_object_m": pregrasp_pinch_object.tolist(),
            "right_pregrasp": transform_debug(
                targets["T_base_right_tcp_pregrasp"]
            ),
            "right_grasp": transform_debug(targets["T_base_right_tcp_grasp"]),
            "right_planner_pregrasp": transform_debug(
                T_base_right_planner_pregrasp
            ),
            "right_planner_grasp": transform_debug(
                T_base_right_planner_grasp
            ),
            "grasp_point_base_m": targets["grasp_point_base_m"].tolist(),
            "planning_strategy": (
                "STAGED_PREGRASP_THEN_REPLAN_LIN"
                if staged_execution
                else "COMBINED_RRTSTAR_PREGRASP_AND_LIN_GRASP"
            ),
            "right_plan": (
                None if right_plan is None else trajectory_summary(right_plan, 2)
            ),
            "right_pregrasp_plan": (
                trajectory_summary(staged_pregrasp_plan, 2)
                if staged_pregrasp_plan is not None
                else None
            ),
            "collision_note": (
                "右手路径按当前MoveIt双臂模型规划；若未启用附着碰撞发布器，"
                "被夹持工件本体不在碰撞模型中。OPEN_JAW_BODY_SIDE_SLIDE "
                "要求右夹爪全开并由操作者确认两指可跨过工件宽度及左夹爪。"
            ),
        }
        if args.verbose_prediction_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        print(
            f"[右手候选通过] approach={approach_mode}, "
            f"symmetry={symmetry_deg:.0f}°, "
            f"strategy={report['planning_strategy']}"
        )
        if not args.execute:
            if staged_execution:
                print(
                    "[完成] 右手预抓取位已规划；最后LIN将在实际到达预抓取位后"
                    "从实时状态重新规划。机械臂和夹爪均未动作。"
                )
            else:
                print("[完成] 只规划右手抓尾部；机械臂和夹爪均未动作。")
            return 0

        right_state = read_gripper_state(args.gripper_host, 2, "right")
        right_opening = float(right_state.get("position", -1.0))
        if right_opening < args.right_open_position - 5.0:
            if not confirmed(
                f"右夹爪当前开度={right_opening:.1f}%，输入 y 张开到"
                f"{args.right_open_position:.1f}%："
            ):
                print("已取消；右臂未运动。")
                return 0
            print(move_gripper(
                args.gripper_host,
                2,
                args.right_open_position,
                args.gripper_force,
                "right",
            ))

        if staged_execution:
            if staged_pregrasp_plan is None:
                raise RuntimeError("分阶段执行缺少右手预抓取轨迹")
            staged_prompt = (
                "确认右夹爪已全开、两指位于工件宽度两侧，且从工件前段"
                "斜向滑到尾部的扫掠路径不会碰工件或左夹爪，输入 y 执行"
                "右手到预抓取位："
                if body_side_slide
                else
                "确认预抓取目标位于工件尾部外侧且双臂间隙安全，"
                "输入 y 执行右手到预抓取位："
            )
            if not confirmed(staged_prompt):
                print("已取消；右臂未运动。")
                return 0
            planner.execute_planned_result(staged_pregrasp_plan)
            pregrasp_live = read_tcp(planner, right_planner_frame)
            require_pose_close(
                pregrasp_live,
                T_base_right_planner_pregrasp,
                args.arrival_translation_tolerance_mm,
                args.arrival_rotation_tolerance_deg,
                "right tail pregrasp pose",
            )
            try:
                right_plan = planner.plan_cartesian_pose(
                    "right",
                    transform_to_object_pose(T_base_right_planner_grasp),
                    planner="lin",
                )
            except Exception as exc:
                raise RuntimeError(
                    "右手已安全到达预抓取位，但从实时状态规划LIN抓取失败；"
                    "右手保持预抓取位，左手仍持有工件"
                ) from exc
            if right_plan is None:
                raise RuntimeError(
                    "右手已到预抓取位，但规划器未返回LIN抓取轨迹；"
                    "右手保持预抓取位，左手仍持有工件"
                )
            print(json.dumps({
                "replanned_right_lin_grasp": trajectory_summary(right_plan, 2)
            }, indent=2, ensure_ascii=False))
            if not confirmed("确认尾部夹持点和直线接近间隙安全，输入 y 执行LIN抓取："):
                print("已停止；右手保持预抓取位，左手仍持有工件。")
                return 0
            planner.execute_planned_result(right_plan)
        else:
            if not confirmed("确认右手从工件尾部外侧接近且双臂间隙安全，输入 y 执行："):
                print("已取消；右臂未运动。")
                return 0
            planner.execute_planned_result(right_plan)
        T_base_right_tcp_live = read_tcp(planner, right_planner_frame)
        arrival_error = require_pose_close(
            T_base_right_tcp_live,
            T_base_right_planner_grasp,
            args.arrival_translation_tolerance_mm,
            args.arrival_rotation_tolerance_deg,
            "right tail grasp pose",
        )
        print(f"右手已到尾部抓取位：{arrival_error}")

        if not confirmed("确认右夹爪已对准工件尾部，输入 y 闭合右夹爪："):
            print("已停止；右手停在尾部抓取位，左手仍持有工件。")
            return 0
        print(move_gripper(
            args.gripper_host,
            2,
            args.right_close_position,
            args.gripper_force,
            "right",
        ))
        if not confirmed("人工确认右手已夹稳尾部，输入 y 释放左夹爪："):
            print("已停止；双手共同夹持工件。")
            return 0

        if args.switch_attachment:
            attachment_ack = request_attachment_owner(
                args.attachment_state, "right", timeout_s=5.0
            )
            print(f"MoveIt附着体已切换到右手：{attachment_ack}")
        else:
            print("[警告] 未切换附着碰撞体；后续规划不包含右手持有的工件几何。")
        print(move_gripper(
            args.gripper_host,
            1,
            args.left_open_position,
            args.gripper_force,
            "left",
        ))

        retreat_plan = planner.plan_cartesian_pose(
            "right",
            transform_to_object_pose(T_base_right_planner_pregrasp),
            planner="lin",
        )
        if retreat_plan is None:
            raise RuntimeError("Planner returned no right retreat plan")
        print(json.dumps({
            "right_retreat_plan": trajectory_summary(retreat_plan, 2)
        }, ensure_ascii=False))
        if not confirmed("左手已释放，输入 y 让右手沿尾部方向直线退回："):
            print("已停止；右手持件停在交接位，左手保持张开。")
            return 0
        planner.execute_planned_result(retreat_plan)

        left_return_plan = planner.plan_joint_trajectory(
            "left", list(args.left_return_joints_deg), planner="rrtstar"
        )
        if left_return_plan is None:
            raise RuntimeError("Planner returned no left return-to-middle plan")
        left_return_summary = validate_joint_return_plan(
            left_return_plan,
            list(args.left_return_joints_deg),
            args.left_return_tolerance_deg,
        )
        print(json.dumps(
            {"left_return_to_middle": left_return_summary},
            indent=2,
            ensure_ascii=False,
        ))
        if not confirmed("右手已持件退开，输入 y 让左臂回到中间初始位："):
            print("已停止；右手持件，左手保持张开。")
            return 0
        planner.execute_planned_result(left_return_plan)
        print("[完成] 右手已抓住工件尾部并退开，左手已释放并回到中间初始位。")
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
