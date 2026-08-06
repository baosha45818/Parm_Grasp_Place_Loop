#!/usr/bin/env python3
"""Execute the taught-neighbour, ICP-corrected workpiece-2 tail handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.reversed_head_observation.check_plan import (  # noqa: E402
    START_JOINTS_DEG,
    validate_joint_vector,
)
from tests.reversed_head_observation.execute_handoff import (  # noqa: E402
    move_gripper,
    read_gripper_state,
    read_tcp,
    validate_joint_return_plan,
)
from tests.reversed_head_observation.execute_taught_tail_handoff import (  # noqa: E402
    validate_arm_joint_plan,
    validate_right_cartesian_plan,
    wait_for_joint_target,
    wait_for_pose_close,
)
from tests.reversed_head_observation.handoff_attachment import (  # noqa: E402
    request_attachment_owner,
)
from tests.reversed_head_observation.handoff_geometry import (  # noqa: E402
    transform_error,
    transform_to_object_pose,
    validate_transform,
)
from tests.reversed_head_observation.move_object_center import (  # noqa: E402
    handoff_axis_errors,
)
from tests.reversed_head_observation.preflight_handoff import (  # noqa: E402
    build_runtime,
    load_icp_object_transform,
    validate_icp_for_motion,
)
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "handoff_config.json"
DEFAULT_RIGHT_PREGRASP_JOINTS_DEG = [
    173.9519, 50.8063, -79.5218, 85.0397, -25.3112, 4.5552, 79.6344,
]
# Previous workpiece-2 taught grasp (trace only):
# [173.1142, 56.1951, -73.7979, 88.5512, -20.1174, 1.6864, 72.4699]
# [162.5673, 58.7975, -71.0898, 78.7033, -21.5511, -1.0972, 79.6811]
DEFAULT_RIGHT_GRASP_JOINTS_DEG = [
    162.5467, 59.2095, -70.8701, 76.9661, -21.2146, 0.4807, 80.0175,
]
DEFAULT_RIGHT_PLACE_READY_JOINTS_DEG = [
    102.2483, 21.8477, -87.2013, 83.4439, -147.2882, -32.2943, -66.0196,
]
DEFAULT_POST_RELEASE_CLEARANCE_MINUS_X_M = 0.03
DEFAULT_POST_RELEASE_CLEARANCE_PLUS_Z_M = 0.03
# Workpiece 2 uses one exact RRTStar handoff orientation.  Two degrees only
# cover arrival/TF error; yaw-redundant handoff poses must be rejected.
DEFAULT_HANDOFF_AXIS_TOLERANCE_DEG = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default="192.168.88.1")
    parser.add_argument("--gripper-host", default="127.0.0.1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--require-confirmation", action="store_true")
    parser.add_argument(
        "--resume-after-left-release",
        action="store_true",
        help="右手已夹稳且左夹爪已打开：跳过抓取，先右手离开交接区再恢复左手",
    )
    parser.add_argument(
        "--expected-center-m", type=float, nargs=3, default=[0.42, 0.10, 0.25]
    )
    parser.add_argument("--center-tolerance-mm", type=float, default=25.0)
    parser.add_argument(
        "--axis-tolerance-deg", type=float,
        default=DEFAULT_HANDOFF_AXIS_TOLERANCE_DEG,
        help="工件2交接轴向到位/TF误差上限；不允许yaw冗余，默认2°",
    )
    parser.add_argument(
        "--right-pregrasp-joints-deg", type=float, nargs=7,
        default=DEFAULT_RIGHT_PREGRASP_JOINTS_DEG,
        help="右手工件2尾部预抓取示教关节角",
    )
    parser.add_argument(
        "--right-grasp-joints-deg", "--right-near-joints-deg",
        dest="right_grasp_joints_deg", type=float, nargs=7,
        default=DEFAULT_RIGHT_GRASP_JOINTS_DEG,
        help="右手工件2尾部最终抓取示教关节角",
    )
    parser.add_argument(
        "--right-place-ready-joints-deg", type=float, nargs=7,
        default=DEFAULT_RIGHT_PLACE_READY_JOINTS_DEG,
    )
    parser.add_argument(
        "--tail-advance-step-mm", type=float, default=5.0,
        help="到示教抓取位后每次沿实时工件+X继续推进的距离，默认5mm",
    )
    parser.add_argument(
        "--tail-advance-max-mm", type=float, default=100.0,
        help="人工分步推进的最大累计距离，默认100mm",
    )
    parser.add_argument(
        "--post-release-clearance-minus-x-m", type=float,
        default=DEFAULT_POST_RELEASE_CLEARANCE_MINUS_X_M,
        help="左手释放后右手沿实时工件-X撤离距离，默认0.03m",
    )
    parser.add_argument(
        "--post-release-clearance-plus-z-m", type=float,
        default=DEFAULT_POST_RELEASE_CLEARANCE_PLUS_Z_M,
        help="左手释放后右手沿实时工件+Z抬升距离，默认0.03m",
    )
    parser.add_argument("--right-open-position", type=float, default=100.0)
    parser.add_argument("--right-close-position", type=float, default=0.0)
    parser.add_argument("--left-open-position", type=float, default=100.0)
    parser.add_argument("--gripper-force", type=float, default=60.0)
    parser.add_argument(
        "--left-return-joints-deg", type=float, nargs=7,
        default=START_JOINTS_DEG,
    )
    parser.add_argument("--joint-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--arrival-translation-tolerance-mm", type=float, default=4.0)
    parser.add_argument("--arrival-rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--attachment-state", type=Path,
        default=HERE / "handoff_attachment_state.json",
    )
    parser.add_argument(
        "--held-object-state", type=Path,
        default=HERE / "right_held_object_state.json",
        help="保存释放瞬间右TCP/工件关系及撤离进度，供安全恢复使用",
    )
    parser.add_argument("--switch-attachment", action="store_true")
    parser.add_argument("--verbose-prediction-json", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for values, label in (
        (args.right_pregrasp_joints_deg, "right pregrasp joints"),
        (args.right_grasp_joints_deg, "right grasp joints"),
        (args.right_place_ready_joints_deg, "right place-ready joints"),
        (args.left_return_joints_deg, "left return joints"),
    ):
        validate_joint_vector(values, label)
    if args.tail_advance_step_mm <= 0.0 or args.tail_advance_max_mm <= 0.0:
        raise ValueError("tail advance distances must be positive")
    if args.tail_advance_step_mm > args.tail_advance_max_mm:
        raise ValueError("tail advance step cannot exceed its maximum")
    if (
        args.post_release_clearance_minus_x_m <= 0.0
        or args.post_release_clearance_plus_z_m <= 0.0
    ):
        raise ValueError("post-release clearance distances must be positive")
    if args.center_tolerance_mm <= 0.0 or args.axis_tolerance_deg <= 0.0:
        raise ValueError("handoff tolerances must be positive")


def confirmed(args: argparse.Namespace, prompt: str) -> bool:
    if not args.require_confirmation:
        print(f"{prompt}[自动确认]", flush=True)
        return True
    return input(prompt).strip().lower() == "y"


def reconstruct_object(planner, left_frame: str, T_left_tcp_object: np.ndarray) -> np.ndarray:
    return validate_transform(
        read_tcp(planner, left_frame) @ T_left_tcp_object,
        "T_base_object_reconstructed_from_live_left_tcp",
    )


def validate_live_object(args: argparse.Namespace, T_base_object: np.ndarray) -> None:
    center_error_mm = float(np.linalg.norm(
        T_base_object[:3, 3] - np.asarray(args.expected_center_m, dtype=float)
    ) * 1000.0)
    front_error_deg, tail_error_deg = handoff_axis_errors(T_base_object)
    if center_error_mm > args.center_tolerance_mm:
        raise RuntimeError(
            f"工件中心偏离交接区域 {center_error_mm:.3f} mm，"
            f"超过 {args.center_tolerance_mm:.3f} mm"
        )
    if max(front_error_deg, tail_error_deg) > args.axis_tolerance_deg:
        raise RuntimeError(
            "工件交接姿态不正确："
            f"front={front_error_deg:.3f}°, tail={tail_error_deg:.3f}°"
        )


def advance_grasp_along_object_x(
    T_base_grasp: np.ndarray,
    T_base_object: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("tail advance distance must be finite and non-negative")
    target = validate_transform(T_base_grasp, "T_base_grasp_before_advance").copy()
    object_pose = validate_transform(T_base_object, "T_base_object_for_advance")
    target[:3, 3] += object_pose[:3, :3] @ np.asarray([distance_m, 0.0, 0.0])
    return validate_transform(target, "T_base_grasp_advanced_along_object_x")


def build_post_release_clearance_target(
    T_base_right_tcp: np.ndarray,
    T_right_tcp_object: np.ndarray,
    minus_x_m: float,
    plus_z_m: float,
) -> np.ndarray:
    """Keep TCP orientation and translate it in the held workpiece frame."""
    tcp = validate_transform(T_base_right_tcp, "T_base_right_tcp_at_release")
    relation = validate_transform(T_right_tcp_object, "T_right_tcp_object_held")
    if minus_x_m <= 0.0 or plus_z_m <= 0.0:
        raise ValueError("post-release clearance distances must be positive")
    object_pose = validate_transform(
        tcp @ relation, "T_base_object_held_at_release"
    )
    target = tcp.copy()
    target[:3, 3] += object_pose[:3, :3] @ np.asarray(
        [-float(minus_x_m), 0.0, float(plus_z_m)]
    )
    return validate_transform(target, "T_base_right_tcp_post_release_clearance")


def save_held_object_state(
    path: Path,
    T_right_tcp_object: np.ndarray,
    T_base_right_tcp_at_release: np.ndarray,
    T_base_right_tcp_clearance: np.ndarray,
    stage: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "T_right_tcp_object": validate_transform(
            T_right_tcp_object, "T_right_tcp_object_to_save"
        ).tolist(),
        "T_base_right_tcp_at_release": validate_transform(
            T_base_right_tcp_at_release, "T_base_right_tcp_at_release_to_save"
        ).tolist(),
        "T_base_right_tcp_clearance": validate_transform(
            T_base_right_tcp_clearance, "T_base_right_tcp_clearance_to_save"
        ).tolist(),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_held_object_state(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "stage": str(payload.get("stage", "unknown")),
        "T_right_tcp_object": validate_transform(
            payload["T_right_tcp_object"], "saved.T_right_tcp_object"
        ),
        "T_base_right_tcp_at_release": validate_transform(
            payload["T_base_right_tcp_at_release"],
            "saved.T_base_right_tcp_at_release",
        ),
        "T_base_right_tcp_clearance": validate_transform(
            payload["T_base_right_tcp_clearance"],
            "saved.T_base_right_tcp_clearance",
        ),
    }


def pose_is_close(
    actual: np.ndarray,
    expected: np.ndarray,
    translation_tolerance_mm: float,
    rotation_tolerance_deg: float,
) -> bool:
    translation_m, rotation_deg = transform_error(actual, expected)
    return (
        translation_m * 1000.0 <= translation_tolerance_mm
        and rotation_deg <= rotation_tolerance_deg
    )


def finish_after_left_release(
    planner,
    args: argparse.Namespace,
    right_ready: list[float],
    left_return: list[float],
    right_frame: str,
    held_state: dict[str, object] | None,
) -> int:
    """Clear the left gripper, move right to ready, then plan left return."""
    current_right = read_tcp(planner, right_frame)
    right_ready_error_deg = current_arm_joint_error_deg(
        args.arm_host, 2, right_ready
    )
    right_at_ready = right_ready_error_deg <= args.joint_tolerance_deg

    if not right_at_ready:
        if held_state is None:
            raise RuntimeError(
                "右手尚未到放置准备位，但缺少释放瞬间的工件关系；"
                "拒绝猜测工件轴向执行3cm撤离"
            )
        clearance = validate_transform(
            held_state["T_base_right_tcp_clearance"],
            "held_state.T_base_right_tcp_clearance",
        )
        at_clearance = pose_is_close(
            current_right, clearance,
            args.arrival_translation_tolerance_mm,
            args.arrival_rotation_tolerance_deg,
        )
        if not at_clearance:
            at_release = pose_is_close(
                current_right,
                validate_transform(
                    held_state["T_base_right_tcp_at_release"],
                    "held_state.T_base_right_tcp_at_release",
                ),
                args.arrival_translation_tolerance_mm,
                args.arrival_rotation_tolerance_deg,
            )
            if not at_release:
                raise RuntimeError(
                    "右手既不在记录的释放位、3cm撤离位，也不在放置准备位；"
                    "拒绝重复或从未知位置执行工件系撤离"
                )
            clearance_plan = planner.plan_cartesian_pose(
                "right", transform_to_object_pose(clearance), planner="lin"
            )
            if clearance_plan is None:
                raise RuntimeError("Planner未返回右手释放后3cm安全撤离轨迹")
            validate_right_cartesian_plan(
                clearance_plan, args.right_grasp_joints_deg, clearance,
                args.arrival_translation_tolerance_mm,
                args.arrival_rotation_tolerance_deg,
            )
            if not args.execute:
                print(
                    "[完成] 左手释放后沿工件-X 30mm、+Z 30mm的右手LIN撤离规划通过；"
                    "到位后才会规划放置准备位和左手恢复。"
                )
                return 0
            if not confirmed(
                args,
                "确认左夹爪已打开，右手沿工件-X 3cm、+Z 3cm的LIN撤离安全，"
                "输入 y 执行：",
            ):
                print("[停止] 右手持件停在交接释放位，左手保持张开。")
                return 0
            planner.execute_planned_result(clearance_plan)
            wait_for_pose_close(
                planner, right_frame, clearance,
                args.arrival_translation_tolerance_mm,
                args.arrival_rotation_tolerance_deg,
                "right post-release object -X/+Z clearance", timeout_s=30.0,
            )
            save_held_object_state(
                args.held_object_state,
                held_state["T_right_tcp_object"],
                held_state["T_base_right_tcp_at_release"],
                clearance,
                "clearance_complete",
            )
            print("[右手已安全撤离] 已沿工件-X 30mm、+Z 30mm离开左夹爪。")

    if not right_at_ready:
        right_ready_plan = planner.plan_joint_trajectory(
            "right", right_ready, planner="rrtstar"
        )
        if right_ready_plan is None:
            raise RuntimeError("Planner returned no right place-ready plan")
        validate_arm_joint_plan(
            right_ready_plan, right_ready, arm_id=2,
            label="right_place_ready", tolerance_deg=args.joint_tolerance_deg,
        )
        if not args.execute:
            print(
                "[完成] 右手安全撤离位到放置准备位规划通过；"
                "左手回位必须等右手真实到位后再规划。"
            )
            return 0
        if not confirmed(
            args,
            "确认右手已完成3cm安全撤离，且到放置准备位RRTStar路径安全，输入 y 执行：",
        ):
            print("[停止] 右手持件停在安全撤离位，左手保持张开。")
            return 0
        planner.execute_planned_result(right_ready_plan)
        wait_for_joint_target(
            args.arm_host, 2, right_ready, args.joint_tolerance_deg,
            "right place-ready", timeout_s=90.0,
        )
        if held_state is not None:
            save_held_object_state(
                args.held_object_state,
                held_state["T_right_tcp_object"],
                held_state["T_base_right_tcp_at_release"],
                held_state["T_base_right_tcp_clearance"],
                "place_ready_complete",
            )
        print("[右手已离开交接区] 现在按新的实时双臂状态规划左手恢复。")
    else:
        print("[恢复模式] 右手已经在放置准备位，不重复执行3cm撤离和右手RRTStar。")

    left_return_plan = planner.plan_joint_trajectory(
        "left", left_return, planner="rrtstar"
    )
    if left_return_plan is None:
        raise RuntimeError(
            "右手已到放置准备位，但Planner未返回左手恢复轨迹；"
            "右手继续持件，左手保持张开"
        )
    validate_joint_return_plan(
        left_return_plan, left_return, args.joint_tolerance_deg
    )
    if not confirmed(args, "确认右手已离开且左手恢复路径安全，输入 y 执行："):
        print("[停止] 右手位于放置准备位，左手保持张开。")
        return 0
    planner.execute_planned_result(left_return_plan)
    wait_for_joint_target(
        args.arm_host, 1, left_return, args.joint_tolerance_deg,
        "left return", timeout_s=90.0,
    )
    print("[完成] 右手已到放置准备位，左手已恢复。")
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    right_ready = list(args.right_place_ready_joints_deg)
    left_return = list(args.left_return_joints_deg)
    if args.resume_after_left_release:
        if not args.execute:
            raise RuntimeError("--resume-after-left-release requires --execute")
        right_state = read_gripper_state(args.gripper_host, 2, "right")
        left_state = read_gripper_state(args.gripper_host, 1, "left")
        right_holding = (
            float(right_state.get("position", 100.0)) < 95.0
            and str(right_state.get("motion_state", "")).lower() == "gripping"
        )
        left_open = float(left_state.get("position", 0.0)) >= 95.0
        if not right_holding or not left_open:
            raise RuntimeError(
                "拒绝恢复：需要右夹爪保持gripping且左夹爪开度>=95%；"
                f"right={right_state}, left={left_state}"
            )
        planner = build_runtime(
            config,
            args.planner_host,
            execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
        )
        try:
            right_frame = str(config["right_tool_frame"])
            planner.open(required_frames={
                "body_base", str(config["left_tool_frame"]), right_frame
            })
            print(
                "[恢复模式] 已确认右手夹持、左夹爪打开；"
                "跳过ICP抓取，先让右手离开交接区。"
            )
            held_state = None
            if args.held_object_state.exists():
                held_state = load_held_object_state(args.held_object_state)
                print(
                    f"[恢复模式] 已加载右手持件关系，阶段={held_state['stage']}。"
                )
            else:
                right_ready_error_deg = current_arm_joint_error_deg(
                    args.arm_host, 2, right_ready
                )
                if right_ready_error_deg > args.joint_tolerance_deg:
                    # Compatibility for a run interrupted before this state file
                    # existed: the released left arm has not moved, so its live
                    # TCP still reconstructs the handoff object pose.
                    T_base_object_observed, result = load_icp_object_transform(
                        args.result
                    )
                    T_base_left_tcp_captured = validate_icp_for_motion(result, config)
                    T_left_tcp_object = validate_transform(
                        np.linalg.inv(T_base_left_tcp_captured)
                        @ T_base_object_observed,
                        "T_left_tcp_object_resume_fallback",
                    )
                    left_frame = str(config["left_tool_frame"])
                    object_at_release = reconstruct_object(
                        planner, left_frame, T_left_tcp_object
                    )
                    relation = validate_transform(
                        np.linalg.inv(current_right) @ object_at_release,
                        "T_right_tcp_object_resume_fallback",
                    )
                    clearance = build_post_release_clearance_target(
                        current_right, relation,
                        args.post_release_clearance_minus_x_m,
                        args.post_release_clearance_plus_z_m,
                    )
                    save_held_object_state(
                        args.held_object_state, relation, current_right,
                        clearance, "left_released_legacy_recovery",
                    )
                    held_state = load_held_object_state(args.held_object_state)
                    print(
                        "[恢复模式] 已用未移动的左TCP和本轮ICP结果重建右手持件关系。"
                    )
            return finish_after_left_release(
                planner, args, right_ready, left_return, right_frame, held_state
            )
        finally:
            planner.close()

    T_base_object_observed, result = load_icp_object_transform(args.result)
    T_base_left_tcp_captured = validate_icp_for_motion(result, config)
    T_left_tcp_object = validate_transform(
        np.linalg.inv(T_base_left_tcp_captured) @ T_base_object_observed,
        "T_left_tcp_object_from_observation",
    )
    right_pregrasp = list(args.right_pregrasp_joints_deg)
    right_grasp = list(args.right_grasp_joints_deg)

    planner = build_runtime(
        config,
        args.planner_host,
        execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
    )
    try:
        left_frame = str(config["left_tool_frame"])
        right_frame = str(config["right_tool_frame"])
        planner.open(required_frames={"body_base", left_frame, right_frame})
        T_base_object_live = reconstruct_object(planner, left_frame, T_left_tcp_object)
        validate_live_object(args, T_base_object_live)
        pregrasp_plan = planner.plan_joint_trajectory(
            "right", right_pregrasp, planner="rrtstar"
        )
        if pregrasp_plan is None:
            raise RuntimeError("Planner returned no taught right pregrasp plan")
        pregrasp_summary = validate_arm_joint_plan(
            pregrasp_plan, right_pregrasp, arm_id=2,
            label="right_taught_pregrasp",
            tolerance_deg=args.joint_tolerance_deg,
        )
        print(
            "[右手交接规划] 固定示教关节序列："
            f"预抓取={np.asarray(right_pregrasp).round(4).tolist()} -> "
            f"尾部抓取={np.asarray(right_grasp).round(4).tolist()}；"
            "两段均使用RRTStar，ICP仅核验工件交接位姿。",
            flush=True,
        )
        if args.verbose_prediction_json:
            print(json.dumps({
                "taught_pregrasp_joints_deg": right_pregrasp,
                "taught_grasp_joints_deg": right_grasp,
                "pregrasp_plan": pregrasp_summary,
                "right_place_ready_joints_deg": right_ready,
                "left_return_joints_deg": left_return,
            }, indent=2, ensure_ascii=False))
        if not args.execute:
            print("[完成] 工件位姿核验和右手示教预抓取RRTStar规划完成；机械臂与夹爪未动作。")
            return 0

        print(move_gripper(
            args.gripper_host, 2, args.right_open_position,
            args.gripper_force, "right",
        ))
        if not confirmed(
            args,
            "确认右手到示教预抓取关节位的RRTStar路径安全，输入 y 执行：",
        ):
            return 0
        planner.execute_planned_result(pregrasp_plan)
        wait_for_joint_target(
            args.arm_host, 2, right_pregrasp, args.joint_tolerance_deg,
            "right taught pregrasp", timeout_s=90.0,
        )
        T_base_right_tcp_live_pregrasp = read_tcp(planner, right_frame)
        if args.verbose_prediction_json:
            print(json.dumps({
                "right_pregrasp_arrival": {
                    "gate": "joint_feedback",
                    "planner_live_tcp_m": (
                        T_base_right_tcp_live_pregrasp[:3, 3].tolist()
                    ),
                }
            }, indent=2, ensure_ascii=False))

        # Revalidate the left-held object before entering the taught grasp pose.
        T_base_object_live = reconstruct_object(planner, left_frame, T_left_tcp_object)
        validate_live_object(args, T_base_object_live)
        grasp_plan = planner.plan_joint_trajectory(
            "right", right_grasp, planner="rrtstar"
        )
        if grasp_plan is None:
            raise RuntimeError("Planner returned no taught right tail-grasp plan")
        validate_arm_joint_plan(
            grasp_plan, right_grasp, arm_id=2,
            label="right_taught_tail_grasp",
            tolerance_deg=args.joint_tolerance_deg,
        )
        if not confirmed(args, "确认从预抓取关节位到尾部示教抓取关节位的RRTStar路径安全，输入 y 执行："):
            return 0
        planner.execute_planned_result(grasp_plan)
        wait_for_joint_target(
            args.arm_host, 2, right_grasp, args.joint_tolerance_deg,
            "right taught tail grasp", timeout_s=90.0,
        )
        T_base_right_tcp_live_grasp = read_tcp(planner, right_frame)

        accumulated_advance_mm = 0.0
        if args.require_confirmation:
            while True:
                answer = input(
                    "检查工件2尾部是否已经进入右夹爪："
                    "输入 y 闭合夹爪；输入 a 沿实时工件+X继续推进"
                    f" {args.tail_advance_step_mm:.1f}mm；输入 q 停止："
                ).strip().lower()
                if answer == "y":
                    break
                if answer in {"q", "n", "no"}:
                    print("[停止] 右夹爪保持张开，左手继续夹持工件。")
                    return 0
                if answer != "a":
                    print("请输入 y、a 或 q。")
                    continue
                next_advance_mm = (
                    accumulated_advance_mm + float(args.tail_advance_step_mm)
                )
                if next_advance_mm > float(args.tail_advance_max_mm) + 1e-9:
                    raise RuntimeError(
                        "沿工件+X累计推进将超过安全上限 "
                        f"{args.tail_advance_max_mm:.1f}mm；左手继续夹持，"
                        "请重新核对示教位和工件TF"
                    )
                T_base_object_live = reconstruct_object(
                    planner, left_frame, T_left_tcp_object
                )
                validate_live_object(args, T_base_object_live)
                advanced_grasp = advance_grasp_along_object_x(
                    T_base_right_tcp_live_grasp,
                    T_base_object_live,
                    next_advance_mm / 1000.0,
                )
                advance_plan = planner.plan_cartesian_pose(
                    "right", transform_to_object_pose(advanced_grasp), planner="lin"
                )
                if advance_plan is None:
                    raise RuntimeError(
                        f"Planner未返回工件+X累计推进{next_advance_mm:.1f}mm轨迹"
                    )
                validate_right_cartesian_plan(
                    advance_plan, right_grasp, advanced_grasp,
                    args.arrival_translation_tolerance_mm,
                    args.arrival_rotation_tolerance_deg,
                )
                planner.execute_planned_result(advance_plan)
                wait_for_pose_close(
                    planner, right_frame, advanced_grasp,
                    args.arrival_translation_tolerance_mm,
                    args.arrival_rotation_tolerance_deg,
                    f"right tail advance {next_advance_mm:.1f}mm",
                    timeout_s=30.0,
                )
                accumulated_advance_mm = next_advance_mm
                print(
                    f"[尾部接近] 已沿实时工件+X累计推进 "
                    f"{accumulated_advance_mm:.1f}mm；右夹爪保持张开。",
                    flush=True,
                )
        else:
            print(
                "[警告] 未启用--require-confirmation，无法人工分步确认尾部距离；"
                "将在示教抓取位直接闭合。",
                flush=True,
            )
        print(move_gripper(
            args.gripper_host, 2, args.right_close_position,
            args.gripper_force, "right",
        ))
        if not confirmed(args, "确认右手已夹稳工件2尾部，输入 y 释放左夹爪："):
            print("[停止] 双手共同夹持工件。")
            return 0
        if args.switch_attachment:
            print(request_attachment_owner(
                args.attachment_state, "right", timeout_s=5.0
            ))

        # Capture the rigid held-object relation while both arms are still at
        # the known handoff pose.  This makes the workpiece-frame clearance
        # deterministic even after the left gripper opens.
        T_base_object_at_release = reconstruct_object(
            planner, left_frame, T_left_tcp_object
        )
        T_base_right_tcp_at_release = read_tcp(planner, right_frame)
        T_right_tcp_object = validate_transform(
            np.linalg.inv(T_base_right_tcp_at_release) @ T_base_object_at_release,
            "T_right_tcp_object_at_release",
        )
        T_base_right_tcp_clearance = build_post_release_clearance_target(
            T_base_right_tcp_at_release,
            T_right_tcp_object,
            args.post_release_clearance_minus_x_m,
            args.post_release_clearance_plus_z_m,
        )
        save_held_object_state(
            args.held_object_state,
            T_right_tcp_object,
            T_base_right_tcp_at_release,
            T_base_right_tcp_clearance,
            "relation_captured_before_left_release",
        )
        print(move_gripper(
            args.gripper_host, 1, args.left_open_position,
            args.gripper_force, "left",
        ))
        save_held_object_state(
            args.held_object_state,
            T_right_tcp_object,
            T_base_right_tcp_at_release,
            T_base_right_tcp_clearance,
            "left_released",
        )
        held_state = load_held_object_state(args.held_object_state)

        return finish_after_left_release(
            planner, args, right_ready, left_return, right_frame, held_state
        )
    finally:
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
