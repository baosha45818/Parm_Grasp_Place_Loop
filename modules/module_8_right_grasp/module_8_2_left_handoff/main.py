#!/usr/bin/env python3
"""Module 8-2：右臂从左手接取工件并撤离到 Module 9 上方准备位。

执行前提：

* Module 5-2 已将左手持件送到交接位；
* ``--result`` 指向同一轮观察生成的 ICP ``result.json``；
* 从观察到本模块释放左夹爪之前，工件没有相对左夹爪滑动。

本模块包含双手交接的原子安全步骤，因此会控制右臂、两侧夹爪，并在右臂
撤离后让左臂回到当前工件profile定义的安全回程位。工件1从 Idle 到预抓取使用
Planner RRTStar（可回放历史轨迹），预抓取到最终抓取使用 Planner LIN；工件2沿用
其现场示教 RRTStar 抓取序列。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules.module_8_right_grasp.module_8_2_left_handoff.gripper_compat import (
    move_gripper as move_gripper_compatible,
    read_gripper_state as read_gripper_state_compatible,
    release_left_gripper,
)
from modules.reverse_grasp_handoff_positions import (
    load_workpiece_handoff_profile,
)
from modules.module_8_right_grasp.module_8_2_left_handoff.trajectory_cache import (
    compatibility as pregrasp_cache_compatibility,
    load as load_pregrasp_trajectory_cache,
    save as save_pregrasp_trajectory_cache,
)
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(
        str(PYTHON),
        [
            str(PYTHON),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )
# 交接预设位全部存于 modules/reverse_grasp_handoff_positions.toml。
# 下面四个名称只保留给旧的独立脚本 import；总控执行路径会在每次运行时重读 TOML。
_WORKPIECE_1_DEFAULTS = load_workpiece_handoff_profile("1")
WORKPIECE_1_HANDOFF_CENTER_BODY_M = list(_WORKPIECE_1_DEFAULTS["handoff"]["center_body_m"])
WORKPIECE_1_RIGHT_GRASP_JOINTS_DEG = list(_WORKPIECE_1_DEFAULTS["right_handoff"]["grasp_joints_deg"])
WORKPIECE_1_RIGHT_PLACE_READY_JOINTS_DEG = list(_WORKPIECE_1_DEFAULTS["right_handoff"]["place_ready_joints_deg"])
WORKPIECE_1_RIGHT_PLACE_READY_TOLERANCE_DEG = float(_WORKPIECE_1_DEFAULTS["right_handoff"]["place_ready_tolerance_deg"])
V2_RIGHT_GRASP_JOINTS_DEG = list(WORKPIECE_1_RIGHT_GRASP_JOINTS_DEG)
V2_RIGHT_ABOVE_MODULE_9_JOINTS_DEG = list(WORKPIECE_1_RIGHT_PLACE_READY_JOINTS_DEG)
V2_RIGHT_ABOVE_MODULE_9_TOLERANCE_DEG = WORKPIECE_1_RIGHT_PLACE_READY_TOLERANCE_DEG
WORKPIECE_2_RUNTIME_DIR = Path(__file__).resolve().parent / "workpiece_2_runtime"
WORKPIECE_2_HANDOFF_EXECUTOR = (
    WORKPIECE_2_RUNTIME_DIR / "execute_icp_tail_handoff.py"
)
WORKPIECE_2_HANDOFF_CONFIG = WORKPIECE_2_RUNTIME_DIR / "handoff_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Module 5-2 本轮生成并用于移动左手到交接位的 result.json",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"))
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument(
        "--workpiece",
        choices=("1", "2"),
        default=os.environ.get("PARM_LOOP_WORKPIECE", "1"),
        help="工件编号；总控默认从 PARM_LOOP_WORKPIECE 传入，缺省为原工件1流程",
    )
    parser.add_argument(
        "--mode", choices=("xiangxin", "gzhu"),
        default=os.environ.get("PARM_DEPLOYMENT_MODE", "gzhu"),
        help="必须与 Module 4-1 / Module 5-2 使用相同的现场部署模式",
    )
    parser.add_argument(
        "--right-place-ready-file",
        type=Path,
        help="可选：右臂到放置准备位后写入的总控同步标记文件",
    )
    parser.add_argument(
        "--right-pregrasp-ready-file",
        type=Path,
        help="Module 5-2 串行右臂阶段写入的预抓取到位标记；缺省时按 result.json 同目录自动查找",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=4.0 * float(os.environ.get("PARM_RIGHT_SPEED_SCALE", "1.0")),
        help="相对原流程速度倍率，默认4.0",
    )
    parser.add_argument(
        "--require-confirmation",
        action="store_true",
        help="恢复各运动阶段人工确认；默认自动确认",
    )
    return parser.parse_args()


def cancel_both_arms(arm_host: str) -> None:
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(
            f"[stop] 无法导入parm_sdk，未能发送cancel_task(): {exc}",
            flush=True,
        )
        return
    for arm_id, label in ((1, "left"), (2, "right")):
        arm = None
        try:
            arm = Arm(host=arm_host, status_port=None)
            arm.cancel_task(arm_id)
            print(
                f"[stop] {label} arm cancel_task({arm_id}) 已发送",
                flush=True,
            )
        except Exception as exc:
            print(f"[stop] {label} arm cancel_task失败: {exc}", flush=True)
        finally:
            if arm is not None:
                try:
                    arm.close()
                except Exception:
                    pass


def execute_movel_handoff(result: Path, args: argparse.Namespace) -> int:
    """Run workpiece-1 at the shared high handoff with its original grasp relation."""
    from tools.planner_runtime import PlannerRuntime
    from tools.handoff_observation import execute_taught_tail_handoff as handoff

    profile = load_workpiece_handoff_profile("1", mode=args.mode)
    right_handoff = profile["right_handoff"]
    high_handoff_center = list(profile["handoff"]["center_body_m"])
    high_grasp_joints_deg = list(right_handoff["grasp_joints_deg"])
    place_ready_joints_deg = list(right_handoff["place_ready_joints_deg"])
    place_ready_tolerance_deg = float(right_handoff["place_ready_tolerance_deg"])
    # This taught pose was recorded directly at the new high handoff center;
    # applying the old->new center TF delta again would double-count the move.
    taught_grasp_tcp = handoff.validate_transform(
        handoff.taught_right_tcp_from_joints(
            high_grasp_joints_deg
        ),
        "workpiece_1_updated_high_right_grasp",
    )

    original_plan_joint = PlannerRuntime.plan_joint_trajectory
    original_execute_plan = PlannerRuntime.execute_planned_result
    original_validate_arm_plan = handoff.validate_arm_joint_plan
    original_wait_for_pose_close = handoff.wait_for_pose_close
    original_move_gripper = handoff.move_gripper
    original_read_gripper_state = handoff.read_gripper_state
    original_confirmed = handoff.confirmed
    original_argv = sys.argv
    pregrasp_cache_context = pregrasp_cache_compatibility(
        mode=args.mode,
        profile_id=str(profile["profile_id"]),
        grasp_joints_deg=high_grasp_joints_deg,
    )
    pending_pregrasp_cache_seed: dict[str, Any] | None = None
    cached_pregrasp_endpoint: list[float] | None = None
    pregrasp_marker = args.right_pregrasp_ready_file or result.with_name(
        f"{result.stem}_right_pregrasp_ready.json"
    )
    parallel_pregrasp_endpoint: list[float] | None = None
    if pregrasp_marker.is_file():
        try:
            marker = json.loads(pregrasp_marker.read_text(encoding="utf-8"))
            endpoint = marker.get("right_joints_deg")
            if marker.get("event") != "right_pregrasp_ready" or not isinstance(endpoint, list) or len(endpoint) != 7:
                raise ValueError("标记内容不完整")
            parallel_pregrasp_endpoint = [float(value) for value in endpoint]
            print(
                f"[module-8-2] 采用 M5.2 预抓取到位标记：{pregrasp_marker}；"
                "跳过重复 RRTstar，等待本轮 ICP 的最终 LIN。",
                flush=True,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"右臂预抓取到位标记无效：{pregrasp_marker}: {exc}") from exc

    def plan_joint_with_right_movel(
        runtime: PlannerRuntime,
        arm: str,
        joints_deg: list[float],
        *,
        planner: str = "rrtstar",
    ) -> dict[str, Any] | None:
        nonlocal cached_pregrasp_endpoint, pending_pregrasp_cache_seed
        if arm != "right":
            return original_plan_joint(
                runtime, arm, joints_deg, planner=planner
            )
        is_above_module_9 = (
            handoff.max_joint_error_deg(
                list(joints_deg), place_ready_joints_deg
            )
            <= 1e-6
        )
        if is_above_module_9:
            print(
                "[V2 MoveJ] 交接撤离完成后，右臂关节规划到"
                f"模块9上方准备位：{[round(value, 4) for value in joints_deg]}。",
                flush=True,
            )
            return original_plan_joint(
                runtime, arm, joints_deg, planner=planner
            )
        is_exact_taught_grasp = (
            handoff.max_joint_error_deg(
                list(joints_deg), high_grasp_joints_deg
            )
            <= 1e-6
        )
        if is_exact_taught_grasp:
            print(
                "[V2 最终交接] 右臂从预抓取位以 Planner LIN 直线到"
                f"godr1验证的示教抓取位：{[round(value, 4) for value in joints_deg]}。",
                flush=True,
            )
            target_tcp = handoff.taught_right_tcp_from_joints(list(joints_deg))
            return runtime.plan_cartesian_pose(
                "right", handoff.transform_to_object_pose(target_tcp), planner="lin"
            )
        cached, cache_status = load_pregrasp_trajectory_cache(
            workpiece="1", compatibility_data=pregrasp_cache_context
        )
        if cached is not None:
            response = cached["response"]
            trajectory = (response.get("data") or {}).get("trajectory") or []
            try:
                handoff.validate_trajectory_start_matches_current_state(
                    response, arm_host=args.arm_host, arm_id=2,
                    label="right_idle_to_pregrasp_cached_trajectory", tolerance_deg=1.0,
                )
                start_matches = True
            except RuntimeError as exc:
                start_matches = False
                print(f"[轨迹缓存] Idle→预抓取首点不匹配：{exc}", flush=True)
            if start_matches:
                cached_pregrasp_endpoint = [float(value) for value in trajectory[-1]]
                print(
                    "[轨迹缓存] 直接回放 Idle→预抓取 RRTstar 历史轨迹；"
                    "本轮 ICP 只用于后续 LIN 最终抓取。", flush=True,
                )
                return response
            cache_status = "start_mismatch"
        print(
            f"[轨迹缓存] Idle→预抓取未命中（{cache_status}），"
            "RRTstar 在线规划并在到位后更新缓存。", flush=True,
        )
        response = original_plan_joint(runtime, arm, joints_deg, planner="rrtstar")
        if response is not None:
            pending_pregrasp_cache_seed = response
        return response

    def execute_plan_and_seed_pregrasp_cache(
        runtime: PlannerRuntime, response: dict[str, Any]
    ) -> None:
        nonlocal pending_pregrasp_cache_seed
        original_execute_plan(runtime, response)
        if response is not pending_pregrasp_cache_seed:
            return
        trajectory = (response.get("data") or {}).get("trajectory") or []
        if not trajectory:
            raise RuntimeError("Idle→预抓取在线轨迹缺少终点")
        handoff.wait_for_joint_target(
            args.arm_host, 2, [float(value) for value in trajectory[-1]], 2.0,
            "right idle-to-pregrasp cache seed", timeout_s=90.0,
        )
        save_pregrasp_trajectory_cache(
            workpiece="1", compatibility_data=pregrasp_cache_context, response=response
        )
        pending_pregrasp_cache_seed = None

    def validate_right_movel_as_arm_plan(
        response: dict,
        target_deg: list[float],
        *,
        arm_id: int,
        label: str,
        tolerance_deg: float,
    ) -> dict[str, Any]:
        if label == "right_pregrasp_trajectory" and cached_pregrasp_endpoint is not None:
            data = response.get("data") or {}
            trajectory = data.get("trajectory") or []
            if int(data.get("arm_id", -1)) != 2 or not trajectory:
                raise RuntimeError("缓存 Idle→预抓取轨迹无有效右臂终点")
            return {
                "arm_id": 2,
                "planner": "RRTSTAR_TRAJECTORY_CACHE",
                "trajectory_points": len(trajectory),
                "sample_frequency_hz": float(data.get("sample_frequency", 10.0)),
                "endpoint_joints_deg": list(cached_pregrasp_endpoint),
            }
        # 在线预抓取和最终 LIN 都保留原有终点关节校验。
        return original_validate_arm_plan(
            response, target_deg, arm_id=arm_id, label=label,
            tolerance_deg=tolerance_deg,
        )

    def wait_for_pose_and_module_9_joints(
        planner: Any,
        frame: str,
        target: Any,
        translation_tolerance_mm: float,
        rotation_tolerance_deg: float,
        label: str,
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        if (
            label == "right taught tail pregrasp pose"
            and cached_pregrasp_endpoint is not None
        ):
            print(
                "[轨迹缓存] 预抓取使用缓存安全接近位；"
                "不要求它与本轮 ICP 预抓取 TCP 重合。",
                flush=True,
            )
            handoff.wait_for_joint_target(
                args.arm_host, 2, cached_pregrasp_endpoint, 0.5,
                "workpiece-1 cached right pregrasp", timeout_s=90.0,
            )
            return {
                "gate": "cached_joint_feedback",
                "target_joints_deg": list(cached_pregrasp_endpoint),
                "tolerance_deg": 0.5,
            }
        if label == "right place-ready pose":
            print(
                "[工件1到位门禁] 放置准备位是关节示教目标；"
                "跳过未标定FK的TCP比较，直接等待右臂七个关节到位。",
                flush=True,
            )
            handoff.wait_for_joint_target(
                args.arm_host,
                2,
                place_ready_joints_deg,
                place_ready_tolerance_deg,
                "workpiece-1 right place-ready",
                timeout_s=90.0,
            )
            return {
                "gate": "joint_feedback",
                "target_joints_deg": list(
                    place_ready_joints_deg
                ),
                "tolerance_deg": float(
                    place_ready_tolerance_deg
                ),
            }
        return original_wait_for_pose_close(
            planner,
            frame,
            target,
            translation_tolerance_mm,
            rotation_tolerance_deg,
            label,
            timeout_s=timeout_s,
        )

    def move_gripper_with_left_release_preemption(
        host: str,
        hand: int,
        position: float,
        force: float,
        label: str,
    ) -> dict[str, Any]:
        if hand != 1 or label != "left" or position <= 5.0:
            return move_gripper_compatible(host, hand, position, force, label)
        return release_left_gripper(host, position, force, label)

    def confirmed_with_v2_prompt(prompt: str) -> bool:
        prompt = prompt.replace(
            "关节PTP到工件尾端外侧预抓取位",
            "关节RRTStar到工件尾端外侧预抓取位",
        )
        return original_confirmed(prompt)

    command_args = [
        "execute_taught_tail_handoff",
        "--result",
        str(result),
        "--planner-host",
        args.planner_host,
        "--arm-host",
        args.arm_host,
        "--gripper-host",
        args.gripper_host,
        "--speed-multiplier",
        str(args.speed_multiplier),
        "--right-grasp-joints-deg",
        *[str(value) for value in high_grasp_joints_deg],
        "--final-grasp-motion",
        "lin",
        "--right-grasp-tolerance-deg",
        "0.5",
        "--expected-center-m",
        *[str(value) for value in high_handoff_center],
        "--taught-object-tilt-deg",
        "0.0",
        "--pregrasp-distance-m",
        "0.05",
        "--pregrasp-axis",
        "object-minus-x",
        *(
            ["--right-pregrasp-ready-joints-deg", *[str(value) for value in parallel_pregrasp_endpoint]]
            if parallel_pregrasp_endpoint is not None else []
        ),
        "--arrival-translation-tolerance-mm",
        "1.0",
        "--right-place-ready-joints-deg",
        *[str(value) for value in place_ready_joints_deg],
        "--right-place-ready-tolerance-deg",
        str(place_ready_tolerance_deg),
        "--left-return-joints-deg",
        *[str(value) for value in profile["left_return_joints_deg"]],
        *(
            ["--right-place-ready-file", str(args.right_place_ready_file)]
            if args.right_place_ready_file
            else []
        ),
        *(
            ["--require-confirmation"]
            if args.require_confirmation
            else []
        ),
        "--execute",
    ]
    try:
        PlannerRuntime.plan_joint_trajectory = plan_joint_with_right_movel
        PlannerRuntime.execute_planned_result = execute_plan_and_seed_pregrasp_cache
        handoff.validate_arm_joint_plan = validate_right_movel_as_arm_plan
        handoff.wait_for_pose_close = wait_for_pose_and_module_9_joints
        handoff.move_gripper = move_gripper_with_left_release_preemption
        handoff.read_gripper_state = read_gripper_state_compatible
        handoff.confirmed = confirmed_with_v2_prompt
        sys.argv = command_args
        print(
            "\n[module-8-2] 工件1进入共用高位右臂交接：Idle→预抓取使用"
            "RRTStar（命中时回放历史轨迹）；预抓取→最终交接使用Planner LIN；"
            "交接后上升、后退使用MoveL，随后右臂MoveJ到模块9上方准备位；"
            "左臂关节回程保持MoveJ。",
            flush=True,
        )
        print(
            "[module-8-2] 工件1高位名义右手抓取TCP：body_base "
            f"xyz(m)={taught_grasp_tcp[:3, 3].tolist()}；"
            "预抓取仍按实时工件关系计算；最终抓取以实时左TCP×ICP关系"
            "重建抓取TCP后执行 LIN。",
            flush=True,
        )
        return handoff.main()
    finally:
        PlannerRuntime.plan_joint_trajectory = original_plan_joint
        PlannerRuntime.execute_planned_result = original_execute_plan
        handoff.validate_arm_joint_plan = original_validate_arm_plan
        handoff.wait_for_pose_close = original_wait_for_pose_close
        handoff.move_gripper = original_move_gripper
        handoff.read_gripper_state = original_read_gripper_state
        handoff.confirmed = original_confirmed
        sys.argv = original_argv


def write_right_place_ready_marker(
    destination: Path, profile: dict[str, Any]
) -> None:
    """Only publish after the workpiece-2 executor has reached its ready pose."""
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "event": "right_place_ready",
                "workpiece": "2",
                "profile_id": profile["profile_id"],
                "right_joints_deg": profile["right_handoff"][
                    "place_ready_joints_deg"
                ],
                "created_at": time.time(),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"[module-8-2/workpiece2] 已写入右臂放置准备标记：{destination}", flush=True)


def execute_workpiece_2_handoff(result: Path, args: argparse.Namespace) -> int:
    """Run only workpiece-2 handoff; final right placement remains Module 9."""
    profile = load_workpiece_handoff_profile("2", mode=args.mode)
    handoff = profile["handoff"]
    right_handoff = profile["right_handoff"]
    for path, label in (
        (WORKPIECE_2_HANDOFF_EXECUTOR, "工件2右手交接执行器"),
        (WORKPIECE_2_HANDOFF_CONFIG, "工件2右手交接配置"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    command = [
        str(PYTHON),
        str(WORKPIECE_2_HANDOFF_EXECUTOR),
        "--result", str(result),
        "--config", str(WORKPIECE_2_HANDOFF_CONFIG),
        "--expected-center-m",
        *(str(value) for value in handoff["center_body_m"]),
        "--center-tolerance-mm", str(handoff["center_tolerance_mm"]),
        "--axis-tolerance-deg", str(handoff["axis_tolerance_deg"]),
        "--right-pregrasp-joints-deg",
        *(str(value) for value in right_handoff["pregrasp_joints_deg"]),
        "--right-grasp-joints-deg",
        *(str(value) for value in right_handoff["grasp_joints_deg"]),
        "--right-place-ready-joints-deg",
        *(str(value) for value in right_handoff["place_ready_joints_deg"]),
        "--left-return-joints-deg",
        *(str(value) for value in profile["left_return_joints_deg"]),
        "--post-release-clearance-minus-x-m",
        str(right_handoff["post_release_clearance_minus_x_m"]),
        "--post-release-clearance-plus-z-m",
        str(right_handoff["post_release_clearance_plus_z_m"]),
        "--speed-multiplier", str(args.speed_multiplier),
        "--planner-host", args.planner_host,
        "--gripper-host", args.gripper_host,
        "--arm-host", args.arm_host,
        "--execute",
        *( ["--require-confirmation"] if args.require_confirmation else [] ),
    ]
    print(
        "\n[module-8-2/workpiece2] 只执行右手从左手接取、"
        "左夹爪释放和双方撤离到模块9准备位；不执行右手最终放置。",
        flush=True,
    )
    print("[module-8-2/workpiece2] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if args.right_place_ready_file:
        write_right_place_ready_marker(args.right_place_ready_file, profile)
    return 0


def main() -> int:
    args = parse_args()
    os.environ["PARM_DEPLOYMENT_MODE"] = args.mode
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    result = args.result.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Module 5-2 ICP结果不存在：{result}")

    if args.workpiece == "2":
        execute_workpiece_2_handoff(result, args)
        print(
            "[module-8-2/workpiece2] 右手已接取工件2并撤离到模块9准备位；"
            "左手已释放并回到工件2较高安全回程位。",
            flush=True,
        )
        return 0

    execute_movel_handoff(result, args)
    print(
        "[module-8-2] 右手在共用高位以实时ICP修正的LIN完成接取并撤离，随后MoveJ到模块9上方准备位；"
        "左手已释放并回到共用较高安全位。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stop] 收到Ctrl+C，直接停止左右臂任务。", flush=True)
        cancel_both_arms(parse_args().arm_host)
        raise SystemExit(130)
