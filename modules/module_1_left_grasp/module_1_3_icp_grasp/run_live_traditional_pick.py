#!/usr/bin/env python3
"""Live pure-traditional pick: D405 -> SAM -> STL ICP -> Parm Planner.

No GraspNet model or inference service is used.  The default mode captures,
generates a Planner plan and performs Planner reachability preflight only.
Actual motion additionally requires both --execute and --confirm-real-grasp.
Once enabled, all traditional-ICP motion confirmations are automatic; the
loop execution permit remains the cross-arm safety gate.  The motion policy
is a 10 cm vertical pregrasp followed by a LIN descent.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from weight_verification import (
    WeightCheckConfig,
    collect_empty_baseline,
    force_sensor,
    verify_graspup_force_signature,
    verify_single_after_lift,
    WeightQualityRejected,
)


LOOP = Path(__file__).resolve().parents[3]
TOOL = LOOP / "tool"
GRASPNET_ROOT = LOOP / "third_party/graspnet"
HERE = Path(__file__).resolve().parent
OFFLINE = HERE / "offline_traditional_sam_stl_icp.py"
PYTHON = LOOP / ".venv/bin/python"
WEIGHT_CONFIG_PATH = LOOP / "config/grasp_weight_thresholds.conf"


class TraditionalPlannerUnreachable(RuntimeError):
    """No candidate from one fresh traditional observation has a safe path."""


def publish_observation_ready() -> None:
    """Tell the loop that the left arm is parked at the D405 observation pose."""
    value = os.environ.get("LOOP_GRASPNET_OBSERVATION_READY_FILE")
    if not value:
        return
    ready_path = Path(value)
    temporary = ready_path.with_name(f".{ready_path.name}.tmp")
    temporary.write_text("ready\n", encoding="utf-8")
    temporary.replace(ready_path)
    print("-> 左臂已到传统 ICP 观察位；已通知总控可以启动右臂 ServoPose。", flush=True)


def wait_for_execution_permit() -> None:
    """Keep real pregrasp motion behind the loop's left/right-arm safety gate."""
    value = os.environ.get("LOOP_GRASPNET_EXECUTION_PERMIT_FILE")
    if not value:
        return
    permit_path = Path(value)
    print(
        "-> 传统 ICP 候选和预抓取规划已完成；等待总控放行真实抓取："
        f"{permit_path}",
        flush=True,
    )
    while True:
        if permit_path.is_file():
            content = permit_path.read_text(encoding="utf-8").strip().lower()
            if content == "allow":
                print("-> 总控已放行；开始执行传统 ICP 预抓取轨迹。", flush=True)
                return
            raise RuntimeError(
                f"invalid traditional ICP execution permit content in {permit_path}: {content!r}"
            )
        time.sleep(0.2)


def load_shared_weight_defaults() -> dict[str, float | int]:
    """Use exactly the same editable force/weight configuration as Version7."""
    fields = {
        "WEIGHT_EMPTY_MAX_G": "weight_empty_max_g",
        "WEIGHT_SINGLE_MIN_G": "weight_single_min_g",
        "WEIGHT_SINGLE_MAX_G": "weight_single_max_g",
        "WEIGHT_MULTIPLE_MIN_G": "weight_multiple_min_g",
        "WEIGHT_MAX_AXIS_MAD_N": "weight_max_axis_mad_n",
        "WEIGHT_CHECK_SAMPLES": "weight_check_samples",
        "WEIGHT_CHECK_TIMEOUT_S": "weight_check_timeout_s",
        "WEIGHT_CHECK_SETTLE_S": "weight_check_settle_s",
    }
    if not WEIGHT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"shared weight configuration is missing: {WEIGHT_CONFIG_PATH}")
    values: dict[str, str] = {}
    for raw_line in WEIGHT_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError(f"invalid shared weight configuration line: {raw_line!r}")
            values[key.strip()] = value.strip()
    missing = [key for key in fields if key not in values]
    if missing:
        raise ValueError(f"shared weight configuration is missing fields: {missing}")
    return {
        source_target: (
            int(values[source_key])
            if source_key == "WEIGHT_CHECK_SAMPLES"
            else float(values[source_key])
        )
        for source_key, source_target in fields.items()
    }


def eef_transform(state: dict) -> np.ndarray:
    left = ((state.get("state") or {}).get("left") or {})
    pose = left.get("eef_pose") or {}
    xyz, rpy = pose.get("xyz"), pose.get("rpy")
    if not (isinstance(xyz, list) and isinstance(rpy, list) and len(xyz) == len(rpy) == 3):
        raise RuntimeError("parm_sdk Arm.get_state() does not contain left eef_pose xyz/rpy")
    values = np.asarray([*xyz, *rpy], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("left eef_pose has non-finite values")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    T[:3, 3] = values[:3]
    return T


def capture_log(run: Path, args: argparse.Namespace) -> Path:
    """Request D405 data from the singleton Camera Broker for traditional ICP."""
    sys.path.insert(0, str(TOOL))
    from parm_sdk import Arm
    sys.path.insert(0, str(GRASPNET_ROOT / "deploy" / "version7"))
    from camera_client import capture as broker_capture
    from camera_client import status as broker_status

    with Arm(host=args.arm_host) as arm:
        state = arm.get_state() or {}
    T_base_eef = eef_transform(state)
    raw = run / "raw/d405"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise RuntimeError(f"traditional raw capture directory already exists: {raw}")

    status = broker_status(args.camera_broker_socket, timeout_s=args.camera_broker_timeout_s)
    allow_root = Path(status["allow_root"]).resolve()
    broker_capture_dir = (
        allow_root / "traditional_icp" /
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex}"
    )
    result = broker_capture(
        args.camera_broker_socket,
        {
            "camera": "d405",
            "output_dir": str(broker_capture_dir),
            "frame_count": int(args.frames),
        },
        timeout_s=args.camera_broker_timeout_s,
    )
    try:
        shutil.copytree(broker_capture_dir, raw)
    finally:
        # Direct-capture workspaces are temporary; preserve only the copy
        # inside this traditional run after the Broker has finished writing.
        shutil.rmtree(broker_capture_dir, ignore_errors=True)

    meta_path = raw / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Keep traditional ICP's established pose source unchanged.  The Broker
    # owns only the camera; this Arm.get_state snapshot remains the recorded
    # T_base_eef consumed by the offline ICP stage.
    meta.update({
        "camera_name": "d405",
        "serial": args.d405_serial,
        "T_base_eef": T_base_eef.tolist(),
        "frame_count": int(args.frames),
        "camera_broker_socket": args.camera_broker_socket,
        "camera_broker_capture_elapsed_s": result.get("capture_elapsed_s"),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        "-> traditional D405 capture received from Camera Broker: "
        f"frames={args.frames}, elapsed={float(result['capture_elapsed_s']):.3f}s",
        flush=True,
    )
    return run


def traditional_perception(run: Path, args: argparse.Namespace) -> dict:
    """Run the non-actuating SAM/depth/ICP stage for one saved D405 view."""
    subprocess.run([str(PYTHON), str(OFFLINE), "--log", str(run), "--output-dir", str(run / "traditional"),
                    "--sam-socket", args.sam_socket,
                    "--min-single-stl-observed-coverage", str(args.min_single_stl_observed_coverage),
                    "--max-observed-to-stl-extent-ratio", str(args.max_observed_to_stl_extent_ratio),
                    "--min-grasp-angle-to-table-deg", str(args.min_grasp_angle_to_table_deg),
                    "--front-axis-center-min-icp-fitness", str(args.front_axis_center_min_icp_fitness),
                    "--min-jaw-side-support-m", str(args.min_jaw_side_support_m),
                    "--jaw-contact-half-span-m", str(args.jaw_contact_half_span_m),
                    "--min-other-instance-gripper-clearance-m", str(args.min_other_instance_gripper_clearance_m),
                    "--allow-other-instance-below-grasp-m", str(args.allow_other_instance_below_grasp_m),
                    "--max-auto-grasp-up-m", str(args.max_auto_grasp_up_m),
                    "--front-grasp-upward-offset-m", str(args.front_grasp_upward_offset_m),
                    "--back-grasp-upward-offset-m", str(args.back_grasp_upward_offset_m),
                    *([] if args.grasp_downward_probe_m is None else [
                        "--grasp-downward-probe-m", str(args.grasp_downward_probe_m)
                    ])], check=True)
    report_path = run / "traditional/traditional_pose_report.json"
    if not report_path.is_file():
        raise RuntimeError("traditional perception did not produce a pose report")
    return json.loads(report_path.read_text())


def ranked_planner_records(report: dict) -> list[dict]:
    return sorted(
        (record for record in report.get("records", []) if record.get("accepted") and record.get("planner_candidate_rank")),
        key=lambda record: int(record["planner_candidate_rank"]),
    )


def planner_args() -> argparse.Namespace:
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe
    args = pe.build_parser().parse_args([])
    args.planner_host = "127.0.0.1"; args.planner_trajectory_port = 5604
    args.planner_tf_port = 5609; args.planner_waypoint_port = 5610
    args.planner_execute_url = "http://127.0.0.1:5667/middle/waypoint/execute"
    args.planner_bridge_calibration = str(
        TOOL / "calibration/graspnet_planner_bridge_5604_20260803/planner_tf_bridge_5604_candidate.json"
    )
    # Exact Module 1-1/V7 production Planner convention.  Its direct-center
    # mode consumes the calibrated plan pose as the left_eef IK target.
    args.planner_base_frame = "body_base"; args.planner_tool_frame = "left_eef"
    args.planner_direct_grasp_center = True
    args.planner_group = "left_arm"; args.arm_id = 1; args.arm_host = "192.168.88.1"
    # Keep the user-requested 0.2 velocity.  Pilz rejected recent LIN paths
    # at 0.2 acceleration by a small joint3 limit overshoot (2.441 > 2.4),
    # so use a conservative acceleration scale without slowing cruise speed.
    args.planner_max_vel = .2; args.planner_max_acc = .18
    args.planner_pregrasp_mode = "rrtstar"; args.planner_descend_mode = "lin"
    args.grasp_weight_check = False; args.stop_after_graspup = True
    return args


def parallel_jaw_variants(plan: dict) -> list[tuple[float, dict]]:
    """Return the two physically equivalent parallel-jaw roll orientations.

    The grasp TCP lies on the EEF +X axis.  A 180-degree roll about that axis
    leaves the chosen pinch point and vertical approach unchanged, but swaps
    which physical finger/hand side faces the table.  In dense piles one of
    those equivalent URDF poses can be collision-free while the other is not.
    """
    original = copy.deepcopy(plan)
    R0 = np.asarray(original["eef_rotation_base"], dtype=float)
    pre_tcp = np.asarray(original["pregrasp_grasp_tcp_base_m"], dtype=float)
    # Preserve the Version7 final-Z probe for the 180-degree symmetry branch.
    grasp_tcp = np.asarray(original.get("grasp_tcp_base_m", original["grasp_center_base_m"]), dtype=float)
    variants = []
    for roll_deg, R in ((0.0, R0), (180.0, R0 @ Rotation.from_euler("x", 180, degrees=True).as_matrix())):
        item = copy.deepcopy(original)
        item["eef_rotation_base"] = R.tolist()
        item["eef_axis_angle_rad"] = Rotation.from_matrix(R).as_rotvec().tolist()
        if item.get("planner_direct_grasp_center") is True:
            item["pregrasp_eef_pose"] = np.r_[pre_tcp, item["eef_axis_angle_rad"]].tolist()
            item["grasp_eef_pose"] = np.r_[grasp_tcp, item["eef_axis_angle_rad"]].tolist()
        else:
            offset = np.asarray(json.loads(Path(item["tcp_translation_calibration"]).read_text())["T_eef_grasp_tcp"], dtype=float)[:3, 3]
            item["pregrasp_eef_pose"] = np.r_[pre_tcp - R @ offset, item["eef_axis_angle_rad"]].tolist()
            item["grasp_eef_pose"] = np.r_[grasp_tcp - R @ offset, item["eef_axis_angle_rad"]].tolist()
        item["parallel_jaw_symmetry_rotation_deg"] = roll_deg
        variants.append((roll_deg, item))
    return variants


def paired_pregrasp_and_descent(pa: argparse.Namespace, candidate: dict, *, attempts: int = 3):
    """Plan hover and descent as one IK-branch-consistent pair.

    RRTStar can legitimately select different valid IK branches for the same
    hover pose.  A LIN descent is only meaningful from the endpoint of the
    particular pregrasp trajectory that will be executed, so never replan it
    later from an unrelated fresh branch.
    """
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe

    bridge_base, bridge_tool, bridge_base_frame, bridge_tool_frame = pe.load_bridge(pa)
    failures = []
    with pe.connect_planner(pa, "traditional paired grasp planning") as planner:
        pe.resolve_planner_tool_frame(pa, planner.list_frame_ids())
        planning_base_bridge_base, bridge_tool_planning_tool = pe.planner_target_transforms(
            planner, pa, bridge_base_frame, bridge_tool_frame
        )
        for attempt in range(1, attempts + 1):
            try:
                pregrasp = pe.plan_cartesian(
                    planner, pa, candidate["plan"]["pregrasp_eef_pose"], pa.planner_pregrasp_mode,
                    f"candidate rank {candidate['rank']} paired pregrasp attempt {attempt}",
                    bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool,
                )
                descent = pe.plan_cartesian_from_joint_state(
                    planner, pa, candidate["plan"]["grasp_eef_pose"], pa.planner_descend_mode,
                    f"candidate rank {candidate['rank']} paired 10cm LIN descent attempt {attempt}",
                    bridge_base, bridge_tool, planning_base_bridge_base, bridge_tool_planning_tool,
                    pregrasp["trajectory"][-1],
                )
                return pregrasp, descent, attempt, failures
            except Exception as exc:
                failures.append(str(exc))
    raise RuntimeError(
        f"candidate rank {candidate['rank']} has no branch-consistent pregrasp/LIN pair after {attempts} attempts: "
        + " | ".join(failures)
    )


def paired_preflight(pa: argparse.Namespace, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Retain candidates only when a single IK branch supports both phases."""
    reachable, records = [], []
    for start in range(0, len(candidates), 2):
        pair = candidates[start:start + 2]
        pair_reachable = []
        for candidate in pair:
            try:
                pregrasp, descent, attempt, failures = paired_pregrasp_and_descent(pa, candidate)
            except Exception as exc:
                records.append({"rank": candidate["rank"], "source_grasp_index": candidate["source_grasp_index"],
                                "grasp_source": candidate["grasp_source"], "score": candidate["score"],
                                "parallel_jaw_symmetry_rotation_deg": candidate["parallel_jaw_symmetry_rotation_deg"],
                                "reachable": False, "reason": str(exc)})
                continue
            item = {"rank": candidate["rank"], "source_grasp_index": candidate["source_grasp_index"],
                    "grasp_source": candidate["grasp_source"], "score": candidate["score"],
                    "parallel_jaw_symmetry_rotation_deg": candidate["parallel_jaw_symmetry_rotation_deg"],
                    "reachable": True, "paired_planning_attempt": attempt,
                    "pregrasp_waypoints": len(pregrasp["trajectory"]), "lin_descent_waypoints": len(descent["trajectory"]),
                    "failed_branch_attempts": len(failures)}
            records.append(item); pair_reachable.append((candidate, item))
        if pair_reachable:
            # Equivalent jaws: the shorter path is the nearer, less
            # roundabout choice from the arm's current state.
            chosen, chosen_record = min(pair_reachable, key=lambda item: (item[1]["pregrasp_waypoints"], item[1]["lin_descent_waypoints"]))
            chosen_record["selected_nearer_parallel_jaw_symmetry"] = True
            reachable.append(chosen)
            break
    return reachable, records


def reverse_lin_trajectory(trajectory: dict) -> dict:
    """Return the exact reverse of an already Planner-validated LIN descent."""
    lift = copy.deepcopy(trajectory)
    waypoints = list(lift.get("trajectory") or [])
    if len(waypoints) < 2:
        raise RuntimeError("cannot reverse a LIN descent with fewer than two waypoints")
    lift["trajectory"] = list(reversed(waypoints))
    return lift


def mark_planner_selected_grasp(run: Path, chosen: dict) -> None:
    """Persist the actual Planner-selected instance, not merely offline OKs."""
    plan = chosen["plan"]
    selected = {
        "selected_mask_index": int(plan["selected_mask_index"]),
        "semantic_candidate_rank": int(chosen["semantic_candidate_rank"]),
        "planner_candidate_rank": int(chosen["rank"]),
        "parallel_jaw_symmetry_rotation_deg": float(chosen["parallel_jaw_symmetry_rotation_deg"]),
        "grasp_eef_pose": plan["grasp_eef_pose"],
    }
    (run / "selected_grasp.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    image_path = run / "traditional/grasp_prediction.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    projection = plan.get("predicted_parallel_jaw_contacts_image") or {}
    center = projection.get("grasp_center_pixel")
    if image is None or not isinstance(center, list) or len(center) != 2:
        return
    point = (int(center[0]), int(center[1]))
    cv2.circle(image, point, 13, (0, 165, 255), 3)
    cv2.putText(image, f"PLANNER SELECTED #{selected['selected_mask_index']} rank {selected['semantic_candidate_rank']}",
                (max(4, point[0] - 100), max(20, point[1] - 22)), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 165, 255), 2)
    cv2.imwrite(str(image_path), image)


def execute_paired_plan(
    pa: argparse.Namespace, candidate: dict, weight_config: WeightCheckConfig,
) -> dict[str, object]:
    """Freshly build, then execute, one branch-consistent hover/LIN pair."""
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe

    pe.check_pregrasp_distance(pa, candidate["plan"])
    # The launcher enables automatic confirmation for every ICP motion.  Do
    # not ask a second stdin question here; the loop permit is the actual
    # cross-arm execution gate.
    print("-> 传统 ICP 动作均自动确认；执行成对预抓取与 10cm LIN 下探")
    pregrasp, descent, attempt, _ = paired_pregrasp_and_descent(pa, candidate)
    print(f"-> executing branch-consistent pair from planning attempt {attempt}")
    # Start fully open, then reduce only to the validated silhouette width
    # (two side margins plus the operator-requested extra 16 mm) before motion.
    with force_sensor(weight_config) as sensor:
        pe.move_gripper(pa, float(candidate["plan"]["opening_percent"]), "set validated predicted jaw opening")
        pe.execute_trajectory(pa, pregrasp, "pregrasp")
        # Baseline and loaded readings use this identical wrist attitude, so
        # gravity projected in the sensor frame cancels out.
        import time
        time.sleep(weight_config.settle_s)
        try:
            baseline = collect_empty_baseline(sensor, weight_config)
        except Exception as exc:
            pe.move_gripper(pa, 100.0, "release after ICP baseline data failure")
            raise WeightQualityRejected(
                "UNKNOWN", f"ICP pregrasp force baseline failed: {exc}"
            ) from exc
        pe.execute_trajectory(pa, descent, "10cm LIN grasp descent")
        pe.close_gripper(pa)
        # The descent is a validated 10-cm vertical LIN segment.  Reversing
        # it gives the same-pose post-grasp lift required by the weight gate.
        pe.execute_trajectory(pa, reverse_lin_trajectory(descent), "post-grasp 10cm vertical LIN lift")
        try:
            handoff = verify_single_after_lift(sensor, baseline, weight_config)
        except WeightQualityRejected:
            # No uncertain grasp may proceed to Module 2.  Release it while
            # stopped at the safe pregrasp height, then start a fresh cycle.
            pe.move_gripper(pa, 100.0, "release rejected ICP grasp")
            raise
        except Exception as exc:
            pe.move_gripper(pa, 100.0, "release after ICP weight data failure")
            raise WeightQualityRejected(
                "UNKNOWN", f"ICP same-pose weight check failed: {exc}"
            ) from exc
        # Match Module 1-1/V7's successful-grasp handoff after the vertical
        # lift: keep the verified single workpiece clamped and return to
        # graspup before validating the physical hold state.
        with pe.connect_planner(pa, "post-grasp return to graspup") as planner:
            graspup = pe.plan_joint(
                planner, pa, pa.postgrasp_lift_joints_deg, "post-grasp graspup"
            )
        pe.execute_trajectory(pa, graspup, "post-grasp graspup")
        try:
            pe.require_left_gripper_holding(pa, phase="graspup")
            verify_graspup_force_signature(sensor, handoff, weight_config)
        except WeightQualityRejected:
            pe.move_gripper(pa, 100.0, "release after ICP graspup force-signature failure")
            raise
    print("-> ICP verified gripper hold and force signature at graspup; Module 2 handoff is allowed")
    return handoff


def prepare_traditional_candidate(
    args: argparse.Namespace, pa: argparse.Namespace, *, attempt_index: int,
) -> tuple[Path, dict]:
    """Capture, run SAM/STL/ICP, and return one Planner-preflighted candidate."""
    base_suffix = "" if attempt_index == 0 else f"_weight_retry_{attempt_index:02d}"
    run: Path | None = None
    report: dict | None = None
    ranked_records: list[dict] = []
    for no_grasp_retry in range(int(args.max_no_grasp_retries) + 1):
        suffix = base_suffix + ("" if no_grasp_retry == 0 else f"_no_grasp_retry_{no_grasp_retry:02d}")
        run = args.output_dir / datetime.now().strftime("live_%Y%m%d_%H%M%S_%f")
        run = run.with_name(run.name + suffix)
        capture_log(run, args)
        report = traditional_perception(run, args)
        ranked_records = ranked_planner_records(report)
        if ranked_records:
            break
        if not args.execute:
            raise RuntimeError(
                "traditional perception did not produce any Planner candidate in plan-only mode; "
                f"see {run}"
            )
        if no_grasp_retry >= int(args.max_no_grasp_retries):
            raise RuntimeError(
                "traditional perception produced no safe Planner candidate after "
                f"{no_grasp_retry + 1} observations; see {run}"
            )
        observation_script = TOOL / ("go_graspnet2.py" if no_grasp_retry == 0 else "go_graspnet.py")
        print(
            "-> no safe ICP candidate; moving to a fresh observation pose "
            f"for no-grasp retry {no_grasp_retry + 1}/{args.max_no_grasp_retries}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, str(observation_script), "--arm-host", args.arm_host, "--speed", "0.2"]
            if observation_script.name == "go_graspnet2.py"
            else [sys.executable, str(observation_script), "--arm-host", args.arm_host],
            input="y\n", text=True, check=True,
        )
    if run is None or report is None or not ranked_records:
        raise RuntimeError("traditional ICP internal error: no selected perception result")
    plans_by_rank: dict[int, dict] = {}
    for record in ranked_records:
        rank = int(record["planner_candidate_rank"])
        plan = report["planner_plan"] if rank == 1 else None
        if plan is None:
            candidate_path = run / "traditional" / f"planner_candidate_{rank}.json"
            if candidate_path.is_file():
                plan = json.loads(candidate_path.read_text())
        if plan is None:
            raise RuntimeError(f"missing persisted Planner candidate rank {rank}")
        plans_by_rank[rank] = plan
    candidates: list[dict] = []
    for semantic_rank, plan in plans_by_rank.items():
        for symmetry_order, (symmetry_deg, variant) in enumerate(parallel_jaw_variants(plan)):
            candidates.append({
                "rank": (semantic_rank - 1) * 2 + symmetry_order + 1,
                "semantic_candidate_rank": semantic_rank,
                "source_grasp_index": int(plan["selected_mask_index"]),
                "grasp_source": "traditional_sam_stl_icp",
                "score": float(plan["icp"]["fitness"]),
                "parallel_jaw_symmetry_rotation_deg": symmetry_deg,
                "plan": variant,
            })
    reachable, records = paired_preflight(pa, candidates)
    (run / "planner_preflight.json").write_text(json.dumps(records, indent=2))
    if not reachable:
        raise TraditionalPlannerUnreachable(
            f"Planner rejected the traditional grasp; see {run / 'planner_preflight.json'}"
        )
    selected = reachable[0]
    (run / "selected_planner_plan.json").write_text(json.dumps(selected["plan"], indent=2))
    mark_planner_selected_grasp(run, selected)
    print(f"-> traditional plan rank {selected['rank']} and Planner preflight succeeded: {run}")
    return run, next(item for item in candidates if item["rank"] == selected["rank"])


def run_traditional_cycles(
    args: argparse.Namespace, pa: argparse.Namespace, pe: object,
) -> None:
    """Apply Version7-equivalent verified-grasp retry semantics to ICP picks."""
    pa.execute_grasp = True
    pa.confirm_real_grasp = True
    weight_config = WeightCheckConfig(
        host=args.force_sensor_host,
        samples=args.weight_check_samples,
        timeout_s=args.weight_check_timeout_s,
        settle_s=args.weight_check_settle_s,
        empty_max_g=args.weight_empty_max_g,
        single_min_g=args.weight_single_min_g,
        single_max_g=args.weight_single_max_g,
        multiple_min_g=args.weight_multiple_min_g,
        max_axis_mad_n=args.weight_max_axis_mad_n,
    )
    for weight_retry in range(int(args.max_weight_retries) + 1):
        if weight_retry:
            print(
                "-> fresh traditional ICP cycle after grasp verification failure "
                f"{weight_retry}/{args.max_weight_retries}; returning to D405 observation",
                flush=True,
            )
            subprocess.run(
                [sys.executable, str(TOOL / "go_graspnet.py"), "--arm-host", args.arm_host],
                input="y\n", text=True, check=True,
            )
        for planner_retry in range(int(args.max_planner_unreachable_retries) + 1):
            try:
                run, chosen = prepare_traditional_candidate(args, pa, attempt_index=weight_retry)
                break
            except TraditionalPlannerUnreachable as exc:
                if planner_retry >= int(args.max_planner_unreachable_retries):
                    raise RuntimeError(
                        "traditional ICP Planner-unreachable retries exhausted: " + str(exc)
                    ) from exc
                print(
                    "-> traditional ICP candidates are Planner-unreachable; "
                    f"fresh capture/ICP retry {planner_retry + 1}/"
                    f"{args.max_planner_unreachable_retries}",
                    flush=True,
                )
        else:
            raise RuntimeError("traditional ICP internal error: planner retry loop exhausted")
        if not args.execute:
            print("-> plan-only: add --execute --confirm-real-grasp only after reviewing artifacts")
            return
        try:
            wait_for_execution_permit()
            handoff = execute_paired_plan(pa, chosen, weight_config)
        except (WeightQualityRejected, pe.RetryGrasp) as exc:
            grasp_class = getattr(exc, "grasp_class", "EMPTY")
            retry_record = {
                "weight_retry_index": weight_retry,
                "classification": str(grasp_class),
                "reason": str(exc),
                "next_step": (
                    "fresh_observation_icp_cycle"
                    if weight_retry < int(args.max_weight_retries)
                    else "stop"
                ),
            }
            (run / "weight_check_result.json").write_text(
                json.dumps(retry_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if weight_retry >= int(args.max_weight_retries):
                raise RuntimeError(
                    "traditional ICP grasp verification retries exhausted: "
                    + json.dumps(retry_record, ensure_ascii=False)
                ) from exc
            print(
                f"-> ICP grasp verification rejected {grasp_class}; releasing/re-observing for a fresh retry",
                flush=True,
            )
            continue
        handoff_path = args.weight_handoff_file or run / "weight_handoff.json"
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> verified ICP graspup handoff: {handoff_path}")
        return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    weight_defaults = load_shared_weight_defaults()
    p.add_argument("--d405-serial", default="260322270242")
    p.add_argument("--arm-host", default="192.168.88.1")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument(
        "--camera-broker-socket",
        default=os.environ.get("CAMERA_BROKER_SOCKET", "/tmp/graspnet-v7-camera.sock"),
        help="singleton D405/Femto Camera Broker Unix socket",
    )
    p.add_argument("--camera-broker-timeout-s", type=float, default=30.0)
    p.add_argument("--output-dir", type=Path, default=HERE / "output")
    p.add_argument("--sam-socket", default=str(TOOL / "sam2-service/runtime/sam2-service.sock"))
    p.add_argument("--min-single-stl-observed-coverage", type=float, default=.65,
                   help="minimum SAM depth coverage explained by one STL (default: 0.65)")
    p.add_argument("--max-observed-to-stl-extent-ratio", type=float, default=1.30,
                   help="reject merged SAM masks exceeding one STL extent by this ratio (default: 1.30)")
    p.add_argument("--min-grasp-angle-to-table-deg", type=float, default=65.0,
                   help="require grasp approach axis angle to the table plane to exceed this value (default: 65)")
    p.add_argument("--front-axis-center-min-icp-fitness", type=float, default=.80,
                   help="front axis-centre is treated as occluded below this ICP fitness and uses rear fallback (default: 0.80)")
    p.add_argument("--min-jaw-side-support-m", type=float, default=.008,
                   help="require STL material on both local -Y/+Y sides of the main X axis (default: 0.008)")
    p.add_argument("--jaw-contact-half-span-m", type=float, default=.012,
                   help="STL-local -Y/+Y contact samples used to prove fingers lie on both sides of X (default: 0.012)")
    p.add_argument("--min-other-instance-gripper-clearance-m", type=float, default=.006,
                   help="minimum depth-cloud clearance from another SAM instance to the jaw keepout (default: 0.006)")
    p.add_argument("--allow-other-instance-below-grasp-m", type=float, default=.003,
                   help="only another instance higher than the grasp surface by this guard enters sweep checking (default: 0.003)")
    p.add_argument("--max-auto-grasp-up-m", type=float, default=.03,
                   help="maximum automatic +Z lift to clear lower stacked points (default: 0.03; 0 disables)")
    p.add_argument("--front-grasp-upward-offset-m", type=float, default=0.015,
                   help="final front-side closure offset in +Z meters (default: 诉讼0; closes at grasp surface)")
    p.add_argument("--back-grasp-upward-offset-m", type=float, default=0.0,
                   help="final back-side closure offset in +Z meters (default: 0; closes at grasp surface)")
    p.add_argument("--grasp-downward-probe-m", type=float, default=None,
                   help="optional V7 signed global override: positive=down, negative=up")
    p.add_argument("--skip-observation-move", action="store_true")
    p.add_argument("--execute", action="store_true", help="permit physical execution only with --confirm-real-grasp")
    p.add_argument("--confirm-real-grasp", action="store_true")
    p.add_argument("--weight-handoff-file", type=Path)
    p.add_argument("--force-sensor-host", default="127.0.0.1")
    p.add_argument("--weight-check-samples", type=int, default=weight_defaults["weight_check_samples"])
    p.add_argument("--weight-check-timeout-s", type=float, default=weight_defaults["weight_check_timeout_s"])
    p.add_argument("--weight-check-settle-s", type=float, default=weight_defaults["weight_check_settle_s"])
    p.add_argument("--weight-empty-max-g", type=float, default=weight_defaults["weight_empty_max_g"])
    p.add_argument("--weight-single-min-g", type=float, default=weight_defaults["weight_single_min_g"])
    p.add_argument("--weight-single-max-g", type=float, default=weight_defaults["weight_single_max_g"])
    p.add_argument("--weight-multiple-min-g", type=float, default=weight_defaults["weight_multiple_min_g"])
    p.add_argument("--weight-max-axis-mad-n", type=float, default=weight_defaults["weight_max_axis_mad_n"])
    p.add_argument(
        "--max-weight-retries", type=int, default=3,
        help="fresh traditional observation/ICP cycles after EMPTY/MULTIPLE/UNKNOWN/gripper-hold failure",
    )
    p.add_argument(
        "--max-no-grasp-retries", type=int, default=3,
        help="fresh traditional observations after SAM/STL/ICP yields no executable candidate",
    )
    p.add_argument(
        "--max-planner-unreachable-retries", type=int, default=30,
        help="fresh traditional observations after every candidate is Planner-unreachable",
    )
    args = p.parse_args()
    if args.frames < 1: p.error("--frames must be positive")
    if not np.isfinite(args.min_single_stl_observed_coverage) or not 0.0 <= args.min_single_stl_observed_coverage <= 1.0:
        p.error("--min-single-stl-observed-coverage must be finite and in [0, 1]")
    if (not np.isfinite(args.max_observed_to_stl_extent_ratio)
            or args.max_observed_to_stl_extent_ratio < 1.0):
        p.error("--max-observed-to-stl-extent-ratio must be finite and >= 1")
    offsets = (args.front_grasp_upward_offset_m, args.back_grasp_upward_offset_m)
    if not all(np.isfinite(value) and -.05 <= value <= .05 for value in offsets):
        p.error("face-specific upward offsets must be finite and in [-0.05, 0.05]")
    if args.grasp_downward_probe_m is not None and (
            not np.isfinite(args.grasp_downward_probe_m) or not -.05 <= args.grasp_downward_probe_m <= .05):
        p.error("--grasp-downward-probe-m must be finite and in [-0.05, 0.05]")
    if not np.isfinite(args.min_grasp_angle_to_table_deg) or not 0.0 <= args.min_grasp_angle_to_table_deg < 90.0:
        p.error("--min-grasp-angle-to-table-deg must be finite and in [0, 90)")
    if not np.isfinite(args.front_axis_center_min_icp_fitness) or not 0.0 <= args.front_axis_center_min_icp_fitness <= 1.0:
        p.error("--front-axis-center-min-icp-fitness must be finite and in [0, 1]")
    if not np.isfinite(args.min_jaw_side_support_m) or args.min_jaw_side_support_m <= 0.0:
        p.error("--min-jaw-side-support-m must be finite and positive")
    if not np.isfinite(args.jaw_contact_half_span_m) or args.jaw_contact_half_span_m <= 0.0:
        p.error("--jaw-contact-half-span-m must be finite and positive")
    if (not np.isfinite(args.min_other_instance_gripper_clearance_m)
            or args.min_other_instance_gripper_clearance_m < 0.0):
        p.error("--min-other-instance-gripper-clearance-m must be finite and nonnegative")
    if (not np.isfinite(args.allow_other_instance_below_grasp_m)
            or args.allow_other_instance_below_grasp_m < 0.0):
        p.error("--allow-other-instance-below-grasp-m must be finite and nonnegative")
    if not np.isfinite(args.max_auto_grasp_up_m) or not 0.0 <= args.max_auto_grasp_up_m <= .05:
        p.error("--max-auto-grasp-up-m must be finite and in [0, 0.05]")
    if args.execute != args.confirm_real_grasp:
        p.error("physical motion requires both --execute and --confirm-real-grasp")
    if args.weight_check_samples < 3 or args.weight_check_timeout_s <= 0 or args.weight_check_settle_s < 0:
        p.error("invalid weight-check sampling settings")
    if args.max_weight_retries < 0:
        p.error("--max-weight-retries must be non-negative")
    if args.max_no_grasp_retries < 0:
        p.error("--max-no-grasp-retries must be non-negative")
    if args.max_planner_unreachable_retries < 0:
        p.error("--max-planner-unreachable-retries must be non-negative")
    weight_threshold_values = (
        args.weight_empty_max_g,
        args.weight_single_min_g,
        args.weight_single_max_g,
        args.weight_multiple_min_g,
        args.weight_max_axis_mad_n,
    )
    if (
        not all(np.isfinite(value) and value > 0 for value in weight_threshold_values)
        or not (
            args.weight_empty_max_g < args.weight_single_min_g
            <= args.weight_single_max_g < args.weight_multiple_min_g
        )
    ):
        p.error("invalid weight-check thresholds")
    pa = planner_args()
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe
    if args.execute:
        # Propagate the policy to Planner helper calls as well as the explicit
        # ``input='y\\n'`` used by the observation-pose helper subprocesses.
        os.environ["LOOP_AUTO_CONFIRM_MOTION"] = "1"
        os.environ["LOOP_AUTO_CONFIRM_GRASPNET_EXECUTION"] = "1"
        # The operator explicitly authorized a real pick.  Open once before
        # moving to the observation pose, rather than unexpectedly opening
        # halfway through the subsequent arm motion.  Plan-only stays fully
        # non-actuating.
        pe.move_gripper(pa, 100.0, "open gripper at real-pick start")
    if not args.skip_observation_move:
        # go_graspnet.py intentionally asks once before executing its
        # RRTStar move to the fixed D405 observation pose.  In this one-shot
        # production workflow that transition is requested by launching the
        # command, so feed its only terminal confirmation automatically.
        print("-> 自动确认：移动到传统 ICP D405 观察位。", flush=True)
        subprocess.run(
            [sys.executable, str(TOOL / "go_graspnet.py"), "--arm-host", args.arm_host],
            input="y\n",
            text=True,
            check=True,
        )
        publish_observation_ready()
    elif os.environ.get("LOOP_GRASPNET_OBSERVATION_READY_FILE"):
        raise RuntimeError(
            "traditional ICP loop mode cannot use --skip-observation-move: "
            "the scheduler needs a verified observation-ready marker"
        )
    run_traditional_cycles(args, pa, pe)
    return

if __name__ == "__main__":
    main()
