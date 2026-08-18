#!/usr/bin/env python3
"""Module 1：D405 → SAM2 → STL/ICP → Planner 的左臂抓取入口。

默认只采集、生成候选并做 Planner 可达性预检；只有同时提供 ``--execute``
和 ``--confirm-real-grasp`` 才允许真实运动。真实抓取采用距目标 10 cm 的垂直
预抓取位和 LIN 下探，抓取后必须通过夹爪状态与力传感器单工件验证。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
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
import icp_pipeline


LOOP = Path(__file__).resolve().parents[2]
TOOL = LOOP / "tools"
GRASPNET_ROOT = TOOL / "graspnet_runtime"
HERE = Path(__file__).resolve().parent
WEIGHT_CONFIG_PATH = LOOP / "config/grasp_weight_thresholds.conf"
DEFAULT_BRIDGE_WRIST_CALIBRATION = (
    TOOL / "calibration/left_wrist_d405/calibration_wrist1.json"
)
DEFAULT_PLANNER_BRIDGE_CALIBRATION = (
    TOOL
    / "calibration/graspnet_planner_bridge_5604_20260803/"
    "planner_tf_bridge_5604_candidate.json"
)
DEFAULT_DIRECT_WRIST_CALIBRATION = (
    TOOL
    / "calibration/wrist1_planner_final_v3/calibration_wrist1_planner.json"
)

MODULE1_MODES = {
    "gzhu": {
        "d405_serial": "260322270242",
        "handeye_mode": "bridge",
        "robot_host": "192.168.88.1",
        "gripper_host": "127.0.0.1",
        "force_sensor_host": "127.0.0.1",
    },
    "xiangxin": {
        "d405_serial": "260322270600",
        "handeye_mode": "planner-direct",
        "robot_host": "192.168.3.11",
        "gripper_host": "192.168.3.11",
        "force_sensor_host": "192.168.3.22",
    },
}

# URDF 中 finger1/finger2 分别是物理左/右手指。gripper_base 固连到
# left_eef，且 +X_gripper == -Y_eef，因此下面的轴表示“物理右指→左指”。
# 该方向只用于正面抓取的整夹爪平移补偿。
FRONT_PHYSICAL_RIGHT_TO_LEFT_JAW_OFFSET_M = 0.00
PHYSICAL_RIGHT_TO_LEFT_JAW_AXIS_EEF = np.array((0.0, -1.0, 0.0))
# 反面公共偏置使用工件 CAD 局部坐标表达，而不是夹爪坐标：整夹爪沿工件 -X
# 平移 1.5 mm。
BACK_OBJECT_NEGATIVE_X_OFFSET_M = 0.0015


class TraditionalPlannerUnreachable(RuntimeError):
    """一次新观测产生的所有传统视觉候选都没有安全 Planner 路径。"""


def wait_for_execution_permit() -> None:
    """如配置许可文件，则在真实预抓取运动前等待外部控制器放行。"""
    value = os.environ.get("MODULE1_EXECUTION_PERMIT_FILE")
    if not value:
        return
    permit_path = Path(value)
    print(
        "-> 传统 ICP 候选和预抓取规划已完成；等待外部许可真实抓取："
        f"{permit_path}",
        flush=True,
    )
    while True:
        if permit_path.is_file():
            content = permit_path.read_text(encoding="utf-8").strip().lower()
            if content == "allow":
                print("-> 已获得外部许可；开始执行传统 ICP 预抓取轨迹。", flush=True)
                return
            raise RuntimeError(
                f"invalid traditional ICP execution permit content in {permit_path}: {content!r}"
            )
        time.sleep(0.2)


def load_shared_weight_defaults() -> dict[str, float | int]:
    """读取项目统一的重量阈值，保证抓取判定与已验证配置完全一致。"""
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
    """从 Arm SDK 状态中提取左臂末端 4×4 位姿。"""
    left = ((state.get("state") or {}).get("left") or {})
    pose = left.get("eef_pose") or {}
    xyz, rpy = pose.get("xyz"), pose.get("rpy")
    if not (
        isinstance(xyz, list)
        and isinstance(rpy, list)
        and len(xyz) == len(rpy) == 3
    ):
        raise RuntimeError(
            "parm_sdk Arm.get_state() does not contain left eef_pose xyz/rpy"
        )
    values = np.asarray([*xyz, *rpy], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("left eef_pose has non-finite values")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def planner_body_base_to_left_arm_base() -> tuple[np.ndarray, dict]:
    """读取 Planner 的 body_base→left_arm_base，并规范网关返回方向。"""
    sys.path.insert(0, str(GRASPNET_ROOT))
    from parm_sdk import Planner

    planner = Planner(host="127.0.0.1", trajectory_port=5604, tf_port=5609, waypoint_port=5610)
    try:
        response = planner.lookup_transform("left_arm_base", "body_base")
        reverse_response = planner.lookup_transform("body_base", "left_arm_base")
    finally:
        planner.close()
    data = (response.get("data") or {})
    reverse_data = (reverse_response.get("data") or {})
    if data.get("source_frame") != "left_arm_base" or data.get("target_frame") != "body_base":
        raise RuntimeError(f"unexpected Planner TF response for left_arm_base -> body_base: {data}")
    if reverse_data.get("source_frame") != "body_base" or reverse_data.get("target_frame") != "left_arm_base":
        raise RuntimeError(f"unexpected Planner TF response for body_base -> left_arm_base: {reverse_data}")
    translation, rotation = data.get("translation") or {}, data.get("rotation") or {}
    raw = np.eye(4)
    try:
        raw[:3, :3] = Rotation.from_quat([rotation[key] for key in "xyzw"]).as_matrix()
        raw[:3, 3] = [translation[key] for key in "xyz"]
        reverse_translation, reverse_rotation = reverse_data.get("translation") or {}, reverse_data.get("rotation") or {}
        reverse_raw = np.eye(4)
        reverse_raw[:3, :3] = Rotation.from_quat([reverse_rotation[key] for key in "xyzw"]).as_matrix()
        reverse_raw[:3, 3] = [reverse_translation[key] for key in "xyz"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Planner TF payload: {data}") from exc
    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(reverse_raw))):
        raise RuntimeError("Planner returned non-finite left-arm TF values")
    # The deployed TF gateway serializes the inverse of the requested
    # transform; this normalization is independently checked in calibration.
    T_body_base_left_arm_base = np.linalg.inv(raw)
    T_left_arm_base_body_base = np.linalg.inv(reverse_raw)
    closure = T_body_base_left_arm_base @ T_left_arm_base_body_base
    closure_mm, closure_deg = pose_delta(np.eye(4), closure)
    if closure_mm > 0.5 or closure_deg > 0.25:
        raise RuntimeError(
            "Planner body_base/left-arm-base TF directions are not reciprocal: "
            f"closure={closure_mm:.3f}mm/{closure_deg:.3f}deg"
        )
    return T_body_base_left_arm_base, {
        "forward_timestamp": data.get("timestamp"),
        "reverse_timestamp": reverse_data.get("timestamp"),
        "forward_reverse_closure_mm": closure_mm,
        "forward_reverse_closure_deg": closure_deg,
    }


def planner_body_base_to_left_end() -> tuple[np.ndarray, dict]:
    """读取 planner-direct 标定所需的 body_base→左臂末端变换。"""
    sys.path.insert(0, str(GRASPNET_ROOT))
    from parm_sdk import Planner

    planner = Planner(host="127.0.0.1", trajectory_port=5604, tf_port=5609, waypoint_port=5610)
    try:
        response = planner.lookup_transform("left_arm_end_link", "body_base")
        reverse_response = planner.lookup_transform("body_base", "left_arm_end_link")
    finally:
        planner.close()
    data = response.get("data") or {}
    reverse_data = reverse_response.get("data") or {}
    if data.get("source_frame") != "left_arm_end_link" or data.get("target_frame") != "body_base":
        raise RuntimeError(f"unexpected Planner TF response for left_arm_end_link -> body_base: {data}")
    if reverse_data.get("source_frame") != "body_base" or reverse_data.get("target_frame") != "left_arm_end_link":
        raise RuntimeError(f"unexpected Planner TF response for body_base -> left_arm_end_link: {reverse_data}")

    def response_matrix(payload: dict) -> np.ndarray:
        """把 Planner TF 响应转换为经过有限值校验的齐次矩阵。"""
        translation, rotation = payload.get("translation") or {}, payload.get("rotation") or {}
        matrix = np.eye(4)
        try:
            matrix[:3, :3] = Rotation.from_quat([rotation[key] for key in "xyzw"]).as_matrix()
            matrix[:3, 3] = [translation[key] for key in "xyz"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid Planner TF payload: {payload}") from exc
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("Planner returned non-finite left-arm TF values")
        return matrix

    # The deployed gateway returns the inverse of the requested transform.
    T_body_base_left_end = np.linalg.inv(response_matrix(data))
    T_left_end_body_base = np.linalg.inv(response_matrix(reverse_data))
    closure_mm, closure_deg = pose_delta(
        np.eye(4), T_body_base_left_end @ T_left_end_body_base
    )
    if closure_mm > 0.5 or closure_deg > 0.25:
        raise RuntimeError(
            "Planner body_base/end-link TF directions are not reciprocal: "
            f"closure={closure_mm:.3f}mm/{closure_deg:.3f}deg"
        )
    return T_body_base_left_end, {
        "direct_timestamp": data.get("timestamp"),
        "reverse_timestamp": reverse_data.get("timestamp"),
        "forward_reverse_closure_mm": closure_mm,
        "forward_reverse_closure_deg": closure_deg,
    }


def planner_timestamp_ns(tf_record: dict) -> int:
    """把 Planner TF 时间戳统一转换为纳秒整数。"""
    stamp = tf_record.get("direct_timestamp")
    if not isinstance(stamp, dict):
        raise RuntimeError("Planner TF response has no dynamic timestamp")
    try:
        return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Planner TF timestamp: {stamp}") from exc


def pose_delta(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """计算两次相机采集间的末端平移差和旋转差。"""
    return (
        float(np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000.0),
        float((Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).magnitude()) * 180.0 / np.pi),
    )


def capture_log(run: Path, args: argparse.Namespace) -> Path:
    """通过 Camera Broker 采集稳定 RGB-D，并记录所选手眼标定链。"""
    sys.path.insert(0, str(GRASPNET_ROOT / "deploy" / "version7"))
    from camera_client import capture as broker_capture
    from camera_client import status as broker_status

    raw = run / "raw/d405"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise RuntimeError(f"traditional raw capture directory already exists: {raw}")

    status = broker_status(args.camera_broker_socket, timeout_s=args.camera_broker_timeout_s)
    if status.get("schema_version") != 1:
        raise RuntimeError("Camera Broker ABI mismatch; restart it with the current project version")
    broker_serial = str(
        (((status.get("camera_config") or {}).get("d405") or {}).get("serial") or "")
    )
    if broker_serial != str(args.d405_serial):
        raise RuntimeError(
            f"Camera Broker D405 serial is {broker_serial!r}; expected {args.d405_serial!r}"
        )
    allow_root = Path(status["allow_root"]).resolve()
    broker_capture_dir: Path | None = None
    result: dict | None = None
    capture_metadata: dict = {}

    def capture_once() -> tuple[Path, dict]:
        """通过 Camera Broker 采集一组 D405 RGB-D 帧及元数据。"""
        capture_dir = (
            allow_root / "traditional_icp" /
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex}"
        )
        capture_result = broker_capture(
            args.camera_broker_socket,
            {
                "camera": "d405",
                "output_dir": str(capture_dir),
                "frame_count": int(args.frames),
                "require_eef_pose": False,
            },
            timeout_s=args.camera_broker_timeout_s,
        )
        return capture_dir, capture_result

    if args.handeye_mode == "bridge":
        sys.path.insert(0, str(TOOL))
        from parm_sdk import Arm

        T_body_base_left_arm_base, base_alignment = planner_body_base_to_left_arm_base()
        with Arm(host=args.arm_host) as arm:
            for attempt in range(1, int(args.capture_stability_retries) + 1):
                T_before = eef_transform(arm.get_state() or {})
                broker_capture_dir, result = capture_once()
                T_after = eef_transform(arm.get_state() or {})
                delta_mm, delta_deg = pose_delta(T_before, T_after)
                if delta_mm <= 0.5 and delta_deg <= 0.25:
                    break
                shutil.rmtree(broker_capture_dir, ignore_errors=True)
                if attempt == int(args.capture_stability_retries):
                    raise RuntimeError(
                        "SDK left eef_pose remained in motion across "
                        f"{attempt} D405 captures: {delta_mm:.2f}mm/{delta_deg:.2f}deg"
                    )
                print(
                    "-> D405 SDK eef_pose static check retry "
                    f"{attempt}/{int(args.capture_stability_retries)}: "
                    f"{delta_mm:.2f}mm/{delta_deg:.2f}deg; waiting "
                    f"{args.capture_stability_retry_wait_s:.2f}s",
                    flush=True,
                )
                time.sleep(args.capture_stability_retry_wait_s)
        capture_metadata = {
            "T_base_eef": T_after.tolist(),
            "T_body_base_left_arm_base": T_body_base_left_arm_base.tolist(),
            "planner_base_alignment": base_alignment,
            "sdk_capture_stability": {
                "translation_delta_mm": delta_mm,
                "rotation_delta_deg": delta_deg,
            },
            "robot_pose_source": (
                "parm_sdk Arm.get_state().state.left.eef_pose + accepted Planner TF bridge"
            ),
        }
    else:
        for attempt in range(1, int(args.capture_stability_retries) + 1):
            T_before, tf_before = planner_body_base_to_left_end()
            broker_capture_dir, result = capture_once()
            T_after, tf_after = planner_body_base_to_left_end()
            delta_mm, delta_deg = pose_delta(T_before, T_after)
            timestamp_before_ns = planner_timestamp_ns(tf_before)
            timestamp_after_ns = planner_timestamp_ns(tf_after)
            if timestamp_after_ns <= timestamp_before_ns:
                shutil.rmtree(broker_capture_dir, ignore_errors=True)
                raise RuntimeError(
                    "Planner TF timestamp did not advance during D405 capture; "
                    "repair the Planner joint-state TF publisher before recapturing"
                )
            if delta_mm <= 0.5 and delta_deg <= 0.25:
                break
            shutil.rmtree(broker_capture_dir, ignore_errors=True)
            if attempt == int(args.capture_stability_retries):
                raise RuntimeError(
                    "Planner left_arm_end_link remained in motion across "
                    f"{attempt} D405 captures: {delta_mm:.2f}mm/{delta_deg:.2f}deg"
                )
            print(
                "-> D405 Planner end-link static check retry "
                f"{attempt}/{int(args.capture_stability_retries)}: "
                f"{delta_mm:.2f}mm/{delta_deg:.2f}deg; waiting "
                f"{args.capture_stability_retry_wait_s:.2f}s",
                flush=True,
            )
            time.sleep(args.capture_stability_retry_wait_s)
        capture_metadata = {
            "T_body_base_left_arm_end_link": T_after.tolist(),
            "planner_tf_before": tf_before,
            "planner_tf_after": tf_after,
            "planner_tf_timestamp_before_ns": timestamp_before_ns,
            "planner_tf_timestamp_after_ns": timestamp_after_ns,
            "planner_capture_stability": {
                "translation_delta_mm": delta_mm,
                "rotation_delta_deg": delta_deg,
            },
            "robot_pose_source": (
                "direct Planner lookup_transform(left_arm_end_link, body_base), inverse-normalized"
            ),
        }

    if broker_capture_dir is None or result is None:
        raise AssertionError("D405 capture did not return a result")
    try:
        shutil.copytree(broker_capture_dir, raw)
    finally:
        # Broker workspaces are temporary; preserve only the copy
        # inside this traditional run after the Broker has finished writing.
        shutil.rmtree(broker_capture_dir, ignore_errors=True)

    meta_path = raw / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    captured_serial = str(meta.get("serial", ""))
    if captured_serial != broker_serial:
        raise RuntimeError(
            f"captured D405 serial is {captured_serial!r}; Broker reported {broker_serial!r}"
        )
    meta.update({
        "camera_name": "d405",
        "serial": broker_serial,
        "handeye_mode": args.handeye_mode,
        "handeye_mode_requested": args.handeye_mode_requested,
        "bridge_wrist_calibration": str(args.bridge_wrist_calibration.resolve()),
        "planner_bridge_calibration": str(args.planner_bridge_calibration.resolve()),
        "direct_wrist_calibration": str(args.direct_wrist_calibration.resolve()),
        "frame_count": int(args.frames),
        "camera_broker_socket": args.camera_broker_socket,
        "camera_broker_capture_elapsed_s": result.get("capture_elapsed_s"),
        **capture_metadata,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        "-> traditional D405 capture received from Camera Broker: "
        f"handeye={args.handeye_mode}, frames={args.frames}, "
        f"elapsed={float(result['capture_elapsed_s']):.3f}s",
        flush=True,
    )
    return run


def traditional_perception(run: Path, args: argparse.Namespace) -> dict:
    """运行 SAM2、STL/ICP 和抓取安全筛选，返回结构化候选报告。"""
    offline_args = [
        "--log", str(run), "--output-dir", str(run / "traditional"),
        "--handeye-mode", args.handeye_mode,
        "--bridge-wrist-calibration", str(args.bridge_wrist_calibration),
        "--planner-bridge-calibration", str(args.planner_bridge_calibration),
        "--direct-wrist-calibration", str(args.direct_wrist_calibration),
        "--sam-socket", args.sam_socket,
        "--icp-workers", str(args.icp_workers),
        "--icp-daemon-timeout-s", str(args.icp_daemon_timeout_s),
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
        ]),
    ]
    runtime_log = os.environ.get("PARM_LOOP_RUNTIME_LOG_FILE")
    if runtime_log:
        # The modular loop already owns this append-only run log.  Keep the
        # verbose ICP diagnostics there rather than flooding its terminal.
        with Path(runtime_log).open("a", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                icp_pipeline.run(offline_args, keep_icp_workers_alive=True)
    else:
        icp_pipeline.run(offline_args, keep_icp_workers_alive=True)
    report_path = run / "traditional/traditional_pose_report.json"
    if not report_path.is_file():
        raise RuntimeError("traditional perception did not produce a pose report")
    report = json.loads(report_path.read_text())
    if runtime_log:
        timing_s = report.get("timing_s", {})
        print(
            "-> ICP匹配耗时: "
            f"{float(timing_s.get('icp_parallel_wall', float('nan'))):.3f}s",
            flush=True,
        )
    return report


def ranked_planner_records(report: dict) -> list[dict]:
    """按视觉语义优先级返回可送入 Planner 的候选记录。"""
    return sorted(
        (record for record in report.get("records", []) if record.get("accepted") and record.get("planner_candidate_rank")),
        key=lambda record: int(record["planner_candidate_rank"]),
    )


def planner_args(args: argparse.Namespace) -> argparse.Namespace:
    """把 Module 1 参数转换为精简 Planner 执行工具所需命名空间。"""
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe
    planner = pe.build_parser().parse_args([])
    planner.planner_host = "127.0.0.1"; planner.planner_trajectory_port = 5604
    planner.planner_tf_port = 5609; planner.planner_waypoint_port = 5610
    planner.planner_execute_url = "http://127.0.0.1:5667/middle/waypoint/execute"
    planner.planner_bridge_calibration = str(args.planner_bridge_calibration)
    # The hand-eye chain is converted through the accepted bridge before ICP
    # emits body_base poses, so the execution helper consumes those poses
    # directly and must not apply the bridge a second time.
    planner.planner_base_frame = "body_base"; planner.planner_tool_frame = "left_eef"
    planner.planner_direct_grasp_center = True; planner.planner_direct_pose = True
    planner.planner_group = "left_arm"; planner.arm_id = 1; planner.arm_host = args.arm_host
    # gzhu runs the arm driver on its controller but exposes the gripper from
    # the workstation.  xiangxin keeps both services on the robot host.
    planner.gripper_host = args.gripper_host
    planner.gripper_hand = 1
    # Keep Module 1 at the requested 0.5 velocity. Pilz rejected recent LIN
    # paths at higher acceleration by a small joint3 limit overshoot (2.441 > 2.4),
    # so use a conservative acceleration scale without slowing cruise speed.
    planner.planner_max_vel = .5; planner.planner_max_acc = .18
    planner.planner_pregrasp_mode = "rrtstar"; planner.planner_descend_mode = "lin"
    planner.grasp_weight_check = False; planner.stop_after_graspup = True
    return planner


def parallel_jaw_variants(plan: dict) -> list[tuple[float, dict]]:
    """生成平行夹爪 0°/180° 等价姿态，并同步应用正反面物理偏置。"""
    original = copy.deepcopy(plan)
    R0 = np.asarray(original["eef_rotation_base"], dtype=float)
    nominal_pre_tcp = np.asarray(original["pregrasp_grasp_tcp_base_m"], dtype=float)
    # Preserve the Version7 final-Z probe for the 180-degree symmetry branch.
    nominal_grasp_tcp = np.asarray(
        original.get("grasp_tcp_base_m", original["grasp_center_base_m"]), dtype=float
    )
    variants = []
    for roll_deg, R in ((0.0, R0), (180.0, R0 @ Rotation.from_euler("x", 180, degrees=True).as_matrix())):
        item = copy.deepcopy(original)
        jaw_offset_base = np.zeros(3, dtype=float)
        if item.get("face_side") == "front":
            # This must be computed after the symmetry roll: 0/180 degrees
            # swaps the physical fingers with respect to the object, while
            # R @ axis always remains physical right -> physical left.
            jaw_offset_base = R @ (
                PHYSICAL_RIGHT_TO_LEFT_JAW_AXIS_EEF
                * FRONT_PHYSICAL_RIGHT_TO_LEFT_JAW_OFFSET_M
            )
            item["front_physical_right_to_left_jaw_offset_m"] = (
                FRONT_PHYSICAL_RIGHT_TO_LEFT_JAW_OFFSET_M
            )
            # The deployment default is deliberately zero.  Record the
            # physical direction even in that case; dividing the zero vector
            # by the zero scalar used to write NaNs into plan diagnostics.
            item["front_physical_right_to_left_jaw_axis_base"] = (
                R @ PHYSICAL_RIGHT_TO_LEFT_JAW_AXIS_EEF
            ).tolist()
            item["front_physical_right_to_left_jaw_offset_base_m"] = jaw_offset_base.tolist()
        elif item.get("face_side") == "back":
            # This is intentionally independent of the 0/180-degree jaw
            # symmetry.  New plans persist the object axis explicitly; the
            # R0 fallback matches the back grasp construction (EEF +Z=-Xobj)
            # and keeps old recorded plans replayable.
            negative_x_axis_base = np.asarray(
                item.get("object_negative_x_axis_base", R0[:, 2]), dtype=float
            )
            if negative_x_axis_base.shape != (3,) or not np.all(np.isfinite(negative_x_axis_base)):
                raise RuntimeError("back plan has invalid object_negative_x_axis_base")
            norm = float(np.linalg.norm(negative_x_axis_base))
            if norm < 1e-8:
                raise RuntimeError("back plan has zero object_negative_x_axis_base")
            negative_x_axis_base /= norm
            jaw_offset_base = negative_x_axis_base * BACK_OBJECT_NEGATIVE_X_OFFSET_M
            item["back_object_negative_x_offset_m"] = BACK_OBJECT_NEGATIVE_X_OFFSET_M
            item["back_object_negative_x_axis_base"] = negative_x_axis_base.tolist()
            item["back_object_negative_x_offset_base_m"] = jaw_offset_base.tolist()
        pre_tcp = nominal_pre_tcp + jaw_offset_base
        grasp_tcp = nominal_grasp_tcp + jaw_offset_base
        # Keep all persisted centre/TCP fields congruent with the actual
        # target, so the plan and its visual/debug artifacts describe the
        # same whole-gripper translation.
        for key in (
            "grasp_center_base_m", "predicted_grasp_tcp_base_m",
            "compensated_grasp_center_base_m", "grasp_tcp_base_m",
        ):
            if key in item:
                item[key] = (np.asarray(original[key], dtype=float) + jaw_offset_base).tolist()
        item["pregrasp_grasp_tcp_base_m"] = pre_tcp.tolist()
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


class PairedTrajectoryUnavailable(RuntimeError):
    """预抓取与垂直 LIN 下探无法组成同一 IK 分支上的成对轨迹。"""


def paired_pregrasp_and_descent(pa: argparse.Namespace, candidate: dict, *, attempts: int = 3):
    """为一个候选规划同分支的预抓取 RRTStar 与 10 cm LIN 下探。"""
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
    raise PairedTrajectoryUnavailable(
        f"candidate rank {candidate['rank']} has no branch-consistent pregrasp/LIN pair after {attempts} attempts: "
        + " | ".join(failures)
    )


def paired_preflight(
    pa: argparse.Namespace, candidates: list[dict], *, max_inflight: int,
) -> tuple[list[dict], list[dict]]:
    """并发预检全部语义/夹爪对称候选，返回可执行项与完整诊断。"""
    if not candidates:
        return [], []
    records_by_rank: dict[int, dict] = {}
    reachable_by_semantic_rank: dict[int, list[tuple[dict, dict]]] = {}
    worker_count = min(max_inflight, len(candidates))
    print(
        f"-> traditional Planner preflight: {len(candidates)} candidates, "
        f"up to {worker_count} concurrent requests (Module 3 default=8)",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="traditional-grasp-preflight",
    ) as executor:
        futures = {
            executor.submit(paired_pregrasp_and_descent, pa, candidate): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                pregrasp, descent, attempt, failures = future.result()
            except Exception as exc:
                records_by_rank[int(candidate["rank"])] = {
                    "rank": candidate["rank"], "source_grasp_index": candidate["source_grasp_index"],
                    "grasp_source": candidate["grasp_source"], "score": candidate["score"],
                    "parallel_jaw_symmetry_rotation_deg": candidate["parallel_jaw_symmetry_rotation_deg"],
                    "reachable": False, "reason": str(exc),
                }
                continue
            item = {
                "rank": candidate["rank"], "source_grasp_index": candidate["source_grasp_index"],
                "grasp_source": candidate["grasp_source"], "score": candidate["score"],
                "parallel_jaw_symmetry_rotation_deg": candidate["parallel_jaw_symmetry_rotation_deg"],
                "reachable": True, "paired_planning_attempt": attempt,
                "pregrasp_waypoints": len(pregrasp["trajectory"]),
                "lin_descent_waypoints": len(descent["trajectory"]),
                "failed_branch_attempts": len(failures),
            }
            records_by_rank[int(candidate["rank"])] = item
            reachable_by_semantic_rank.setdefault(
                int(candidate["semantic_candidate_rank"]), []
            ).append((candidate, item))

    reachable: list[dict] = []
    for semantic_rank in sorted(reachable_by_semantic_rank):
        pair_reachable = reachable_by_semantic_rank[semantic_rank]
        # Equivalent jaws: the shorter path is the nearer, less roundabout
        # choice from the arm's current state.
        chosen, chosen_record = min(
            pair_reachable,
            key=lambda item: (item[1]["pregrasp_waypoints"], item[1]["lin_descent_waypoints"]),
        )
        chosen_record["selected_nearer_parallel_jaw_symmetry"] = True
        reachable.append(chosen)
    return reachable, [records_by_rank[rank] for rank in sorted(records_by_rank)]


def select_reachable_candidate(
    reachable: list[dict], *, failed_grasp_retry_index: int,
) -> tuple[dict, str]:
    """首次按视觉得分选择；重试时在其余合法候选中随机选择。"""
    if not reachable:
        raise RuntimeError("cannot select from an empty reachable candidate set")
    if failed_grasp_retry_index == 0:
        return reachable[0], "highest_score_first_attempt"
    return secrets.choice(reachable), "random_legal_after_grasp_failure"


def reverse_lin_trajectory(trajectory: dict) -> dict:
    """精确反转已验证的 LIN 下探轨迹，用作同姿态抓取后抬升。"""
    lift = copy.deepcopy(trajectory)
    waypoints = list(lift.get("trajectory") or [])
    if len(waypoints) < 2:
        raise RuntimeError("cannot reverse a LIN descent with fewer than two waypoints")
    lift["trajectory"] = list(reversed(waypoints))
    return lift


def mark_planner_selected_grasp(run: Path, chosen: dict) -> None:
    """保存 Planner 最终采用的实例，并在抓取图上标出选择结果。"""
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
    """重新规划并执行一组成对轨迹，随后完成夹爪与重量验证。"""
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe

    pe.check_pregrasp_distance(pa, candidate["plan"])
    print("-> 执行成对预抓取与 10cm LIN 下探")
    pregrasp, descent, attempt, _ = paired_pregrasp_and_descent(pa, candidate)
    print(f"-> executing branch-consistent pair from planning attempt {attempt}")
    # Start fully open, then reduce only to the validated silhouette width
    # (two side margins plus the operator-requested extra 12 mm) before motion.
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
            # No uncertain grasp may produce a verified handoff. Release it while
            # stopped at the safe pregrasp height, then start a fresh cycle.
            pe.move_gripper(pa, 100.0, "release rejected ICP grasp")
            raise
        except Exception as exc:
            pe.move_gripper(pa, 100.0, "release after ICP weight data failure")
            raise WeightQualityRejected(
                "UNKNOWN", f"ICP same-pose weight check failed: {exc}"
            ) from exc
        # Keep the verified single workpiece clamped and return to graspup
        # before validating the physical hold state.
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
    print("-> ICP verified gripper hold and force signature at graspup; verified handoff is available")
    return handoff


def prepare_traditional_candidate(
    args: argparse.Namespace, pa: argparse.Namespace, *, attempt_index: int,
) -> tuple[Path, dict, list[dict]]:
    """完成一次新采集、SAM/STL/ICP 与 Planner 预检，返回候选集合。"""
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
            [sys.executable, str(observation_script), "--arm-host", args.arm_host, "--speed", "0.5"]
            if observation_script.name == "go_graspnet2.py"
            else [sys.executable, str(observation_script), "--arm-host", args.arm_host],
            check=True,
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
    reachable, records = paired_preflight(
        pa, candidates, max_inflight=int(args.planner_preflight_inflight),
    )
    (run / "planner_preflight.json").write_text(json.dumps(records, indent=2))
    if not reachable:
        raise TraditionalPlannerUnreachable(
            f"Planner rejected the traditional grasp; see {run / 'planner_preflight.json'}"
        )
    selected, selection_policy = select_reachable_candidate(
        reachable, failed_grasp_retry_index=attempt_index
    )
    (run / "selected_planner_plan.json").write_text(json.dumps(selected["plan"], indent=2))
    (run / "candidate_selection.json").write_text(
        json.dumps(
            {
                "policy": selection_policy,
                "failed_grasp_retry_index": attempt_index,
                "legal_candidate_count": len(reachable),
                "selected_planner_candidate_rank": selected["rank"],
                "selected_semantic_candidate_rank": selected["semantic_candidate_rank"],
                "selected_icp_score": selected["score"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    mark_planner_selected_grasp(run, selected)
    print(
        f"-> traditional plan rank {selected['rank']} selected by {selection_policy}; "
        f"Planner preflight succeeded: {run}"
    )
    selected_candidate = next(
        item for item in candidates if item["rank"] == selected["rank"]
    )
    # The initial preflight proves a candidate was possible at that instant,
    # but a fresh planner request may choose another IK branch. Preserve the
    # remaining legal candidates so that one fresh paired-LIN failure does not
    # discard this RGB-D/ICP observation.
    execution_candidates = [selected_candidate] + [
        candidate for candidate in reachable
        if candidate["rank"] != selected_candidate["rank"]
    ]
    return run, selected_candidate, execution_candidates


def run_traditional_cycles(
    args: argparse.Namespace, pa: argparse.Namespace, pe: object,
) -> None:
    """管理新观测、无候选、规划失败和重量失败的分层重试循环。"""
    # Keep Planner helper authorization aligned with the top-level safety
    # gate. Plan-only may preflight trajectories but must never be marked as
    # execution-authorized inside the helper namespace.
    pa.execute_grasp = bool(args.execute)
    pa.confirm_real_grasp = bool(args.confirm_real_grasp)
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
                check=True,
            )
        weight_failure: Exception | None = None
        handoff = None
        for planner_retry in range(int(args.max_planner_unreachable_retries) + 1):
            try:
                run, chosen, execution_candidates = prepare_traditional_candidate(
                    args, pa, attempt_index=weight_retry,
                )
                if not args.execute:
                    print("-> plan-only: add --execute --confirm-real-grasp only after reviewing artifacts")
                    return
                wait_for_execution_permit()
                pair_failures: list[str] = []
                for execution_index, candidate in enumerate(execution_candidates, 1):
                    try:
                        if candidate["rank"] != chosen["rank"]:
                            mark_planner_selected_grasp(run, candidate)
                            print(
                                "-> fresh paired planning failed for an earlier candidate; "
                                f"trying preflight-legal candidate rank {candidate['rank']} "
                                f"({execution_index}/{len(execution_candidates)})",
                                flush=True,
                            )
                        handoff = execute_paired_plan(pa, candidate, weight_config)
                        break
                    except PairedTrajectoryUnavailable as exc:
                        pair_failures.append(str(exc))
                        print(
                            "-> candidate rank "
                            f"{candidate['rank']} cannot form a fresh RRTStar/LIN pair; "
                            "trying the next preflight-legal candidate.",
                            flush=True,
                        )
                if handoff is None:
                    raise TraditionalPlannerUnreachable(
                        "all preflight-legal ICP candidates lost their fresh "
                        "RRTStar/LIN pair: " + " | ".join(pair_failures)
                    )
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
            except (WeightQualityRejected, pe.RetryGrasp) as exc:
                # A physical grasp/weight failure must preserve the existing
                # fresh-observation retry path rather than consume a planner
                # reachability retry.
                weight_failure = exc
                break
        else:
            raise RuntimeError("traditional ICP internal error: planner retry loop exhausted")
        if weight_failure is not None:
            exc = weight_failure
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
        if handoff is None:
            raise RuntimeError("traditional ICP internal error: no verified grasp handoff")
        handoff_path = args.weight_handoff_file or run / "weight_handoff.json"
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> verified ICP graspup handoff: {handoff_path}")
        return


def move_to_initial_observation_if_authorized(args: argparse.Namespace) -> None:
    """仅在真实执行已授权时把左臂移动到固定 D405 观察关节位。"""
    if not args.execute:
        print(
            "-> plan-only: robot/gripper motion is disabled; capturing at the current pose",
            flush=True,
        )
        return
    if args.skip_observation_move:
        return
    print("-> 移动到传统 ICP D405 观察位。", flush=True)
    subprocess.run(
        [sys.executable, str(TOOL / "go_graspnet.py"), "--arm-host", args.arm_host],
        check=True,
    )


def main() -> None:
    """解析部署模式与安全参数，然后启动 Module 1 完整抓取流程。"""
    # 阶段 1：定义命令行接口。参数较多，但都只在入口集中声明，业务函数不再
    # 直接读取 sys.argv。
    p = argparse.ArgumentParser(description=__doc__)
    weight_defaults = load_shared_weight_defaults()
    p.add_argument("--mode", choices=tuple(MODULE1_MODES), required=True)
    p.add_argument("--d405-serial", default=None)
    p.add_argument(
        "--handeye-mode", choices=("auto", "bridge", "planner-direct"), default="auto",
        help="D405 hand-eye chain; auto enforces the camera-serial binding (default: auto)",
    )
    p.add_argument(
        "--bridge-wrist-calibration", type=Path,
        default=DEFAULT_BRIDGE_WRIST_CALIBRATION,
        help="SDK eef -> D405 calibration used by --handeye-mode bridge",
    )
    p.add_argument(
        "--planner-bridge-calibration", type=Path,
        default=DEFAULT_PLANNER_BRIDGE_CALIBRATION,
        help="SDK base/eef -> Planner frame bridge used by bridge mode",
    )
    p.add_argument(
        "--direct-wrist-calibration", type=Path,
        default=DEFAULT_DIRECT_WRIST_CALIBRATION,
        help="Planner left_arm_end_link -> D405 calibration used by planner-direct mode",
    )
    p.add_argument("--arm-host", default=None)
    p.add_argument("--gripper-host", default=None)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--icp-workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="parallel STL-ICP worker processes per D405 observation (default: min(4, CPU count))")
    p.add_argument("--icp-daemon-timeout-s", type=float, default=30.0,
                   help="restart/retry the persistent ICP daemon after this request timeout (default: 30)")
    p.add_argument(
        "--planner-preflight-inflight", type=int, default=8,
        help="maximum concurrent Planner preflight requests; matches Module 3 default (default: 8)",
    )
    p.add_argument(
        "--camera-broker-socket",
        default=os.environ.get("CAMERA_BROKER_SOCKET", "/tmp/graspnet-v7-camera.sock"),
        help="singleton D405/Femto Camera Broker Unix socket",
    )
    p.add_argument("--camera-broker-timeout-s", type=float, default=30.0)
    p.add_argument(
        "--capture-stability-retries", type=int, default=3,
        help="D405 capture static-check attempts after observation motion (default: 3)",
    )
    p.add_argument(
        "--capture-stability-retry-wait-s", type=float, default=0.25,
        help="wait before a D405 static-check recapture, in seconds (default: 0.25)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=LOOP / "output/module_1_left_grasp",
        help="directory for per-attempt captures, plans, and verified handoffs",
    )
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
    p.add_argument("--front-grasp-upward-offset-m", type=float, default=None,
                   help="override the mode-bound front +Z offset (bridge: 0.021 m; planner-direct: 0)")
    p.add_argument("--back-grasp-upward-offset-m", type=float, default=None,
                   help="override the mode-bound back +Z offset (bridge: 0.027 m; planner-direct: 0)")
    p.add_argument("--grasp-downward-probe-m", type=float, default=None,
                   help="optional V7 signed global override: positive=down, negative=up")
    p.add_argument("--skip-observation-move", action="store_true")
    p.add_argument("--execute", action="store_true", help="permit physical execution only with --confirm-real-grasp")
    p.add_argument("--confirm-real-grasp", action="store_true")
    p.add_argument("--weight-handoff-file", type=Path)
    p.add_argument("--force-sensor-host", default=None)
    p.add_argument("--weight-check-samples", type=int, default=weight_defaults["weight_check_samples"])
    p.add_argument("--weight-check-timeout-s", type=float, default=weight_defaults["weight_check_timeout_s"])
    p.add_argument("--weight-check-settle-s", type=float, default=weight_defaults["weight_check_settle_s"])
    p.add_argument("--weight-empty-max-g", type=float, default=weight_defaults["weight_empty_max_g"])
    p.add_argument("--weight-single-min-g", type=float, default=weight_defaults["weight_single_min_g"])
    p.add_argument("--weight-single-max-g", type=float, default=weight_defaults["weight_single_max_g"])
    p.add_argument("--weight-multiple-min-g", type=float, default=weight_defaults["weight_multiple_min_g"])
    p.add_argument("--weight-max-axis-mad-n", type=float, default=weight_defaults["weight_max_axis_mad_n"])
    p.add_argument(
        "--max-weight-retries", type=int, default=30,
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
    # 阶段 2：绑定部署模式。一个 mode 必须原子决定相机、标定和三类服务地址，
    # 防止把两套实验环境的参数交叉使用。
    args = p.parse_args()
    mode_profile = MODULE1_MODES[args.mode]
    expected_serial = str(mode_profile["d405_serial"])
    expected_handeye = str(mode_profile["handeye_mode"])
    expected_robot_host = str(mode_profile["robot_host"])
    expected_gripper_host = str(mode_profile["gripper_host"])
    expected_force_sensor_host = str(mode_profile["force_sensor_host"])
    if args.d405_serial is not None and str(args.d405_serial) != expected_serial:
        p.error(
            f"--mode {args.mode} requires --d405-serial {expected_serial}, "
            f"not {args.d405_serial}"
        )
    if args.handeye_mode not in ("auto", expected_handeye):
        p.error(
            f"--mode {args.mode} requires --handeye-mode {expected_handeye}, "
            f"not {args.handeye_mode}"
        )
    if args.arm_host is not None and str(args.arm_host) != expected_robot_host:
        p.error(
            f"--mode {args.mode} requires --arm-host {expected_robot_host}, "
            f"not {args.arm_host}"
        )
    if args.gripper_host is not None and str(args.gripper_host) != expected_gripper_host:
        p.error(
            f"--mode {args.mode} requires --gripper-host {expected_gripper_host}, "
            f"not {args.gripper_host}"
        )
    if (
        args.force_sensor_host is not None
        and str(args.force_sensor_host) != expected_force_sensor_host
    ):
        p.error(
            f"--mode {args.mode} requires --force-sensor-host "
            f"{expected_force_sensor_host}, not {args.force_sensor_host}"
        )
    args.d405_serial = expected_serial
    args.handeye_mode = expected_handeye
    args.arm_host = expected_robot_host
    args.gripper_host = expected_gripper_host
    args.force_sensor_host = expected_force_sensor_host
    args.handeye_mode_requested = args.handeye_mode
    try:
        args.handeye_mode = icp_pipeline.resolve_handeye_mode(
            args.d405_serial, args.handeye_mode_requested
        )
    except ValueError as exc:
        p.error(str(exc))
    # 阶段 3：解析相机绑定的手眼链，并加载该标定独有的抓取偏置策略。
    default_front_offset, default_back_offset = (
        icp_pipeline.default_face_grasp_upward_offsets(args.handeye_mode)
    )
    if args.front_grasp_upward_offset_m is None:
        args.front_grasp_upward_offset_m = default_front_offset
    if args.back_grasp_upward_offset_m is None:
        args.back_grasp_upward_offset_m = default_back_offset
    required_calibrations = (
        (args.bridge_wrist_calibration, args.planner_bridge_calibration)
        if args.handeye_mode == "bridge"
        else (args.direct_wrist_calibration,)
    )
    missing_calibrations = [str(path) for path in required_calibrations if not path.is_file()]
    if missing_calibrations:
        p.error("selected hand-eye calibration file is missing: " + ", ".join(missing_calibrations))
    # 阶段 4：在启动相机、Planner 或机器人之前完成全部数值和文件校验。
    if args.frames < 1: p.error("--frames must be positive")
    if args.capture_stability_retries < 1:
        p.error("--capture-stability-retries must be positive")
    if args.capture_stability_retry_wait_s < 0:
        p.error("--capture-stability-retry-wait-s must be non-negative")
    if args.icp_workers < 1: p.error("--icp-workers must be positive")
    if not np.isfinite(args.icp_daemon_timeout_s) or args.icp_daemon_timeout_s <= 0:
        p.error("--icp-daemon-timeout-s must be finite and positive")
    if not 1 <= args.planner_preflight_inflight <= 16:
        p.error("--planner-preflight-inflight must be in [1, 16]")
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
    # 阶段 5：构造执行工具参数。纯规划只做感知和轨迹预检；真实模式先开夹爪，
    # 再进入观察位、抓取、重量验证和 handoff 流程。
    pa = planner_args(args)
    sys.path.insert(0, str(GRASPNET_ROOT))
    from deploy.tools import graspnet_planner_execute as pe
    if args.execute:
        # The operator explicitly authorized a real pick.  Open once before
        # moving to the observation pose, rather than unexpectedly opening
        # halfway through the subsequent arm motion.  Plan-only stays fully
        # non-actuating.
        pe.move_gripper(pa, 100.0, "open gripper at real-pick start")
    move_to_initial_observation_if_authorized(args)
    try:
        run_traditional_cycles(args, pa, pe)
    finally:
        # This is only relevant to the explicit in-process fallback; the
        # default Unix-socket daemon intentionally survives separate CLI runs.
        icp_pipeline.shutdown_persistent_icp_workers()
    return

if __name__ == "__main__":
    main()
