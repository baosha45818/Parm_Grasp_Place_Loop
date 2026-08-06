#!/usr/bin/env python3
"""用头部 Intel RealSense D435i + SAM2 + ICP 估计被夹持工件位姿。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"位姿估计运行环境不存在: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable) != PYTHON:
    os.execv(str(PYTHON), [str(PYTHON), "-m", "tests.reversed_head_observation.estimate_pose_head_d435", *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


CONFIG_PATH = Path(__file__).with_name("head_d435_pose_config.json")
ALIGNMENT_PATH = (
    ROOT / "calibration/sdk_base_alignment/"
    "sdk_base_to_planner_alignment_pitch2_yaw26p5_kinematic.json"
)
CAMERA_FRAME = "head_d435i_color_optical_frame"
DEFAULT_SAM2_URL = os.environ.get("SAM2_SERVICE_URL", "unix:///tmp/sam2-service.sock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用头部 D435i 估计被夹持工件位姿")
    parser.add_argument("--sam2-url", default=DEFAULT_SAM2_URL)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output/reversed_head_pose_d435",
    )
    parser.add_argument("--driver-host", default="127.0.0.1")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--left-tcp-frame", default="left_hand_transfer_end")
    parser.add_argument("--left-tcp-static-translation-mm", type=float, default=2.0)
    parser.add_argument("--left-tcp-static-rotation-deg", type=float, default=1.0)
    parser.add_argument("--expected-head-pitch", type=float, default=2.0)
    parser.add_argument("--expected-head-yaw", type=float, default=26.5)
    parser.add_argument("--head-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--output-frame", choices=("camera", "body_base"), default="camera",
        help=("camera 始终可用；body_base 仅在实时云台角与标定云台角一致时允许"),
    )
    parser.add_argument(
        "--auto-gripper-prompt", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--mask-mode", choices=("interactive", "auto"), default="auto",
                        help="auto uses current RGB-D depth/colour prompts; interactive opens the manual OpenCV annotator")
    parser.add_argument("--auto-mask-config", type=Path,
                        default=Path(__file__).with_name("head_d435_auto_mask_config.json"))
    return parser.parse_args()


def read_head_pose_command_deg(host: str, timeout_s: float = 3.0) -> tuple[dict[str, float], dict[str, Any]]:
    """读取云台并转换为 move_to() 的命令角符号约定。"""
    from parm_sdk import Head

    deadline = time.monotonic() + timeout_s
    state = None
    with Head(host=host) as head:
        head.ping()
        while time.monotonic() < deadline:
            candidate = head.get_state()
            if isinstance(candidate, dict) and candidate.get("pitch") and candidate.get("yaw"):
                state = candidate
                break
            time.sleep(0.05)
    if state is None:
        raise RuntimeError("未收到云台状态，拒绝在未知观察角拍摄")
    pitch = state["pitch"]
    yaw = state["yaw"]
    if pitch.get("alarm_message") or yaw.get("alarm_message"):
        raise RuntimeError(f"云台存在报警: {state}")
    if not pitch.get("enabled") or not yaw.get("enabled"):
        raise RuntimeError(f"云台未使能: {state}")
    # 当前 SDK 的 move_to() 下发 pitch 时取反，但状态反馈保留驱动符号。
    command_pose = {
        "pitch": -float(pitch["position"]),
        "yaw": float(yaw["position"]),
    }
    return command_pose, state


def pose_error_deg(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(abs(actual[axis] - expected[axis]) for axis in ("pitch", "yaw"))


def load_calibrated_transform(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["T_body_base_head_camera"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"无效 T_body_base_head_camera: {path}")
    pose = data.get("head_pose_deg") or {}
    calibration_pose = {"pitch": float(pose["pitch"]), "yaw": float(pose["yaw"])}
    return matrix, calibration_pose


def choose_output_transform(
    output_frame: str,
    T_camera_object: np.ndarray,
    actual_head_pose: dict[str, float],
    alignment_path: Path,
    tolerance_deg: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if output_frame == "camera":
        return T_camera_object, CAMERA_FRAME, {
            "status": "camera_frame_only",
            "reason": "未对不同云台角套用固定外参",
        }
    T_body_camera, calibration_pose = load_calibrated_transform(alignment_path)
    error = pose_error_deg(actual_head_pose, calibration_pose)
    if error > tolerance_deg:
        raise RuntimeError(
            "不能输出 body_base 位姿：实时云台命令角 "
            f"pitch={actual_head_pose['pitch']:.3f}°, yaw={actual_head_pose['yaw']:.3f}°，"
            "但现有 D435 外参只在 "
            f"pitch={calibration_pose['pitch']:.3f}°, yaw={calibration_pose['yaw']:.3f}° 有效。"
            "请在当前观察角重新标定，或改用 --output-frame camera。"
        )
    return T_body_camera @ T_camera_object, "body_base", {
        "status": "calibrated_fixed_head_pose",
        "calibration_path": str(alignment_path),
        "calibration_head_pose_deg": calibration_pose,
        "head_pose_error_deg": error,
    }


def main() -> int:
    # 重依赖只在正式执行时加载，纯变换校验单元测试不依赖 Open3D/相机 SDK。
    from scipy.spatial.transform import Rotation
    from tool.inspection_support.sam2_icp import (
        capture_view,
        icp_pose,
        save_pose_visualization,
    )
    from tool.inspection_support.robot_camera import D405Camera
    from tests.reversed_head_observation.handoff_geometry import (
        planner_tf_target_source,
        transform_error,
    )
    from tool.inspection_support.stl_model import (
        load_stl_triangles_m,
        sample_surface_points,
        triangles_in_object_frame,
    )

    args = parse_args()
    if args.head_tolerance_deg <= 0 or not np.isfinite(args.head_tolerance_deg):
        raise ValueError("--head-tolerance-deg 必须是大于 0 的有限值")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {"pitch": args.expected_head_pitch, "yaw": args.expected_head_yaw}
    actual, raw_state = read_head_pose_command_deg(args.driver_host)
    error = pose_error_deg(actual, expected)
    if error > args.head_tolerance_deg:
        raise RuntimeError(
            f"云台不在指定观察角：实时命令角={actual}，预期={expected}，"
            f"最大误差={error:.3f}°"
        )
    print(f"[head] 云台观察角已确认: pitch={actual['pitch']:.3f}°, yaw={actual['yaw']:.3f}°")
    print(f"[camera] 将连接头部 D435i serial={cfg['camera_serial']}（不会连接 Femto Bolt）")
    if args.output_frame == "body_base":
        # 在打开相机、要求人工分割前先拒绝姿态不匹配的固定外参。
        choose_output_transform(
            "body_base", np.eye(4), actual, args.alignment, args.head_tolerance_deg
        )

    def read_left_tcp() -> np.ndarray:
        from parm_sdk import Planner
        with Planner(host=args.planner_host) as planner:
            frames = planner.list_frame_ids()
            if args.left_tcp_frame not in frames or "body_base" not in frames:
                raise RuntimeError(
                    f"Planner TF missing body_base or {args.left_tcp_frame!r}"
                )
            response = planner.lookup_transform(args.left_tcp_frame, "body_base")
        return planner_tf_target_source(
            response, args.left_tcp_frame, "body_base"
        )

    T_base_left_tcp_before = None
    if args.output_frame == "body_base":
        T_base_left_tcp_before = read_left_tcp()

    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    with D405Camera(
        serial=str(cfg["camera_serial"]),
        width=int(cfg["color_width"]),
        height=int(cfg["color_height"]),
        depth_width=int(cfg["depth_width"]),
        depth_height=int(cfg["depth_height"]),
        fps=int(cfg["fps"]),
        warmup_frames=int(cfg["warmup_frames"]),
        # D435i 的 1280x720 对齐流直接用于多帧中值；RealSense 时域/空间
        # filter 在该设备配置上可能阻塞管线，因此这里不启用设备端后处理。
        enable_postprocess=False,
        camera_name="head_d435i",
        # 当前固定观察位中右后方白板作为中性参考，修正荧光灯下的青绿色偏色。
        color_balance_mode="neutral_roi",
        color_balance_roi_xyxy=(730, 90, 1020, 315),
        depth_visual_preset=4.0,
        laser_power=360.0,
    ) as camera:
        if args.mask_mode == "auto":
            from tests.reversed_head_observation.auto_head_d435_mask import capture_auto_view
            points, report = capture_auto_view(
                camera, cfg, "head_d435", run_dir, args.sam2_url, args.auto_mask_config
            )
        else:
            points, report = capture_view(
                camera, cfg, "head_d435", run_dir, args.sam2_url, args.auto_gripper_prompt
            )

    actual_after_capture, raw_state_after_capture = read_head_pose_command_deg(
        args.driver_host
    )
    head_capture_drift_deg = pose_error_deg(actual_after_capture, actual)
    if head_capture_drift_deg > args.head_tolerance_deg:
        raise RuntimeError(
            "云台在D435采集/人工标注期间发生移动，拒绝生成位姿："
            f"采集前={actual}，采集后={actual_after_capture}"
        )

    T_base_left_tcp_after = None
    left_tcp_static_error = None
    if args.output_frame == "body_base":
        T_base_left_tcp_after = read_left_tcp()
        translation_m, rotation_deg = transform_error(
            T_base_left_tcp_before, T_base_left_tcp_after
        )
        left_tcp_static_error = {
            "translation_mm": translation_m * 1000.0,
            "rotation_deg": rotation_deg,
        }
        if (
            translation_m * 1000.0 > args.left_tcp_static_translation_mm
            or rotation_deg > args.left_tcp_static_rotation_deg
        ):
            raise RuntimeError(
                "左臂在D435采集/人工标注期间发生移动，拒绝生成交接位姿："
                f"TCP变化={translation_m * 1000.0:.3f} mm/{rotation_deg:.3f}°"
            )

    if len(points) < int(cfg["min_registration_points"]):
        raise RuntimeError(
            f"有效工件点云只有 {len(points)} 点，少于 ICP 最低要求 "
            f"{cfg['min_registration_points']} 点"
        )
    triangles = triangles_in_object_frame(
        load_stl_triangles_m(Path(cfg["stl_path"])),
        R_object_stl=cfg.get("R_object_stl"),
        object_origin_stl_m=cfg.get("object_origin_stl_m"),
    )
    cad = sample_surface_points(triangles, int(cfg["cad_sample_points"]), int(cfg["seed"]))
    import cv2
    selection_mask = cv2.imread(
        str(run_dir / "head_d435_mask.png"), cv2.IMREAD_GRAYSCALE
    )
    if selection_mask is None:
        raise RuntimeError("无法读取head_d435_mask.png用于ICP对称解筛选")
    T_camera_object, metrics = icp_pose(
        cad,
        points,
        cfg,
        selection_mask=selection_mask,
        camera_intrinsics=np.asarray(report["camera_intrinsics"], dtype=np.float64),
    )
    T_output_object, output_frame, transform_info = choose_output_transform(
        args.output_frame, T_camera_object, actual, args.alignment, args.head_tolerance_deg
    )
    visualization = save_pose_visualization(
        run_dir, "head_d435", cad, T_camera_object, T_output_object, output_frame,
        np.asarray(report["camera_intrinsics"], dtype=np.float64),
    )
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": {"role": "head_d435i", "serial": str(cfg["camera_serial"])},
        "head_pose_command_deg": actual,
        "head_state_raw": raw_state,
        "head_pose_after_capture_deg": actual_after_capture,
        "head_state_after_capture_raw": raw_state_after_capture,
        "head_capture_drift_deg": head_capture_drift_deg,
        "object_pose": {
            "frame_id": output_frame,
            "position_m": T_output_object[:3, 3].tolist(),
            "orientation_xyzw": Rotation.from_matrix(T_output_object[:3, :3]).as_quat().tolist(),
        },
        "T_camera_object": T_camera_object.tolist(),
        "T_output_object": T_output_object.tolist(),
        "transform_info": transform_info,
        "registration": metrics,
        "point_report": report,
        "pose_visualization": str(visualization),
        "model_frame": {
            "stl_path": str(Path(cfg["stl_path"]).resolve()),
            "R_object_stl": cfg.get("R_object_stl", np.eye(3).tolist()),
            "object_origin_stl_m": cfg.get("object_origin_stl_m", [0.0, 0.0, 0.0]),
        },
    }
    if T_base_left_tcp_before is not None:
        result["left_tcp_snapshot"] = {
            "frame_id": "body_base",
            "tcp_frame": args.left_tcp_frame,
            "T_base_left_tcp": T_base_left_tcp_before.tolist(),
            "static_check_after_capture": left_tcp_static_error,
            "static_translation_tolerance_mm": args.left_tcp_static_translation_mm,
            "static_rotation_tolerance_deg": args.left_tcp_static_rotation_deg,
        }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[完成] 头部 D435 位姿估计结果: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
