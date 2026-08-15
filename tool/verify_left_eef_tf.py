#!/usr/bin/env python3
"""只读验证 SDK 左臂 EEF 与 GraspNet 规划器左臂末端 TF 的固定变换。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_PATH = Path(__file__).resolve()
PUBLIC_DIR = SCRIPT_PATH.parents[2]
for candidate in sorted(
    (PUBLIC_DIR / "parm_sdk_1.0.0").glob("parm_sdk-*.whl"), reverse=True
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from parm_sdk import Arm, Planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读拟合 T_sdk_eef_left_arm_end_link；不会发送任何运动指令。"
    )
    parser.add_argument("--arm-host", default="192.168.3.11")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--trajectory-port", type=int, default=5605)
    parser.add_argument("--tf-port", type=int, default=5614)
    parser.add_argument("--waypoint-port", type=int, default=5615)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PUBLIC_DIR / "Parm_Grasp_Place_Loop" / "output" / "tf_validation",
    )
    return parser.parse_args()


def make_transform(translation: np.ndarray, rotation: Rotation) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    return transform


def transform_from_driver(eef: dict) -> tuple[np.ndarray, np.ndarray, Rotation]:
    xyz = np.asarray(eef["xyz"], dtype=float)
    rotation = Rotation.from_euler("xyz", eef["rpy"])
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"SDK eef xyz 无效: {eef!r}")
    return make_transform(xyz, rotation), xyz, rotation


def transform_from_planner(response: dict) -> tuple[np.ndarray, np.ndarray, Rotation]:
    data = response.get("data") or {}
    translation = data.get("translation") or {}
    quaternion = data.get("rotation") or {}
    xyz = np.asarray([translation[key] for key in ("x", "y", "z")], dtype=float)
    rotation = Rotation.from_quat([quaternion[key] for key in ("x", "y", "z", "w")])
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise RuntimeError(f"规划器 TF 无效: {response!r}")
    return make_transform(xyz, rotation), xyz, rotation


def pose_error(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    position_error_m = float(np.linalg.norm(reference[:3, 3] - actual[:3, 3]))
    rotation_error_deg = math.degrees(
        Rotation.from_matrix(reference[:3, :3]).inv()
        .__mul__(Rotation.from_matrix(actual[:3, :3]))
        .magnitude()
    )
    return position_error_m, float(rotation_error_deg)


def json_pose(transform: np.ndarray) -> dict:
    rotation = Rotation.from_matrix(transform[:3, :3])
    return {
        "xyz_m": transform[:3, 3].tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
        "rpy_xyz_rad": rotation.as_euler("xyz").tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples 至少为 2")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    log("=== 左臂 SDK EEF / GraspNet Planner TF 固定变换验证（只读） ===")
    log("不会规划、不会执行轨迹、不会发送任何机械臂控制命令。")
    log(
        "规划器端口: "
        f"trajectory={args.trajectory_port}, tf={args.tf_port}, waypoint={args.waypoint_port}"
    )
    log("每次采样前请将左臂停在一个新的安全姿态并等待完全稳定。")

    arm = Arm(host=args.arm_host)
    planner = Planner(
        host=args.planner_host,
        trajectory_port=args.trajectory_port,
        tf_port=args.tf_port,
        waypoint_port=args.waypoint_port,
    )
    records: list[dict] = []
    try:
        frame_ids = set(planner.list_frame_ids())
        if "body_base" not in frame_ids or "left_arm_end_link" not in frame_ids:
            raise RuntimeError("GraspNet 规划器缺少 body_base 或 left_arm_end_link TF")

        for index in range(1, args.samples + 1):
            answer = input(
                f"\n[{index}/{args.samples}] 左臂已停稳后按回车采样；输入 q 后回车提前结束："
            ).strip().lower()
            if answer == "q":
                break

            state = arm.get_state() or {}
            left_state = (state.get("state") or {}).get("left") or {}
            eef = left_state.get("eef_pose")
            if not isinstance(eef, dict):
                raise RuntimeError(f"SDK 未返回左臂 eef_pose: {left_state!r}")

            sdk_transform, sdk_xyz, _ = transform_from_driver(eef)
            planner_response = planner.lookup_transform("body_base", "left_arm_end_link")
            end_transform, end_xyz, _ = transform_from_planner(planner_response)
            relative = np.linalg.inv(sdk_transform) @ end_transform
            direct_position_error, direct_rotation_error = pose_error(sdk_transform, end_transform)

            record = {
                "sample_index": index,
                "wall_time_unix_s": time.time(),
                "joint_position_deg": left_state.get("position"),
                "sdk_eef_body_base": json_pose(sdk_transform),
                "planner_left_arm_end_link_body_base": json_pose(end_transform),
                "T_sdk_eef_left_arm_end_link": json_pose(relative),
                "direct_origin_difference": {
                    "position_error_m": direct_position_error,
                    "rotation_error_deg": direct_rotation_error,
                    "sdk_minus_planner_xyz_m": (sdk_xyz - end_xyz).tolist(),
                },
            }
            records.append(record)
            relative_pose = record["T_sdk_eef_left_arm_end_link"]
            log(
                f"[{index}] relative xyz(m)={np.round(relative_pose['xyz_m'], 7).tolist()} "
                f"rpy(deg)={np.round(np.rad2deg(relative_pose['rpy_xyz_rad']), 5).tolist()}"
            )
    finally:
        planner.close()
        arm.close()

    if len(records) < 2:
        raise RuntimeError("至少需要两次采样才能验证固定变换")

    translations = np.asarray(
        [record["T_sdk_eef_left_arm_end_link"]["xyz_m"] for record in records], dtype=float
    )
    rotations = Rotation.from_quat(
        [record["T_sdk_eef_left_arm_end_link"]["quaternion_xyzw"] for record in records]
    )
    mean_rotation = rotations.mean()
    mean_transform = make_transform(translations.mean(axis=0), mean_rotation)

    residuals = []
    for record in records:
        sdk_pose = record["sdk_eef_body_base"]
        end_pose = record["planner_left_arm_end_link_body_base"]
        sdk_transform = make_transform(
            np.asarray(sdk_pose["xyz_m"], dtype=float),
            Rotation.from_quat(sdk_pose["quaternion_xyzw"]),
        )
        end_transform = make_transform(
            np.asarray(end_pose["xyz_m"], dtype=float),
            Rotation.from_quat(end_pose["quaternion_xyzw"]),
        )
        predicted_end = sdk_transform @ mean_transform
        position_error, rotation_error = pose_error(predicted_end, end_transform)
        residuals.append({
            "sample_index": record["sample_index"],
            "position_error_m": position_error,
            "rotation_error_deg": rotation_error,
        })

    summary = {
        "frame_definition": "T_sdk_eef_left_arm_end_link = inv(T_body_base_sdk_eef) @ T_body_base_left_arm_end_link",
        "planner": {
            "host": args.planner_host,
            "trajectory_port": args.trajectory_port,
            "tf_port": args.tf_port,
            "waypoint_port": args.waypoint_port,
        },
        "sample_count": len(records),
        "mean_T_sdk_eef_left_arm_end_link": json_pose(mean_transform),
        "translation_std_m": translations.std(axis=0).tolist(),
        "fit_residuals": residuals,
        "max_fit_position_error_m": max(item["position_error_m"] for item in residuals),
        "max_fit_rotation_error_deg": max(item["rotation_error_deg"] for item in residuals),
        "samples": records,
    }
    (run_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    log("\n=== 拟合结果 ===")
    log(
        "mean T_sdk_eef_left_arm_end_link xyz(m)="
        f"{np.round(mean_transform[:3, 3], 8).tolist()}"
    )
    log(
        "mean relative rpy(deg)="
        f"{np.round(np.rad2deg(mean_rotation.as_euler('xyz')), 6).tolist()}"
    )
    log(
        "max fitted residual: "
        f"position={summary['max_fit_position_error_m'] * 1000:.4f} mm, "
        f"rotation={summary['max_fit_rotation_error_deg']:.5f} deg"
    )
    log(f"JSON: {run_dir / 'report.json'}")
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    log(f"LOG:  {run_dir / 'run.log'}")


if __name__ == "__main__":
    main()
