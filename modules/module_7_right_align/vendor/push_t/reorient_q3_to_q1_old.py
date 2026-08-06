#!/usr/bin/env python3
"""Realtime Q1--Q4-to-Q1 reorientation using the current TCP and latest mask pose.

No taught pose is used as a destination.  The planner reads the live right
fingertip-centre TCP, reads one workpiece centre/red direction from a fresh
mask_pose JSON, generates downward-grasp attitude candidates, and selects one
for which every operation target is plannable.  The recorded table-contact
pose is used only as a hard lower bound for the virtual TCP height.

Run without --execute first.  --execute opens, grips at the visual centre,
lifts, turns the object towards Q1, places it in a reachable negative-X/negative-Y
body_base region, releases, and lifts off.
Capture and segment again before the subsequent Q1 operation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import time

from move_by_eef_pose import (
    RIGHT_FINGERTIP_CENTER_CONFIG,
    compose_pose,
    inverse_pose,
    load_virtual_tcp_offset,
    pose_from_lookup_transform,
)
from move_by_joints import current_pose, finite, import_sdk, move_l, rotvec_to_quat, rotvec_to_rpy, rpy_to_rotvec
from push_workpiece_to_q1 import (
    can_plan_target,
    ensure_ready,
    load_reference_tcp_pose,
    plan_and_follow,
    set_gripper_state,
)
from right_trajectory_recorder import capture_tool_pose, record_plan


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOWEST_TCP_POSE = SCRIPT_DIR / "saved_poses" / "7.json"
DEFAULT_CALIBRATION_RECORD_DIR = SCRIPT_DIR / "reorient_q3_records"
DEFAULT_MEASURED_ACTUAL_POSE = SCRIPT_DIR / "saved_poses" / "test_1.json"
DEFAULT_MEASURED_DESIRED_POSE = SCRIPT_DIR / "saved_poses" / "test_2.json"
# After releasing the workpiece, always clear it by this fixed vertical distance
# before the outer pipeline plans the return to pose 3.
POST_RELEASE_LIFT_M = 0.020


def normalize_deg(value: float) -> float:
    return value % 360.0


def shortest_angle_delta_deg(start_deg: float, end_deg: float) -> float:
    """Signed base-Z turn from start to end in [-180, 180)."""
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def signed_offsets(limit_deg: float, step_deg: float) -> list[float]:
    values = [0.0]
    for index in range(1, int(math.floor(limit_deg / step_deg + 1e-9)) + 1):
        values.extend((index * step_deg, -index * step_deg))
    return values


def load_workpiece(path: Path, index: int) -> tuple[list[float], float, int, float, dict, Path]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取工件位姿 JSON: {path}: {exc}") from exc
    if payload.get("coordinate_frame") not in (None, "body_base"):
        raise ValueError(f"工件 JSON 坐标系不是 body_base: {payload.get('coordinate_frame')!r}")
    plane_z = finite([payload.get("plane_z_m")], 1, "plane_z_m")[0]
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list) or not 0 <= index < len(workpieces):
        count = len(workpieces) if isinstance(workpieces, list) else 0
        raise ValueError(f"--workpiece-index={index} 超出范围；文件中共有 {count} 个工件")
    item = workpieces[index]
    if not isinstance(item, dict):
        raise ValueError(f"workpieces[{index}] 不是对象")
    # Prefer the tail/head-centreline midpoint for grasping.  Unlike the mask
    # area centroid it is guaranteed to sit on the red direction line, so the
    # perpendicular jaw-closing line passes through the intended grasp centre.
    grasp_centre = item.get("grasp_center_base_xy_m")
    if grasp_centre is None:
        centre = finite(item.get("center_base_xy_m"), 2, "center_base_xy_m")
    else:
        centre = finite(grasp_centre, 2, "grasp_center_base_xy_m")
    yaw = finite([item.get("head_direction_base_angle_deg")], 1, "head_direction_base_angle_deg")[0]
    quadrant = item.get("head_direction_base_quadrant")
    if not isinstance(quadrant, int):
        raise ValueError("JSON 缺少整数 head_direction_base_quadrant")
    calibration_value = payload.get("calibration")
    if not isinstance(calibration_value, str) or not calibration_value:
        raise ValueError("工件 JSON 缺少 calibration；无法对放置点执行 ROI 投影过滤")
    calibration_path = Path(calibration_value)
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    return centre, normalize_deg(yaw), quadrant, plane_z, item, calibration_path


def load_projection_calibration(path: Path) -> tuple[list[list[float]], list[float], list[list[float]], tuple[int, int]]:
    """Load the camera model used to project a body_base placement into the RGB image."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 ROI 投影标定文件 {path}: {exc}") from exc
    try:
        camera_matrix = [[float(value) for value in row] for row in payload["camera_matrix"]]
        distortion = [float(value) for value in payload.get("distortion_coeffs", [])]
        base_camera = [[float(value) for value in row] for row in payload["T_body_base_head_camera"]]
        image_size = tuple(int(value) for value in payload["image_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ROI 投影标定文件格式无效: {path}") from exc
    if (len(camera_matrix) != 3 or any(len(row) != 3 for row in camera_matrix)
            or len(base_camera) != 4 or any(len(row) != 4 for row in base_camera)
            or len(image_size) != 2):
        raise ValueError(f"ROI 投影标定文件尺寸无效: {path}")
    return camera_matrix, distortion, base_camera, image_size


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(value * vector[index] for index, value in enumerate(row)) for row in matrix]


def undistort_pixel(pixel: list[float], camera_matrix: list[list[float]], distortion: list[float]) -> tuple[float, float]:
    """Convert a distorted pixel into a normalized camera ray without OpenCV."""
    fx, skew, cx = camera_matrix[0]
    _, fy, cy = camera_matrix[1]
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("ROI 投影标定相机内参无效")
    xd = (pixel[0] - cx - skew * (pixel[1] - cy) / fy) / fx
    yd = (pixel[1] - cy) / fy
    coefficients = distortion + [0.0] * max(0, 5 - len(distortion))
    k1, k2, p1, p2, k3 = coefficients[:5]
    xu, yu = xd, yd
    for _ in range(8):
        r2 = xu * xu + yu * yu
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        if abs(radial) < 1e-12:
            raise ValueError("ROI 投影标定畸变参数无效")
        dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        xu, yu = (xd - dx) / radial, (yd - dy) / radial
    return xu, yu


def distort_normalized(x: float, y: float, distortion: list[float]) -> tuple[float, float]:
    coefficients = distortion + [0.0] * max(0, 5 - len(distortion))
    k1, k2, p1, p2, k3 = coefficients[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def footprint_local_base_xy(
    item: dict,
    grasp_centre_xy: list[float],
    plane_z: float,
    camera_matrix: list[list[float]],
    distortion: list[float],
    base_camera: list[list[float]],
) -> list[list[float]]:
    """Return the mask rotated-box footprint relative to its grasp centre in body_base."""
    corners = item.get("rotated_box_pixel_corners_xy")
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError("工件 JSON 缺少有效 rotated_box_pixel_corners_xy；无法进行 ROI 放置过滤")
    rotation = [row[:3] for row in base_camera[:3]]
    camera_origin = [row[3] for row in base_camera[:3]]
    local: list[list[float]] = []
    for corner in corners:
        if not isinstance(corner, list) or len(corner) != 2:
            raise ValueError("工件 JSON 的 rotated_box_pixel_corners_xy 格式无效")
        x, y = undistort_pixel([float(corner[0]), float(corner[1])], camera_matrix, distortion)
        ray = mat_vec(rotation, [x, y, 1.0])
        if abs(ray[2]) < 1e-9:
            raise ValueError("ROI 投影失败：工件角点射线与台面平行")
        distance = (plane_z - camera_origin[2]) / ray[2]
        if distance <= 0.0:
            raise ValueError("ROI 投影失败：工件角点位于相机后方")
        local.append([
            camera_origin[0] + distance * ray[0] - grasp_centre_xy[0],
            camera_origin[1] + distance * ray[1] - grasp_centre_xy[1],
        ])
    return local


def project_placement_footprint(
    place_xy: list[float],
    local_footprint_xy: list[list[float]],
    yaw_delta_deg: float,
    plane_z: float,
    camera_matrix: list[list[float]],
    distortion: list[float],
    base_camera: list[list[float]],
) -> list[list[float]]:
    """Project the reoriented workpiece footprint at a candidate placement centre."""
    angle = math.radians(yaw_delta_deg)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    base_rotation = [row[:3] for row in base_camera[:3]]
    camera_origin = [row[3] for row in base_camera[:3]]
    pixels: list[list[float]] = []
    for local_x, local_y in local_footprint_xy:
        point_base = [
            place_xy[0] + cos_angle * local_x - sin_angle * local_y,
            place_xy[1] + sin_angle * local_x + cos_angle * local_y,
            plane_z,
        ]
        relative = [point_base[index] - camera_origin[index] for index in range(3)]
        # The inverse of a rigid camera-to-base rotation is its transpose.
        point_camera = [sum(base_rotation[row][column] * relative[row] for row in range(3)) for column in range(3)]
        if point_camera[2] <= 0.0:
            raise ValueError("ROI 投影失败：放置候选位于相机后方")
        xd, yd = distort_normalized(point_camera[0] / point_camera[2], point_camera[1] / point_camera[2], distortion)
        pixels.append([
            camera_matrix[0][0] * xd + camera_matrix[0][1] * yd + camera_matrix[0][2],
            camera_matrix[1][1] * yd + camera_matrix[1][2],
        ])
    return pixels


def normalized_roi_bounds(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(roi) != 4 or any(value < 0.0 or value > 1.0 for value in roi):
        raise ValueError("--placement-visibility-roi 必须是 [0, 1] 内四个值")
    x0, y0, x1, y1 = roi
    if x0 >= x1 or y0 >= y1:
        raise ValueError("--placement-visibility-roi 必须满足 X0 < X1 且 Y0 < Y1")
    return int(math.floor(x0 * width)), int(math.floor(y0 * height)), int(math.ceil(x1 * width)), int(math.ceil(y1 * height))


def footprint_is_inside_roi(
    pixels: list[list[float]], roi_bounds: tuple[int, int, int, int], margin_px: float
) -> bool:
    left, top, right, bottom = roi_bounds
    return all(
        left + margin_px <= x <= right - 1 - margin_px
        and top + margin_px <= y <= bottom - 1 - margin_px
        for x, y in pixels
    )


def tool_x_from_rpy(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    return [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)]


def tool_z_from_rpy(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    return [
        math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll),
        math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll),
        math.cos(pitch) * math.cos(roll),
    ]


def align_z_projection(rpy: list[float], red_yaw_rad: float, opposite: bool) -> list[float] | None:
    """Body-Z rotate an attitude so tool +Z projects along the red line."""
    z_axis = tool_z_from_rpy(rpy)
    if math.hypot(z_axis[0], z_axis[1]) < 1e-6:
        return None
    current_heading = math.atan2(z_axis[1], z_axis[0])
    desired_heading = red_yaw_rad + (math.pi if opposite else 0.0)
    delta = (desired_heading - current_heading + math.pi) % (2.0 * math.pi) - math.pi
    return [rpy[0], rpy[1], rpy[2] + delta]


def attitude_candidates(
    current_rpy: list[float], red_yaw_deg: float, x_down_tilt_deg: float,
    tilt_search_deg: float, roll_search_deg: float, step_deg: float,
    minimum_x_down: float,
) -> list[list[float]]:
    """Return rotvec candidates with +X down and +Z/+Y tabletop projections.

    +Z's tabletop projection follows the red tail-to-head line; +Y's
    projection consequently follows the perpendicular green jaw-closing line.
    The opposite directed line is also valid for symmetric parallel jaws.
    """
    red_yaw_rad = math.radians(red_yaw_deg)
    raw_rpy: list[list[float]] = []
    live_x = tool_x_from_rpy(current_rpy)
    if -live_x[2] >= minimum_x_down:
        raw_rpy.append(current_rpy)
    for tilt_offset in signed_offsets(tilt_search_deg, step_deg):
        pitch = math.radians(90.0 - max(0.0, min(80.0, x_down_tilt_deg + tilt_offset)))
        for roll in signed_offsets(roll_search_deg, step_deg):
            raw_rpy.append([math.radians(roll), pitch, 0.0])

    candidates: list[list[float]] = []
    for rpy in raw_rpy:
        for opposite in (False, True):
            aligned = align_z_projection(rpy, red_yaw_rad, opposite)
            if aligned is None:
                continue
            if -tool_x_from_rpy(aligned)[2] < minimum_x_down:
                continue
            rotvec = rpy_to_rotvec(aligned)
            if all(any(abs(a - b) > 1e-8 for a, b in zip(rotvec, known)) for known in candidates):
                candidates.append(rotvec)
    return candidates


def close_gripper_for_grasp(
    gripper: object, *, arm_id: int, timeout_s: float, fully_closed_tolerance: float,
    minimum_close_travel: float, settle_s: float,
) -> None:
    """Close on a workpiece: accept full closure or a stable contact position."""
    initial_position: float | None = None
    try:
        states = gripper.get_state()
        state = states.get(arm_id) if isinstance(states, dict) else None
        if isinstance(state, dict) and state.get("position") is not None:
            initial_position = float(state["position"])
    except Exception:
        pass
    gripper.move(arm_id, 0.0, mode="async", preempt=True)
    print("已下发右夹爪闭合命令，等待完全闭合或稳定接触工件...")
    deadline = time.monotonic() + timeout_s
    last_position: float | None = None
    last_change_time = time.monotonic()
    while time.monotonic() < deadline:
        states = gripper.get_state()
        state = states.get(arm_id) if isinstance(states, dict) else None
        if isinstance(state, dict) and state.get("position") is not None:
            position = float(state["position"])
            now = time.monotonic()
            motion_state = state.get("motion_state")
            if motion_state == "gripping":
                print(f"右夹爪状态为 gripping，已夹住工件（position={position:.1f}）。")
                return
            if initial_position is None:
                initial_position = position
            if last_position is None or abs(position - last_position) > 0.5:
                last_position = position
                last_change_time = now
            if position <= fully_closed_tolerance:
                print(f"右夹爪已完全闭合（position={position:.1f}）。")
                return
            travelled = initial_position - position
            if travelled >= minimum_close_travel and now - last_change_time >= settle_s:
                print(f"右夹爪已接触并稳定夹住工件（position={position:.1f}，闭合行程={travelled:.1f}）。")
                return
        time.sleep(0.05)
    observed = "无有效状态" if last_position is None else f"position={last_position:.1f}"
    raise RuntimeError(f"右夹爪闭合等待超时（{observed}）；机械臂不会继续运动。")


def preplan_tcp_trajectory(
    planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str,
    tcp_offset: list[float], target_tcp: list[float], planner_kind: str,
    max_vel: float, max_acc: float, timeout_s: float,
) -> tuple[list, float]:
    """Plan once and retain the exact trajectory for immediate execution."""
    planner_target = compose_pose(target_tcp, inverse_pose(tcp_offset))
    record_target = {
        "frame_id": frame_id, "ik_frame": ik_frame,
        "position_m": planner_target[:3], "orientation_xyzw": rotvec_to_quat(planner_target[3:]),
        "tcp_target_xyz_rotvec": target_tcp, "max_vel": max_vel, "max_acc": max_acc, "timeout_s": timeout_s,
    }
    start_tool_pose = capture_tool_pose(planner, frame_id, ik_frame)
    try:
        if planner_kind == "rrtstar":
            # The SDK helper rejects rrtstar locally.  Submit the same cartesian
            # waypoint format directly to the planner service instead.
            result = planner._trajectory.send_payload({
                "waypoints": [{
                    "name": "module_7_reorient_rrtstar",
                    "type": "cart",
                    "group": group,
                    "planner": "rrtstar",
                    "max_velocity_scaling_factor": max_vel,
                    "max_acceleration_scaling_factor": max_acc,
                    "frame_id": frame_id,
                    "ik_frame": ik_frame,
                    "position": planner_target[:3],
                    "orientation": rotvec_to_quat(planner_target[3:]),
                }],
                "topp": {"max_vel_factor": max_vel, "max_acc_factor": max_acc},
            }, timeout=timeout_s)
        else:
            result = planner.plan_cart_trajectory(
                group=group, planner=planner_kind, frame_id=frame_id, ik_frame=ik_frame,
                position=planner_target[:3], orientation=rotvec_to_quat(planner_target[3:]),
                max_vel=max_vel, max_acc=max_acc, timeout=timeout_s,
            )
    except Exception as exc:
        record_plan(module="7", stage="reorient_preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, error=exc, start_tool_pose=start_tool_pose)
        raise
    data = result.get("data") or {}
    trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
    frequency = data.get("sample_frequency", 10.0)
    if planned_arm_id != arm_id:
        error = RuntimeError(f"规划结果 arm_id={planned_arm_id!r}，与请求不一致")
        record_plan(module="7", stage="reorient_preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        raise error
    if not isinstance(trajectory, list) or not trajectory:
        error = RuntimeError("规划器未返回有效 trajectory")
        record_plan(module="7", stage="reorient_preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        raise error
    if not isinstance(frequency, (int, float)) or frequency <= 0.0:
        error = RuntimeError(f"规划器返回的 sample_frequency 无效: {frequency!r}")
        record_plan(module="7", stage="reorient_preplan_tcp", status="failure", planner=planner_kind,
                    request_type="cartesian", target=record_target, response=result, error=error,
                    start_tool_pose=start_tool_pose)
        raise error
    record_plan(module="7", stage="reorient_preplan_tcp", status="success", planner=planner_kind,
                request_type="cartesian", target=record_target, response=result, start_tool_pose=start_tool_pose)
    return trajectory, float(frequency)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", type=int, choices=(2,), required=True)
    parser.add_argument("--workpiece-pose-file", type=Path, required=True)
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument("--lowest-tcp-pose-file", type=Path, default=DEFAULT_LOWEST_TCP_POSE)
    parser.add_argument(
        "--measured-actual-pose-file", type=Path, default=DEFAULT_MEASURED_ACTUAL_POSE,
        help="实测脚本实际停点（test_1）；用于重定向抓取 XY 补偿",
    )
    parser.add_argument(
        "--measured-desired-pose-file", type=Path, default=DEFAULT_MEASURED_DESIRED_POSE,
        help="实测应停点（test_2）；与 test_1 的 TCP XY 差作为补偿",
    )
    parser.add_argument(
        "--grasp-xy-offset-m", nargs=2, type=float, metavar=("DX", "DY"),
        help="直接指定重定向抓取 TCP 的 body_base XY 偏置（米）；指定后覆盖 test_2-test_1 测得值",
    )
    parser.add_argument("--minimum-tcp-z-margin-m", type=float, default=0.0)
    parser.add_argument("--workpiece-height-m", type=float, required=True,
                        help="工件从台面到夹持中心对应高度（米）；用于确定抓取 TCP z")
    parser.add_argument("--grasp-tcp-extra-clearance-m", type=float, default=0.0,
                        help="在由工件高度计算出的抓取 TCP z 之上附加的余量（默认 0）")
    parser.add_argument("--place-tcp-extra-clearance-m", type=float, default=0.005,
                        help="仅在放下并松爪时额外抬高 TCP（米；默认 0.005m，不影响抓取高度）")
    parser.add_argument("--approach-height-m", type=float, default=0.060,
                        help="相对桌面最低 TCP 高度的中心上方安全高度（默认 60 mm）")
    parser.add_argument("--line-entry-clearance-m", type=float, default=0.010,
                        help="调姿点距工件中心的最小 tail 侧余量（默认 10 mm）")
    parser.add_argument("--line-max-rear-distance-m", type=float, default=0.030,
                        help="调姿点距工件中心的最大 tail 侧距离（默认 30 mm）")
    parser.add_argument("--lift-height-m", type=float, default=0.080,
                        help="相对桌面最低 TCP 高度的翻转抬升高度（默认 80 mm）")
    parser.add_argument("--x-down-tilt-deg", type=float, default=25.0,
                        help="工具 +X 相对竖直向下的首选倾角（默认 25 度）")
    parser.add_argument("--tilt-search-deg", type=float, default=15.0)
    parser.add_argument("--roll-search-deg", type=float, default=60.0)
    parser.add_argument("--attitude-search-step-deg", type=float, default=10.0)
    parser.add_argument("--max-attitude-candidates", type=int, default=12,
                        help="最多预规划的姿态候选数（默认 12；必要时增大）")
    parser.add_argument("--planning-timeout-s", type=float, default=5.0,
                        help="单个预规划 RPC 的超时秒数（默认 5）")
    parser.add_argument("--minimum-x-down-component", type=float, default=0.70,
                        help="工具 +X 向下分量下限（默认 0.70）")
    parser.add_argument(
        "--q1-target-yaw-deg", action="append", type=float,
        help="重定向后工件红线的 Q1 目标 yaw（度）；可重复，给定顺序即优先级；默认优先 45 度附近",
    )
    parser.add_argument("--q1-turn-step-deg", type=float, default=0.0,
                        help="兼容旧命令的保留参数；整体后撤+转向始终一次规划执行")
    parser.add_argument("--turn-relocation-step-m", type=float, default=0.040,
                        help="旋转小段不可达时，翻转站 XY 重定位搜索步长（默认 40 mm）")
    parser.add_argument("--turn-retreat-negative-xy-m", type=float, default=0.100,
                        help="已废弃：改用 --place-negative-xy-min/max-m 定义翻转/放置区域")
    parser.add_argument("--place-negative-xy-min-m", type=float, default=0.050,
                        help="放置区相对抓取中心的 X、Y 负向最小距离（默认各 50 mm）")
    parser.add_argument("--place-negative-xy-max-m", type=float, default=0.200,
                        help="放置区相对抓取中心的 X 负向最大距离（默认 200 mm）")
    parser.add_argument("--place-positive-x-max-m", type=float, default=0.200,
                        help="仅下放区相对抓取中心的 X 正向最大距离（默认 200 mm）")
    parser.add_argument("--place-negative-y-max-m", type=float, default=0.350,
                        help="放置区相对抓取中心的 Y 负向最大距离（默认 350 mm）")
    parser.add_argument("--place-region-grid-step-m", type=float, default=0.050,
                        help="翻转/放置区域的候选网格间距（默认 50 mm）")
    parser.add_argument(
        "--placement-visibility-roi", nargs=4, type=float,
        default=(0.08, 0.20, 0.98, 0.90), metavar=("X0", "Y0", "X1", "Y1"),
        help="放置后工件必须完整位于其中的归一化相机 ROI（默认与分割 ROI 相同）",
    )
    parser.add_argument("--placement-visibility-margin-px", type=float, default=30.0,
                        help="工件轮廓距放置可见 ROI 边界的最小像素余量（默认 30）")
    parser.add_argument("--planner", choices=("lin", "rrtstar"), default="lin")
    parser.add_argument("--turn-planner", choices=("lin", "rrtstar"), default="lin",
                        help="仅用于抬起后的区域后撤、重定位、转向（默认 lin）")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    parser.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "localhost"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
    parser.add_argument("--frame-id", default=os.environ.get("PARM_PLANNER_FRAME", "body_base"))
    parser.add_argument("--max-vel", type=float, default=0.08)
    parser.add_argument("--max-acc", type=float, default=0.08)
    parser.add_argument("--start-sync-retries", type=int, default=3)
    parser.add_argument("--gripper-timeout-s", type=float, default=5.0)
    parser.add_argument("--gripper-position-tolerance", type=float, default=5.0)
    parser.add_argument("--grasp-minimum-close-travel", type=float, default=5.0,
                        help="判定接触工件前，夹爪至少需从打开位置闭合的行程（默认 5）")
    parser.add_argument("--grasp-contact-settle-s", type=float, default=0.5,
                        help="闭合位置稳定多久后判定已接触工件（默认 0.5 秒）")
    parser.add_argument(
        "--calibrate-grasp-xy", action="store_true",
        help="下探到绿点后暂停，不闭爪/翻转；记录规划器实际 TCP XY 与绿点 XY 的差值",
    )
    parser.add_argument(
        "--calibration-record-file", type=Path,
        help="--calibrate-grasp-xy 的 JSON 记录路径（默认 reorient_q3_records/ 下按时间命名）",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.workpiece_pose_file.is_file():
        print(f"错误: 工件位姿文件不存在: {args.workpiece_pose_file}", file=sys.stderr)
        return 1
    positive = (args.approach_height_m, args.lift_height_m, args.attitude_search_step_deg,
                args.turn_relocation_step_m, args.place_region_grid_step_m)
    if (any(value <= 0.0 for value in positive) or args.minimum_tcp_z_margin_m < 0.0
            or args.workpiece_height_m <= 0.0 or args.grasp_tcp_extra_clearance_m < 0.0
            or args.place_tcp_extra_clearance_m < 0.0):
        print("错误: 高度和步长必须大于 0，最低高度余量不能小于 0", file=sys.stderr)
        return 1
    if args.max_attitude_candidates < 1 or args.planning_timeout_s <= 0.0:
        print("错误: 候选数量和预规划超时必须大于 0", file=sys.stderr)
        return 1
    if args.q1_turn_step_deg < 0.0:
        print("错误: --q1-turn-step-deg 不能小于 0", file=sys.stderr)
        return 1
    if args.grasp_minimum_close_travel < 0.0 or args.grasp_contact_settle_s <= 0.0:
        print("错误: 夹爪接触判定参数无效", file=sys.stderr)
        return 1
    if args.line_entry_clearance_m < 0.0 or args.line_max_rear_distance_m < args.line_entry_clearance_m:
        print("错误: 红线后方余量必须非负，最大后方距离必须不小于该余量", file=sys.stderr)
        return 1
    if (args.place_negative_xy_min_m < 0.0
            or args.place_negative_xy_max_m < args.place_negative_xy_min_m
            or args.place_positive_x_max_m < args.place_negative_xy_min_m
            or args.place_negative_y_max_m < args.place_negative_xy_min_m):
        print("错误: 放置区域距离无效", file=sys.stderr)
        return 1
    if args.placement_visibility_margin_px < 0.0:
        print("错误: --placement-visibility-margin-px 不能小于 0", file=sys.stderr)
        return 1
    if not 0.0 < args.minimum_x_down_component <= 1.0 or args.tilt_search_deg < 0.0 or args.roll_search_deg < 0.0:
        print("错误: 姿态搜索范围或 +X 向下分量无效", file=sys.stderr)
        return 1
    try:
        centre_xy, red_yaw_deg, quadrant, table_plane_z, workpiece_item, calibration_path = load_workpiece(
            args.workpiece_pose_file, args.workpiece_index
        )
        if quadrant not in (1, 2, 3, 4):
            raise ValueError(f"工件朝向象限无效: Q{quadrant}")
        camera_matrix, distortion, base_camera, calibration_size = load_projection_calibration(calibration_path)
        roi_bounds = normalized_roi_bounds(list(args.placement_visibility_roi), *calibration_size)
        if (roi_bounds[2] - roi_bounds[0] <= 2.0 * args.placement_visibility_margin_px
                or roi_bounds[3] - roi_bounds[1] <= 2.0 * args.placement_visibility_margin_px):
            raise ValueError("放置可见 ROI 小于两侧安全边距，无法容纳工件")
        local_footprint_xy = footprint_local_base_xy(
            workpiece_item, centre_xy, table_plane_z, camera_matrix, distortion, base_camera
        )
        _, tcp_offset, correction_frame, correction = load_virtual_tcp_offset(RIGHT_FINGERTIP_CENTER_CONFIG, "right_hand_transfer_base")
        if correction_frame != args.frame_id:
            raise ValueError(f"TCP 校正坐标系为 {correction_frame!r}，规划坐标系为 {args.frame_id!r}")
        lowest_tcp = load_reference_tcp_pose(args.lowest_tcp_pose_file, tcp_offset, args.frame_id)
        if args.grasp_xy_offset_m is None:
            measured_actual_tcp = load_reference_tcp_pose(args.measured_actual_pose_file, tcp_offset, args.frame_id)
            measured_desired_tcp = load_reference_tcp_pose(args.measured_desired_pose_file, tcp_offset, args.frame_id)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    q1_target_yaws = args.q1_target_yaw_deg or [45.0, 40.0, 50.0, 35.0, 55.0, 30.0, 60.0, 25.0, 65.0, 20.0, 70.0, 15.0, 75.0]
    q1_target_yaws = [normalize_deg(value) for value in q1_target_yaws]
    if any(not 0.0 <= value < 90.0 for value in q1_target_yaws):
        print("错误: --q1-target-yaw-deg 必须位于 Q1 的 [0, 90) 度", file=sys.stderr)
        return 1

    vision_green_xy = centre_xy.copy()
    if args.grasp_xy_offset_m is None:
        measured_grasp_xy_offset = [
            measured_desired_tcp[axis] - measured_actual_tcp[axis] for axis in range(2)
        ]
        offset_source = f"test_2-test_1 ({args.measured_actual_pose_file.name} -> {args.measured_desired_pose_file.name})"
    else:
        measured_grasp_xy_offset = list(args.grasp_xy_offset_m)
        offset_source = "--grasp-xy-offset-m"
    # This script targets the fingertip-centre virtual TCP, so derive the
    # correction from the same TCP in test_2 - test_1.  Do not reuse the Q1
    # push workflow's independently measured compensation.
    centre_xy = [
        centre_xy[axis] + correction[axis] + measured_grasp_xy_offset[axis]
        for axis in range(2)
    ]
    lowest_z = lowest_tcp[2] + args.minimum_tcp_z_margin_m
    # The table-contact pose gives the virtual TCP's fixed clearance over the
    # calibrated support plane.  It is a collision limit, not a grasp target.
    tcp_table_clearance = lowest_tcp[2] - table_plane_z
    requested_grasp_z = (
        table_plane_z + tcp_table_clearance + args.workpiece_height_m + args.grasp_tcp_extra_clearance_m
    )
    grasp_z = max(requested_grasp_z, lowest_z)
    prepare_z = grasp_z + args.approach_height_m
    lift_z = grasp_z + args.lift_height_m
    if args.place_tcp_extra_clearance_m > args.lift_height_m:
        print("错误: --place-tcp-extra-clearance-m 不能高于 --lift-height-m", file=sys.stderr)
        return 1
    red_yaw_rad = math.radians(red_yaw_deg)
    red_direction = [math.cos(red_yaw_rad), math.sin(red_yaw_rad)]
    print(f"输入工件朝向=Q{quadrant}；目标将优先选择 Q1 中间 yaw（默认 45°）。")
    print(f"工件中心（已应用 TCP 及 test_2-test_1 实测校正）xy(m)={centre_xy}")
    print(f"重定向抓取 TCP 补偿 xy(m)={measured_grasp_xy_offset}（来源: {offset_source}）")
    print(
        "红线调姿点将在执行时按当前 TCP 投影选择："
        f"中心 tail 侧 {args.line_entry_clearance_m * 1000.0:.1f}--"
        f"{args.line_max_rear_distance_m * 1000.0:.1f} mm。"
    )
    print(f"红线 tail→head yaw={red_yaw_deg:.2f}°；绿线（工具 +Y、夹爪闭合线）yaw={normalize_deg(red_yaw_deg + 90.0):.2f}°")
    print(
        f"TCP 高度：台面 z={table_plane_z:.6f}，最低安全值={lowest_z:.6f}，"
        f"工件高度={args.workpiece_height_m:.6f}，准备={prepare_z:.6f}，"
        f"抓取={grasp_z:.6f}，标准放下={grasp_z + args.place_tcp_extra_clearance_m:.6f}，"
        f"抬起={lift_z:.6f} m"
    )
    if grasp_z != requested_grasp_z:
        print(f"抓取目标低于安全下限，已钳制到 {grasp_z:.6f} m。")
    if args.place_tcp_extra_clearance_m:
        print(
            f"放下/松爪将比抓取 TCP 高 {args.place_tcp_extra_clearance_m * 1000.0:.1f} mm，"
            f"目标 z={grasp_z + args.place_tcp_extra_clearance_m:.6f} m。"
        )
    if not args.execute:
        print(
            "翻转区位于 body_base 负 X/负 Y；下放区扩展至正 X/负 Y。"
            f"X 负向 {args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_xy_max_m * 1000.0:.1f} mm，正向 "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_positive_x_max_m * 1000.0:.1f} mm；Y 负向 "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_y_max_m * 1000.0:.1f} mm；"
            f"网格 {args.place_region_grid_step_m * 1000.0:.1f} mm。"
        )
        print(
            "放置 ROI 投影过滤："
            f"像素 [{roi_bounds[0]}, {roi_bounds[1]})-({roi_bounds[2]}, {roi_bounds[3]})，"
            f"边界余量 {args.placement_visibility_margin_px:.1f} px。"
        )
        print(f"Q1 目标红线 yaw 候选={q1_target_yaws}°。")
        print("预览：--execute 后将读取当前 TCP，筛选全段可规划的实时抓取姿态；未连接机械臂。")
        return 0

    arm = planner = gripper = None
    try:
        sdk = import_sdk()
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=args.planner_host)
        gripper = sdk.Gripper(host=args.gripper_host)
        group, ik_frame = "right_arm", "right_hand_transfer_base"
        current_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
        current_tcp = compose_pose(current_ik, tcp_offset)
        current_rpy = rotvec_to_rpy(current_tcp[3:])
        print(f"当前 TCP xyz(m)={[round(v, 6) for v in current_tcp[:3]]}；rpy(deg)={[round(math.degrees(v), 2) for v in current_rpy]}")
        current_line_projection = sum(
            (current_tcp[axis] - centre_xy[axis]) * red_direction[axis] for axis in range(2)
        )
        line_parameter = max(
            -args.line_max_rear_distance_m,
            min(current_line_projection, -args.line_entry_clearance_m),
        )
        line_alignment_xy = [
            centre_xy[axis] + line_parameter * red_direction[axis] for axis in range(2)
        ]
        print(
            f"红线调姿点 xy(m)={line_alignment_xy}；距中心 tail 侧 "
            f"{-line_parameter * 1000.0:.1f} mm（当前投影 {current_line_projection * 1000.0:.1f} mm）。"
        )
        line_transit_target = line_alignment_xy + [prepare_z] + current_tcp[3:]
        print("先检查红线调姿点的平移...", end="", flush=True)
        feasible, detail = can_plan_target(planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                                            ik_frame=ik_frame, tcp_offset=tcp_offset, target=line_transit_target,
                                            target_is_tcp=True, planner_kind=args.planner, max_vel=args.max_vel,
                                            max_acc=args.max_acc, timeout_s=args.planning_timeout_s)
        print("可达" if feasible else f"失败（{detail}）", flush=True)
        if not feasible:
            raise RuntimeError("无法先平移到红线调姿点；请缩小 --red-line-approach-distance-m。")
        candidates = attitude_candidates(current_rpy, red_yaw_deg, args.x_down_tilt_deg, args.tilt_search_deg,
                                         args.roll_search_deg, args.attitude_search_step_deg, args.minimum_x_down_component)
        if not candidates:
            raise RuntimeError("没有满足 +X 向下约束的姿态候选")
        if len(candidates) > args.max_attitude_candidates:
            candidates = candidates[:args.max_attitude_candidates]
        print(f"开始预规划 {len(candidates)} 个姿态候选；单次 RPC 最多等待 {args.planning_timeout_s:g} 秒。", flush=True)

        chosen: list[float] | None = None
        failures: list[str] = []
        for index, orientation in enumerate(candidates, start=1):
            rpy = rotvec_to_rpy(orientation)
            print(f"预规划候选 {index}/{len(candidates)}：rpy(deg)={[round(math.degrees(v), 2) for v in rpy]}", flush=True)
            checks = (
                ("红线点调姿", line_alignment_xy + [prepare_z] + orientation),
                ("沿红线到中心", centre_xy + [prepare_z] + orientation),
                ("下探", centre_xy + [grasp_z] + orientation),
                ("抬起", centre_xy + [lift_z] + orientation),
            )
            failed = None
            detail = ""
            for name, target in checks:
                print(f"  检查 {name}...", end="", flush=True)
                feasible, detail = can_plan_target(planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                                                    ik_frame=ik_frame, tcp_offset=tcp_offset, target=target,
                                                    target_is_tcp=True, planner_kind=args.planner,
                                                    max_vel=args.max_vel, max_acc=args.max_acc,
                                                    timeout_s=args.planning_timeout_s)
                print("可达" if feasible else f"失败（{detail}）", flush=True)
                if not feasible:
                    failed = name
                    break
            if failed is None:
                chosen = orientation
                print(
                    f"候选 {index}/{len(candidates)} 的抓取前四段可规划（{detail}）；"
                    f"rpy(deg)={[round(math.degrees(v), 2) for v in rpy]}。"
                )
                print("翻转与放下将在夹住并抬起后，基于实际 TCP 位姿逐段规划。")
                break
            if len(failures) < 8:
                failures.append(f"候选 {index} {failed}: {detail}")
        if chosen is None:
            raise RuntimeError("所有实时抓取候选都不可规划。前几个失败原因：" + "；".join(failures))

        chosen_rpy = rotvec_to_rpy(chosen)
        align_target = line_alignment_xy + [prepare_z] + chosen
        prepare_target = centre_xy + [prepare_z] + chosen
        grasp_target = centre_xy + [grasp_z] + chosen
        lift_target = centre_xy + [lift_z] + chosen
        set_gripper_state(gripper, arm_id=args.arm_id, target_open=True, timeout_s=args.gripper_timeout_s,
                          tolerance=args.gripper_position_tolerance)
        for stage, target in (
            ("阶段 1/8：保持当前姿态平移到红线 tail 侧调姿点", line_transit_target),
            ("阶段 2/8：固定红线调姿点，令 +Y 平行绿线", align_target),
            ("阶段 3/8：保持抓取姿态沿红线移动到工件中心上方", prepare_target),
            ("阶段 4/8：固定工件中心 XY 下探抓取", grasp_target),
        ):
            plan_and_follow(arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                            tcp_offset=tcp_offset, target=target, target_is_tcp=True, planner_kind=args.planner,
                            max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries, stage=stage)
        if args.calibrate_grasp_xy:
            actual_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
            actual_tcp = compose_pose(actual_ik, tcp_offset)
            actual_xy = actual_tcp[:2]
            actual_minus_target = [actual_xy[axis] - centre_xy[axis] for axis in range(2)]
            recommended_offset = [-value for value in actual_minus_target]
            record = {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "coordinate_frame": args.frame_id,
                "workpiece_pose_file": str(args.workpiece_pose_file),
                "workpiece_index": args.workpiece_index,
                "vision_green_center_xy_m": vision_green_xy,
                "commanded_tcp_target_xy_m": centre_xy,
                "measured_grasp_xy_offset_m": measured_grasp_xy_offset,
                "commanded_grasp_tcp_z_m": grasp_z,
                "actual_tcp_xy_m": actual_xy,
                "actual_minus_commanded_tcp_xy_m": actual_minus_target,
                "recommended_command_xy_offset_m": recommended_offset,
                "note": (
                    "The command target includes the measured closed-fingertip XY correction. "
                    "This is the planner-reported virtual TCP tracking error. "
                    "A near-zero value does not validate camera-to-physical-fingertip alignment."
                ),
            }
            output = args.calibration_record_file
            if output is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                output = DEFAULT_CALIBRATION_RECORD_DIR / f"grasp_xy_{stamp}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("校准模式：已下探到绿点，未闭爪、未翻转；机械臂保持当前位置。")
            print(f"视觉绿点 xy(m)={vision_green_xy}")
            print(f"已补偿的 TCP 命令目标 xy(m)={centre_xy}")
            print(f"规划器实际 TCP xy(m)={actual_xy}")
            print(f"实际 - TCP 命令目标 xy(m)={actual_minus_target}")
            print(f"记录已写入: {output}")
            return 0
        close_gripper_for_grasp(
            gripper, arm_id=args.arm_id, timeout_s=args.gripper_timeout_s,
            fully_closed_tolerance=args.gripper_position_tolerance,
            minimum_close_travel=args.grasp_minimum_close_travel,
            settle_s=args.grasp_contact_settle_s,
        )
        plan_and_follow(arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target=lift_target, target_is_tcp=True, planner_kind=args.planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, retries=args.start_sync_retries, stage="阶段 5：固定中心 XY 抬起")
        reoriented_rpy: list[float] | None = None
        selected_q1_yaw: float | None = None
        turn_station_xy: list[float] | None = None
        selected_trajectory: list | None = None
        selected_frequency: float | None = None
        # The region is intentionally expressed in body_base coordinates relative
        # to the grasp centre: both axes must be negative.  Try nearer cells first.
        x_region_distances: list[float] = []
        distance = args.place_negative_xy_min_m
        while distance <= args.place_negative_xy_max_m + 1e-9:
            x_region_distances.append(min(distance, args.place_negative_xy_max_m))
            distance += args.place_region_grid_step_m
        if not x_region_distances or x_region_distances[-1] < args.place_negative_xy_max_m - 1e-9:
            x_region_distances.append(args.place_negative_xy_max_m)
        x_positive_place_distances: list[float] = []
        distance = args.place_negative_xy_min_m
        while distance <= args.place_positive_x_max_m + 1e-9:
            x_positive_place_distances.append(min(distance, args.place_positive_x_max_m))
            distance += args.place_region_grid_step_m
        if (not x_positive_place_distances
                or x_positive_place_distances[-1] < args.place_positive_x_max_m - 1e-9):
            x_positive_place_distances.append(args.place_positive_x_max_m)
        y_region_distances: list[float] = []
        distance = args.place_negative_xy_min_m
        while distance <= args.place_negative_y_max_m + 1e-9:
            y_region_distances.append(min(distance, args.place_negative_y_max_m))
            distance += args.place_region_grid_step_m
        if not y_region_distances or y_region_distances[-1] < args.place_negative_y_max_m - 1e-9:
            y_region_distances.append(args.place_negative_y_max_m)
        turn_station_candidates = sorted(
            ([centre_xy[0] - dx, centre_xy[1] - dy]
             for dx in x_region_distances for dy in y_region_distances),
            key=lambda xy: (centre_xy[0] - xy[0]) ** 2 + (centre_xy[1] - xy[1]) ** 2,
        )
        print(
            f"阶段 6：在负 X/负 Y 翻转/放置区搜索 {len(turn_station_candidates)} 个候选，"
            f"X 距中心 {args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_xy_max_m * 1000.0:.1f} mm，Y "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_y_max_m * 1000.0:.1f} mm。"
        )
        turn_failures: list[str] = []
        # Preserve caller/default ordering: the default deliberately tries the
        # middle of Q1 first, rather than mechanically minimizing turn angle.
        q1_candidates = q1_target_yaws
        for station_xy in turn_station_candidates:
            for q1_yaw in q1_candidates:
                delta_deg = shortest_angle_delta_deg(red_yaw_deg, q1_yaw)
                candidate_rpy = [chosen_rpy[0], chosen_rpy[1], chosen_rpy[2] + math.radians(delta_deg)]
                final_turn_target = station_xy + [lift_z] + rpy_to_rotvec(candidate_rpy)
                print(f"规划整体后撤+转向：xy={station_xy}，Q1 yaw={q1_yaw:.2f}°（总转角={delta_deg:.2f}°）...", flush=True)
                try:
                    trajectory, frequency = preplan_tcp_trajectory(
                        planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target_tcp=final_turn_target, planner_kind=args.turn_planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                    )
                except Exception as exc:
                    turn_failures.append(f"xy={station_xy}, Q1 {q1_yaw:.2f}°: {exc}")
                    continue
                turn_station_xy = station_xy
                reoriented_rpy = candidate_rpy
                selected_q1_yaw = q1_yaw
                selected_trajectory = trajectory
                selected_frequency = frequency
                print(f"选择翻转/放置点 xy(m)={station_xy}；Q1 yaw={q1_yaw:.2f}°；已保存 {len(trajectory)} 个路点的整体轨迹。")
                break
            if turn_station_xy is not None:
                break
        if (reoriented_rpy is None or selected_q1_yaw is None
                or turn_station_xy is None or selected_trajectory is None or selected_frequency is None):
            raise RuntimeError("负 X/负 Y 翻转/放置区没有可规划的 Q1 朝向；" + "；".join(turn_failures[-12:]))
        print(
            f"阶段 6/7：直接执行已保存的整体后撤+转向轨迹，"
            f"Q1 yaw={selected_q1_yaw:.2f}°，{len(selected_trajectory)} 个关节路点。"
        )
        try:
            arm.follow_trajectory(
                arm_id=args.arm_id, trajectory=selected_trajectory, sample_frequency=selected_frequency,
            )
        except Exception as exc:
            raise RuntimeError(f"整体后撤+转向轨迹执行失败（不会改为分段重规划）: {exc}") from exc
        # The turn station and the placement point are deliberately separate
        # searches.  A station that can reach/lift/turn is not necessarily a
        # point from which the low placement pose is reachable.  Search the
        # expanded positive-or-negative-X/negative-Y placement region afresh
        # from the actual post-turn state before committing to the descent.
        place_candidates = sorted(
            ([centre_xy[0] + x_offset, centre_xy[1] - dy]
             for x_offset in ([-dx for dx in x_region_distances] + x_positive_place_distances)
             for dy in y_region_distances),
            key=lambda xy: (centre_xy[0] - xy[0]) ** 2 + (centre_xy[1] - xy[1]) ** 2,
        )
        yaw_delta_deg = shortest_angle_delta_deg(red_yaw_deg, selected_q1_yaw)
        visible_place_candidates: list[list[float]] = []
        for place_xy in place_candidates:
            projected_footprint = project_placement_footprint(
                place_xy, local_footprint_xy, yaw_delta_deg, table_plane_z,
                camera_matrix, distortion, base_camera,
            )
            if footprint_is_inside_roi(
                    projected_footprint, roi_bounds, args.placement_visibility_margin_px):
                visible_place_candidates.append(place_xy)
        filtered_count = len(place_candidates) - len(visible_place_candidates)
        place_candidates = visible_place_candidates
        print(
            f"阶段 8：翻转已完成；ROI 过滤掉 {filtered_count} 个候选，"
            f"在正/负 X、负 Y 区独立搜索剩余 {len(place_candidates)} 个下放候选；"
            f"X 负向 {args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_xy_max_m * 1000.0:.1f} mm、正向 "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_positive_x_max_m * 1000.0:.1f} mm，Y 负向 "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_y_max_m * 1000.0:.1f} mm。"
        )
        if not place_candidates:
            raise RuntimeError("下放操作无法规划，程序暂停")
        selected_place_xy: list[float] | None = None
        selected_place_trajectory: list | None = None
        selected_place_frequency: float | None = None
        selected_place_z: float | None = None
        place_failures: list[str] = []
        # Try the desired table-placement height first.  If it is not
        # reachable, try progressively higher targets so the workpiece is put
        # down as far as the planner can safely lower it instead of aborting
        # while still grasping it.
        standard_place_z = grasp_z + args.place_tcp_extra_clearance_m
        place_z_candidates = [standard_place_z]
        fallback_step_m = 0.010
        fallback_z = standard_place_z + fallback_step_m
        while fallback_z < lift_z - 1e-9:
            place_z_candidates.append(fallback_z)
            fallback_z += fallback_step_m
        for candidate_z in place_z_candidates:
            for place_xy in place_candidates:
                place_target = place_xy + [candidate_z] + rpy_to_rotvec(reoriented_rpy)
                height_note = "标准放下高度" if abs(candidate_z - standard_place_z) < 1e-9 else (
                    f"降级下放高度，比标准高 {(candidate_z - standard_place_z) * 1000.0:.1f} mm"
                )
                print(f"预规划下放候选：xy={place_xy}，{height_note}...", flush=True)
                try:
                    trajectory, frequency = preplan_tcp_trajectory(
                        planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target_tcp=place_target, planner_kind=args.planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                    )
                except Exception as exc:
                    place_failures.append(f"xy={place_xy}, z={candidate_z:.6f}: {exc}")
                    continue
                selected_place_xy = place_xy
                selected_place_trajectory = trajectory
                selected_place_frequency = frequency
                selected_place_z = candidate_z
                print(
                    f"选择可下放点 xy(m)={place_xy}，z={candidate_z:.6f}；"
                    f"已保存 {len(trajectory)} 个关节路点的下放轨迹。"
                )
                break
            if selected_place_xy is not None:
                break
        if (selected_place_xy is None or selected_place_trajectory is None
                or selected_place_frequency is None or selected_place_z is None):
            raise RuntimeError("下放操作无法规划，程序暂停")
        print(
            f"阶段 8：执行已保存的下放轨迹，放置 xy(m)={selected_place_xy}，z={selected_place_z:.6f}，"
            f"{len(selected_place_trajectory)} 个关节路点。"
        )
        try:
            arm.follow_trajectory(
                arm_id=args.arm_id, trajectory=selected_place_trajectory, sample_frequency=selected_place_frequency,
            )
        except Exception as exc:
            raise RuntimeError(f"下放轨迹执行失败：{exc}") from exc
        set_gripper_state(gripper, arm_id=args.arm_id, target_open=True, timeout_s=args.gripper_timeout_s,
                          tolerance=args.gripper_position_tolerance)
        # After release, do not ask the global planner to solve another IK
        # problem merely to clear the workpiece.  Lift straight up by a fixed
        # 20 mm from the measured release pose, holding its actual XY and
        # orientation.  The outer pipeline then plans the return to pose 3.
        released_pose = current_pose(arm, args.arm_id)
        direct_lift_m = POST_RELEASE_LIFT_M
        direct_lift_target = [
            released_pose[0], released_pose[1], released_pose[2] + direct_lift_m,
            *released_pose[3:],
        ]
        print(
            "阶段 9：开爪后直接沿 +Z 抬起（不经规划器） "
            f"{direct_lift_m * 1000.0:.1f} mm，目标 [xyz(m), rotvec(rad)]="
            f"{[round(value, 6) for value in direct_lift_target]}"
        )
        move_l(
            arm, args.arm_id, direct_lift_target,
            linear_speed_mps=0.05, angular_speed_radps=0.30,
            max_linear_accel_mps2=0.10, max_angular_accel_radps2=0.60,
            hz=50.0, position_tolerance_m=0.0005,
            orientation_tolerance_rad=0.035, settle_timeout_s=8.0,
        )
        print("工件重定向到 Q1 完成。请重新拍照、分割并生成 pose JSON，再执行后续 Q1 流程。")
        return 0
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        for client in (arm, planner, gripper):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
