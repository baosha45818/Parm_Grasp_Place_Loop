#!/usr/bin/env python3
"""Reorient a JSON-configured workpiece from Q2--Q4 into Q1.

No taught pose is used as a destination.  The planner reads the live native
right_eef TCP, reads one semantic 6-D workpiece pose from a fresh
mask_pose JSON, constructs a configured semantic grasp pose, generates
tolerant six-dimensional pregrasp candidates, and selects a plannable one.
The final approach follows the observed workpiece -Z axis.  Workpiece and
motion parameters come from ``--config``.

Run without ``--execute`` first.  Execution opens the gripper, grasps at the
configured semantic offset, lifts, turns the object towards Q1, places it in a
reachable body_base region, then releases.
Capture and segment again before the subsequent Q1 operation.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import time
import threading

from scipy.spatial.transform import Rotation

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

from move_by_eef_pose import compose_pose, inverse_pose, pose_from_lookup_transform
from move_by_joints import current_pose, finite, import_sdk, move_l, rotvec_to_quat, rotvec_to_rpy, rpy_to_rotvec
from grasp_motion_support import (
    can_plan_target as pilz_can_plan_target,
    ensure_ready,
    plan_and_follow as pilz_plan_and_follow,
    set_gripper_state,
)
from planning_log import append_event, planner_start_pose, planning_event


SCRIPT_DIR = Path(__file__).resolve().parent
# ``arm.get_state().state.right.eef_pose`` is the driver tool frame, while
# Module 7 targets are the physical jaw-centre frame ``right_eef``.  This fixed
# transform is expressed in the driver EEF's local frame; compose_pose rotates
# it into body_base for each live arm attitude.
RIGHT_EEF_FROM_TRANSFER_BASE = [0.274, 0.0, -0.05075, 0.0, 0.0, 0.0]
DEFAULT_OUTPUT_DIR = MODULE_ROOT.parents[1] / "output/module_7"
DEFAULT_CALIBRATION_RECORD_DIR = DEFAULT_OUTPUT_DIR / "reorient_q3_records"
RRTCONNECT_PLANNER_KIND = "rrtconnect"
RRTSTAR_PLANNER_KIND = "rrtstar"
RRTCONNECT_TIMEOUT_S = 30.0
# The 5604 RRTConnect profile has three verified cartesian workers.  Keep this
# process below that limit; a fourth client would only sit in the server queue.
RRTCONNECT_PLACE_MAX_PARALLELISM = 3
RRTCONNECT_INFLIGHT = threading.BoundedSemaphore(RRTCONNECT_PLACE_MAX_PARALLELISM)
CONFIG_SCHEMA = "reorient-workpiece-config/v2"
POSE_OUTPUT_ROOT = MODULE_ROOT.parents[1] / "output/module_4_2"
WORKPIECE_CONFIGS = {
    1: MODULE_ROOT / "configs/workpiece_1_reorient.json",
    2: MODULE_ROOT / "configs/workpiece_2_reorient.json",
}
TURN_CANDIDATE_CONFIGS = {
    1: MODULE_ROOT / "configs/workpiece_1_turn_candidates.toml",
    2: MODULE_ROOT / "configs/workpiece_2_turn_candidates.toml",
}


def newest_workpiece_pose_file(workpiece: int) -> Path:
    """Return the newest pose result from module 4.2 for one workpiece."""
    root = POSE_OUTPUT_ROOT / f"workpiece_{workpiece}"
    candidates = sorted(
        (path for path in root.glob("*/mask_pose.json") if path.parent.name.isdecimal()),
        key=lambda path: int(path.parent.name), reverse=True,
    )
    if not candidates:
        raise ValueError(
            f"工件 {workpiece} 尚无视觉结果；请先运行 module_4_2，"
            f"期望目录：{root}"
        )
    return candidates[0]


def validate_pose_workpiece(path: Path, workpiece: int) -> None:
    """Reject a pose JSON produced for a different configured workpiece."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取工件位姿 JSON: {path}: {exc}") from exc
    expected = {str(workpiece), f"workpiece_{workpiece}"}
    if payload.get("workpiece_asset") not in expected:
        raise ValueError(
            f"位姿文件属于工件 {payload.get('workpiece_asset')!r}，"
            f"与 --workpiece {workpiece} 不一致：{path}"
        )


def normalize_deg(value: float) -> float:
    return value % 360.0


def shortest_angle_delta_deg(start_deg: float, end_deg: float) -> float:
    """Signed base-Z turn from start to end in [-180, 180)."""
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def signed_offsets(limit_deg: float, step_deg: float) -> list[float]:
    values = [0.0]
    for index in range(1, int(math.floor(limit_deg / step_deg + 1e-9)) + 1):
        values.extend((index * step_deg, -index * step_deg))
    sampled_limit = (len(values) - 1) // 2 * step_deg
    if limit_deg > sampled_limit + 1e-9:
        values.extend((limit_deg, -limit_deg))
    return values


def prioritise_q1_yaws(yaws_deg: list[float], preferred_yaw_deg: float) -> list[float]:
    """Try the configured preferred Q1 yaw, then expand outwards."""
    unique_yaws = list(dict.fromkeys(normalize_deg(value) for value in yaws_deg))
    return sorted(unique_yaws, key=lambda value: (abs(value - preferred_yaw_deg), value))


def finite_matrix4(value: object, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise ValueError(f"{name} 必须是 4x4 矩阵")
    matrix = [[float(entry) for entry in row] for row in value]
    if any(not math.isfinite(entry) for row in matrix for entry in row):
        raise ValueError(f"{name} 包含非有限数")
    if any(abs(matrix[3][column] - expected) > 1e-6 for column, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f"{name} 不是齐次刚体变换")
    return matrix


def transform_point(transform: list[list[float]], point: list[float]) -> list[float]:
    if len(point) != 3:
        raise ValueError("局部点必须有三个坐标")
    return [sum(transform[row][column] * point[column] for column in range(3)) + transform[row][3] for row in range(3)]


def matrix4_to_pose(transform: list[list[float]]) -> list[float]:
    """Convert a rigid 4x4 matrix to the module's [xyz, rotvec] pose form."""
    return [transform[row][3] for row in range(3)] + rpy_to_rotvec(matrix_to_rpy([
        [transform[row][column] for column in range(3)] for row in range(3)
    ]))


def pose_heading_yaw_deg(pose: list[float]) -> float:
    """Return the body_base yaw of the pose's local +X axis."""
    rotation = rotation_matrix_from_rotvec(pose[3:])
    return normalize_deg(math.degrees(math.atan2(rotation[1][0], rotation[0][0])))


def workpiece_centre_pose(item: dict, semantic_pose: list[float]) -> list[float]:
    """Use Module 4-2's ICP/CAD centre, with the semantic-frame orientation."""
    value = item.get("pose_body_base")
    if not isinstance(value, dict):
        raise ValueError("工件 JSON 缺少 pose_body_base，无法建立工件中心刚性关系")
    centre = finite([value.get("x_m"), value.get("y_m"), value.get("z_m")], 3, "工件中心 xyz")
    return centre + semantic_pose[3:]


def normalize_vector(vector: list[float], name: str) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-9:
        raise ValueError(f"{name} 是零向量")
    return [value / norm for value in vector]


def load_workpiece(path: Path, index: int) -> tuple[list[list[float]], float, int, float, dict, Path]:
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
    transform = finite_matrix4(
        item.get("T_body_base_workpiece_semantic"), "T_body_base_workpiece_semantic"
    )
    workpiece_x = normalize_vector([transform[row][0] for row in range(3)], "工件 +X")
    if math.hypot(workpiece_x[0], workpiece_x[1]) < 1e-7:
        raise ValueError("工件 +X 几乎竖直，无法定义红线水平 yaw")
    yaw = math.degrees(math.atan2(workpiece_x[1], workpiece_x[0]))
    quadrant = item.get("head_direction_base_quadrant")
    if not isinstance(quadrant, int):
        raise ValueError("JSON 缺少整数 head_direction_base_quadrant")
    calibration_value = payload.get("calibration")
    if not isinstance(calibration_value, str) or not calibration_value:
        raise ValueError("工件 JSON 缺少 calibration；无法对放置点执行 ROI 投影过滤")
    calibration_path = Path(calibration_value)
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    return transform, normalize_deg(yaw), quadrant, plane_z, item, calibration_path


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


def cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def matrix_to_rpy(rotation: list[list[float]]) -> list[float]:
    """Convert a body-base rotation matrix to intrinsic ZYX roll/pitch/yaw."""
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0][1], rotation[1][1])
    return [roll, pitch, yaw]


def orientation_distance_rad(first_rotvec: list[float], second_rotvec: list[float]) -> float:
    """Shortest physical rotation angle between two orientation targets."""
    first_quat = rotvec_to_quat(first_rotvec)
    second_quat = rotvec_to_quat(second_rotvec)
    dot = abs(sum(first_quat[index] * second_quat[index] for index in range(4)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def rotation_matrix_from_rotvec(rotvec: list[float]) -> list[list[float]]:
    """Return the standard body-base rotation matrix for a rotation vector."""
    x, y, z, w = rotvec_to_quat(rotvec)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def angle_between_vectors_deg(left: list[float], right: list[float]) -> float:
    dot = sum(left[index] * right[index] for index in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def nominal_tcp_orientation(
    workpiece_y_base: list[float], workpiece_z_base: list[float], y_sign: int,
) -> list[float]:
    """Build TCP +X=-workpiece Z, +Y=±workpiece Y, +Z=±workpiece X."""
    tool_x = normalize_vector([-value for value in workpiece_z_base], "工件 -Z")
    raw_tool_y = [float(y_sign) * value for value in workpiece_y_base]
    x_dot_y = sum(tool_x[index] * raw_tool_y[index] for index in range(3))
    tool_y = normalize_vector(
        [raw_tool_y[index] - x_dot_y * tool_x[index] for index in range(3)],
        "与工件 +Z 正交的夹爪 +Y",
    )
    tool_z = normalize_vector(cross(tool_x, tool_y), "抓取姿态 +Z")
    return rpy_to_rotvec(matrix_to_rpy([
        [tool_x[0], tool_y[0], tool_z[0]],
        [tool_x[1], tool_y[1], tool_z[1]],
        [tool_x[2], tool_y[2], tool_z[2]],
    ]))


def attitude_candidates(
    workpiece_x_base: list[float], workpiece_y_base: list[float], workpiece_z_base: list[float], *,
    rpy_tolerance_deg: list[float], sample_step_deg: float,
    x_negative_z_tolerance_deg: float, y_parallel_y_tolerance_deg: float,
    z_parallel_x_tolerance_deg: float,
) -> list[tuple[list[float], float, float, float, int]]:
    """Return local-TCP RPY neighbourhoods around both symmetric TCP poses.

    The semantic axes are full 3-D directions. Tool +Y may be either +Y or
    -Y of the workpiece because the two jaws are symmetric. The nominal TCP
    axes are +X=-workpiece Z, +Y=±workpiece Y, and +Z=±workpiece X.
    Candidates must also meet the configured physical axis angular limits. The configured
    roll, pitch and yaw offsets are right-multiplied onto that nominal
    orientation, so they are rotations about the TCP's own local X/Y/Z axes.
    """
    candidates: list[tuple[list[float], float, float, int]] = []
    offset_sets = [signed_offsets(value, sample_step_deg) for value in rpy_tolerance_deg]
    desired_tool_x = normalize_vector([-value for value in workpiece_z_base], "工件 -Z")
    for sign in (1, -1):
        try:
            nominal = nominal_tcp_orientation(workpiece_y_base, workpiece_z_base, sign)
        except ValueError:
            continue
        desired_tool_z = normalize_vector(
            [float(sign) * value for value in workpiece_x_base], "工件 ±X"
        )
        desired_tool_y = normalize_vector(
            [float(sign) * value for value in workpiece_y_base], "工件 ±Y"
        )
        for roll_deg, pitch_deg, yaw_deg in product(*offset_sets):
            local_delta = rpy_to_rotvec([
                math.radians(roll_deg),
                math.radians(pitch_deg),
                math.radians(yaw_deg),
            ])
            rotvec = compose_pose(
                [0.0, 0.0, 0.0] + nominal,
                [0.0, 0.0, 0.0] + local_delta,
            )[3:]
            rotation = rotation_matrix_from_rotvec(rotvec)
            candidate_x = [rotation[row][0] for row in range(3)]
            candidate_y = [rotation[row][1] for row in range(3)]
            candidate_z = [rotation[row][2] for row in range(3)]
            x_error_deg = angle_between_vectors_deg(candidate_x, desired_tool_x)
            y_error_deg = angle_between_vectors_deg(candidate_y, desired_tool_y)
            z_error_deg = angle_between_vectors_deg(candidate_z, desired_tool_z)
            if (x_error_deg <= x_negative_z_tolerance_deg + 1e-9
                    and y_error_deg <= y_parallel_y_tolerance_deg + 1e-9
                    and z_error_deg <= z_parallel_x_tolerance_deg + 1e-9
                    and all(any(abs(a - b) > 1e-8 for a, b in zip(rotvec, known[0])) for known in candidates)):
                candidates.append((rotvec, x_error_deg, y_error_deg, z_error_deg, sign))
    return candidates


def servo_follow_tcp(
    arm: object, *, arm_id: int, tcp_target: list[float], servo_eef_offset: list[float], stage: str,
    linear_speed_mps: float, angular_speed_radps: float,
    max_linear_accel_mps2: float, max_angular_accel_radps2: float,
    hz: float, position_tolerance_m: float, orientation_tolerance_rad: float,
    settle_timeout_s: float,
) -> None:
    """Move the native right_eef target on a controller Cartesian line."""
    print(f"{stage}：servo 直线目标={[round(value, 6) for value in tcp_target]}")
    # ServoPose accepts a calibrated TCP. Keep the target itself in the native
    # right_eef frame, matching every planner target in this module.
    eef_target = tcp_target
    move_l(
        arm, arm_id, eef_target,
        linear_speed_mps=linear_speed_mps,
        angular_speed_radps=angular_speed_radps,
        max_linear_accel_mps2=max_linear_accel_mps2,
        max_angular_accel_radps2=max_angular_accel_radps2,
        hz=hz, position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
        settle_timeout_s=settle_timeout_s, tcp=servo_eef_offset,
    )


def actual_right_eef_from_driver(
    arm: object, arm_id: int, driver_to_right_eef: list[float],
) -> list[float]:
    """Return the live jaw-centre pose in body_base from the arm status stream."""
    return compose_pose(current_pose(arm, arm_id), driver_to_right_eef)


def require_driver_right_eef_at_target(
    arm: object, *, arm_id: int, driver_to_right_eef: list[float], target: list[float],
    position_tolerance_m: float, orientation_tolerance_rad: float, timeout_s: float, stage: str,
) -> list[float]:
    """Refuse to start servo until the driver confirms right_eef is at its expected start pose."""
    deadline = time.monotonic() + timeout_s
    last_pose: list[float] | None = None
    while time.monotonic() < deadline:
        last_pose = actual_right_eef_from_driver(arm, arm_id, driver_to_right_eef)
        position_error = math.sqrt(sum((last_pose[axis] - target[axis]) ** 2 for axis in range(3)))
        orientation_error = orientation_distance_rad(last_pose[3:], target[3:])
        if position_error <= position_tolerance_m and orientation_error <= orientation_tolerance_rad:
            print(
                f"{stage}：驱动状态确认 right_eef 已到预抓取点；"
                f"位置误差={position_error * 1000.0:.2f} mm，"
                f"姿态误差={math.degrees(orientation_error):.2f}°。"
            )
            return last_pose
        time.sleep(0.05)
    assert last_pose is not None
    position_error = math.sqrt(sum((last_pose[axis] - target[axis]) ** 2 for axis in range(3)))
    orientation_error = orientation_distance_rad(last_pose[3:], target[3:])
    raise RuntimeError(
        f"{stage}：驱动状态的 right_eef 未到预抓取点，禁止 servo 下探；"
        f"实际 xyz={[round(value, 6) for value in last_pose[:3]]}，"
        f"目标 xyz={[round(value, 6) for value in target[:3]]}，"
        f"位置误差={position_error * 1000.0:.1f} mm，"
        f"姿态误差={math.degrees(orientation_error):.2f}°。"
    )


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


def rrtconnect_plan_tcp_trajectory(
    planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str,
    tcp_offset: list[float], target: list[float], target_is_tcp: bool,
    max_vel: float, max_acc: float, timeout_s: float,
    stage: str = "reorient_rrtconnect_plan",
) -> tuple[list, float]:
    """Plan a TCP goal through the RRTConnect cart+ompl private SDK payload."""
    target_for_ik = compose_pose(target, inverse_pose(tcp_offset)) if target_is_tcp else target
    start = planner_start_pose(planner, frame_id, ik_frame)
    started = time.perf_counter()
    target_record = {
        "frame_id": frame_id, "ik_frame": ik_frame,
        "tcp_pose_xyz_rotvec": target if target_is_tcp else None,
        "ik_pose_xyz_rotvec": target_for_ik,
    }
    payload = {
        "waypoints": [{
            "name": "workpiece_reorient_rrtconnect_target",
            "type": "cart",
            "group": group,
            "planner": "ompl",
            "max_velocity_scaling_factor": max_vel,
            "max_acceleration_scaling_factor": max_acc,
            "frame_id": frame_id,
            "ik_frame": ik_frame,
            "position": target_for_ik[:3],
            "orientation": rotvec_to_quat(target_for_ik[3:]),
        }],
        "topp": {
            "max_vel_factor": max_vel,
            "max_acc_factor": max_acc,
            "final_phase_ratio": 0.05,
            "final_phase_oversample": 2.0,
        },
    }
    try:
        result = planner._trajectory.send_payload(payload, timeout=max(timeout_s, RRTCONNECT_TIMEOUT_S))
        data = result.get("data") or {}
        trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
        frequency = data.get("sample_frequency", 10.0)
        if planned_arm_id != arm_id:
            raise RuntimeError(f"RRTConnect 返回 arm_id={planned_arm_id!r}，与请求不一致")
        if not isinstance(trajectory, list) or not trajectory:
            raise RuntimeError("RRTConnect 未返回有效 trajectory")
        if not isinstance(frequency, (int, float)) or frequency <= 0.0:
            raise RuntimeError(f"RRTConnect 返回的 sample_frequency 无效: {frequency!r}")
    except Exception as exc:
        planning_event(stage=stage, method="ompl/RRTConnect", start=start,
                       target=target_record, started_at=started, success=False,
                       failure_reason=f"{type(exc).__name__}: {exc}", arm_id=arm_id)
        raise
    planning_event(stage=stage, method="ompl/RRTConnect", start=start,
                   target=target_record, started_at=started, success=True,
                   arm_id=arm_id, trajectory_waypoints=len(trajectory),
                   sample_frequency_hz=float(frequency))
    return trajectory, float(frequency)


def can_plan_target(
    planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str,
    tcp_offset: list[float], target: list[float], target_is_tcp: bool,
    planner_kind: str, max_vel: float, max_acc: float, timeout_s: float = 5.0,
) -> tuple[bool, str]:
    if planner_kind == RRTSTAR_PLANNER_KIND:
        if not target_is_tcp:
            return False, "RRTstar 精确目标检查只接受 TCP 目标"
        try:
            trajectory, _ = preplan_tcp_trajectory(
                planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target_tcp=target, planner_kind=RRTSTAR_PLANNER_KIND,
                max_vel=max_vel, max_acc=max_acc, timeout_s=timeout_s,
                stage="reorient RRTstar exact feasibility check",
            )
        except Exception as exc:
            return False, f"RRTstar 精确目标: {exc}"
        return True, f"RRTstar 精确目标，{len(trajectory)} 个关节路点"
    if planner_kind != RRTCONNECT_PLANNER_KIND:
        return pilz_can_plan_target(
            planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
            tcp_offset=tcp_offset, target=target, target_is_tcp=target_is_tcp,
            planner_kind=planner_kind, max_vel=max_vel, max_acc=max_acc, timeout_s=timeout_s,
        )
    try:
        trajectory, _ = rrtconnect_plan_tcp_trajectory(
            planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
            tcp_offset=tcp_offset, target=target, target_is_tcp=target_is_tcp,
            max_vel=max_vel, max_acc=max_acc, timeout_s=timeout_s,
            stage="reorient feasibility check",
        )
    except Exception as exc:
        return False, f"RRTConnect: {exc}"
    return True, f"RRTConnect {len(trajectory)} 个关节路点"


def plan_and_follow(
    arm: object, planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str,
    tcp_offset: list[float], target: list[float], target_is_tcp: bool, planner_kind: str,
    max_vel: float, max_acc: float, retries: int, stage: str, execution_speed_scale: float,
) -> None:
    if planner_kind == RRTSTAR_PLANNER_KIND:
        if not target_is_tcp:
            raise RuntimeError(f"{stage} RRTstar 精确规划只接受 TCP 目标")
        print(f"{stage} RRTstar 精确 TCP target [xyz(m), rotvec(rad)]: "
              f"{[round(value, 6) for value in target]}")
        try:
            trajectory, frequency = preplan_tcp_trajectory(
                planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
                tcp_offset=tcp_offset, target_tcp=target, planner_kind=RRTSTAR_PLANNER_KIND,
                max_vel=max_vel, max_acc=max_acc, timeout_s=RRTCONNECT_TIMEOUT_S,
                stage=stage,
            )
            arm.follow_trajectory(arm_id=arm_id, trajectory=trajectory,
                                  sample_frequency=frequency * execution_speed_scale)
        except Exception as exc:
            raise RuntimeError(f"{stage} RRTstar 精确规划或执行失败: {exc}") from exc
        return
    if planner_kind != RRTCONNECT_PLANNER_KIND:
        return pilz_plan_and_follow(
            arm, planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
            tcp_offset=tcp_offset, target=target, target_is_tcp=target_is_tcp,
            planner_kind=planner_kind, max_vel=max_vel, max_acc=max_acc, retries=retries, stage=stage,
        )
    print(f"{stage} RRTConnect TCP target [xyz(m), rotvec(rad)]: {[round(value, 6) for value in target]}")
    try:
        trajectory, frequency = rrtconnect_plan_tcp_trajectory(
            planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
            tcp_offset=tcp_offset, target=target, target_is_tcp=target_is_tcp,
            max_vel=max_vel, max_acc=max_acc, timeout_s=RRTCONNECT_TIMEOUT_S,
            stage=stage,
        )
        arm.follow_trajectory(arm_id=arm_id, trajectory=trajectory,
                              sample_frequency=frequency * execution_speed_scale)
    except Exception as exc:
        raise RuntimeError(f"{stage} RRTConnect 规划或执行失败: {exc}") from exc


def preplan_tcp_trajectory(
    planner: object, *, arm_id: int, group: str, frame_id: str, ik_frame: str,
    tcp_offset: list[float], target_tcp: list[float], planner_kind: str,
    max_vel: float, max_acc: float, timeout_s: float,
    stage: str = "reorient_preplan", rrtstar_goal_tolerance: bool | None = None,
) -> tuple[list, float]:
    """Plan once and retain the exact trajectory for immediate execution."""
    if planner_kind == RRTCONNECT_PLANNER_KIND:
        return rrtconnect_plan_tcp_trajectory(
            planner, arm_id=arm_id, group=group, frame_id=frame_id, ik_frame=ik_frame,
            tcp_offset=tcp_offset, target=target_tcp, target_is_tcp=True,
            max_vel=max_vel, max_acc=max_acc, timeout_s=timeout_s,
            stage=stage,
        )
    planner_target = compose_pose(target_tcp, inverse_pose(tcp_offset))
    start = planner_start_pose(planner, frame_id, ik_frame)
    started = time.perf_counter()
    target_record = {"frame_id": frame_id, "ik_frame": ik_frame,
                     "tcp_pose_xyz_rotvec": target_tcp,
                     "ik_pose_xyz_rotvec": planner_target}
    try:
        request = dict(
            group=group, planner=planner_kind, frame_id=frame_id, ik_frame=ik_frame,
            position=planner_target[:3], orientation=rotvec_to_quat(planner_target[3:]),
            max_vel=max_vel, max_acc=max_acc, timeout=timeout_s,
        )
        # Omit this field unless a caller explicitly asks for tolerant RRTstar.
        # The planner default is exact Cartesian goal matching.
        if planner_kind == RRTSTAR_PLANNER_KIND and rrtstar_goal_tolerance is True:
            request["rrtstar_goal_tolerance"] = True
        result = planner.plan_cart_trajectory(**request)
        data = result.get("data") or {}
        trajectory, planned_arm_id = data.get("trajectory"), data.get("arm_id")
        frequency = data.get("sample_frequency", 10.0)
        if planned_arm_id != arm_id:
            raise RuntimeError(f"规划结果 arm_id={planned_arm_id!r}，与请求不一致")
        if not isinstance(trajectory, list) or not trajectory:
            raise RuntimeError("规划器未返回有效 trajectory")
        if not isinstance(frequency, (int, float)) or frequency <= 0.0:
            raise RuntimeError(f"规划器返回的 sample_frequency 无效: {frequency!r}")
    except Exception as exc:
        planning_event(stage=stage, method=planner_kind, start=start,
                       target=target_record, started_at=started, success=False,
                       failure_reason=f"{type(exc).__name__}: {exc}", arm_id=arm_id)
        raise
    planning_event(stage=stage, method=planner_kind, start=start,
                   target=target_record, started_at=started, success=True,
                   arm_id=arm_id, trajectory_waypoints=len(trajectory),
                   sample_frequency_hz=float(frequency))
    return trajectory, float(frequency)


def rrtconnect_plan_tcp_batch(
    sdk: object, *, planner_host: str, arm_id: int, group: str, frame_id: str,
    ik_frame: str, tcp_offset: list[float], jobs: list[tuple[int, list[float], float, list[float]]],
    max_vel: float, max_acc: float, timeout_s: float, parallelism: int,
) -> list[tuple[int, list[float], float, list | None, float | None, str | None]]:
    """Plan one priority batch with independent SDK connections per RRT worker.

    A Planner connection serializes its own requests, so sharing the main
    connection here would not create real parallel RRT searches.  Every worker
    therefore owns and closes a separate Planner client.
    """
    def plan_one(job: tuple[int, list[float], float, list[float]]) -> tuple[int, list[float], float, list | None, float | None, str | None]:
        priority, place_xy, candidate_z, target_tcp = job
        worker_planner = None
        try:
            with RRTCONNECT_INFLIGHT:
                worker_planner = sdk.Planner(host=planner_host)
                trajectory, frequency = rrtconnect_plan_tcp_trajectory(
                    worker_planner, arm_id=arm_id, group=group, frame_id=frame_id,
                    ik_frame=ik_frame, tcp_offset=tcp_offset, target=target_tcp,
                    target_is_tcp=True, max_vel=max_vel, max_acc=max_acc,
                    timeout_s=timeout_s,
                )
            return priority, place_xy, candidate_z, trajectory, frequency, None
        except Exception as exc:
            return priority, place_xy, candidate_z, None, None, str(exc)
        finally:
            if worker_planner is not None:
                try:
                    worker_planner.close()
                except Exception:
                    pass

    results: list[tuple[int, list[float], float, list | None, float | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=min(parallelism, len(jobs))) as executor:
        futures = [executor.submit(plan_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result[0])


CONFIGURABLE_OPTIONS = frozenset({
    "--grasp-point-semantic-offset-m", "--pregrasp-positive-workpiece-z-m",
    "--final-extra-descent-negative-workpiece-z-m",
    "--pregrasp-position-tolerance-m", "--pregrasp-rpy-tolerance-deg",
    "--tcp-x-negative-workpiece-z-tolerance-deg",
    "--tcp-y-parallel-workpiece-y-tolerance-deg",
    "--tcp-z-parallel-workpiece-x-tolerance-deg",
    "--servo-linear-speed-mps", "--servo-angular-speed-radps",
    "--servo-max-linear-accel-mps2", "--servo-max-angular-accel-radps2",
    "--servo-hz", "--servo-position-tolerance-m",
    "--servo-orientation-tolerance-rad", "--servo-settle-timeout-s",
    "--place-tcp-extra-clearance-m", "--lift-height-m", "--post-release-lift-height-m",
    "--attitude-search-step-deg",
    "--planning-timeout-s", "--q1-yaw-min-deg", "--q1-yaw-max-deg",
    "--q1-yaw-search-step-deg", "--q1-preferred-yaw-deg",
    "--turn-roll-tolerance-deg", "--turn-pitch-tolerance-deg",
    "--turn-attitude-search-step-deg",
    "--turn-relocation-step-m", "--post-grasp-turn-search-negative-base-x-m",
    "--post-grasp-turn-search-negative-base-y-m", "--turn-station-tcp-height-m",
    "--turn-search-grid-step-m", "--place-negative-xy-min-m",
    "--place-negative-xy-max-m", "--place-positive-x-max-m",
    "--turn-positive-x-min-m", "--turn-positive-x-max-m",
    "--place-negative-y-max-m", "--place-region-grid-step-m",
    "--place-search-grid-step-m", "--rrt-place-parallelism",
    "--rrt-turn-parallelism", "--placement-visibility-roi",
    "--placement-visibility-margin-px", "--planner", "--turn-planner",
    "--max-vel", "--max-acc", "--start-sync-retries",
    "--gripper-timeout-s", "--gripper-position-tolerance",
    "--grasp-minimum-close-travel", "--grasp-contact-settle-s",
    "--trajectory-execution-speed-scale",
})
def load_reorient_config(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取调整配置 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("调整配置顶层必须是 JSON 对象")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"配置 schema 必须是 {CONFIG_SCHEMA!r}")
    if not isinstance(payload.get("reorient_arguments"), dict):
        raise ValueError("配置缺少对象 reorient_arguments")
    return payload


def apply_reorient_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    payload = load_reorient_config(args.config)
    configured = payload["reorient_arguments"]
    assert isinstance(configured, dict)
    missing = sorted(CONFIGURABLE_OPTIONS - configured.keys())
    if missing:
        raise ValueError(f"配置缺少调整参数: {', '.join(missing)}")
    by_option = {
        option: action for action in parser._actions for option in action.option_strings
    }
    for option, raw_value in configured.items():
        if option not in CONFIGURABLE_OPTIONS or option not in by_option:
            raise ValueError(f"配置包含未知调整参数: {option!r}")
        action = by_option[option]
        try:
            if action.nargs in ("+", "*") or isinstance(action.nargs, int):
                if not isinstance(raw_value, list) or not raw_value:
                    raise ValueError("必须是非空数组")
                if isinstance(action.nargs, int) and len(raw_value) != action.nargs:
                    raise ValueError(f"必须包含 {action.nargs} 项")
                value = [action.type(str(item)) if action.type else item for item in raw_value]
            else:
                value = action.type(str(raw_value)) if action.type else raw_value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"配置参数 {option} 的值无效: {raw_value!r} ({exc})") from exc
        if action.choices is not None:
            values = value if isinstance(value, list) else [value]
            if any(item not in action.choices for item in values):
                raise ValueError(f"配置参数 {option} 不在允许范围内: {value!r}")
        setattr(args, action.dest, value)
    workpiece = payload.get("workpiece", args.config.stem)
    if not isinstance(workpiece, str) or not workpiece.strip():
        raise ValueError("配置参数 workpiece 必须是非空字符串")
    args.workpiece_name = workpiece
    print(f"加载调整配置: {workpiece} ({args.config})")


def load_turn_candidates(path: Path) -> dict[str, object]:
    """Load one workpiece's editable closed-grasp adjustment candidates.

    Python 3.10 is intentionally kept for this robot environment, so this
    parser supports only the TOML syntax used by our editable parameter files:
    one [turn] table and numeric arrays, including multiline nested arrays.
    """
    table: dict[str, object] = {}
    in_turn_table = False
    pending_key: str | None = None
    pending_value: list[str] = []

    def store(key: str, raw: str, line_number: int) -> None:
        if key in table:
            raise ValueError(f"翻转候选配置键重复: {path}:{line_number}: {key}")
        try:
            table[key] = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"翻转候选 TOML 无法解析: {path}:{line_number}: {raw!r}") from exc

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if pending_key is not None:
            pending_value.append(line)
            joined = " ".join(pending_value)
            if joined.count("[") == joined.count("]"):
                store(pending_key, joined, line_number)
                pending_key = None
                pending_value = []
            continue
        if line.startswith("[") and line.endswith("]"):
            in_turn_table = line == "[turn]"
            if not in_turn_table:
                raise ValueError(f"翻转候选配置只支持 [turn]: {path}:{line_number}")
            continue
        if not in_turn_table or "=" not in line:
            raise ValueError(f"翻转候选配置只支持 [turn] 内的 key = value: {path}:{line_number}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise ValueError(f"翻转候选配置键为空: {path}:{line_number}")
        if raw_value.count("[") != raw_value.count("]"):
            pending_key = key
            pending_value = [raw_value]
        else:
            store(key, raw_value, line_number)
    if pending_key is not None:
        raise ValueError(f"翻转候选配置数组未闭合: {path}")

    def vector(name: str, length: int | None = None) -> list[float]:
        raw = table.get(name)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"翻转候选配置 {name} 必须是非空数值数组")
        if length is not None and len(raw) != length:
            raise ValueError(f"翻转候选配置 {name} 必须恰好有 {length} 项")
        try:
            values = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"翻转候选配置 {name} 必须全为数值") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"翻转候选配置 {name} 含非有限数")
        return values

    planner = table.get("planner")
    if planner not in {"rrtconnect", "rrtstar"}:
        raise ValueError("翻转候选配置 planner 必须是 'rrtconnect' 或 'rrtstar'")
    values: dict[str, object] = {
        "planner": planner,
        "front_base_rpy_deg": vector("front_base_rpy_deg", 3),
        "x_offset_candidates_m": vector("x_offset_candidates_m"),
        "y_offset_candidates_m": vector("y_offset_candidates_m"),
        "z_offset_candidates_m": vector("z_offset_candidates_m"),
        "roll_candidates_deg": vector("roll_candidates_deg"),
        "pitch_candidates_deg": vector("pitch_candidates_deg"),
        "yaw_candidates_deg": vector("yaw_candidates_deg"),
    }
    if any(abs(value) > 360.0 for name, candidate in values.items()
           if name.endswith("_deg") for value in candidate):
        raise ValueError("翻转候选 RPY 不能超过 360 度")
    if any(abs(value) > 0.5 for name, candidate in values.items()
           if name.endswith("_m") for value in candidate):
        raise ValueError("翻转候选工件中心 XYZ 增量不能超过 0.5 m")
    return values


def record_turn_candidate_statistics(
    *, workpiece: int, config_path: Path, details_by_priority: dict[int, dict[str, object]],
    outcomes: dict[int, bool], selected_priority: int | None,
) -> Path:
    """Persist aggregate planning outcomes for one workpiece's turn candidates.

    ``attempts`` counts submitted plans, ``plannable`` counts plans that returned
    a trajectory, and ``selected`` counts candidates actually chosen for execution.
    Candidate identity intentionally excludes the live object position: it is the
    editable TOML combination (centre offset and local RPY).
    """
    statistics_path = DEFAULT_OUTPUT_DIR / f"workpiece_{workpiece}_turn_candidate_statistics.json"
    try:
        payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = {
            "schema": "module-7-turn-candidate-statistics/v1",
            "workpiece": workpiece,
            "runs": 0,
            "candidates": {},
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"无法读取翻转候选统计文件 {statistics_path}: {exc}"
        ) from exc
    if (not isinstance(payload, dict) or payload.get("workpiece") != workpiece
            or not isinstance(payload.get("candidates"), dict)):
        raise RuntimeError(f"翻转候选统计文件格式无效: {statistics_path}")

    payload["runs"] = int(payload.get("runs", 0)) + 1
    payload["turn_candidates_config"] = str(config_path)
    candidates = payload["candidates"]

    for priority, plannable in outcomes.items():
        detail = details_by_priority[priority]
        centre_offset = detail["centre_offset_m"]
        local_rpy = detail["local_rpy_deg"]
        if not isinstance(centre_offset, list) or not isinstance(local_rpy, list):
            raise RuntimeError("翻转候选详情格式无效")
        identity = json.dumps(
            {"centre_offset_m": centre_offset, "local_rpy_deg": local_rpy},
            separators=(",", ":"), sort_keys=True,
        )
        record = candidates.setdefault(identity, {
            "centre_offset_m": centre_offset,
            "local_rpy_deg": local_rpy,
            "attempts": 0,
            "plannable": 0,
            "selected": 0,
        })
        if not isinstance(record, dict):
            raise RuntimeError(f"翻转候选统计记录格式无效: {identity}")
        record["attempts"] = int(record.get("attempts", 0)) + 1
        if plannable:
            record["plannable"] = int(record.get("plannable", 0)) + 1
        if priority == selected_priority:
            record["selected"] = int(record.get("selected", 0)) + 1

    payload["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = statistics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, statistics_path)
    return statistics_path


def append_turn_planning_records(
    *, run_id: str, workpiece: int, config_path: Path, pose_file: Path, workpiece_index: int,
    planner_kind: str, details_by_priority: dict[int, dict[str, object]], outcomes: dict[int, bool],
    failure_reasons: dict[int, str], selected_priority: int | None,
) -> Path:
    """Append one analysis record for every candidate actually sent to a planner."""
    history_path = DEFAULT_OUTPUT_DIR / f"workpiece_{workpiece}_turn_planning_records.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(timezone.utc).isoformat()
    with history_path.open("a", encoding="utf-8") as stream:
        for priority, plannable in outcomes.items():
            detail = details_by_priority[priority]
            record = {
                "schema": "module-7-turn-planning-record/v1",
                "recorded_at_utc": recorded_at,
                "planning_run_id": run_id,
                "workpiece": workpiece,
                "turn_candidates_config": str(config_path),
                "workpiece_pose_file": str(pose_file),
                "workpiece_index": workpiece_index,
                "planner": planner_kind,
                "candidate_priority": priority + 1,
                "centre_offset_m": detail["centre_offset_m"],
                "local_rpy_deg": detail["local_rpy_deg"],
                "plannable": plannable,
                "selected": priority == selected_priority,
                "failure_reason": failure_reasons.get(priority),
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return history_path


def execute_workpiece_1_legacy_turn(
    *, args: argparse.Namespace, sdk: object, arm: object, planner: object,
    group: str, ik_frame: str, tcp_offset: list[float], lifted_tcp: list[float],
    chosen_rpy: list[float], red_yaw_deg: float, grasp_z: float,
) -> tuple[list[float], float]:
    """Run Workpiece 1's original post-lift relocation and Q1 turn search."""
    requested_xy = [
        args.post_grasp_turn_search_negative_base_x_m,
        args.post_grasp_turn_search_negative_base_y_m,
    ]
    requested_distance = math.hypot(*requested_xy)
    relocation_distances: list[float] = []
    if requested_distance > 1e-9:
        distance = requested_distance
        while distance > 1e-9:
            relocation_distances.append(distance)
            distance -= args.turn_relocation_step_m
    selected_relocation: tuple[list, float] | None = None
    failures: list[str] = []
    print(
        "阶段 4b：从实际抬起 TCP 沿 body_base -X/-Y 尽量后撤到翻转搜索原点；"
        f"最大偏移=−{requested_xy[0] * 1000.0:.1f}/−{requested_xy[1] * 1000.0:.1f} mm，"
        f"降级步长={args.turn_relocation_step_m * 1000.0:.1f} mm。"
    )
    for distance in relocation_distances:
        scale = distance / requested_distance
        target = [
            lifted_tcp[0] - scale * requested_xy[0],
            lifted_tcp[1] - scale * requested_xy[1], lifted_tcp[2], *lifted_tcp[3:],
        ]
        try:
            trajectory, frequency = preplan_tcp_trajectory(
                planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                ik_frame=ik_frame, tcp_offset=tcp_offset, target_tcp=target,
                planner_kind=args.planner, max_vel=args.max_vel, max_acc=args.max_acc,
                timeout_s=args.planning_timeout_s,
                stage="阶段 4b：抬起后后撤到翻转搜索原点",
            )
        except Exception as exc:
            failures.append(f"{distance * 1000.0:.1f} mm: {exc}")
            continue
        selected_relocation = (trajectory, frequency)
        print(f"阶段 4b：可规划；采用后撤 {distance * 1000.0:.1f} mm。")
        break
    if selected_relocation is not None:
        try:
            arm.follow_trajectory(
                arm_id=args.arm_id, trajectory=selected_relocation[0], sample_frequency=selected_relocation[1],
            )
        except Exception as exc:
            raise RuntimeError(f"阶段 4b 后撤轨迹执行失败: {exc}") from exc
    elif relocation_distances:
        print("阶段 4b：所有后撤候选均不可规划；保持抬起后的当前位置，直接进入翻转搜索。"
              + (f" 最近失败：{failures[-1]}" if failures else ""))

    actual_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
    origin_tcp = compose_pose(actual_ik, tcp_offset)
    origin_xy = origin_tcp[:2].copy()
    turn_z = grasp_z + args.turn_station_tcp_height_m

    def distances(minimum: float, maximum: float) -> list[float]:
        values: list[float] = []
        value = minimum
        while value <= maximum + 1e-9:
            values.append(min(value, maximum))
            value += args.turn_search_grid_step_m
        if not values or values[-1] < maximum - 1e-9:
            values.append(maximum)
        return values

    negative_x = distances(args.place_negative_xy_min_m, args.place_negative_xy_max_m)
    positive_x = distances(max(args.turn_positive_x_min_m, args.turn_search_grid_step_m), args.turn_positive_x_max_m)
    negative_y = distances(args.place_negative_xy_min_m, args.place_negative_y_max_m)
    stations = sorted(
        ([origin_xy[0] + x_offset, origin_xy[1] - dy]
         for x_offset in ([-value for value in negative_x] + positive_x)
         for dy in negative_y),
        key=lambda xy: (xy[1], abs(origin_xy[0] - xy[0])),
    )
    q1_count = int(math.floor((args.q1_yaw_max_deg - args.q1_yaw_min_deg) / args.q1_yaw_search_step_deg + 1e-9))
    q1_yaws = [args.q1_yaw_min_deg + index * args.q1_yaw_search_step_deg for index in range(q1_count + 1)]
    if q1_yaws[-1] < args.q1_yaw_max_deg - 1e-9:
        q1_yaws.append(args.q1_yaw_max_deg)
    q1_yaws = prioritise_q1_yaws(q1_yaws + [args.q1_preferred_yaw_deg], args.q1_preferred_yaw_deg)
    attitude_offsets = list(product(
        signed_offsets(args.turn_roll_tolerance_deg, args.turn_attitude_search_step_deg),
        signed_offsets(args.turn_pitch_tolerance_deg, args.turn_attitude_search_step_deg),
    ))
    attitude_offsets.sort(key=lambda item: (max(abs(item[0]), abs(item[1])), abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1]), item[0], item[1]))
    jobs: list[tuple[int, list[float], float, list[float]]] = []
    rpy_by_priority: dict[int, list[float]] = {}
    priority = 0
    for yaw in q1_yaws:
        delta = shortest_angle_delta_deg(red_yaw_deg, yaw)
        for roll, pitch in attitude_offsets:
            rpy = [chosen_rpy[0] + math.radians(roll), chosen_rpy[1] + math.radians(pitch), chosen_rpy[2] + math.radians(delta)]
            for station in stations:
                jobs.append((priority, station, yaw, station + [turn_z] + rpy_to_rotvec(rpy)))
                rpy_by_priority[priority] = rpy
                priority += 1
    failures = []
    selected: tuple[list[float], float, list, float] | None = None
    for batch_start in range(0, len(jobs), args.rrt_turn_parallelism):
        batch = jobs[batch_start:batch_start + args.rrt_turn_parallelism]
        if args.turn_planner == RRTCONNECT_PLANNER_KIND:
            results = rrtconnect_plan_tcp_batch(
                sdk, planner_host=args.planner_host, arm_id=args.arm_id, group=group,
                frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, jobs=batch,
                max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                parallelism=args.rrt_turn_parallelism,
            )
            for job_priority, _station, yaw, trajectory, frequency, failure in results:
                if failure is None:
                    selected = (rpy_by_priority[job_priority], yaw, trajectory, frequency)
                    break
                failures.append(str(failure))
        else:
            for job_priority, _station, yaw, target in batch:
                try:
                    trajectory, frequency = preplan_tcp_trajectory(
                        planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                        ik_frame=ik_frame, tcp_offset=tcp_offset, target_tcp=target,
                        planner_kind=args.turn_planner, max_vel=args.max_vel, max_acc=args.max_acc,
                        timeout_s=args.planning_timeout_s,
                    )
                except Exception as exc:
                    failures.append(str(exc))
                    continue
                selected = (rpy_by_priority[job_priority], yaw, trajectory, frequency)
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("翻转区没有可规划的 Q1 朝向；" + "；".join(failures[-12:]))
    reoriented_rpy, selected_yaw, trajectory, frequency = selected
    print(f"阶段 5/6：执行原工件1整体后撤+转向轨迹，Q1 yaw={selected_yaw:.2f}°，{len(trajectory)} 个关节路点。")
    try:
        arm.follow_trajectory(arm_id=args.arm_id, trajectory=trajectory, sample_frequency=frequency)
    except Exception as exc:
        raise RuntimeError(f"整体后撤+转向轨迹执行失败（不会改为分段重规划）: {exc}") from exc
    return reoriented_rpy, selected_yaw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", type=int, choices=(1, 2), required=True,
                        help="工件编号；自动选择对应调整配置和 module_4_2 视觉结果")
    parser.add_argument("--config", type=Path,
                        help="可选的调整配置覆盖；其 workpiece 必须与 --workpiece 一致")
    parser.add_argument(
        "--turn-candidates-config", type=Path,
        help="闭爪后工件绝对姿态/中心候选 TOML；默认使用该工件配置",
    )
    parser.add_argument("--arm-id", type=int, choices=(2,), required=True)
    parser.add_argument("--workpiece-pose-file", type=Path,
                        help="可选的位姿 JSON 覆盖；默认使用 module_4_2 对应工件的最新结果")
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument("--grasp-point-semantic-offset-m", nargs=3, type=float,
                        default=(-0.025, 0.0, 0.01), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--pregrasp-positive-workpiece-z-m", type=float, default=0.07)
    parser.add_argument("--final-extra-descent-negative-workpiece-z-m", type=float, default=0.0)
    parser.add_argument("--pregrasp-position-tolerance-m", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                        help="到达预抓取点后，沿实际 TCP 局部 X/Y/Z 的固定平移（m）")
    parser.add_argument("--pregrasp-rpy-tolerance-deg", nargs=3, type=float,
                        default=(15.0, 15.0, 15.0), metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--tcp-x-negative-workpiece-z-tolerance-deg", type=float, default=45.0,
                        help="TCP +X 与工件 -Z 的最大夹角")
    parser.add_argument("--tcp-y-parallel-workpiece-y-tolerance-deg", type=float, default=45.0,
                        help="TCP +Y 与工件 ±Y 的最大夹角")
    parser.add_argument("--tcp-z-parallel-workpiece-x-tolerance-deg", type=float, default=45.0,
                        help="TCP +Z 与工件 ±X 的最大夹角")
    parser.add_argument("--servo-linear-speed-mps", type=float, default=0.05)
    parser.add_argument("--servo-angular-speed-radps", type=float, default=0.30)
    parser.add_argument("--servo-max-linear-accel-mps2", type=float, default=0.10)
    parser.add_argument("--servo-max-angular-accel-radps2", type=float, default=0.60)
    parser.add_argument("--servo-hz", type=float, default=50.0)
    parser.add_argument("--servo-position-tolerance-m", type=float, default=0.0005)
    parser.add_argument("--servo-orientation-tolerance-rad", type=float, default=0.035)
    parser.add_argument("--servo-settle-timeout-s", type=float, default=8.0)
    parser.add_argument("--place-tcp-extra-clearance-m", type=float, default=0.005)
    parser.add_argument("--lift-height-m", nargs="+", type=float, default=(0.080,),
                        help="闭爪后 RRTstar 抬起高度候选（m），按给定顺序依次尝试")
    parser.add_argument("--post-release-lift-height-m", type=float, default=0.020,
                        help="松爪后 TCP 沿 body_base +Z 直线抬起的距离（m）")
    parser.add_argument("--attitude-search-step-deg", type=float, default=5.0)
    parser.add_argument("--planning-timeout-s", type=float, default=5.0)
    parser.add_argument("--q1-yaw-min-deg", type=float, default=15.0)
    parser.add_argument("--q1-yaw-max-deg", type=float, default=75.0)
    parser.add_argument("--q1-yaw-search-step-deg", type=float, default=2.5)
    parser.add_argument("--q1-preferred-yaw-deg", type=float, default=35.0)
    parser.add_argument("--turn-roll-tolerance-deg", type=float, default=0.0)
    parser.add_argument("--turn-pitch-tolerance-deg", type=float, default=0.0)
    parser.add_argument("--turn-attitude-search-step-deg", type=float, default=5.0)
    parser.add_argument("--turn-relocation-step-m", type=float, default=0.040)
    parser.add_argument("--post-grasp-turn-search-negative-base-x-m", type=float, default=0.030)
    parser.add_argument("--post-grasp-turn-search-negative-base-y-m", type=float, default=0.200)
    parser.add_argument("--turn-station-tcp-height-m", type=float, default=0.160)
    parser.add_argument("--turn-search-grid-step-m", type=float, default=0.050)
    parser.add_argument("--place-negative-xy-min-m", type=float, default=0.0)
    parser.add_argument("--place-negative-xy-max-m", type=float, default=0.200)
    parser.add_argument("--place-positive-x-max-m", type=float, default=0.200)
    parser.add_argument("--turn-positive-x-min-m", type=float, default=0.0)
    parser.add_argument("--turn-positive-x-max-m", type=float, default=0.200)
    parser.add_argument("--place-negative-y-max-m", type=float, default=0.250)
    parser.add_argument("--place-region-grid-step-m", type=float, default=0.050)
    parser.add_argument("--place-search-grid-step-m", type=float, default=0.025)
    parser.add_argument("--rrt-place-parallelism", type=int, default=RRTCONNECT_PLACE_MAX_PARALLELISM)
    parser.add_argument("--rrt-turn-parallelism", type=int, default=RRTCONNECT_PLACE_MAX_PARALLELISM)
    parser.add_argument("--placement-visibility-roi", nargs=4, type=float,
                        default=(0.08, 0.20, 0.98, 0.90))
    parser.add_argument("--placement-visibility-margin-px", type=float, default=30.0)
    parser.add_argument("--planner", choices=("lin", "ptp", RRTCONNECT_PLANNER_KIND, RRTSTAR_PLANNER_KIND),
                        default=RRTCONNECT_PLANNER_KIND)
    parser.add_argument("--turn-planner", choices=("lin", "ptp", RRTCONNECT_PLANNER_KIND),
                        default=RRTCONNECT_PLANNER_KIND)
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.88.1"))
    parser.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "localhost"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "localhost"))
    parser.add_argument("--frame-id", default=os.environ.get("PARM_PLANNER_FRAME", "body_base"))
    parser.add_argument("--max-vel", type=float, default=0.08)
    parser.add_argument("--max-acc", type=float, default=0.08)
    parser.add_argument("--trajectory-execution-speed-scale", type=float, default=2.0,
                        help="已规划轨迹执行采样频率倍率；2 表示以原轨迹两倍速度执行")
    parser.add_argument("--start-sync-retries", type=int, default=3)
    parser.add_argument("--gripper-timeout-s", type=float, default=5.0)
    parser.add_argument("--gripper-position-tolerance", type=float, default=5.0)
    parser.add_argument("--grasp-minimum-close-travel", type=float, default=5.0)
    parser.add_argument("--grasp-contact-settle-s", type=float, default=0.5)
    parser.add_argument("--calibrate-grasp-xy", action="store_true")
    parser.add_argument("--calibration-record-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.config = args.config or WORKPIECE_CONFIGS[args.workpiece]
    try:
        apply_reorient_config(parser, args)
        expected_name = f"workpiece_{args.workpiece}"
        if args.workpiece_name != expected_name:
            raise ValueError(
                f"调整配置属于 {args.workpiece_name!r}，与 --workpiece {args.workpiece} 不一致"
            )
        args.workpiece_pose_file = args.workpiece_pose_file or newest_workpiece_pose_file(args.workpiece)
        validate_pose_workpiece(args.workpiece_pose_file, args.workpiece)
        args.turn_candidates_config = (
            args.turn_candidates_config or TURN_CANDIDATE_CONFIGS[args.workpiece]
        )
        args.turn_candidates = load_turn_candidates(args.turn_candidates_config)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    if not args.workpiece_pose_file.is_file():
        print(f"错误: 工件位姿文件不存在: {args.workpiece_pose_file}", file=sys.stderr)
        return 1
    positive = (args.pregrasp_positive_workpiece_z_m, *args.lift_height_m,
                args.attitude_search_step_deg, args.turn_search_grid_step_m,
                args.turn_relocation_step_m, args.place_region_grid_step_m,
                args.place_search_grid_step_m)
    if (any(value <= 0.0 for value in positive)
            or args.place_tcp_extra_clearance_m < 0.0):
        print("错误: 高度和步长必须大于 0，放置余量必须非负", file=sys.stderr)
        return 1
    if (any(abs(value) > 0.2 for value in args.grasp_point_semantic_offset_m)
            or abs(args.final_extra_descent_negative_workpiece_z_m) > 0.1):
        print("错误: 抓取点语义偏移或额外下探量超出合理范围", file=sys.stderr)
        return 1
    if (any(abs(value) > 0.1 for value in args.pregrasp_position_tolerance_m)
            or any(not 0.0 <= value <= 45.0 for value in args.pregrasp_rpy_tolerance_deg)):
        print("错误: 预抓取后 TCP 局部 XYZ 平移不得超过 ±0.1 m，RPY 容差必须在 0--45 度", file=sys.stderr)
        return 1
    if (not 0.0 <= args.tcp_x_negative_workpiece_z_tolerance_deg <= 45.0
            or not 0.0 <= args.tcp_y_parallel_workpiece_y_tolerance_deg <= 45.0
            or not 0.0 <= args.tcp_z_parallel_workpiece_x_tolerance_deg <= 45.0):
        print("错误: TCP +X/-Z、+Y/±Y 和 +Z/±X 的轴夹角容差必须在 0--45 度", file=sys.stderr)
        return 1
    if (args.pregrasp_positive_workpiece_z_m
            + args.final_extra_descent_negative_workpiece_z_m <= 0.0):
        print("错误: 预抓取到最终闭爪点必须沿工件 -Z 保持正的下探距离", file=sys.stderr)
        return 1
    if args.planning_timeout_s <= 0.0:
        print("错误: 预规划超时必须大于 0", file=sys.stderr)
        return 1
    if (not 0.0 <= args.turn_roll_tolerance_deg <= 45.0
            or not 0.0 <= args.turn_pitch_tolerance_deg <= 45.0
            or args.turn_attitude_search_step_deg <= 0.0):
        print("错误: 阶段 5 roll/pitch 容差必须在 0--45 度，姿态搜索间隔必须大于 0", file=sys.stderr)
        return 1
    servo_positive = (
        args.servo_linear_speed_mps, args.servo_angular_speed_radps,
        args.servo_max_linear_accel_mps2, args.servo_max_angular_accel_radps2,
        args.servo_hz, args.servo_position_tolerance_m,
        args.servo_orientation_tolerance_rad, args.servo_settle_timeout_s,
        args.post_release_lift_height_m,
    )
    if any(value <= 0.0 for value in servo_positive):
        print("错误: servo 速度、加速度、频率、容差和稳定超时必须大于 0", file=sys.stderr)
        return 1
    if not 1 <= args.rrt_place_parallelism <= RRTCONNECT_PLACE_MAX_PARALLELISM:
        print(f"错误: --rrt-place-parallelism 必须在 1--{RRTCONNECT_PLACE_MAX_PARALLELISM}；5604 没有第 4 个 RRT worker", file=sys.stderr)
        return 1
    if not 1 <= args.rrt_turn_parallelism <= RRTCONNECT_PLACE_MAX_PARALLELISM:
        print(f"错误: --rrt-turn-parallelism 必须在 1--{RRTCONNECT_PLACE_MAX_PARALLELISM}；5604 没有第 4 个 RRT worker", file=sys.stderr)
        return 1
    if args.grasp_minimum_close_travel < 0.0 or args.grasp_contact_settle_s <= 0.0:
        print("错误: 夹爪接触判定参数无效", file=sys.stderr)
        return 1
    if (args.place_negative_xy_min_m < 0.0
            or args.place_negative_xy_max_m < args.place_negative_xy_min_m
            or args.place_positive_x_max_m < args.place_negative_xy_min_m
            or args.place_negative_y_max_m < args.place_negative_xy_min_m
            or args.turn_positive_x_min_m < 0.0
            or args.turn_positive_x_max_m < args.turn_positive_x_min_m):
        print("错误: 放置区域距离无效", file=sys.stderr)
        return 1
    if abs(args.turn_station_tcp_height_m) > 0.5:
        print("错误: --turn-station-tcp-height-m 的相对高度偏移不能超过 0.5 m", file=sys.stderr)
        return 1
    if (args.post_grasp_turn_search_negative_base_x_m < 0.0
            or args.post_grasp_turn_search_negative_base_y_m < 0.0):
        print("错误: 翻转搜索前的 -X/-Y 后撤距离不能小于 0", file=sys.stderr)
        return 1
    if args.placement_visibility_margin_px < 0.0:
        print("错误: --placement-visibility-margin-px 不能小于 0", file=sys.stderr)
        return 1
    try:
        workpiece_transform, red_yaw_deg, quadrant, table_plane_z, workpiece_item, calibration_path = load_workpiece(
            args.workpiece_pose_file, args.workpiece_index
        )
        # JSON offsets are expressed on the observed workpiece's own +X/+Y/+Z.
        semantic_local_grasp = list(args.grasp_point_semantic_offset_m)
        semantic_grasp_point = transform_point(workpiece_transform, semantic_local_grasp)
        workpiece_x_base = normalize_vector(
            [workpiece_transform[row][0] for row in range(3)], "工件 +X"
        )
        workpiece_y_base = normalize_vector(
            [workpiece_transform[row][1] for row in range(3)], "工件 +Y"
        )
        workpiece_z_base = normalize_vector(
            [workpiece_transform[row][2] for row in range(3)], "工件 +Z"
        )
        workpiece_semantic_pose = matrix4_to_pose(workpiece_transform)
        observed_workpiece_centre_pose = workpiece_centre_pose(
            workpiece_item, workpiece_semantic_pose
        )
        grasp_reference_rpy = matrix_to_rpy([
            [workpiece_transform[row][column] for column in range(3)]
            for row in range(3)
        ])
        if quadrant not in (1, 2, 3, 4):
            raise ValueError(f"工件朝向象限无效: Q{quadrant}")
        camera_matrix, distortion, base_camera, calibration_size = load_projection_calibration(calibration_path)
        roi_bounds = normalized_roi_bounds(list(args.placement_visibility_roi), *calibration_size)
        if (roi_bounds[2] - roi_bounds[0] <= 2.0 * args.placement_visibility_margin_px
                or roi_bounds[3] - roi_bounds[1] <= 2.0 * args.placement_visibility_margin_px):
            raise ValueError("放置可见 ROI 小于两侧安全边距，无法容纳工件")
        # Planner targets are native right_eef poses.  Servo starts from the
        # driver's eef_pose, so it needs the fixed local driver->right_eef
        # transform above to represent exactly the same jaw-centre point.
        tcp_offset = [0.0] * 6
        servo_eef_offset = RIGHT_EEF_FROM_TRANSFER_BASE
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    # Recompute all absolute targets from this observation.  Both the approach
    # and descent follow the observed workpiece Z axis, not body_base Z.
    grasp_xyz = semantic_grasp_point.copy()
    pregrasp_xyz = [
        grasp_xyz[index] + args.pregrasp_positive_workpiece_z_m * workpiece_z_base[index]
        for index in range(3)
    ]
    final_grasp_xyz = [
        grasp_xyz[index]
        - args.final_extra_descent_negative_workpiece_z_m * workpiece_z_base[index]
        for index in range(3)
    ]
    centre_xy = final_grasp_xyz[:2].copy()
    vision_grasp_xy = semantic_grasp_point[:2].copy()
    grasp_z = final_grasp_xyz[2]
    lift_z = grasp_z + args.lift_height_m[0]
    turn_station_z = grasp_z + args.turn_station_tcp_height_m
    if args.place_tcp_extra_clearance_m > min(args.lift_height_m):
        print("错误: --place-tcp-extra-clearance-m 不能高于任一 --lift-height-m 候选", file=sys.stderr)
        return 1
    print(
        f"输入工件朝向=Q{quadrant}；闭爪后将读取 right_hand_transfer_end，"
        f"以工件{args.workpiece}正面基准姿态和 TOML 的局部 RPY/中心增量生成候选。"
    )
    print(
        f"语义抓取点：local={[round(value, 9) for value in semantic_local_grasp]} m；"
        f"body_base xyz(m)={semantic_grasp_point}；"
        f"参考 rpy 与本次工件一致(deg)="
        f"{[round(math.degrees(value), 3) for value in grasp_reference_rpy]}"
    )
    print(
        f"语义抓取点 xyz(m)={grasp_xyz}；预抓取点沿工件 +Z "
        f"{args.pregrasp_positive_workpiece_z_m * 1000.0:.1f} mm，xyz(m)={pregrasp_xyz}；"
        f"最终闭爪点相对抓取点沿工件 -Z "
        f"{args.final_extra_descent_negative_workpiece_z_m * 1000.0:.1f} mm，xyz(m)={final_grasp_xyz}。"
    )
    if args.planner == RRTSTAR_PLANNER_KIND:
        print(
            "抓取前流程：RRTstar 精确规划到预抓取位姿（不传 goal tolerance），"
            "若驱动状态仍未到该点，则再以同样的精确 RRTstar 修正；"
            "之后 servo 沿工件 -Z 下探。"
        )
    else:
        print(
            "抓取前流程：RRT 直接规划到预抓取位姿，"
            "锁定所选 TCP 姿态，再沿工件 -Z 直线下探。"
        )
    print(
        f"ICP 语义 +X 的 body_base 水平投影 yaw={red_yaw_deg:.2f}°；"
        f"工具 +X 将与工件完整 3-D -Z 同向平行（即与工件 +Z 反向；工件 +Z={workpiece_z_base}），"
        f"工具 +Y 将与工件完整 3-D ±Y 平行（工件 +Y={workpiece_y_base}），"
        "工具 +Z 相应与工件 ±X 平行。"
    )
    print(
        f"TCP 高度：构造抓取点 z={grasp_xyz[2]:.6f}，预抓取 z={pregrasp_xyz[2]:.6f}，"
        f"抓取={grasp_z:.6f}，标准放下={grasp_z + args.place_tcp_extra_clearance_m:.6f}，"
        f"阶段 4 首选抬起={lift_z:.6f} m（+{args.lift_height_m[0] * 1000.0:.1f} mm）；"
        f"翻转站={turn_station_z:.6f} m（相对抓取点 {args.turn_station_tcp_height_m * 1000.0:+.1f} mm）"
    )
    if args.place_tcp_extra_clearance_m:
        print(
            f"放下/松爪将比抓取 TCP 高 {args.place_tcp_extra_clearance_m * 1000.0:.1f} mm，"
            f"目标 z={grasp_z + args.place_tcp_extra_clearance_m:.6f} m。"
        )
    if not args.execute:
        turn = args.turn_candidates
        total = (len(turn["x_offset_candidates_m"])
                 * len(turn["y_offset_candidates_m"])
                 * len(turn["z_offset_candidates_m"])
                 * len(turn["roll_candidates_deg"])
                 * len(turn["pitch_candidates_deg"])
                 * len(turn["yaw_candidates_deg"]))
        print(
            f"工件{args.workpiece}闭爪后调整候选 TOML：{args.turn_candidates_config}；"
            f"规划器={turn['planner']}；"
            f"x/y/z/r/p/y 数量={len(turn['x_offset_candidates_m'])}/"
            f"{len(turn['y_offset_candidates_m'])}/{len(turn['z_offset_candidates_m'])}/"
            f"{len(turn['roll_candidates_deg'])}/{len(turn['pitch_candidates_deg'])}/"
            f"{len(turn['yaw_candidates_deg'])}，共 {total} 个。"
        )
        print(
            "放置 ROI 投影过滤："
            f"像素 [{roi_bounds[0]}, {roi_bounds[1]})-({roi_bounds[2]}, {roi_bounds[3]})，"
            f"边界余量 {args.placement_visibility_margin_px:.1f} px。"
        )
        print("预览：--execute 后将读取当前 TCP，筛选全段可规划的实时抓取姿态；未连接机械臂。")
        return 0

    arm = planner = gripper = None
    try:
        sdk = import_sdk()
        arm = sdk.Arm(host=args.arm_host)
        ensure_ready(arm, args.arm_id)
        planner = sdk.Planner(host=args.planner_host)
        gripper = sdk.Gripper(host=args.gripper_host)
        group, ik_frame = "right_arm", "right_eef"
        current_ik = pose_from_lookup_transform(planner.lookup_transform(args.frame_id, ik_frame).get("data"))
        current_tcp = compose_pose(current_ik, tcp_offset)
        current_rpy = rotvec_to_rpy(current_tcp[3:])
        print(f"当前 TCP xyz(m)={[round(v, 6) for v in current_tcp[:3]]}；rpy(deg)={[round(math.degrees(v), 2) for v in current_rpy]}")
        orientation_candidates = attitude_candidates(
            workpiece_x_base, workpiece_y_base, workpiece_z_base,
            rpy_tolerance_deg=list(args.pregrasp_rpy_tolerance_deg),
            sample_step_deg=args.attitude_search_step_deg,
            x_negative_z_tolerance_deg=args.tcp_x_negative_workpiece_z_tolerance_deg,
            y_parallel_y_tolerance_deg=args.tcp_y_parallel_workpiece_y_tolerance_deg,
            z_parallel_x_tolerance_deg=args.tcp_z_parallel_workpiece_x_tolerance_deg,
        )
        if not orientation_candidates:
            raise RuntimeError("无法构造 TCP +X=-工件 Z、+Y=±工件 Y 的预抓取姿态")
        candidates: list[tuple[list[float], list[float], float, float, float, int]] = []
        for orientation, x_error_deg, y_error_deg, z_error_deg, y_sign in orientation_candidates:
            candidates.append(
                (pregrasp_xyz, orientation, x_error_deg, y_error_deg, z_error_deg, y_sign)
            )
        # Try candidates from strict axis alignment to wider tolerances.  The
        # first plannable one is therefore the closest-to-parallel solution;
        # current wrist travel is only a tie-breaker at equal axis errors.
        candidates.sort(key=lambda candidate: (
            max(candidate[2], candidate[3], candidate[4]),
            candidate[2] + candidate[3] + candidate[4],
            orientation_distance_rad(current_tcp[3:], candidate[1]),
        ))
        print(
            f"开始按轴偏差从严到宽预规划 {len(candidates)} 个预抓取六维候选；"
            "预抓取点不再搜索 XYZ 偏差；到达后将沿实际 TCP 局部 X/Y/Z 固定平移(m)="
            f"{list(args.pregrasp_position_tolerance_m)}，"
            f"绕 TCP 自身 X/Y/Z 的 RPY 容差(deg)="
            f"{list(args.pregrasp_rpy_tolerance_deg)}，姿态采样间隔 "
            f"{args.attitude_search_step_deg:g}°；TCP +X/-Z 最大夹角 "
            f"{args.tcp_x_negative_workpiece_z_tolerance_deg:g}°，"
            f"TCP +Y/±Y 最大夹角 {args.tcp_y_parallel_workpiece_y_tolerance_deg:g}°，"
            f"TCP +Z/±X 最大夹角 {args.tcp_z_parallel_workpiece_x_tolerance_deg:g}°；"
            f"单次 RPC 最多等待 {args.planning_timeout_s:g} 秒。",
            flush=True,
        )

        chosen_xyz: list[float] | None = None
        chosen_orientation: list[float] | None = None
        chosen_x_error_deg: float | None = None
        chosen_y_error_deg: float | None = None
        chosen_z_error_deg: float | None = None
        chosen_y_sign: int | None = None
        failures: list[str] = []
        for index, (candidate_xyz, orientation, x_error_deg, y_error_deg, z_error_deg, y_sign) in enumerate(candidates, start=1):
            rpy = rotvec_to_rpy(orientation)
            turn_angle_deg = math.degrees(orientation_distance_rad(current_tcp[3:], orientation))
            print(
                f"预规划候选 {index}/{len(candidates)}：相对当前转动={turn_angle_deg:.2f}°；"
                f"+X/-Z 偏差={x_error_deg:.2f}°，+Y/{'+' if y_sign > 0 else '-'}Y 偏差={y_error_deg:.2f}°，"
                f"+Z/{'+' if y_sign > 0 else '-'}X 偏差={z_error_deg:.2f}°；"
                f"xyz(m)={[round(value, 6) for value in candidate_xyz]}；"
                f"rpy(deg)={[round(math.degrees(v), 2) for v in rpy]}",
                flush=True,
            )
            target = candidate_xyz + orientation
            print("  检查精确六维预抓取目标...", end="", flush=True)
            feasible, detail = can_plan_target(
                planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                ik_frame=ik_frame, tcp_offset=tcp_offset, target=target,
                target_is_tcp=True, planner_kind=args.planner,
                max_vel=args.max_vel, max_acc=args.max_acc,
                timeout_s=args.planning_timeout_s,
            )
            print("可达" if feasible else f"失败（{detail}）", flush=True)
            if feasible:
                chosen_xyz = candidate_xyz
                chosen_orientation = orientation
                chosen_x_error_deg = x_error_deg
                chosen_y_error_deg = y_error_deg
                chosen_z_error_deg = z_error_deg
                chosen_y_sign = y_sign
                print(f"候选 {index}/{len(candidates)} 可规划（{detail}）；采用该最严格可达候选。")
                break
            if len(failures) < 8:
                failures.append(f"候选 {index}: {detail}")
        if (chosen_xyz is None or chosen_orientation is None
                or chosen_x_error_deg is None or chosen_y_error_deg is None
                or chosen_z_error_deg is None or chosen_y_sign is None):
            raise RuntimeError("所有实时预抓取候选都不可规划。前几个失败原因：" + "；".join(failures))
        print(
            "选择轴对齐最严格的可规划候选："
            f"+X/-Z 偏差={chosen_x_error_deg:.2f}°，"
            f"+Y/{'+' if chosen_y_sign > 0 else '-'}Y 偏差={chosen_y_error_deg:.2f}°，"
            f"+Z/{'+' if chosen_y_sign > 0 else '-'}X 偏差={chosen_z_error_deg:.2f}°；"
            f"同等轴偏差时优先较小腕部转动。"
        )

        selected_pregrasp_target = chosen_xyz + chosen_orientation
        set_gripper_state(gripper, arm_id=args.arm_id, target_open=True, timeout_s=args.gripper_timeout_s,
                          tolerance=args.gripper_position_tolerance)
        plan_and_follow(
            arm, planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
            ik_frame=ik_frame, tcp_offset=tcp_offset, target=selected_pregrasp_target,
            target_is_tcp=True, planner_kind=args.planner, max_vel=args.max_vel,
            max_acc=args.max_acc, retries=args.start_sync_retries,
            stage="阶段 1：RRTstar 精确规划到工件 +Z 预抓取位姿",
            execution_speed_scale=args.trajectory_execution_speed_scale,
        )
        try:
            reached_pregrasp_tcp = require_driver_right_eef_at_target(
                arm, arm_id=args.arm_id, driver_to_right_eef=servo_eef_offset,
                target=selected_pregrasp_target,
                position_tolerance_m=args.servo_position_tolerance_m,
                orientation_tolerance_rad=args.servo_orientation_tolerance_rad,
                timeout_s=args.servo_settle_timeout_s,
                stage="阶段 1 后同步",
            )
        except RuntimeError as rrt_sync_error:
            # A real driver/planner state mismatch must be corrected before
            # servo motion.  This RRTstar request deliberately omits
            # rrtstar_goal_tolerance, so its goal is the exact right_eef pose.
            print(
                f"阶段 1 后同步未通过（{rrt_sync_error}）；"
                "阶段 1c：用精确 RRTstar 轨迹修正到名义预抓取点。"
            )
            try:
                correction_trajectory, correction_frequency = preplan_tcp_trajectory(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                    ik_frame=ik_frame, tcp_offset=tcp_offset,
                    target_tcp=selected_pregrasp_target, planner_kind=RRTSTAR_PLANNER_KIND,
                    max_vel=args.max_vel, max_acc=args.max_acc,
                    timeout_s=args.planning_timeout_s,
                    stage="阶段 1c：精确 RRTstar 预抓取修正",
                )
                arm.follow_trajectory(
                    arm_id=args.arm_id, trajectory=correction_trajectory,
                    sample_frequency=correction_frequency * args.trajectory_execution_speed_scale,
                )
            except Exception as exc:
                raise RuntimeError(
                    "阶段 1c 精确 RRTstar 预抓取修正不可规划或执行失败；"
                    "不会开始 servo 下探。"
                ) from exc
            reached_pregrasp_tcp = require_driver_right_eef_at_target(
                arm, arm_id=args.arm_id, driver_to_right_eef=servo_eef_offset,
                target=selected_pregrasp_target,
                position_tolerance_m=args.servo_position_tolerance_m,
                orientation_tolerance_rad=args.servo_orientation_tolerance_rad,
                timeout_s=args.servo_settle_timeout_s,
                stage="阶段 1c 后同步",
            )
        tcp_local_adjustment = list(args.pregrasp_position_tolerance_m)
        if any(abs(value) > 1e-9 for value in tcp_local_adjustment):
            adjusted_pregrasp_target = compose_pose(
                reached_pregrasp_tcp, tcp_local_adjustment + [0.0, 0.0, 0.0]
            )
            print(
                "阶段 1b：从实际到达的预抓取 TCP 沿其自身局部 X/Y/Z 平移 "
                f"{[round(value * 1000.0, 2) for value in tcp_local_adjustment]} mm；"
                f"目标 xyz(m)={[round(value, 6) for value in adjusted_pregrasp_target[:3]]}。"
            )
            servo_follow_tcp(
                arm, arm_id=args.arm_id, tcp_target=adjusted_pregrasp_target,
                servo_eef_offset=servo_eef_offset,
                stage="阶段 1b：预抓取后 TCP 局部 XYZ 调整",
                linear_speed_mps=args.servo_linear_speed_mps,
                angular_speed_radps=args.servo_angular_speed_radps,
                max_linear_accel_mps2=args.servo_max_linear_accel_mps2,
                max_angular_accel_radps2=args.servo_max_angular_accel_radps2,
                hz=args.servo_hz,
                position_tolerance_m=args.servo_position_tolerance_m,
                orientation_tolerance_rad=args.servo_orientation_tolerance_rad,
                settle_timeout_s=args.servo_settle_timeout_s,
            )
            reached_pregrasp_tcp = actual_right_eef_from_driver(
                arm, args.arm_id, servo_eef_offset
            )
        locked_orientation = reached_pregrasp_tcp[3:].copy()
        chosen_rpy = rotvec_to_rpy(locked_orientation)
        descent_m = (
            args.pregrasp_positive_workpiece_z_m
            + args.final_extra_descent_negative_workpiece_z_m
        )
        # The configured semantic grasp point is the endpoint.  A delayed or
        # inaccurate pregrasp state must never shift right_eef past the object.
        selected_final_grasp_xyz = final_grasp_xyz.copy()
        # Keep the actually reached orientation, but use the configured
        # jaw-centre position exactly for the close target.
        grasp_target = selected_final_grasp_xyz + locked_orientation
        centre_xy = selected_final_grasp_xyz[:2].copy()
        grasp_z = selected_final_grasp_xyz[2]
        lift_z = grasp_z + args.lift_height_m[0]
        print(
            f"锁定实际到达的候选 TCP 姿态 rpy(deg)="
            f"{[round(math.degrees(value), 2) for value in chosen_rpy]}；"
            f"从实际预抓取 xyz(m)="
            f"{[round(value, 6) for value in reached_pregrasp_tcp[:3]]} "
            f"到配置定义的闭爪点 xyz(m)="
            f"{[round(value, 6) for value in selected_final_grasp_xyz]}；"
            f"名义预抓取到闭爪点间距={descent_m * 1000.0:.1f} mm。"
        )
        print(
            "阶段 2：以 RRTstar 将 right_eef 从预抓取位姿规划到闭爪点；"
            "目标为精确 right_eef 抓取点。"
        )
        descent_trajectory, descent_frequency = preplan_tcp_trajectory(
            planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
            ik_frame=ik_frame, tcp_offset=tcp_offset, target_tcp=grasp_target,
            planner_kind=RRTSTAR_PLANNER_KIND, max_vel=args.max_vel,
            max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
            stage="阶段 2：RRTstar 规划 right_eef 下探到闭爪点",
        )
        arm.follow_trajectory(
            arm_id=args.arm_id, trajectory=descent_trajectory,
            sample_frequency=descent_frequency * args.trajectory_execution_speed_scale,
        )
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
                "vision_semantic_negative_x_grasp_xy_m": vision_grasp_xy,
                "commanded_tcp_target_xy_m": centre_xy,
                "measured_grasp_xy_offset_m": [0.0, 0.0],
                "commanded_grasp_tcp_z_m": grasp_z,
                "actual_tcp_xy_m": actual_xy,
                "actual_minus_commanded_tcp_xy_m": actual_minus_target,
                "recommended_command_xy_offset_m": recommended_offset,
                "note": (
                    "The command target is the unmodified semantic purple point. "
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
            print("校准模式：已下探到紫色抓取点，未闭爪、未翻转；机械臂保持当前位置。")
            print(f"视觉语义 -X 抓取点 xy(m)={vision_grasp_xy}")
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
        # Once the jaws are closed, the object is rigid relative to the actual
        # transfer-end frame.  Record that live relation before lifting; every
        # later object candidate is converted back through this same relation.
        transfer_frame = "right_hand_transfer_end"
        closed_transfer = pose_from_lookup_transform(
            planner.lookup_transform(args.frame_id, transfer_frame).get("data")
        )
        closed_eef = pose_from_lookup_transform(
            planner.lookup_transform(args.frame_id, ik_frame).get("data")
        )
        transfer_from_object_centre = compose_pose(
            inverse_pose(closed_transfer), observed_workpiece_centre_pose
        )
        transfer_from_eef = compose_pose(inverse_pose(closed_transfer), closed_eef)
        append_event({
            "event": "module_7_closed_grasp_rigid_relation",
            "frame_id": args.frame_id,
            "transfer_frame": transfer_frame,
            "object_frame": "workpiece_semantic_centre",
            "T_right_hand_transfer_end_workpiece_centre": transfer_from_object_centre,
            "T_right_hand_transfer_end_right_eef": transfer_from_eef,
            "observed_workpiece_centre_body_base": observed_workpiece_centre_pose,
        })
        print(
            "阶段 3b：已记录闭爪瞬间刚性关系 "
            "T_right_hand_transfer_end_workpiece_centre；"
            f"工件中心 body_base xyz(m)={[round(value, 6) for value in observed_workpiece_centre_pose[:3]]}。"
        )
        # Lift from the actual reached TCP.  RRTstar tries the configured
        # height candidates strictly in their listed priority order.
        grasped_ik = pose_from_lookup_transform(
            planner.lookup_transform(args.frame_id, ik_frame).get("data")
        )
        grasped_tcp = compose_pose(grasped_ik, tcp_offset)
        lift_heights = list(args.lift_height_m)
        print(
            f"阶段 4：从实际 TCP xyz={[round(value, 6) for value in grasped_tcp[:3]]} "
            "保持 XY/姿态，以 RRTstar（rrtstar_goal_tolerance=True）按配置列表顺序规划抬起；"
            f"候选高度(mm)={[round(value * 1000.0, 1) for value in lift_heights]}。"
        )
        selected_lift_height: float | None = None
        lift_errors: list[str] = []
        for candidate_index, candidate_height in enumerate(lift_heights, start=1):
            lift_target = [
                grasped_tcp[0], grasped_tcp[1], grasped_tcp[2] + candidate_height,
                *grasped_tcp[3:],
            ]
            stage = (
                f"阶段 4：RRTstar 闭爪后抬起候选 {candidate_index}/{len(lift_heights)} "
                f"(+{candidate_height * 1000.0:.1f} mm)"
            )
            try:
                trajectory, frequency = preplan_tcp_trajectory(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                    ik_frame=ik_frame, tcp_offset=tcp_offset, target_tcp=lift_target,
                    planner_kind=RRTSTAR_PLANNER_KIND, max_vel=args.max_vel,
                    max_acc=args.max_acc, timeout_s=RRTCONNECT_TIMEOUT_S, stage=stage,
                    rrtstar_goal_tolerance=True,
                )
            except Exception as exc:
                error = f"+{candidate_height * 1000.0:.1f} mm: {exc}"
                lift_errors.append(error)
                print(f"{stage} 不可规划（{exc}）；尝试列表中的下一个高度。")
                continue
            try:
                arm.follow_trajectory(
                    arm_id=args.arm_id, trajectory=trajectory,
                    sample_frequency=frequency * args.trajectory_execution_speed_scale,
                )
            except Exception as exc:
                raise RuntimeError(f"{stage} 已规划但执行失败；停止后续高度尝试：{exc}") from exc
            selected_lift_height = candidate_height
            lift_z = grasped_tcp[2] + candidate_height
            print(f"{stage} 成功并已执行。")
            break
        if selected_lift_height is None:
            raise RuntimeError(
                "阶段 4：所有 RRTstar 抬起高度候选均不可规划或执行失败："
                + "; ".join(lift_errors)
            )
        # From this point on all targets are described as object-centre
        # candidates, then converted through the closed-grasp transfer-end
        # relation recorded above.
        lifted_transfer = pose_from_lookup_transform(
            planner.lookup_transform(args.frame_id, transfer_frame).get("data")
        )
        lifted_object_centre = compose_pose(lifted_transfer, transfer_from_object_centre)
        turn = args.turn_candidates
        turn_planner = str(turn["planner"])
        print(
            f"阶段 5：抬起后工件中心 body_base xyz(m)="
            f"{[round(value, 6) for value in lifted_object_centre[:3]]}；"
            f"目标姿态使用工件{args.workpiece} STL 正面基准 "
            "R_body_object_front @ R_local_rpy；"
            f"规划器={turn_planner}；"
            f"候选总数={len(turn['x_offset_candidates_m'])}×{len(turn['y_offset_candidates_m'])}×"
            f"{len(turn['z_offset_candidates_m'])}×{len(turn['roll_candidates_deg'])}×"
            f"{len(turn['pitch_candidates_deg'])}×{len(turn['yaw_candidates_deg'])}。"
        )
        turn_jobs: list[tuple[int, list[float], float, list[float]]] = []
        turn_details_by_priority: dict[int, dict[str, object]] = {}
        priority = 0
        # TOML list order is the priority.  X/Y/Z are outer loops, then local
        # roll/pitch/yaw, so yaw changes fastest within one position.
        for offset_x in turn["x_offset_candidates_m"]:
            for offset_y in turn["y_offset_candidates_m"]:
                for offset_z in turn["z_offset_candidates_m"]:
                    centre_offset = [offset_x, offset_y, offset_z]
                    for roll_deg in turn["roll_candidates_deg"]:
                        for pitch_deg in turn["pitch_candidates_deg"]:
                            for yaw_deg in turn["yaw_candidates_deg"]:
                                local_rpy_deg = [roll_deg, pitch_deg, yaw_deg]
                                # Match Module 3 exactly: uppercase ``XYZ``
                                # is SciPy's intrinsic local-axis convention.
                                # The local term is therefore applied in the
                                # already defined STL-front object frame.
                                target_rotation = (
                                    Rotation.from_euler(
                                        "XYZ", turn["front_base_rpy_deg"], degrees=True
                                    )
                                    * Rotation.from_euler(
                                        "XYZ", local_rpy_deg, degrees=True
                                    )
                                )
                                object_target = [0.0, 0.0, 0.0, *target_rotation.as_rotvec().tolist()]
                                object_target[:3] = [
                                    lifted_object_centre[axis] + centre_offset[axis]
                                    for axis in range(3)
                                ]
                                object_heading_yaw = pose_heading_yaw_deg(object_target)
                                transfer_target = compose_pose(
                                    object_target, inverse_pose(transfer_from_object_centre)
                                )
                                eef_target = compose_pose(transfer_target, transfer_from_eef)
                                turn_jobs.append((priority, object_target[:2], object_heading_yaw, eef_target))
                                turn_details_by_priority[priority] = {
                                    "local_rpy_deg": local_rpy_deg,
                                    "centre_offset_m": centre_offset,
                                    "object_target": object_target,
                                    "eef_target": eef_target,
                                }
                                priority += 1
        turn_failures: list[str] = []
        turn_outcomes: dict[int, bool] = {}
        turn_failure_reasons: dict[int, str] = {}
        selected_priority: int | None = None
        turn_planning_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        reoriented_rpy: list[float] | None = None
        selected_object_heading_yaw: float | None = None
        selected_turn_object_centre: list[float] | None = None
        selected_trajectory: list | None = None
        selected_frequency: float | None = None
        if turn_planner == RRTCONNECT_PLANNER_KIND:
            for batch_start in range(0, len(turn_jobs), args.rrt_turn_parallelism):
                batch = turn_jobs[batch_start:batch_start + args.rrt_turn_parallelism]
                print(
                    f"并行 RRTConnect 翻转规划：候选 {batch_start + 1}--{batch_start + len(batch)}/{len(turn_jobs)}，"
                    f"并发 {len(batch)}。",
                    flush=True,
                )
                for job_priority, _, object_heading_yaw, target in batch:
                    detail = turn_details_by_priority[job_priority]
                    print(
                        f"  规划工件局部调整：local rpy(deg)={detail['local_rpy_deg']}，"
                        f"中心增量(m)={detail['centre_offset_m']}，目标工件 +X yaw="
                        f"{object_heading_yaw:.2f}°...",
                        flush=True,
                    )
                results = rrtconnect_plan_tcp_batch(
                    sdk, planner_host=args.planner_host, arm_id=args.arm_id, group=group,
                    frame_id=args.frame_id, ik_frame=ik_frame, tcp_offset=tcp_offset, jobs=batch,
                    max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                    parallelism=args.rrt_turn_parallelism,
                )
                successes = []
                for job_priority, _, object_heading_yaw, trajectory, frequency, failure in results:
                    if failure is not None:
                        turn_outcomes[job_priority] = False
                        turn_failure_reasons[job_priority] = str(failure)
                        turn_failures.append(
                            f"目标工件 +X yaw={object_heading_yaw:.2f}°: {failure}"
                        )
                    else:
                        turn_outcomes[job_priority] = True
                        successes.append((job_priority, object_heading_yaw, trajectory, frequency))
                if successes:
                    (selected_priority, selected_object_heading_yaw,
                     selected_trajectory, selected_frequency) = successes[0]
                    detail = turn_details_by_priority[selected_priority]
                    selected_turn_object_centre = detail["object_target"]
                    selected_eef_target = detail["eef_target"]
                    assert isinstance(selected_turn_object_centre, list)
                    assert isinstance(selected_eef_target, list)
                    reoriented_rpy = rotvec_to_rpy(selected_eef_target[3:])
                    print(
                        f"选择工件{args.workpiece}调整候选 #{selected_priority + 1}/{len(turn_jobs)}："
                        f"中心增量 xyz(m)={detail['centre_offset_m']}，"
                        f"局部 rpy(deg)={detail['local_rpy_deg']}；"
                        f"目标工件中心 body_base xyz(m)={selected_turn_object_centre[:3]}，"
                        f"工件 +X yaw={selected_object_heading_yaw:.2f}°；"
                        f"已保存 {len(selected_trajectory)} 个路点的整体轨迹。"
                    )
                    break
        else:
            for job_priority, _, object_heading_yaw, final_turn_target in turn_jobs:
                detail = turn_details_by_priority[job_priority]
                print(
                    f"规划工件局部调整：local rpy(deg)={detail['local_rpy_deg']}，"
                    f"中心增量(m)={detail['centre_offset_m']}，目标工件 +X yaw="
                    f"{object_heading_yaw:.2f}°...",
                    flush=True,
                )
                try:
                    trajectory, frequency = preplan_tcp_trajectory(
                        planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                        tcp_offset=tcp_offset, target_tcp=final_turn_target, planner_kind=turn_planner,
                        max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                    )
                except Exception as exc:
                    turn_outcomes[job_priority] = False
                    turn_failure_reasons[job_priority] = str(exc)
                    turn_failures.append(f"目标工件 +X yaw={object_heading_yaw:.2f}°: {exc}")
                    continue
                turn_outcomes[job_priority] = True
                selected_priority = job_priority
                selected_object_heading_yaw = object_heading_yaw
                selected_turn_object_centre = detail["object_target"]
                selected_eef_target = detail["eef_target"]
                assert isinstance(selected_turn_object_centre, list)
                assert isinstance(selected_eef_target, list)
                reoriented_rpy = rotvec_to_rpy(selected_eef_target[3:])
                selected_trajectory = trajectory
                selected_frequency = frequency
                print(f"选择工件{args.workpiece}调整候选 #{job_priority + 1}/{len(turn_jobs)}："
                      f"中心增量 xyz(m)={detail['centre_offset_m']}，"
                      f"局部 rpy(deg)={detail['local_rpy_deg']}；"
                      f"目标工件中心 body_base xyz(m)={selected_turn_object_centre[:3]}，"
                      f"工件 +X yaw={selected_object_heading_yaw:.2f}°；已保存 {len(trajectory)} 个路点的整体轨迹。")
                break
        try:
            statistics_path = record_turn_candidate_statistics(
                workpiece=args.workpiece, config_path=args.turn_candidates_config,
                details_by_priority=turn_details_by_priority, outcomes=turn_outcomes,
                selected_priority=selected_priority,
            )
            history_path = append_turn_planning_records(
                run_id=turn_planning_run_id, workpiece=args.workpiece,
                config_path=args.turn_candidates_config, pose_file=args.workpiece_pose_file,
                workpiece_index=args.workpiece_index, planner_kind=turn_planner,
                details_by_priority=turn_details_by_priority, outcomes=turn_outcomes,
                failure_reasons=turn_failure_reasons,
                selected_priority=selected_priority,
            )
            print(
                f"翻转候选统计已更新：{statistics_path}；本次尝试 {len(turn_outcomes)} 个，"
                f"可规划 {sum(turn_outcomes.values())} 个；逐候选记录已追加：{history_path}；"
                f"选择={selected_priority + 1 if selected_priority is not None else '无'}。"
            )
        except (OSError, RuntimeError) as exc:
            print(f"警告：翻转候选统计未写入，但将继续执行已保存轨迹：{exc}", file=sys.stderr)
        if (reoriented_rpy is None or selected_object_heading_yaw is None
                or selected_turn_object_centre is None or selected_trajectory is None
                or selected_frequency is None or selected_priority is None):
            raise RuntimeError("所有工件局部调整候选均不可规划；" + "；".join(turn_failures[-12:]))
        print(
            f"阶段 5/6：直接执行已保存的工件局部调整轨迹，"
            f"工件 +X yaw={selected_object_heading_yaw:.2f}°，{len(selected_trajectory)} 个关节路点。"
        )
        try:
            arm.follow_trajectory(
                arm_id=args.arm_id, trajectory=selected_trajectory, sample_frequency=selected_frequency * args.trajectory_execution_speed_scale,
            )
        except Exception as exc:
            raise RuntimeError(f"工件局部调整轨迹执行失败（不会改为分段重规划）: {exc}") from exc
        # Stage 6 is an object-placement search, not a TCP-placement search.
        # The XY region is still expressed relative to the initial grasp point,
        # but every candidate names the *workpiece centre*.  Convert it back
        # through the closed-grasp rigid relation before asking the arm planner
        # for a TCP trajectory, exactly as stage 5 does for the turn target.
        # This matters because a rotated workpiece centre is generally offset
        # from right_eef; treating a desired object centre as a TCP target can
        # move the workpiece outside the requested placement region.
        reoriented_transfer = pose_from_lookup_transform(
            planner.lookup_transform(args.frame_id, transfer_frame).get("data")
        )
        reoriented_object_centre = compose_pose(
            reoriented_transfer, transfer_from_object_centre
        )
        place_x_negative_distances: list[float] = []
        distance = args.place_negative_xy_min_m
        while distance <= args.place_negative_xy_max_m + 1e-9:
            place_x_negative_distances.append(min(distance, args.place_negative_xy_max_m))
            distance += args.place_search_grid_step_m
        if (not place_x_negative_distances
                or place_x_negative_distances[-1] < args.place_negative_xy_max_m - 1e-9):
            place_x_negative_distances.append(args.place_negative_xy_max_m)
        place_x_positive_distances: list[float] = []
        # X=0 is already represented by the negative-side list when the
        # minimum is zero; start positive samples at one grid step to avoid
        # planning the same centre point twice.
        distance = max(args.place_negative_xy_min_m, args.place_search_grid_step_m)
        while distance <= args.place_positive_x_max_m + 1e-9:
            place_x_positive_distances.append(min(distance, args.place_positive_x_max_m))
            distance += args.place_search_grid_step_m
        if (not place_x_positive_distances
                or place_x_positive_distances[-1] < args.place_positive_x_max_m - 1e-9):
            place_x_positive_distances.append(args.place_positive_x_max_m)
        place_y_distances: list[float] = []
        distance = args.place_negative_xy_min_m
        while distance <= args.place_negative_y_max_m + 1e-9:
            place_y_distances.append(min(distance, args.place_negative_y_max_m))
            distance += args.place_search_grid_step_m
        if (not place_y_distances
                or place_y_distances[-1] < args.place_negative_y_max_m - 1e-9):
            place_y_distances.append(args.place_negative_y_max_m)
        object_place_candidates = sorted(
            ([centre_xy[0] + x_offset, centre_xy[1] - dy]
             for x_offset in ([-dx for dx in place_x_negative_distances] + place_x_positive_distances)
             for dy in place_y_distances),
            # Prefer the farthest reachable negative-Y placement.  For points
            # at the same Y, search the positive-X side before centre and
            # negative-X side.
            key=lambda xy: (xy[1], -xy[0]),
        )
        yaw_delta_deg = shortest_angle_delta_deg(red_yaw_deg, selected_object_heading_yaw)
        # The footprint helper needs offsets relative to the point represented
        # by a placement candidate.  Candidates below represent object centres,
        # whereas the grasp-time footprint was originally expressed relative to
        # the semantic grasp point.
        object_centre_footprint_xy = footprint_local_base_xy(
            workpiece_item, observed_workpiece_centre_pose[:2], table_plane_z,
            camera_matrix, distortion, base_camera,
        )
        visible_object_place_candidates: list[list[float]] = []
        for object_place_xy in object_place_candidates:
            projected_footprint = project_placement_footprint(
                object_place_xy, object_centre_footprint_xy, yaw_delta_deg, table_plane_z,
                camera_matrix, distortion, base_camera,
            )
            if footprint_is_inside_roi(
                    projected_footprint, roi_bounds, args.placement_visibility_margin_px):
                visible_object_place_candidates.append(object_place_xy)
        filtered_count = len(object_place_candidates) - len(visible_object_place_candidates)
        object_place_candidates = visible_object_place_candidates
        print(
            f"阶段 6：翻转已完成；ROI 过滤掉 {filtered_count} 个候选，"
            f"当前工件中心 xy={reoriented_object_centre[:2]}，以初始抓取点 xy={centre_xy} 为原点，"
            f"在正/负 X、负 Y 区搜索剩余 {len(object_place_candidates)} 个工件中心下放候选；"
            f"X 负向 {args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_xy_max_m * 1000.0:.1f} mm、正向 "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_positive_x_max_m * 1000.0:.1f} mm，Y 负向 "
            f"{args.place_negative_xy_min_m * 1000.0:.1f}--"
            f"{args.place_negative_y_max_m * 1000.0:.1f} mm；下放网格 "
            f"{args.place_search_grid_step_m * 1000.0:.1f} mm。"
        )
        selected_object_place_xy: list[float] | None = None
        selected_tcp_place_target: list[float] | None = None
        selected_place_trajectory: list | None = None
        selected_place_frequency: float | None = None
        selected_place_z: float | None = None
        place_failures: list[str] = []
        # Try the desired table-placement height first.  If it is not
        # reachable, try progressively higher targets so the workpiece is put
        # down as far as the planner can safely lower it instead of aborting
        # while still grasping it.
        # Preserve the configured release-height semantics while changing the
        # planning semantics to object-centre placement: first map the legacy
        # nominal TCP release pose to its corresponding held-object pose, then
        # vary that object-centre height for fallback attempts.
        nominal_tcp_release = centre_xy + [
            grasp_z + args.place_tcp_extra_clearance_m, *rpy_to_rotvec(reoriented_rpy)
        ]
        nominal_eef_release = compose_pose(nominal_tcp_release, inverse_pose(tcp_offset))
        nominal_transfer_release = compose_pose(
            nominal_eef_release, inverse_pose(transfer_from_eef)
        )
        nominal_object_release = compose_pose(
            nominal_transfer_release, transfer_from_object_centre
        )
        standard_object_place_z = nominal_object_release[2]
        place_z_candidates = [standard_object_place_z]
        fallback_step_m = 0.010
        # Do not command a fallback lower than the present held object.  The
        # old TCP-based lift limit is retained as a conservative upper bound.
        fallback_z = standard_object_place_z + fallback_step_m
        max_object_place_z = max(lift_z, reoriented_object_centre[2])
        while fallback_z < max_object_place_z - 1e-9:
            place_z_candidates.append(fallback_z)
            fallback_z += fallback_step_m
        priority = 0
        place_jobs: list[tuple[int, list[float], float, list[float]]] = []
        for candidate_object_z in place_z_candidates:
            for object_place_xy in object_place_candidates:
                object_target = object_place_xy + [candidate_object_z] + selected_turn_object_centre[3:]
                transfer_target = compose_pose(
                    object_target, inverse_pose(transfer_from_object_centre)
                )
                eef_target = compose_pose(transfer_target, transfer_from_eef)
                tcp_target = compose_pose(eef_target, tcp_offset)
                place_jobs.append((priority, object_place_xy, candidate_object_z, tcp_target))
                priority += 1
        # Placement is intentionally searched serially with Cartesian RRTstar.
        # This is independent of --planner, which continues to govern the
        # pregrasp motion and other non-placement stages.
        for _, object_place_xy, candidate_object_z, tcp_place_target in place_jobs:
            height_note = "标准工件下放高度" if abs(candidate_object_z - standard_object_place_z) < 1e-9 else (
                f"降级工件下放高度，比标准高 {(candidate_object_z - standard_object_place_z) * 1000.0:.1f} mm"
            )
            print(
                f"RRTstar 预规划工件下放候选：工件中心 xy={object_place_xy}，{height_note}；"
                f"反算 TCP xyz={tcp_place_target[:3]}...",
                flush=True,
            )
            try:
                trajectory, frequency = preplan_tcp_trajectory(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target_tcp=tcp_place_target, planner_kind=RRTSTAR_PLANNER_KIND,
                    max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                    stage="阶段 6：RRTstar 工件下放候选",
                    rrtstar_goal_tolerance=True,
                )
            except Exception as exc:
                place_failures.append(
                    f"object_xy={object_place_xy}, object_z={candidate_object_z:.6f}: {exc}"
                )
                continue
            selected_object_place_xy = object_place_xy
            selected_tcp_place_target = tcp_place_target
            selected_place_trajectory = trajectory
            selected_place_frequency = frequency
            selected_place_z = candidate_object_z
            print(
                f"选择 RRTstar 可下放工件中心 xy(m)={object_place_xy}，z={candidate_object_z:.6f}；"
                f"反算 TCP xyz(m)={tcp_place_target[:3]}；"
                f"已保存 {len(trajectory)} 个关节路点的下放轨迹。"
            )
            break
        if (selected_object_place_xy is None or selected_tcp_place_target is None
                or selected_place_trajectory is None or selected_place_frequency is None
                or selected_place_z is None):
            # All offset candidates either failed the ROI test or did not
            # plan.  Keep the fallback consistent with the placement search:
            # lower the object centre at the original grasp-point XY, not at
            # the turn station, using the same closed-grasp inverse mapping.
            fallback_object_target = centre_xy + [
                standard_object_place_z, *selected_turn_object_centre[3:]
            ]
            fallback_transfer_target = compose_pose(
                fallback_object_target, inverse_pose(transfer_from_object_centre)
            )
            fallback_eef_target = compose_pose(fallback_transfer_target, transfer_from_eef)
            fallback_tcp_target = compose_pose(fallback_eef_target, tcp_offset)
            print(
                f"所有偏移工件下放候选不可用；尝试将工件中心放到初始抓取点 xy={centre_xy} "
                f"和标准 z={standard_object_place_z:.6f}...",
                flush=True,
            )
            try:
                trajectory, frequency = preplan_tcp_trajectory(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id, ik_frame=ik_frame,
                    tcp_offset=tcp_offset, target_tcp=fallback_tcp_target, planner_kind=RRTSTAR_PLANNER_KIND,
                    max_vel=args.max_vel, max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                    stage="阶段 6：RRTstar 工件原地兜底下放",
                    rrtstar_goal_tolerance=True,
                )
            except Exception as exc:
                details = "；".join(place_failures[-12:])
                raise RuntimeError(
                    f"工件下放操作无法规划，初始抓取点原地下放也失败：{exc}。最后偏移候选失败原因：{details}"
                ) from exc
            selected_object_place_xy = centre_xy.copy()
            selected_tcp_place_target = fallback_tcp_target
            selected_place_trajectory = trajectory
            selected_place_frequency = frequency
            selected_place_z = standard_object_place_z
            print(
                f"已选择初始抓取点工件中心下放 xy(m)={selected_object_place_xy}，z={selected_place_z:.6f}；"
                f"反算 TCP xyz(m)={selected_tcp_place_target[:3]}；"
                f"已保存 {len(trajectory)} 个关节路点的下放轨迹。"
            )
        print(
            f"阶段 6：执行已保存的下放轨迹，目标工件中心 xy(m)={selected_object_place_xy}，"
            f"z={selected_place_z:.6f}，目标 TCP xyz(m)={selected_tcp_place_target[:3]}，"
            f"{len(selected_place_trajectory)} 个关节路点。"
        )
        try:
            arm.follow_trajectory(
                arm_id=args.arm_id, trajectory=selected_place_trajectory, sample_frequency=selected_place_frequency * args.trajectory_execution_speed_scale,
            )
        except Exception as exc:
            raise RuntimeError(f"下放轨迹执行失败：{exc}") from exc
        set_gripper_state(gripper, arm_id=args.arm_id, target_open=True, timeout_s=args.gripper_timeout_s,
                          tolerance=args.gripper_position_tolerance)
        released_ik = pose_from_lookup_transform(
            planner.lookup_transform(args.frame_id, ik_frame).get("data")
        )
        released_tcp = compose_pose(released_ik, tcp_offset)
        post_release_lift_target = [
            released_tcp[0], released_tcp[1],
            released_tcp[2] + args.post_release_lift_height_m,
            *released_tcp[3:],
        ]
        print(
            "阶段 7：松爪后从当前 TCP 保持 XY/姿态，"
            f"沿 body_base +Z 使用 RRTstar（rrtstar_goal_tolerance=True）抬起 "
            f"{args.post_release_lift_height_m * 1000.0:.1f} mm。"
        )
        try:
            trajectory, frequency = preplan_tcp_trajectory(
                planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                ik_frame=ik_frame, tcp_offset=tcp_offset,
                target_tcp=post_release_lift_target, planner_kind=RRTSTAR_PLANNER_KIND,
                max_vel=args.max_vel, max_acc=args.max_acc,
                timeout_s=RRTCONNECT_TIMEOUT_S,
                stage="阶段 7：松爪后 RRTstar 沿 body_base +Z 抬起",
                rrtstar_goal_tolerance=True,
            )
            arm.follow_trajectory(
                arm_id=args.arm_id, trajectory=trajectory,
                sample_frequency=frequency * args.trajectory_execution_speed_scale,
            )
        except Exception as exc:
            raise RuntimeError(f"阶段 7：松爪后 RRTstar 抬起规划或执行失败: {exc}") from exc
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
