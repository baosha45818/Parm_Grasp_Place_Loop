#!/usr/bin/env python3
"""Reorient a TOML-configured workpiece from Q2--Q4 into Q1.

No taught pose is used as a destination.  The planner reads the live native
right_eef TCP, reads one semantic 6-D workpiece pose from a fresh
mask_pose JSON, constructs a configured semantic grasp pose, generates
tolerant six-dimensional pregrasp candidates, and selects a plannable one.
The shared pregrasp point comes from the first configured grasp point.  From
there, exact RRTstar targets are tried for every configured grasp point and
its paired workpiece -Z extra descent.  All editable motion parameters come
from one workpiece TOML file.

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
DEFAULT_RRTSTAR_GOAL_TOLERANCE = False
POSE_OUTPUT_ROOT = MODULE_ROOT.parents[1] / "output/module_4_2"
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
    roll_candidates_deg: list[float], pitch_candidates_deg: list[float],
    yaw_candidates_deg: list[float],
) -> list[tuple[list[float], int]]:
    """Return configured local-TCP RPY candidates around both symmetric poses.

    The semantic axes are full 3-D directions. Tool +Y may be either +Y or
    -Y of the workpiece because the two jaws are symmetric. The nominal TCP
    axes are +X=-workpiece Z, +Y=±workpiece Y, and +Z=±workpiece X.
    The TOML roll, pitch and yaw offsets are right-multiplied onto that nominal
    orientation, so they are rotations about the TCP's own local X/Y/Z axes.
    Their list order is the planning priority: sign, roll, pitch, then yaw.
    """
    candidates: list[tuple[list[float], int]] = []
    for sign in (1, -1):
        try:
            nominal = nominal_tcp_orientation(workpiece_y_base, workpiece_z_base, sign)
        except ValueError:
            continue
        for roll_deg, pitch_deg, yaw_deg in product(
            roll_candidates_deg, pitch_candidates_deg, yaw_candidates_deg,
        ):
            local_delta = rpy_to_rotvec([
                math.radians(roll_deg),
                math.radians(pitch_deg),
                math.radians(yaw_deg),
            ])
            rotvec = compose_pose(
                [0.0, 0.0, 0.0] + nominal,
                [0.0, 0.0, 0.0] + local_delta,
            )[3:]
            if all(any(abs(a - b) > 1e-8 for a, b in zip(rotvec, known[0])) for known in candidates):
                candidates.append((rotvec, sign))
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
        f"{stage}：驱动状态的 right_eef 未到预抓取点，禁止 RRTstar 下探；"
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
        enabled_goal_tolerance = (DEFAULT_RRTSTAR_GOAL_TOLERANCE
                                  if rrtstar_goal_tolerance is None else rrtstar_goal_tolerance)
        if planner_kind == RRTSTAR_PLANNER_KIND and enabled_goal_tolerance:
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
    "--grasp-point-semantic-offset-m", "--grasp-point-tcp-local-rpy-offset-candidates-deg",
    "--pregrasp-positive-workpiece-z-m",
    "--pregrasp-positive-workpiece-x-m", "--pregrasp-positive-workpiece-y-m",
    "--final-extra-descent-negative-workpiece-z-m",
    "--pregrasp-position-tolerance-m",
    "--servo-linear-speed-mps", "--servo-angular-speed-radps",
    "--servo-max-linear-accel-mps2", "--servo-max-angular-accel-radps2",
    "--servo-hz", "--servo-position-tolerance-m",
    "--servo-orientation-tolerance-rad", "--servo-settle-timeout-s",
    "--place-tcp-extra-clearance-m", "--lift-height-m", "--post-release-lift-height-m",
    "--planning-timeout-s", "--q1-yaw-min-deg", "--q1-yaw-max-deg",
    "--q1-yaw-search-step-deg", "--q1-preferred-yaw-deg",
    "--turn-attitude-search-step-deg",
    "--turn-relocation-step-m", "--post-grasp-turn-search-negative-base-x-m",
    "--post-grasp-turn-search-negative-base-y-m", "--turn-station-tcp-height-m",
    "--turn-search-grid-step-m", "--turn-positive-x-min-m",
    "--turn-positive-x-max-m", "--rrt-place-parallelism",
    "--rrt-turn-parallelism", "--planner", "--turn-planner",
    "--max-vel", "--max-acc", "--start-sync-retries",
    "--gripper-timeout-s", "--gripper-position-tolerance",
    "--grasp-minimum-close-travel", "--grasp-contact-settle-s",
    "--trajectory-execution-speed-scale",
    "--rrtstar-goal-tolerance", "--lift-rrtstar-goal-tolerance",
    "--pregrasp-rrtstar-goal-tolerance", "--turn-rrtstar-goal-tolerance",
    "--grasp-rrtstar-goal-tolerance",
    "--place-rrtstar-goal-tolerance", "--place-fallback-rrtstar-goal-tolerance",
    "--release-rrtstar-goal-tolerance",
})
def load_reorient_config(path: Path) -> dict[str, object]:
    """Read Module 7's single editable TOML file without requiring Python 3.11."""
    tables: dict[str, dict[str, object]] = {}
    current_table: dict[str, object] | None = None
    pending_key: str | None = None
    pending_value: list[str] = []

    def store(table: dict[str, object], key: str, raw: str, line_number: int) -> None:
        if key in table:
            raise ValueError(f"调整 TOML 键重复: {path}:{line_number}: {key}")
        try:
            table[key] = True if raw == "true" else False if raw == "false" else ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"调整 TOML 无法解析: {path}:{line_number}: {raw!r}") from exc

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取调整 TOML {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if pending_key is not None:
            pending_value.append(line)
            joined = " ".join(pending_value)
            if joined.count("[") == joined.count("]"):
                assert current_table is not None
                store(current_table, pending_key, joined, line_number)
                pending_key, pending_value = None, []
            continue
        if line.startswith("[") and line.endswith("]"):
            table_name = line[1:-1].strip()
            if not table_name:
                raise ValueError(f"调整 TOML 表名为空: {path}:{line_number}")
            if table_name in tables:
                raise ValueError(f"调整 TOML 表重复: {path}:{line_number}: [{table_name}]")
            current_table = tables[table_name] = {}
            continue
        if current_table is None or "=" not in line:
            raise ValueError(f"调整 TOML 只支持 [table] 内的 key = value: {path}:{line_number}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise ValueError(f"调整 TOML 键为空: {path}:{line_number}")
        if raw_value.count("[") != raw_value.count("]"):
            pending_key, pending_value = key, [raw_value]
        else:
            store(current_table, key, raw_value, line_number)
    if pending_key is not None:
        raise ValueError(f"调整 TOML 数组未闭合: {path}")

    meta = tables.get("meta", {})
    workpiece = meta.get("workpiece")
    if not isinstance(workpiece, str) or not workpiece.strip():
        raise ValueError("调整 TOML 的 [meta] 必须包含非空 workpiece")
    configured: dict[str, object] = {}
    grasp_rpy_axes: object | None = None
    for table_name, table in tables.items():
        if table_name == "meta":
            continue
        for key, value in table.items():
            if key == "grasp_point_tcp_local_rpy_candidates_deg":
                grasp_rpy_axes = value
                continue
            option = "--" + key.replace("_", "-")
            # Candidate arrays are consumed separately by load_turn_candidates.
            # Only argparse-backed runtime knobs belong to reorient_arguments.
            if option not in CONFIGURABLE_OPTIONS or (table_name == "turn" and key == "planner"):
                continue
            if option in configured:
                raise ValueError(f"调整 TOML 参数重复: {option}")
            configured[option] = value
    if grasp_rpy_axes is not None:
        if not isinstance(grasp_rpy_axes, list) or not grasp_rpy_axes:
            raise ValueError("grasp_point_tcp_local_rpy_candidates_deg 必须是非空抓取点数组")
        converted_rpy: list[list[list[float]]] = []
        try:
            for point_axes in grasp_rpy_axes:
                if not isinstance(point_axes, list) or len(point_axes) != 3:
                    raise ValueError("每个抓取点必须包含 roll/pitch/yaw 三个数组")
                roll, pitch, yaw = ([float(v) for v in axis] for axis in point_axes)
                if not roll or not pitch or not yaw:
                    raise ValueError("每个抓取点的 roll/pitch/yaw 数组不能为空")
                converted_rpy.append([[r, p, y] for r in roll for p in pitch for y in yaw])
        except (TypeError, ValueError) as exc:
            raise ValueError("抓取点 RPY 候选必须是三组有限数值数组") from exc
        configured["--grasp-point-tcp-local-rpy-offset-candidates-deg"] = converted_rpy
    rpy_prefix = "grasp_point_"
    rpy_suffix = "_tcp_local_rpy_offset_candidates_deg"
    rpy_entries = []
    for table in tables.values():
        for key, value in table.items():
            if key.startswith(rpy_prefix) and key.endswith(rpy_suffix):
                try:
                    index = int(key[len(rpy_prefix):-len(rpy_suffix)])
                except ValueError as exc:
                    raise ValueError(f"抓取点 RPY 参数编号无效: {key}") from exc
                rpy_entries.append((index, value))
    if rpy_entries:
        rpy_entries.sort()
        if [index for index, _ in rpy_entries] != list(range(1, len(rpy_entries) + 1)):
            raise ValueError("抓取点 RPY 参数必须从 grasp_point_1 连续编号")
        configured["--grasp-point-tcp-local-rpy-offset-candidates-deg"] = [value for _, value in rpy_entries]
    return {"workpiece": workpiece, "reorient_arguments": configured}


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
        if option == "--grasp-point-semantic-offset-m":
            if not isinstance(raw_value, list) or not raw_value:
                raise ValueError(f"配置参数 {option} 必须是非空三维点列表")
            try:
                value = [
                    [float(component) for component in point]
                    for point in raw_value
                    if isinstance(point, list) and len(point) == 3
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"配置参数 {option} 必须是非空三维数值点列表") from exc
            if len(value) != len(raw_value) or any(
                not math.isfinite(component) for point in value for component in point
            ):
                raise ValueError(f"配置参数 {option} 必须是非空三维有限数点列表")
            setattr(args, "grasp_point_semantic_offset_m", value)
            continue
        if option == "--grasp-point-tcp-local-rpy-offset-candidates-deg":
            if not isinstance(raw_value, list) or not raw_value:
                raise ValueError(f"配置参数 {option} 必须是每个抓取点对应的非空 RPY 候选列表")
            value: list[list[list[float]]] = []
            try:
                for point_candidates in raw_value:
                    if not isinstance(point_candidates, list) or not point_candidates:
                        raise ValueError("每个抓取点必须至少有一个 RPY 候选")
                    converted: list[list[float]] = []
                    for candidate in point_candidates:
                        if not isinstance(candidate, list) or len(candidate) != 3:
                            raise ValueError("每个 RPY 候选必须恰好包含 roll/pitch/yaw 三项")
                        converted.append([float(component) for component in candidate])
                    value.append(converted)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"配置参数 {option} 必须是每个抓取点对应的非空三维数值候选列表"
                ) from exc
            if any(
                not math.isfinite(component)
                for point_candidates in value
                for candidate in point_candidates
                for component in candidate
            ):
                raise ValueError(f"配置参数 {option} 含非有限数")
            setattr(args, "grasp_point_tcp_local_rpy_offset_candidates_deg", value)
            continue
        if option == "--final-extra-descent-negative-workpiece-z-m":
            if not isinstance(raw_value, list) or not raw_value:
                raise ValueError(f"配置参数 {option} 必须是非空数值列表")
            try:
                value = [float(item) for item in raw_value]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"配置参数 {option} 必须是非空数值列表") from exc
            if not all(math.isfinite(item) for item in value):
                raise ValueError(f"配置参数 {option} 含非有限数")
            setattr(args, "final_extra_descent_negative_workpiece_z_m", value)
            continue
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
    """Load one workpiece's editable pregrasp, turn and placement candidates.

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
            # The single-file layout keeps the pregrasp planner and the turn
            # candidate planner in their respective motion stages.  They
            # deliberately share the same key and must agree.
            if key == "planner":
                try:
                    duplicate = ast.literal_eval(raw)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"翻转候选 TOML 无法解析: {path}:{line_number}: {raw!r}") from exc
                if table[key] == duplicate:
                    return
            raise ValueError(f"翻转候选配置键重复: {path}:{line_number}: {key}")
        try:
            table[key] = True if raw == "true" else False if raw == "false" else ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"翻转候选 TOML 无法解析: {path}:{line_number}: {raw!r}") from exc

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
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
            # These three pregrasp RPY arrays belong to motion stage 1, while
            # the remaining candidate arrays belong to stage 4.
            in_turn_table = line in {"[pregrasp]", "[turn]", "[place]"}
            continue
        if not in_turn_table:
            continue
        if "=" not in line:
            raise ValueError(f"翻转候选配置 [turn] 内只支持 key = value: {path}:{line_number}")
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
        "pregrasp_roll_candidates_deg": vector("pregrasp_roll_candidates_deg"),
        "pregrasp_pitch_candidates_deg": vector("pregrasp_pitch_candidates_deg"),
        "pregrasp_yaw_candidates_deg": vector("pregrasp_yaw_candidates_deg"),
        "x_offset_candidates_m": vector("x_offset_candidates_m"),
        "y_offset_candidates_m": vector("y_offset_candidates_m"),
        "z_offset_candidates_m": vector("z_offset_candidates_m"),
        "roll_candidates_deg": vector("roll_candidates_deg"),
        "pitch_candidates_deg": vector("pitch_candidates_deg"),
        "yaw_candidates_deg": vector("yaw_candidates_deg"),
        "place_x_offset_candidates_m": vector("place_x_offset_candidates_m"),
        "place_y_offset_candidates_m": vector("place_y_offset_candidates_m"),
    }
    if any(abs(value) > 360.0 for name, candidate in values.items()
           if name.endswith("_deg") for value in candidate):
        raise ValueError("翻转候选 RPY 不能超过 360 度")
    if any(abs(value) > 45.0 for name, candidate in values.items()
           if name.startswith("pregrasp_") and name.endswith("_deg")
           for value in candidate):
        raise ValueError("预抓取候选局部 RPY 不能超过 ±45 度")
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

    place_x_offsets = args.turn_candidates["place_x_offset_candidates_m"]
    place_y_offsets = args.turn_candidates["place_y_offset_candidates_m"]
    assert isinstance(place_x_offsets, list) and isinstance(place_y_offsets, list)
    stations = [
        [origin_xy[0] + x_offset, origin_xy[1] + y_offset]
        for y_offset in place_y_offsets for x_offset in place_x_offsets
    ]
    q1_count = int(math.floor((args.q1_yaw_max_deg - args.q1_yaw_min_deg) / args.q1_yaw_search_step_deg + 1e-9))
    q1_yaws = [args.q1_yaw_min_deg + index * args.q1_yaw_search_step_deg for index in range(q1_count + 1)]
    if q1_yaws[-1] < args.q1_yaw_max_deg - 1e-9:
        q1_yaws.append(args.q1_yaw_max_deg)
    q1_yaws = prioritise_q1_yaws(q1_yaws + [args.q1_preferred_yaw_deg], args.q1_preferred_yaw_deg)
    attitude_offsets = [(0.0, 0.0)]
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
                        help="可选的单一 Module 7 TOML 配置；默认使用该工件的 TOML")
    parser.add_argument(
        "--turn-candidates-config", type=Path,
                        help="兼容选项：覆盖同一 TOML 内 [turn] 的候选；通常不需要使用",
    )
    parser.add_argument("--arm-id", type=int, choices=(2,), required=True)
    parser.add_argument("--workpiece-pose-file", type=Path,
                        help="可选的位姿 JSON 覆盖；默认使用 module_4_2 对应工件的最新结果")
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument("--grasp-point-semantic-offset-m", nargs=3, type=float,
                        default=((-0.025, 0.0, 0.01),), metavar=("DX", "DY", "DZ"),
                        help="配置文件提供按优先级排列的工件语义坐标抓取点列表")
    parser.add_argument(
        "--grasp-point-tcp-local-rpy-offset-candidates-deg", nargs=3, type=float,
        default=(((0.0, 0.0, 0.0),),), metavar=("ROLL", "PITCH", "YAW"),
        help="配置文件提供每个抓取点对应的 TCP 局部 RPY 偏移候选列表",
    )
    parser.add_argument("--pregrasp-positive-workpiece-x-m", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--pregrasp-positive-workpiece-y-m", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--pregrasp-positive-workpiece-z-m", type=float, nargs="+", default=(0.07,))
    parser.add_argument("--final-extra-descent-negative-workpiece-z-m", type=float, nargs="+",
                        default=(0.0,), help="与语义抓取点列表一一对应的工件 -Z 额外下探值")
    parser.add_argument("--pregrasp-position-tolerance-m", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                        help="到达预抓取点后，沿实际 TCP 局部 X/Y/Z 的固定平移（m）")
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
    parser.add_argument("--post-release-lift-height-m", type=float, nargs="+", default=(0.020,),
                        help="松爪后 TCP 沿 body_base +Z 直线抬起的距离（m）")
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
    parser.add_argument("--turn-positive-x-min-m", type=float, default=0.0)
    parser.add_argument("--turn-positive-x-max-m", type=float, default=0.200)
    parser.add_argument("--rrt-place-parallelism", type=int, default=RRTCONNECT_PLACE_MAX_PARALLELISM)
    parser.add_argument("--rrt-turn-parallelism", type=int, default=RRTCONNECT_PLACE_MAX_PARALLELISM)
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
    parser.add_argument("--rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=False,
                        help="统一控制 Module 7 所有 RRTstar 请求是否传 rrtstar_goal_tolerance")
    parser.add_argument("--pregrasp-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grasp-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--turn-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lift-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--place-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--place-fallback-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--release-rrtstar-goal-tolerance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-sync-retries", type=int, default=3)
    parser.add_argument("--gripper-timeout-s", type=float, default=5.0)
    parser.add_argument("--gripper-position-tolerance", type=float, default=5.0)
    parser.add_argument("--grasp-minimum-close-travel", type=float, default=5.0)
    parser.add_argument("--grasp-contact-settle-s", type=float, default=0.5)
    parser.add_argument("--calibrate-grasp-xy", action="store_true")
    parser.add_argument("--calibration-record-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.config = args.config or TURN_CANDIDATE_CONFIGS[args.workpiece]
    try:
        apply_reorient_config(parser, args)
        expected_name = f"workpiece_{args.workpiece}"
        if args.workpiece_name != expected_name:
            raise ValueError(
                f"调整配置属于 {args.workpiece_name!r}，与 --workpiece {args.workpiece} 不一致"
            )
        args.workpiece_pose_file = args.workpiece_pose_file or newest_workpiece_pose_file(args.workpiece)
        validate_pose_workpiece(args.workpiece_pose_file, args.workpiece)
        args.turn_candidates_config = args.turn_candidates_config or args.config
        args.turn_candidates = load_turn_candidates(args.turn_candidates_config)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    global DEFAULT_RRTSTAR_GOAL_TOLERANCE
    args = parse_args()
    DEFAULT_RRTSTAR_GOAL_TOLERANCE = args.pregrasp_rrtstar_goal_tolerance
    if not args.workpiece_pose_file.is_file():
        print(f"错误: 工件位姿文件不存在: {args.workpiece_pose_file}", file=sys.stderr)
        return 1
    positive = (*args.pregrasp_positive_workpiece_z_m, *args.lift_height_m,
                args.turn_search_grid_step_m,
                args.turn_relocation_step_m)
    if (any(value <= 0.0 for value in positive)
            or args.place_tcp_extra_clearance_m < 0.0):
        print("错误: 高度和步长必须大于 0，放置余量必须非负", file=sys.stderr)
        return 1
    if not (
        len(args.grasp_point_semantic_offset_m)
        == len(args.final_extra_descent_negative_workpiece_z_m)
        == len(args.grasp_point_tcp_local_rpy_offset_candidates_deg)
    ):
        print(
            "错误: 抓取点、额外下探值和抓取点 TCP 局部 RPY 候选列表必须一一对应且长度相同",
            file=sys.stderr,
        )
        return 1
    if (any(abs(value) > 0.2 for point in args.grasp_point_semantic_offset_m for value in point)
            or any(abs(value) > 0.1 for value in args.final_extra_descent_negative_workpiece_z_m)):
        print("错误: 抓取点语义偏移或额外下探量超出合理范围", file=sys.stderr)
        return 1
    if any(
        abs(value) > 45.0
        for point_candidates in args.grasp_point_tcp_local_rpy_offset_candidates_deg
        for candidate in point_candidates
        for value in candidate
    ):
        print("错误: 抓取点 TCP 局部 RPY 偏移候选不得超过 ±45 度", file=sys.stderr)
        return 1
    if any(abs(value) > 0.1 for value in args.pregrasp_position_tolerance_m):
        print("错误: 预抓取后 TCP 局部 XYZ 平移不得超过 ±0.1 m", file=sys.stderr)
        return 1
    if any(
        min(args.pregrasp_positive_workpiece_z_m) + extra_descent <= 0.0
        for extra_descent in args.final_extra_descent_negative_workpiece_z_m
    ):
        print("错误: 预抓取到最终闭爪点必须保持正的工件 -Z 下探分量", file=sys.stderr)
        return 1
    if args.planning_timeout_s <= 0.0:
        print("错误: 预规划超时必须大于 0", file=sys.stderr)
        return 1
    if args.turn_attitude_search_step_deg <= 0.0:
        print("错误: 阶段 5 姿态搜索间隔必须大于 0", file=sys.stderr)
        return 1
    servo_positive = (
        args.servo_linear_speed_mps, args.servo_angular_speed_radps,
        args.servo_max_linear_accel_mps2, args.servo_max_angular_accel_radps2,
        args.servo_hz, args.servo_position_tolerance_m,
        args.servo_orientation_tolerance_rad, args.servo_settle_timeout_s,
    )
    if any(value <= 0.0 for value in servo_positive):
        print("错误: servo 速度、加速度、频率、容差和稳定超时必须大于 0", file=sys.stderr)
        return 1
    if any(value < 0.0 for value in args.post_release_lift_height_m):
        print("错误: 松爪后撤离高度候选不能小于 0", file=sys.stderr)
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
    if (args.turn_positive_x_min_m < 0.0
            or args.turn_positive_x_max_m < args.turn_positive_x_min_m):
        print("错误: 翻转区域距离无效", file=sys.stderr)
        return 1
    if abs(args.turn_station_tcp_height_m) > 0.5:
        print("错误: --turn-station-tcp-height-m 的相对高度偏移不能超过 0.5 m", file=sys.stderr)
        return 1
    if (args.post_grasp_turn_search_negative_base_x_m < 0.0
            or args.post_grasp_turn_search_negative_base_y_m < 0.0):
        print("错误: 翻转搜索前的 -X/-Y 后撤距离不能小于 0", file=sys.stderr)
        return 1
    try:
        workpiece_transform, red_yaw_deg, quadrant, table_plane_z, workpiece_item, calibration_path = load_workpiece(
            args.workpiece_pose_file, args.workpiece_index
        )
        # JSON offsets are expressed on the observed workpiece's own +X/+Y/+Z.
        semantic_local_grasps = [
            list(point) for point in args.grasp_point_semantic_offset_m
        ]
        semantic_grasp_points = [
            transform_point(workpiece_transform, point)
            for point in semantic_local_grasps
        ]
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
        # Planner targets are native right_eef poses.  Servo starts from the
        # driver's eef_pose, so it needs the fixed local driver->right_eef
        # transform above to represent exactly the same jaw-centre point.
        tcp_offset = [0.0] * 6
        servo_eef_offset = RIGHT_EEF_FROM_TRANSFER_BASE
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    # Recompute all absolute targets from this observation.  The shared
    # pregrasp uses candidate 1; each final target uses its own semantic XYZ
    # plus its paired offset along the observed workpiece Z axis.
    # The shared pregrasp entry is defined only by the first grasp point.
    grasp_xyz = semantic_grasp_points[0].copy()
    pregrasp_xyz_candidates = [
        [semantic_grasp_point[index] + x * workpiece_x_base[index] + y * workpiece_y_base[index]
         + z * workpiece_z_base[index] for index in range(3)]
        for semantic_grasp_point in semantic_grasp_points
        for x, y, z in product(args.pregrasp_positive_workpiece_x_m,
                               args.pregrasp_positive_workpiece_y_m,
                               args.pregrasp_positive_workpiece_z_m)
    ]
    pregrasp_xyz = pregrasp_xyz_candidates[0]
    final_grasp_xyz_candidates = [
        [
            semantic_grasp_point[index] - extra_descent * workpiece_z_base[index]
            for index in range(3)
        ]
        for semantic_grasp_point, extra_descent in zip(
            semantic_grasp_points, args.final_extra_descent_negative_workpiece_z_m
        )
    ]
    final_grasp_xyz = final_grasp_xyz_candidates[0]
    centre_xy = final_grasp_xyz[:2].copy()
    vision_grasp_xy = semantic_grasp_points[0][:2].copy()
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
        f"语义抓取点候选数={len(semantic_local_grasps)}；"
        f"参考 rpy 与本次工件一致(deg)="
        f"{[round(math.degrees(value), 3) for value in grasp_reference_rpy]}"
    )
    for candidate_index, (
        local_grasp, semantic_grasp_point, extra_descent, final_xyz, local_rpy_candidates,
    ) in enumerate(
        zip(
            semantic_local_grasps, semantic_grasp_points,
            args.final_extra_descent_negative_workpiece_z_m,
            final_grasp_xyz_candidates,
            args.grasp_point_tcp_local_rpy_offset_candidates_deg,
        ),
        start=1,
    ):
        print(
            f"抓取点候选 {candidate_index}/{len(semantic_local_grasps)}："
            f"semantic local(m)={[round(value, 9) for value in local_grasp]}；"
            f"body_base xyz(m)={[round(value, 9) for value in semantic_grasp_point]}；"
            f"沿工件 -Z 额外下探={extra_descent * 1000.0:+.1f} mm；"
            f"精确闭爪 xyz(m)={[round(value, 9) for value in final_xyz]}；"
            f"TCP 局部 RPY 偏移候选(deg)={local_rpy_candidates}。"
        )
    print(
        f"预抓取位置候选数={len(pregrasp_xyz_candidates)}；每个抓取点均使用同一组工件局部 X/Y/Z 偏移，首选候选为抓取点 1 的 "
        f"{[args.pregrasp_positive_workpiece_x_m[0], args.pregrasp_positive_workpiece_y_m[0], args.pregrasp_positive_workpiece_z_m[0]]} m 计算："
        f"xyz(m)={[round(value, 9) for value in pregrasp_xyz]}。"
    )
    if args.planner == RRTSTAR_PLANNER_KIND:
        print(
            "抓取前流程：RRTstar 精确规划到预抓取位姿（不传 goal tolerance），"
            "若驱动状态仍未到该点，则再以同样的精确 RRTstar 修正；"
            "之后从该预抓取位姿按抓取点列表顺序做无容差 RRTstar 精确下探规划。"
        )
    else:
        print(
            "抓取前流程：RRT 直接规划到预抓取位姿，"
            "锁定所选 TCP 姿态，再按抓取点列表顺序做无容差 RRTstar 精确下探规划。"
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
        pregrasp_total = (
            2 * len(turn["pregrasp_roll_candidates_deg"])
            * len(turn["pregrasp_pitch_candidates_deg"])
            * len(turn["pregrasp_yaw_candidates_deg"])
        )
        total = (len(turn["x_offset_candidates_m"])
                 * len(turn["y_offset_candidates_m"])
                 * len(turn["z_offset_candidates_m"])
                 * len(turn["roll_candidates_deg"])
                 * len(turn["pitch_candidates_deg"])
                 * len(turn["yaw_candidates_deg"]))
        print(
            f"工件{args.workpiece}预抓取/闭爪后调整候选 TOML：{args.turn_candidates_config}；"
            f"预抓取 sign/r/p/y 数量=2/"
            f"{len(turn['pregrasp_roll_candidates_deg'])}/"
            f"{len(turn['pregrasp_pitch_candidates_deg'])}/"
            f"{len(turn['pregrasp_yaw_candidates_deg'])}，过滤前共 {pregrasp_total} 个；"
            f"规划器={turn['planner']}；"
            f"x/y/z/r/p/y 数量={len(turn['x_offset_candidates_m'])}/"
            f"{len(turn['y_offset_candidates_m'])}/{len(turn['z_offset_candidates_m'])}/"
            f"{len(turn['roll_candidates_deg'])}/{len(turn['pitch_candidates_deg'])}/"
            f"{len(turn['yaw_candidates_deg'])}，共 {total} 个。"
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
            roll_candidates_deg=list(args.turn_candidates["pregrasp_roll_candidates_deg"]),
            pitch_candidates_deg=list(args.turn_candidates["pregrasp_pitch_candidates_deg"]),
            yaw_candidates_deg=list(args.turn_candidates["pregrasp_yaw_candidates_deg"]),
        )
        if not orientation_candidates:
            raise RuntimeError("无法构造 TCP +X=-工件 Z、+Y=±工件 Y 的预抓取姿态")
        candidates: list[tuple[list[float], list[float], int]] = [
            (position, orientation, y_sign)
            for position in pregrasp_xyz_candidates
            for orientation, y_sign in orientation_candidates
        ]
        print(
            f"开始按 TOML 列表顺序预规划 {len(candidates)} 个预抓取六维候选；"
            "预抓取点不再搜索 XYZ 偏差；到达后将沿实际 TCP 局部 X/Y/Z 固定平移(m)="
            f"{list(args.pregrasp_position_tolerance_m)}，"
            "绕 TCP 自身 X/Y/Z 的候选数量="
            f"{len(args.turn_candidates['pregrasp_roll_candidates_deg'])}/"
            f"{len(args.turn_candidates['pregrasp_pitch_candidates_deg'])}/"
            f"{len(args.turn_candidates['pregrasp_yaw_candidates_deg'])}；"
            f"单次 RPC 最多等待 {args.planning_timeout_s:g} 秒。",
            flush=True,
        )

        chosen_xyz: list[float] | None = None
        chosen_orientation: list[float] | None = None
        chosen_y_sign: int | None = None
        failures: list[str] = []
        for index, (candidate_xyz, orientation, y_sign) in enumerate(candidates, start=1):
            rpy = rotvec_to_rpy(orientation)
            turn_angle_deg = math.degrees(orientation_distance_rad(current_tcp[3:], orientation))
            print(
                f"预规划候选 {index}/{len(candidates)}：相对当前转动={turn_angle_deg:.2f}°；"
                f"夹爪方向=工件 {'+' if y_sign > 0 else '-'}Y；"
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
                chosen_y_sign = y_sign
                print(f"候选 {index}/{len(candidates)} 可规划（{detail}）；采用该 TOML 最高优先级可达候选。")
                break
            if len(failures) < 8:
                failures.append(f"候选 {index}: {detail}")
        if (chosen_xyz is None or chosen_orientation is None
                or chosen_y_sign is None):
            raise RuntimeError("所有实时预抓取候选都不可规划。前几个失败原因：" + "；".join(failures))
        print("选择 TOML 列表中最高优先级的可规划候选："
              f"夹爪方向=工件 {'+' if chosen_y_sign > 0 else '-'}Y。")

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
            # descent planning.  This RRTstar request deliberately omits
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
                    rrtstar_goal_tolerance=args.pregrasp_rrtstar_goal_tolerance,
                    stage="阶段 1c：精确 RRTstar 预抓取修正",
                )
                arm.follow_trajectory(
                    arm_id=args.arm_id, trajectory=correction_trajectory,
                    sample_frequency=correction_frequency * args.trajectory_execution_speed_scale,
                )
            except Exception as exc:
                raise RuntimeError(
                    "阶段 1c 精确 RRTstar 预抓取修正不可规划或执行失败；"
                    "不会开始 RRTstar 下探。"
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
        # The planning start is the live right_eef state at pregrasp, but every
        # grasp endpoint must retain the exact selected TCP/workpiece-axis
        # relationship.  Do not turn the small measured tracking error at the
        # pregrasp into a new grasp orientation.
        locked_orientation = selected_pregrasp_target[3:].copy()
        chosen_rpy = rotvec_to_rpy(locked_orientation)
        print(
            f"锁定所选 TCP/工件对应姿态 rpy(deg)="
            f"{[round(math.degrees(value), 2) for value in chosen_rpy]}；"
            f"RRTstar 规划起点使用实际预抓取 right_eef xyz(m)="
            f"{[round(value, 6) for value in reached_pregrasp_tcp[:3]]}；"
            "抓取终点使用该精确姿态。"
        )
        selected_grasp_index: int | None = None
        selected_final_grasp_xyz: list[float] | None = None
        selected_grasp_orientation: list[float] | None = None
        selected_grasp_local_rpy_offset_deg: list[float] | None = None
        selected_descent_trajectory: list | None = None
        selected_descent_frequency: float | None = None
        descent_failures: list[str] = []
        for candidate_index, (
            semantic_local_grasp, semantic_grasp_point, extra_descent, candidate_final_xyz,
            local_rpy_candidates_deg,
        ) in enumerate(
            zip(
                semantic_local_grasps, semantic_grasp_points,
                args.final_extra_descent_negative_workpiece_z_m,
                final_grasp_xyz_candidates,
                args.grasp_point_tcp_local_rpy_offset_candidates_deg,
            ),
            start=1,
        ):
            for rpy_candidate_index, local_rpy_offset_deg in enumerate(
                local_rpy_candidates_deg, start=1,
            ):
                # Right multiplication applies roll/pitch/yaw about the TCP's
                # own local axes while retaining the selected workpiece-relative
                # pregrasp orientation as the zero-offset reference.
                local_orientation_delta = rpy_to_rotvec([
                    math.radians(value) for value in local_rpy_offset_deg
                ])
                candidate_orientation = compose_pose(
                    [0.0, 0.0, 0.0] + locked_orientation,
                    [0.0, 0.0, 0.0] + local_orientation_delta,
                )[3:]
                # No trajectory has been executed yet, so every request starts
                # from the same live right_eef state at the shared pregrasp.
                grasp_target = candidate_final_xyz + candidate_orientation
                stage = (
                    f"阶段 2：精确 RRTstar 抓取点 {candidate_index}/"
                    f"{len(final_grasp_xyz_candidates)}，姿态候选 "
                    f"{rpy_candidate_index}/{len(local_rpy_candidates_deg)}"
                )
                print(
                    f"{stage}：semantic local(m)="
                    f"{[round(value, 9) for value in semantic_local_grasp]}；"
                    f"额外工件 -Z={extra_descent * 1000.0:+.1f} mm；"
                    f"right_eef 目标 xyz(m)="
                    f"{[round(value, 6) for value in candidate_final_xyz]}；"
                    f"TCP 局部 RPY 偏移(deg)={local_rpy_offset_deg}；"
                    "不启用 goal tolerance。",
                    flush=True,
                )
                try:
                    trajectory, frequency = preplan_tcp_trajectory(
                        planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                        ik_frame=ik_frame, tcp_offset=tcp_offset, target_tcp=grasp_target,
                        planner_kind=RRTSTAR_PLANNER_KIND, max_vel=args.max_vel,
                        max_acc=args.max_acc, timeout_s=args.planning_timeout_s,
                        stage=stage, rrtstar_goal_tolerance=args.grasp_rrtstar_goal_tolerance,
                    )
                except Exception as exc:
                    descent_failures.append(
                        f"抓取点 {candidate_index} 姿态 {local_rpy_offset_deg}: {exc}"
                    )
                    print(f"{stage} 不可规划（{exc}）；继续下一个候选。", flush=True)
                    continue
                selected_grasp_index = candidate_index - 1
                selected_final_grasp_xyz = candidate_final_xyz.copy()
                selected_grasp_orientation = candidate_orientation
                selected_grasp_local_rpy_offset_deg = list(local_rpy_offset_deg)
                selected_descent_trajectory = trajectory
                selected_descent_frequency = frequency
                print(
                    f"{stage} 可规划；采用列表中第一个可达抓取位姿，"
                    f"保存 {len(trajectory)} 个关节路点。"
                )
                break
            if selected_grasp_index is not None:
                break
        if (selected_grasp_index is None or selected_final_grasp_xyz is None
                or selected_grasp_orientation is None
                or selected_grasp_local_rpy_offset_deg is None
                or selected_descent_trajectory is None
                or selected_descent_frequency is None):
            raise RuntimeError(
                "阶段 2：所有抓取点的无容差 RRTstar 下探均不可规划："
                + "；".join(descent_failures)
            )
        vision_grasp_xy = semantic_grasp_points[selected_grasp_index][:2].copy()
        centre_xy = selected_final_grasp_xyz[:2].copy()
        grasp_z = selected_final_grasp_xyz[2]
        lift_z = grasp_z + args.lift_height_m[0]
        selected_extra_descent = args.final_extra_descent_negative_workpiece_z_m[selected_grasp_index]
        chosen_rpy = rotvec_to_rpy(selected_grasp_orientation)
        descent_distance = math.sqrt(sum(
            (selected_final_grasp_xyz[axis] - reached_pregrasp_tcp[axis]) ** 2
            for axis in range(3)
        ))
        print(
            f"选择抓取点 {selected_grasp_index + 1}/{len(final_grasp_xyz_candidates)}；"
            f"额外工件 -Z={selected_extra_descent * 1000.0:+.1f} mm；"
            f"TCP 局部 RPY 偏移(deg)={selected_grasp_local_rpy_offset_deg}；"
            f"从实际预抓取点到精确闭爪点的三维距离="
            f"{descent_distance * 1000.0:.1f} mm。"
        )
        print(
            "阶段 2：执行已保存的无容差 RRTstar 轨迹，从预抓取位姿到精确闭爪位姿。"
        )
        arm.follow_trajectory(
            arm_id=args.arm_id, trajectory=selected_descent_trajectory,
            sample_frequency=(
                selected_descent_frequency * args.trajectory_execution_speed_scale
            ),
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
                "selected_grasp_candidate_index": selected_grasp_index,
                "selected_grasp_semantic_offset_m": semantic_local_grasps[selected_grasp_index],
                "selected_extra_descent_negative_workpiece_z_m": selected_extra_descent,
                "selected_grasp_tcp_local_rpy_offset_deg": selected_grasp_local_rpy_offset_deg,
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
            print(
                f"校准模式：已下探到抓取点候选 {selected_grasp_index + 1}，"
                "未闭爪、未翻转；机械臂保持当前位置。"
            )
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
            "selected_grasp_candidate_index": selected_grasp_index,
            "selected_grasp_semantic_offset_m": semantic_local_grasps[selected_grasp_index],
            "selected_extra_descent_negative_workpiece_z_m": selected_extra_descent,
            "selected_grasp_tcp_local_rpy_offset_deg": selected_grasp_local_rpy_offset_deg,
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
            f"保持 XY/姿态，以 RRTstar（rrtstar_goal_tolerance={args.lift_rrtstar_goal_tolerance}）按配置列表顺序规划抬起；"
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
                    rrtstar_goal_tolerance=args.lift_rrtstar_goal_tolerance,
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
                        rrtstar_goal_tolerance=args.turn_rrtstar_goal_tolerance,
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
        place_x_offsets = args.turn_candidates["place_x_offset_candidates_m"]
        place_y_offsets = args.turn_candidates["place_y_offset_candidates_m"]
        assert isinstance(place_x_offsets, list) and isinstance(place_y_offsets, list)
        # The TOML array order is the planning priority: Y is outer and X is
        # inner.  Each offset is relative to the initial semantic grasp point
        # and names the workpiece centre.
        object_place_candidates = [
            [centre_xy[0] + x_offset, centre_xy[1] + y_offset]
            for y_offset in place_y_offsets for x_offset in place_x_offsets
        ]
        print(
            "阶段 6：翻转已完成；不使用相机 ROI 过滤下放候选，"
            f"当前工件中心 xy={reoriented_object_centre[:2]}，以初始抓取点 xy={centre_xy} 为原点，"
            f"按 TOML 数组顺序搜索 {len(object_place_candidates)} 个工件中心下放候选。"
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
                    rrtstar_goal_tolerance=args.place_rrtstar_goal_tolerance,
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
            # All offset candidates failed planning. Keep the fallback
            # consistent with the placement search:
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
                    rrtstar_goal_tolerance=args.place_fallback_rrtstar_goal_tolerance,
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
        print(
            "阶段 7：松爪后从当前 TCP 保持 XY/姿态，"
            f"沿 body_base +Z 使用 RRTstar（rrtstar_goal_tolerance={args.release_rrtstar_goal_tolerance}）抬起 "
            f"候选高度(mm)={[value * 1000.0 for value in args.post_release_lift_height_m]}。"
        )
        release_errors = []
        for height in args.post_release_lift_height_m:
            post_release_lift_target = [released_tcp[0], released_tcp[1], released_tcp[2] + height, *released_tcp[3:]]
            try:
                trajectory, frequency = preplan_tcp_trajectory(
                    planner, arm_id=args.arm_id, group=group, frame_id=args.frame_id,
                    ik_frame=ik_frame, tcp_offset=tcp_offset, target_tcp=post_release_lift_target,
                    planner_kind=RRTSTAR_PLANNER_KIND, max_vel=args.max_vel, max_acc=args.max_acc,
                    timeout_s=RRTCONNECT_TIMEOUT_S, stage="阶段 7：松爪后 RRTstar 沿 body_base +Z 抬起",
                    rrtstar_goal_tolerance=args.release_rrtstar_goal_tolerance)
                arm.follow_trajectory(arm_id=args.arm_id, trajectory=trajectory,
                                    sample_frequency=frequency * args.trajectory_execution_speed_scale)
                break
            except Exception as exc:
                release_errors.append(str(exc))
        else:
            raise RuntimeError("阶段 7：所有松爪后抬起高度候选均失败：" + "；".join(release_errors))
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
