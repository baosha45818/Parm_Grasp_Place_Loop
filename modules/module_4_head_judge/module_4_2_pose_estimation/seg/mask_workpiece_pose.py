#!/usr/bin/env python3
"""Estimate robot-base XY positions and head-to-tail yaw from a binary mask.

The mask must have the same resolution as the calibrated head D435i RGB
stream.  Each connected component is treated as one workpiece.  The wider end
of the component is its head; yaw points from the narrow tail to that head.

Examples:
    cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_2_pose_estimation
    uv run python seg/mask_workpiece_pose.py output/mask.png

    # If the tabletop is 12 mm above robot-base z=0:
    uv run python my_robot/seg/mask_workpiece_pose.py mask.png --plane-z 0.012 \
      --debug-image pose_debug.png --output poses.json

Two directions are reported.  ``head_direction_base_*`` is measured in robot
base XY from +X towards +Y (counter-clockwise), and is also the quadrant used
to select the gripper push-entry point.  ``head_direction_image_*`` uses the
visual convention requested for four quadrants: right/up is I, left/up II,
left/down III, and right/down IV; it is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "calibration/sdk_base_to_planner_alignment.json"
)
# Measured workpiece support plane in Planner body_base.  The head-camera
# calibration and SDK-base alignment used by this script are valid at this
# fixed table plane.
DEFAULT_PLANE_Z_M = -0.391809
# Side push contacts are deliberately ahead of the geometric middle for
# better leverage: measured from tail to head, use this fraction as the
# short-edge-line anchor.
PUSH_SIDE_ANCHOR_FRACTION_FROM_TAIL = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask", type=Path, help="binary workpiece mask PNG")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=(
            "body_base head-camera calibration JSON (default: "
            f"{DEFAULT_CALIBRATION})"
        ),
    )
    parser.add_argument(
        "--plane-z",
        type=float,
        default=DEFAULT_PLANE_Z_M,
        help=(
            "workpiece support-plane height in body_base coordinates, metres "
            f"(default: {DEFAULT_PLANE_Z_M:.6f})"
        ),
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="discard mask components smaller than this many pixels (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON result path (default: <mask_stem>_pose.json beside the mask)",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        help="optional annotated mask image; arrows point from tail to head",
    )
    return parser.parse_args()


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in (None, "ACCEPTED"):
        raise RuntimeError(f"calibration is not accepted: {path}")
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(payload.get("distortion_coeffs", []), dtype=np.float64).reshape(-1)
    if "T_body_base_head_camera" not in payload:
        raise RuntimeError(
            "calibration lacks T_body_base_head_camera. Use "
            "calibration/output/sdk_base_alignment/sdk_base_to_planner_alignment.json, "
            "not the legacy SDK-base calibration_head.json."
        )
    transform = np.asarray(payload["T_body_base_head_camera"], dtype=np.float64)
    image_size = tuple(int(value) for value in payload["image_size"])
    if camera_matrix.shape != (3, 3) or transform.shape != (4, 4) or len(image_size) != 2:
        raise RuntimeError(f"invalid calibration payload: {path}")
    return camera_matrix, distortion, transform, image_size


def pixel_to_base_xy(
    pixel: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    t_base_camera: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    """Intersect the calibrated pixel ray with z=plane_z in robot-base frame."""
    normalized = cv2.undistortPoints(
        np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2), camera_matrix, distortion
    ).reshape(2)
    ray_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    ray_base = t_base_camera[:3, :3] @ ray_camera
    camera_origin = t_base_camera[:3, 3]
    if abs(ray_base[2]) < 1e-9:
        raise RuntimeError("pixel ray is parallel to the workpiece plane")
    distance = (plane_z - camera_origin[2]) / ray_base[2]
    if distance <= 0:
        raise RuntimeError("workpiece plane is behind the camera; check --plane-z and calibration")
    point_base = camera_origin + distance * ray_base
    return point_base[:2]


def _rotated_box_axis(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit a coarse rectangle used only to identify the wider head end."""
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("component has no contour")
    rectangle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    centre = np.asarray(rectangle[0], dtype=np.float64)
    corners = cv2.boxPoints(rectangle).astype(np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    long_edge_index = int(np.argmax(lengths))
    long_length = float(lengths[long_edge_index])
    if long_length <= 0:
        raise RuntimeError("component rotated rectangle has zero length")
    axis = edges[long_edge_index] / long_length
    return centre, axis, long_length, corners


def _end_width(points_xy: np.ndarray, axis: np.ndarray, coordinates: np.ndarray, high_end: bool) -> float:
    """Robust cross-axis width measured over the outer quarter of one end."""
    threshold = np.percentile(coordinates, 75 if high_end else 25)
    end_points = points_xy[coordinates >= threshold] if high_end else points_xy[coordinates <= threshold]
    cross_axis = np.array([-axis[1], axis[0]])
    cross_coordinates = end_points @ cross_axis
    return float(np.percentile(cross_coordinates, 95) - np.percentile(cross_coordinates, 5))


def _tail_mask_centerline(
    component: np.ndarray, points_xy: np.ndarray, coarse_tail_to_head: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the actual tail midpoint and direction parallel to mask sidewalls.

    A min-area rectangle is biased by a broad/forked head.  Prefer the longest
    contour segments that are mutually parallel to its coarse long direction:
    they are the actual two sidewalls that guide the gripper.  A tail-only
    centreline fit is retained as a fallback when the mask has no sufficiently
    long straight contour segments.  The reported tail point itself is the
    middle of the outer five percent of the *actual mask*.
    """
    coarse_tail_to_head = coarse_tail_to_head / np.linalg.norm(coarse_tail_to_head)
    coarse_coordinate = points_xy @ coarse_tail_to_head
    tail_limit = np.percentile(coarse_coordinate, 5)
    tail_points = points_xy[coarse_coordinate <= tail_limit]
    if tail_points.shape[0] < 3:
        raise RuntimeError("mask tail has too few pixels to determine its midpoint")
    tail_midpoint = np.mean(tail_points, axis=0)

    edges = cv2.Canny(component.astype(np.uint8) * 255, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=12,
        minLineLength=max(12, int(0.20 * math.sqrt(points_xy.shape[0]))),
        maxLineGap=6,
    )
    side_candidates: list[tuple[float, np.ndarray]] = []
    # 35 degrees is deliberately loose enough for a skewed coarse rectangle,
    # but excludes end caps, which are near perpendicular to the sidewalls.
    min_parallel_dot = math.cos(math.radians(35.0))
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            vector = np.array([float(x2 - x1), float(y2 - y1)])
            length = float(np.linalg.norm(vector))
            if length <= 1e-9:
                continue
            vector /= length
            if float(vector @ coarse_tail_to_head) < 0.0:
                vector = -vector
            if float(vector @ coarse_tail_to_head) >= min_parallel_dot:
                side_candidates.append((length, vector))
    if side_candidates:
        # A head edge can still be roughly aligned with the coarse rectangle.
        # Select the *largest parallel cluster* first, rather than averaging
        # every broadly compatible segment with the true two sidewalls.
        cluster_parallel_dot = math.cos(math.radians(15.0))
        selected = max(
            (
                [
                    other
                    for other in side_candidates
                    if float(vector @ other[1]) >= cluster_parallel_dot
                ]
                for _, vector in side_candidates
            ),
            key=lambda cluster: sum(length for length, _ in cluster),
        )
        weighted_axis = sum((length * vector for length, vector in selected), np.zeros(2))
        if float(np.linalg.norm(weighted_axis)) > 1e-9:
            sidewall_axis = weighted_axis / np.linalg.norm(weighted_axis)
        else:
            side_candidates = []
    if not side_candidates:
        centres: list[np.ndarray] = []
        for low_percentile, high_percentile in zip(
            np.linspace(2.0, 25.0, 12), np.linspace(7.0, 30.0, 12)
        ):
            low, high = np.percentile(coarse_coordinate, [low_percentile, high_percentile])
            slice_points = points_xy[(coarse_coordinate >= low) & (coarse_coordinate <= high)]
            if slice_points.shape[0] >= 3:
                centres.append(np.mean(slice_points, axis=0))
        if len(centres) < 2:
            raise RuntimeError("mask has too few centreline slices")
        centre_points = np.asarray(centres, dtype=np.float64)
        _, _, right_vectors = np.linalg.svd(
            centre_points - np.mean(centre_points, axis=0), full_matrices=False
        )
        sidewall_axis = right_vectors[0]
        if float(sidewall_axis @ coarse_tail_to_head) < 0.0:
            sidewall_axis = -sidewall_axis

    fitted_coordinates = (points_xy - tail_midpoint) @ sidewall_axis
    forward_length = float(np.percentile(fitted_coordinates, 97.5))
    if forward_length <= 0.0:
        raise RuntimeError("mask centreline has zero tail-to-head length")
    return tail_midpoint, sidewall_axis, forward_length


def _angle_and_quadrant(direction_xy: np.ndarray) -> tuple[float, int]:
    """Return a [0, 360) CCW angle and its standard one-to-four quadrant."""
    angle_deg = math.degrees(math.atan2(float(direction_xy[1]), float(direction_xy[0])))
    if angle_deg < 0:
        angle_deg += 360.0
    quadrant = int(angle_deg // 90.0) + 1
    return float(angle_deg), 1 if quadrant == 5 else quadrant


def _mask_contains(component: np.ndarray, point_xy: np.ndarray) -> bool:
    """Return whether the nearest pixel to ``point_xy`` belongs to the component."""
    x, y = np.rint(point_xy).astype(int)
    return 0 <= x < component.shape[1] and 0 <= y < component.shape[0] and bool(component[y, x])


def _point_along_mask_ray(
    component: np.ndarray,
    start_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    want_inside: bool,
    max_distance_px: float,
    margin_px: float = 2.0,
) -> np.ndarray:
    """Find a nearby point just inside or just outside the mask along one ray.

    Pixel coordinates are used deliberately: the requested push directions are
    parallel to the rotated rectangle's short edge in the RGB image.  The
    returned outside point is two pixels clear of the mask, while an inside
    point is two pixels clear of its edge.
    """
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9:
        raise RuntimeError("push-point direction has zero length")
    direction = direction_xy / norm
    step_px = 0.25
    if want_inside:
        first_inside_distance: float | None = None
        last_inside_point: np.ndarray | None = None
        for distance in np.arange(0.0, max_distance_px + step_px, step_px):
            point = start_xy + distance * direction
            if _mask_contains(component, point):
                if first_inside_distance is None:
                    first_inside_distance = float(distance)
                last_inside_point = point
                if distance - first_inside_distance >= margin_px:
                    return point
            elif last_inside_point is not None:
                # Thin tails can be narrower than the requested margin.
                return last_inside_point
        if last_inside_point is not None:
            return last_inside_point
        raise RuntimeError("could not find mask interior along requested push-point ray")

    seen_inside = False
    edge_distance: float | None = None
    for distance in np.arange(0.0, max_distance_px + step_px, step_px):
        point = start_xy + distance * direction
        is_inside = _mask_contains(component, point)
        if is_inside:
            seen_inside = True
        elif seen_inside:
            if edge_distance is None:
                edge_distance = float(distance)
            if distance - edge_distance >= margin_px:
                return point

    if edge_distance is not None:
        # A component touching the image edge has no visible outside pixel.
        # Return the closest usable point rather than silently selecting a
        # location on the wrong side.
        return start_xy + edge_distance * direction
    raise RuntimeError("could not find a mask edge along requested push-point ray")


def _push_entry_point(
    component: np.ndarray,
    head_pixel: np.ndarray,
    tail_pixel: np.ndarray,
    base_angle_deg: float,
    base_quadrant: int,
    long_length: float,
) -> tuple[np.ndarray, str]:
    """Choose the gripper push/entry point from the base-XY quadrant policy."""
    tail_to_head = head_pixel - tail_pixel
    # In image coordinates (+y down), this is the right-hand normal when
    # looking from the tail toward the head.  It is parallel to the box's
    # short edges.
    right_normal = np.array([-tail_to_head[1], tail_to_head[0]], dtype=np.float64)
    max_distance = float(math.hypot(*component.shape))

    if base_quadrant == 1:
        return (
            _point_along_mask_ray(
                component,
                tail_pixel,
                tail_to_head,
                want_inside=True,
                max_distance_px=long_length,
            ),
            "tail_inside",
        )

    if base_quadrant == 2:
        direction, policy = -right_normal, "left_outside"
    elif base_quadrant == 4:
        direction, policy = right_normal, "right_outside"
    else:
        # In base XY, Q3's Q2-facing half is 180--225 degrees and uses the
        # left side; its Q4-facing half uses the right.
        direction, policy = (
            (-right_normal, "left_outside_q3_toward_q2")
            if base_angle_deg < 225.0
            else (right_normal, "right_outside_q3_toward_q4")
        )
    # The side contact must not share the rotated-box centre's short-edge
    # line.  Shift its line toward the head, while keeping the outward ray
    # strictly parallel to the gripper's short-edge direction.
    side_anchor = tail_pixel + PUSH_SIDE_ANCHOR_FRACTION_FROM_TAIL * tail_to_head
    return (
        _point_along_mask_ray(
            component,
            side_anchor,
            direction,
            want_inside=False,
            max_distance_px=max_distance,
        ),
        policy,
    )


def component_pose(
    component: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    t_base_camera: np.ndarray,
    plane_z: float,
) -> dict:
    """Estimate centroid, wider head endpoint, and robot-base yaw for one component."""
    # np.where returns (row=y, column=x); convert to image (x, y).
    rows, columns = np.where(component)
    points_xy = np.column_stack((columns, rows)).astype(np.float64)
    # Position uses the actual mask centroid.  The rectangle is used only for
    # orientation, so an asymmetric wide head does not shift the reported
    # workpiece centre.
    mask_centroid = np.mean(points_xy, axis=0)
    rectangle_centre, coarse_axis, _coarse_long_length, box_corners = _rotated_box_axis(component)
    coordinates = (points_xy - rectangle_centre) @ coarse_axis
    width_low = _end_width(points_xy, coarse_axis, coordinates, high_end=False)
    width_high = _end_width(points_xy, coarse_axis, coordinates, high_end=True)

    head_is_high = width_high > width_low
    coarse_tail_to_head = coarse_axis if head_is_high else -coarse_axis
    # The pose direction is the mask's sidewall centreline, not the long axis
    # of its minimum-area rectangle.  In particular, its origin is the real
    # tail-mask midpoint instead of a rectangle corner/endpoint.
    tail_pixel, tail_to_head_axis, long_length = _tail_mask_centerline(
        component, points_xy, coarse_tail_to_head
    )
    head_pixel = tail_pixel + long_length * tail_to_head_axis
    # The area centroid is useful for diagnostics, but a tapered workpiece's
    # centroid need not lie on its longitudinal centreline.  Grasping uses the
    # midpoint of the semantic tail/head centreline instead, so the jaw closing
    # line is perpendicular to (and crosses) the red tail-to-head line.
    grasp_center_pixel = (tail_pixel + head_pixel) / 2.0

    center_xy = pixel_to_base_xy(mask_centroid, camera_matrix, distortion, t_base_camera, plane_z)
    grasp_center_xy = pixel_to_base_xy(grasp_center_pixel, camera_matrix, distortion, t_base_camera, plane_z)
    head_xy = pixel_to_base_xy(head_pixel, camera_matrix, distortion, t_base_camera, plane_z)
    tail_xy = pixel_to_base_xy(tail_pixel, camera_matrix, distortion, t_base_camera, plane_z)
    base_angle_deg, base_quadrant = _angle_and_quadrant(head_xy - tail_xy)
    base_yaw_rad = math.radians(base_angle_deg)
    # Image pixel y grows down.  Negate it so +y is visually upward.
    image_direction_xy = np.array(
        [head_pixel[0] - tail_pixel[0], tail_pixel[1] - head_pixel[1]], dtype=np.float64
    )
    image_angle_deg, image_quadrant = _angle_and_quadrant(image_direction_xy)
    push_entry_pixel, push_entry_policy = _push_entry_point(
        component,
        head_pixel,
        tail_pixel,
        base_angle_deg,
        base_quadrant,
        long_length,
    )
    push_entry_xy = pixel_to_base_xy(
        push_entry_pixel, camera_matrix, distortion, t_base_camera, plane_z
    )

    return {
        "center_pixel_xy": [float(mask_centroid[0]), float(mask_centroid[1])],
        "center_base_xy_m": [float(center_xy[0]), float(center_xy[1])],
        "center_base_xy_mm": [float(center_xy[0] * 1000.0), float(center_xy[1] * 1000.0)],
        "grasp_center_pixel_xy": [float(grasp_center_pixel[0]), float(grasp_center_pixel[1])],
        "grasp_center_base_xy_m": [float(grasp_center_xy[0]), float(grasp_center_xy[1])],
        "grasp_center_base_xy_mm": [float(grasp_center_xy[0] * 1000.0), float(grasp_center_xy[1] * 1000.0)],
        "head_pixel_xy": [float(head_pixel[0]), float(head_pixel[1])],
        "tail_pixel_xy": [float(tail_pixel[0]), float(tail_pixel[1])],
        "head_base_xy_m": [float(head_xy[0]), float(head_xy[1])],
        "tail_base_xy_m": [float(tail_xy[0]), float(tail_xy[1])],
        "head_direction_base_angle_deg": base_angle_deg,
        "head_direction_base_quadrant": base_quadrant,
        "robot_base_xy_yaw": {
            "x_m": float(center_xy[0]),
            "y_m": float(center_xy[1]),
            "x_mm": float(center_xy[0] * 1000.0),
            "y_mm": float(center_xy[1] * 1000.0),
            "yaw_rad": float(base_yaw_rad),
            "yaw_deg": base_angle_deg,
            "yaw_quadrant": base_quadrant,
            "yaw_definition": "tail-mask midpoint to mask-sidewall centreline, in body_base XY",
        },
        "head_direction_image_angle_deg": image_angle_deg,
        "head_direction_image_quadrant": image_quadrant,
        "push_entry_point": {
            "pixel_xy": [float(push_entry_pixel[0]), float(push_entry_pixel[1])],
            "base_xy_m": [float(push_entry_xy[0]), float(push_entry_xy[1])],
            "base_xy_mm": [float(push_entry_xy[0] * 1000.0), float(push_entry_xy[1] * 1000.0)],
            "policy": push_entry_policy,
            "definition": (
                "Q1: point inside the tail; Q2: point just outside the left side; "
                "Q4: point just outside the right side; Q3: left/right according to "
                "whether the base-XY angle is toward Q2/Q4. The quadrant is "
                "head_direction_base_quadrant; left/right are viewed tail-to-head. "
                "Side contacts use the section 80 percent of the way from tail to head."
            ),
        },
        "head_width_px": float(width_high if head_is_high else width_low),
        "tail_width_px": float(width_low if head_is_high else width_high),
        "head_to_tail_width_ratio": float(max(width_low, width_high) / max(min(width_low, width_high), 1e-6)),
        "area_px": int(points_xy.shape[0]),
        "rotated_box_pixel_corners_xy": box_corners.tolist(),
    }


def write_debug_image(mask: np.ndarray, poses: list[dict], output: Path) -> None:
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for index, pose in enumerate(poses, start=1):
        center = tuple(np.rint(pose["center_pixel_xy"]).astype(int))
        grasp_center = tuple(np.rint(pose.get("grasp_center_pixel_xy", pose["center_pixel_xy"])).astype(int))
        head = tuple(np.rint(pose["head_pixel_xy"]).astype(int))
        tail = tuple(np.rint(pose["tail_pixel_xy"]).astype(int))
        push_entry = tuple(np.rint(pose["push_entry_point"]["pixel_xy"]).astype(int))
        box = np.rint(np.asarray(pose["rotated_box_pixel_corners_xy"])).astype(np.int32)
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 255), thickness=1)
        cv2.arrowedLine(image, tail, head, (0, 0, 255), 2, tipLength=0.18)
        cv2.circle(image, center, 4, (0, 255, 0), -1)
        cv2.drawMarker(image, grasp_center, (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        cv2.circle(image, push_entry, 4, (255, 0, 255), -1)
        cv2.putText(
            image,
            f"#{index} base {pose['head_direction_base_angle_deg']:.1f} deg Q{pose['head_direction_base_quadrant']}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write debug image: {output}")


def main() -> int:
    args = parse_args()
    if args.min_area <= 0:
        raise ValueError("--min-area must be positive")
    if not args.mask.is_file():
        raise FileNotFoundError(f"mask does not exist: {args.mask}")
    if not args.calibration.is_file():
        raise FileNotFoundError(f"calibration does not exist: {args.calibration}")

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"could not decode mask: {args.mask}")
    camera_matrix, distortion, t_base_camera, calibration_size = load_calibration(args.calibration)
    image_size = (mask.shape[1], mask.shape[0])
    if image_size != calibration_size:
        raise RuntimeError(
            f"mask size {image_size} differs from calibration image size {calibration_size}; "
            "resize is not supported because it changes camera intrinsics"
        )

    foreground = np.where(mask > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    components = [
        (label, int(stats[label, cv2.CC_STAT_AREA]))
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= args.min_area
    ]
    if not components:
        raise RuntimeError(f"no components with area >= {args.min_area} pixels in {args.mask}")
    components.sort(key=lambda item: item[1], reverse=True)
    poses = [
        component_pose(labels == label, camera_matrix, distortion, t_base_camera, args.plane_z)
        for label, _ in components
    ]
    result = {
        "mask": str(args.mask),
        "calibration": str(args.calibration),
        "coordinate_frame": "body_base",
        "plane_z_m": args.plane_z,
        "angle_convention": {
            "base": "tail_to_head; body_base +X toward +Y, counter-clockwise",
            "image": "tail_to_head; right/up is quadrant I, left/up II, left/down III, right/down IV",
        },
        "workpieces": poses,
    }
    output_path = args.output or args.mask.with_name(f"{args.mask.stem}_pose.json")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.debug_image is not None:
        write_debug_image(foreground, poses, args.debug_image)
    print(
        f"[done] workpieces={len(poses)} result={output_path} "
        f"debug_image={args.debug_image if args.debug_image is not None else 'not-written'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
