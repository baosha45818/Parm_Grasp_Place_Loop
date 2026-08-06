#!/usr/bin/env python3
"""Detect reverse-side workpieces on a table and enqueue reviewed grasp candidates.

This program is deliberately perception-only: it never opens an arm, gripper, or
motion-planning connection.  Every queued pose is expressed in the D435i colour
optical frame and is marked ``PENDING_REVIEW``.  A calibrated head-camera-to-robot
transform and collision checking are required before any robot may execute it.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent
PLACE_LOOP_ROOT = PROJECT_ROOT.parents[4]
STL_PATH = PLACE_LOOP_ROOT / "tool/welding_up_side_1118_only_2_small_screws_removed.stl"
if str(PLACE_LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(PLACE_LOOP_ROOT))

from tool.inspection_support.registration import register_cad_to_observed_candidates  # noqa: E402
from tool.inspection_support.stl_model import load_stl_triangles_m, sample_surface_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="Directory created by capture_head_rgbd.py")
    parser.add_argument("--stl", type=Path, default=STL_PATH)
    parser.add_argument("--queue", type=Path, default=PROJECT_ROOT / "output/grasp_queue.jsonl")
    parser.add_argument("--seen-cache", type=Path, default=PROJECT_ROOT / "output/seen_workpieces.jsonl",
                        help="Append-only cache of recognised positive-side workpieces")
    parser.add_argument("--remember-seen", action="store_true",
                        help="Enable duplicate association with prior queue/cache records (disabled by default)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/detections")
    parser.add_argument("--depth-name", default="depth_aligned_000.png")
    parser.add_argument("--rgb-name", default="rgb_000.png")
    parser.add_argument("--plane-distance-m", type=float, default=0.004)
    parser.add_argument("--object-min-height-m", type=float, default=0.004)
    parser.add_argument("--object-max-height-m", type=float, default=0.080)
    parser.add_argument("--voxel-m", type=float, default=0.003)
    parser.add_argument("--cluster-eps-m", type=float, default=0.012)
    parser.add_argument("--cluster-min-points", type=int, default=80)
    parser.add_argument("--min-cluster-points", type=int, default=150)
    parser.add_argument("--icp-max-corr-m", type=float, default=0.008)
    parser.add_argument("--icp-min-fitness", type=float, default=0.30)
    parser.add_argument("--icp-max-rmse-m", type=float, default=0.006)
    parser.add_argument("--reverse-face-axis-stl", nargs=3, type=float, default=[0.0, 0.0, -1.0],
                        metavar=("X", "Y", "Z"),
                        help="STL face normal that denotes the reverse side; verify once against a labelled part")
    parser.add_argument("--reverse-facing-min-dot", type=float, default=0.45,
                        help="Minimum dot product of the reverse face normal with object-to-camera direction")
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.080)
    parser.add_argument("--dedup-center-m", type=float, default=0.025,
                        help="Maximum camera-frame centroid difference for an already queued workpiece")
    parser.add_argument("--dedup-extent-m", type=float, default=0.015,
                        help="Maximum L2 difference between point-cloud bounding-box extents")
    parser.add_argument("--dedup-chamfer-m", type=float, default=0.008,
                        help="Maximum 90th-percentile bidirectional cloud distance for duplicate association")
    parser.add_argument("--overlay-axis-m", type=float, default=0.040,
                        help="Length of each grasp-frame axis rendered onto the RGB overlay")
    parser.add_argument("--overlay-positive", action="store_true",
                        help="Render positive-side debug grasp frames too; they are never queued")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.object_min_height_m <= 0 or args.object_max_height_m <= args.object_min_height_m:
        parser.error("object height limits must satisfy 0 < min < max")
    if args.cluster_min_points < 3 or args.min_cluster_points < args.cluster_min_points:
        parser.error("cluster point limits are invalid")
    return args


def load_capture(capture_dir: Path, rgb_name: str, depth_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    metadata_path = capture_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rgb = cv2.imread(str(capture_dir / rgb_name), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture_dir / depth_name), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or depth.dtype != np.uint16:
        raise RuntimeError("Capture directory must contain a BGR RGB PNG and a uint16 aligned depth PNG")
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(f"RGB/depth dimensions differ: {rgb.shape[:2]} vs {depth.shape}")
    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]])
    return rgb, depth, K, float(metadata["depth_scale_m"]), metadata


def depth_to_points(depth: np.ndarray, K: np.ndarray, depth_scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(depth.shape)
    z = depth.astype(np.float64) * depth_scale_m
    valid = (depth > 0) & (depth < np.iinfo(np.uint16).max) & (z < 3.0)
    x = (xx - K[0, 2]) * z / K[0, 0]
    y = (yy - K[1, 2]) * z / K[1, 1]
    xyz_image = np.dstack((x, y, z))
    return xyz_image, xyz_image[valid]


def orient_plane_towards_camera(plane: np.ndarray, plane_points: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float64).copy()
    normal = plane[:3] / np.linalg.norm(plane[:3])
    centroid = np.asarray(plane_points, dtype=np.float64).mean(axis=0)
    if np.dot(normal, -centroid) < 0.0:
        plane *= -1.0
    return plane


def segment_table_and_objects(points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    coarse = cloud.voxel_down_sample(args.voxel_m)
    if len(coarse.points) < 500:
        raise RuntimeError("Too few valid depth points to estimate the table plane")
    plane, inliers = coarse.segment_plane(
        distance_threshold=args.plane_distance_m, ransac_n=3, num_iterations=1500
    )
    plane = orient_plane_towards_camera(np.asarray(plane), np.asarray(coarse.points)[inliers])
    signed_height = points @ plane[:3] + plane[3]
    object_points = points[(signed_height >= args.object_min_height_m) & (signed_height <= args.object_max_height_m)]
    if len(object_points) == 0:
        raise RuntimeError("No points remain above the detected table plane")
    objects = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_points)).voxel_down_sample(args.voxel_m)
    labels = np.asarray(
        objects.cluster_dbscan(eps=args.cluster_eps_m, min_points=args.cluster_min_points, print_progress=False)
    )
    return plane, np.asarray(objects.points), labels


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("zero-length direction")
    return vector / norm


def widest_head_in_stl(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Find the broadest cross-section along the CAD's dominant in-plane axis."""
    centered = vertices - vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(values)[::-1]
    long_axis = normalize(vectors[:, order[0]])
    width_axis = normalize(vectors[:, order[1]])
    # Keep a right-handed basis; the third axis is not used as a grasp direction.
    long_projection = centered @ long_axis
    width_projection = centered @ width_axis
    edges = np.linspace(long_projection.min(), long_projection.max(), 42)
    best: tuple[float, np.ndarray] | None = None
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (long_projection >= low) & (long_projection < high)
        if np.count_nonzero(mask) < 12:
            continue
        width = float(np.percentile(width_projection[mask], 95) - np.percentile(width_projection[mask], 5))
        center = vertices[mask].mean(axis=0)
        if best is None or width > best[0]:
            best = (width, center)
    if best is None:
        raise RuntimeError("STL does not contain enough vertices to find its widest cross-section")
    return best[1], long_axis, width_axis, best[0]


def make_grasp_frame(width_axis_cam: np.ndarray, table_normal_cam: np.ndarray) -> np.ndarray:
    """Create the Version7/GraspNet parallel-jaw frame in camera coordinates.

    Version7 treats gripper +X as the approach direction and +Y as the jaw
    opening direction.  The calibrated pinch TCP is also displaced along EEF
    +X, so a top-down grasp must put +X *into* the table.  The former frame
    used +X for the workpiece width; that made the 270 mm TCP offset lateral.
    """
    # table_normal is oriented from the table towards the camera.  Therefore
    # -normal is the physical downward/approach direction.
    x = -normalize(table_normal_cam)
    y = width_axis_cam - x * np.dot(width_axis_cam, x)
    y = normalize(y)
    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))
    R = np.column_stack((x, y, z))
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1.0
    return R


def pose_record(R: np.ndarray, translation: np.ndarray) -> dict[str, object]:
    return {"translation_m": np.asarray(translation).round(7).tolist(), "rotation_matrix": np.asarray(R).round(8).tolist()}


def write_ply(path: Path, points: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=True)


def project_point(K: np.ndarray, point_cam: np.ndarray) -> tuple[int, int] | None:
    point_cam = np.asarray(point_cam, dtype=np.float64)
    if point_cam.shape != (3,) or point_cam[2] <= 1e-6:
        return None
    uvw = K @ point_cam
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def draw_grasp_overlay(rgb: np.ndarray, K: np.ndarray, records: list[dict], axis_m: float) -> np.ndarray:
    """Draw queued grasp frames in the RGB image using their camera-frame poses."""
    canvas = rgb.copy()
    for record in records:
        pose = record["grasp_pose"]
        origin = np.asarray(pose["translation_m"], dtype=np.float64)
        R = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        p0 = project_point(K, origin)
        if p0 is None:
            continue
        cv2.circle(canvas, p0, 7, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        # BGR: X red, Y green, Z blue. +Z is the retreat/pre-grasp direction.
        for axis, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            p1 = project_point(K, origin + R[:, axis] * axis_m)
            if p1 is not None:
                cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.18)
        pregrasp = np.asarray(record["pregrasp_pose"]["translation_m"], dtype=np.float64)
        pp = project_point(K, pregrasp)
        if pp is not None:
            cv2.circle(canvas, pp, 5, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(canvas, p0, pp, (255, 255, 0), 1, cv2.LINE_AA)
        label = f"{record.get('object_side', 'reverse')} {record.get('object_key', record.get('id', ''))[-6:]}"
        cv2.putText(canvas, label, (p0[0] + 9, p0[1] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "X:red  Y:green  Z(retreat):blue  pregrasp:yellow", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "X:red  Y:green  Z(retreat):blue  pregrasp:yellow", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def cloud_signature(points: np.ndarray) -> dict[str, object]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return {
        "centroid_m": points.mean(axis=0).round(7).tolist(),
        "bbox_extent_m": np.ptp(points, axis=0).round(7).tolist(),
        "points": int(len(points)),
    }


def load_active_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    active: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL queue line {line_no}: {exc}") from exc
        if record.get("status") in {"PENDING_REVIEW", "APPROVED", "EXECUTING"}:
            active.append(record)
    return active


def load_seen_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL seen-cache line {line_no}: {exc}") from exc
        if record.get("status") == "IGNORED_POSITIVE":
            records.append(record)
    return records


def duplicate_queue_record(candidate: np.ndarray, queue_records: list[dict], args: argparse.Namespace) -> dict | None:
    """Match a current cluster to an active queued object before running costly ICP.

    A centroid is only a coarse spatial index.  Extent and bidirectional cloud
    distance are both required before considering the object already queued.
    """
    sig = cloud_signature(candidate)
    center = np.asarray(sig["centroid_m"], dtype=np.float64)
    extent = np.asarray(sig["bbox_extent_m"], dtype=np.float64)
    for record in queue_records:
        old_sig = record.get("object_signature")
        if not isinstance(old_sig, dict) or record.get("frame_id") != "head_d435i_color_optical_frame":
            continue  # Old records without a cloud signature are deliberately re-verified.
        old_center = np.asarray(old_sig.get("centroid_m", []), dtype=np.float64)
        old_extent = np.asarray(old_sig.get("bbox_extent_m", []), dtype=np.float64)
        cloud_path = Path(str(old_sig.get("cloud_path", "")))
        if old_center.shape != (3,) or old_extent.shape != (3,) or not cloud_path.is_file():
            continue
        if np.linalg.norm(center - old_center) > args.dedup_center_m:
            continue
        if np.linalg.norm(extent - old_extent) > args.dedup_extent_m:
            continue
        old_cloud = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
        if len(old_cloud) < 3:
            continue
        forward = cKDTree(old_cloud).query(candidate, k=1, workers=-1)[0]
        backward = cKDTree(candidate).query(old_cloud, k=1, workers=-1)[0]
        if max(np.quantile(forward, 0.90), np.quantile(backward, 0.90)) <= args.dedup_chamfer_m:
            return record
    return None


def append_queue(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        __import__("os").fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    rgb, depth, K, scale_m, metadata = load_capture(args.capture_dir, args.rgb_name, args.depth_name)
    xyz_image, all_points = depth_to_points(depth, K, scale_m)
    plane, clustered_points, labels = segment_table_and_objects(all_points, args)
    run_dir = args.output_dir / datetime.now().strftime("reverse_detection_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(run_dir / "rgb.png"), rgb)
    write_ply(run_dir / "table_objects_clustered.ply", clustered_points)

    triangles = load_stl_triangles_m(args.stl)
    vertices = triangles.reshape(-1, 3)
    cad_points = sample_surface_points(triangles, n_points=12000, seed=args.seed)
    head_stl, _long_stl, width_stl, widest_width_m = widest_head_in_stl(vertices)
    reverse_axis_stl = normalize(np.asarray(args.reverse_face_axis_stl))
    table_normal = normalize(plane[:3])
    records: list[dict] = []
    active_queue = load_active_queue(args.queue) if args.remember_seen else []
    positive_cache = load_seen_cache(args.seen_cache) if args.remember_seen else []
    known_records = active_queue + positive_cache
    seen_positive_records: list[dict] = []
    overlay_records: list[dict] = []
    report: dict[str, object] = {
        "capture_dir": str(args.capture_dir), "stl": str(args.stl), "camera_frame": "head_d435i_color_optical_frame",
        "table_plane_camera": plane.tolist(), "clusters": [], "queued_ids": [],
    }

    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        observed = clustered_points[labels == label]
        if len(observed) < args.min_cluster_points:
            continue
        cluster_path = run_dir / f"cluster_{label:02d}.ply"
        write_ply(cluster_path, observed)
        existing = duplicate_queue_record(observed, known_records, args)
        if existing is not None:
            cluster_report = {
                "label": int(label), "points": int(len(observed)), "cloud": cluster_path.name,
                "decision": "skipped_existing_queue" if existing.get("status") != "IGNORED_POSITIVE" else "skipped_known_positive",
                "queue_id": existing.get("id"),
            }
            if existing.get("status") != "IGNORED_POSITIVE" and "grasp_pose" in existing:
                overlay_records.append(existing)
            report["clusters"].append(cluster_report)
            continue
        candidates = register_cad_to_observed_candidates(
            cad_points, observed, max_corr_m=args.icp_max_corr_m, max_iters=60,
            max_cad_points=6000, max_obs_points=6000, seed=args.seed + int(label),
        )
        best = candidates[0]
        cluster_report = {
            "label": int(label), "points": int(len(observed)), "cloud": cluster_path.name,
            "rmse_m": float(best.rmse_m), "fitness": float(best.fitness), "init": best.init_name,
        }
        accepted_icp = best.fitness >= args.icp_min_fitness and best.rmse_m <= args.icp_max_rmse_m
        if not accepted_icp:
            cluster_report["decision"] = "rejected_icp"
            report["clusters"].append(cluster_report)
            continue

        T_camera_stl = best.T_target_source
        object_center = transform_points(T_camera_stl, vertices.mean(axis=0)[None, :])[0]
        object_to_camera = normalize(-object_center)
        reverse_facing = float(np.dot(T_camera_stl[:3, :3] @ reverse_axis_stl, object_to_camera))
        cluster_report["reverse_facing_dot"] = reverse_facing
        grasp_position = transform_points(T_camera_stl, head_stl[None, :])[0]
        grasp_R = make_grasp_frame(T_camera_stl[:3, :3] @ width_stl, table_normal)
        pregrasp_position = grasp_position + table_normal * args.pregrasp_clearance_m
        if reverse_facing < args.reverse_facing_min_dot:
            positive_record = {
                "id": str(uuid.uuid4()), "status": "IGNORED_POSITIVE", "created_at": datetime.now(timezone.utc).isoformat(),
                "object_key": "positive-" + hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:16],
                "frame_id": "head_d435i_color_optical_frame", "object_side": "positive", "stl": str(args.stl),
                "source_capture": str(args.capture_dir),
                "icp": {"fitness": float(best.fitness), "rmse_m": float(best.rmse_m), "init": best.init_name},
                "grasp_pose": pose_record(grasp_R, grasp_position),
                "pregrasp_pose": pose_record(grasp_R, pregrasp_position),
                "visualization_only": True,
                "object_signature": {
                    **cloud_signature(observed), "cloud_path": str(cluster_path),
                    "association": "centroid + bounding-box extent + bidirectional 90th-percentile cloud distance",
                },
            }
            seen_positive_records.append(positive_record)
            if args.overlay_positive:
                overlay_records.append(positive_record)
            cluster_report["decision"] = "cached_positive_side"
            cluster_report["cache_id"] = positive_record["id"]
            report["clusters"].append(cluster_report)
            continue

        record = {
            "id": str(uuid.uuid4()), "status": "PENDING_REVIEW", "created_at": datetime.now(timezone.utc).isoformat(),
            "object_key": "reverse-" + hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:16],
            "source_capture": str(args.capture_dir), "frame_id": "head_d435i_color_optical_frame",
            "object_side": "reverse", "stl": str(args.stl),
            "icp": {"fitness": float(best.fitness), "rmse_m": float(best.rmse_m), "init": best.init_name},
            "big_head": {"center_m": grasp_position.round(7).tolist(), "width_m": float(widest_width_m)},
            "object_signature": {
                **cloud_signature(observed),
                "cloud_path": str(cluster_path),
                "association": "centroid + bounding-box extent + bidirectional 90th-percentile cloud distance",
            },
            "grasp_pose": pose_record(grasp_R, grasp_position),
            "pregrasp_pose": pose_record(grasp_R, pregrasp_position),
            "safety": {
                "requires_head_camera_extrinsic": True,
                "requires_ik_collision_check": True,
                "requires_human_review": True,
                "note": "+Z of each pose points from table toward camera; robot approach is along -Z.",
            },
        }
        records.append(record)
        overlay_records.append(record)
        report["queued_ids"].append(record["id"])
        cluster_report["decision"] = "queued_reverse"
        cluster_report["queue_id"] = record["id"]
        report["clusters"].append(cluster_report)

    np.save(run_dir / "xyz_camera.npy", xyz_image)
    overlay_path = run_dir / "grasp_overlay.png"
    if not cv2.imwrite(str(overlay_path), draw_grasp_overlay(rgb, K, overlay_records, args.overlay_axis_m)):
        raise RuntimeError(f"Failed to write {overlay_path}")
    report["grasp_overlay"] = overlay_path.name
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if records:
        append_queue(args.queue, records)
    if args.remember_seen and seen_positive_records:
        append_queue(args.seen_cache, seen_positive_records)
    print(json.dumps({"run_dir": str(run_dir), "queued": len(records), "queue": str(args.queue)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
