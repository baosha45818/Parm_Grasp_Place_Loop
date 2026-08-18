#!/usr/bin/env python3
"""Run D405/Femto/fused GraspNet inference, then use version4 grasp execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.version4.run_d405_graspnet import (  # noqa: E402
    build_execution_plan,
    build_parser as build_version4_parser,
    capture_eef_rotation,
    execute_grasp,
    grasp_result,
    grasp_tcp_is_certified,
    load_tcp_translation,
    load_wrist_calibration,
    move_to_observation,
    preview_pregrasp,
    show_official_visualization,
)

VERSION1 = ROOT / "deploy" / "version1" / "run_dual_camera_inference.py"
DEFAULT_LASER_CALIBRATION = ROOT / "deploy" / "calibration" / "output" / "laser" / "calibration_laser.json"
DEFAULT_TCP_CALIBRATION = ROOT / "deploy" / "calibration" / "output" / "tcp_grasp_tcp" / "calibration_translation2.json"
GRASPUP_JOINTS_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]
BUFFER_JOINTS_DEG = [49.671, -56.037, -115.398, 72.201, 9.075, 53.369, 11.563]


def build_parser():
    parser = build_version4_parser()
    parser.description = "Run D405, Femto Bolt, and fused GraspNet inference, then execute like version4."
    legacy_action = next(
        (action for action in parser._actions if "--gripper-yaw-offset-deg" in action.option_strings),
        None,
    )
    if legacy_action is not None:
        for option in legacy_action.option_strings:
            parser._option_string_actions.pop(option, None)
        parser._actions.remove(legacy_action)
        for group in parser._action_groups:
            if legacy_action in group._group_actions:
                group._group_actions.remove(legacy_action)
    parser.add_argument("--femto-serial", default="CL8L56301LK")
    parser.add_argument("--laser-calibration", default=str(DEFAULT_LASER_CALIBRATION))
    parser.add_argument("--femto-color-width", type=int, default=1280)
    parser.add_argument("--femto-color-height", type=int, default=720)
    parser.add_argument("--femto-depth-width", type=int, default=640)
    parser.add_argument("--femto-depth-height", type=int, default=480)
    parser.add_argument("--femto-z-min", type=float, default=0.20)
    parser.add_argument("--femto-z-max", type=float, default=2.00)
    parser.add_argument("--femto-x-abs-max", type=float, default=1.20)
    parser.add_argument("--femto-y-abs-max", type=float, default=1.20)
    parser.add_argument(
        "--femto-roi",
        type=int,
        nargs=4,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Femto aligned color/depth ROI selected with deploy/tools/roi_graspnet_laser.py",
    )
    parser.add_argument("--fuse-voxel-size-m", type=float, default=0.004)
    parser.add_argument("--fusion-object-min-height-m", type=float, default=0.005)
    parser.add_argument("--fusion-object-max-height-m", type=float, default=0.150)
    parser.add_argument(
        "--max-grasp-object-distance-m", type=float, default=0.015,
        help="maximum center-to-workpiece-reference distance after source-local object extraction",
    )
    parser.add_argument(
        "--object-cluster-eps-m", type=float, default=0.008,
        help="DBSCAN radius for selecting the single workpiece cluster from elevated points",
    )
    parser.add_argument(
        "--object-cluster-min-points", type=int, default=100,
        help="minimum DBSCAN cluster size for the workpiece reference",
    )
    parser.add_argument(
        "--object-cluster-min-z-span-m", type=float, default=0.008,
        help="reject flat tabletop residual clusters with less vertical extent than this",
    )
    parser.add_argument(
        "--min-grasp-object-support-points", type=int, default=30,
        help="minimum workpiece-reference points required inside the predicted closing volume",
    )
    parser.add_argument(
        "--enable-fused-inference",
        action="store_true",
        help="experimentally run a third GraspNet inference on the fused object cloud",
    )
    parser.add_argument(
        "--min-grasp-approach-angle-deg",
        type=float,
        default=65.0,
        help="minimum angle between the downward GraspNet approach axis and the XOY plane",
    )
    parser.add_argument(
        "--confirm-real-pregrasp",
        action="store_true",
        help="explicitly allow --preview-pregrasp to send real robot/gripper commands",
    )
    parser.set_defaults(
        tcp_translation_calibration=str(DEFAULT_TCP_CALIBRATION),
        postgrasp_lift_joints_deg=GRASPUP_JOINTS_DEG,
        place_joints_deg=BUFFER_JOINTS_DEG,
    )
    return parser


def validate_transform(matrix: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"{label} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise RuntimeError(f"{label} does not contain a proper rotation matrix")
    return matrix


def load_transform(path: Path, key: str, *, require_accepted: bool = False) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"missing calibration file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if require_accepted and payload.get("status") != "ACCEPTED":
        raise RuntimeError(f"calibration is not ACCEPTED: {path}")
    return validate_transform(payload.get(key), f"{path}:{key}")


def newest_run(output_dir: Path) -> Path:
    runs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"expected one inference run under {output_dir}, found {len(runs)}")
    return runs[0]


def camera_command(args, camera: str, output_dir: Path) -> list[str]:
    cmd = [
        args.python_bin,
        str(VERSION1),
        "--camera", camera,
        "--apply_calibration",
        "--checkpoint_path", str(args.checkpoint_path),
        "--num_point", str(args.num_point),
        "--num_view", str(args.num_view),
        "--collision_thresh", str(args.collision_thresh),
        "--voxel_size", str(args.collision_voxel_size),
        "--approach_dist", str(args.collision_approach_dist),
        "--collision_table_thickness", str(args.collision_table_thickness),
        "--collision_table_plane_dist", str(args.collision_table_plane_dist),
        "--collision_table_min_inlier_ratio", str(args.collision_table_min_inlier_ratio),
        "--collision_table_min_normal_z", str(args.collision_table_min_normal_z),
        "--top_k", str(args.top_k),
        "--output_dir", str(output_dir),
        "--arm_host", str(args.arm_host),
        "--arm_id", str(args.arm_id),
        "--wrist_calibration", str(args.wrist_calibration),
        "--laser_calibration", str(args.laser_calibration),
    ]
    if camera == "d405":
        cmd.extend([
            "--d405_serial", str(args.d405_serial),
            "--d405_frame_count", str(args.d405_frame_count),
            "--d405_z_min", str(args.d405_z_min),
            "--d405_z_max", str(args.d405_z_max),
            "--d405_x_abs_max", str(args.d405_x_abs_max),
            "--d405_y_abs_max", str(args.d405_y_abs_max),
        ])
        if args.d405_postprocess:
            cmd.append("--d405_postprocess")
        if args.d405_hole_filling:
            cmd.append("--d405_hole_filling")
        if args.roi is not None:
            cmd.extend(["--roi", *(str(value) for value in args.roi)])
    else:
        cmd.extend([
            "--femto_serial", str(args.femto_serial),
            "--femto_color_width", str(args.femto_color_width),
            "--femto_color_height", str(args.femto_color_height),
            "--femto_depth_width", str(args.femto_depth_width),
            "--femto_depth_height", str(args.femto_depth_height),
            "--femto_z_min", str(args.femto_z_min),
            "--femto_z_max", str(args.femto_z_max),
            "--femto_x_abs_max", str(args.femto_x_abs_max),
            "--femto_y_abs_max", str(args.femto_y_abs_max),
        ])
        if args.femto_roi is not None:
            cmd.extend(["--roi", *(str(value) for value in args.femto_roi)])
    return cmd


def run_camera_inference(args, camera: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    cmd = camera_command(args, camera, output_dir)
    print(f"-> {camera} GraspNet:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=os.environ.copy())
    run = newest_run(output_dir)
    name = camera
    if not (run / name / "point_cloud.npy").exists() or not (run / name / "grasps.npy").exists():
        raise RuntimeError(f"{camera} inference output is incomplete: {run}")
    expected_roi = args.roi if camera == "d405" else args.femto_roi
    meta_path = run / name / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    applied_roi = meta.get("roi_xywh")
    if expected_roi is None:
        raise RuntimeError(f"{camera} ROI is required; refusing to fuse a full-frame cloud")
    if [int(value) for value in applied_roi or []] != [int(value) for value in expected_roi]:
        raise RuntimeError(
            f"{camera} ROI was not applied: expected {list(expected_roi)}, got {applied_roi}"
        )
    print(f"[{camera}] verified ROI for inference and fusion: {applied_roi}")
    return run


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0:
        return np.arange(len(points), dtype=np.int64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, selected = np.unique(keys, axis=0, return_index=True)
    return np.sort(selected)


def estimate_horizontal_table_z(points: np.ndarray, bin_size_m: float = 0.001) -> float:
    z = np.asarray(points[:, 2], dtype=np.float64)
    z = z[np.isfinite(z)]
    if len(z) < 100:
        raise RuntimeError("too few points to estimate the tabletop")
    z_min, z_max = float(z.min()), float(z.max())
    edges = np.arange(z_min, z_max + 2.0 * bin_size_m, bin_size_m)
    counts, edges = np.histogram(z, bins=edges)
    peak = int(np.argmax(counts))
    in_peak = z[(z >= edges[peak]) & (z < edges[peak + 1])]
    return float(np.median(in_peak))


def extract_object_points(
    points: np.ndarray, colors: np.ndarray, min_height: float, max_height: float
) -> tuple[np.ndarray, np.ndarray, float]:
    table_z = estimate_horizontal_table_z(points)
    mask = (
        (points[:, 2] >= table_z + min_height)
        & (points[:, 2] <= table_z + max_height)
    )
    return points[mask], colors[mask], table_z


def select_workpiece_clusters(
    points: np.ndarray, colors: np.ndarray, *, eps_m: float, min_points: int, min_z_span_m: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep every spatially coherent, non-flat workpiece cloud.

    The height band alone also captures small tabletop-depth residuals. Those
    residuals are usually broad but nearly flat; each workpiece is a connected
    3-D component. Keep all qualifying components: a scene can contain several
    rusty workpieces and selecting only the largest one silently excludes the
    others before GraspNet candidate validation.
    """
    if eps_m <= 0.0 or min_points < 3 or min_z_span_m < 0.0:
        raise RuntimeError("invalid object-cluster parameters")
    if len(points) < min_points:
        raise RuntimeError(f"only {len(points)} elevated points; cannot identify the workpiece cluster")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(cloud.cluster_dbscan(eps=eps_m, min_points=min_points, print_progress=False), dtype=np.int64)
    candidates: list[tuple[int, int, float]] = []
    for label in range(int(labels.max()) + 1):
        mask = labels == label
        count = int(np.count_nonzero(mask))
        z_span = float(np.ptp(points[mask, 2]))
        if count >= min_points and z_span >= min_z_span_m:
            candidates.append((label, count, z_span))
    if not candidates:
        raise RuntimeError(
            "no non-flat workpiece cluster found; refusing to use elevated tabletop residuals as an object reference"
        )
    selected_labels = np.asarray([item[0] for item in candidates], dtype=np.int64)
    selected = np.isin(labels, selected_labels)
    selected_records = [
        {"label": int(label), "points": int(count), "z_span_m": float(z_span)}
        for label, count, z_span in sorted(candidates, key=lambda item: (-item[1], -item[2], item[0]))
    ]
    return points[selected], colors[selected], {
        "elevated_points": int(len(points)),
        "cluster_count": int(labels.max() + 1),
        "noise_points": int(np.count_nonzero(labels < 0)),
        "selected_cluster_count": int(len(selected_labels)),
        "selected_cluster_points": int(np.count_nonzero(selected)),
        "selected_clusters": selected_records,
        "eps_m": float(eps_m),
        "min_points": int(min_points),
        "min_z_span_m": float(min_z_span_m),
    }


def fuse_clouds(args, d405_run: Path, femto_run: Path, output: Path) -> Path:
    d405_dir = d405_run / "d405"
    femto_dir = femto_run / "femto_bolt"
    d405_meta = json.loads((d405_dir / "meta.json").read_text(encoding="utf-8"))
    femto_meta = json.loads((femto_dir / "meta.json").read_text(encoding="utf-8"))
    T_base_d405 = validate_transform(d405_meta.get("T_base_camera"), "D405 T_base_camera")
    T_base_femto = validate_transform(femto_meta.get("T_base_camera"), "Femto T_base_camera")
    d405 = np.asarray(np.load(d405_dir / "point_cloud.npy"), dtype=np.float64)
    femto = np.asarray(np.load(femto_dir / "point_cloud.npy"), dtype=np.float64)
    d405_colors = np.asarray(np.load(d405_dir / "point_colors.npy"), dtype=np.float64)
    femto_colors = np.asarray(np.load(femto_dir / "point_colors.npy"), dtype=np.float64)
    for label, cloud in (("D405", d405), ("Femto", femto)):
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise RuntimeError(f"{label} point cloud must have shape [N, 3], got {cloud.shape}")
    for label, cloud, colors in (
        ("D405", d405, d405_colors),
        ("Femto", femto, femto_colors),
    ):
        if colors.shape != cloud.shape:
            raise RuntimeError(f"{label} point colors must match point cloud shape")
    d405_base = transform_points(d405, T_base_d405)
    femto_base = transform_points(femto, T_base_femto)
    min_height = float(args.fusion_object_min_height_m)
    max_height = float(args.fusion_object_max_height_m)
    if min_height < 0.0 or max_height <= min_height:
        raise RuntimeError("invalid fusion object height range")
    d405_objects, d405_object_colors, d405_table_z = extract_object_points(
        d405_base, d405_colors, min_height, max_height
    )
    femto_objects, femto_object_colors, femto_table_z = extract_object_points(
        femto_base, femto_colors, min_height, max_height
    )
    merged_before_filter = np.concatenate((d405_objects, femto_objects), axis=0)
    colors_before_filter = np.concatenate(
        (d405_object_colors, femto_object_colors), axis=0
    )
    valid = (
        np.all(np.isfinite(merged_before_filter), axis=1)
        & np.all(np.isfinite(colors_before_filter), axis=1)
        & (np.linalg.norm(merged_before_filter, axis=1) > 1e-6)
    )
    merged = merged_before_filter[valid]
    merged_colors = np.clip(colors_before_filter[valid], 0.0, 1.0)
    selected = voxel_downsample_indices(merged, float(args.fuse_voxel_size_m))
    merged = merged[selected]
    merged_colors = merged_colors[selected]
    if len(merged) < 100:
        raise RuntimeError(f"fused point cloud has too few points: {len(merged)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    d405_object_path = output.with_name("d405_object_cloud_base.npy")
    femto_object_path = output.with_name("femto_object_cloud_base.npy")
    np.save(d405_object_path, d405_objects.astype(np.float32))
    np.save(femto_object_path, femto_objects.astype(np.float32))
    np.save(output, merged.astype(np.float32))
    color_output = output.with_name("fused_point_colors.npy")
    np.save(color_output, merged_colors.astype(np.float32))
    (output.parent / "fused_cloud_summary.json").write_text(
        json.dumps(
            {
                "d405_points_before_filter": int(len(d405)),
                "femto_points_before_filter": int(len(femto)),
                "d405_object_points": int(len(d405_objects)),
                "femto_object_points": int(len(femto_objects)),
                "d405_table_z_base_m": d405_table_z,
                "femto_table_z_base_m": femto_table_z,
                "table_height_disagreement_m": abs(d405_table_z - femto_table_z),
                "conservative_table_z_base_m": max(d405_table_z, femto_table_z),
                "object_min_height_m": min_height,
                "object_max_height_m": max_height,
                "d405_object_cloud_base": str(d405_object_path),
                "femto_object_cloud_base": str(femto_object_path),
                "merged_points_before_filter": int(len(merged_before_filter)),
                "merged_points_after_finite_filter": int(np.count_nonzero(np.all(np.isfinite(merged_before_filter), axis=1))),
                "merged_points_after_voxel_downsample": int(len(merged)),
                "voxel_size_m": float(args.fuse_voxel_size_m),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[fusion] table z: D405={:+.4f}m Femto={:+.4f}m disagreement={:.1f}mm".format(
            d405_table_z, femto_table_z, abs(d405_table_z - femto_table_z) * 1000.0
        )
    )
    if abs(d405_table_z - femto_table_z) > 0.005:
        print("[fusion] WARNING: camera table heights differ by more than 5 mm; check extrinsic calibration")
    print(f"-> fused point cloud: {len(merged)} points -> {output}")
    return output


def prepare_independent_object_references(
    args, d405_run: Path, femto_run: Path | None, output_dir: Path
) -> Path:
    """Save per-camera object clouds without registering or concatenating them."""
    output_dir.mkdir(parents=True, exist_ok=False)
    min_height = float(args.fusion_object_min_height_m)
    max_height = float(args.fusion_object_max_height_m)
    if min_height < 0.0 or max_height <= min_height:
        raise RuntimeError("invalid independent object height range")
    records = {}
    sources = [("d405", d405_run, "d405")]
    if femto_run is not None:
        sources.append(("femto_bolt", femto_run, "femto_bolt"))
    for source, run_dir, camera_dir in sources:
        assets_dir = run_dir / camera_dir
        meta = json.loads((assets_dir / "meta.json").read_text(encoding="utf-8"))
        transform = validate_transform(meta.get("T_base_camera"), f"{source} T_base_camera")
        points = np.asarray(np.load(assets_dir / "point_cloud.npy"), dtype=np.float64)
        colors = np.asarray(np.load(assets_dir / "point_colors.npy"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise RuntimeError(f"invalid {source} point-cloud assets")
        object_points, object_colors, table_z = extract_object_points(
            transform_points(points, transform), colors, min_height, max_height
        )
        object_points, object_colors, cluster_record = select_workpiece_clusters(
            object_points, object_colors,
            eps_m=float(args.object_cluster_eps_m),
            min_points=int(args.object_cluster_min_points),
            min_z_span_m=float(args.object_cluster_min_z_span_m),
        )
        cloud_path = output_dir / f"{source}_object_cloud_base.npy"
        color_path = output_dir / f"{source}_object_colors.npy"
        np.save(cloud_path, object_points.astype(np.float32))
        np.save(color_path, object_colors.astype(np.float32))
        records[source] = {
            "input_points": int(len(points)),
            "object_points": int(len(object_points)),
            "table_z_base_m": table_z,
            "object_cloud_base": str(cloud_path),
            "object_colors": str(color_path),
            "workpiece_cluster": cluster_record,
        }
        print(
            f"[{source}] workpiece reference: {len(object_points)} points "
            f"(elevated={cluster_record['elevated_points']}, clusters={cluster_record['selected_cluster_count']})"
        )
    summary_path = output_dir / "independent_object_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "selection_mode": "independent_d405_femto_graspnet" if femto_run is not None else "d405_only_graspnet",
                "object_min_height_m": min_height,
                "object_max_height_m": max_height,
                "sources": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def transform_grasps(grasps: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transformed = np.array(grasps, dtype=np.float64, copy=True)
    if transformed.ndim != 2 or transformed.shape[1] < 16:
        raise RuntimeError(f"grasps must have shape [N, >=16], got {transformed.shape}")
    if len(transformed) == 0:
        return transformed
    rotations = transformed[:, 4:13].reshape(-1, 3, 3)
    transformed[:, 4:13] = (transform[:3, :3] @ rotations).reshape(-1, 9)
    transformed[:, 13:16] = transform_points(transformed[:, 13:16], transform)
    return transformed


def run_fused_inference(
    args, cloud_path: Path, output_dir: Path, d405_run: Path
) -> Path:
    # GraspNet is trained in camera coordinates. Keep fusion and final filtering in
    # arm1_base, but rotate the fused cloud back into a camera frame for inference.
    d405_meta = json.loads((d405_run / "d405" / "meta.json").read_text(encoding="utf-8"))
    T_base_d405 = validate_transform(d405_meta.get("T_base_camera"), "D405 T_base_camera")
    T_d405_base = np.linalg.inv(T_base_d405)
    fused_base = np.asarray(np.load(cloud_path), dtype=np.float64)
    fused_colors = np.asarray(
        np.load(cloud_path.with_name("fused_point_colors.npy")), dtype=np.float64
    )
    fused_camera_path = cloud_path.with_name("fused_point_cloud_d405_camera.npy")
    fused_camera_colors_path = cloud_path.with_name("fused_point_colors_d405_camera.npy")
    np.save(fused_camera_path, transform_points(fused_base, T_d405_base).astype(np.float32))
    np.save(fused_camera_colors_path, fused_colors.astype(np.float32))

    output_dir.mkdir(parents=True, exist_ok=False)
    cmd = [
        args.python_bin, str(VERSION1),
        "--merged_point_cloud", str(fused_camera_path),
        "--merged_point_colors", str(fused_camera_colors_path),
        "--checkpoint_path", str(args.checkpoint_path),
        "--num_point", str(args.num_point),
        "--num_view", str(args.num_view),
        "--collision_thresh", str(args.collision_thresh),
        "--voxel_size", str(args.collision_voxel_size),
        "--approach_dist", str(args.collision_approach_dist),
        "--collision_table_thickness", str(args.collision_table_thickness),
        "--collision_table_plane_dist", str(args.collision_table_plane_dist),
        "--collision_table_min_inlier_ratio", str(args.collision_table_min_inlier_ratio),
        "--collision_table_min_normal_z", str(args.collision_table_min_normal_z),
        "--top_k", str(args.top_k),
        "--output_dir", str(output_dir),
    ]
    print("-> fused GraspNet:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=os.environ.copy())
    run = newest_run(output_dir)
    merged_dir = run / "merged"
    if not (merged_dir / "grasps.npy").exists():
        raise RuntimeError(f"fused inference output is incomplete: {run}")

    grasps_camera = np.asarray(np.load(merged_dir / "grasps.npy"), dtype=np.float64)
    np.save(merged_dir / "grasps_camera.npy", grasps_camera.astype(np.float32))
    np.save(
        merged_dir / "grasps.npy",
        transform_grasps(grasps_camera, T_base_d405).astype(np.float32),
    )
    np.save(merged_dir / "point_cloud_camera.npy", np.load(merged_dir / "point_cloud.npy"))
    np.save(merged_dir / "point_cloud.npy", fused_base.astype(np.float32))
    np.save(merged_dir / "point_colors.npy", fused_colors.astype(np.float32))
    meta_path = merged_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "coordinate_frame": "arm1_base",
        "inference_coordinate_frame": "d405_camera",
        "T_base_inference_camera": T_base_d405.tolist(),
        "source_point_cloud_base": str(cloud_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[fused] inferred {len(grasps_camera)} grasps in D405 camera coordinates "
        "and transformed them to arm1_base"
    )
    return run


def load_all_grasp_candidates(
    d405_run: Path, femto_run: Path | None, result_run: Path, fused_run: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    groups = [("d405", np.load(d405_run / "d405" / "grasps.npy"))]
    if femto_run is not None:
        groups.append(("femto_bolt", np.load(femto_run / "femto_bolt" / "grasps.npy")))
    if fused_run is not None:
        groups.append(("fused", np.load(fused_run / "merged" / "grasps.npy")))
    arrays = [np.asarray(group, dtype=np.float64) for _, group in groups]
    for (source, _), array in zip(groups, arrays):
        if array.ndim != 2 or array.shape[1] < 16:
            raise RuntimeError(f"{source} grasps must have shape [N, >=16], got {array.shape}")
    grasps = np.concatenate(arrays, axis=0)
    sources = np.concatenate([
        np.full(len(array), source, dtype="<U16")
        for (source, _), array in zip(groups, arrays)
    ])
    np.save(result_run / "merged" / "all_source_grasps.npy", grasps.astype(np.float32))
    (result_run / "merged" / "all_source_grasp_sources.json").write_text(
        json.dumps(sources.tolist(), indent=2), encoding="utf-8"
    )
    print(
        "[all-sources] candidates: "
        + ", ".join(f"{source}={len(array)}" for (source, _), array in zip(groups, arrays))
    )
    return grasps, sources


def count_object_points_in_closing_volume(
    grasps: np.ndarray, sources: np.ndarray, source_cloud_paths: dict[str, str | None],
) -> np.ndarray:
    """Count source-local workpiece points between the two predicted fingers.

    A grasp center merely being close to a workpiece is insufficient: it can
    still be a table grasp next to the part.  This uses the same GraspNet frame
    convention as the collision detector and tests the actual closing volume.
    """
    counts = np.zeros(len(grasps), dtype=np.int32)
    rotations = grasps[:, 4:13].reshape(-1, 3, 3)
    centers = grasps[:, 13:16]
    for source, cloud_path in source_cloud_paths.items():
        source_indices = np.flatnonzero(sources == source)
        if len(source_indices) == 0:
            continue
        if cloud_path is None:
            raise RuntimeError(f"no object cloud available for {source} candidates")
        points = np.asarray(np.load(cloud_path), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise RuntimeError(f"invalid workpiece reference cloud for {source}")
        # Limit the temporary [candidate, point, xyz] tensor.  Femto can have
        # many raw proposals, while all candidates must still be checked.
        for start in range(0, len(source_indices), 64):
            indices = source_indices[start:start + 64]
            local = np.matmul(points[None, :, :] - centers[indices, None, :], rotations[indices])
            heights = grasps[indices, 2, None]
            depths = grasps[indices, 3, None]
            widths = grasps[indices, 1, None]
            inner = (
                (local[:, :, 2] > -heights / 2.0)
                & (local[:, :, 2] < heights / 2.0)
                & (local[:, :, 0] > depths - 0.060)
                & (local[:, :, 0] < depths)
                & (local[:, :, 1] > -widths / 2.0)
                & (local[:, :, 1] < widths / 2.0)
            )
            counts[indices] = np.count_nonzero(inner, axis=1)
    return counts


def select_all_source_grasp(
    args, run_dir: Path, grasps: np.ndarray, sources: np.ndarray
) -> tuple[dict | None, np.ndarray]:
    if grasps.ndim != 2 or grasps.shape[1] < 16:
        return None, grasps
    max_width = float(args.gripper_open_width_m) - float(args.gripper_width_margin_m)
    min_angle = float(args.min_grasp_approach_angle_deg)
    if not np.isfinite(min_angle) or not 0.0 <= min_angle <= 90.0:
        raise RuntimeError("--min-grasp-approach-angle-deg must be finite and in [0, 90]")
    finite_mask = np.isfinite(grasps).all(axis=1)
    width_mask = finite_mask & (grasps[:, 1] <= max_width)

    rotations = grasps[:, 4:13].reshape(-1, 3, 3)
    approach_axis = rotations[:, :, 0]
    opening_axis = rotations[:, :, 1]
    horizontal_norm = np.linalg.norm(approach_axis[:, :2], axis=1)
    approach_angles = np.degrees(
        np.arctan2(np.abs(approach_axis[:, 2]), horizontal_norm)
    )
    downward_mask = approach_axis[:, 2] < 0.0
    approach_angle_mask = (
        np.isfinite(approach_angles) & (approach_angles >= min_angle) & downward_mask
    )
    mask = width_mask & approach_angle_mask

    meta = json.loads((run_dir / "merged" / "meta.json").read_text(encoding="utf-8"))
    if "object_summary_path" in meta:
        summary_path = Path(meta["object_summary_path"])
    else:
        summary_path = Path(meta["source_point_cloud_base"]).parent / "fused_cloud_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "sources" in summary:
        source_summary = summary["sources"]
        table_z_by_source = {
            source: float(record["table_z_base_m"])
            for source, record in source_summary.items()
        }
        source_cloud_paths = {
            source: record["object_cloud_base"]
            for source, record in source_summary.items()
        }
        if "fused" in set(sources.tolist()):
            table_z_by_source["fused"] = 0.0
            source_cloud_paths["fused"] = meta.get("source_point_cloud_base")
    else:
        table_z_by_source = {
            "d405": float(summary["d405_table_z_base_m"]),
            "femto_bolt": float(summary["femto_table_z_base_m"]),
            "fused": float(summary["conservative_table_z_base_m"]),
        }
        source_cloud_paths = {
            "d405": summary["d405_object_cloud_base"],
            "femto_bolt": summary["femto_object_cloud_base"],
            "fused": meta["source_point_cloud_base"],
        }
    max_object_distance = float(args.max_grasp_object_distance_m)
    if max_object_distance <= 0.0:
        raise RuntimeError("--max-grasp-object-distance-m must be positive")
    centers = grasps[:, 13:16]
    min_dist_sq = np.full(len(grasps), np.inf, dtype=np.float64)
    for source, source_cloud_path in source_cloud_paths.items():
        source_indices = np.flatnonzero(sources == source)
        if len(source_indices) == 0:
            continue
        if source_cloud_path is None:
            raise RuntimeError(f"no object cloud available for {source} candidates")
        source_cloud = np.asarray(np.load(source_cloud_path), dtype=np.float64)
        source_centers = centers[source_indices]
        source_min_dist_sq = np.full(len(source_indices), np.inf, dtype=np.float64)
        for start in range(0, len(source_cloud), 4096):
            chunk = source_cloud[start:start + 4096]
            distances_sq = np.sum(
                (source_centers[:, None, :] - chunk[None, :, :]) ** 2, axis=2
            )
            source_min_dist_sq = np.minimum(
                source_min_dist_sq, np.min(distances_sq, axis=1)
            )
        min_dist_sq[source_indices] = source_min_dist_sq
    object_proximity_mask = min_dist_sq <= max_object_distance ** 2
    mask &= object_proximity_mask

    min_support = int(args.min_grasp_object_support_points)
    if min_support < 1:
        raise RuntimeError("--min-grasp-object-support-points must be at least 1")
    object_support_counts = count_object_points_in_closing_volume(
        grasps, sources, source_cloud_paths
    )
    object_support_mask = object_support_counts >= min_support
    mask &= object_support_mask

    tabletop_mask = np.ones(len(grasps), dtype=bool)
    xy_mask = np.ones(len(grasps), dtype=bool)
    if not args.disable_tabletop_filter:
        table_z_per_grasp = np.array(
            [table_z_by_source[str(source)] for source in sources], dtype=np.float64
        )
        tabletop_mask = grasps[:, 15] >= table_z_per_grasp + float(args.table_clearance_m)
        mask &= tabletop_mask

        if args.table_xy_bounds is not None:
            x_min, x_max, y_min, y_max = (float(value) for value in args.table_xy_bounds)
            margin = float(args.table_edge_margin_m)
            if x_min >= x_max or y_min >= y_max or margin < 0.0:
                raise RuntimeError("invalid --table-xy-bounds or --table-edge-margin-m")
            opening_axis_xy = opening_axis[:, :2]
            half_width = grasps[:, 1:2] * 0.5
            centers_xy = grasps[:, 13:15]
            endpoints_xy = np.stack(
                (centers_xy - opening_axis_xy * half_width, centers_xy + opening_axis_xy * half_width),
                axis=1,
            )
            xy_mask = (
                (endpoints_xy[:, :, 0] >= x_min + margin)
                & (endpoints_xy[:, :, 0] <= x_max - margin)
                & (endpoints_xy[:, :, 1] >= y_min + margin)
                & (endpoints_xy[:, :, 1] <= y_max - margin)
            ).all(axis=1)
            mask &= xy_mask

    indices = np.flatnonzero(mask).astype(np.int64)
    source_stats = {
        source: {
            "input": int(np.count_nonzero(sources == source)),
            "kept_after_width": int(np.count_nonzero((sources == source) & width_mask)),
            "kept_after_downward_approach_angle": int(np.count_nonzero(
                (sources == source) & width_mask & approach_angle_mask
            )),
            "kept_after_object_proximity": int(np.count_nonzero(
                (sources == source) & width_mask & approach_angle_mask & object_proximity_mask
            )),
            "kept": int(np.count_nonzero((sources == source) & mask)),
            "max_object_support_points": int(object_support_counts[sources == source].max()),
            "kept_after_object_support": int(np.count_nonzero(
                (sources == source) & width_mask & approach_angle_mask & object_proximity_mask & object_support_mask
            )),
        }
        for source in sorted(set(sources.tolist()))
    }
    for source, stats in source_stats.items():
        print(
            f"[{source}] grasp filter stages: input={stats['input']}, "
            f"width={stats['kept_after_width']}, "
            f"downward={stats['kept_after_downward_approach_angle']}, "
            f"near_workpiece={stats['kept_after_object_proximity']}, "
            f"support_max={stats['max_object_support_points']}, "
            f"kept_after_support={stats['kept_after_object_support']}, final={stats['kept']}"
        )
    (run_dir / "all_source_grasp_filter.json").write_text(
        json.dumps(
            {
                "input_grasps": int(len(grasps)),
                "kept": int(len(indices)),
                "kept_after_width": int(np.count_nonzero(width_mask)),
                "kept_after_downward_approach_angle": int(np.count_nonzero(width_mask & approach_angle_mask)),
                "kept_after_object_proximity": int(np.count_nonzero(width_mask & approach_angle_mask & object_proximity_mask)),
                "min_grasp_object_support_points": min_support,
                "kept_after_object_support": int(np.count_nonzero(width_mask & approach_angle_mask & object_proximity_mask & object_support_mask)),
                "kept_after_tabletop": int(np.count_nonzero(width_mask & approach_angle_mask & object_proximity_mask & object_support_mask & tabletop_mask)),
                "kept_after_xy_bounds": int(np.count_nonzero(width_mask & approach_angle_mask & object_proximity_mask & object_support_mask & tabletop_mask & xy_mask)),
                "collision_policy": "already_filtered_per_source_during_each_graspnet_inference",
                "min_grasp_approach_angle_deg": min_angle,
                "approach_direction": "arm1_base_negative_z",
                "tabletop_filter_enabled": not args.disable_tabletop_filter,
                "estimated_table_z_base_m_by_source": table_z_by_source,
                "table_clearance_m": float(args.table_clearance_m),
                "max_grasp_object_distance_m": max_object_distance,
                "object_support_points": {
                    "min": int(object_support_counts.min()) if len(object_support_counts) else 0,
                    "max": int(object_support_counts.max()) if len(object_support_counts) else 0,
                },
                "sources": source_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[all-sources] grasp filter: downward approach_angle>={:.1f}deg tabletop={} kept={}/{}".format(
            min_angle, not args.disable_tabletop_filter, len(indices), len(grasps)
        )
    )
    if len(indices) == 0:
        return None, indices
    ranked_indices = indices[np.argsort(grasps[indices, 0])[::-1]]
    ranked = [
        {
            "rank": rank,
            "source_grasp_index": int(index),
            "grasp_source": str(sources[index]),
            "score": float(grasps[index, 0]),
            "center_base_m": grasps[index, 13:16].tolist(),
            "ik_status": "not_checked_sdk_has_no_query_api",
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]
    (run_dir / "ranked_grasp_candidates.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )
    selected_index = int(ranked_indices[0])
    selected = grasp_result(run_dir, grasps[selected_index])
    selected["source_grasp_index"] = selected_index
    selected["grasp_source"] = str(sources[selected_index])
    (run_dir / "selected_executable_grasp.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    return selected, indices


def show_fused_visualization(
    args,
    run_dir: Path,
    grasps: np.ndarray,
    sources: np.ndarray,
    indices: np.ndarray,
    selected_index: int | None,
    execution_plan: dict | None = None,
) -> None:
    if not args.vis:
        return
    if not args.display:
        raise RuntimeError("fused visualization requires --display or use --no-vis")
    os.environ["DISPLAY"] = str(args.display)
    os.environ.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))
    import open3d as o3d
    from graspnetAPI import GraspGroup

    merged_dir = run_dir / "merged"
    cloud_path = merged_dir / "point_cloud.npy"
    colors_path = merged_dir / "point_colors.npy"
    allowed_indices = np.arange(len(grasps), dtype=np.int64)
    if not cloud_path.exists():
        meta = json.loads((merged_dir / "meta.json").read_text(encoding="utf-8"))
        summary = json.loads(Path(meta["object_summary_path"]).read_text(encoding="utf-8"))
        active_source = (
            str(sources[selected_index])
            if selected_index is not None
            else str(sources[int(np.argmax(grasps[:, 0]))]) if len(grasps) else "d405"
        )
        cloud_path = Path(summary["sources"][active_source]["object_cloud_base"])
        colors_path = Path(summary["sources"][active_source]["object_colors"])
        allowed_indices = np.flatnonzero(sources == active_source)
        print(f"[visualization] independent source: {active_source}")
    cloud = np.asarray(np.load(cloud_path), dtype=np.float64)
    geometries = [o3d.geometry.PointCloud()]
    geometries[0].points = o3d.utility.Vector3dVector(cloud)
    if colors_path.exists():
        colors = np.asarray(np.load(colors_path), dtype=np.float64)
        if colors.shape != cloud.shape or not np.all(np.isfinite(colors)):
            raise RuntimeError("fused visualization colors do not match the point cloud")
        geometries[0].colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    else:
        geometries[0].paint_uniform_color([0.55, 0.55, 0.55])
    executable_indices = np.intersect1d(indices, allowed_indices, assume_unique=True)
    if len(executable_indices) > 0:
        display_indices = executable_indices[np.argsort(grasps[executable_indices, 0])[::-1]][: args.top_k]
    else:
        display_indices = allowed_indices[np.argsort(grasps[allowed_indices, 0])[::-1]][: args.top_k]
    if selected_index is not None and selected_index not in display_indices:
        display_indices = np.r_[display_indices, selected_index]
    grippers = []
    if len(display_indices) > 0:
        group = GraspGroup(grasps[display_indices])
        grippers = group.to_open3d_geometry_list()
    executable = set(int(index) for index in indices)
    for index, gripper in zip(display_indices, grippers):
        if selected_index is not None and int(index) == selected_index:
            # This is still the raw network pose.  Candidate planning may
            # choose the 180-degree parallel-jaw equivalent, so do not label
            # it as the pose that will actually be sent to Planner.
            color = [1.0, 0.0, 1.0]
        elif int(index) in executable:
            color = [1.0, 0.75, 0.0]
        else:
            color = [1.0, 0.0, 0.0]
        gripper.paint_uniform_color(color)

    if execution_plan is not None:
        if selected_index is None:
            raise RuntimeError("execution-plan visualization requires a selected grasp")
        commanded = np.array(grasps[int(selected_index)], copy=True)
        commanded_rotation = np.asarray(
            execution_plan.get("eef_rotation_base"), dtype=np.float64
        )
        commanded_center = np.asarray(
            execution_plan.get("grasp_center_base_m"), dtype=np.float64
        )
        if (
            commanded_rotation.shape != (3, 3)
            or commanded_center.shape != (3,)
            or not np.all(np.isfinite(commanded_rotation))
            or not np.all(np.isfinite(commanded_center))
        ):
            raise RuntimeError(
                "execution plan lacks a finite EEF rotation/grasp center for visualization"
            )
        # GraspNet rows and the object cloud are both expressed in arm1_base.
        # Render the final SDK-base left_eef command before the fixed
        # SDK->Planner base conversion.  This includes the selected 0/180
        # parallel-jaw equivalent and is the pose execution consumes.
        commanded[4:13] = commanded_rotation.reshape(-1)
        commanded[13:16] = commanded_center
        commanded_geometry = GraspGroup(
            commanded[None, :]
        ).to_open3d_geometry_list()[0]
        commanded_geometry.paint_uniform_color([0.0, 1.0, 0.0])
        grippers.append(commanded_geometry)
        print(
            "[visualization] colors: green=final commanded left_eef pose, "
            "magenta=raw selected GraspNet pose, orange=other executable candidates, "
            "red=non-executable candidates"
        )
    o3d.visualization.draw_geometries(
        geometries + grippers, window_name="Version5 independent camera candidates"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.roi is None or args.femto_roi is None:
        raise RuntimeError(
            "Version5 requires both --roi for D405 and --femto-roi for Femto; "
            "full-frame point clouds are not allowed in fusion"
        )
    if args.preview_pregrasp and args.execute_grasp:
        raise RuntimeError("--preview-pregrasp and --execute-grasp cannot be used together")
    if args.preview_pregrasp and not args.confirm_real_pregrasp:
        raise RuntimeError("--preview-pregrasp requires --confirm-real-pregrasp")
    if args.gripper_speed is not None:
        if not np.isfinite(float(args.gripper_speed)) or not 1.0 <= float(args.gripper_speed) <= 100.0:
            raise RuntimeError("--gripper-speed is a percentage and must be in [1, 100]; omit it for AG grippers")
    # The approach angle is a candidate filter, never a fixed EEF rotation.
    args.gripper_yaw_offset_deg = 0.0
    load_wrist_calibration(Path(args.wrist_calibration))
    load_transform(Path(args.laser_calibration), "T_base_laser", require_accepted=True)
    load_tcp_translation(Path(args.tcp_translation_calibration))
    output_root = Path(args.output_dir) if args.output_dir else HERE.parent / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root.mkdir(parents=True, exist_ok=False)

    if not move_to_observation(args):
        return
    d405_run = run_camera_inference(args, "d405", run_root / "d405_inference")
    femto_run = run_camera_inference(args, "femto_bolt", run_root / "femto_inference")
    if args.enable_fused_inference:
        fused_path = fuse_clouds(args, d405_run, femto_run, run_root / "fused_point_cloud.npy")
        result_run = run_fused_inference(
            args, fused_path, run_root / "fused_inference", d405_run
        )
        fused_run = result_run
        print("-> experimental fused GraspNet enabled")
    else:
        fused_run = None
        object_summary_path = prepare_independent_object_references(
            args, d405_run, femto_run, run_root / "independent_object_references"
        )
        result_run = run_root / "independent_selection"
        merged_dir = result_run / "merged"
        merged_dir.mkdir(parents=True, exist_ok=False)
        (merged_dir / "meta.json").write_text(
            json.dumps(
                {
                    "coordinate_frame": "arm1_base",
                    "selection_mode": "independent_d405_femto_graspnet",
                    "object_summary_path": str(object_summary_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("-> fused GraspNet disabled; selecting only independent D405/Femto candidates")
    grasps, grasp_sources = load_all_grasp_candidates(
        d405_run, femto_run, result_run, fused_run
    )
    selected, indices = select_all_source_grasp(args, result_run, grasps, grasp_sources)
    if selected is None:
        print("-> no grasp from any source survived; see all_source_grasp_filter.json")
        show_fused_visualization(args, result_run, grasps, grasp_sources, indices, None)
        return
    selected_index = int(selected["source_grasp_index"])
    show_fused_visualization(args, result_run, grasps, grasp_sources, indices, selected_index)
    if not grasp_tcp_is_certified(Path(args.tcp_translation_calibration)):
        print(
            "-> selected grasp is visualization-only: TCP calibration is not certified "
            "as the pinch/grasp TCP; real execution is blocked"
        )
        return
    plan = build_execution_plan(args, selected, capture_eef_rotation(d405_run, load_wrist_calibration(Path(args.wrist_calibration))))
    plan["source"] = (
        "d405_femto_bolt_and_experimental_fused"
        if fused_run is not None
        else "independent_d405_femto_bolt"
    )
    plan["selected_grasp_source"] = selected["grasp_source"]
    plan["d405_inference_run"] = str(d405_run)
    plan["femto_inference_run"] = str(femto_run)
    plan["fused_inference_run"] = None if fused_run is None else str(fused_run)
    plan_path = run_root / "grasp_execution_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("[fused] grasp execution plan:", plan_path)
    if args.preview_pregrasp:
        preview_pregrasp(args, plan)
    else:
        execute_grasp(args, plan)


if __name__ == "__main__":
    main()
