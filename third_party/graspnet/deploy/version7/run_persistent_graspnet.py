#!/usr/bin/env python3
"""Version5-compatible dual-camera flow using the persistent Version7 service."""
from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEPLOY = HERE.parent
for path in (ROOT, DEPLOY, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deploy.tools import graspnet_planner_execute as planner_execute  # noqa: E402
from deploy.version5 import run_dual_camera_fused_graspnet as version5  # noqa: E402
from camera_client import capture, capture_pair, status as camera_status  # noqa: E402
from graspnet_client import infer, infer_batch, status as graspnet_status  # noqa: E402
from pointcloud_preprocess import prepare_capture  # noqa: E402
from workpiece2_cloud_thickening import (  # noqa: E402
    build_thickened_inference_cloud,
    rank_candidate_indices_by_steep_surface,
    validate_args as validate_workpiece2_cloud_thickening_args,
)


SERVICE_JOBS = HERE / "jobs"
_BASE_BUILD_PARSER = planner_execute.build_parser
NO_GRASP_EXIT_CODE = 20
WEIGHT_CONFIG_PATH = HERE.parents[3] / "config" / "grasp_weight_thresholds.conf"


def load_shared_weight_defaults() -> dict[str, float | int]:
    """Read the Loop-wide weight settings for direct Python invocations too."""
    required = {
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
        raise FileNotFoundError(f"共享重量配置不存在: {WEIGHT_CONFIG_PATH}")

    values: dict[str, str] = {}
    for raw_line in WEIGHT_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"共享重量配置格式错误: {raw_line!r}")
        values[key.strip()] = value.strip()

    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"共享重量配置缺少字段: {missing}")
    result: dict[str, float | int] = {}
    for source, target in required.items():
        result[target] = int(values[source]) if source == "WEIGHT_CHECK_SAMPLES" else float(values[source])
    return result


def print_elapsed(label: str, started: float) -> float:
    elapsed = time.monotonic() - started
    print(f"[timing] {label}: {elapsed:.3f}s")
    return elapsed


def build_parser():
    parser = _BASE_BUILD_PARSER()
    # The Shell entry and direct Python entry share one editable Loop config.
    weight_defaults = load_shared_weight_defaults()
    parser.set_defaults(
        **weight_defaults,
        weight_contiguous_single_band=True,
        # The resident Camera Broker enables D405 hole filling by default.
        # Keep the front-end's expected configuration identical so its
        # preflight remains a useful ownership/configuration guard.
        d405_hole_filling=True,
    )
    group = parser.add_argument_group("Persistent GraspNet service")
    group.add_argument("--graspnet-socket", default="/tmp/graspnet-v7.sock")
    group.add_argument("--graspnet-service-jobs", default=str(SERVICE_JOBS))
    group.add_argument("--graspnet-timeout-s", type=float, default=120.0)
    group.add_argument("--camera-socket", default="/tmp/graspnet-v7-camera.sock")
    group.add_argument("--camera-timeout-s", type=float, default=30.0)
    group.add_argument("--camera-max-start-skew-s", type=float, default=.2,
                       help="maximum allowed D405/Femto capture-start skew; a larger value means non-synchronous inputs")
    group.add_argument("--femto-warmup-frames", type=int, default=3,
                       help="must match the resident Camera Broker profile")
    group.add_argument("--femto-request-flush-frames", type=int,
                       default=int(os.environ.get("CAMERA_BROKER_FEMTO_REQUEST_FLUSH_FRAMES", "30")),
                       help="must match the resident Camera Broker request-boundary stale-frame flush")
    group.add_argument("--femto-frame-read-retries", type=int,
                       default=int(os.environ.get("CAMERA_BROKER_FEMTO_FRAME_READ_RETRIES", "10")),
                       help="must match the resident Camera Broker incomplete-frame retry budget")
    group.add_argument("--max-no-grasp-retries", type=int, default=3,
                       help="immediately re-capture/re-infer at the observation pose when all candidate grasps are rejected")
    group.add_argument("--max-weight-retries", type=int, default=3,
                       help="maximum fresh Version7 observation/inference cycles after EMPTY/MULTIPLE/UNKNOWN force checks")
    group.add_argument("--max-operator-declines", type=int, default=20,
                       help="separate safety limit for manual non-y confirmations; these do not consume --max-weight-retries")
    group.add_argument(
        "--min-grasp-edge-clearance-m", type=float, default=.004,
        help=("minimum workpiece-point extent required on both sides of the grasp center in each "
              "in-plane gripper axis; rejects centers too close to an outer edge/hole edge (0 disables)"),
    )
    group.add_argument("--planner-candidate-top-k", type=int, default=3,
                       help="number of score-ranked GraspNet candidates to preflight with the Planner")
    group.add_argument("--max-planner-unreachable-retries", type=int, default=30,
                       help="fresh capture/inference cycles after every top-K candidate is Planner-unreachable")
    group.add_argument("--empty-failure-ban-count", type=int, default=2,
                       help="blacklist a local grasp pose after this many EMPTY weight results (default: 2)")
    group.add_argument("--empty-failure-position-radius-m", type=float, default=.020,
                       help="translation radius for grouping repeated EMPTY grasp failures (meters)")
    group.add_argument("--empty-failure-orientation-deg", type=float, default=15.0,
                       help="parallel-jaw-symmetry-aware orientation radius for grouping EMPTY failures (degrees)")
    group.add_argument("--handoff-success-file", default=None,
                       help="write this JSON marker only after a verified real grasp reaches graspup")
    group.add_argument("--d405-color-profile", default=None,
                       help="optional D405 color filter before GraspNet; omitted by default so the full ROI is inferred")
    group.add_argument("--femto-color-profile", default=None,
                       help="optional Femto color filter before GraspNet; omitted by default so the full ROI is inferred")
    group.add_argument(
        "--disable-femto",
        action="store_true",
        help="run D405-only capture, inference, filtering, and grasp selection; Femto frames and calibration are not used",
    )
    group.add_argument(
        "--grasp-downward-probe-m", type=float, default=0.0,
        help=("signed final Z adjustment from the predicted grasp TCP in arm1_base (meters): "
              "positive descends along -Z, negative raises along +Z; Version7 defaults to zero"),
    )
    group.add_argument(
        "--stop-after-graspup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("stop with the gripper closed after the post-grasp/graspup joint pose; "
              "Version7 hands the held workpiece to Module 2 by default"),
    )
    group.add_argument(
        "--workpiece-id", choices=("1", "2"), default=None,
        help="Loop workpiece identity; required only for workpiece-specific inference options",
    )
    group.add_argument(
        "--enable-workpiece2-cloud-thickening", action="store_true",
        help=("workpiece-2 only: create an inference-only D405 cloud with normal offsets on "
              "local surfaces steep relative to the table"),
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-offsets-mm", type=float, nargs="+", default=[1.5, 3.0],
        help="positive normal offsets; each generates both +/- virtual layers (default: 1.5 3.0)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-max-source-points", type=int, default=10000,
        help="maximum elevated D405 points used to make virtual layers (default: 10000)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-plane-distance-m", type=float, default=.003,
        help="RANSAC table-plane inlier distance in metres (default: 0.003)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-min-height-m", type=float, default=.004,
        help="minimum signed height above table eligible for extrusion (default: 0.004)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-max-height-m", type=float, default=.120,
        help="maximum signed height above table eligible for extrusion (default: 0.120)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-normal-radius-m", type=float, default=.010,
        help="Open3D normal-estimation radius in metres (default: 0.010)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-normal-max-nn", type=int, default=30,
        help="Open3D normal-estimation maximum neighbours (default: 30)",
    )
    group.add_argument(
        "--workpiece2-cloud-thickening-min-plane-angle-deg", type=float, default=65.0,
        help=("only extrude local planes at least this angle from the table "
              "(default: 65; vertical is 90)"),
    )
    group.add_argument(
        "--workpiece2-steep-surface-priority", action=argparse.BooleanOptionalAction,
        default=True,
        help=("workpiece-2 only: rank candidates whose centre and jaw-contact samples are near "
              "physical D405 surfaces >= the configured table angle before raw GraspNet score"),
    )
    group.add_argument(
        "--workpiece2-steep-surface-contact-distance-m", type=float, default=.015,
        help=("maximum distance from a candidate centre/jaw-contact sample to a physical steep "
              "surface for workpiece-2 ranking (default: 0.015 m)"),
    )
    group.add_argument(
        "--workpiece2-steep-surface-min-score-ratio", type=float, default=.60,
        help=("only a steep-surface candidate whose raw GraspNet score is at least this ratio of "
              "the frame best score may be promoted (default: 0.60)"),
    )
    return parser


def _service_request(
    args, run: Path, camera: str, num_point: int, *, point_cloud_path: Path | None = None,
) -> dict:
    asset_name = "d405" if camera == "d405" else "femto_bolt"
    assets = run / asset_name
    cloud_path = assets / "point_cloud.npy" if point_cloud_path is None else Path(point_cloud_path)
    grasps_camera_path = assets / "grasps_camera.npy"
    return {"point_cloud_path": str(cloud_path.resolve()), "output_grasps_path": str(grasps_camera_path.resolve()),
            "num_point": int(num_point), "num_view": int(args.num_view), "collision_thresh": float(args.collision_thresh),
            "voxel_size": float(args.collision_voxel_size), "approach_dist": float(args.collision_approach_dist),
            "collision_table_thickness": float(args.collision_table_thickness),
            "collision_table_plane_dist": float(args.collision_table_plane_dist),
            "collision_table_min_inlier_ratio": float(args.collision_table_min_inlier_ratio),
            "collision_table_min_normal_z": float(args.collision_table_min_normal_z)}


def _expected_camera_config(args) -> dict:
    config = {"arm": {"host": str(args.arm_host), "id": int(args.arm_id)},
              "d405": {"serial": str(args.d405_serial), "width": 640, "height": 480,
                        "postprocess": bool(args.d405_postprocess), "hole_filling": bool(args.d405_hole_filling)}}
    if not args.disable_femto:
        config["femto_bolt"] = {"serial": str(args.femto_serial),
                                 "color_width": int(args.femto_color_width), "color_height": int(args.femto_color_height,
                                 ), "depth_width": int(args.femto_depth_width), "depth_height": int(args.femto_depth_height),
                                 "warmup_frames": int(args.femto_warmup_frames),
                                 "request_flush_frames": int(args.femto_request_flush_frames),
                                 "frame_read_retries": int(args.femto_frame_read_retries)}
    return config


def _config_differences(
    actual: dict,
    expected: dict,
    prefix: str = "",
    *,
    allow_extra_actual: bool = False,
) -> list[str]:
    """Return leaf-level service/command differences for actionable preflight errors."""
    differences = []
    for key in sorted(set(actual) | set(expected)):
        name = f"{prefix}.{key}" if prefix else str(key)
        if key not in actual:
            differences.append(f"{name}: service=<missing>, command={expected[key]!r}")
        elif key not in expected:
            if allow_extra_actual:
                continue
            differences.append(f"{name}: service={actual[key]!r}, command=<missing>")
        elif isinstance(actual[key], dict) and isinstance(expected[key], dict):
            differences.extend(
                _config_differences(
                    actual[key],
                    expected[key],
                    name,
                    allow_extra_actual=allow_extra_actual,
                )
            )
        elif actual[key] != expected[key]:
            differences.append(f"{name}: service={actual[key]!r}, command={expected[key]!r}")
    return differences


def verify_persistent_services(args) -> None:
    """Fail before robot motion when a front-end command disagrees with a warm service."""
    jobs_root = str(Path(args.graspnet_service_jobs).resolve())
    camera = camera_status(args.camera_socket)
    graspnet = graspnet_status(args.graspnet_socket)
    errors = []
    if camera.get("allow_root") != jobs_root:
        errors.append(f"camera allow-root={camera.get('allow_root')} but --graspnet-service-jobs={jobs_root}")
    if graspnet.get("allow_root") != jobs_root:
        errors.append(f"GraspNet allow-root={graspnet.get('allow_root')} but --graspnet-service-jobs={jobs_root}")
    if int(graspnet.get("num_view", -1)) != int(args.num_view):
        errors.append(f"GraspNet num_view={graspnet.get('num_view')} but command requests {args.num_view}")
    expected_camera = _expected_camera_config(args)
    actual_camera = camera.get("camera_config")
    # The resident broker owns both cameras for the whole loop.  A D405-only
    # grasp command may therefore legitimately omit Femto from its request.
    differences = _config_differences(
        actual_camera if isinstance(actual_camera, dict) else {},
        expected_camera,
        allow_extra_actual=True,
    )
    if differences:
        errors.append("camera startup configuration differs from this command:\n  - " + "\n  - ".join(differences))
    if errors:
        raise RuntimeError("persistent service preflight failed:\n- " + "\n- ".join(errors))
    print("-> persistent service preflight passed (paths, num_view, camera configuration)")


def _finalize_inference(args, camera: str, run: Path, response: dict) -> Path:
    asset_name = "d405" if camera == "d405" else "femto_bolt"
    assets = run / asset_name
    grasps_camera_path = assets / "grasps_camera.npy"
    raw = np.asarray(np.load(grasps_camera_path), dtype=np.float32)
    meta_path = assets / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if camera == "d405":
        T_base_camera = np.asarray(meta.get("T_base_eef"), dtype=np.float64) @ version5.load_wrist_calibration(Path(args.wrist_calibration))
    else:
        T_base_camera = version5.load_transform(Path(args.laser_calibration), "T_base_laser", require_accepted=True)
    grasps_base = version5.transform_grasps(raw, T_base_camera)
    np.save(assets / "grasps.npy", grasps_base.astype(np.float32))
    meta.update({
        "coordinate_frame": "arm1_base",
        "T_base_camera": T_base_camera.tolist(),
        "num_grasps": int(len(raw)),
        "persistent_graspnet_service": response,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (assets / "service_inference.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    expected_roi = args.roi if camera == "d405" else args.femto_roi
    if [int(value) for value in meta.get("roi_xywh") or []] != [int(value) for value in expected_roi or []]:
        raise RuntimeError(f"{camera} ROI was not applied during capture: {meta.get('roi_xywh')}")
    print("[{}] persistent inference: points={}, grasps={}, model={:.3f}s, collision={:.3f}s, total={:.3f}s".format(
        camera, response["num_input_points"], response["num_grasps"], response.get("model_elapsed_s", 0.0),
        response.get("collision_elapsed_s", 0.0), response.get("inference_elapsed_s", 0.0)))
    return run


def capture_preprocess_both(args) -> tuple[Path, Path | None]:
    """Capture D405, and optionally Femto, then prepare active point clouds."""
    root = Path(args.graspnet_service_jobs).resolve()
    root.mkdir(parents=True, exist_ok=True)
    job = root / ("dual_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    raw_d405, raw_femto = job / "raw" / "d405", job / "raw" / "femto_bolt"
    started = time.monotonic()
    d405_run, femto_run = job / "d405_inference", job / "femto_inference"
    if args.disable_femto:
        print("-> D405-only mode: Femto capture, inference, and candidates are disabled")
        capture_result = capture(args.camera_socket,
            {"camera": "d405", "output_dir": str(raw_d405), "frame_count": int(args.d405_frame_count)},
            float(args.camera_timeout_s))
        print("-> D405 capture: {:.3f}s".format(capture_result["capture_elapsed_s"]))
        d405_info = prepare_capture(raw_d405, d405_run / "d405", roi_xywh=list(args.roi),
                                    z_min=float(args.d405_z_min), z_max=float(args.d405_z_max),
                                    x_abs_max=float(args.d405_x_abs_max), y_abs_max=float(args.d405_y_abs_max),
                                    color_profile_path=args.d405_color_profile)
        print("-> D405 CPU preprocessing: {} points ({:.3f}s), capture+CPU={:.3f}s".format(
            d405_info["num_points"], d405_info["elapsed_s"], time.monotonic() - started))
        infos = (("D405", d405_info),)
        femto_run = None
    else:
        print("-> requesting concurrent D405 + Femto RGB-D from persistent camera service")
        capture_result = capture_pair(args.camera_socket,
                     {"output_dir": str(raw_d405), "frame_count": int(args.d405_frame_count)},
                     {"output_dir": str(raw_femto)}, float(args.camera_timeout_s), float(args.camera_max_start_skew_s))
        print("-> paired capture: start_skew={:.3f}s, D405={:.3f}s, Femto={:.3f}s".format(
            capture_result["start_skew_s"], capture_result["d405"]["capture_elapsed_s"],
            capture_result["femto_bolt"]["capture_elapsed_s"]))
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="v7-cloud") as workers:
            d405_future = workers.submit(prepare_capture, raw_d405, d405_run / "d405", roi_xywh=list(args.roi),
                                         z_min=float(args.d405_z_min), z_max=float(args.d405_z_max),
                                         x_abs_max=float(args.d405_x_abs_max), y_abs_max=float(args.d405_y_abs_max),
                                         color_profile_path=args.d405_color_profile)
            femto_future = workers.submit(prepare_capture, raw_femto, femto_run / "femto_bolt", roi_xywh=list(args.femto_roi),
                                         z_min=float(args.femto_z_min), z_max=float(args.femto_z_max),
                                         x_abs_max=float(args.femto_x_abs_max), y_abs_max=float(args.femto_y_abs_max),
                                         color_profile_path=args.femto_color_profile)
            d405_info, femto_info = d405_future.result(), femto_future.result()
        print("-> CPU preprocessing complete in parallel: D405={} points ({:.3f}s), Femto={} points ({:.3f}s), capture+CPU={:.3f}s".format(
            d405_info["num_points"], d405_info["elapsed_s"], femto_info["num_points"], femto_info["elapsed_s"], time.monotonic() - started))
        infos = (("D405", d405_info), ("Femto", femto_info))
    for label, info in infos:
        record = info["color_filter"]
        if record.get("enabled"):
            print("-> {} optional color filter: {} -> {} points ({})".format(
                label, record["points_before_color_filter"], record["points_after_color_filter"], record["profile_name"]
            ))
        else:
            print("-> {} inference cloud: full workspace-filtered ROI, {} points".format(
                label, record["points_after_color_filter"]
            ))
    return d405_run, femto_run


def no_fused_subprocess(*_args, **_kwargs):
    raise RuntimeError("Version7 persistent service currently handles D405/Femto independent inference only; omit --enable-fused-inference")


def _is_batch_resource_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("out of memory", "cuda error: out of memory", "cuda oom", "cublas", "cuda memory"))


def infer_with_batch_fallback(args, requests: list[dict]) -> list[dict]:
    """Use the fast batch path, but recover safely when a deployed GPU cannot fit it."""
    try:
        return infer_batch(args.graspnet_socket, requests, timeout_s=float(args.graspnet_timeout_s))
    except RuntimeError as exc:
        if not _is_batch_resource_error(exc):
            raise
        print(f"-> batch=2 GraspNet exhausted GPU resources ({exc}); retrying each camera sequentially")
        return [infer(args.graspnet_socket, item, timeout_s=float(args.graspnet_timeout_s)) for item in requests]


def move_to_observation_and_open(args) -> bool:
    stage_started = time.monotonic()
    if not planner_execute.move_to_observation_with_planner(args):
        return False
    print_elapsed("go_graspnet / D405 observation motion", stage_started)
    observation_ready_value = os.environ.get("LOOP_GRASPNET_OBSERVATION_READY_FILE")
    if observation_ready_value:
        observation_ready_path = Path(observation_ready_value)
        temporary = observation_ready_path.with_name(
            f".{observation_ready_path.name}.tmp"
        )
        temporary.write_text("ready\n", encoding="utf-8")
        temporary.replace(observation_ready_path)
        print(
            "-> 左臂已到 GraspNet 观察位；已通知总控可以启动右臂 ServoPose。",
            flush=True,
        )
    stage_started = time.monotonic()
    planner_execute.move_gripper(args, 100.0, "open at GraspNet observation")
    print_elapsed("open gripper at GraspNet observation", stage_started)
    return True


def filter_grasp_edge_clearance(
    args: argparse.Namespace, grasps: np.ndarray, sources: np.ndarray, indices: np.ndarray,
    object_summary: Path, output_dir: Path,
) -> np.ndarray:
    """Reject candidates whose closing center is too near a workpiece edge.

    Version5 already proves that enough workpiece points are inside the closing
    volume.  That alone still accepts a grasp placed near an outer contour: all
    support can be on one side of the center.  Here the same closing volume is
    split about the center along local +Y/-Y (jaw opening) and +Z/-Z (the other
    in-plane gripper axis).  A valid centered grasp must have support extending
    by ``edge_clearance`` in every direction.
    """
    clearance = float(args.min_grasp_edge_clearance_m)
    if not np.isfinite(clearance) or clearance < 0.0:
        raise RuntimeError("--min-grasp-edge-clearance-m must be finite and non-negative")
    if clearance == 0.0 or len(indices) == 0:
        return indices
    summary = json.loads(Path(object_summary).read_text(encoding="utf-8"))
    source_info = summary.get("sources", {})
    retained: list[int] = []
    records: list[dict] = []
    cloud_cache: dict[str, np.ndarray] = {}
    for index in (int(value) for value in indices):
        source = str(sources[index])
        record: dict = {"source_grasp_index": index, "source": source}
        try:
            cloud_path = Path(source_info[source]["object_cloud_base"])
            if source not in cloud_cache:
                points = np.asarray(np.load(cloud_path), dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                    raise RuntimeError("invalid workpiece reference cloud")
                cloud_cache[source] = points
            points = cloud_cache[source]
            grasp = np.asarray(grasps[index], dtype=np.float64)
            rotation = grasp[4:13].reshape(3, 3)
            center = grasp[13:16]
            width, height, depth = (float(grasp[1]), float(grasp[2]), float(grasp[3]))
            local = (points - center) @ rotation
            # Same local closing region used by Version5's support test.
            inside = (
                (local[:, 2] > -height / 2.0) & (local[:, 2] < height / 2.0)
                & (local[:, 0] > depth - 0.060) & (local[:, 0] < depth)
                & (local[:, 1] > -width / 2.0) & (local[:, 1] < width / 2.0)
            )
            support = local[inside]
            if len(support) == 0:
                extents = {"minus_y_m": 0.0, "plus_y_m": 0.0, "minus_z_m": 0.0, "plus_z_m": 0.0}
            else:
                y, z = support[:, 1], support[:, 2]
                extents = {
                    "minus_y_m": max(0.0, float(-np.min(y))),
                    "plus_y_m": max(0.0, float(np.max(y))),
                    "minus_z_m": max(0.0, float(-np.min(z))),
                    "plus_z_m": max(0.0, float(np.max(z))),
                }
            minimum_extent = min(extents.values())
            accepted = minimum_extent >= clearance
            record.update({
                "support_points": int(len(support)), "edge_extents_m": extents,
                "minimum_extent_m": minimum_extent, "required_clearance_m": clearance,
                "accepted": accepted,
            })
            if accepted:
                retained.append(index)
        except Exception as exc:
            # A malformed reference cloud must never make an edge candidate
            # look safe.  Keep the error observable in the filter artifact.
            record.update({"accepted": False, "reason": str(exc)})
        records.append(record)
    payload = {
        "method": "Version7 symmetric workpiece support in gripper local Y/Z",
        "minimum_edge_clearance_m": clearance,
        "input_candidates": int(len(indices)), "kept": int(len(retained)),
        "rejected": int(len(indices) - len(retained)), "candidates": records,
    }
    (output_dir / "grasp_edge_clearance_filter.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        "[all-sources] edge-clearance filter: required={:.1f}mm kept={}/{}".format(
            clearance * 1000.0, len(retained), len(indices)
        )
    )
    return np.asarray(retained, dtype=np.int64)


def find_selected_grasp(args, cycle_root: Path) -> tuple[dict | None, tuple | None]:
    """Run Version5 candidate filtering, with only vision-stage retries here."""
    total_attempts = int(args.max_no_grasp_retries) + 1
    for attempt in range(1, total_attempts + 1):
        attempt_root = cycle_root / f"attempt_{attempt:02d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        print(f"-> grasp attempt {attempt}/{total_attempts}: capture and inference")
        attempt_started = time.monotonic()
        d405_run, femto_run = capture_preprocess_both(args)
        # The persistent service batches the two camera requests and therefore
        # requires one shared point count.  Use the smaller retained cloud so
        # neither source is upsampled with replacement after table filtering.
        d405_points = int(np.load(d405_run / "d405" / "point_cloud.npy", mmap_mode="r").shape[0])
        femto_points = None if femto_run is None else int(np.load(femto_run / "femto_bolt" / "point_cloud.npy", mmap_mode="r").shape[0])
        inference_num_point = min(int(args.num_point), d405_points) if femto_points is None else min(int(args.num_point), d405_points, femto_points)
        if inference_num_point < 100:
            raise RuntimeError(
                "too few shared workpiece points for GraspNet after preprocessing: "
                f"D405={d405_points}, Femto={femto_points}, shared={inference_num_point}"
            )
        print("-> {} GraspNet batch size: {} points (D405 retained={}, Femto retained={}, configured cap={})".format(
            "D405-only" if femto_points is None else "shared", inference_num_point,
            d405_points, femto_points, int(args.num_point)))
        d405_inference_cloud = d405_run / "d405" / "point_cloud.npy"
        if args.enable_workpiece2_cloud_thickening:
            d405_inference_cloud, thickening_record = build_thickened_inference_cloud(
                d405_inference_cloud, args
            )
            print(
                "-> workpiece-2 D405 cloud thickening: raw={} + virtual={} = {} points; "
                "offsets=+/-{} mm; plane-to-table>={:.1f}deg ({:.3f}s)".format(
                    thickening_record["raw_points"],
                    thickening_record["virtual_points"],
                    thickening_record["inference_points"],
                    ",".join(f"{value:g}" for value in thickening_record["offsets_mm_each_side"]),
                    thickening_record["minimum_plane_to_table_angle_deg"],
                    thickening_record["elapsed_s"],
                )
            )
        stage_started = time.monotonic()
        requests = [_service_request(
            args, d405_run, "d405", inference_num_point, point_cloud_path=d405_inference_cloud,
        )]
        if femto_run is not None:
            requests.append(_service_request(args, femto_run, "femto_bolt", inference_num_point))
        responses = infer_with_batch_fallback(args, requests)
        print_elapsed(f"attempt {attempt} GraspNet request", stage_started)
        stage_started = time.monotonic()
        d405_run = _finalize_inference(args, "d405", d405_run, responses[0])
        if femto_run is not None:
            femto_run = _finalize_inference(args, "femto_bolt", femto_run, responses[1])
        try:
            object_summary = version5.prepare_independent_object_references(
                args, d405_run, femto_run, attempt_root / "independent_object_references")
        except RuntimeError as exc:
            # Depth can briefly be too incomplete for DBSCAN to form a valid
            # non-flat reference cluster.  This is a vision no-result, not a
            # program fault and must follow the normal no-grasp retry policy.
            message = str(exc)
            cluster_failure = (
                "no non-flat workpiece cluster found" in message
                or "elevated points; cannot identify the workpiece cluster" in message
            )
            if not cluster_failure:
                raise
            retry_record = {
                "attempt": attempt,
                "max_no_grasp_retries": int(args.max_no_grasp_retries),
                "reason": "workpiece_reference_unavailable",
                "detail": message,
                "next_step": "immediate_retry" if attempt < total_attempts else "stop",
            }
            (attempt_root / "object_reference_failure.json").write_text(
                json.dumps(retry_record, indent=2), encoding="utf-8"
            )
            print(f"-> workpiece reference unavailable: {message}")
            print_elapsed(f"attempt {attempt} transform/object-reference/filter", stage_started)
            print_elapsed(f"attempt {attempt} total", attempt_started)
            if attempt < total_attempts:
                print("-> no usable workpiece reference; immediately retrying from the current observation pose")
                continue
            return None, None
        merged = attempt_root / "independent_selection" / "merged"
        merged.mkdir(parents=True, exist_ok=False)
        (merged / "meta.json").write_text(json.dumps({
            "coordinate_frame": "arm1_base", "selection_mode": "persistent_d405_only" if femto_run is None else "persistent_independent_d405_femto",
            "attempt": attempt, "object_summary_path": str(object_summary)}, indent=2), encoding="utf-8")
        grasps, sources = version5.load_all_grasp_candidates(d405_run, femto_run, merged.parent, None)
        # Permit a predicted opening all the way to the physical maximum.
        # build_ranked_planner_candidates below retains the normal +margin
        # policy for narrower grasps, but saturates wide ones at full opening.
        # Version5 owns the geometric/object-support filters; only its maximum
        # width check needs this Version7 execution-policy override.
        selection_args = copy.copy(args)
        selection_args.gripper_width_margin_m = 0.0
        selected, indices = version5.select_all_source_grasp(selection_args, merged.parent, grasps, sources)
        filter_path = merged.parent / "all_source_grasp_filter.json"
        if filter_path.exists():
            filter_record = json.loads(filter_path.read_text(encoding="utf-8"))
            filter_record["gripper_opening_policy"] = {
                "predicted_width_limit_m": float(args.gripper_open_width_m),
                "normal_margin_m": float(args.gripper_width_margin_m),
                "policy": (
                    "predicted_width <= open_width; predicted+margin below open_width uses margin; "
                    "predicted+margin above open_width uses fixed full opening"
                ),
            }
            filter_path.write_text(json.dumps(filter_record, indent=2), encoding="utf-8")
        indices = filter_grasp_edge_clearance(
            args, grasps, sources, indices, object_summary, merged.parent,
        )
        indices, steep_surface_record = rank_candidate_indices_by_steep_surface(
            args, d405_run / "d405", grasps, indices,
        )
        (merged.parent / "workpiece2_steep_surface_priority.json").write_text(
            json.dumps(steep_surface_record, indent=2), encoding="utf-8"
        )
        if steep_surface_record.get("reason") == "preferred_steep_surface_candidates_ranked_first":
            print(
                "[workpiece-2] steep-surface priority: preferred {}/{} candidates "
                "({} high-score eligible; plane-to-table>={:.1f}deg; contact distance<={:.1f}mm; "
                "score floor={:.3f})".format(
                    steep_surface_record["preferred_candidates"], len(indices),
                    steep_surface_record["priority_eligible_candidates"],
                    float(steep_surface_record["minimum_plane_to_table_angle_deg"]),
                    float(args.workpiece2_steep_surface_contact_distance_m) * 1000.0,
                    float(steep_surface_record["raw_score_priority_floor"]),
                )
            )
        elif bool(args.enable_workpiece2_cloud_thickening):
            print(
                "[workpiece-2] steep-surface priority fallback: {}; preserving raw GraspNet score order".format(
                    steep_surface_record.get("reason", "unavailable")
                )
            )
        if len(indices):
            selected_index = int(indices[0])
            selected = version5.grasp_result(merged.parent, grasps[selected_index])
            selected["source_grasp_index"] = selected_index
            selected["grasp_source"] = str(sources[selected_index])
            (merged / "selected_executable_grasp.json").write_text(
                json.dumps(selected, indent=2), encoding="utf-8"
            )
        else:
            selected = None
        print(
            "[all-sources] gripper opening policy: predicted width <= {:.1f}mm; "
            "wide candidates use fixed {:.1f}mm opening".format(
                float(args.gripper_open_width_m) * 1000.0,
                float(args.gripper_open_width_m) * 1000.0,
            )
        )
        print_elapsed(f"attempt {attempt} transform/object-reference/filter", stage_started)
        print_elapsed(f"attempt {attempt} total", attempt_started)
        if selected is not None:
            return selected, (d405_run, femto_run, merged, grasps, sources, indices)
        retry_record = {"attempt": attempt, "max_no_grasp_retries": int(args.max_no_grasp_retries),
                        "reason": "no_candidate_survived_all_source_filters", "next_step": "immediate_retry" if attempt < total_attempts else "stop"}
        (attempt_root / "no_valid_grasp.json").write_text(json.dumps(retry_record, indent=2), encoding="utf-8")
        if attempt < total_attempts:
            print("-> no valid predicted grasp; immediately retrying from the current observation pose")
    return None, None


def build_ranked_planner_candidates(
    args, grasps: np.ndarray, sources: np.ndarray, indices: np.ndarray, d405_run: Path,
) -> list[dict]:
    """Keep the nearest parallel-jaw-equivalent orientation for each ranked grasp."""
    # find_selected_grasp has already applied the W2 steep-surface preference
    # (or the exact legacy raw-score fallback) to this ordered index array.
    ranked_indices = np.asarray(indices, dtype=np.int64)
    current_eef_rotation = version5.capture_eef_rotation(
        d405_run, version5.load_wrist_calibration(Path(args.wrist_calibration))
    )
    candidates: list[dict] = []
    for rank, source_index in enumerate(ranked_indices[:int(args.planner_candidate_top_k)], start=1):
        grasp = version5.grasp_result(Path("."), grasps[int(source_index)])
        grasp["source_grasp_index"] = int(source_index)
        grasp["grasp_source"] = str(sources[int(source_index)])
        predicted_width = float(grasp["width_m"])
        physical_opening = float(args.gripper_open_width_m)
        nominal_opening = predicted_width + float(args.gripper_width_margin_m)
        if predicted_width <= 0.0 or predicted_width > physical_opening:
            raise RuntimeError(
                f"predicted width {predicted_width:.4f} m is outside physical gripper range "
                f"(0, {physical_opening:.4f}]"
            )
        plan_args = args
        opening_saturated = nominal_opening > physical_opening
        if opening_saturated:
            # Version4 rejects predicted+margin above the physical opening.
            # Build its TF/TCP plan with zero margin, then explicitly command
            # full opening below.  This leaves Version4/Version5 untouched.
            plan_args = copy.copy(args)
            plan_args.gripper_width_margin_m = 0.0
        plan = version5.build_execution_plan(plan_args, grasp, current_eef_rotation)
        if opening_saturated:
            plan.update({
                "opening_width_m": physical_opening,
                "opening_percent": 100.0,
                "width_margin_m": physical_opening - predicted_width,
                "configured_width_margin_m": float(args.gripper_width_margin_m),
                "opening_policy": "saturated_at_physical_maximum",
            })
            print(
                "-> candidate rank {} predicted width {:.1f}mm exceeds margin-safe limit {:.1f}mm; "
                "using fixed full opening {:.1f}mm".format(
                    rank, predicted_width * 1000.0,
                    (physical_opening - float(args.gripper_width_margin_m)) * 1000.0,
                    physical_opening * 1000.0,
                )
            )
        else:
            plan["opening_policy"] = "predicted_width_plus_configured_margin"
        probe_m = float(args.grasp_downward_probe_m)
        if probe_m != 0.0:
            # Keep the full GraspNet orientation and the active grasp-center
            # model. Only the final target is translated vertically in
            # arm1_base, so Planner checks the configured signed adjustment as
            # a real LIN motion in both legacy-TCP and direct-left_eef modes.
            # Positive values descend; negative values raise the final
            # closure point while leaving pregrasp unchanged.
            predicted_tcp = np.asarray(plan["grasp_center_base_m"], dtype=np.float64)
            probed_tcp = predicted_tcp - np.array([0.0, 0.0, probe_m], dtype=np.float64)
            predicted_eef = np.asarray(plan["grasp_eef_pose"], dtype=np.float64)
            probed_eef = predicted_eef.copy()
            probed_eef[:3] -= np.array([0.0, 0.0, probe_m], dtype=np.float64)
            plan["predicted_grasp_tcp_base_m"] = predicted_tcp.tolist()
            plan["grasp_tcp_base_m"] = probed_tcp.tolist()
            plan["predicted_grasp_eef_pose"] = predicted_eef.tolist()
            plan["grasp_eef_pose"] = probed_eef.tolist()
            plan["grasp_downward_probe_m"] = probe_m
        candidates.append({
            "rank": rank,
            "source_grasp_index": int(source_index),
            "grasp_source": str(sources[int(source_index)]),
            "score": float(grasp["score"]),
            "grasp": grasp,
            "parallel_jaw_symmetry_rotation_deg": float(plan["parallel_jaw_symmetry_rotation_deg"]),
            "plan": plan,
        })
    return candidates


def _grasp_rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the nearest SO(3) distance under 180-degree parallel-jaw symmetry."""
    symmetry = np.diag([1.0, -1.0, -1.0])

    def angle_degrees(relative: np.ndarray) -> float:
        cosine = float(np.clip((np.trace(relative) - 1.0) * .5, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    return min(
        angle_degrees(first.T @ second),
        angle_degrees(first.T @ (second @ symmetry)),
    )


def _grasp_signature(grasp: dict) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(grasp["translation_m"], dtype=np.float64)
    rotation = np.asarray(grasp["rotation_matrix"], dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (3, 3):
        raise RuntimeError("selected GraspNet candidate has an invalid translation or rotation")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        raise RuntimeError("selected GraspNet candidate contains non-finite pose values")
    return translation, rotation


def _matches_empty_failure_cluster(
    translation: np.ndarray, rotation: np.ndarray, cluster: dict, args,
) -> tuple[bool, float, float]:
    cluster_translation = np.asarray(cluster["translation_m"], dtype=np.float64)
    cluster_rotation = np.asarray(cluster["rotation_matrix"], dtype=np.float64)
    distance_m = float(np.linalg.norm(translation - cluster_translation))
    orientation_deg = _grasp_rotation_distance_deg(rotation, cluster_rotation)
    return (
        distance_m <= float(args.empty_failure_position_radius_m)
        and orientation_deg <= float(args.empty_failure_orientation_deg),
        distance_m,
        orientation_deg,
    )


def record_empty_failure(
    memory: list[dict], grasp: dict, args, *, weight_cycle: int,
) -> dict:
    """Record a confirmed EMPTY execution and blacklist its cluster at the configured count."""
    translation, rotation = _grasp_signature(grasp)
    matching: list[tuple[int, float, float]] = []
    for index, cluster in enumerate(memory):
        matches, distance_m, orientation_deg = _matches_empty_failure_cluster(
            translation, rotation, cluster, args,
        )
        if matches:
            matching.append((index, distance_m, orientation_deg))

    timestamp = datetime.now().isoformat(timespec="seconds")
    if matching:
        cluster_index, distance_m, orientation_deg = min(
            matching, key=lambda item: (item[1], item[2]),
        )
        cluster = memory[cluster_index]
        cluster["empty_failure_count"] = int(cluster["empty_failure_count"]) + 1
        cluster["last_empty_at"] = timestamp
        cluster["last_weight_cycle"] = int(weight_cycle)
        cluster["last_score"] = float(grasp["score"])
        cluster["last_source_grasp_index"] = int(grasp["source_grasp_index"])
        cluster["last_grasp_source"] = str(grasp["grasp_source"])
    else:
        cluster_index = len(memory)
        distance_m = 0.0
        orientation_deg = 0.0
        cluster = {
            "translation_m": translation.tolist(),
            "rotation_matrix": rotation.tolist(),
            "empty_failure_count": 1,
            "first_empty_at": timestamp,
            "last_empty_at": timestamp,
            "last_weight_cycle": int(weight_cycle),
            "last_score": float(grasp["score"]),
            "last_source_grasp_index": int(grasp["source_grasp_index"]),
            "last_grasp_source": str(grasp["grasp_source"]),
        }
        memory.append(cluster)

    cluster["blacklisted"] = bool(
        int(cluster["empty_failure_count"]) >= int(args.empty_failure_ban_count)
    )
    return {
        "cluster_index": int(cluster_index),
        "matched_existing_cluster": bool(matching),
        "match_translation_distance_m": float(distance_m),
        "match_orientation_deg": float(orientation_deg),
        "empty_failure_count": int(cluster["empty_failure_count"]),
        "blacklisted": bool(cluster["blacklisted"]),
    }


def write_empty_failure_memory(path: Path, memory: list[dict], args) -> None:
    payload = {
        "coordinate_frame": "arm1_base",
        "scope": "current Version7 process only; cleared at the next process start",
        "failure_class_recorded": "EMPTY only",
        "ban_count": int(args.empty_failure_ban_count),
        "position_radius_m": float(args.empty_failure_position_radius_m),
        "orientation_deg": float(args.empty_failure_orientation_deg),
        "clusters": memory,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def filter_empty_failure_blacklist(
    grasps: np.ndarray, indices: np.ndarray, memory: list[dict], args, output_dir: Path,
) -> np.ndarray:
    """Remove current candidates matching an EMPTY-failure cluster already blacklisted."""
    blacklisted = [
        (cluster_index, cluster)
        for cluster_index, cluster in enumerate(memory)
        if cluster.get("blacklisted")
    ]
    if not blacklisted:
        return indices
    kept: list[int] = []
    rejected: list[dict] = []
    for index in indices:
        grasp = np.asarray(grasps[int(index)], dtype=np.float64)
        translation = grasp[13:16]
        rotation = grasp[4:13].reshape(3, 3)
        matches: list[tuple[int, float, float]] = []
        for cluster_index, cluster in blacklisted:
            matched, distance_m, orientation_deg = _matches_empty_failure_cluster(
                translation, rotation, cluster, args,
            )
            if matched:
                matches.append((cluster_index, distance_m, orientation_deg))
        if matches:
            cluster_index, distance_m, orientation_deg = min(
                matches, key=lambda item: (item[1], item[2]),
            )
            rejected.append({
                "source_grasp_index": int(index),
                "blacklist_cluster_index": int(cluster_index),
                "translation_distance_m": float(distance_m),
                "orientation_distance_deg": float(orientation_deg),
            })
        else:
            kept.append(int(index))
    payload = {
        "input_candidates": int(len(indices)),
        "blacklisted_cluster_count": int(len(blacklisted)),
        "rejected_candidates": int(len(rejected)),
        "kept_candidates": int(len(kept)),
        "rejections": rejected,
    }
    (output_dir / "empty_failure_blacklist_filter.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    print(
        "[all-sources] EMPTY-failure blacklist: clusters={} rejected={} kept={}".format(
            len(blacklisted), len(rejected), len(kept),
        )
    )
    return np.asarray(kept, dtype=np.int64)


def weight_result_record(exc: planner_execute.RetryGrasp, retry_index: int) -> dict:
    result = exc.weight_result
    payload = {"weight_retry_index": retry_index, "classification": exc.grasp_class}
    if result is not None:
        payload.update({"mass_g": float(result.mass_g), "delta_force_n": [float(v) for v in result.delta_force_n],
                        "reason": str(result.reason),
                        "baseline": {"median_force_n": [float(v) for v in result.baseline.median_force_n],
                                     "axis_mad_n": [float(v) for v in result.baseline.axis_mad_n],
                                     "sample_count": int(result.baseline.sample_count)},
                        "loaded": {"median_force_n": [float(v) for v in result.loaded.median_force_n],
                                   "axis_mad_n": [float(v) for v in result.loaded.axis_mad_n],
                                   "sample_count": int(result.loaded.sample_count)}})
    return payload


def main() -> None:
    flow_started = time.monotonic()
    args = build_parser().parse_args()
    margin_safe_width_mm = (
        float(args.gripper_open_width_m) - float(args.gripper_width_margin_m)
    ) * 1000.0
    physical_width_mm = float(args.gripper_open_width_m) * 1000.0
    print(
        "-> Version7 runner: {}; executable width policy: selector accepts "
        "width <= {:.1f}mm (selection margin forced to 0); <= {:.1f}mm opens "
        "with configured margin, ({:.1f}, {:.1f}]mm opens at the physical maximum.".format(
            Path(__file__).resolve(),
            physical_width_mm,
            margin_safe_width_mm,
            margin_safe_width_mm,
            physical_width_mm,
        )
    )
    # Version7 uses the collector-compatible persistent gripper RPC client.
    # Always close its ZMQ sockets when this process returns or fails.
    atexit.register(planner_execute.close_collector_gripper_session, args)
    if args.roi is None or (not args.disable_femto and args.femto_roi is None):
        raise RuntimeError("Version7 requires --roi and, unless --disable-femto is set, --femto-roi")
    validate_workpiece2_cloud_thickening_args(args)
    if args.enable_fused_inference:
        no_fused_subprocess()
    if args.preview_pregrasp and args.execute_grasp:
        raise RuntimeError("--preview-pregrasp and --execute-grasp cannot be used together")
    if args.preview_pregrasp and not args.confirm_real_pregrasp:
        raise RuntimeError("--preview-pregrasp requires --confirm-real-pregrasp")
    if args.gripper_speed is not None and (not np.isfinite(float(args.gripper_speed)) or not 1.0 <= float(args.gripper_speed) <= 100.0):
        raise RuntimeError("--gripper-speed is a percentage and must be in [1, 100]; omit it for AG grippers")
    if not np.isfinite(float(args.camera_max_start_skew_s)) or float(args.camera_max_start_skew_s) < 0:
        raise RuntimeError("--camera-max-start-skew-s must be finite and non-negative")
    if int(args.max_no_grasp_retries) < 0:
        raise RuntimeError("--max-no-grasp-retries must be non-negative")
    if int(args.max_weight_retries) < 0:
        raise RuntimeError("--max-weight-retries must be non-negative")
    if int(args.max_operator_declines) < 0:
        raise RuntimeError("--max-operator-declines must be non-negative")
    if int(args.planner_candidate_top_k) < 1:
        raise RuntimeError("--planner-candidate-top-k must be at least 1")
    if int(args.max_planner_unreachable_retries) < 0:
        raise RuntimeError("--max-planner-unreachable-retries must be non-negative")
    if int(args.empty_failure_ban_count) < 1:
        raise RuntimeError("--empty-failure-ban-count must be at least 1")
    if (
        not np.isfinite(float(args.empty_failure_position_radius_m))
        or float(args.empty_failure_position_radius_m) <= 0.0
    ):
        raise RuntimeError("--empty-failure-position-radius-m must be finite and positive")
    if (
        not np.isfinite(float(args.empty_failure_orientation_deg))
        or not 0.0 < float(args.empty_failure_orientation_deg) <= 180.0
    ):
        raise RuntimeError("--empty-failure-orientation-deg must be in (0, 180]")
    if not np.isfinite(float(args.grasp_downward_probe_m)) or not -0.05 <= float(args.grasp_downward_probe_m) <= 0.05:
        raise RuntimeError("--grasp-downward-probe-m must be finite and in [-0.05, 0.05]; positive=down, negative=up")
    profiles = [("D405", args.d405_color_profile)]
    if not args.disable_femto:
        profiles.append(("Femto", args.femto_color_profile))
    for label, profile in profiles:
        if profile is not None and not Path(profile).is_file():
            raise RuntimeError(f"{label} color profile is missing: {profile}")
    args.gripper_yaw_offset_deg = 0.0
    version5.load_wrist_calibration(Path(args.wrist_calibration))
    if not args.disable_femto:
        version5.load_transform(Path(args.laser_calibration), "T_base_laser", require_accepted=True)
    if args.planner_direct_grasp_center:
        print(
            "-> Planner direct grasp center enabled: left_eef is the open-jaw "
            "GraspNet center; legacy TCP calibration is bypassed"
        )
    else:
        version5.load_tcp_translation(Path(args.tcp_translation_calibration))
    output_root = Path(args.output_dir) if args.output_dir else HERE / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root.mkdir(parents=True, exist_ok=False)
    empty_failure_memory: list[dict] = []
    empty_failure_memory_path = run_root / "empty_failure_blacklist.json"
    stage_started = time.monotonic()
    verify_persistent_services(args)
    print_elapsed("persistent service preflight", stage_started)
    if not move_to_observation_and_open(args): return
    retry_reason: str | None = None
    operator_declines = 0
    # A cycle is either an actual force-check retry or a manual rejection.
    # Keeping a separate decline count prevents a non-y response from spending
    # the finite EMPTY/MULTIPLE/UNKNOWN retry budget.
    max_weight_retries = int(args.max_weight_retries)
    max_operator_declines = int(args.max_operator_declines)
    for cycle_index in range(max_weight_retries + 1 + max_operator_declines):
        weight_retry = cycle_index - operator_declines
        cycle_root = run_root / (
            f"weight_cycle_{weight_retry:02d}_operator_decline_{operator_declines:02d}"
        )
        cycle_root.mkdir(parents=True, exist_ok=False)
        if retry_reason == "operator_declined":
            print(
                "-> fresh Version7 cycle after operator declined the previous grasp "
                f"{operator_declines}/{max_operator_declines}; "
                f"weight retries remain {max_weight_retries - weight_retry}/{max_weight_retries}"
            )
        elif weight_retry:
            print(f"-> fresh Version7 cycle after force rejection {weight_retry}/{max_weight_retries}")
        for reachability_retry in range(int(args.max_planner_unreachable_retries) + 1):
            reachability_root = cycle_root / f"reachability_cycle_{reachability_retry:02d}"
            reachability_root.mkdir(parents=True, exist_ok=False)
            selected, trial_data = find_selected_grasp(args, reachability_root)
            if selected is None or trial_data is None:
                print("-> no valid predicted grasp; no robot grasp command was sent")
                print_elapsed("TOTAL (all no-grasp retries exhausted)", flow_started)
                # This is a normal perception outcome, not a motion or
                # hardware failure.  The loop controller uses this distinct
                # code to keep the arm at graspup and begin a later cycle.
                return NO_GRASP_EXIT_CODE
            d405_run, femto_run, merged, grasps, sources, indices = trial_data
            indices = filter_empty_failure_blacklist(
                grasps, indices, empty_failure_memory, args, merged.parent,
            )
            if len(indices) == 0:
                print("-> every current candidate is in an EMPTY-failure blacklist region; re-running GraspNet")
                if reachability_retry >= int(args.max_planner_unreachable_retries):
                    print("-> EMPTY-failure blacklist left no candidates after all fresh inference retries")
                    print_elapsed("TOTAL (EMPTY-failure blacklist exhausted)", flow_started)
                    return NO_GRASP_EXIT_CODE
                continue
            planner_candidates = build_ranked_planner_candidates(
                args, grasps, sources, indices, d405_run
            )
            reachability_path = reachability_root / "planner_candidate_reachability.json"
            reachability_payload = {
                "top_k": int(args.planner_candidate_top_k),
                "candidate_count": len(planner_candidates),
                "parallel_jaw_policy": "nearest_equivalent_only",
                "records": [],
            }
            if (
                not args.planner_direct_grasp_center
                and not version5.grasp_tcp_is_certified(Path(args.tcp_translation_calibration))
            ):
                print("-> selected grasp is visualization-only: TCP calibration is not certified"); return
            force_rejected = False
            prediction_timed = False
            for execution_attempt, selected_plan_item in enumerate(planner_candidates, start=1):
                # Lazy reachability policy: only ask the Planner about this
                # item after every higher-priority candidate has failed.  The
                # list order is rank1-nearest, rank2-nearest, rank3-nearest.
                reachable_items, preflight_records = planner_execute.preflight_candidate_plans(
                    args, [selected_plan_item]
                )
                reachability_payload["records"].extend(preflight_records)
                reachability_path.write_text(json.dumps(reachability_payload, indent=2), encoding="utf-8")
                if not reachable_items:
                    continue
                if not prediction_timed:
                    print_elapsed("TOTAL (reachable prediction selected; timing stops before visualization)", flow_started)
                    prediction_timed = True
                selected_index = int(selected_plan_item["source_grasp_index"])
                print(
                    "-> executing lazily-checked candidate {}/{}: rank={}, symmetry={:.0f}deg".format(
                        execution_attempt, len(planner_candidates),
                        int(selected_plan_item["rank"]),
                        float(selected_plan_item["parallel_jaw_symmetry_rotation_deg"]),
                    )
                )
                version5.show_fused_visualization(
                    args,
                    merged.parent,
                    grasps,
                    sources,
                    indices,
                    selected_index,
                    execution_plan=selected_plan_item["plan"],
                )
                plan = dict(selected_plan_item["plan"])
                plan.update({"source": "persistent_d405_only" if args.disable_femto else "persistent_independent_d405_femto", "selected_grasp_source": selected_plan_item["grasp_source"],
                             "source_grasp_index": selected_index, "candidate_rank": int(selected_plan_item["rank"]),
                             "weight_cycle": weight_retry, "reachability_cycle": reachability_retry,
                             "d405_inference_run": str(d405_run), "femto_inference_run": None if femto_run is None else str(femto_run),
                             "fused_inference_run": None})
                actual_selected = dict(selected_plan_item["grasp"])
                actual_selected.update({
                    "run_dir": str(merged.parent),
                    "source_grasp_index": selected_index,
                    "grasp_source": selected_plan_item["grasp_source"],
                    "candidate_rank": int(selected_plan_item["rank"]),
                    "parallel_jaw_symmetry_rotation_deg": float(selected_plan_item["parallel_jaw_symmetry_rotation_deg"]),
                    "execution_plan": None,
                })
                plan_path = reachability_root / (
                    f"grasp_execution_plan_rank_{int(selected_plan_item['rank']):02d}_"
                    f"sym_{int(selected_plan_item['parallel_jaw_symmetry_rotation_deg']):03d}.json"
                )
                plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
                actual_selected["execution_plan"] = str(plan_path)
                (merged / "selected_executable_grasp.json").write_text(
                    json.dumps(actual_selected, indent=2), encoding="utf-8"
                )
                print("[persistent] grasp execution plan:", plan_path)
                try:
                    if args.preview_pregrasp:
                        planner_execute.preview_pregrasp_with_planner(args, plan)
                    else:
                        graspup_complete = planner_execute.execute_grasp_with_planner(args, plan)
                        if not graspup_complete:
                            raise RuntimeError(
                                "Planner grasp did not complete a verified stop-after-graspup handoff"
                            )
                        if args.handoff_success_file:
                            handoff_path = Path(args.handoff_success_file)
                            handoff_path.parent.mkdir(parents=True, exist_ok=True)
                            handoff_path.write_text(json.dumps({
                                "status": "verified_graspup",
                                "candidate_rank": int(selected_plan_item["rank"]),
                                "grasp_source": str(selected_plan_item["grasp_source"]),
                                "weight_check_enabled": bool(args.grasp_weight_check),
                                "weight_check": getattr(
                                    args, "verified_weight_handoff", None
                                ),
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                            }, indent=2), encoding="utf-8")
                            print(f"-> verified graspup handoff marker written: {handoff_path}")
                    return
                except planner_execute.CandidatePlanRejected as exc:
                    reachability_payload["records"].append({
                        "rank": int(selected_plan_item["rank"]),
                        "source_grasp_index": selected_index,
                        "grasp_source": str(selected_plan_item["grasp_source"]),
                        "score": float(selected_plan_item["score"]),
                        "parallel_jaw_symmetry_rotation_deg": float(selected_plan_item["parallel_jaw_symmetry_rotation_deg"]),
                        "reachable": False,
                        "phase": exc.phase,
                        "reason": str(exc.cause),
                    })
                    reachability_path.write_text(json.dumps(reachability_payload, indent=2), encoding="utf-8")
                    print(f"-> candidate rank {selected_plan_item['rank']} rejected during {exc.phase}; returning safely to observation for the next candidate")
                    if not move_to_observation_and_open(args):
                        return
                    continue
                except planner_execute.RetryGrasp as exc:
                    record = weight_result_record(exc, weight_retry)
                    if exc.grasp_class == "EMPTY":
                        empty_event = record_empty_failure(
                            empty_failure_memory, selected_plan_item["grasp"], args,
                            weight_cycle=weight_retry,
                        )
                        record["empty_failure_memory"] = empty_event
                        write_empty_failure_memory(
                            empty_failure_memory_path, empty_failure_memory, args,
                        )
                        print(
                            "-> EMPTY failure memory: cluster={cluster_index} count={empty_failure_count}/{limit} "
                            "matched_existing={matched_existing_cluster} blacklisted={blacklisted}".format(
                                limit=int(args.empty_failure_ban_count), **empty_event,
                            )
                        )
                    if exc.grasp_class == "operator_declined":
                        operator_declines += 1
                        record["operator_decline_index"] = operator_declines
                        record["max_operator_declines"] = max_operator_declines
                        (cycle_root / "operator_declined.json").write_text(
                            json.dumps(record, indent=2), encoding="utf-8"
                        )
                        if operator_declines >= max_operator_declines:
                            print("-> operator declined grasp; separate operator-decline limit exhausted, stopping safely")
                            return
                        print("-> operator declined grasp; returning to go_graspnet for fresh inference")
                    else:
                        (cycle_root / "weight_check_result.json").write_text(
                            json.dumps(record, indent=2), encoding="utf-8"
                        )
                        if weight_retry >= max_weight_retries:
                            print(f"-> force check rejected {exc.grasp_class}; max weight retries exhausted, stopping safely")
                            return
                        print(f"-> force check rejected {exc.grasp_class}; returning to go_graspnet for fresh inference")
                    if not move_to_observation_and_open(args):
                        return
                    retry_reason = exc.grasp_class
                    force_rejected = True
                    break
                except planner_execute.ForceSensorDataError as exc:
                    failure_record = {
                        "phase": exc.phase,
                        "reason": str(exc.cause),
                        "safety_action": (
                            "return_to_observation_with_gripper_open"
                            if exc.phase in {"stream", "baseline"}
                            else "hold_current_pose_with_gripper_closed"
                        ),
                    }
                    (reachability_root / "force_sensor_failure.json").write_text(
                        json.dumps(failure_record, indent=2), encoding="utf-8"
                    )
                    print(f"-> force sensor {exc.phase} data failure: {exc.cause}")
                    if exc.phase in {"stream", "baseline"}:
                        print("-> grasp descent/closure was not completed; returning safely to observation with gripper open")
                        move_to_observation_and_open(args)
                    else:
                        print("-> force data failed after closure; grasp state is UNKNOWN. Holding current pose with gripper closed; Module 2 handoff is prohibited")
                    return
            if force_rejected:
                # An operator decline or force-sensor rejection has already
                # moved back to observation; continue the next independent
                # cycle. Only force rejections advance ``weight_retry``.
                break
            if reachability_retry >= int(args.max_planner_unreachable_retries):
                print("-> all top-K candidates failed before gripper close; planner-unreachable retries exhausted, stopping safely")
                print_elapsed("TOTAL (planner-unreachable retries exhausted)", flow_started)
                return
            print("-> all top-K candidates failed before gripper close; immediately recapturing and re-running GraspNet")
            continue
        else:
            print("-> every top-K candidate failed before gripper close; planner-unreachable retries exhausted, stopping safely")
            print_elapsed("TOTAL (planner-unreachable retries exhausted)", flow_started)
            return


if __name__ == "__main__":
    raise SystemExit(main() or 0)
