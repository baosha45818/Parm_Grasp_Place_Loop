#!/usr/bin/env python3
"""Offline traditional grasp-pose selection from a saved D405 capture.

This program intentionally has no GraspNet import or inference call.  It uses
SAM instance masks, depth back-projection and STL ICP, then produces one
face-specific fixed grasp template per accepted visible instance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import http.client
import io
import itertools
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time
import signal
from urllib.parse import urlencode

# One native thread per process avoids N worker processes each attempting to
# occupy every CPU through OpenMP/BLAS.  An explicit user environment value
# remains authoritative.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


LOOP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STL = LOOP_ROOT / "tool/workpiece_1/workpiece_1.stl"
WRIST_CALIBRATION = LOOP_ROOT / "tool/calibration/left_wrist_d405/calibration_wrist1.json"
TCP_CALIBRATION = LOOP_ROOT / "tool/calibration/graspnet_tcp/calibration_translation2.json"
LEFT_EEF_MIN_GRASP_Z_BASE_M = -0.541
ICP_DAEMON_SOCKET = Path(os.environ.get("TRADITIONAL_ICP_DAEMON_SOCKET", "/tmp/traditional-stl-icp.sock"))
ICP_DAEMON_AUTHKEY = b"traditional-stl-icp-v1"
ICP_DAEMON_GENERATION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# These are installed once in each worker, so the sampled STL is not sent
# with every SAM instance.  Keep the worker entry point at module scope: it
# must be pickleable when Python uses the ``spawn`` start method.
_WORKER_CAD: np.ndarray | None = None
_WORKER_K: np.ndarray | None = None
_PERSISTENT_ICP_EXECUTOR: ProcessPoolExecutor | None = None
_PERSISTENT_ICP_SIGNATURE: tuple[int, bytes] | None = None


def _init_icp_worker(cad: np.ndarray, K: np.ndarray) -> None:
    global _WORKER_CAD, _WORKER_K
    _WORKER_CAD = cad
    _WORKER_K = K


def _icp_pool_signature(cad: np.ndarray, K: np.ndarray, workers: int) -> tuple[int, bytes]:
    digest = hashlib.sha256()
    for array in (cad, K):
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode())
        digest.update(contiguous.tobytes())
    return workers, digest.digest()


def persistent_icp_executor(cad: np.ndarray, K: np.ndarray, workers: int) -> tuple[ProcessPoolExecutor, bool]:
    """Return workers reused by successive live capture/retry cycles."""
    global _PERSISTENT_ICP_EXECUTOR, _PERSISTENT_ICP_SIGNATURE
    signature = _icp_pool_signature(cad, K, workers)
    if _PERSISTENT_ICP_EXECUTOR is not None and _PERSISTENT_ICP_SIGNATURE == signature:
        return _PERSISTENT_ICP_EXECUTOR, True
    shutdown_persistent_icp_workers()
    _PERSISTENT_ICP_EXECUTOR = ProcessPoolExecutor(
        max_workers=workers, initializer=_init_icp_worker, initargs=(cad, K),
        mp_context=mp.get_context("spawn"),
    )
    _PERSISTENT_ICP_SIGNATURE = signature
    print(f"-> started persistent STL-ICP worker pool: {workers} process(es)", flush=True)
    return _PERSISTENT_ICP_EXECUTOR, False


def shutdown_persistent_icp_workers() -> None:
    """Release reusable ICP workers when the live picker terminates."""
    global _PERSISTENT_ICP_EXECUTOR, _PERSISTENT_ICP_SIGNATURE
    if _PERSISTENT_ICP_EXECUTOR is not None:
        _PERSISTENT_ICP_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        _PERSISTENT_ICP_EXECUTOR = None
        _PERSISTENT_ICP_SIGNATURE = None


def _daemon_client(socket_path: Path):
    return Client(str(socket_path), family="AF_UNIX", authkey=ICP_DAEMON_AUTHKEY)


def _daemon_request(socket_path: Path, request: dict, timeout_s: float) -> dict:
    connection = _daemon_client(socket_path)
    try:
        connection.send(request)
        if not connection.poll(timeout_s):
            raise TimeoutError(f"ICP daemon request exceeded {timeout_s:.1f}s")
        return connection.recv()
    finally:
        connection.close()


def _daemon_pid_path(socket_path: Path) -> Path:
    return socket_path.with_name(socket_path.name + ".pid")


def _terminate_icp_daemon(socket_path: Path) -> None:
    """Ask the known local daemon to exit, then force-stop only its own group."""
    try:
        _daemon_request(socket_path, {"action": "shutdown"}, timeout_s=1.0)
    except (ConnectionError, EOFError, OSError, TimeoutError):
        pass
    deadline = time.monotonic() + 3.0
    while socket_path.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    if not socket_path.exists():
        return
    pid_path = _daemon_pid_path(socket_path)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        expected = str(Path(__file__).resolve()).encode()
        if expected not in command or b"--serve-icp-worker-pool" not in command:
            raise RuntimeError("daemon PID does not identify this ICP service")
        os.killpg(pid, signal.SIGTERM)
    except FileNotFoundError:
        pass
    finally:
        deadline = time.monotonic() + 3.0
        while socket_path.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        if socket_path.exists():
            socket_path.unlink()


def _start_icp_daemon(socket_path: Path) -> None:
    """Start the local worker service only when no healthy service exists."""
    try:
        health = _daemon_request(socket_path, {"action": "health"}, timeout_s=1.0)
        healthy = bool(health.get("ok")) and health.get("generation") == ICP_DAEMON_GENERATION
    except (ConnectionError, EOFError, OSError, TimeoutError):
        healthy = False
    if healthy:
        return
    if socket_path.exists():
        mode = socket_path.stat().st_mode
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"ICP daemon path exists but is not a socket: {socket_path}")
        _terminate_icp_daemon(socket_path)
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--serve-icp-worker-pool", "--socket", str(socket_path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None
    while True:
        try:
            health = _daemon_request(socket_path, {"action": "health"}, timeout_s=1.0)
            ready = bool(health.get("ok")) and health.get("generation") == ICP_DAEMON_GENERATION
        except (ConnectionError, EOFError, OSError, TimeoutError) as exc:
            ready = False
            last_error = exc
        if not ready:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"ICP daemon did not become ready at {socket_path}") from last_error
            time.sleep(.05)
            continue
        return


def daemon_icp_hypotheses(
    cad: np.ndarray, K: np.ndarray, tasks: list[tuple[int, np.ndarray, np.ndarray]], workers: int,
    socket_path: Path = ICP_DAEMON_SOCKET, timeout_s: float = 30.0,
) -> tuple[dict[int, list[tuple[np.ndarray, dict]]], dict[int, float], dict]:
    """Use the process-persistent local daemon for ICP across CLI launches."""
    started = time.perf_counter()
    request = {"action": "icp", "cad": cad, "K": K, "tasks": tasks, "workers": workers}
    for attempt in range(2):
        _start_icp_daemon(socket_path)
        try:
            response = _daemon_request(socket_path, request, timeout_s=timeout_s)
        except (ConnectionError, EOFError, OSError, TimeoutError):
            if attempt:
                raise
            _terminate_icp_daemon(socket_path)
            continue
        break
    if not response.get("ok"):
        raise RuntimeError("ICP daemon failed: " + str(response.get("error", "unknown error")))
    meta = dict(response["meta"])
    meta["round_trip_wall_s"] = time.perf_counter() - started
    return response["hypotheses"], response["worker_seconds"], meta


def serve_icp_worker_pool(socket_path: Path) -> None:
    """Run the single-owner local ICP worker daemon until externally stopped."""
    if socket_path.exists():
        raise RuntimeError(f"ICP daemon socket is already occupied: {socket_path}")
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=ICP_DAEMON_AUTHKEY)
    _daemon_pid_path(socket_path).write_text(str(os.getpid()), encoding="utf-8")
    running = True
    try:
        while running:
            connection = listener.accept()
            try:
                try:
                    request = connection.recv()
                except EOFError:
                    # Readiness probes connect and close without a request.
                    continue
                if request.get("action") == "health":
                    connection.send({"ok": True, "generation": ICP_DAEMON_GENERATION})
                    continue
                if request.get("action") == "shutdown":
                    connection.send({"ok": True})
                    running = False
                    continue
                if request.get("action") != "icp":
                    connection.send({"ok": False, "error": "unsupported daemon action"})
                    continue
                cad = np.asarray(request["cad"], dtype=float)
                K = np.asarray(request["K"], dtype=float)
                tasks = request["tasks"]
                workers = int(request["workers"])
                executor, reused = persistent_icp_executor(cad, K, workers)
                started = time.perf_counter()
                hypotheses, worker_seconds = parallel_icp_hypotheses(
                    cad, K, tasks, workers, executor=executor,
                )
                connection.send({
                    "ok": True, "hypotheses": hypotheses, "worker_seconds": worker_seconds,
                    "meta": {"pool_reused": reused, "service_icp_wall_s": time.perf_counter() - started},
                })
            except Exception as exc:
                try:
                    connection.send({"ok": False, "error": repr(exc)})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            finally:
                connection.close()
    finally:
        listener.close()
        shutdown_persistent_icp_workers()
        if socket_path.exists():
            socket_path.unlink()
        _daemon_pid_path(socket_path).unlink(missing_ok=True)


def _run_icp_worker(
    task: tuple[int, np.ndarray, np.ndarray],
) -> tuple[int, list[tuple[np.ndarray, dict]], float]:
    """Run all ICP branches for one independent SAM instance."""
    index, points, mask = task
    if _WORKER_CAD is None or _WORKER_K is None:
        raise RuntimeError("ICP worker was not initialized")
    started = time.perf_counter()
    hypotheses = icp_hypotheses(_WORKER_CAD, points, mask, _WORKER_K)
    return index, hypotheses, time.perf_counter() - started


def parallel_icp_hypotheses(
    cad: np.ndarray, K: np.ndarray, tasks: list[tuple[int, np.ndarray, np.ndarray]], workers: int,
    *, executor: ProcessPoolExecutor | None = None,
) -> tuple[dict[int, list[tuple[np.ndarray, dict]]], dict[int, float]]:
    """Evaluate independent SAM-instance ICP jobs in separate processes.

    Open3D releases the GIL inconsistently across builds, so processes give a
    reliable speed-up and isolate its native registration state.  A value of
    one deliberately retains a serial, debugging-friendly path.
    """
    if workers <= 1 or len(tasks) <= 1:
        results: dict[int, list[tuple[np.ndarray, dict]]] = {}
        worker_seconds: dict[int, float] = {}
        for index, points, mask in tasks:
            started = time.perf_counter()
            results[index] = icp_hypotheses(cad, points, mask, K)
            worker_seconds[index] = time.perf_counter() - started
        return results, worker_seconds
    if executor is not None:
        completed = list(executor.map(_run_icp_worker, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_icp_worker, initargs=(cad, K),
            mp_context=mp.get_context("spawn"),
        ) as temporary_executor:
            completed = list(temporary_executor.map(_run_icp_worker, tasks))
    return (
        {index: hypotheses for index, hypotheses, _ in completed},
        {index: elapsed_s for index, _, elapsed_s in completed},
    )


def sam_masks(
    socket_path: str, bgr: np.ndarray, depth: np.ndarray, K: np.ndarray, scale: float, min_area: int,
) -> np.ndarray:
    """Get RGB-D-ranked instance candidates from the resident SAM service."""
    class UnixConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(90.0)
            self.sock.connect(socket_path)

    request = io.BytesIO()
    np.savez_compressed(
        request, image_bgr=np.ascontiguousarray(bgr), depth_image=np.ascontiguousarray(depth),
        intrinsics=np.asarray(K, dtype=np.float64), depth_scale=float(scale),
    )
    # Keep this caller aligned with the current candicates-max endpoint.  In
    # particular, do not disable the endpoint's boundary-mask refinement or
    # impose the former 640x391 ROI: both would make the live result differ
    # from the validated max-service visualizations.
    options: dict[str, object] = {}
    params = urlencode({"num_candidates": 20, "options": json.dumps(options, separators=(",", ":"))})
    conn = UnixConnection("sam2-service", timeout=90.0)
    try:
        conn.request("POST", f"/v1/first-frame/candicates-max?{params}", body=request.getvalue(),
                     headers={"Content-Type": "application/x-sam2-rgbd-npz"})
        response = conn.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM first-frame candicates-max returned HTTP {response.status}: {payload.decode(errors='replace')}")
    finally:
        conn.close()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        masks = np.asarray(archive["candidate_masks"], dtype=bool)
        present = np.asarray(archive["candidate_present"], dtype=bool)
        masks = masks[present]
    # Keep the existing caller-side area and geometric gates; this endpoint's
    # RGB-D ranking, rather than raw AMG area ranking, now supplies the masks.
    print(f"-> SAM /v1/first-frame/candicates-max: {len(masks)} RGB-D-ranked masks")
    return masks


def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, scale: float) -> np.ndarray:
    valid = mask & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64) * scale
    return np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))


def local_depth_point(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, scale: float, pixel: list[int] | None) -> np.ndarray | None:
    """Robust 3-D point at a SAM-selected XY grasp centre, from D405 depth."""
    if pixel is None:
        return None
    u, v = pixel; h, w = depth.shape
    x0, x1 = max(0, u - 6), min(w, u + 7)
    y0, y1 = max(0, v - 6), min(h, v + 7)
    region = mask[y0:y1, x0:x1] & (depth[y0:y1, x0:x1] > 0)
    ys, xs = np.nonzero(region)
    if len(xs) < 8:
        return None
    # Median over nearby segmented pixels rejects holes and isolated depth noise.
    uu, vv = float(np.median(xs + x0)), float(np.median(ys + y0))
    z = float(np.median(depth[y0:y1, x0:x1][region])) * scale
    return np.array(((uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z))


def mesh_points(stl_path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise RuntimeError(f"cannot read STL: {stl_path}")
    # Stable replay is essential when the selected face determines a different
    # canonical grasp.  Open3D uniform sampling otherwise varies per process.
    o3d.utility.random.seed(0)
    cloud = mesh.sample_points_uniformly(number_of_points=12000)
    points = np.asarray(cloud.voxel_down_sample(.0015).points, dtype=np.float64)
    if len(points) < 100:
        raise RuntimeError("STL has too few sampled points")
    return points


def pca_initials(source: np.ndarray, target: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Match module_2_left_inspect_apart's proper all-axis PCA hypotheses."""
    def frame(p: np.ndarray) -> np.ndarray:
        centered = p - p.mean(0)
        covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
        values, r = np.linalg.eigh(covariance)
        r = r[:, np.argsort(values)[::-1]]
        if np.linalg.det(r) < 0: r[:, -1] *= -1
        return r
    rs, rt = frame(source), frame(target)
    initials: list[tuple[str, np.ndarray]] = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        S = np.diag(signs)
        if np.linalg.det(S) < 0:
            continue
        r = rt @ S @ rs.T
        t = target.mean(0) - r @ source.mean(0)
        T = np.eye(4); T[:3, :3] = r; T[:3, 3] = t
        initials.append((f"pca_flip_{int(signs[0]):+d}{int(signs[1]):+d}{int(signs[2]):+d}", T))
    T = np.eye(4); T[:3, 3] = target.mean(0) - source.mean(0)
    initials.append(("centroid_identity", T))
    return initials


def projected_mask_consistency(cad: np.ndarray, T: np.ndarray, mask: np.ndarray, K: np.ndarray, *, dilation_px: int = 3) -> dict:
    """The Module 2 projected-CAD / selected-SAM-mask consistency metrics."""
    h, w = mask.shape
    transformed = cad @ T[:3, :3].T + T[:3, 3]
    transformed = transformed[transformed[:, 2] > 1e-6]
    rendered = np.zeros((h, w), dtype=np.uint8)
    if len(transformed):
        pixels = (K @ transformed.T).T
        u = np.rint(pixels[:, 0] / pixels[:, 2]).astype(np.int32)
        v = np.rint(pixels[:, 1] / pixels[:, 2]).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        rendered[v[inside], u[inside]] = 1
    selected = np.asarray(mask, dtype=bool)
    if not np.any(rendered) or not np.any(selected):
        return {"projected_cad_pixels": int(rendered.sum()), "mask_near_cad_ratio": 0.0,
                "projected_iou": 0.0, "mask_to_cad_median_px": float("inf"),
                "projection_quality_passed": False}
    kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
    dilated = cv2.dilate(rendered, kernel) > 0
    distance = cv2.distanceTransform((~(rendered > 0)).astype(np.uint8), cv2.DIST_L2, 3)
    mask_distances = distance[selected]
    union = int(np.count_nonzero(dilated | selected))
    near = float(np.mean(mask_distances <= dilation_px))
    iou = float(np.count_nonzero(dilated & selected) / union) if union else 0.0
    median = float(np.median(mask_distances))
    return {"projected_cad_pixels": int(rendered.sum()), "mask_near_cad_ratio": near,
            "projected_iou": iou, "mask_to_cad_median_px": median,
            "projection_quality_passed": bool(near >= .70 and iou >= .50 and median <= 8.0)}


def icp_hypotheses(cad: np.ndarray, observed: np.ndarray, mask: np.ndarray, K: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    for cloud in (source, target):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.012, max_nn=40))
    results: list[tuple[np.ndarray, dict]] = []
    for method, estimator in (("point_to_plane", o3d.pipelines.registration.TransformationEstimationPointToPlane()),
                              ("point_to_point", o3d.pipelines.registration.TransformationEstimationPointToPoint())):
        for initial_name, initial in pca_initials(cad, observed):
            result = o3d.pipelines.registration.registration_icp(
                source, target, .008, initial, estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
            )
            metrics = {"method": method, "initialization": initial_name,
                       "fitness": float(result.fitness), "rmse_m": float(result.inlier_rmse)}
            metrics.update(projected_mask_consistency(cad, np.asarray(result.transformation), mask, K))
            metrics["quality_passed"] = bool(metrics["fitness"] >= .60 and metrics["rmse_m"] <= .004
                                               and metrics["projection_quality_passed"])
            results.append((np.asarray(result.transformation), metrics))
    return results


def icp(cad: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compatibility helper returning the raw highest-fitness ICP branch."""
    raise RuntimeError("icp() needs a SAM mask/intrinsics; use icp_hypotheses()")


def single_instance_completeness(cad: np.ndarray, observed: np.ndarray, T_camera_object: np.ndarray) -> dict:
    """Measure whether one transformed STL explains this entire SAM cloud.

    ICP fitness is source-to-target and can stay excellent when SAM merges a
    second stacked workpiece into the target cloud.  This complementary
    target-to-STL coverage and extent test detects that failure mode.
    """
    transformed = cad @ T_camera_object[:3, :3].T + T_camera_object[:3, 3]
    residual = cKDTree(transformed).query(observed, k=1)[0]
    expected_extent = transformed.ptp(axis=0)
    observed_extent = observed.ptp(axis=0)
    extent_ratio = observed_extent / np.maximum(expected_extent, 1e-6)
    return {
        "observed_points_explained_by_single_stl_ratio": float(np.mean(residual <= .006)),
        "observed_to_stl_extent_ratio_xyz": extent_ratio.tolist(),
        "maximum_observed_to_stl_extent_ratio": float(extent_ratio.max()),
        "observed_to_stl_median_residual_m": float(np.median(residual)),
    }


FRONT_AXIS_CENTER_LOCAL_M = np.array((0., 0., 0.))
FRONT_GRASP_X_RANGE_M = (.010, -.021)
# The operator-approved front search range includes the usable workpiece
# span, but its outermost 5 mm is an end feature rather than a robust pinch
# section.  Never let clearance scoring choose those extreme positions.
FRONT_GRASP_END_EXCLUSION_M = .005
# An 8-mm inboard section rejects an extreme tip while retaining the intended
# back-face tail boss (about 9.4 mm of CAD support on its short side).
MIN_GRASP_LONGITUDINAL_SUPPORT_M = .008
MAX_ICP_X_TO_SAM_AXIS_ANGLE_DEG = 30.0
MIN_SAM_MAIN_AXIS_ASPECT_RATIO = 1.35
SWEEP_INTRUSION_CLUSTER_RADIUS_M = .008
SWEEP_INTRUSION_MIN_CLUSTER_POINTS = 100
MIN_MASK_CLOSING_SECTION_WIDTH_M = .010
BACK_BOSS_CENTER_LOCAL_M = np.array((-.0395, 0., 0.))
BACK_HEAD_HIGH_CENTER_LOCAL_M = np.array((-.0100, 0., 0.))


def grasp_template(face: str, icp_fitness: float, *, front_axis_center_min_fitness: float) -> tuple[np.ndarray, str]:
    """Select the user-specified STL-local pinch centre.

    The front-face origin is the workpiece coordinate-axis centre.  Low ICP
    fitness is the available per-instance visibility signal: it is normally
    caused by the centre being hidden by another piled workpiece, so use the
    approved rear-area fallback instead of trying to close through that pile.
    """
    if face == "front":
        # Replaced after all instances are available by
        # choose_front_clearance_grasp().  Keep the axis centre here so an
        # isolated front workpiece remains sensible.
        return FRONT_AXIS_CENTER_LOCAL_M, "front_axis_center_pending_clearance"
    return BACK_BOSS_CENTER_LOCAL_M, "back_boss_center"


def choose_front_clearance_grasp(
    record: dict, other_points_camera: np.ndarray, mask: np.ndarray, K: np.ndarray, *, samples: int = 33,
    contact_half_span_m: float = .012, contact_T_camera_object: np.ndarray | None = None,
) -> tuple[np.ndarray, str, float]:
    """Choose the front local-X grasp point farthest from other instances.

    The interval is supplied by the operator: [0.010, 0, 0] to
    [-0.021, 0, 0], excluding the outermost 5 mm at either end.  Clearance
    is measured directly against the other SAM instance depth clouds,
    avoiding a potentially stale CAD-only occlusion guess.  With no other
    workpiece, choose the axis centre (x=0).
    """
    T = (np.asarray(record["T_camera_object"], dtype=float) if contact_T_camera_object is None
         else np.asarray(contact_T_camera_object, dtype=float))
    local_x = np.linspace(FRONT_GRASP_X_RANGE_M[0], FRONT_GRASP_X_RANGE_M[1], samples)
    local = np.column_stack((local_x, np.zeros(samples), np.zeros(samples)))
    candidates = local @ T[:3, :3].T + T[:3, 3]
    distances = (cKDTree(other_points_camera).query(candidates, k=1)[0]
                 if len(other_points_camera) else np.full(samples, np.inf))
    # Do not select a high-clearance point unless its two closing contacts are
    # visibly on the selected instance.  This is the same hard check later
    # applied before Planner, pushed earlier so it participates in selection.
    valid = np.array([
        parallel_jaw_bracket_from_mask(T, point, mask, K)[1]
        for point in local
    ], dtype=bool)
    x_low, x_high = sorted(FRONT_GRASP_X_RANGE_M)
    inboard = ((local_x >= x_low + FRONT_GRASP_END_EXCLUSION_M)
               & (local_x <= x_high - FRONT_GRASP_END_EXCLUSION_M))
    valid &= inboard
    if not np.any(valid):
        return FRONT_AXIS_CENTER_LOCAL_M.copy(), "front_segment_no_two_sided_mask_support", float("-inf")
    # A tolerance makes ties deterministic and retains the coordinate-axis
    # centre whenever it is equally clear.
    best = np.flatnonzero(valid & (distances >= distances[valid].max() - 1e-6))
    index = int(best[np.argmin(np.abs(local_x[best]))])
    source = ("front_axis_center_no_other_instance"
              if len(other_points_camera) == 0 else "front_segment_max_other_instance_clearance")
    return local[index], source, float(distances[index])


def choose_back_high_grasp(
    T_camera_object: np.ndarray, mask: np.ndarray, depth: np.ndarray, K: np.ndarray, scale: float,
) -> tuple[np.ndarray, str]:
    """Use the visible tail boss or head high pad, never the low back middle."""
    choices = (
        (BACK_BOSS_CENTER_LOCAL_M, "back_tail_boss_high"),
        (BACK_HEAD_HIGH_CENTER_LOCAL_M, "back_head_high"),
    )
    ranked = []
    T = np.asarray(T_camera_object, dtype=float)
    for local, source in choices:
        point = T[:3, :3] @ local + T[:3, 3]
        pixel = project_pixel(K, point)
        depth_point = local_depth_point(depth, mask, K, scale, pixel)
        if depth_point is not None:
            # Smaller camera Z is the exposed/high visible region.
            ranked.append((float(depth_point[2]), local.copy(), source))
    if ranked:
        _, local, source = min(ranked, key=lambda item: item[0])
        return local, source
    # Downstream SAM/depth gates will reject this if it has no real support.
    return BACK_BOSS_CENTER_LOCAL_M.copy(), "back_tail_boss_high_no_local_depth"


def jaw_side_support(cad: np.ndarray, local_center: np.ndarray, *, x_half_band_m: float = .004) -> tuple[float, float, int]:
    """Measure STL material extent on the two sides of local X (-Y/+Y)."""
    section = cad[np.abs(cad[:, 0] - local_center[0]) <= x_half_band_m]
    if len(section) == 0:
        return 0.0, 0.0, 0
    return (float(local_center[1] - section[:, 1].min()),
            float(section[:, 1].max() - local_center[1]), int(len(section)))


def jaw_longitudinal_support(cad: np.ndarray, local_center: np.ndarray, *, y_half_band_m: float = .004) -> tuple[float, float, int]:
    """Require a real inboard section on both local-X sides of a pinch."""
    section = cad[np.abs(cad[:, 1] - local_center[1]) <= y_half_band_m]
    if len(section) == 0:
        return 0.0, 0.0, 0
    return (float(local_center[0] - section[:, 0].min()),
            float(section[:, 0].max() - local_center[0]), int(len(section)))


def sam_main_axis_alignment(
    T_camera_object: np.ndarray, mask: np.ndarray, K: np.ndarray,
) -> dict:
    """Independently verify that ICP local X agrees with this SAM instance."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return {"valid": False, "reason": "too_few_mask_pixels"}
    image_points = np.column_stack((xs, ys)).astype(float)
    covariance = np.cov(image_points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    aspect = float(np.sqrt(values.max() / max(values.min(), 1e-9)))
    T = np.asarray(T_camera_object, dtype=float)
    endpoints = np.array(((-.025, 0., 0.), (.025, 0., 0.))) @ T[:3, :3].T + T[:3, 3]
    pixels = [project_pixel(K, point) for point in endpoints]
    if pixels[0] is None or pixels[1] is None:
        return {"valid": False, "reason": "icp_local_x_not_projectable", "sam_main_axis_aspect_ratio": aspect}
    direction = np.asarray(pixels[1], dtype=float) - np.asarray(pixels[0], dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 2.0:
        return {"valid": False, "reason": "icp_local_x_projection_too_short", "sam_main_axis_aspect_ratio": aspect}
    cosine = float(np.clip(abs(np.dot(direction / direction_norm, major)), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    valid = bool(aspect >= MIN_SAM_MAIN_AXIS_ASPECT_RATIO and angle <= MAX_ICP_X_TO_SAM_AXIS_ANGLE_DEG)
    return {
        "valid": valid, "sam_main_axis_aspect_ratio": aspect,
        "icp_x_to_sam_main_axis_angle_deg": angle,
        "maximum_angle_deg": MAX_ICP_X_TO_SAM_AXIS_ANGLE_DEG,
        "minimum_aspect_ratio": MIN_SAM_MAIN_AXIS_ASPECT_RATIO,
        "icp_local_x_projected_pixels": pixels,
    }


def project_pixel(K: np.ndarray, point_camera: np.ndarray) -> list[int] | None:
    if point_camera[2] <= 0.0 or not np.all(np.isfinite(point_camera)):
        return None
    homogeneous = K @ point_camera
    return [int(np.rint(homogeneous[0] / homogeneous[2])), int(np.rint(homogeneous[1] / homogeneous[2]))]


def local_x_axis_is_image_vertical(T_camera_object: np.ndarray, K: np.ndarray) -> bool:
    """Whether the visible workpiece main X axis is vertical in the D405 view."""
    T = np.asarray(T_camera_object, dtype=float)
    endpoints = np.array(((-.025, 0., 0.), (.025, 0., 0.))) @ T[:3, :3].T + T[:3, 3]
    first, second = (project_pixel(K, point) for point in endpoints)
    if first is None or second is None:
        return False
    dx, dy = abs(second[0] - first[0]), abs(second[1] - first[1])
    return bool(dy > dx)


def jaw_contacts_in_instance_mask(
    T_camera_object: np.ndarray, local_center: np.ndarray, mask: np.ndarray, K: np.ndarray,
    *, contact_half_span_m: float,
) -> tuple[dict, bool]:
    """Validate that both closing fingers contact opposite sides of this mask."""
    contact_local = np.vstack((
        local_center,
        local_center + np.array((0.0, -contact_half_span_m, 0.0)),
        local_center + np.array((0.0, contact_half_span_m, 0.0)),
    ))
    contact_camera = contact_local @ T_camera_object[:3, :3].T + T_camera_object[:3, 3]
    pixels = [project_pixel(K, point) for point in contact_camera]
    # Small image-space tolerance handles pixel quantization at a CAD edge,
    # while still rejecting a different stacked workpiece or empty tabletop.
    tolerant_mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    h, w = mask.shape
    contacts_on_mask = []
    for pixel in pixels[1:]:
        on_mask = pixel is not None and 0 <= pixel[0] < w and 0 <= pixel[1] < h and bool(tolerant_mask[pixel[1], pixel[0]])
        contacts_on_mask.append(on_mask)
    return {
        "grasp_center_pixel": pixels[0],
        "local_minus_y_contact_pixel": pixels[1],
        "local_plus_y_contact_pixel": pixels[2],
        "contact_half_span_m": float(contact_half_span_m),
        "both_contacts_on_selected_sam_mask": bool(all(contacts_on_mask)),
    }, bool(all(contacts_on_mask))


def parallel_jaw_bracket_from_mask(
    T_camera_object: np.ndarray, local_center: np.ndarray, mask: np.ndarray, K: np.ndarray,
    *, finger_margin_m: float = .004, extra_opening_m: float = .012, max_opening_m: float = .078,
) -> tuple[dict, bool]:
    """Predict fingers outside the two opposite SAM silhouette boundaries."""
    # SAM holes (bolt/shaft holes in this workpiece) are not free space for a
    # finger.  Fill them and use only the external silhouette; otherwise a
    # finger in a central hole would be incorrectly labelled "outside".
    silhouette = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(silhouette, contours, -1, 1, thickness=cv2.FILLED)
    silhouette = silhouette.astype(bool)
    # Search sufficiently beyond both sides of the main X axis.  The real
    # 78-mm gripper limit is enforced below.
    offsets = np.arange(-.090, .0901, .001)
    # The operator-defined main axis is local X.  Two fingers must be on its
    # two sides (-Y/+Y), making their connecting line perpendicular to X.
    points_local = local_center[None, :] + offsets[:, None] * np.array((0., 1., 0.))
    points_camera = points_local @ T_camera_object[:3, :3].T + T_camera_object[:3, 3]
    pixels = [project_pixel(K, point) for point in points_camera]
    h, w = mask.shape
    inside = np.array([
        pixel is not None and 0 <= pixel[0] < w and 0 <= pixel[1] < h and bool(silhouette[pixel[1], pixel[0]])
        for pixel in pixels
    ])
    # A valid parallel-jaw grasp crosses the *same connected silhouette
    # section* crossing the object-local X axis (local y=0).  Looking for any
    # negative and any positive sample independently is unsafe: it can bridge
    # a hole, a discontinuous/merged SAM mask, or a neighbouring workpiece and
    # put one finger inside the physical part.  Start at the local X axis and
    # walk outwards on both sides, stopping at the first boundary.
    center_i = int(np.argmin(np.abs(offsets)))
    center_on_silhouette = bool(inside[center_i])
    center_pixel = pixels[center_i]
    # Fill holes only for deciding where the *fingers* must stay outside the
    # external silhouette.  The closing centre itself must remain on original
    # SAM foreground with real object support; otherwise a hole or an
    # inter-object gap is incorrectly treated as a graspable surface.
    center_on_raw_mask = bool(
        center_pixel is not None and 0 <= center_pixel[0] < w and 0 <= center_pixel[1] < h
        and mask[center_pixel[1], center_pixel[0]]
    )
    if not center_on_silhouette or not center_on_raw_mask:
        return {
            "grasp_center_pixel": center_pixel,
            "grasp_center_on_selected_sam_silhouette": center_on_silhouette,
            "grasp_center_on_raw_selected_sam_mask": center_on_raw_mask,
            "fingers_on_opposite_sides_of_object_local_x_axis": False,
            "both_fingers_outside_selected_sam_mask": False,
            "validation_mask": "filled_external_sam_silhouette",
        }, False
    left_i = center_i
    while left_i - 1 >= 0 and inside[left_i - 1]:
        left_i -= 1
    right_i = center_i
    while right_i + 1 < len(inside) and inside[right_i + 1]:
        right_i += 1
    if left_i == center_i or right_i == center_i:
        return {
            "grasp_center_pixel": pixels[center_i],
            "grasp_center_on_selected_sam_silhouette": True,
            "fingers_on_opposite_sides_of_object_local_x_axis": False,
            "both_fingers_outside_selected_sam_mask": False,
            "validation_mask": "filled_external_sam_silhouette",
        }, False
    if left_i == 0 or right_i == len(inside) - 1:
        return {
            "grasp_center_pixel": pixels[center_i],
            "grasp_center_on_selected_sam_silhouette": True,
            "fingers_on_opposite_sides_of_object_local_x_axis": False,
            "opening_axis_outer_boundary_not_found_within_m": .09,
            "both_fingers_outside_selected_sam_mask": False,
            "validation_mask": "filled_external_sam_silhouette",
        }, False
    # Contact is at the outer silhouette edge; the open fingers stand just
    # beyond it, rather than at two points inside the material.
    left_finger = offsets[left_i] - finger_margin_m - extra_opening_m / 2.0
    right_finger = offsets[right_i] + finger_margin_m + extra_opening_m / 2.0
    finger_local = np.vstack((
        local_center + np.array((0., left_finger, 0.)),
        local_center + np.array((0., right_finger, 0.)),
    ))
    finger_camera = finger_local @ T_camera_object[:3, :3].T + T_camera_object[:3, 3]
    finger_pixels = [project_pixel(K, point) for point in finger_camera]
    # Merely landing one raster pixel outside can still visually/physically
    # put a finger on the selected contour.  Require a real image-space
    # clearance from the filled outer silhouette as well.
    outside_distance_px = cv2.distanceTransform((~silhouette).astype(np.uint8), cv2.DIST_L2, 3)
    finger_clearance_px = [
        (float(outside_distance_px[p[1], p[0]]) if p is not None and 0 <= p[0] < w and 0 <= p[1] < h else 0.0)
        for p in finger_pixels
    ]
    outside = [clearance >= 3.0 for clearance in finger_clearance_px]
    width = right_finger - left_finger
    # Offsets are signed around the selected grasp centre, proving one finger
    # on each side of the object's local X axis (-Y/+Y).
    opposite_x_axis_sides = bool(left_finger < 0.0 < right_finger)
    closing_section_width = float(offsets[right_i] - offsets[left_i])
    opening_within_limit = bool(0.010 <= width <= max_opening_m)
    closing_section_wide_enough = bool(closing_section_width >= MIN_MASK_CLOSING_SECTION_WIDTH_M)
    accepted = bool(all(outside) and opposite_x_axis_sides and opening_within_limit and closing_section_wide_enough)
    return {
        "grasp_center_pixel": pixels[center_i],
        "local_minus_y_contact_pixel": pixels[left_i], "local_plus_y_contact_pixel": pixels[right_i],
        "local_minus_y_finger_center_pixel": finger_pixels[0], "local_plus_y_finger_center_pixel": finger_pixels[1],
        "predicted_opening_width_m": float(width), "finger_margin_m": float(finger_margin_m),
        "extra_opening_m": float(extra_opening_m),
        "commanded_pregrasp_opening_width_m": float(width),
        "opening_within_gripper_limit": opening_within_limit,
        "max_gripper_opening_width_m": float(max_opening_m),
        "grasp_center_on_selected_sam_silhouette": True,
        "grasp_center_on_raw_selected_sam_mask": True,
        "fingers_on_opposite_sides_of_object_local_x_axis": opposite_x_axis_sides,
        "continuous_local_y_opening_section_m": [float(offsets[left_i]), float(offsets[right_i])],
        "continuous_local_y_opening_section_width_m": closing_section_width,
        "minimum_closing_section_width_m": MIN_MASK_CLOSING_SECTION_WIDTH_M,
        "closing_section_wide_enough": closing_section_wide_enough,
        "both_fingers_outside_selected_sam_mask": bool(all(outside)),
        "finger_clearance_from_selected_silhouette_px": finger_clearance_px,
        "validation_mask": "filled_external_sam_silhouette",
    }, accepted


def pointcloud_gripper_screen(
    T_camera_object: np.ndarray, face: str, local_center: np.ndarray,
    target_points_camera: np.ndarray, other_points_camera: np.ndarray, cad: np.ndarray,
    *, contact_half_span_m: float, min_clearance_m: float = .006,
    center_offset_camera_m: np.ndarray | None = None,
    pregrasp_vertical_sweep_camera_m: np.ndarray | None = None,
) -> tuple[dict, bool]:
    """Conservative depth-cloud check of the open parallel-jaw volume.

    The selected object's surface is allowed at the two planned closing
    contacts.  Points belonging to *other* SAM instances are not: they must
    stay outside the finger/palm swept box.  In addition, a selected-mask
    point that is far from every transformed STL point inside this box is
    treated as an occluder incorrectly merged into the selected object
    (the practical point-cloud signature of gripper/object penetration).
    """
    local_approach = np.array((0., 0., -1. if face == "front" else 1.))
    local_opening = np.array((0., 1., 0.))
    local_side = np.cross(local_approach, local_opening)
    R_camera_eef = T_camera_object[:3, :3] @ np.column_stack((local_approach, local_opening, local_side))
    center = T_camera_object[:3, :3] @ local_center + T_camera_object[:3, 3]
    if center_offset_camera_m is not None:
        center = center + np.asarray(center_offset_camera_m, dtype=float)

    def eef_coordinates(points: np.ndarray) -> np.ndarray:
        return (points - center) @ R_camera_eef

    # The box covers the two fingers, their closing sweep and the palm-facing
    # side of the grasp.  Dimensions are deliberately smaller than the full
    # gripper, so this rejects severe visual collisions without claiming to be
    # a robot-link collision model.
    def in_keepout(q: np.ndarray) -> np.ndarray:
        return ((q[:, 0] >= -.030) & (q[:, 0] <= .035)
                & (np.abs(q[:, 1]) <= contact_half_span_m + .026)
                & (np.abs(q[:, 2]) <= .024))

    obstacle_eef = eef_coordinates(other_points_camera) if len(other_points_camera) else np.empty((0, 3))
    # Check the actual final-to-pregrasp vertical sweep, not a 2-D overlap or
    # only one final pose.  A lower stacked workpiece is permitted whenever
    # it stays below/outside every sampled finger/palm volume.
    sweep_camera = (np.zeros(3) if pregrasp_vertical_sweep_camera_m is None
                    else np.asarray(pregrasp_vertical_sweep_camera_m, dtype=float))
    sweep_eef = sweep_camera @ R_camera_eef
    if len(obstacle_eef):
        sweep_inside = np.zeros(len(obstacle_eef), dtype=bool)
        for fraction in np.linspace(0.0, 1.0, 21):
            sweep_inside |= in_keepout(obstacle_eef - fraction * sweep_eef)
        obstacle_inside = obstacle_eef[sweep_inside]
    else:
        obstacle_inside = obstacle_eef
    # Distance to the planned pinch centre is a stable severity indicator;
    # any other-object point close to it means the jaws would close through
    # two objects rather than the selected instance.
    obstacle_min = (float(np.linalg.norm(obstacle_inside, axis=1).min())
                    if len(obstacle_inside) else float("inf"))

    target_local = (target_points_camera - T_camera_object[:3, 3]) @ T_camera_object[:3, :3]
    residual = cKDTree(cad).query(target_local, k=1)[0]
    target_eef = eef_coordinates(target_points_camera)
    merged_occluder = in_keepout(target_eef) & (residual > .009)
    report = {
        "model": "sam_depth_open_parallel_jaw_keepout",
        "min_other_instance_clearance_m": obstacle_min,
        "other_instance_points_in_gripper_keepout": int(len(obstacle_inside)),
        "vertical_sweep_height_m": float(np.linalg.norm(sweep_camera)),
        "selected_mask_unmodelled_points_in_keepout": int(np.count_nonzero(merged_occluder)),
        "minimum_clearance_m": float(min_clearance_m),
    }
    # Sparse D405 edge/segmentation points must not veto an otherwise valid
    # grasp.  Treat an intrusion as real only when it forms a compact 8-mm
    # point-cloud cluster; a true stacked workpiece produces hundreds of
    # neighbouring samples, while isolated boundary noise does not.
    if len(obstacle_inside):
        intrusion_tree = cKDTree(obstacle_inside)
        intrusion_cluster_sizes = np.asarray([
            len(neighbours) for neighbours in intrusion_tree.query_ball_point(
                obstacle_inside, r=SWEEP_INTRUSION_CLUSTER_RADIUS_M,
            )
        ], dtype=int)
        largest_cluster = int(intrusion_cluster_sizes.max())
    else:
        largest_cluster = 0
    report.update({
        "sweep_intrusion_cluster_radius_m": SWEEP_INTRUSION_CLUSTER_RADIUS_M,
        "largest_other_instance_intrusion_cluster_points": largest_cluster,
        "minimum_intrusion_cluster_points_to_reject": SWEEP_INTRUSION_MIN_CLUSTER_POINTS,
    })
    accepted = largest_cluster < SWEEP_INTRUSION_MIN_CLUSTER_POINTS
    return report, accepted


def template(
    T: np.ndarray, icp_fitness: float = float("inf"), *, front_axis_center_min_fitness: float = .80
) -> tuple[str, np.ndarray, str]:
    # Local +Z faces the camera when it points toward camera -Z.
    front = float(np.dot(T[:3, 2], np.array((0., 0., -1.)))) >= 0
    face = "front" if front else "back"
    local, source = grasp_template(face, icp_fitness, front_axis_center_min_fitness=front_axis_center_min_fitness)
    return face, T[:3, :3] @ local + T[:3, 3], source


def final_target_camera_transform(T_camera_object: np.ndarray, capture_meta: dict, upward_offset_m: float) -> np.ndarray:
    """Translate an object-local contact template to the configured final +base-Z target."""
    T_base_camera = (np.asarray(capture_meta["T_base_eef"], dtype=float)
                     @ np.asarray(json.loads(WRIST_CALIBRATION.read_text())["T_eef_wrist1"], dtype=float))
    shifted = np.asarray(T_camera_object, dtype=float).copy()
    shifted[:3, 3] += T_base_camera[:3, :3].T @ np.array((0., 0., upward_offset_m))
    return shifted


def base_execution_plan(
    selected: dict, capture_meta: dict, *, grasp_downward_probe_m: float = 0.0,
    min_grasp_angle_to_table_deg: float = 65.0, front_axis_center_min_fitness: float = .80,
) -> dict | None:
    """Use the deployed GraspNet wrist/TCP calibration convention, without inference."""
    if not capture_meta.get("T_base_eef"):
        return None
    wrist = json.loads(WRIST_CALIBRATION.read_text())
    T_base_eef_capture = np.asarray(capture_meta["T_base_eef"], dtype=float)
    T_eef_camera = np.asarray(wrist["T_eef_wrist1"], dtype=float)
    T_base_object = T_base_eef_capture @ T_eef_camera @ np.asarray(selected["T_camera_object"], dtype=float)
    front = selected["face_side"] == "front"
    if "grasp_template_local_m" in selected:
        local_center = np.asarray(selected["grasp_template_local_m"], dtype=float)
        grasp_template_source = str(selected.get("grasp_template_source", "recorded_template"))
    else:
        local_center, grasp_template_source = grasp_template(
            selected["face_side"], float(selected["icp"]["fitness"]),
            front_axis_center_min_fitness=front_axis_center_min_fitness,
        )
    local_approach = np.array((0., 0., -1. if front else 1.))
    local_opening = np.array((0., 1., 0.))
    local_side = np.cross(local_approach, local_opening)
    R_eef = T_base_object[:3, :3] @ np.column_stack((local_approach, local_opening, local_side))
    # The EEF +X column is the commanded pinch/approach axis.  The table is
    # the arm-base XY plane, so asin(|axis·Z|) is its acute angle to the table.
    grasp_angle_to_table_deg = float(np.degrees(np.arcsin(np.clip(abs(R_eef[2, 0]), 0., 1.))))
    center = T_base_object[:3, :3] @ local_center + T_base_object[:3, 3]
    # SAM selects XY; use the local D405 depth cloud, not mask geometry or
    # ICP alone, to correct the physical table-height coordinate.
    center[2] += float(selected.get("pointcloud_height_correction_base_m", 0.0))
    # Version7 convention: positive descends and negative raises the final
    # closure point.  The hover below is recomputed from that actual target.
    final_center = center - np.array((0., 0., grasp_downward_probe_m))
    requested_final_z = float(final_center[2])
    final_center[2] = max(final_center[2], LEFT_EEF_MIN_GRASP_Z_BASE_M)
    z_clamped = bool(final_center[2] != requested_final_z)
    # The hover is always measured from the actual, clamped closing target.
    pregrasp_tcp = final_center + np.array((0., 0., .10))
    rotvec = Rotation.from_matrix(R_eef).as_rotvec()
    return {
        "coordinate_frame": "arm1_base", "source": "traditional_sam_stl_icp",
        "selected_mask_index": int(selected["mask_index"]), "icp": selected["icp"],
        "face_side": selected["face_side"], "grasp_center_base_m": center.tolist(),
        "pointcloud_height_correction_base_m": float(selected.get("pointcloud_height_correction_base_m", 0.0)),
        "grasp_template_source": grasp_template_source,
        "grasp_template_local_m": local_center.tolist(),
        "front_axis_center_min_icp_fitness": float(front_axis_center_min_fitness),
        "predicted_grasp_tcp_base_m": center.tolist(),
        "grasp_tcp_base_m": final_center.tolist(),
        "requested_grasp_tcp_z_base_m": requested_final_z,
        "left_eef_min_grasp_z_base_m": LEFT_EEF_MIN_GRASP_Z_BASE_M,
        "left_eef_min_grasp_z_clamped": z_clamped,
        "grasp_downward_probe_m": float(grasp_downward_probe_m),
        "eef_rotation_base": R_eef.tolist(), "eef_axis_angle_rad": rotvec.tolist(),
        "grasp_angle_to_table_deg": grasp_angle_to_table_deg,
        "min_grasp_angle_to_table_deg": float(min_grasp_angle_to_table_deg),
        "pregrasp_grasp_tcp_base_m": pregrasp_tcp.tolist(),
        "vertical_pregrasp_height_m": .10,
        # Exact Module 1-1/V7 direct-center behavior: left_eef is the
        # open-jaw center, so the legacy closed-tip TCP pivot is deliberately
        # not applied to these Planner target poses.
        "planner_target_mode": "direct_left_eef_open_jaw_center",
        "planner_direct_grasp_center": True,
        "pregrasp_eef_pose": np.r_[pregrasp_tcp, rotvec].tolist(),
        "predicted_grasp_eef_pose": np.r_[center, rotvec].tolist(),
        "grasp_eef_pose": np.r_[final_center, rotvec].tolist(),
        "opening_percent": 100.0, "tcp_translation_calibration": None,
        "wrist_calibration": str(WRIST_CALIBRATION),
        "execution_authorization": "plan_only; requires separate Planner preflight and operator confirmation",
    }


def apply_auto_upward_clearance(plan: dict, extra_up_m: float) -> None:
    """Raise only the final target and keep the hover exactly 10 cm above it."""
    final = np.asarray(plan["grasp_tcp_base_m"], dtype=float) + np.array((0., 0., extra_up_m))
    hover = final + np.array((0., 0., .10))
    plan["grasp_tcp_base_m"] = final.tolist()
    plan["grasp_eef_pose"] = np.r_[final, plan["eef_axis_angle_rad"]].tolist()
    plan["pregrasp_grasp_tcp_base_m"] = hover.tolist()
    plan["pregrasp_eef_pose"] = np.r_[hover, plan["eef_axis_angle_rad"]].tolist()
    plan["auto_grasp_upward_clearance_m"] = float(extra_up_m)
    plan["effective_grasp_upward_offset_m"] = float(plan["grasp_upward_offset_m"] + extra_up_m)


def force_plan_grasp_angle(plan: dict, minimum_deg: float) -> dict:
    """Tilt only around the parallel-jaw opening axis to the requested angle."""
    adjusted = dict(plan)
    R = np.asarray(plan["eef_rotation_base"], dtype=float)
    xz, zz = float(R[2, 0]), float(R[2, 2])
    target = np.copysign(np.sin(np.deg2rad(minimum_deg)), xz if abs(xz) > 1e-8 else 1.0)
    phase = np.arctan2(xz, zz)
    ratio = np.clip(target / max(np.hypot(xz, zz), 1e-9), -1., 1.)
    theta = float(phase - np.arcsin(ratio))
    R_new = R @ Rotation.from_euler("y", theta).as_matrix()
    adjusted["eef_rotation_base"] = R_new.tolist()
    adjusted["eef_axis_angle_rad"] = Rotation.from_matrix(R_new).as_rotvec().tolist()
    for key in ("pregrasp_eef_pose", "grasp_eef_pose", "predicted_grasp_eef_pose"):
        if key in adjusted:
            adjusted[key] = [*adjusted[key][:3], *adjusted["eef_axis_angle_rad"]]
    adjusted["grasp_angle_to_table_deg"] = float(minimum_deg)
    adjusted["angle_fallback_to_minimum"] = True
    return adjusted


def run(argv: list[str] | None = None, *, keep_icp_workers_alive: bool = False) -> None:
    """Run one perception pass; live mode can reuse workers across retries."""
    total_started = time.perf_counter()
    timing_s: dict[str, object] = {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Version7 saved dual_* job directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL)
    parser.add_argument(
        "--icp-workers", type=int, default=min(4, os.cpu_count() or 1),
        help="processes used to register independent SAM instances; 1 disables parallelism (default: min(4, CPU count))",
    )
    parser.add_argument(
        "--icp-daemon-socket", type=Path, default=ICP_DAEMON_SOCKET,
        help="persistent local ICP-worker Unix socket (default: %(default)s)",
    )
    parser.add_argument("--no-icp-daemon", action="store_true",
                        help="use a one-shot local process pool instead of the persistent daemon")
    parser.add_argument("--icp-daemon-timeout-s", type=float, default=30.0,
                        help="restart and retry once if a daemon ICP request exceeds this time (default: 30)")
    parser.add_argument(
        "--sam-socket",
        default=str(LOOP_ROOT / "tool/sam2-service/runtime/sam2-service.sock"),
    )
    parser.add_argument("--min-mask-area-px", type=int, default=1200)
    parser.add_argument("--min-single-stl-observed-coverage", type=float, default=.65,
                        help="minimum fraction of SAM depth points explained by one transformed STL (default: 0.65)")
    parser.add_argument("--relaxed-single-stl-observed-coverage", type=float, default=.45,
                        help="only if no normal-completeness instance exists, retry at this coverage (default: 0.45)")
    parser.add_argument("--max-observed-to-stl-extent-ratio", type=float, default=1.30,
                        help="reject SAM masks wider/deeper than one STL by this ratio (default: 1.30)")
    parser.add_argument("--min-grasp-angle-to-table-deg", type=float, default=65.0,
                        help="require grasp approach axis angle to the table plane to exceed this value (default: 65)")
    parser.add_argument("--front-axis-center-min-icp-fitness", type=float, default=.80,
                        help="front axis-centre is treated as occluded below this ICP fitness and uses rear fallback (default: 0.80)")
    parser.add_argument("--min-jaw-side-support-m", type=float, default=.008,
                        help="require STL material at both local -Y/+Y sides of the main X axis (default: 0.008)")
    parser.add_argument("--jaw-contact-half-span-m", type=float, default=.012,
                        help="STL-local -Y/+Y contact samples used to prove fingers lie on both sides of X (default: 0.012)")
    parser.add_argument("--min-other-instance-gripper-clearance-m", type=float, default=.006,
                        help="minimum depth-cloud clearance from another SAM instance to the jaw keepout (default: 0.006)")
    parser.add_argument("--allow-other-instance-below-grasp-m", type=float, default=.003,
                        help="vertical depth/calibration guard: only other points higher than the grasp surface by this amount enter sweep checking (default: 0.003)")
    parser.add_argument("--max-auto-grasp-up-m", type=float, default=.03,
                        help="search this far upward in 5-mm steps to clear lower stacked points (default: 0.03; 0 disables)")
    parser.add_argument("--front-grasp-upward-offset-m", type=float, default=0.0,
                        help="final front-side closure offset in +Z meters (default: 0; closes at grasp surface)")
    parser.add_argument("--back-grasp-upward-offset-m", type=float, default=0.0,
                        help="final back-side closure offset in +Z meters (default: 0; closes at grasp surface)")
    parser.add_argument("--grasp-downward-probe-m", type=float, default=None,
                        help="optional V7 signed global override: positive=down, negative=up")
    args = parser.parse_args(argv)
    if args.icp_workers < 1:
        parser.error("--icp-workers must be positive")
    if not np.isfinite(args.icp_daemon_timeout_s) or args.icp_daemon_timeout_s <= 0.0:
        parser.error("--icp-daemon-timeout-s must be finite and positive")
    offsets = (args.front_grasp_upward_offset_m, args.back_grasp_upward_offset_m)
    if not all(np.isfinite(value) and -.05 <= value <= .05 for value in offsets):
        parser.error("face-specific upward offsets must be finite and in [-0.05, 0.05]")
    if args.grasp_downward_probe_m is not None and (
            not np.isfinite(args.grasp_downward_probe_m) or not -.05 <= args.grasp_downward_probe_m <= .05):
        parser.error("--grasp-downward-probe-m must be finite and in [-0.05, 0.05]")
    if not np.isfinite(args.min_grasp_angle_to_table_deg) or not 0.0 <= args.min_grasp_angle_to_table_deg < 90.0:
        parser.error("--min-grasp-angle-to-table-deg must be finite and in [0, 90)")
    if not np.isfinite(args.min_single_stl_observed_coverage) or not 0.0 <= args.min_single_stl_observed_coverage <= 1.0:
        parser.error("--min-single-stl-observed-coverage must be finite and in [0, 1]")
    if (not np.isfinite(args.relaxed_single_stl_observed_coverage)
            or not 0.0 <= args.relaxed_single_stl_observed_coverage <= args.min_single_stl_observed_coverage):
        parser.error("--relaxed-single-stl-observed-coverage must be finite and in [0, min coverage]")
    if (not np.isfinite(args.max_observed_to_stl_extent_ratio)
            or args.max_observed_to_stl_extent_ratio < 1.0):
        parser.error("--max-observed-to-stl-extent-ratio must be finite and >= 1")
    if not np.isfinite(args.front_axis_center_min_icp_fitness) or not 0.0 <= args.front_axis_center_min_icp_fitness <= 1.0:
        parser.error("--front-axis-center-min-icp-fitness must be finite and in [0, 1]")
    if not np.isfinite(args.min_jaw_side_support_m) or args.min_jaw_side_support_m <= 0.0:
        parser.error("--min-jaw-side-support-m must be finite and positive")
    if not np.isfinite(args.jaw_contact_half_span_m) or args.jaw_contact_half_span_m <= 0.0:
        parser.error("--jaw-contact-half-span-m must be finite and positive")
    if (not np.isfinite(args.min_other_instance_gripper_clearance_m)
            or args.min_other_instance_gripper_clearance_m < 0.0):
        parser.error("--min-other-instance-gripper-clearance-m must be finite and nonnegative")
    if (not np.isfinite(args.allow_other_instance_below_grasp_m)
            or args.allow_other_instance_below_grasp_m < 0.0):
        parser.error("--allow-other-instance-below-grasp-m must be finite and nonnegative")
    if not np.isfinite(args.max_auto_grasp_up_m) or not 0.0 <= args.max_auto_grasp_up_m <= .05:
        parser.error("--max-auto-grasp-up-m must be finite and in [0, 0.05]")
    raw = args.log.resolve() / "raw/d405"
    color_npy = raw / "color.npy"
    image = np.load(color_npy) if color_npy.is_file() else cv2.imread(str(raw / "color.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"missing readable D405 color image under {raw}")
    depth = np.load(raw / "depth.npy")
    K = np.load(raw / "depth_intrinsics.npy")
    meta = json.loads((raw / "meta.json").read_text())
    scale = float(meta["depth_scale_m"])
    sam_started = time.perf_counter()
    masks = sam_masks(args.sam_socket, image, depth, K, scale, args.min_mask_area_px)
    timing_s["sam_request_and_decode"] = time.perf_counter() - sam_started
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mesh_started = time.perf_counter()
    cad = mesh_points(args.stl.resolve())
    timing_s["stl_load_and_sample"] = time.perf_counter() - mesh_started
    overlay = image.copy()
    records = []
    saturation_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    def basic_mask_indices(relaxed: bool) -> list[int]:
        selected = []
        for index, mask in enumerate(masks):
            ys0, xs0 = np.nonzero(mask)
            if len(xs0) == 0:
                continue
            area0 = int(mask.sum()); x0, x1, y0, y1 = int(xs0.min()), int(xs0.max()), int(ys0.min()), int(ys0.max())
            area_ok = (1200 <= area0 <= 30000) if relaxed else (2000 <= area0 <= 24000)
            if (area_ok and x0 >= 0 and y0 >= 0 and x1 < image.shape[1] and y1 < image.shape[0]
                    and float(saturation_image[mask].mean()) <= 130.0):
                selected.append(index)
        return selected
    strict_indices = basic_mask_indices(False)
    relaxed_retry = not strict_indices
    process_indices = strict_indices if strict_indices else basic_mask_indices(True)

    # Only the ICP stage is parallel: it is independent for each mask.  SAM,
    # candidate ordering and all inter-instance collision checks deliberately
    # stay in this process.  Pre-filter here so workers receive no obvious
    # background/edge masks.
    icp_prepare_started = time.perf_counter()
    icp_tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for index in process_indices:
        mask = masks[index]
        points = backproject(depth, mask, K, scale)
        if len(points) < 400:
            continue
        bbox = points.max(axis=0) - points.min(axis=0)
        if float(bbox.max()) > (.25 if relaxed_retry else .13) or float(bbox.min()) < (.006 if relaxed_retry else .008):
            continue
        icp_tasks.append((index, points, mask))
    print(
        f"-> STL ICP: {len(icp_tasks)} SAM instances, "
        f"{min(args.icp_workers, len(icp_tasks)) if icp_tasks else 0} worker process(es)",
        flush=True,
    )
    timing_s["icp_task_preparation"] = time.perf_counter() - icp_prepare_started
    reusable_executor: ProcessPoolExecutor | None = None
    pool_reused = False
    daemon_meta: dict | None = None
    if not args.no_icp_daemon and args.icp_workers > 1 and len(icp_tasks) > 1:
        timing_s["icp_worker_pool"] = "persistent_local_daemon"
        icp_started = time.perf_counter()
        hypotheses_by_mask, icp_worker_seconds, daemon_meta = daemon_icp_hypotheses(
            cad, K, icp_tasks, args.icp_workers, args.icp_daemon_socket, args.icp_daemon_timeout_s,
        )
        pool_reused = bool(daemon_meta["pool_reused"])
    else:
        if keep_icp_workers_alive and args.icp_workers > 1 and len(icp_tasks) > 1:
            reusable_executor, pool_reused = persistent_icp_executor(cad, K, args.icp_workers)
        timing_s["icp_worker_pool"] = (
            "persistent_live_pool" if reusable_executor is not None else "temporary_process_pool_or_serial"
        )
        icp_started = time.perf_counter()
        hypotheses_by_mask, icp_worker_seconds = parallel_icp_hypotheses(
            cad, K, icp_tasks, args.icp_workers, executor=reusable_executor,
        )
    timing_s["icp_persistent_pool_reused"] = pool_reused
    timing_s["icp_parallel_wall"] = time.perf_counter() - icp_started
    if daemon_meta is not None:
        timing_s["icp_daemon_service_wall"] = daemon_meta["service_icp_wall_s"]
        timing_s["icp_daemon_round_trip_wall"] = daemon_meta["round_trip_wall_s"]
    timing_s["icp_worker_seconds_by_mask"] = {
        str(index): elapsed_s for index, elapsed_s in sorted(icp_worker_seconds.items())
    }
    timing_s["icp_worker_seconds_sum"] = sum(icp_worker_seconds.values())
    timing_s["icp_task_count"] = len(icp_tasks)
    timing_s["icp_worker_processes"] = min(args.icp_workers, len(icp_tasks)) if icp_tasks else 0
    if icp_worker_seconds:
        timing_s["icp_worker_seconds_min"] = min(icp_worker_seconds.values())
        timing_s["icp_worker_seconds_max"] = max(icp_worker_seconds.values())
        # Approximate process creation, module imports, data transfer and
        # scheduling cost.  It is especially useful when this exceeds the
        # slowest actual ICP task.
        timing_s["icp_process_startup_transfer_and_scheduling"] = max(
            0.0, float(timing_s["icp_parallel_wall"]) - max(icp_worker_seconds.values())
        )
    records_started = time.perf_counter()

    for index in process_indices:
        mask = masks[index]
        ys, xs = np.nonzero(mask)
        area = int(mask.sum())
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        saturation = float(saturation_image[mask].mean())
        # Borrow the bracket-like mask gates from pick/sam_segment.py.  They
        # prevent a tabletop/background mask or a cropped edge object from
        # winning ICP merely because it contains many points.
        points = backproject(depth, mask, K, scale)
        if len(points) < 400: continue
        bbox = points.max(axis=0) - points.min(axis=0)
        if float(bbox.max()) > (.25 if relaxed_retry else .13) or float(bbox.min()) < (.006 if relaxed_retry else .008):
            continue
        # ICP can have several visually similar branches.  Choose the branch
        # that best explains the *whole* SAM point cloud first.  Selecting an
        # ICP branch from a provisional jaw projection is circular: an
        # isolated correct workpiece can otherwise lose to a poor branch only
        # because the later, face-specific x search has not run yet.
        # This lookup is the result of the process-parallel registration
        # above.  The repeated cheap filters preserve the original record and
        # overlay behavior exactly.
        hypotheses = hypotheses_by_mask[index]
        scored_hypotheses = []
        for T_trial, quality_trial in hypotheses:
            trial_completeness = single_instance_completeness(cad, points, T_trial)
            extent_ok = trial_completeness["maximum_observed_to_stl_extent_ratio"] <= args.max_observed_to_stl_extent_ratio
            scored_hypotheses.append((
                quality_trial["quality_passed"], quality_trial["mask_near_cad_ratio"],
                quality_trial["projected_iou"], extent_ok,
                trial_completeness["observed_points_explained_by_single_stl_ratio"],
                quality_trial["fitness"], -quality_trial["rmse_m"], T_trial, quality_trial,
            ))
        _, _, _, _, _, _, _, T, quality = max(scored_hypotheses, key=lambda item: item[:7])
        upright_image_side_grasp = local_x_axis_is_image_vertical(T, K)
        face, center, template_source = template(
            T, quality["fitness"], front_axis_center_min_fitness=args.front_axis_center_min_icp_fitness
        )
        if upright_image_side_grasp and face == "front":
            # A workpiece whose main X axis is upright in the view is allowed
            # to be grasped directly from its visible side.  This exception
            # applies only to the front face: the back middle is low, so a
            # back-facing workpiece must still use its tail boss or head pad.
            center = T[:3, 3].copy()
            template_source = "upright_side_axis_center"
        # SAM owns grasp XY.  ICP may retain a good orientation while its
        # translation projects the template outside this instance (common in
        # piles).  Re-anchor that template centre to a robust SAM pixel with
        # real D405 depth, then re-run all bracket checks downstream.
        template_pixel = project_pixel(K, center)
        template_on_mask = (template_pixel is not None and 0 <= template_pixel[0] < mask.shape[1]
                            and 0 <= template_pixel[1] < mask.shape[0]
                            and bool(mask[template_pixel[1], template_pixel[0]]))
        sam_xy_recentered = False
        if not template_on_mask:
            anchor_pixel = [int(np.median(xs)), int(np.median(ys))]
            anchor_camera = local_depth_point(depth, mask, K, scale, anchor_pixel)
            if anchor_camera is not None:
                T = T.copy()
                T[:3, 3] += anchor_camera - center
                face, center, template_source = template(
                    T, quality["fitness"], front_axis_center_min_fitness=args.front_axis_center_min_icp_fitness
                )
                if upright_image_side_grasp and face == "front":
                    center = T[:3, 3].copy()
                    template_source = "upright_side_axis_center"
                sam_xy_recentered = True
        visible_depth = float(np.median(points[:, 2]))
        completeness = single_instance_completeness(cad, points, T)
        coverage_complete = (
            completeness["observed_points_explained_by_single_stl_ratio"] >= args.min_single_stl_observed_coverage
        )
        # An oversized cloud is direct evidence of a merged SAM instance and
        # is unsafe.  Low coverage alone can also mean a single workpiece is
        # partially hidden, so retain it but rank it behind complete masks.
        extent_consistent = (
            completeness["maximum_observed_to_stl_extent_ratio"] <= args.max_observed_to_stl_extent_ratio
        )
        accepted = bool(quality["quality_passed"] and extent_consistent)
        records.append({"mask_index": index, "mask_area_px": area,
                        "mask_center_px": [int(np.median(xs)), int(np.median(ys))],
                        "sam_filter_tier": "relaxed_retry" if relaxed_retry else "strict",
                        "mask_bbox_xyxy": [x0, y0, x1, y1], "mean_saturation": saturation,
                        "bbox_camera_m": bbox.tolist(),
                        "accepted": accepted, "face_side": face,
                        "grasp_template_source": template_source,
                        "grasp_template_local_m": (FRONT_AXIS_CENTER_LOCAL_M.tolist()
                                                   if face == "front" else BACK_BOSS_CENTER_LOCAL_M.tolist()),
                        "upright_image_side_grasp": upright_image_side_grasp,
                        "visible_depth_m": visible_depth, "icp": quality,
                        "single_instance_completeness": completeness,
                        "single_instance_completeness_accepted": extent_consistent,
                        "single_instance_coverage_complete": coverage_complete,
                        "template_center_initial_pixel": template_pixel,
                        "sam_xy_recentered_from_template_outside_mask": sam_xy_recentered,
                        "T_camera_object": T.tolist(), "template_grasp_center_camera_m": center.tolist(),
                        "reason": ("accepted" if accepted else
                                   ("sam_mask_exceeds_single_stl_extent"
                                    if not extent_consistent else "icp_quality"))})
        color = ((37 * index) % 255, (97 * index) % 255, (181 * index) % 255)
        overlay[mask] = (0.55 * overlay[mask] + .45 * np.asarray(color)).astype(np.uint8)
    timing_s["icp_result_scoring_and_record_build"] = time.perf_counter() - records_started
    # A small SAM fragment inside a pile can still pass ICP and even have a
    # locally continuous silhouette.  It is not an independently graspable
    # workpiece.  First require the normal single-STL coverage; only when
    # that leaves no candidate do a deliberately limited retry.  Never let a
    # very incomplete fragment win merely because it is the only ICP match.
    normal_coverage = [r for r in records if r["accepted"] and r["single_instance_coverage_complete"]]
    if normal_coverage:
        eligible_ids = {id(r) for r in normal_coverage}
        coverage_tier = "normal"
    else:
        relaxed_coverage = [
            r for r in records
            if r["accepted"] and r["single_instance_completeness"]["observed_points_explained_by_single_stl_ratio"]
            >= args.relaxed_single_stl_observed_coverage
        ]
        eligible_ids = {id(r) for r in relaxed_coverage}
        coverage_tier = "relaxed_retry"
    for record in records:
        if not record["accepted"]:
            continue
        record["single_instance_coverage_selection_tier"] = coverage_tier if id(record) in eligible_ids else "rejected"
        if id(record) not in eligible_ids:
            record["accepted"] = False
            record["reason"] = "insufficient_single_stl_coverage_for_independent_grasp"
    accepted = [r for r in records if r["accepted"]]
    # The dense ``candicates-max`` endpoint can return two prompts for the
    # same physical workpiece.  Those masks are nearly identical and must
    # never be treated as two stacked obstacles: otherwise the selected mask
    # collides with a duplicate copy of itself during the vertical sweep.
    mask_areas = np.asarray([int(mask.sum()) for mask in masks], dtype=int)
    duplicate_mask_pairs: list[tuple[int, int]] = []
    same_instance = np.eye(len(masks), dtype=bool)
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            overlap = int(np.count_nonzero(masks[first] & masks[second]))
            same = overlap / max(min(mask_areas[first], mask_areas[second]), 1) >= .85
            same_instance[first, second] = same_instance[second, first] = same
            if same:
                duplicate_mask_pairs.append((first, second))
    for record in records:
        record["sam_duplicate_instance_mask_indices"] = [
            index for index in range(len(masks))
            if index != int(record["mask_index"]) and same_instance[int(record["mask_index"]), index]
        ]
    # Every segmented mask is an obstacle, including masks rejected as a
    # grasp target for low completeness.  Otherwise a finger can be accepted
    # merely because the workpiece above/below it was not itself graspable.
    all_clouds_by_mask = {
        int(record["mask_index"]): backproject(depth, masks[int(record["mask_index"])], K, scale)
        for record in records
    }
    clouds_by_mask = {int(record["mask_index"]): all_clouds_by_mask[int(record["mask_index"])] for record in accepted}
    for record in accepted:
        if record["face_side"] == "back":
            local, source = choose_back_high_grasp(
                np.asarray(record["T_camera_object"], dtype=float), masks[int(record["mask_index"])],
                depth, K, scale,
            )
            T = np.asarray(record["T_camera_object"], dtype=float)
            record["grasp_template_local_m"] = local.tolist()
            record["grasp_template_source"] = source
            record["template_grasp_center_camera_m"] = (T[:3, :3] @ local + T[:3, 3]).tolist()
            continue
        if record["face_side"] != "front":
            continue
        target_index = int(record["mask_index"])
        other_clouds = [
            cloud for index, cloud in all_clouds_by_mask.items()
            if not same_instance[target_index, index]
        ]
        others = np.vstack(other_clouds) if other_clouds else np.empty((0, 3), dtype=float)
        upward_offset = (-args.grasp_downward_probe_m if args.grasp_downward_probe_m is not None
                         else args.front_grasp_upward_offset_m)
        final_T = final_target_camera_transform(
            np.asarray(record["T_camera_object"], dtype=float), meta, upward_offset
        )
        local, source, clearance = choose_front_clearance_grasp(
            record, others, masks[int(record["mask_index"])], K,
            contact_half_span_m=float(args.jaw_contact_half_span_m),
            contact_T_camera_object=final_T,
        )
        T = np.asarray(record["T_camera_object"], dtype=float)
        record["grasp_template_local_m"] = local.tolist()
        record["grasp_template_source"] = source
        record["front_clearance_to_other_instances_m"] = clearance
        record["front_grasp_end_exclusion_m"] = FRONT_GRASP_END_EXCLUSION_M
        record["template_grasp_center_camera_m"] = (T[:3, :3] @ local + T[:3, 3]).tolist()
    T_base_camera = (np.asarray(meta["T_base_eef"], dtype=float)
                     @ np.asarray(json.loads(WRIST_CALIBRATION.read_text())["T_eef_wrist1"], dtype=float))
    for record in accepted:
        T = np.asarray(record["T_camera_object"], dtype=float)
        local = np.asarray(record["grasp_template_local_m"], dtype=float)
        icp_center_camera = T[:3, :3] @ local + T[:3, 3]
        depth_center = local_depth_point(depth, masks[int(record["mask_index"])], K, scale,
                                         project_pixel(K, icp_center_camera))
        if depth_center is None:
            record["pointcloud_height_correction_base_m"] = 0.0
            record["pointcloud_height_source"] = "unavailable_keep_icp_height"
            continue
        depth_base = T_base_camera[:3, :3] @ depth_center + T_base_camera[:3, 3]
        icp_base = T_base_camera[:3, :3] @ icp_center_camera + T_base_camera[:3, 3]
        correction = float(depth_base[2] - icp_base[2])
        record["pointcloud_height_correction_base_m"] = correction
        record["pointcloud_height_source"] = "local_sam_depth_median"
        record["pointcloud_grasp_center_camera_m"] = depth_center.tolist()
    # Same priority as pick/plan_stl.py: choose the most credible STL-ICP pose;
    # optical depth only breaks a quality tie and is never treated as stack height.
    selected = max(accepted, key=lambda r: (r["icp"]["fitness"], -r["icp"]["rmse_m"], -r["visible_depth_m"])) if accepted else None
    if selected: selected["selected_for_planning"] = True
    for record in records:
        x, y = record["mask_center_px"]
        color = (0, 255, 0) if record is selected else (255, 255, 255)
        cv2.putText(overlay, f"#{record['mask_index']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        if record is selected:
            cv2.circle(overlay, (x, y), 14, (0, 255, 0), 2)
    cv2.imwrite(str(out / "sam_instances.png"), overlay)
    # In a pile the visually/ICP-best instance can still be occluded in a way
    # that makes the arm target invalid.  Preserve every credible semantic
    # template as an ordered Planner candidate; the live runner chooses only
    # the first candidate for which both the 5-cm hover and vertical descent
    # pass Planner preflight.  ICP is thus a ranking signal, never a reason to
    # attempt a known-unreachable grasp.
    ranked = sorted(accepted, key=lambda r: (
        # First prefer the SAM instance that is most completely explained by
        # one STL.  This avoids choosing a partial/occluded fragment merely
        # because it happens to be front-facing or has a cleaner local path.
        -int(bool(r["single_instance_coverage_complete"])),
        -r["single_instance_completeness"]["observed_points_explained_by_single_stl_ratio"],
        # With comparable segmentation completeness, retain the requested
        # front-face preference and leave a back face as fallback.
        0 if r["face_side"] == "front" else 1,
        r["single_instance_completeness"]["maximum_observed_to_stl_extent_ratio"],
        -r["icp"]["fitness"], r["icp"]["rmse_m"], r["visible_depth_m"]
    ))
    planner_candidates = []
    angle_fallbacks: list[tuple[dict, dict]] = []
    for record in ranked:
        if record.get("pointcloud_height_source") != "local_sam_depth_median":
            record["planner_rejection_reason"] = "no_local_sam_depth_at_predicted_grasp_xy"
            continue
        upward_offset = (
            args.front_grasp_upward_offset_m if record["face_side"] == "front"
            else args.back_grasp_upward_offset_m
        )
        # Preserve the deployed V7 spelling/semantics internally.  It is the
        # negative of the operator-facing +Z offset.  A specified global V7
        # probe remains an explicit override for compatibility.
        probe_m = (-upward_offset if args.grasp_downward_probe_m is None
                   else args.grasp_downward_probe_m)
        candidate_plan = base_execution_plan(
            record, meta, grasp_downward_probe_m=probe_m,
            min_grasp_angle_to_table_deg=args.min_grasp_angle_to_table_deg,
            front_axis_center_min_fitness=args.front_axis_center_min_icp_fitness,
        )
        if candidate_plan is None:
            continue
        candidate_plan["grasp_upward_offset_m"] = -probe_m
        left_support, right_support, support_samples = jaw_side_support(
            cad, np.asarray(record["grasp_template_local_m"], dtype=float)
        )
        candidate_plan["parallel_jaw_opening_axis_object_local"] = [0.0, 1.0, 0.0]
        candidate_plan["jaw_side_support_m"] = {
            "local_minus_y": left_support, "local_plus_y": right_support,
            "minimum_required": float(args.min_jaw_side_support_m), "section_samples": support_samples,
        }
        record["jaw_side_support_m"] = candidate_plan["jaw_side_support_m"]
        record["jaw_side_support_accepted"] = (
            left_support >= float(args.min_jaw_side_support_m)
            and right_support >= float(args.min_jaw_side_support_m)
        )
        if not record["jaw_side_support_accepted"]:
            record["planner_rejection_reason"] = "insufficient_stl_material_on_parallel_jaw_both_sides"
            continue
        longitudinal_minus_x, longitudinal_plus_x, longitudinal_samples = jaw_longitudinal_support(
            cad, np.asarray(record["grasp_template_local_m"], dtype=float)
        )
        candidate_plan["grasp_longitudinal_support_m"] = {
            "local_minus_x": longitudinal_minus_x, "local_plus_x": longitudinal_plus_x,
            "minimum_required": MIN_GRASP_LONGITUDINAL_SUPPORT_M, "section_samples": longitudinal_samples,
        }
        record["grasp_longitudinal_support_m"] = candidate_plan["grasp_longitudinal_support_m"]
        record["grasp_longitudinal_support_accepted"] = (
            longitudinal_minus_x >= MIN_GRASP_LONGITUDINAL_SUPPORT_M
            and longitudinal_plus_x >= MIN_GRASP_LONGITUDINAL_SUPPORT_M
        )
        if not record["grasp_longitudinal_support_accepted"]:
            record["planner_rejection_reason"] = "grasp_too_close_to_workpiece_local_x_end"
            continue
        axis_alignment = sam_main_axis_alignment(
            np.asarray(record["T_camera_object"], dtype=float), masks[int(record["mask_index"])], K,
        )
        candidate_plan["icp_x_to_sam_main_axis"] = axis_alignment
        record["icp_x_to_sam_main_axis"] = axis_alignment
        if not axis_alignment["valid"]:
            record["planner_rejection_reason"] = "icp_local_x_not_aligned_with_selected_sam_main_axis"
            continue
        # SAM owns the *surface* pinch XY.  A later +body_base-Z clearance
        # offset is a robot motion quantity, not an image-plane grasp shift:
        # under the oblique D405 view it visibly projects into an empty gap.
        # Keep silhouette validation and the review overlay on this original
        # surface transform; use the offset only in 3-D keepout/planning.
        surface_T = np.asarray(record["T_camera_object"], dtype=float)
        contact_projection, contacts_on_full_mask = parallel_jaw_bracket_from_mask(
            surface_T, np.asarray(record["grasp_template_local_m"], dtype=float),
            masks[int(record["mask_index"])], K,
        )
        candidate_plan["predicted_parallel_jaw_contacts_image"] = contact_projection
        record["predicted_parallel_jaw_contacts_image"] = contact_projection
        # Every face uses SAM first: the two predicted finger centres must be
        # outside opposite sides of the selected object's outer silhouette.
        # Back-side STL boss support remains an additional, not replacement,
        # constraint.
        contacts_accepted = contacts_on_full_mask
        candidate_plan["parallel_jaw_contact_validation"] = "filled_external_selected_sam_silhouette"
        record["jaw_contacts_on_selected_mask_accepted"] = contacts_accepted
        record["jaw_contacts_on_full_selected_sam_mask"] = contacts_on_full_mask
        if not contacts_accepted:
            if contact_projection.get("grasp_center_on_raw_selected_sam_mask") is False:
                record["planner_rejection_reason"] = "grasp_center_in_sam_hole_or_inter_object_gap"
            elif contact_projection.get("opening_axis_outer_boundary_not_found_within_m") is not None:
                record["planner_rejection_reason"] = "parallel_jaw_opening_axis_extent_exceeds_90mm_search"
            elif contact_projection.get("opening_within_gripper_limit") is False:
                record["planner_rejection_reason"] = "parallel_jaw_opening_exceeds_78mm_gripper_limit"
            elif contact_projection.get("closing_section_wide_enough") is False:
                record["planner_rejection_reason"] = "selected_sam_closing_section_too_narrow"
            else:
                record["planner_rejection_reason"] = "parallel_jaw_contacts_not_on_selected_sam_instance"
            continue
        opening_width = float(contact_projection["predicted_opening_width_m"])
        candidate_plan["opening_width_m"] = opening_width
        candidate_plan["opening_percent"] = 100.0 * opening_width / .078
        mask_index = int(record["mask_index"])
        other_clouds = [
            cloud for index, cloud in all_clouds_by_mask.items()
            if not same_instance[mask_index, index]
        ]
        other_points = (np.vstack(other_clouds) if other_clouds
                        else np.empty((0, 3), dtype=float))
        T_base_camera = (np.asarray(meta["T_base_eef"], dtype=float)
                         @ np.asarray(json.loads(WRIST_CALIBRATION.read_text())["T_eef_wrist1"], dtype=float))
        # Start from the operator-facing face offset already embedded in the
        # final plan, then try the smallest additional +body_base Z lift.
        nominal_center_base = np.asarray(candidate_plan["grasp_center_base_m"], dtype=float)
        final_center_base = np.asarray(candidate_plan["grasp_tcp_base_m"], dtype=float)
        # A lower stacked part can support the selected workpiece without
        # touching the fingers.  Keep only same/higher points (or points less
        # than the configured gap below the grasp surface) as sweep obstacles.
        if len(other_points):
            other_base_z = (other_points @ T_base_camera[:3, :3].T + T_base_camera[:3, 3])[:, 2]
            collision_other_points = other_points[
                # The vertical approach only needs to avoid material that is
                # definitively *above* the actual closing plane.  Lower
                # layers, and a small +/- depth/calibration uncertainty band,
                # are underneath an upper workpiece and must not veto its
                # top-down grasp.  Use the executed (possibly Z-clamped) TCP,
                # never the pre-clamp nominal prediction.
                other_base_z > final_center_base[2] + args.allow_other_instance_below_grasp_m
            ]
            allowed_below_count = int(len(other_points) - len(collision_other_points))
        else:
            collision_other_points = other_points
            allowed_below_count = 0
        base_offset = final_center_base - nominal_center_base
        upward_trials = np.arange(0.0, args.max_auto_grasp_up_m + 1e-9, .005)
        chosen = None
        trial_screen = None
        final_target_contacts_supported = False
        for extra_up in upward_trials:
            total_base_offset = base_offset + np.array((0., 0., extra_up))
            camera_offset = T_base_camera[:3, :3].T @ total_base_offset
            # The pixel-space finger placement is invariant under the
            # vertical motion offset by design (see surface_T above).
            trial_projection, placement_accepted = contact_projection, contacts_on_full_mask
            if not placement_accepted:
                continue
            final_target_contacts_supported = True
            trial_screen, trial_accepted = pointcloud_gripper_screen(
                np.asarray(record["T_camera_object"], dtype=float), record["face_side"],
                np.asarray(record["grasp_template_local_m"], dtype=float),
                clouds_by_mask[mask_index], collision_other_points, cad,
                contact_half_span_m=float(args.jaw_contact_half_span_m),
                min_clearance_m=float(args.min_other_instance_gripper_clearance_m),
                center_offset_camera_m=camera_offset,
                pregrasp_vertical_sweep_camera_m=(
                    T_base_camera[:3, :3].T @ np.array((0., 0., .10))
                ),
            )
            trial_screen["other_instance_points_allowed_below_grasp_surface"] = allowed_below_count
            trial_screen["allow_other_instance_below_grasp_m"] = float(args.allow_other_instance_below_grasp_m)
            if trial_accepted:
                chosen = (float(extra_up), camera_offset, trial_screen, trial_projection)
                break
        cloud_accepted = chosen is not None
        if chosen is None:
            cloud_screen = trial_screen or {
                "model": "sam_depth_open_parallel_jaw_keepout",
                "note": "no upward target kept both parallel-jaw contacts on selected SAM instance",
            }
            record["auto_grasp_upward_clearance_m"] = None
            # Do not conflate a later depth-cloud collision rejection with a
            # failed two-sided jaw placement in the review image/report.
            record["jaw_contacts_on_selected_mask_accepted"] = final_target_contacts_supported
        else:
            extra_up, camera_offset, cloud_screen, projection = chosen
            apply_auto_upward_clearance(candidate_plan, extra_up)
            record["auto_grasp_upward_clearance_m"] = extra_up
            # The review image must show the actually shifted closing point,
            # not the unadjusted STL template centre.
            candidate_plan["predicted_parallel_jaw_contacts_image"] = projection
            record["predicted_parallel_jaw_contacts_image"] = projection
        candidate_plan["pointcloud_gripper_screen"] = cloud_screen
        record["pointcloud_gripper_screen"] = cloud_screen
        record["pointcloud_gripper_screen_accepted"] = cloud_accepted
        if not cloud_accepted:
            record["planner_rejection_reason"] = (
                "final_target_parallel_jaw_contacts_not_on_selected_sam_instance"
                if not final_target_contacts_supported else "depth_cloud_gripper_keepout_intrusion"
            )
            continue
        angle = float(candidate_plan["grasp_angle_to_table_deg"])
        record["grasp_angle_to_table_deg"] = angle
        record["min_grasp_angle_to_table_deg"] = float(args.min_grasp_angle_to_table_deg)
        upright_side = bool(record.get("upright_image_side_grasp"))
        # The 65-degree rule is for a top-down grasp of a flat workpiece.
        # An image-upright workpiece is explicitly requested to be grasped
        # from the side; Planner remains the feasibility authority for it.
        record["grasp_angle_accepted"] = upright_side or angle > float(args.min_grasp_angle_to_table_deg)
        candidate_plan["upright_image_side_grasp"] = upright_side
        if not record["grasp_angle_accepted"]:
            record["planner_rejection_reason"] = "grasp_angle_to_table_below_minimum"
            angle_fallbacks.append((record, candidate_plan))
            continue
        rank = len(planner_candidates) + 1
        candidate_plan["candidate_rank"] = rank
        record["planner_candidate_rank"] = rank
        planner_candidates.append(candidate_plan)
        (out / f"planner_candidate_{rank}.json").write_text(
            json.dumps(candidate_plan, ensure_ascii=False, indent=2)
        )
    # A usable front face is preferred over a normally angled back face.  Its
    # low-angle pose is therefore promoted by tilting to the 65-degree limit
    # before rank assignment.  Planner still remains the feasibility gate.
    normal_candidate_coverage = [
        next(
            record["single_instance_completeness"]["observed_points_explained_by_single_stl_ratio"]
            for record in records if int(record["mask_index"]) == int(plan["selected_mask_index"])
        )
        for plan in planner_candidates
    ]
    best_normal_coverage = max(normal_candidate_coverage, default=-float("inf"))
    # A low-angle front fallback may only outrank normal candidates when its
    # segmented single-STL coverage is effectively just as complete (within
    # 1 percentage point).  Completeness therefore remains the first policy.
    front_angle_fallbacks = [
        item for item in angle_fallbacks
        if item[0]["face_side"] == "front"
        and item[0]["single_instance_completeness"]["observed_points_explained_by_single_stl_ratio"]
        >= best_normal_coverage - .01
    ]
    if front_angle_fallbacks:
        promoted = []
        for record, plan in front_angle_fallbacks:
            fallback = force_plan_grasp_angle(plan, float(args.min_grasp_angle_to_table_deg))
            record["angle_fallback_to_minimum"] = True
            record.pop("planner_rejection_reason", None)
            promoted.append(fallback)
        planner_candidates = promoted + planner_candidates
    # With no usable front fallback or normal candidate, retain every
    # geometrically valid low-angle option as the final fallback.
    elif not planner_candidates:
        for record, plan in angle_fallbacks:
            fallback = force_plan_grasp_angle(plan, float(args.min_grasp_angle_to_table_deg))
            record["angle_fallback_to_minimum"] = True
            record.pop("planner_rejection_reason", None)
            planner_candidates.append(fallback)
    # Candidates may have been promoted above an existing back candidate;
    # rewrite ranks/files once so the live runner sees the same ordering as
    # the review report.
    for rank, candidate in enumerate(planner_candidates, start=1):
        candidate["candidate_rank"] = rank
        for record in records:
            if int(record["mask_index"]) == int(candidate["selected_mask_index"]):
                record["planner_candidate_rank"] = rank
        (out / f"planner_candidate_{rank}.json").write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2)
        )
    execution_plan = planner_candidates[0] if planner_candidates else None
    # Human-reviewable image: a yellow dot is the grasp centre; blue/red dots
    # are the two local -Y/+Y closing contacts and the line joins the fingers.
    # Green means the candidate survives all geometric gates; red means it was
    # rejected before Planner.  This uses the exact ICP pose and local grasp
    # point persisted in the JSON plan.
    grasp_overlay = image.copy()
    for record in records:
        if not record["accepted"] or "predicted_parallel_jaw_contacts_image" not in record:
            continue
        projection = record["predicted_parallel_jaw_contacts_image"]
        # A rejected bracket can intentionally contain only a centre and a
        # rejection flag (for example when its local X-axis cross-section is
        # discontinuous).  It must still be reviewable, never crash the whole
        # perception run while rendering the diagnostic overlay.
        center = projection.get("grasp_center_pixel")
        left = projection.get("local_minus_y_finger_center_pixel",
                              projection.get("local_minus_y_contact_pixel"))
        right = projection.get("local_plus_y_finger_center_pixel",
                               projection.get("local_plus_y_contact_pixel"))
        valid = bool(record.get("jaw_contacts_on_selected_mask_accepted")) and not record.get("planner_rejection_reason")
        line_color = (0, 255, 0) if valid else (0, 0, 255)
        if left is not None and right is not None:
            cv2.line(grasp_overlay, tuple(left), tuple(right), line_color, 2)
            cv2.circle(grasp_overlay, tuple(left), 5, (255, 0, 0), -1)
            cv2.circle(grasp_overlay, tuple(right), 5, (0, 0, 255), -1)
        if center is not None:
            cv2.circle(grasp_overlay, tuple(center), 5, (0, 255, 255), -1)
            cv2.putText(grasp_overlay, f"#{record['mask_index']} {'OK' if valid else 'REJECT'}",
                        (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, .45, line_color, 1)
    cv2.imwrite(str(out / "grasp_prediction.png"), grasp_overlay)
    if execution_plan is not None:
        (out / "planner_plan.json").write_text(json.dumps(execution_plan, ensure_ascii=False, indent=2))
    timing_s["post_icp_candidate_screening_and_output"] = time.perf_counter() - records_started - float(
        timing_s["icp_result_scoring_and_record_build"]
    )
    timing_s["total_before_report_write"] = time.perf_counter() - total_started
    report = {"mode": "traditional_sam_depth_stl_icp_no_graspnet", "input_log": str(args.log.resolve()),
              "stl": str(args.stl.resolve()), "instance_count": len(records),
              "sam_duplicate_instance_mask_pairs": duplicate_mask_pairs, "selected": selected,
              "planner_plan": execution_plan, "planner_candidate_count": len(planner_candidates),
              "records": records, "timing_s": timing_s,
              "note": "offline plan only; no robot command was sent"}
    (out / "traditional_pose_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("-> traditional perception timing (s): " + json.dumps(timing_s, ensure_ascii=False), flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    try:
        run()
    finally:
        shutdown_persistent_icp_workers()


if __name__ == "__main__":
    if "--serve-icp-worker-pool" in sys.argv:
        service_parser = argparse.ArgumentParser()
        service_parser.add_argument("--serve-icp-worker-pool", action="store_true")
        service_parser.add_argument("--socket", type=Path, required=True)
        service_args = service_parser.parse_args()
        serve_icp_worker_pool(service_args.socket)
    else:
        main()
