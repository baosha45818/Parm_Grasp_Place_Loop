#!/usr/bin/env python3
"""Persistent local GraspNet inference service for Version7.

The service owns CUDA/model lifetime.  Clients only submit an existing camera
point cloud path over a Unix-domain socket, so capture cycles do not reload the
checkpoint.  It deliberately accepts point clouds rather than robot commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socketserver
import sys
import threading
import time

import numpy as np
import open3d as o3d
import torch
from graspnetAPI import GraspGroup

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "models", ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from graspnet import GraspNet, pred_decode  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent local GraspNet point-cloud service")
    p.add_argument("--checkpoint-path", default=str(ROOT / "checkpoint" / "checkpoint-rs.tar"))
    p.add_argument("--num-view", type=int, default=300)
    p.add_argument("--warmup-num-point", type=int, default=50000,
                   help="run one real forward/pred_decode with this many synthetic points before listening")
    p.add_argument("--no-warmup", action="store_true", help="skip GPU warmup (diagnostic use only)")
    p.add_argument("--socket-path", default="/tmp/graspnet-v7.sock")
    p.add_argument("--allow-root", default=str(ROOT / "deploy" / "version7" / "jobs"))
    p.add_argument("--pid-file", default=os.environ.get("GRASPNET_SERVICE_PID_FILE", ""),
                   help="optional PID file managed by the service process")
    return p.parse_args()


def sample_points(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) >= count:
        indices = np.random.choice(len(points), count, replace=False)
    else:
        indices = np.r_[np.arange(len(points)), np.random.choice(len(points), count - len(points), replace=True)]
    return points[indices]


def collision_cloud(points: np.ndarray, request: dict) -> np.ndarray:
    thickness = float(request.get("collision_table_thickness", 0.0))
    if thickness <= 0.0:
        return points
    plane_dist = float(request.get("collision_table_plane_dist", 0.005))
    min_ratio = float(request.get("collision_table_min_inlier_ratio", 0.15))
    min_normal_z = float(request.get("collision_table_min_normal_z", 0.5))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, inliers = cloud.segment_plane(distance_threshold=plane_dist, ransac_n=3, num_iterations=1000)
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    if len(inliers) / len(points) < min_ratio or abs(normal[2]) < min_normal_z:
        return points
    surface = np.asarray(cloud.select_by_index(inliers).voxel_down_sample(max(float(request["voxel_size"]) * .5, .003)).points)
    if len(surface) == 0:
        return points
    if np.dot(normal, -surface.mean(axis=0)) < 0:
        normal = -normal
    spacing = max(float(request["voxel_size"]) * .5, .003)
    offsets = np.arange(0.0, thickness + spacing * .5, spacing)
    solid = np.concatenate([surface - normal * offset for offset in offsets], axis=0)
    return np.concatenate((points, solid.astype(points.dtype)), axis=0)


class Engine:
    def __init__(self, args: argparse.Namespace):
        self.num_view = int(args.num_view)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = str(Path(args.checkpoint_path).resolve())
        self.started_unix_ms = int(time.time() * 1000)
        self.lock = threading.Lock()
        self.net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                            cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04], is_training=False)
        self.net.to(self.device)
        checkpoint = torch.load(args.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
        print(f"-> GraspNet service ready: checkpoint={args.checkpoint_path}, epoch={checkpoint.get('epoch')}, device={self.device}", flush=True)
        if not args.no_warmup:
            self.warmup(int(args.warmup_num_point))

    def warmup(self, num_point: int) -> None:
        """Materialize CUDA context, kernels and allocator before the first workpiece."""
        if num_point <= 0:
            raise RuntimeError("--warmup-num-point must be positive")
        # A shallow, non-degenerate cloud keeps GraspNet's geometric operators
        # valid while exercising the same batch=1 shape used by a client.
        side = int(np.ceil(np.sqrt(num_point)))
        x, y = np.meshgrid(np.linspace(-.1, .1, side, dtype=np.float32),
                           np.linspace(-.1, .1, side, dtype=np.float32))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .5, dtype=np.float32)))[:num_point]
        started = time.monotonic()
        with self.lock, torch.no_grad():
            endpoints = {"point_clouds": torch.from_numpy(points[None]).to(self.device),
                         "cloud_colors": np.zeros_like(points, dtype=np.float32)}
            pred_decode(self.net(endpoints))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        print(f"-> GraspNet warmup complete: num_point={num_point}, elapsed={time.monotonic() - started:.2f}s", flush=True)

    def infer(self, request: dict) -> dict:
        return self.infer_batch([request])[0]

    def infer_batch(self, requests: list[dict]) -> list[dict]:
        """Run one GPU forward for equally sampled independent camera clouds."""
        if not requests:
            raise RuntimeError("batch request is empty")
        if any(int(request["num_point"]) != int(requests[0]["num_point"]) for request in requests):
            raise RuntimeError("all batch requests must use the same num_point")
        started = time.monotonic()
        clouds = []
        for request in requests:
            points = np.asarray(np.load(request["point_cloud_path"]), dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise RuntimeError("point_cloud_path must contain finite Nx3 points")
            points = points[np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)]
            if len(points) < 100:
                raise RuntimeError(f"point cloud has too few valid points: {len(points)}")
            clouds.append(points)
        count = int(requests[0]["num_point"])
        if count <= 0:
            raise RuntimeError("num_point must be positive")
        sampled = np.stack([sample_points(points, count) for points in clouds], axis=0)
        model_started = time.monotonic()
        with self.lock, torch.no_grad():
            endpoints = {"point_clouds": torch.from_numpy(sampled).to(self.device),
                         "cloud_colors": np.zeros_like(sampled, dtype=np.float32)}
            predictions = pred_decode(self.net(endpoints))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        model_elapsed_s = time.monotonic() - model_started
        results = []
        for request, points, prediction in zip(requests, clouds, predictions):
            group = GraspGroup(prediction.detach().cpu().numpy())
            threshold = float(request["collision_thresh"])
            collision_started = time.monotonic()
            if threshold > 0:
                detector = ModelFreeCollisionDetector(collision_cloud(points, request), voxel_size=float(request["voxel_size"]))
                group = group[~detector.detect(group, approach_dist=float(request["approach_dist"]), collision_thresh=threshold)]
            output = Path(request["output_grasps_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, group.grasp_group_array.astype(np.float32))
            results.append({"grasps_path": str(output), "num_input_points": int(len(points)),
                            "num_grasps": int(len(group)), "num_view": self.num_view,
                            "model_elapsed_s": model_elapsed_s,
                            "collision_elapsed_s": time.monotonic() - collision_started,
                            "inference_elapsed_s": time.monotonic() - started})
        return results

class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline().decode("utf-8"))
            response = self.server.handle_request(request)  # type: ignore[attr-defined]
            self.wfile.write((json.dumps({"ok": True, "result": response}) + "\n").encode("utf-8"))
        except Exception as exc:
            self.wfile.write((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode("utf-8"))


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: str, engine: Engine, allow_root: Path):
        super().__init__(socket_path, Handler)
        self.engine = engine
        self.allow_root = allow_root.resolve()

    def handle_request(self, request: dict) -> dict:
        if request.get("op") == "status":
            return {
                "allow_root": str(self.allow_root),
                "num_view": self.engine.num_view,
                "device": str(self.engine.device),
                "checkpoint_path": self.engine.checkpoint_path,
                "started_unix_ms": self.engine.started_unix_ms,
                "pid": os.getpid(),
            }
        requests = request.get("requests") if isinstance(request.get("requests"), list) else [request]
        if not requests:
            raise RuntimeError("batch request is empty")
        for item in requests:
            for key in ("point_cloud_path", "output_grasps_path"):
                path = Path(item[key]).resolve()
                if self.allow_root not in (path, *path.parents):
                    raise RuntimeError(f"{key} must stay under service allow-root: {self.allow_root}")
            if int(item.get("num_view", self.engine.num_view)) != self.engine.num_view:
                raise RuntimeError(f"service uses num_view={self.engine.num_view}; restart it to change num_view")
        results = self.engine.infer_batch(requests)
        return {"batch": results} if len(requests) > 1 else results[0]


def main() -> None:
    args = parse_args()
    allow_root = Path(args.allow_root).resolve()
    allow_root.mkdir(parents=True, exist_ok=True)
    socket_path = Path(args.socket_path)
    if socket_path.exists() or socket_path.is_symlink():
        raise RuntimeError(f"refusing to replace existing GraspNet socket: {socket_path}; stop the old service or remove a confirmed stale socket")
    pid_file = Path(args.pid_file).resolve() if args.pid_file else None
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def stop_service(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_service)
    signal.signal(signal.SIGINT, stop_service)
    socket_created = False
    try:
        server = Server(str(socket_path), Engine(args), allow_root)
        socket_created = True
        with server:
            os.chmod(socket_path, 0o600)
            print(f"-> listening on {socket_path}; allow-root={allow_root}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("-> GraspNet service stopping", flush=True)
    finally:
        if socket_created and socket_path.exists() and socket_path.is_socket():
            socket_path.unlink()
        if pid_file is not None and pid_file.exists():
            try:
                recorded_pid = int(pid_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                recorded_pid = -1
            if recorded_pid == os.getpid():
                pid_file.unlink()


if __name__ == "__main__":
    main()
