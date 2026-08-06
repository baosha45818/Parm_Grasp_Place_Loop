#!/usr/bin/env python3
"""Persistent Version6 CAD sampling and ICP service."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socketserver
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "deploy" / "version6"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from workpiece_pose import RegistrationQualityError  # noqa: E402
from example.place.pc_pose.registration import register_cad_to_observed_candidates  # noqa: E402
from example.place.pc_pose.stl_model import load_stl_triangles_m, sample_surface_points  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Persistent Version6 STL/CAD ICP service")
    p.add_argument("--config", default=str(ROOT / "deploy" / "version6" / "config_template.json"))
    p.add_argument("--socket-path", default="/tmp/graspnet-v7-pose.sock")
    p.add_argument("--allow-root", default=str(ROOT / "deploy" / "version6" / "output"))
    return p.parse_args()


class Engine:
    def __init__(self, config: dict, config_sha256: str):
        self.cfg = dict(config["inspection"])
        stl = Path(config["workpiece"]["stl_path"])
        self.triangles = load_stl_triangles_m(stl)
        self.cad = sample_surface_points(self.triangles, n_points=int(self.cfg["cad_sample_points"]), seed=int(self.cfg.get("seed", 0)))
        self.lock = threading.Lock()
        self.config_sha256 = config_sha256
        print(f"-> pose service ready: STL={stl}, cached_cad_points={len(self.cad)}", flush=True)

    def estimate(self, request: dict) -> dict:
        clouds = [np.asarray(np.load(path), dtype=np.float64) for path in request["cloud_paths"]]
        minimum = int(self.cfg.get("min_points_per_view", 3))
        if not clouds or any(cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < minimum or not np.isfinite(cloud).all() for cloud in clouds):
            raise RuntimeError(f"cloud_paths must contain finite Nx3 EEF clouds with at least {minimum} points each")
        observed = np.concatenate(clouds, axis=0)
        from example.place.pc_pose.mask_depth import voxel_downsample_np
        observed = voxel_downsample_np(observed, float(self.cfg["observed_voxel_m"]))
        if len(observed) < 3 or not np.isfinite(observed).all():
            raise RuntimeError("observed EEF cloud is invalid after voxel downsampling")
        with self.lock:
            candidates = register_cad_to_observed_candidates(self.cad, observed,
                max_corr_m=float(self.cfg["icp_max_corr_m"]), max_iters=int(self.cfg["icp_max_iters"]),
                max_cad_points=int(self.cfg["icp_max_cad_points"]), max_obs_points=int(self.cfg["icp_max_observed_points"]),
                seed=int(self.cfg.get("seed", 0)))
        best = candidates[0]
        if best.rmse_m > float(self.cfg["quality_max_rmse_m"]): raise RegistrationQualityError(f"pose RMSE {best.rmse_m:.4f} m exceeds quality gate")
        if best.fitness < float(self.cfg["quality_min_fitness"]): raise RegistrationQualityError(f"pose fitness {best.fitness:.3f} is below quality gate")
        output = Path(request["output_path"]); output.parent.mkdir(parents=True, exist_ok=True)
        data = {"T_eef_object": best.T_target_source.tolist(), "observed_eef_path": str(output),
                "registration": {"fused_observed_points": int(len(observed)), "rmse_m": float(best.rmse_m),
                "fitness": float(best.fitness), "inlier_ratio": float(best.inlier_ratio), "correspondences": int(best.n_corr),
                "iterations": int(best.iterations), "initialization": best.init_name,
                "orientation_candidates": [{"T_eef_object": item.T_target_source.tolist(), "rmse_m": float(item.rmse_m),
                "fitness": float(item.fitness), "initialization": item.init_name} for item in candidates]}}
        np.save(output, observed.astype(np.float32)); return data


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            response = self.server.handle_request(json.loads(self.rfile.readline().decode()))  # type: ignore[attr-defined]
            self.wfile.write((json.dumps({"ok": True, "result": response}) + "\n").encode())
        except Exception as exc: self.wfile.write((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    def __init__(self, path, engine, root): super().__init__(path, Handler); self.engine, self.root = engine, root.resolve()
    def handle_request(self, request):
        if request.get("op") == "status":
            return {"allow_root": str(self.root), "config_sha256": self.engine.config_sha256}
        for path in [*request["cloud_paths"], request["output_path"]]:
            resolved = Path(path).resolve()
            if self.root not in (resolved, *resolved.parents): raise RuntimeError("pose paths must stay under allow-root")
        return self.engine.estimate(request)


def main():
    args = parse_args(); config_bytes = Path(args.config).read_bytes(); config = json.loads(config_bytes); root = Path(args.allow_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    sock = Path(args.socket_path)
    if sock.exists() or sock.is_symlink():
        raise RuntimeError(f"refusing to replace existing pose socket: {sock}; stop the old service or remove a confirmed stale socket")
    with Server(str(sock), Engine(config, hashlib.sha256(config_bytes).hexdigest()), root) as server:
        os.chmod(sock, 0o600); print(f"-> pose service listening on {sock}", flush=True); server.serve_forever()

if __name__ == "__main__": main()
