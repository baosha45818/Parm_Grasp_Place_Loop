#!/usr/bin/env python3
"""Offline CAD-to-point-cloud pose estimation for the gripped workpiece.

This is deliberately an observation-only program.  It reads a SAM2 mask, an
aligned Femto depth image and the accepted laser calibration, then writes a
``T_base_object`` estimate.  It does not import the robot SDK and cannot send
any robot or gripper command.

The implementation follows ``graspnet/example/place/pc_pose``:
mask + depth -> object cloud -> CAD surface sampling -> PCA initial hypotheses
-> point-to-point ICP.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def load_stl_triangles_m(path: Path) -> np.ndarray:
    """Read binary or ASCII STL and return triangles in metres."""
    blob = path.read_bytes()
    if len(blob) >= 84:
        count = struct.unpack("<I", blob[80:84])[0]
        if len(blob) == 84 + count * 50:
            records = np.frombuffer(
                blob,
                dtype=np.dtype([
                    ("normal", "<f4", (3,)),
                    ("v0", "<f4", (3,)),
                    ("v1", "<f4", (3,)),
                    ("v2", "<f4", (3,)),
                    ("attr", "<u2"),
                ]),
                offset=84,
                count=count,
            )
            triangles = np.stack([records["v0"], records["v1"], records["v2"]], axis=1)
        else:
            triangles = _load_ascii_stl(blob)
    else:
        triangles = _load_ascii_stl(blob)
    triangles = np.asarray(triangles, dtype=np.float64)
    # The deployed workpiece STL is already metres; retain compatibility with
    # common CAD exports expressed in millimetres.
    if float(np.ptp(triangles.reshape(-1, 3), axis=0).max()) > 1.0:
        triangles *= 1e-3
    return triangles


def _load_ascii_stl(blob: bytes) -> np.ndarray:
    values: list[list[float]] = []
    for line in blob.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            values.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not values or len(values) % 3:
        raise ValueError("STL contains no valid triangles")
    return np.asarray(values, dtype=np.float64).reshape(-1, 3, 3)


def sample_surface_points(triangles: np.ndarray, count: int, seed: int) -> np.ndarray:
    tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(tris), size=count, replace=True, p=probabilities)
    r1, r2 = rng.random(count), rng.random(count)
    root_r1 = np.sqrt(r1)
    return ((1 - root_r1)[:, None] * tris[selected, 0]
            + (root_r1 * (1 - r2))[:, None] * tris[selected, 1]
            + (root_r1 * r2)[:, None] * tris[selected, 2])


def mask_depth_to_points_laser(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                               depth_scale: float, min_depth_m: float,
                               max_depth_m: float, erode_px: int) -> np.ndarray:
    if mask.shape[:2] != depth.shape[:2]:
        raise ValueError(f"mask shape {mask.shape[:2]} != depth shape {depth.shape[:2]}; "
                         "mask and depth must be aligned")
    selected = (np.asarray(mask) > 0).astype(np.uint8)
    if erode_px:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        selected = cv2.erode(selected, kernel, iterations=1)
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = np.asarray(depth, dtype=np.float64) * depth_scale
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    valid = (selected > 0) & np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    return np.stack([x[valid], y[valid], z[valid]], axis=1)


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    if voxel_m <= 0:
        return np.asarray(points, dtype=np.float64)
    keys = np.floor(np.asarray(points) / voxel_m).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points, dtype=np.float64)[np.sort(index)]


def _best_fit_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_center, tgt_center = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - src_center).T @ (target - tgt_center))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return make_T(R, tgt_center - R @ src_center)


def _pca_frame(points: np.ndarray) -> np.ndarray:
    X = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(X.T @ X / max(len(points) - 1, 1))
    R = vectors[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def pca_initial_guesses(source: np.ndarray, target: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    source_frame, target_frame = _pca_frame(source), _pca_frame(target)
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                flips = np.diag([sx, sy, sz])
                if np.linalg.det(flips) > 0:
                    R = target_frame @ flips @ source_frame.T
                    yield f"pca_flip_{int(sx):+d}{int(sy):+d}{int(sz):+d}", make_T(R, target_center - R @ source_center)
    yield "centroid_identity", make_T(np.eye(3), target_center - source_center)


@dataclass
class Candidate:
    init_name: str
    T_base_object: list[list[float]]
    rmse_m: float | None
    fitness: float
    inlier_ratio: float
    n_corr: int
    iterations: int


def run_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray,
            max_corr_m: float, max_iters: int, trim_quantile: float) -> Candidate:
    tree, T = cKDTree(target), initial.copy()
    for iteration in range(max_iters):
        transformed = transform_points(T, source)
        distances, indices = tree.query(transformed, workers=-1)
        inlier = distances <= max_corr_m
        if inlier.sum() < 12:
            inlier[np.argsort(distances)[:max(12, min(200, len(distances)))]] = True
        if trim_quantile < 1.0 and inlier.sum() > 20:
            inlier &= distances <= np.quantile(distances[inlier], trim_quantile)
        delta = _best_fit_transform(transformed[inlier], target[indices[inlier]])
        next_T = delta @ T
        d_translation = np.linalg.norm(next_T[:3, 3] - T[:3, 3])
        d_rotation = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
        T = next_T
        if d_translation < 1e-6 and d_rotation < 1e-5:
            break
    final_distances, _ = tree.query(transform_points(T, source), workers=-1)
    final_inliers = final_distances <= max_corr_m
    n_corr = int(final_inliers.sum())
    return Candidate(
        init_name="",
        T_base_object=T.tolist(),
        rmse_m=float(np.sqrt(np.mean(final_distances[final_inliers] ** 2))) if n_corr else None,
        fitness=float(n_corr / len(source)),
        inlier_ratio=float(n_corr / len(source)),
        n_corr=n_corr,
        iterations=iteration + 1,
    )


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        result = np.load(path)
    else:
        result = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if result is None or result.ndim != 2:
        raise ValueError("depth must be a one-channel .npy or 16-bit image")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Observation-only Femto CAD pose estimator")
    parser.add_argument("--mask", type=Path, required=True, help="SAM2 binary mask aligned with depth")
    parser.add_argument("--depth", type=Path, required=True, help="aligned depth (.npy or 16-bit png)")
    parser.add_argument("--depth-scale", type=float, required=True, help="metres per depth unit")
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--laser-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--erode-px", type=int, default=1)
    parser.add_argument("--voxel-m", type=float, default=0.003)
    parser.add_argument("--cad-points", type=int, default=12000)
    parser.add_argument("--max-corr-m", type=float, default=0.006)
    parser.add_argument("--max-iters", type=int, default=45)
    parser.add_argument("--trim-quantile", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {args.mask}")
    calibration = json.loads(args.laser_calibration.read_text())
    if calibration.get("status") != "ACCEPTED":
        raise RuntimeError("laser calibration status must be ACCEPTED")
    K = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    T_base_laser = np.asarray(calibration["T_base_laser"], dtype=np.float64)
    if K.shape != (3, 3) or T_base_laser.shape != (4, 4):
        raise ValueError("invalid camera_matrix or T_base_laser in calibration")

    points_laser = mask_depth_to_points_laser(
        mask, load_depth(args.depth), K, args.depth_scale,
        args.min_depth_m, args.max_depth_m, args.erode_px,
    )
    if len(points_laser) < 100:
        raise RuntimeError(f"only {len(points_laser)} valid object points after masking")
    points_base = voxel_downsample(transform_points(T_base_laser, points_laser), args.voxel_m)
    cad_points = sample_surface_points(load_stl_triangles_m(args.stl), args.cad_points, args.seed)
    candidates = []
    for name, initial in pca_initial_guesses(cad_points, points_base):
        candidate = run_icp(cad_points, points_base, initial, args.max_corr_m, args.max_iters, args.trim_quantile)
        candidate.init_name = name
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (item.fitness, -(item.rmse_m if item.rmse_m is not None else float("inf"))),
        reverse=True,
    )
    result = {
        "pipeline": "SAM2 mask -> aligned depth -> CAD sampling -> PCA hypotheses -> point-to-point ICP",
        "frame_convention": "T_base_object maps STL/object coordinates into arm1 base coordinates",
        "inputs": {"mask": str(args.mask), "depth": str(args.depth), "stl": str(args.stl), "laser_calibration": str(args.laser_calibration)},
        "point_counts": {"masked_depth_points": int(len(points_laser)), "observed_base_points": int(len(points_base)), "cad_surface_points": int(len(cad_points))},
        "best": asdict(candidates[0]),
        "candidates": [asdict(item) for item in candidates],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")
    rmse_text = "n/a" if candidates[0].rmse_m is None else f"{candidates[0].rmse_m * 1000:.2f} mm"
    print(f"best: {candidates[0].init_name}, fitness={candidates[0].fitness:.3f}, rmse={rmse_text}")


if __name__ == "__main__":
    main()
