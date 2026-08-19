"""Lightweight CAD-to-observation point-cloud registration."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.spatial import cKDTree

from .geometry import make_T, transform_points


@dataclass
class RegistrationResult:
    T_target_source: np.ndarray
    rmse_m: float
    fitness: float
    inlier_ratio: float
    n_corr: int
    iterations: int
    init_name: str


def _best_fit_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    cs = src.mean(axis=0)
    ct = tgt.mean(axis=0)
    X = src - cs
    Y = tgt - ct
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = ct - R @ cs
    return make_T(R, t)


def _pca_frame(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    X = pts - pts.mean(axis=0)
    C = (X.T @ X) / max(len(pts) - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    R = vecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1.0
    return R


def make_pca_initial_guesses(source: np.ndarray,
                             target: np.ndarray,
                             use_all_axis_flips: bool = True) -> list[tuple[str, np.ndarray]]:
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    Rs = _pca_frame(src)
    Rt = _pca_frame(tgt)
    cs = src.mean(axis=0)
    ct = tgt.mean(axis=0)
    guesses: list[tuple[str, np.ndarray]] = []
    signs = list(itertools.product([-1.0, 1.0], repeat=3)) if use_all_axis_flips else [(1, 1, 1)]
    for sx, sy, sz in signs:
        S = np.diag([sx, sy, sz])
        if np.linalg.det(S) < 0:
            continue
        R = Rt @ S @ Rs.T
        t = ct - R @ cs
        name = f"pca_flip_{int(sx):+d}{int(sy):+d}{int(sz):+d}"
        guesses.append((name, make_T(R, t)))
    guesses.append(("centroid_identity", make_T(np.eye(3), ct - cs)))
    return guesses


def run_icp_point_to_point(source_points: np.ndarray,
                           target_points: np.ndarray,
                           init_T: np.ndarray,
                           max_corr_m: float = 0.006,
                           max_iters: int = 40,
                           trim_quantile: float = 0.85) -> RegistrationResult:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    tree = cKDTree(target)
    T = np.asarray(init_T, dtype=np.float64).copy()
    n_corr = 0
    rmse = float("inf")
    fitness = 0.0
    for it in range(int(max_iters)):
        src_w = transform_points(T, source)
        dists, idx = tree.query(src_w, k=1, workers=-1)
        inlier = dists <= float(max_corr_m)
        if np.count_nonzero(inlier) < 12:
            order = np.argsort(dists)[:max(12, min(200, len(dists)))]
            inlier = np.zeros_like(dists, dtype=bool)
            inlier[order] = True
        if trim_quantile < 1.0 and np.count_nonzero(inlier) > 20:
            cutoff = np.quantile(dists[inlier], trim_quantile)
            inlier &= dists <= cutoff
        src_sel = src_w[inlier]
        tgt_sel = target[idx[inlier]]
        dT = _best_fit_transform(src_sel, tgt_sel)
        T_new = dT @ T
        translation_delta = np.linalg.norm(T_new[:3, 3] - T[:3, 3])
        rotation_cos = np.clip((np.trace(dT[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        rotation_delta = float(np.arccos(rotation_cos))
        T = T_new

        # Quality metrics must describe the updated transform, not the previous iteration.
        updated = transform_points(T, source)
        updated_dists, _ = tree.query(updated, k=1, workers=-1)
        updated_inlier = updated_dists <= float(max_corr_m)
        n_corr = int(np.count_nonzero(updated_inlier))
        fitness = float(n_corr / max(len(updated_dists), 1))
        rmse = (
            float(np.sqrt(np.mean(np.square(updated_dists[updated_inlier]))))
            if n_corr > 0 else float("inf")
        )
        if translation_delta < 1e-6 and rotation_delta < 1e-5:
            return RegistrationResult(T, rmse, fitness, n_corr / max(len(dists), 1), n_corr, it + 1, "")
    return RegistrationResult(T, rmse, fitness, n_corr / max(len(source), 1), n_corr, int(max_iters), "")


def register_cad_to_observed_candidates(cad_points_stl: np.ndarray,
                                        observed_points_world: np.ndarray,
                                        max_corr_m: float = 0.006,
                                        max_iters: int = 45,
                                        max_cad_points: int = 6000,
                                        max_obs_points: int = 6000,
                                        seed: int = 0) -> list[RegistrationResult]:
    rng = np.random.default_rng(seed)
    cad = np.asarray(cad_points_stl, dtype=np.float64).reshape(-1, 3)
    obs = np.asarray(observed_points_world, dtype=np.float64).reshape(-1, 3)
    if len(cad) > max_cad_points:
        cad = cad[rng.choice(len(cad), max_cad_points, replace=False)]
    if len(obs) > max_obs_points:
        obs = obs[rng.choice(len(obs), max_obs_points, replace=False)]
    candidates: list[RegistrationResult] = []
    for name, T0 in make_pca_initial_guesses(cad, obs):
        res = run_icp_point_to_point(
            cad, obs, T0,
            max_corr_m=max_corr_m,
            max_iters=max_iters,
        )
        res.init_name = name
        candidates.append(res)
    if not candidates:
        raise RuntimeError("ICP produced no candidate")
    return sorted(candidates, key=lambda result: (result.fitness, -result.rmse_m), reverse=True)


def register_cad_to_observed(cad_points_stl: np.ndarray,
                             observed_points_world: np.ndarray,
                             max_corr_m: float = 0.006,
                             max_iters: int = 45,
                             max_cad_points: int = 6000,
                             max_obs_points: int = 6000,
                             seed: int = 0) -> RegistrationResult:
    return register_cad_to_observed_candidates(
        cad_points_stl,
        observed_points_world,
        max_corr_m=max_corr_m,
        max_iters=max_iters,
        max_cad_points=max_cad_points,
        max_obs_points=max_obs_points,
        seed=seed,
    )[0]
