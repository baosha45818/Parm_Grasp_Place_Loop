#!/usr/bin/env python3
"""工件 2 反面朝上的全局初始化与全 6DoF ICP。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def fit_table_plane(points: np.ndarray) -> np.ndarray:
    """拟合整帧桌面，法向统一指向相机一侧。"""
    o3d.utility.random.seed(17)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float))).voxel_down_sample(.003)
    plane, _ = cloud.segment_plane(.004, 3, 1600)
    plane = np.asarray(plane, dtype=float)
    plane /= max(float(np.linalg.norm(plane[:3])), 1e-12)
    if float(np.dot(plane[:3], -np.asarray(points, dtype=float).mean(axis=0))) < 0.0:
        plane *= -1.0
    return plane


def fpfh_features(points: np.ndarray, voxel_m: float) -> tuple[o3d.geometry.PointCloud, object]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float)))
    down = cloud.voxel_down_sample(voxel_m)
    if len(down.points) < 40:
        raise RuntimeError("全局 ICP 初始化降采样后点数不足")
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 2.5, max_nn=40))
    return down, o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 5.0, max_nn=100)
    )


def fpfh_ransac_initial(source_features: tuple, target_features: tuple, voxel_m: float,
                        iterations: int, seed: int) -> np.ndarray:
    o3d.utility.random.seed(seed)
    source_down, source_feature = source_features
    target_down, target_feature = target_features
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_feature, target_feature, True, voxel_m * 3.0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.80),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_m * 3.0),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, .999),
        )
    if result.fitness < .03 or not np.isfinite(result.inlier_rmse):
        raise RuntimeError(f"FPFH/RANSAC 无有效全局初值 (fitness={result.fitness:.3f})")
    return np.asarray(result.transformation, dtype=float)


def full_6d_point_to_point_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray,
                                iterations: int, max_corr_m: float, workers: int) -> tuple[np.ndarray, float, float]:
    """点到点 ICP；允许全部旋转和三轴平移变化。"""
    tree, T = cKDTree(target), initial.copy()
    for _ in range(iterations):
        current = transform(T, source)
        distances, nearest = tree.query(current, k=1, workers=workers)
        keep = distances < max_corr_m
        if int(np.count_nonzero(keep)) < 30:
            keep[np.argsort(distances)[:min(250, len(distances))]] = True
        keep &= distances <= np.quantile(distances[keep], .82)
        moving, fixed = current[keep], target[nearest[keep]]
        moving_center, fixed_center = moving.mean(axis=0), fixed.mean(axis=0)
        U, _, Vt = np.linalg.svd((moving - moving_center).T @ (fixed - fixed_center))
        rotation = Vt.T @ U.T
        if np.linalg.det(rotation) < 0.0:
            Vt[-1] *= -1.0
            rotation = Vt.T @ U.T
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = fixed_center - rotation @ moving_center
        T = delta @ T
        if np.linalg.norm(rotation - np.eye(3), ord="fro") < 2e-5 and np.linalg.norm(delta[:3, 3]) < 2e-6:
            break
    distances, _ = tree.query(transform(T, source), k=1, workers=workers)
    inliers = distances < max_corr_m
    rmse = float(np.sqrt(np.mean(distances[inliers] ** 2))) if np.any(inliers) else float("inf")
    return T, rmse, float(np.mean(inliers))


def semantic_half_turns(initial: np.ndarray, stl_to_object: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """以工件语义坐标系生成四个正交的朝向初值。"""
    turns = (
        ("fpfh", np.eye(3)),
        ("flip_about_object_x", np.diag((1.0, -1.0, -1.0))),
        ("flip_about_object_y", np.diag((-1.0, 1.0, -1.0))),
        ("flip_about_object_z", np.diag((-1.0, -1.0, 1.0))),
    )
    inverse = np.linalg.inv(stl_to_object)
    result = []
    for name, turn in turns:
        candidate = initial.copy()
        candidate[:3, :3] = initial[:3, :3] @ stl_to_object[:3, :3] @ turn @ inverse[:3, :3]
        result.append((name, candidate))
    return result


def reverse_full_6d_trials(source: np.ndarray, observed: np.ndarray, mask: np.ndarray, K: np.ndarray,
                           table_plane: np.ndarray, stl_to_object: np.ndarray, *, global_seeds: int,
                           ransac_iterations: int, voxel_m: float, icp_iterations: int,
                           max_corr_m: float, workers: int, minimum_reverse_alignment: float,
                           max_table_penetration_m: float) -> list[dict]:
    """返回通过反面、mask 原点和桌面穿透约束的全部 ICP 候选。"""
    source_features = fpfh_features(source, voxel_m)
    target_features = fpfh_features(observed, voxel_m)
    table_plane = np.asarray(table_plane, dtype=float)
    table_normal = normalize(table_plane[:3])
    center = np.median(observed, axis=0)

    def refine(item: tuple[str, np.ndarray]) -> tuple[str, np.ndarray, float, float]:
        name, candidate = item
        candidate = candidate.copy()
        candidate[:3, 3] = center
        T, rmse, fitness = full_6d_point_to_point_icp(source, observed, candidate, icp_iterations, max_corr_m, workers)
        return name, T, rmse, fitness

    trials: list[dict] = []
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="reverse-icp") as executor:
        for seed in range(global_seeds):
            try:
                initial = fpfh_ransac_initial(source_features, target_features, voxel_m, ransac_iterations, seed)
            except RuntimeError:
                continue
            for name, T, rmse, fitness in executor.map(refine, semantic_half_turns(initial, stl_to_object)):
                stl_z_dot_table_normal = float(T[:3, 2] @ table_normal)
                # Confirmed reverse tabletop pose: STL/documented object +Z points into the table.
                reverse_alignment = -stl_z_dot_table_normal
                point = T[:3, 3]
                if point[2] <= 1e-6:
                    continue
                pixel = K @ point
                x, y = np.rint(pixel[:2] / pixel[2]).astype(int)
                origin_in_mask = 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])
                table_height = transform(T, source) @ table_normal + table_plane[3]
                penetration_m = float(max(0.0, -np.min(table_height)))
                if reverse_alignment < minimum_reverse_alignment or not origin_in_mask or penetration_m > max_table_penetration_m:
                    continue
                trials.append({
                    "seed_index": seed,
                    "initialization": f"fpfh_seed_{seed}/{name}",
                    "T": T,
                    "rmse_m": rmse,
                    "fitness": fitness,
                    "stl_z_dot_table_normal": stl_z_dot_table_normal,
                    "reverse_alignment": reverse_alignment,
                    "origin_pixel_xy": [int(x), int(y)],
                    "table_penetration_m": penetration_m,
                })
    return trials
