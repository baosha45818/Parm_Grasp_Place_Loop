"""Shared, solver-only ICP primitives for the grasp/place modules.

This module deliberately does not know about SAM masks, CAD face semantics,
grasp templates, or robot motion.  Callers own those business decisions and
use the returned transforms and quality metrics.
"""

from __future__ import annotations

import itertools

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def pca_initial_guesses(source: np.ndarray, target: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return proper-rotation PCA alignment hypotheses plus a centroid seed."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    def frame(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0)
        covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
        _, vectors = np.linalg.eigh(covariance)
        rotation = vectors[:, ::-1]
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1.0
        return rotation

    source_frame, target_frame = frame(source), frame(target)
    guesses: list[tuple[str, np.ndarray]] = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        sign_matrix = np.diag(signs)
        if np.linalg.det(sign_matrix) < 0:
            continue
        rotation = target_frame @ sign_matrix @ source_frame.T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = target.mean(axis=0) - rotation @ source.mean(axis=0)
        guesses.append((f"pca_flip_{int(signs[0]):+d}{int(signs[1]):+d}{int(signs[2]):+d}", transform))
    centroid = np.eye(4)
    centroid[:3, 3] = target.mean(axis=0) - source.mean(axis=0)
    guesses.append(("centroid_identity", centroid))
    return guesses


def solve_multi_hypothesis_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    initial_guesses: list[tuple[str, np.ndarray]] | None = None,
    max_corr_m: float = 0.008,
    max_iterations: int = 60,
    normal_radius_m: float = 0.012,
    methods: tuple[str, ...] = ("point_to_plane", "point_to_point"),
) -> list[tuple[np.ndarray, dict[str, float | str]]]:
    """Run Open3D ICP for every requested method and initial pose."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    for cloud in (source_cloud, target_cloud):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_m, max_nn=40)
        )
    estimators = {
        "point_to_plane": o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        "point_to_point": o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    }
    guesses = initial_guesses or pca_initial_guesses(source, target)
    results: list[tuple[np.ndarray, dict[str, float | str]]] = []
    for method in methods:
        if method not in estimators:
            raise ValueError(f"unsupported ICP method: {method}")
        for initialization, initial in guesses:
            result = o3d.pipelines.registration.registration_icp(
                source_cloud, target_cloud, float(max_corr_m), np.asarray(initial, dtype=np.float64),
                estimators[method],
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iterations)),
            )
            results.append((
                np.asarray(result.transformation, dtype=np.float64),
                {"method": method, "initialization": initialization,
                 "fitness": float(result.fitness), "rmse_m": float(result.inlier_rmse)},
            ))
    return results


def full_6d_point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    max_iterations: int = 45,
    max_corr_m: float = 0.009,
) -> tuple[np.ndarray, float, float]:
    """Refine a pose with unrestricted 6DoF point-to-plane ICP.

    Target normals are estimated from the observed point cloud.  This solver
    optimizes all three rotations and translations; it does not apply any
    tabletop, yaw, or face-direction constraint.
    """
    source_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(source, dtype=np.float64))
    )
    target_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(target, dtype=np.float64))
    )
    normal_radius_m = max(2.0 * float(max_corr_m), 0.008)
    target_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_m, max_nn=40)
    )
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        float(max_corr_m),
        np.asarray(initial, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=float(max_corr_m) * 0.5)
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iterations)
        ),
    )
    return (
        np.asarray(result.transformation, dtype=np.float64),
        float(result.inlier_rmse),
        float(result.fitness),
    )


def tabletop_constrained_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    table_normal: np.ndarray,
    *,
    max_iterations: int = 45,
    max_corr_m: float = 0.009,
) -> tuple[np.ndarray, float, float]:
    """Rigid ICP restricted to translation and yaw about ``table_normal``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    normal = _normalize(table_normal)
    transform = np.asarray(initial, dtype=np.float64).copy()
    tree = cKDTree(target)
    basis_x, basis_y = _planar_basis(target, normal)
    for _ in range(int(max_iterations)):
        transformed = _transform_points(transform, source)
        distances, nearest = tree.query(transformed, k=1, workers=-1)
        keep = distances < float(max_corr_m)
        if np.count_nonzero(keep) < 30:
            keep[np.argsort(distances)[: min(250, len(distances))]] = True
        keep &= distances <= np.quantile(distances[keep], 0.82)
        matched_source, matched_target = transformed[keep], target[nearest[keep]]
        source_center, target_center = matched_source.mean(axis=0), matched_target.mean(axis=0)
        source_2d = np.column_stack(((matched_source - source_center) @ basis_x,
                                    (matched_source - source_center) @ basis_y))
        target_2d = np.column_stack(((matched_target - target_center) @ basis_x,
                                    (matched_target - target_center) @ basis_y))
        covariance = source_2d.T @ target_2d
        yaw = float(np.arctan2(covariance[0, 1] - covariance[1, 0],
                               covariance[0, 0] + covariance[1, 1]))
        rotation = _axis_angle(normal, yaw)
        translation = target_center - rotation @ source_center
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rotation, translation
        transform = delta @ transform
        if abs(yaw) < 2e-5 and np.linalg.norm(translation) < 2e-6:
            break
    final_distances, _ = tree.query(_transform_points(transform, source), k=1, workers=-1)
    inlier = final_distances < float(max_corr_m)
    rmse = float(np.sqrt(np.mean(final_distances[inlier] ** 2))) if np.any(inlier) else float("inf")
    return transform, rmse, float(np.mean(inlier))


def tabletop_constrained_point_to_plane_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    table_normal: np.ndarray,
    *,
    max_iterations: int = 45,
    max_corr_m: float = 0.009,
) -> tuple[np.ndarray, float, float]:
    """Point-to-plane ICP restricted to translation and tabletop yaw.

    The target normals come from the observed cloud.  Every Gauss-Newton step
    solves four variables: one rotation about ``table_normal`` and XYZ
    translation.  Roll and pitch therefore cannot drift into the opposite
    face hypothesis.  The returned RMSE remains the conventional Euclidean
    correspondence RMSE so existing quality gates keep the same units.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    normal = _normalize(table_normal)
    transform = np.asarray(initial, dtype=np.float64).copy()
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    normal_radius_m = max(2.0 * float(max_corr_m), 0.008)
    target_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_m, max_nn=40
        )
    )
    target_normals = np.asarray(target_cloud.normals, dtype=np.float64)
    normal_lengths = np.linalg.norm(target_normals, axis=1)
    valid_normals = np.isfinite(target_normals).all(axis=1) & (normal_lengths > 1e-8)
    target_normals[valid_normals] /= np.maximum(
        normal_lengths[valid_normals, None],
        1e-12,
    )
    tree = cKDTree(target)
    robust_scale = max(float(max_corr_m) * 0.5, 1e-6)

    for _ in range(int(max_iterations)):
        transformed = _transform_points(transform, source)
        distances, nearest = tree.query(transformed, k=1, workers=-1)
        keep = (distances < float(max_corr_m)) & valid_normals[nearest]
        if np.count_nonzero(keep) < 30:
            closest = np.argsort(distances)[: min(250, len(distances))]
            keep[closest] = valid_normals[nearest[closest]]
        kept_distances = distances[keep]
        if len(kept_distances) < 4:
            break
        cutoff = np.quantile(kept_distances, 0.82)
        keep &= distances <= cutoff
        points = transformed[keep]
        matches = target[nearest[keep]]
        normals = target_normals[nearest[keep]]
        residual = np.einsum("ij,ij->i", normals, points - matches)

        # Tukey biweight suppresses depth-edge and segmentation outliers.
        ratio = residual / robust_scale
        weights = np.square(1.0 - np.square(ratio))
        weights[np.abs(ratio) >= 1.0] = 0.0
        if np.count_nonzero(weights > 0.0) < 20:
            weights = np.ones_like(residual)

        yaw_jacobian = np.einsum(
            "ij,ij->i", normals, np.cross(normal, points)
        )
        jacobian = np.column_stack((yaw_jacobian, normals))
        weighted_jacobian = jacobian * np.sqrt(weights)[:, None]
        weighted_rhs = -residual * np.sqrt(weights)
        # Mild damping prevents an under-observed flat face from producing a
        # large unconstrained tangent update while preserving point-plane ICP.
        damping = np.diag((1e-7, 1e-6, 1e-6, 1e-6))
        normal_matrix = weighted_jacobian.T @ weighted_jacobian + damping
        rhs = weighted_jacobian.T @ weighted_rhs
        try:
            step = np.linalg.solve(normal_matrix, rhs)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(weighted_jacobian, weighted_rhs, rcond=None)[0]

        yaw = float(np.clip(step[0], -np.deg2rad(5.0), np.deg2rad(5.0)))
        translation = np.clip(np.asarray(step[1:4]), -0.005, 0.005)
        rotation = _axis_angle(normal, yaw)
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = translation
        transform = delta @ transform
        if abs(yaw) < 2e-5 and np.linalg.norm(translation) < 2e-6:
            break

    final_distances, _ = tree.query(
        _transform_points(transform, source), k=1, workers=-1
    )
    inlier = final_distances < float(max_corr_m)
    rmse = (
        float(np.sqrt(np.mean(final_distances[inlier] ** 2)))
        if np.any(inlier) else float("inf")
    )
    return transform, rmse, float(np.mean(inlier))


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _planar_basis(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    centered -= (centered @ normal)[:, None] * normal
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    axis_x = _normalize(vectors[:, -1] - normal * np.dot(vectors[:, -1], normal))
    return axis_x, _normalize(np.cross(normal, axis_x))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = _normalize(axis)
    cosine, sine, remainder = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [cosine + x*x*remainder, x*y*remainder-z*sine, x*z*remainder+y*sine],
        [y*x*remainder+z*sine, cosine + y*y*remainder, y*z*remainder-x*sine],
        [z*x*remainder-y*sine, z*y*remainder+x*sine, cosine + z*z*remainder],
    ])
