"""SAM2 提示分割和头部相机 STL-点云 ICP 的共用实现。"""

from __future__ import annotations

import http.client
import io
from pathlib import Path
import socket
from urllib.parse import urlparse

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from tools.inspection_support.registration import make_pca_initial_guesses


def prompt_segment(image: np.ndarray, box: list[int], positives: list[list[int]],
                   negatives: list[list[int]], sam2_url: str) -> np.ndarray:
    """调用驻留 SAM2 服务，返回与输入 RGB 同尺寸的二值 mask。"""
    if not positives:
        raise ValueError("SAM2分割至少需要一个正点")
    points = np.asarray(positives + negatives, dtype=np.int32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    payload = io.BytesIO()
    np.savez_compressed(payload, image_bgr=image, points=points, labels=labels,
                        box=np.asarray(box, dtype=np.int32))
    endpoint = urlparse(sam2_url)
    if endpoint.scheme == "unix":
        if not endpoint.path:
            raise ValueError(f"SAM2 Unix Socket 地址无路径: {sam2_url!r}")

        class UnixConnection(http.client.HTTPConnection):
            def connect(self) -> None:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(endpoint.path)

        connection = UnixConnection("sam2-service", timeout=90.0)
        request_path = "/v1/prompt-segment-sam2"
    elif endpoint.scheme in {"http", "https"} and endpoint.hostname:
        klass = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
        connection = klass(endpoint.hostname, port=endpoint.port, timeout=90.0)
        request_path = (endpoint.path or "/") + (f"?{endpoint.query}" if endpoint.query else "")
    else:
        raise ValueError(f"无效 SAM2 地址: {sam2_url!r}")
    try:
        connection.request("POST", request_path, body=payload.getvalue(),
                           headers={"Content-Type": "application/x-sam2-prompt-npz"})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM2返回HTTP {response.status}: {body.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return (archive["mask"] > 0).astype(np.uint8) * 255


def capture_view(camera, cfg: dict, name: str, output: Path, sam2_url: str,
                 auto_gripper_prompt: bool):
    """D435 interactive ROI/positive/negative-point annotation entry."""
    # Reuse the established OpenCV annotator.  Keep this branch distinct from
    # capture_auto_view: callers selecting ``interactive`` must never silently
    # fall back to fixed automatic prompts.
    from tools.inspection_support.legacy_inspect.sam2_icp_runtime import (
        capture_view as capture_interactive_view,
    )

    return capture_interactive_view(
        camera, cfg, name, output, sam2_url, auto_gripper_prompt
    )


def _projected_mask_overlap(cad: np.ndarray, transform: np.ndarray, mask: np.ndarray,
                            intrinsics: np.ndarray) -> float:
    points = (transform[:3, :3] @ np.asarray(cad, dtype=np.float64).T).T + transform[:3, 3]
    valid = points[:, 2] > 1e-4
    projected = (np.asarray(intrinsics, dtype=np.float64) @ points[valid].T).T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
    height, width = mask.shape[:2]
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    hits = int(np.count_nonzero(np.asarray(mask, dtype=bool)[pixels[inside, 1], pixels[inside, 0]]))
    return float(hits / max(1, len(cad)))


def icp_pose(cad: np.ndarray, observed: np.ndarray, cfg: dict, *,
             selection_mask: np.ndarray | None = None,
             camera_intrinsics: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """保留原流程：先 Point-to-plane，未通过质量门限再 Point-to-point。"""
    cad_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
    obs_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    radius = float(cfg.get("normal_radius_m", float(cfg["observed_voxel_m"]) * 4.0))
    for cloud in (cad_cloud, obs_cloud):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    attempts, all_results, global_best = [], [], None
    methods = [
        ("point_to_plane", o3d.pipelines.registration.TransformationEstimationPointToPlane()),
        ("point_to_point", o3d.pipelines.registration.TransformationEstimationPointToPoint()),
    ]
    quality_gate = bool(cfg.get("quality_gate_enabled", True))
    for method, estimator in methods:
        best = None
        for init_name, initial in make_pca_initial_guesses(cad, observed):
            result = o3d.pipelines.registration.registration_icp(
                cad_cloud, obs_cloud, float(cfg["icp_max_corr_m"]), initial, estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(cfg["icp_max_iters"])),
            )
            item = {"method": method, "initialization": init_name,
                    "fitness": float(result.fitness), "rmse_m": float(result.inlier_rmse)}
            transform = np.asarray(result.transformation)
            if selection_mask is not None and camera_intrinsics is not None:
                item["mask_overlap_ratio"] = _projected_mask_overlap(cad, transform, selection_mask, camera_intrinsics)
            attempts.append(item)
            all_results.append((item, transform))
            if best is None or (item["fitness"], -item["rmse_m"]) > (best[0]["fitness"], -best[0]["rmse_m"]):
                best = (item, transform)
        if best is not None and (global_best is None or (best[0]["fitness"], -best[0]["rmse_m"]) > (global_best[0]["fitness"], -global_best[0]["rmse_m"])):
            global_best = best
        passed = bool(best and best[0]["fitness"] >= float(cfg["quality_min_fitness"])
                      and best[0]["rmse_m"] <= float(cfg["quality_max_rmse_m"]))
        if quality_gate and passed:
            return best[1], {**best[0], "attempts": attempts, "quality_gate_enabled": True,
                             "quality_gate_passed": True}
        if quality_gate:
            print(f"[ICP] {method}未通过质量门限，尝试下一方法")
    if not quality_gate and global_best is not None:
        item, transform = global_best
        if selection_mask is not None and camera_intrinsics is not None:
            max_fitness = max(candidate[0]["fitness"] for candidate in all_results)
            margin = float(cfg.get("icp_selection_fitness_margin", 0.03))
            eligible = [candidate for candidate in all_results if candidate[0]["fitness"] >= max_fitness - margin]
            item, transform = max(eligible, key=lambda candidate: (
                candidate[0].get("mask_overlap_ratio", 0.0), -candidate[0]["rmse_m"], candidate[0]["fitness"]
            ))
            item = {**item, "selection_policy": "max_mask_overlap_within_fitness_margin",
                    "selection_max_fitness": float(max_fitness), "selection_fitness_margin": margin}
        passed = bool(item["fitness"] >= float(cfg["quality_min_fitness"])
                      and item["rmse_m"] <= float(cfg["quality_max_rmse_m"]))
        return transform, {**item, "attempts": attempts, "quality_gate_enabled": False,
                           "quality_gate_passed": passed,
                           "configured_quality_min_fitness": float(cfg["quality_min_fitness"]),
                           "configured_quality_max_rmse_m": float(cfg["quality_max_rmse_m"])}
    raise RuntimeError(f"Point-to-plane和Point-to-point均未通过质量门限: {attempts}")


def save_pose_visualization(run_dir: Path, view_name: str, cad: np.ndarray,
                            T_camera_object: np.ndarray, T_output_object: np.ndarray,
                            output_frame: str, intrinsics: np.ndarray,
                            output_filename: str = "pose_estimation_visualization.png") -> Path:
    """保存 SAM2 mask、STL 投影和物体坐标轴图，供人工核对 ICP。"""
    image = cv2.imread(str(run_dir / f"{view_name}_rgb.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(run_dir / f"{view_name}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise RuntimeError(f"位姿可视化缺少图像或mask: {run_dir}/{view_name}")
    left, right = image.copy(), image.copy()
    selected = mask > 0
    left[selected] = (left[selected] * 0.55 + np.array([0, 0, 255]) * 0.45).astype(np.uint8)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    points = (T_camera_object[:3, :3] @ np.asarray(cad, dtype=np.float64).T).T + T_camera_object[:3, 3]
    valid = points[:, 2] > 1e-4
    projected = (K @ points[valid].T).T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
    height, width = right.shape[:2]
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    for u, v in pixels[inside][::max(1, int(np.count_nonzero(inside) // 4000))]:
        cv2.circle(right, (int(u), int(v)), 1, (0, 255, 255), -1)
    extent = np.ptp(np.asarray(cad, dtype=np.float64), axis=0)
    length = max(0.03, min(0.12, float(np.max(extent)) * 0.55))
    axes = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=float)
    axes_cam = (T_camera_object[:3, :3] @ axes.T).T + T_camera_object[:3, 3]
    axes_px = np.rint((K @ axes_cam.T).T[:, :2] / axes_cam[:, 2:3]).astype(np.int32)
    if np.all(axes_cam[:, 2] > 1e-4):
        for endpoint, color, label in zip(axes_px[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 255)], ["+X", "+Y", "FRONT(+Z)"]):
            cv2.arrowedLine(right, tuple(axes_px[0]), tuple(endpoint), color, 4, tipLength=0.18)
            cv2.putText(right, label, tuple(endpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    path = run_dir / output_filename
    if not cv2.imwrite(str(path), np.concatenate([left, right], axis=1)):
        raise RuntimeError(f"位姿可视化保存失败: {path}")
    return path
