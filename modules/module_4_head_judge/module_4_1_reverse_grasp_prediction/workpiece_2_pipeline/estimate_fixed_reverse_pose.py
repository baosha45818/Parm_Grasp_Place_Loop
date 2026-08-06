#!/usr/bin/env python3
"""以已确认的桌面反面假设，对 SAM2 完整 mask 做 workpiece_2 的 CAD/ICP 位姿估计。

输入必须是本目录完整视觉链路已经生成的 ``capture`` 与 ``01_sam2_mask``。本工具
固定使用工件实际落桌的反面（STL +Z 朝桌面、STL -Z 朝相机）假设。每次会在该反面
假设的全部 yaw 初值中动态选择当前 RGB-D 质量最佳的结果，不假定桌面上的工件 yaw 固定；
它输出工件正式坐标系 ``workpiece2_top_view`` 的相机位姿和 RGB/3D 坐标轴可视化。
不控制任何硬件，也不写输入目录外的文件。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
LOOP_ROOT = HERE.parents[3]
VISION_ROOT = HERE / "vision_support"
STL = LOOP_ROOT / "tool/workpiece_2/workpiece_2.stl"
FRAME = "workpiece2_top_view"
R_OBJECT_STL = np.diag([-1.0, -1.0, 1.0])
# With this directory's orientation config, the classifier's "reverse" hypothesis
# is STL -Z towards the camera.  Therefore documented object +Z (= STL +Z) points
# into the table, which is the confirmed tabletop reverse.
TABLE_REVERSE_SOURCE_SIDE = "reverse"
QUALITY_GATE = HERE / "config/execution_quality_gate.json"


def inside_here(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须位于 {HERE}: {resolved}") from exc
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True,
                        help="reverse_grasp/runs/.../capture")
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="默认由完整流程传入 01_sam2_mask；也可显式传入历史 masks 目录")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认 <capture-dir>/fixed_reverse_pose")
    parser.add_argument("--mask-index", type=int, default=None,
                        help="仅估计指定的几何 mask；默认评估全部 mask 并自动选择最佳者")
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def trial_passes_quality_gate(trial: dict, gate: dict) -> bool:
    """与抓取候选的执行门限保持一致，避免选到必然会被后续拒绝的 yaw。"""
    render = trial.get("render") or {}
    return (
        float(trial["fitness"]) >= float(gate["minimum_icp_fitness"])
        and float(trial["rmse_m"]) <= float(gate["maximum_icp_rmse_m"])
        and float(render.get("match_fraction", 0.0)) >= float(gate["minimum_depth_match_fraction"])
    )


def trial_quality_key(trial: dict) -> tuple[float, float, float, float]:
    """质量从高到低：深度匹配率、ICP fitness、低 RMSE、低渲染误差。"""
    render = trial.get("render") or {}
    return (
        float(render.get("match_fraction", float("-inf"))),
        float(trial["fitness"]),
        -float(trial["rmse_m"]),
        -float(trial["render_score_m"]),
    )


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ np.asarray(points).T).T + T[:3, 3]


def relative_table_yaw_deg(rotation: np.ndarray, reference_rotation: np.ndarray, table_normal: np.ndarray) -> float:
    """Signed yaw relative to the documented yaw-index-2 reference."""
    relative = np.asarray(rotation, dtype=float) @ np.asarray(reference_rotation, dtype=float).T
    skew_vector = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) * .5
    return float(np.rad2deg(np.arctan2(np.dot(normalize(table_normal), skew_vector), (np.trace(relative) - 1.) * .5)))


def project(K: np.ndarray, point: np.ndarray) -> tuple[int, int] | None:
    if point[2] <= 1e-6:
        return None
    uvw = K @ point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def silhouette_refine(T: np.ndarray, cad_render: np.ndarray, mask: np.ndarray, K: np.ndarray,
                     table_n: np.ndarray, observed_points: np.ndarray,
                     yaw_reference_rotation: np.ndarray, yaw_min_deg: float,
                     yaw_max_deg: float) -> tuple[np.ndarray, dict]:
    """Refine ICP's planar translation/yaw by CAD-to-RGB-mask silhouette overlap.

    ICP is vulnerable to a few remaining depth outliers on reflective metal.
    This refinement keeps the confirmed reverse face, but searches only small
    in-plane X/Y and yaw corrections and scores both CAD->mask and mask->CAD
    pixel distances.
    """
    foreground = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    dt_to_foreground = cv2.distanceTransform((1 - foreground).astype(np.uint8), cv2.DIST_L2, 3)
    yy, xx = np.nonzero(foreground)
    observed_uv = np.column_stack((xx, yy))
    if len(observed_uv) > 5000:
        observed_uv = observed_uv[np.linspace(0, len(observed_uv) - 1, 5000).astype(int)]
    rng = np.random.default_rng(23)
    sample = cad_render[rng.choice(len(cad_render), min(7000, len(cad_render)), replace=False)]
    centered = observed_points - np.mean(observed_points, axis=0)
    centered -= (centered @ table_n)[:, None] * table_n
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    ex = normalize(vectors[:, -1] - table_n * np.dot(vectors[:, -1], table_n))
    ey = normalize(np.cross(table_n, ex))
    h, w = foreground.shape

    def score(candidate: np.ndarray) -> float:
        points = transform(candidate, sample)
        good = points[:, 2] > .1
        points = points[good]
        uv = np.rint(np.column_stack((K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
                                      K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]))).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if int(np.count_nonzero(inside)) < 600:
            return float("inf")
        uv = uv[inside]
        forward = float(np.mean(np.minimum(dt_to_foreground[uv[:, 1], uv[:, 0]], 35.0)))
        render = np.zeros_like(foreground)
        render[uv[:, 1], uv[:, 0]] = 1
        render = cv2.dilate(render, np.ones((5, 5), np.uint8), iterations=1)
        dt_to_render = cv2.distanceTransform((1 - render).astype(np.uint8), cv2.DIST_L2, 3)
        reverse = float(np.mean(np.minimum(dt_to_render[observed_uv[:, 1], observed_uv[:, 0]], 35.0)))
        return forward + reverse

    def candidate(dx: float, dy: float, yaw_deg: float) -> np.ndarray:
        base_yaw_deg = relative_table_yaw_deg(T[:3, :3], yaw_reference_rotation, table_n)
        if not yaw_min_deg <= base_yaw_deg + yaw_deg <= yaw_max_deg:
            return None
        x, y, z = table_n; angle = np.deg2rad(yaw_deg); c, s = np.cos(angle), np.sin(angle); C = 1. - c
        Q = np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                      [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                      [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
        result = T.copy(); result[:3, :3] = Q @ T[:3, :3]
        result[:3, 3] = T[:3, 3] + ex * dx + ey * dy
        return result

    best = (score(T), 0.0, 0.0, 0.0, T)
    for step_m, step_deg, span_m, span_deg in ((.005, 5.0, .030, 25.0), (.0015, 1.5, .006, 6.0)):
        _, cx, cy, ca, _ = best
        for dx in np.arange(cx - span_m, cx + span_m + 1e-9, step_m):
            for dy in np.arange(cy - span_m, cy + span_m + 1e-9, step_m):
                for angle in np.arange(ca - span_deg, ca + span_deg + 1e-9, step_deg):
                    pose = candidate(float(dx), float(dy), float(angle))
                    value = float("inf") if pose is None else score(pose)
                    if value < best[0]:
                        best = (value, float(dx), float(dy), float(angle), pose)
    return best[4], {"method": "cad_rgb_mask_symmetric_chamfer", "score_px": best[0],
                     "planar_dx_m": best[1], "planar_dy_m": best[2], "yaw_correction_deg": best[3]}


def draw_axes(rgb: np.ndarray, K: np.ndarray, T_camera_object: np.ndarray, path: Path) -> None:
    image = rgb.copy()
    center = T_camera_object[:3, 3]
    origin = project(K, center)
    if origin is None:
        raise RuntimeError("工件原点不在相机前方，无法可视化")
    cv2.drawMarker(image, origin, (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    # BGR: object X red, Y green, Z blue.
    for axis, color, label in ((0, (0, 0, 255), "X"), (1, (0, 255, 0), "Y"), (2, (255, 0, 0), "Z")):
        endpoint = project(K, center + T_camera_object[:3, axis] * .055)
        if endpoint is not None:
            cv2.arrowedLine(image, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=.16)
            cv2.putText(image, label, (endpoint[0] + 5, endpoint[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
    cv2.putText(image, "Reverse full-6DoF ICP | X:red Y:green Z:blue", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "Reverse full-6DoF ICP | X:red Y:green Z:blue", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法写入 {path}")


def draw_cad_axes_overlay(rgb: np.ndarray, K: np.ndarray, cad_object: np.ndarray,
                          T_camera_object: np.ndarray, path: Path, label: str) -> None:
    """Render the actual STL silhouette and documented object axes in the RGB frame."""
    image = rgb.copy()
    points = transform(T_camera_object, cad_object)
    good = points[:, 2] > 1e-6
    uv = np.column_stack((K[0, 0] * points[good, 0] / points[good, 2] + K[0, 2],
                          K[1, 1] * points[good, 1] / points[good, 2] + K[1, 2]))
    canvas = image.copy()
    for x, y in np.rint(uv).astype(int):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(canvas, (x, y), 1, (0, 165, 255), -1, cv2.LINE_AA)
    image = cv2.addWeighted(canvas, .48, image, .52, 0)
    center = T_camera_object[:3, 3]
    origin = project(K, center)
    if origin is None:
        raise RuntimeError("工件原点不在相机前方，无法可视化")
    cv2.drawMarker(image, origin, (0, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    for axis, color, name in ((0, (0, 0, 255), "X"), (1, (0, 255, 0), "Y"), (2, (255, 0, 0), "Z")):
        endpoint = project(K, center + T_camera_object[:3, axis] * .055)
        if endpoint is not None:
            cv2.arrowedLine(image, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=.16)
            cv2.putText(image, name, (endpoint[0] + 5, endpoint[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
    cv2.putText(image, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法写入 {path}")


def draw_3d(scene: np.ndarray, cad_object: np.ndarray, T_camera_object: np.ndarray, path: Path) -> None:
    rng = np.random.default_rng(7)
    sample = scene[rng.choice(len(scene), min(len(scene), 30000), replace=False)]
    cad_camera = transform(T_camera_object, cad_object)
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=.18, c="#808080", alpha=.35, label="observed depth")
    axis.scatter(cad_camera[:, 0], cad_camera[:, 1], cad_camera[:, 2], s=.4, c="#e69f00", alpha=.7, label="workpiece_2 CAD")
    center = T_camera_object[:3, 3]
    for i, color in enumerate(("r", "g", "b")):
        direction = T_camera_object[:3, i]
        axis.quiver(*center, *direction, length=.07, color=color, arrow_length_ratio=.15)
    axis.set_xlabel("camera X (m)")
    axis.set_ylabel("camera Y (m)")
    axis.set_zlabel("camera Z (m)")
    axis.set_title("Fixed tabletop reverse pose | object axes X(red) Y(green) Z(blue)")
    axis.legend(loc="upper right")
    extent_points = np.vstack((scene, cad_camera, center[None, :]))
    lower, upper = extent_points.min(axis=0), extent_points.max(axis=0)
    padding = np.maximum((upper - lower) * .18, .035)
    axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    axis.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
    axis.view_init(28, -72)
    axis.set_box_aspect((1, 1, .55))
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    capture = args.capture_dir.expanduser().resolve()
    if args.masks_dir is not None:
        masks_dir = args.masks_dir.expanduser().resolve()
    else:
        # Fresh runs keep all step outputs beside capture/.  The old capture
        # subdirectory remains a fallback only for archived geometry runs.
        masks_dir = capture.parent / "01_sam2_mask"
        if not masks_dir.is_dir():
            masks_dir = capture / "geometry_masks"
    output = inside_here(args.output_dir) if args.output_dir else capture / "fixed_reverse_pose"
    output = inside_here(output)
    if not capture.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("capture-dir 或 SAM2 masks 目录不存在")
    for path in (capture / "metadata.json", capture / "rgb_000.png", capture / "depth_aligned_000.png", masks_dir / "masks.npz"):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入: {path}")
    output.mkdir(parents=True, exist_ok=False)

    masks_metadata_path = masks_dir / "meta.json"
    masks_metadata = json.loads(masks_metadata_path.read_text(encoding="utf-8")) if masks_metadata_path.is_file() else {}

    sys.path.insert(0, str(VISION_ROOT))
    import sam2_icp_grasp as icp
    from reverse_full6d_icp import fit_table_plane, reverse_full_6d_trials
    from tool.inspection_support.mask_depth import mask_depth_to_points_cam, voxel_downsample_np, remove_radius_outliers_np
    from tool.inspection_support.stl_model import load_stl_triangles_m, sample_surface_points

    meta = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    rgb = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture / "depth_aligned_000.png"), cv2.IMREAD_UNCHANGED)
    intr = meta["color_intrinsics"]
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])
    scale = float(meta["depth_scale_m"])
    masks = np.load(masks_dir / "masks.npz")["masks"]
    if args.mask_index is not None and not 0 <= args.mask_index < len(masks):
        raise ValueError(f"--mask-index={args.mask_index} 不在 [0,{len(masks) - 1}] 内")
    triangles_stl = load_stl_triangles_m(STL)
    cad_stl = sample_surface_points(triangles_stl, 14000, 3)
    cad_render = sample_surface_points(triangles_stl, 90000, 4)
    cad_object = cad_stl @ R_OBJECT_STL.T
    all_scene = mask_depth_to_points_cam(np.ones(depth.shape, dtype=bool), depth, K, scale, .35, 1.2, 1)
    table_plane = fit_table_plane(all_scene)
    table_n = table_plane[:3]
    orientation = json.loads((HERE / "workpiece_2_orientation_config.json").read_text(encoding="utf-8"))
    quality_gate = json.loads(QUALITY_GATE.read_text(encoding="utf-8"))
    stl_to_object = np.eye(4)
    stl_to_object[:3, :3] = R_OBJECT_STL
    # In the confirmed tabletop reverse, STL -Z is the visible/camera-facing
    # surface. It is only a render-score weighting; the actual pose search is
    # unrestricted full 6DoF and accepts STL +Z pointing into the table.
    visible_face_stl = np.array([0.0, 0.0, -1.0])
    visible_feature = cad_render @ visible_face_stl >= np.percentile(cad_render @ visible_face_stl, 88)
    candidates: list[dict] = []
    indices = range(len(masks)) if args.mask_index is None else (args.mask_index,)
    for index in indices:
        raw_mask = masks[index]
        mask = raw_mask.astype(bool)
        points = mask_depth_to_points_cam(mask, depth, K, scale, .35, 1.2, 1)
        points = voxel_downsample_np(remove_radius_outliers_np(points), .003)
        if len(points) < 120:
            continue
        trials = reverse_full_6d_trials(
            cad_stl, points, mask, K, table_plane, stl_to_object,
            global_seeds=int(orientation["reverse_full_6d_global_seeds"]),
            ransac_iterations=int(orientation["reverse_full_6d_ransac_iterations"]),
            voxel_m=float(orientation["reverse_full_6d_global_voxel_m"]),
            icp_iterations=int(orientation["reverse_full_6d_icp_iterations"]),
            max_corr_m=float(orientation["reverse_full_6d_icp_max_correspondence_m"]),
            workers=int(orientation["reverse_full_6d_icp_workers"]),
            minimum_reverse_alignment=float(orientation["reverse_full_6d_minimum_alignment"]),
            max_table_penetration_m=float(orientation["reverse_full_6d_max_table_penetration_m"]),
        )
        for trial in trials:
            T = trial["T"]
            render_score, render = icp.depth_score(T, cad_render, visible_feature, depth, mask, K, scale)
            trial.update({"render_score_m": float(render_score), "render": render})
        if not trials:
            continue
        accepted_trials = [trial for trial in trials if trial_passes_quality_gate(trial, quality_gate)]
        chosen = max(accepted_trials or trials, key=trial_quality_key)
        initialization_selection = (
            "reverse_full_6d_global_candidates_quality_gate_pass"
            if accepted_trials else "reverse_full_6d_global_candidates_no_quality_gate_pass"
        )
        T_stl, rmse, fitness = chosen["T"], chosen["rmse_m"], chosen["fitness"]
        # SAM2 only supplies the observed workpiece mask. Pose remains the
        # global full-6DoF CAD/ICP result; no 2-D silhouette refinement changes it.
        silhouette = {"method": "disabled_after_sam2_full_6d_icp"}
        render_score, render = icp.depth_score(T_stl, cad_render, visible_feature, depth, mask, K, scale)
        # p_object = R_object_stl p_stl; therefore p_camera = T_camera_stl R_object_stl^T p_object.
        T_object = np.eye(4)
        T_object[:3, :3] = T_stl[:3, :3] @ R_OBJECT_STL.T
        T_object[:3, 3] = T_stl[:3, 3]
        candidates.append({"mask_index": index, "T_camera_stl": T_stl, "T_camera_object": T_object,
                           "icp_rmse_m": float(rmse), "icp_fitness": float(fitness),
                           "render_score_m": float(render_score), "match_fraction": float(render.get("match_fraction", 0.0)),
                           "quality_gate_passed": trial_passes_quality_gate(chosen, quality_gate),
                           "initialization_selection": initialization_selection,
                           "initialization": chosen["initialization"],
                           "reverse_alignment": chosen["reverse_alignment"],
                           "stl_z_dot_table_normal": chosen["stl_z_dot_table_normal"],
                           "origin_pixel_xy": chosen["origin_pixel_xy"],
                           "table_penetration_m": chosen["table_penetration_m"],
                           "full_6d_trials": trials, "silhouette_refinement": silhouette, "points": points})
    if not candidates:
        if len(masks) == 0:
            raise RuntimeError("SAM2 没有任何工件掩码，无法进入 ICP")
        raise RuntimeError("SAM2 工件掩码存在，但没有候选达到至少 120 个有效深度点的要求")
    passing_candidates = [item for item in candidates if item["quality_gate_passed"]]
    best = max(
        passing_candidates or candidates,
        key=lambda item: (
            item["match_fraction"], item["icp_fitness"], -item["icp_rmse_m"], -item["render_score_m"],
        ),
    )
    draw_axes(rgb, K, best["T_camera_object"], output / "fixed_reverse_axes_overlay.png")
    initialization_overlay_paths = []
    for trial_index, trial in enumerate(best["full_6d_trials"]):
        T_object_trial = np.eye(4)
        T_object_trial[:3, :3] = trial["T"][:3, :3] @ R_OBJECT_STL.T
        T_object_trial[:3, 3] = trial["T"][:3, 3]
        path = output / f"full6d_{trial_index:03d}_cad_axes_overlay.png"
        draw_cad_axes_overlay(
            rgb, K, cad_object, T_object_trial, path,
            f"reverse full6d {trial['initialization']} | CAD:orange X:red Y:green Z:blue",
        )
        initialization_overlay_paths.append(str(path))
    draw_3d(best["points"], cad_object, best["T_camera_object"], output / "fixed_reverse_pose_3d.png")
    payload = {
        "schema_version": 1,
        "event": "workpiece_2_fixed_table_reverse_pose",
        "face_definition": "Operator-confirmed tabletop reverse: STL +Z (= documented workpiece +Z) points into the table, and STL -Z faces the camera. No score-margin rejection was applied.",
        "legacy_source_hypothesis": TABLE_REVERSE_SOURCE_SIDE,
        "frame_id": "head_d435i_color_optical_frame",
        "object_frame": FRAME,
        "R_object_stl": R_OBJECT_STL.tolist(),
        "mask_index": best["mask_index"],
        "T_camera_object": best["T_camera_object"].round(9).tolist(),
        "object_pose": {"translation_m": best["T_camera_object"][:3, 3].round(9).tolist(), "rotation_matrix": best["T_camera_object"][:3, :3].round(9).tolist()},
        "table_normal_camera": table_n.round(9).tolist(),
        "table_plane_camera": table_plane.round(9).tolist(),
        "input_mask_method": masks_metadata.get("method", "legacy_or_unknown_mask_source"),
        "icp": {"method": "FPFH_RANSAC_initialized_full_6DoF_point_to_point_ICP", "rmse_definition": "RMS Euclidean distance from transformed CAD source points to nearest observed depth points", "rmse_m": best["icp_rmse_m"], "fitness": best["icp_fitness"],
                "render_score_m": best["render_score_m"], "global_seeds": int(orientation["reverse_full_6d_global_seeds"]),
                "initial_hypotheses": int(orientation["reverse_full_6d_global_seeds"]) * 4,
                "match_fraction": best["match_fraction"], "quality_gate_passed": best["quality_gate_passed"],
                "initialization_selection": best["initialization_selection"], "initialization": best["initialization"],
                "reverse_alignment": best["reverse_alignment"], "stl_z_dot_table_normal": best["stl_z_dot_table_normal"],
                "origin_pixel_xy": best["origin_pixel_xy"], "table_penetration_m": best["table_penetration_m"],
                "reverse_constraint": "STL +Z must point into the table: dot(STL +Z, table_normal) <= -minimum_alignment."},
        "silhouette_refinement": best["silhouette_refinement"],
        "full_6d_trials": [{key: value for key, value in trial.items() if key != "T"} for trial in best["full_6d_trials"]],
        "artifacts": {"axes_overlay": str(output / "fixed_reverse_axes_overlay.png"),
                      "pose_3d": str(output / "fixed_reverse_pose_3d.png"),
                      "full_6d_cad_axes_overlays": initialization_overlay_paths},
        "created_at": time.time(),
    }
    (output / "fixed_reverse_pose.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"FIXED_REVERSE_POSE={output / 'fixed_reverse_pose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
