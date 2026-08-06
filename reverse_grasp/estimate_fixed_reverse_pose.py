#!/usr/bin/env python3
"""以已确认的桌面反面假设，对 SAM2 mask 做 workpiece_2 的 CAD/ICP 位姿估计。

输入必须是本目录完整视觉链路已经生成的 ``capture`` 与 ``sam2_masks``。本工具
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
ROOT = HERE.parent
VISION_ROOT = ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/implementation"
STL = ROOT / "tool/workpiece_2/workpiece_2.stl"
FRAME = "workpiece2_top_view"
R_OBJECT_STL = np.diag([-1.0, -1.0, 1.0])
# With this directory's orientation config, the classifier's "reverse" hypothesis
# is STL -Z towards the camera.  Therefore documented object +Z (= STL +Z) points
# into the table, which is the confirmed tabletop reverse.
TABLE_REVERSE_SOURCE_SIDE = "reverse"
QUALITY_GATE = HERE / "grasp/execution_quality_gate.json"


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
                        help="默认 <capture-dir>/sam2_masks")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认 <capture-dir>/fixed_reverse_pose")
    parser.add_argument("--mask-index", type=int, default=None,
                        help="仅估计指定的 SAM mask；默认评估全部 mask 并自动选择最佳者")
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


def project(K: np.ndarray, point: np.ndarray) -> tuple[int, int] | None:
    if point[2] <= 1e-6:
        return None
    uvw = K @ point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


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
    cv2.putText(image, "Fixed reverse ICP | X:red Y:green Z:blue", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "Fixed reverse ICP | X:red Y:green Z:blue", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
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
    masks_dir = args.masks_dir.expanduser().resolve() if args.masks_dir else capture / "sam2_masks"
    output = inside_here(args.output_dir) if args.output_dir else capture / "fixed_reverse_pose"
    output = inside_here(output)
    if not capture.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("capture-dir 或 sam2_masks 不存在")
    for path in (capture / "metadata.json", capture / "rgb_000.png", capture / "depth_aligned_000.png", masks_dir / "masks.npz"):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入: {path}")
    output.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(VISION_ROOT))
    import sam2_icp_grasp as icp
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
    table_n = icp.table_normal(all_scene)
    orientation = json.loads((HERE / "workpiece_2_orientation_config.json").read_text(encoding="utf-8"))
    quality_gate = json.loads(QUALITY_GATE.read_text(encoding="utf-8"))
    cad_up = normalize(np.asarray(orientation["positive_face_normal_stl"], dtype=float))
    # In the confirmed tabletop reverse, STL -Z is the visible/camera-facing
    # surface. Weight that asymmetric visible face when resolving the four yaw
    # hypotheses; ICP RMSE alone can select a 90/180-degree wrong yaw.
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
        reverse_starts = [item for item in icp.face_initials(cad_stl, points, table_n, cad_up) if item[0] == TABLE_REVERSE_SOURCE_SIDE]
        trials = []
        for yaw_index, (_, initial) in enumerate(reverse_starts):
            T, rmse, fitness = icp.constrained_icp(cad_stl, points, initial, table_n)
            render_score, render = icp.depth_score(T, cad_render, visible_feature, depth, mask, K, scale)
            trials.append({"yaw_index": yaw_index, "T": T, "rmse_m": float(rmse), "fitness": float(fitness),
                           "render_score_m": float(render_score), "render": render})
        if not trials:
            continue
        fixed_yaw = int(orientation["table_reverse_yaw_index"])
        if not 0 <= fixed_yaw < len(trials):
            raise RuntimeError(f"table_reverse_yaw_index 超出范围: {fixed_yaw}")
        chosen = trials[fixed_yaw]
        yaw_selection = "operator_confirmed_fixed_table_reverse_yaw"
        T_stl, rmse, fitness = chosen["T"], chosen["rmse_m"], chosen["fitness"]
        # p_object = R_object_stl p_stl; therefore p_camera = T_camera_stl R_object_stl^T p_object.
        T_object = np.eye(4)
        T_object[:3, :3] = T_stl[:3, :3] @ R_OBJECT_STL.T
        T_object[:3, 3] = T_stl[:3, 3]
        candidates.append({"mask_index": index, "T_camera_stl": T_stl, "T_camera_object": T_object,
                           "icp_rmse_m": float(rmse), "icp_fitness": float(fitness),
                           "render_score_m": chosen["render_score_m"], "match_fraction": float(chosen["render"].get("match_fraction", 0.0)),
                           "quality_gate_passed": trial_passes_quality_gate(chosen, quality_gate), "yaw_selection": yaw_selection,
                           "yaw_index": int(chosen["yaw_index"]), "yaw_trials": trials, "points": points})
    if not candidates:
        raise RuntimeError("没有具有足够深度点的 SAM2 工件候选")
    passing_candidates = [item for item in candidates if item["quality_gate_passed"]]
    best = max(
        passing_candidates or candidates,
        key=lambda item: (
            item["match_fraction"], item["icp_fitness"], -item["icp_rmse_m"], -item["render_score_m"],
        ),
    )
    draw_axes(rgb, K, best["T_camera_object"], output / "fixed_reverse_axes_overlay.png")
    yaw_overlay_paths = []
    for trial in best["yaw_trials"]:
        T_object_trial = np.eye(4)
        T_object_trial[:3, :3] = trial["T"][:3, :3] @ R_OBJECT_STL.T
        T_object_trial[:3, 3] = trial["T"][:3, 3]
        path = output / f"yaw_{trial['yaw_index']}_cad_axes_overlay.png"
        draw_cad_axes_overlay(
            rgb, K, cad_object, T_object_trial, path,
            f"table reverse yaw {trial['yaw_index']} | CAD:orange X:red Y:green Z:blue",
        )
        yaw_overlay_paths.append(str(path))
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
        "icp": {"rmse_m": best["icp_rmse_m"], "fitness": best["icp_fitness"],
                "render_score_m": best["render_score_m"], "yaw_starts": 4,
                "match_fraction": best["match_fraction"], "quality_gate_passed": best["quality_gate_passed"],
                "yaw_selection": best["yaw_selection"], "yaw_index": best["yaw_index"],
                "yaw_basis": "Operator-confirmed fixed tabletop-reverse yaw 2; ICP scores validate this pose but cannot change documented object axes."},
        "yaw_trials": [{key: value for key, value in trial.items() if key != "T"} for trial in best["yaw_trials"]],
        "artifacts": {"axes_overlay": str(output / "fixed_reverse_axes_overlay.png"),
                      "pose_3d": str(output / "fixed_reverse_pose_3d.png"),
                      "yaw_cad_axes_overlays": yaw_overlay_paths},
        "created_at": time.time(),
    }
    (output / "fixed_reverse_pose.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"FIXED_REVERSE_POSE={output / 'fixed_reverse_pose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
