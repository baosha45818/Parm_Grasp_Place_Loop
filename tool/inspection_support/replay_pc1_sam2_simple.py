#!/usr/bin/env python3
"""Replay one-pass PC1/PC2 SAM2 with URDF used only before prompting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_2_left_inspect_apart import standalone_inspect as inspect
from tool.inspection_support.legacy_inspect.remove_pc2_urdf_gripper_depth import (
    URDF_ROOT,
    load_self_surface_points,
    projected_gripper_depth,
)


FALLBACK_K = np.array(
    [[751.8960571289062, 0.0, 632.2559204101562],
     [0.0, 751.3369750976562, 359.5553283691406],
     [0.0, 0.0, 1.0]], dtype=float,
)


def _rendered_depth(result: dict, cfg: dict, depth: np.ndarray, view: str = "pc1") -> np.ndarray:
    registration = result.get("registration") or {}
    K = np.asarray(registration.get("camera_intrinsics") or FALLBACK_K, dtype=float)
    T_camera_object = np.asarray(result["T_camera_object"], dtype=float)
    if "T_body_base_object" in result:
        T_body_object = np.asarray(result["T_body_base_object"], dtype=float)
    else:
        pose = result["object_pose"]
        T_body_object = np.eye(4)
        T_body_object[:3, :3] = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
        T_body_object[:3, 3] = np.asarray(pose["position"], dtype=float)
    T_tool_object = np.asarray(
        result.get("T_tool_object") or result.get("placement", {}).get("T_tool_object"),
        dtype=float,
    )
    T_body_camera = T_body_object @ np.linalg.inv(T_camera_object)
    T_body_tool = T_body_object @ np.linalg.inv(T_tool_object)
    T_camera_tool = np.linalg.inv(T_body_camera) @ T_body_tool
    # Older xiayufei logs do not retain the opening after the successful
    # grasp; render the conservative fully-closed gripper in that case.
    opening = float(result.get("gripper_opening_percent", 0.0) or 0.0)
    active = 0.65 * (1.0 - opening / 100.0)
    joints = cfg["pc_joints_deg"] if view == "pc1" else cfg["pc2_joints_deg"]
    points = load_self_surface_points(URDF_ROOT, active, 200000, joints)
    rendered = projected_gripper_depth(points, T_camera_tool, K, depth.shape)
    projected = np.isfinite(rendered)
    closed = cv2.morphologyEx(projected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    newly_closed = closed & ~projected
    if newly_closed.any():
        _, nearest = ndimage.distance_transform_edt(~projected, return_indices=True)
        rendered = rendered.copy()
        rendered[newly_closed] = rendered[nearest[0][newly_closed], nearest[1][newly_closed]]
    return rendered


def _body_camera_from_result(result: dict) -> np.ndarray:
    """Recover the fixed chest-camera pose from a successful inspection result."""
    T_camera_object = np.asarray(result["T_camera_object"], dtype=float)
    if "T_body_base_object" in result:
        T_body_object = np.asarray(result["T_body_base_object"], dtype=float)
    else:
        pose = result["object_pose"]
        T_body_object = np.eye(4)
        T_body_object[:3, :3] = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
        T_body_object[:3, 3] = np.asarray(pose["position"], dtype=float)
    return T_body_object @ np.linalg.inv(T_camera_object)


def _legacy_capture_rendered_depth(capture: Path, camera_pose_result: Path, depth: np.ndarray) -> np.ndarray:
    """Render URDF depth for a standalone RGB-D capture with recorded robot state."""
    state = json.loads((capture / "robot_state.json").read_text(encoding="utf-8"))
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    if state.get("status") != "ok":
        raise RuntimeError(f"Recorded robot state is unavailable: {state.get('error', 'unknown error')}")
    tool = state["T_body_base_left_hand_transfer_end"]
    T_body_tool = np.eye(4)
    T_body_tool[:3, :3] = Rotation.from_quat(
        [tool["quaternion_xyzw"][axis] for axis in "xyzw"]
    ).as_matrix()
    T_body_tool[:3, 3] = [tool["translation_m"][axis] for axis in "xyz"]
    T_body_camera = _body_camera_from_result(
        json.loads(camera_pose_result.read_text(encoding="utf-8"))
    )
    gripper = state.get("left_gripper_state") or {}
    opening = float(gripper.get("position", 0.0))
    active = 0.65 * (1.0 - opening / 100.0)
    joints = [float(value) for value in state["left_arm_joint_position_deg"]]
    if len(joints) != 7:
        raise RuntimeError("left_arm_joint_position_deg must contain seven values")
    points = load_self_surface_points(URDF_ROOT, active, 200000, joints)
    K = np.asarray(metadata["color_intrinsics"], dtype=float)
    rendered = projected_gripper_depth(points, np.linalg.inv(T_body_camera) @ T_body_tool, K, depth.shape)
    projected = np.isfinite(rendered)
    closed = cv2.morphologyEx(projected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    newly_closed = closed & ~projected
    if newly_closed.any():
        _, nearest = ndimage.distance_transform_edt(~projected, return_indices=True)
        rendered = rendered.copy()
        rendered[newly_closed] = rendered[nearest[0][newly_closed], nearest[1][newly_closed]]
    return rendered


def _config_for_run(run_dir: Path) -> dict:
    """Use a saved run's pose configuration when it is available."""
    cfg = json.loads((ROOT / "modules/module_2_left_inspect_apart/inspect_config.json").read_text())
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved = manifest.get("inspect_config")
        if isinstance(saved, dict):
            cfg.update(saved)
    return cfg


def _workpiece_for_run(run_dir: Path) -> str:
    """Read the recorded workpiece type; older runs are workpiece 1."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return "1"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("arguments", {}).get("workpiece", manifest.get("workpiece", "1"))
    return "2" if str(value) == "2" else "1"


def _expand_candidate_allowing_missing_depth(
    seed: np.ndarray,
    measured: np.ndarray,
    rgb: np.ndarray,
    self_pixels: np.ndarray,
    near_min_m: float,
    near_max_m: float,
    passes: int,
    color_difference: int = 30,
    depth_difference_m: float = 0.015,
) -> np.ndarray:
    """Grow through RGB-coherent zero-depth holes without crossing URDF pixels."""
    grown = seed.astype(bool).copy()
    missing = ~np.isfinite(measured) | (measured <= 0)
    expandable = (
        (missing | ((measured >= near_min_m - 0.030) & (measured <= near_max_m + 0.030)))
        & ~self_pixels
    )
    rgb_i16 = rgb.astype(np.int16)
    height, width = grown.shape
    for _ in range(int(passes)):
        frontier = cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        frontier &= ~grown & expandable
        accepted = np.zeros_like(grown)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
            neighbour_rgb = np.roll(np.roll(rgb_i16, dy, axis=0), dx, axis=1)
            neighbour_depth = np.roll(np.roll(measured, dy, axis=0), dx, axis=1)
            neighbour_missing = np.roll(np.roll(missing, dy, axis=0), dx, axis=1)
            if dy > 0:
                neighbour[:dy] = False
            elif dy < 0:
                neighbour[dy:] = False
            if dx > 0:
                neighbour[:, :dx] = False
            elif dx < 0:
                neighbour[:, dx:] = False
            colour_ok = np.max(np.abs(rgb_i16 - neighbour_rgb), axis=2) <= color_difference
            depth_ok = missing | neighbour_missing | (np.abs(measured - neighbour_depth) <= depth_difference_m)
            accepted |= frontier & neighbour & colour_ok & depth_ok
        if not accepted.any():
            break
        grown |= accepted
    return grown


def _expand_main_mask_rgbd_consistent(
    mask: np.ndarray,
    measured: np.ndarray,
    rgb: np.ndarray,
    self_pixels: np.ndarray,
    depth_support: np.ndarray,
    passes: int = 16,
    colour_difference: int = 30,
    depth_margin_m: float = 0.015,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Grow the final main SAM mask only into RGB-connected, depth-consistent pixels.

    The depth interval is measured where the current main mask overlaps the
    trusted pre-SAM depth support, rather than using the global 0.10--0.28 m
    prompt band.  This is deliberately a local repair: a disconnected depth
    island, or a neighbouring surface at a noticeably different depth,
    cannot be added by this post-process.
    """
    grown = mask.astype(bool).copy()
    trusted_main = grown & depth_support.astype(bool)
    valid_main_depth = measured[trusted_main & np.isfinite(measured) & (measured > 0)]
    if valid_main_depth.size < 20:
        return grown, {"added_pixels": 0, "trusted_main_depth_pixels": int(valid_main_depth.size)}

    # Robust endpoints avoid one or two flying-depth pixels making the allowed
    # range unrealistically broad.  The margin still accommodates ordinary
    # depth noise at the object boundary.
    core_low, core_high = np.quantile(valid_main_depth, [0.05, 0.95])
    depth_low = max(0.0, float(core_low) - depth_margin_m)
    depth_high = float(core_high) + depth_margin_m
    expandable = (
        np.isfinite(measured)
        & (measured > 0)
        & (measured >= depth_low)
        & (measured <= depth_high)
        & ~self_pixels
    )
    rgb_i16 = rgb.astype(np.int16)
    original = grown.copy()
    for _ in range(int(passes)):
        frontier = cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        frontier &= ~grown & expandable
        accepted = np.zeros_like(grown)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
            neighbour_rgb = np.roll(np.roll(rgb_i16, dy, axis=0), dx, axis=1)
            if dy > 0:
                neighbour[:dy] = False
            elif dy < 0:
                neighbour[dy:] = False
            if dx > 0:
                neighbour[:, :dx] = False
            elif dx < 0:
                neighbour[:, dx:] = False
            colour_ok = np.max(np.abs(rgb_i16 - neighbour_rgb), axis=2) <= colour_difference
            accepted |= frontier & neighbour & colour_ok
        if not accepted.any():
            break
        grown |= accepted
    return grown, {
        "added_pixels": int((grown & ~original).sum()),
        "trusted_main_depth_pixels": int(valid_main_depth.size),
        "main_depth_low_m": depth_low,
        "main_depth_high_m": depth_high,
        "passes": int(passes),
        "colour_difference": int(colour_difference),
    }


def _main_positive_cluster(
    points: list[list[int]], max_distance_px: float = 300.0, weights: list[float] | None = None,
) -> list[list[int]]:
    """Keep positives close to the spatial majority, weighted by depth support."""
    if len(points) < 2:
        return points
    xy = np.asarray(points, dtype=np.float32)
    distance_sq = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    inlier = distance_sq <= max_distance_px ** 2
    support = (
        inlier.astype(np.float64) @ np.asarray(weights, dtype=np.float64)
        if weights is not None
        else inlier.sum(axis=1).astype(np.float64)
    )
    # Break ties by selecting the geometric centre of the main prompt group.
    candidates = np.flatnonzero(support == support.max())
    centre = int(candidates[np.argmin(distance_sq[candidates].sum(axis=1))])
    keep = distance_sq[centre] <= max_distance_px ** 2
    return [point for point, accepted in zip(points, keep, strict=True) if accepted]


def _pc1_workpiece2_expanded_prompt(
    measured: np.ndarray,
    self_pixels: np.ndarray,
    rgb: np.ndarray,
    runtime: SimpleNamespace,
) -> dict | None:
    """Build a conservative SAM2 prompt from the expanded workpiece-2 seed."""
    brightness = np.max(rgb, axis=2)
    seed = (
        (measured >= runtime.pc1_near_min_m)
        & (measured <= runtime.pc1_near_max_m)
        & ~self_pixels
        & (brightness >= 55)
        & (brightness <= 130)
    )
    if not seed.any():
        return None
    expanded = _expand_candidate_allowing_missing_depth(
        seed, measured, rgb, self_pixels,
        runtime.pc1_near_min_m, runtime.pc1_near_max_m,
        passes=10,
    )
    # This is a deliberate hard RGB rejection after expansion: unlike the
    # previous context-only expansion preview, these pixels must not reach
    # ROI construction or SAM2.  In particular, max(R,G,B)<55 is a black
    # gripper/background cue and is deleted before connected components and
    # prompts are built.
    expanded &= (brightness >= 55) & (brightness <= 130)
    # Ignore salt-and-pepper depth islands.  200 px is approximately a
    # 14x14-px visibly discernible region at the PC1 image scale.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        expanded.astype(np.uint8), connectivity=8
    )
    visible_labels = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 200) + 1
    expanded = np.isin(labels, visible_labels)
    if not expanded.any():
        return None

    # Keep every visible island in the candidate preview.  Prompt association
    # is intentionally handled below, so thin genuine workpiece lobes are not
    # thrown away merely because URDF removal disconnected them.
    all_candidate = expanded.copy()
    valid_depth = all_candidate & (measured > 0)
    interior_distance = cv2.distanceTransform(all_candidate.astype(np.uint8), cv2.DIST_L2, 5)
    self_distance = cv2.distanceTransform((~self_pixels).astype(np.uint8), cv2.DIST_L2, 5)
    negatives = inspect._pc1_fixed_negative_points(rgb)
    forbidden = np.zeros_like(expanded, dtype=np.uint8)
    for x, y in negatives:
        cv2.circle(forbidden, (x, y), 42, 1, -1)
    # A point must have a measured depth; zero-depth pixels may remain in the
    # candidate/ROI after RGB-D expansion but must never become a primary cue.
    component_count, labels = cv2.connectedComponents(all_candidate.astype(np.uint8), connectivity=8)
    component_points: dict[int, list[int]] = {}
    point_eligible = valid_depth & (brightness >= 50) & (brightness < 100) & ~(forbidden > 0)
    for label in range(1, component_count):
        component = labels == label
        candidates = component & point_eligible
        if not candidates.any():
            continue
        ys_component, xs_component = np.where(candidates)
        best = int(np.argmax(interior_distance[ys_component, xs_component]))
        component_points[label] = [int(xs_component[best]), int(ys_component[best])]
    if not component_points:
        return None

    # Start from the main spatial majority, then iteratively attach nearby
    # depth-consistent lobes.  This preserves a long thin held part while an
    # unrelated far workpiece remains visible in the candidate preview only.
    provisional_labels = list(component_points)
    component_area = np.bincount(labels.ravel())
    main_points = _main_positive_cluster(
        [component_points[label] for label in provisional_labels],
        weights=[float(component_area[label]) for label in provisional_labels],
    )
    associated_labels = {
        label for label, point in component_points.items() if point in main_points
    }
    if not associated_labels:
        return None
    while True:
        associated = np.isin(labels, list(associated_labels))
        distance_to_associated = cv2.distanceTransform((~associated).astype(np.uint8), cv2.DIST_L2, 5)
        anchor_depth = measured[associated & (measured > 0)]
        if not anchor_depth.size:
            break
        anchor_median = float(np.median(anchor_depth))
        added: list[int] = []
        for label, point in component_points.items():
            if label in associated_labels:
                continue
            component = labels == label
            component_depth = measured[component & (measured > 0)]
            if not component_depth.size:
                continue
            if (
                float(distance_to_associated[component].min()) <= 120.0
                and abs(float(np.median(component_depth)) - anchor_median) <= 0.040
            ):
                added.append(label)
        if not added:
            break
        associated_labels.update(added)
    prompt_support = np.isin(labels, list(associated_labels))
    positives = [component_points[label] for label in associated_labels if label in component_points]
    interior_distance = cv2.distanceTransform(prompt_support.astype(np.uint8), cv2.DIST_L2, 5)
    point_eligible &= prompt_support
    # Prefer a wide URDF clearance, but do not discard an otherwise valid
    # frame solely because the held part touches the gripper projection.
    for self_clearance_px in (42.0, 20.0, 0.0):
        for inset_px in (16.0, 10.0, 6.0):
            eligible = (
                point_eligible
                & (interior_distance >= inset_px)
                & (self_distance >= self_clearance_px)
                & ~(forbidden > 0)
            )
            supplemental = inspect._spatially_uniform_positive_points(eligible, runtime.auto_positive_count)
            for point in supplemental:
                if point not in positives:
                    positives.append(point)
            if positives:
                break
        if positives:
            break
    if not positives:
        return None
    eligible &= prompt_support
    ys, xs = np.where(prompt_support)
    return {
        "box": [
            max(0, int(xs.min()) - 100),
            max(0, int(ys.min()) - 100),
            min(prompt_support.shape[1] - 1, int(xs.max()) + 100),
            min(prompt_support.shape[0] - 1, int(ys.max()) + 100),
        ],
        "positives": positives,
        "negatives": negatives,
        "strategy": "pc1_workpiece2_keep_all_candidate_associate_rgbd_lobes_roi100",
        "debug_masks": {
            "initial_depth_seed": seed,
            "processed_near": all_candidate,
            "expanded_depth_candidate": all_candidate,
            "prompt_support": prompt_support,
            "prompt_core": eligible,
        },
    }


def _pc2_all_components_prompt(
    measured: np.ndarray,
    self_pixels: np.ndarray,
    runtime: SimpleNamespace,
) -> dict | None:
    """PC2 prompt that represents every visibly sized post-URDF depth island."""
    raw = (
        (measured >= runtime.pc2_near_min_m)
        & (measured <= runtime.pc2_near_max_m)
        & ~self_pixels
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    # A 200-px island is visually discernible in the 1280x720 PC2 image;
    # keeping smaller islands would turn isolated depth noise into SAM2 cues.
    labels_to_keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 200) + 1
    source = np.isin(labels, labels_to_keep)
    if not source.any():
        return None
    positives: list[list[int]] = []
    for label in labels_to_keep:
        component = labels == label
        point = inspect._spatially_uniform_positive_points(component, 1)
        if point:
            positives.extend(point)
    if not positives:
        return None
    ys, xs = np.where(source)
    box = [
        max(0, int(xs.min()) - 30),
        max(0, int(ys.min()) - 30),
        min(source.shape[1] - 1, int(xs.max()) + 30),
        min(source.shape[0] - 1, int(ys.max()) + 30),
    ]
    fixed = [point for group in inspect.load_fixed_negatives().values() for point in group]
    fixed = [
        point for point in fixed
        if not (
            0 <= point[0] < source.shape[1]
            and 0 <= point[1] < source.shape[0]
            and source[point[1], point[0]]
        )
    ]
    return {
        "box": box,
        "positives": positives,
        "negatives": fixed + inspect.far_depth_negative_points(measured, runtime.pc2_near_max_m),
        "strategy": "pc2_depth100_250_one_positive_per_visible_component",
        "debug_masks": {
            "processed_near": raw,
            "prompt_core": source,
            "prompt_roi": source,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-capture-dir", type=Path,
        help="Standalone rgbd_* capture containing rgb_000.png, depth_aligned_000.npy and robot_state.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("pc1", "pc2"), default="pc1")
    parser.add_argument(
        "--pc1-near-max-m", type=float, default=0.300,
        help="Upper depth bound for the PC1 pre-SAM2 candidate band (metres).",
    )
    parser.add_argument("--pc2-near-min-m", type=float, default=0.110)
    parser.add_argument("--pc2-near-max-m", type=float, default=0.280)
    parser.add_argument(
        "--pc2-all-components-prompt", action="store_true",
        help="Use every visibly sized post-URDF PC2 depth component as a positive-point source.",
    )
    parser.add_argument(
        "--pc2-use-pc1-workpiece2-rule", action="store_true",
        help="Apply the current PC1 workpiece-2 depth/RGB/expansion/prompt policy unchanged to PC2.",
    )
    parser.add_argument("--candidate-rgb-min", type=int)
    parser.add_argument("--candidate-rgb-max", type=int)
    parser.add_argument(
        "--candidate-largest-component", action="store_true",
        help="For a candidate-only preview, retain only its largest 8-connected component.",
    )
    parser.add_argument(
        "--candidate-rgbd-expand-passes", type=int, default=0,
        help="Number of local RGB-D coherent expansion passes after candidate filtering.",
    )
    parser.add_argument(
        "--candidate-rgbd-expand-allow-missing-depth", action="store_true",
        help="Allow RGB-coherent expansion through zero/invalid-depth neighbours.",
    )
    parser.add_argument(
        "--pc1-workpiece2-expanded-prompt", action="store_true",
        help="Use the 0.10--max-depth RGB seed plus 10-pass missing-depth expansion for PC1 workpiece 2.",
    )
    parser.add_argument(
        "--sam2-url",
        default="unix:///home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service/runtime/sam2-service.sock",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        help="Optional successful result.json to supply the URDF pose for an incomplete capture log.",
    )
    parser.add_argument(
        "--workpiece", choices=("1", "2"), required=True,
        help="Explicit workpiece type. Required: strategy selection never infers this from old logs.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Stop after the pre-SAM2 URDF/depth candidate filter and save its debug masks.",
    )
    args = parser.parse_args()
    prefix = args.view
    capture = args.legacy_capture_dir
    if capture is None:
        rgb = cv2.imread(str(args.run_dir / f"{prefix}_rgb.png"), cv2.IMREAD_COLOR)
        depth = np.load(args.run_dir / f"{prefix}_depth.npy", allow_pickle=False)
    else:
        rgb = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
        depth = np.load(capture / "depth_aligned_000.npy", allow_pickle=False)
    if rgb is None or depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"Missing or inconsistent {args.view} RGB-D")
    cfg = _config_for_run(args.run_dir)
    workpiece = args.workpiece
    result_path = args.result_path or (args.run_dir / "result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    measured = depth.astype(np.float32) * 0.001
    rendered = (
        _legacy_capture_rendered_depth(capture, result_path, depth)
        if capture is not None
        else _rendered_depth(result, cfg, depth, args.view)
    )
    valid = np.isfinite(rendered) & (measured > 0)
    error = np.abs(measured - rendered)
    near_black_limit = 20 if workpiece == "1" else 35
    near_black = np.max(rgb, axis=2) <= near_black_limit
    # 250 mm for close black material, ramping to 300 mm once the expected
    # URDF depth reaches 250 mm (the 200--250 mm interval is linear).
    black_ramp = np.clip((rendered - 0.200) / 0.050, 0.0, 1.0)
    black_tolerance = 0.250 + black_ramp * 0.050
    depth_matched = valid & ((error <= 0.035) | (near_black & (error <= black_tolerance)))
    # No signed URDF-vs-measured mismatch exclusion: a local URDF depth error
    # must not remove a possible held-workpiece surface from PC2 prompts.
    projected_nearer_mismatch = np.zeros(depth.shape, dtype=bool)
    depth_matched |= projected_nearer_mismatch
    self_pixels = cv2.dilate(depth_matched.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    runtime = SimpleNamespace(
        workpiece=workpiece,
        rgb_for_prompt=rgb,
        pc1_near_min_m=0.100, pc1_near_max_m=float(args.pc1_near_max_m),
        pc1_candidate_opening_kernel_px=0,
        pc1_roi_margin_px=100,
        pc1_workpiece2_expand_px=8,
        pc1_workpiece2_expand_color_diff=30,
        pc1_workpiece2_expand_depth_diff_m=0.015,
        pc1_urdf_barrier_signed_depth_m=0.120,
        pc1_urdf_barrier_min_area_px=200,
        pc2_near_min_m=float(args.pc2_near_min_m), pc2_near_max_m=float(args.pc2_near_max_m),
        pc2_gripper_box_margin_px=50, auto_positive_count=6,
    )
    if (
        workpiece == "2"
        and args.pc1_workpiece2_expanded_prompt
        and (args.view == "pc1" or args.pc2_use_pc1_workpiece2_rule)
    ):
        auto = _pc1_workpiece2_expanded_prompt(measured, self_pixels, rgb, runtime)
    elif args.view == "pc2" and args.pc2_all_components_prompt:
        auto = _pc2_all_components_prompt(measured, self_pixels, runtime)
    else:
        auto = (
            inspect._pc1_prompt(measured, rendered, self_pixels, runtime, cfg)
            if args.view == "pc1"
            else inspect._pc2_prompt(measured, rendered, self_pixels, runtime)
        )
    if auto is None:
        if not args.candidate_only:
            raise RuntimeError(f"No reliable {args.view} prompt from the depth band")
        # A preview is still valuable for an incomplete capture: retain the
        # exact post-URDF depth band even when it is too fragmented to form a
        # reliable SAM2 prompt/ROI.
        near_min = runtime.pc1_near_min_m if args.view == "pc1" else runtime.pc2_near_min_m
        near_max = runtime.pc1_near_max_m if args.view == "pc1" else runtime.pc2_near_max_m
        auto = {
            "box": None,
            "positives": [],
            "negatives": [],
            "debug_masks": {
                "processed_near": (measured >= near_min) & (measured <= near_max) & ~self_pixels,
            },
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / f"{prefix}_rgb.png"), rgb)
    cv2.imwrite(str(args.output_dir / f"{prefix}_urdf_self_prompt_filter.png"), self_pixels.astype(np.uint8) * 255)
    for name, debug_mask in auto["debug_masks"].items():
        cv2.imwrite(str(args.output_dir / f"{prefix}_{name}.png"), debug_mask.astype(np.uint8) * 255)
    np.save(args.output_dir / f"{prefix}_depth.npy", depth)
    if args.candidate_only:
        candidate = auto["debug_masks"]["processed_near"]
        brightness = np.max(rgb, axis=2)
        if args.candidate_rgb_min is not None:
            candidate = candidate & (brightness >= args.candidate_rgb_min)
        if args.candidate_rgb_max is not None:
            candidate = candidate & (brightness <= args.candidate_rgb_max)
        if args.candidate_largest_component:
            candidate = inspect._largest_component(candidate)
        if args.candidate_rgbd_expand_passes > 0:
            expand_runtime = SimpleNamespace(
                pc1_workpiece2_expand_px=int(args.candidate_rgbd_expand_passes),
                pc1_workpiece2_expand_color_diff=30,
                pc1_workpiece2_expand_depth_diff_m=0.015,
            )
            if args.candidate_rgbd_expand_allow_missing_depth:
                candidate = _expand_candidate_allowing_missing_depth(
                    candidate, measured, rgb, self_pixels,
                    runtime.pc1_near_min_m, runtime.pc1_near_max_m,
                    int(args.candidate_rgbd_expand_passes),
                )
            else:
                candidate = inspect._pc1_workpiece2_expand_candidate(
                    candidate,
                    measured,
                    rgb,
                    self_pixels,
                    runtime.pc1_near_min_m,
                    runtime.pc1_near_max_m,
                    expand_runtime,
                )
            # Expansion can touch an unrelated thin fragment; preserve only
            # the component connected to the original main candidate.
            candidate = inspect._largest_component(candidate)
        auto["debug_masks"]["processed_near"] = candidate
        if "expanded_depth_candidate" in auto["debug_masks"]:
            auto["debug_masks"]["expanded_depth_candidate"] = candidate
        cv2.imwrite(
            str(args.output_dir / f"{prefix}_processed_near.png"),
            candidate.astype(np.uint8) * 255,
        )
        report = {
            "source_run": str(args.run_dir),
            "urdf_pose_result": str(result_path),
            "view": args.view,
            "workpiece": workpiece,
            "workpiece_selection": "explicit_cli",
            "policy": "pre_sam2_urdf_depth_candidate_filter_only",
            "candidate_pixels": int(candidate.sum()),
            "urdf_self_prompt_filter_pixels": int(self_pixels.sum()),
            "near_black_max_rgb": near_black_limit,
            "ordinary_urdf_depth_tolerance_m": 0.035,
            "black_urdf_depth_tolerance_m": [0.250, 0.300],
            "candidate_rgb_range": [args.candidate_rgb_min, args.candidate_rgb_max],
            "candidate_largest_component_only": args.candidate_largest_component,
            "candidate_rgbd_expand_passes": args.candidate_rgbd_expand_passes,
            "candidate_rgbd_expand_color_diff": 30,
            "candidate_rgbd_expand_depth_diff_m": 0.015,
            "candidate_rgbd_expand_allow_missing_depth": args.candidate_rgbd_expand_allow_missing_depth,
            "roi_box_xyxy": auto["box"],
        }
        (args.output_dir / "replay_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))
        return
    using_pc1_workpiece2_policy = workpiece == "2" and (
        args.view == "pc1" or args.pc2_use_pc1_workpiece2_rule
    )
    raw = inspect._sam2_prompt(rgb, auto["box"], auto["positives"], auto["negatives"], args.sam2_url)
    mask, _ = inspect._main_mask_with_filled_holes(
        raw,
        auto["positives"] if args.view == "pc2" or using_pc1_workpiece2_policy else None,
    )
    if not using_pc1_workpiece2_policy:
        mask, _ = inspect._filter_mask_by_positive_depth(
            mask, depth, 0.001, auto["positives"], 0.050, float(cfg["min_depth_m"]),
        )
    mask, _ = inspect._fill_small_enclosed_holes(mask)
    main_mask_rgbd_expansion: dict[str, float | int] | None = None
    if using_pc1_workpiece2_policy:
        mask, _ = inspect._pc1_workpiece2_outer_dark_cleanup(
            mask,
            rgb,
            auto["debug_masks"].get("prompt_support", auto["debug_masks"]["processed_near"]),
            auto["positives"],
        )
        # Repair only the immediately adjacent SAM boundary.  Unlike the
        # pre-SAM candidate expansion this requires a real measured depth in
        # the robust depth range of the selected main mask, so it cannot jump
        # through a zero-depth gap into another visible object.
        mask, main_mask_rgbd_expansion = _expand_main_mask_rgbd_consistent(
            mask,
            measured,
            rgb,
            self_pixels,
            auto["debug_masks"].get("prompt_support", auto["debug_masks"]["processed_near"]),
        )
        mask, _ = inspect._pc1_workpiece2_outer_dark_cleanup(
            mask,
            rgb,
            auto["debug_masks"].get("prompt_support", auto["debug_masks"]["processed_near"]),
            auto["positives"],
        )
        cv2.imwrite(
            str(args.output_dir / f"{prefix}_main_mask_rgbd_expanded.png"),
            mask.astype(np.uint8),
        )
    cv2.imwrite(str(args.output_dir / f"{prefix}_mask.png"), mask)
    cv2.imwrite(str(args.output_dir / f"{prefix}_sam2_raw_mask.png"), raw)
    report = {
        "source_run": str(args.run_dir),
        "urdf_pose_result": str(result_path),
        "view": args.view,
        "policy": "one_sam2_no_opening_urdf_initial_prompt_filter_only",
        "initial_urdf_filter": {
            "ordinary_depth_tolerance_m": 0.035,
            "near_black_max_rgb": near_black_limit,
            "near_black_depth_tolerance_m": 0.250,
            "far_black_depth_tolerance_m": 0.300,
            "far_black_depth_reached_at_urdf_depth_m": 0.250,
            "projected_nearer_mismatch_exclusion_m": 0.125,
        },
        "candidate_pixels": int(auto["debug_masks"]["processed_near"].sum()),
        "urdf_self_prompt_filter_pixels": int(self_pixels.sum()),
        "projected_nearer_mismatch_pixels": int(projected_nearer_mismatch.sum()),
        "final_mask_pixels": int((mask > 0).sum()),
        "workpiece": workpiece,
        "workpiece_selection": "explicit_cli",
        "roi_box_xyxy": auto["box"],
        "positive_points": auto["positives"],
        "negative_points": auto["negatives"],
        "main_mask_rgbd_expansion": main_mask_rgbd_expansion,
    }
    (args.output_dir / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
