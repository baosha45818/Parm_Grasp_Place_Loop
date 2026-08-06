#!/usr/bin/env python3
"""Replay PC1 SAM2 from saved RGB-D using the current URDF depth policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from scipy import ndimage

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-url", default="unix:///tmp/sam2-service.sock")
    return parser.parse_args()


def rendered_depth(run: Path, result: dict, cfg: dict, depth: np.ndarray) -> tuple[np.ndarray, bool]:
    registration = result.get("registration") or {}
    intrinsics_saved = registration.get("camera_intrinsics")
    K = np.asarray(intrinsics_saved if intrinsics_saved is not None else FALLBACK_K, dtype=float)
    T_camera_object = np.asarray(result["T_camera_object"], dtype=float)
    T_body_object = np.asarray(result["T_body_base_object"], dtype=float)
    T_tool_object = np.asarray(result["T_tool_object"], dtype=float)
    T_body_camera = T_body_object @ np.linalg.inv(T_camera_object)
    T_body_tool = T_body_object @ np.linalg.inv(T_tool_object)
    T_camera_tool = np.linalg.inv(T_body_camera) @ T_body_tool
    active = 0.65 * (1.0 - float(result["gripper_opening_percent"]) / 100.0)
    points = load_self_surface_points(URDF_ROOT, active, 200000, cfg["pc_joints_deg"])
    rendered = projected_gripper_depth(points, T_camera_tool, K, depth.shape)
    projected = np.isfinite(rendered)
    closed = cv2.morphologyEx(projected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    new_pixels = closed & ~projected
    if new_pixels.any():
        _, nearest = ndimage.distance_transform_edt(~projected, return_indices=True)
        rendered = rendered.copy()
        rendered[new_pixels] = rendered[nearest[0][new_pixels], nearest[1][new_pixels]]
    return rendered, intrinsics_saved is None


def main() -> None:
    args = parse_args()
    run = args.run_dir
    output = args.output_dir
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    cfg = json.loads((Path(__file__).parents[2] / "modules/module_2_left_inspect_apart/inspect_config.json").read_text())
    rgb = cv2.imread(str(run / "pc1_rgb.png"), cv2.IMREAD_COLOR)
    depth = np.load(run / "pc1_depth.npy", allow_pickle=False)
    if rgb is None or depth.shape != rgb.shape[:2]:
        raise RuntimeError("Missing or inconsistent PC1 RGB-D")
    rendered, fallback_intrinsics = rendered_depth(run, result, cfg, depth)
    measured = depth.astype(np.float32) * 0.001
    valid = np.isfinite(rendered) & (measured > 0)
    depth_error = np.abs(measured - rendered)
    near_black = np.max(rgb, axis=2) <= 20
    exact = valid & ((depth_error <= 0.035) | (near_black & (depth_error <= 0.200)))
    self_pixels = cv2.dilate(exact.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    runtime_args = SimpleNamespace(
        workpiece="1",
        rgb_for_prompt=rgb,
        pc1_near_min_m=0.100,
        pc1_near_max_m=0.300,
        pc1_candidate_opening_kernel_px=0,
        pc1_workpiece2_expand_px=8,
        pc1_workpiece2_expand_color_diff=30,
        pc1_workpiece2_expand_depth_diff_m=0.015,
        auto_positive_count=6,
        sam2_url=args.sam2_url,
        mask_positive_depth_tolerance_m=0.050,
    )
    auto = inspect._pc1_prompt(measured, rendered, self_pixels, runtime_args, cfg)
    if auto is None:
        raise RuntimeError("No reliable PC1 prompt after new URDF filtering")
    box = auto["box"]
    positives, rejected = inspect._exclude_urdf_silhouette_positive_points(auto["positives"], rendered)
    negatives = auto["negatives"]
    seed = positives or [auto["positives"][0]]
    raw = inspect._sam2_prompt(rgb, box, seed, negatives, args.sam2_url)
    mask, _ = inspect._main_mask_with_filled_holes(raw)
    if positives:
        mask, _ = inspect._filter_mask_by_positive_depth(mask, depth, 0.001, positives, 0.050, float(cfg["min_depth_m"]))
    protected = inspect._prompt_components_containing_points(auto["debug_masks"]["processed_near"], positives)
    mask, _ = inspect._filter_mask_by_urdf_silhouette(mask, rendered, protected)
    mask, _ = inspect._open_and_remove_distant_small_components(mask, opening_kernel_px=21, keep_only_main_component=True)
    mask, _ = inspect._fill_small_enclosed_holes(mask)
    # PC1 second pass uses positive points from the clean first-pass support.
    reselection = inspect.positive_points(mask > 0, 3)
    second_pass = bool(len(reselection) >= 3)
    if second_pass:
        first = mask > 0
        positives = reselection
        raw = inspect._sam2_prompt(rgb, box, positives, negatives, args.sam2_url)
        mask, _ = inspect._main_mask_with_filled_holes(raw)
        mask, _ = inspect._filter_mask_by_positive_depth(mask, depth, 0.001, positives, 0.050, float(cfg["min_depth_m"]))
        mask, _ = inspect._filter_mask_by_urdf_silhouette(mask, rendered, first)
        mask, _ = inspect._open_and_remove_distant_small_components(mask, opening_kernel_px=21, keep_only_main_component=True)
        mask, _ = inspect._fill_small_enclosed_holes(mask)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "pc1_rgb.png"), rgb)
    cv2.imwrite(str(output / "pc1_mask.png"), mask)
    cv2.imwrite(str(output / "pc1_sam2_raw_mask.png"), raw)
    cv2.imwrite(str(output / "pc1_urdf_self_removed.png"), self_pixels.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "pc1_processed_near.png"), auto["debug_masks"]["processed_near"].astype(np.uint8) * 255)
    np.save(output / "pc1_depth.npy", depth)
    np.save(output / "pc1_urdf_rendered_depth_m.npy", rendered)
    overlay = rgb.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(output / "pc1_overlay.png"), overlay)
    report = {
        "source_run": str(run), "fallback_intrinsics": fallback_intrinsics,
        "policy": {"ordinary_depth_tolerance_m": 0.035, "near_black_max_rgb": 20, "near_black_depth_tolerance_m": 0.200},
        "candidate_pixels": int(auto["debug_masks"]["processed_near"].sum()),
        "urdf_self_pixels": int(self_pixels.sum()), "final_mask_pixels": int((mask > 0).sum()),
        "positive_points": positives, "rejected_urdf_points": rejected, "second_pass": second_pass,
    }
    (output / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
