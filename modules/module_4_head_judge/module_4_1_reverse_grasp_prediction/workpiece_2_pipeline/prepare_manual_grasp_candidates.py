#!/usr/bin/env python3
"""Build, display-ready and select reverse-grasp candidates for manual jog mode.

This helper is used only by run_manual_adjust_reverse_grasp.sh.  It estimates
each SAM mask separately, suppresses overlapping masks for the same physical
part, and leaves a manifest identifying the operator-selected prediction.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ESTIMATE = HERE / "estimate_fixed_reverse_pose.py"
PREDICT = HERE / "predict_convex_tab_grasp.py"
VISUALIZE = HERE / "grasp/visualize_grasp_prediction.py"
GATE = HERE / "grasp/execution_quality_gate.json"


def inside_here(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_relative_to(HERE):
        raise ValueError(f"输出必须位于 {HERE}: {path}")
    return path


def accepted_pose(pose: dict) -> bool:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    icp = pose.get("icp") or {}
    trial = next((x for x in pose.get("yaw_trials", []) if int(x.get("yaw_index", -1)) == int(icp.get("yaw_index", -2))), {})
    match = ((trial.get("render") or {}).get("match_fraction"))
    return bool(float(icp.get("fitness", -1)) >= float(gate["minimum_icp_fitness"])
                and float(icp.get("rmse_m", float("inf"))) <= float(gate["maximum_icp_rmse_m"])
                and match is not None and float(match) >= float(gate["minimum_depth_match_fraction"]))


def make_overview(candidates: list[dict], path: Path) -> None:
    images = [cv2.imread(item["overlay"], cv2.IMREAD_COLOR) for item in candidates]
    if any(image is None for image in images):
        raise RuntimeError("无法读取候选 grasp_prediction_overlay")
    thumb_w, thumb_h = 640, 360
    tiles = []
    for index, (item, image) in enumerate(zip(candidates, images), start=1):
        image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (thumb_w - 1, thumb_h - 1), (0, 255, 255), 3)
        cv2.putText(image, f"[{index}] SAM mask {item['mask_index']}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, f"[{index}] SAM mask {item['mask_index']}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(image)
    rows = [np.hstack(tiles[i:i + 2] + ([np.zeros_like(tiles[0])] if len(tiles[i:i + 2]) == 1 else [])) for i in range(0, len(tiles), 2)]
    canvas = np.vstack(rows)
    cv2.imwrite(str(path), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    capture = args.capture_dir.expanduser().resolve()
    output = inside_here(args.output_dir)
    if output.exists():
        raise FileExistsError(f"输出已存在: {output}")
    masks = np.load(capture / "sam2_masks/masks.npz")["masks"]
    output.mkdir(parents=True)
    candidates, failures = [], []
    for mask_index in range(len(masks)):
        candidate_dir = output / f"mask_{mask_index:03d}"
        pose_dir = candidate_dir / "pose"
        point_dir = candidate_dir / "convex_tab_grasp"
        visual_dir = candidate_dir / "visualization"
        try:
            subprocess.run([sys.executable, str(ESTIMATE), "--capture-dir", str(capture), "--mask-index", str(mask_index), "--output-dir", str(pose_dir)], check=True, capture_output=True, text=True)
            pose_path = pose_dir / "fixed_reverse_pose.json"
            pose = json.loads(pose_path.read_text(encoding="utf-8"))
            if not accepted_pose(pose):
                failures.append({"mask_index": mask_index, "reason": "ICP quality gate rejected"})
                continue
            subprocess.run([sys.executable, str(PREDICT), "--pose", str(pose_path), "--capture-dir", str(capture), "--output-dir", str(point_dir)], check=True, capture_output=True, text=True)
            prediction = point_dir / "convex_tab_grasp_prediction.json"
            subprocess.run([sys.executable, str(VISUALIZE), "--prediction-json", str(prediction), "--output-dir", str(visual_dir)], check=True, capture_output=True, text=True)
            payload = json.loads(prediction.read_text(encoding="utf-8"))
            candidates.append({"mask_index": mask_index, "pose": str(pose_path), "prediction": str(prediction),
                               "overlay": str(visual_dir / "grasp_prediction_overlay.png"),
                               "contact_camera_m": payload["grasp_contact_point_camera_m"]})
        except Exception as exc:
            failures.append({"mask_index": mask_index, "reason": str(exc)})
    # SAM frequently emits overlapping masks for one item. Keep only one grasp
    # candidate per 25 mm contact neighbourhood, retaining the first valid one.
    unique = []
    for item in candidates:
        p = np.asarray(item["contact_camera_m"], dtype=float)
        if any(np.linalg.norm(p - np.asarray(old["contact_camera_m"], dtype=float)) < .025 for old in unique):
            continue
        unique.append(item)
    candidates = unique
    if not candidates:
        details = "; ".join(
            f"mask {item['mask_index']}: {item['reason']}" for item in failures
        ) or "没有生成候选"
        raise RuntimeError(f"没有可用于反面抓取的候选。详情: {details}")
    overview = output / "grasp_prediction_overview.png"
    make_overview(candidates, overview)
    print(f"MANUAL_GRASP_CANDIDATE_OVERVIEW={overview}")
    subprocess.run([sys.executable, str(HERE / "show_grasp_prediction.py"),
                    "--image", str(overview), "--title", "reverse grasp candidates"], check=True)
    if len(candidates) == 1:
        selected = 0
        print("只检测到一个有效反面工件候选，自动选中 [1]。")
    else:
        print(f"检测到 {len(candidates)} 个有效反面工件候选。请先查看总览图，然后输入要抓取的编号：")
        while True:
            raw = input(f"选择 [1-{len(candidates)}]: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(candidates):
                selected = int(raw) - 1
                break
            print("请输入有效编号。")
    manifest = {"event": "manual_reverse_grasp_candidate_selection", "capture_dir": str(capture),
                "overview": str(overview), "candidates": candidates, "failures": failures,
                "selected_index_1_based": selected + 1, "selected": candidates[selected]}
    path = output / "manual_selected_candidate.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MANUAL_SELECTED_CANDIDATE={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
