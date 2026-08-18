#!/usr/bin/env python3
"""用同一套桌面几何和 SAM2 流程生成工件 1/2 mask。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .geometry import depth_to_xyz, farthest_points, retain_prompt_components, workspace_roi_mask
from .sam2_client import prompt_segment


MODULE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODULE_ROOT.parents[2]
CONFIG_ROOT = MODULE_ROOT / "configs"
DEFAULT_SOCKET = PROJECT_ROOT / "tool/sam2-service/runtime/sam2-service.sock"
SAM2_BINARY = PROJECT_ROOT / "tool/sam2-service/.venv/bin/sam2-service"


def _fit_table_plane(points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.003)
    if len(cloud.points) < 200:
        raise RuntimeError("有效深度点不足，无法拟合桌面平面")
    o3d.utility.random.seed(17)
    plane, _ = cloud.segment_plane(0.004, 3, 1200)
    plane = np.asarray(plane, dtype=np.float64)
    plane /= max(float(np.linalg.norm(plane[:3])), 1e-12)
    if float(plane[:3] @ (-points.mean(axis=0))) < 0.0:
        plane *= -1.0
    return plane


def _ensure_sam2(url: str) -> None:
    if not url.startswith("unix://"):
        return
    socket_path = Path(url.removeprefix("unix://"))
    if socket_path.exists():
        return
    if not SAM2_BINARY.is_file():
        raise RuntimeError(f"SAM2 service binary 不存在: {SAM2_BINARY}")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = socket_path.parent / "sam2-service-module-4-1.log"
    with log_path.open("ab") as log:
        subprocess.Popen(
            [
                str(SAM2_BINARY), "--socket", str(socket_path),
                "--backend", "efficientvit", "--queue-capacity", "4",
                "--request-timeout-s", "90",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(120):
        if socket_path.exists():
            return
        time.sleep(0.5)
    raise RuntimeError(f"SAM2 服务未启动: {socket_path}；查看 {log_path}")


def _load_inputs(capture: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    metadata_path = capture / "metadata.json"
    rgb_path = capture / "rgb_000.png"
    depth_path = capture / "depth_aligned_000.png"
    missing = [path for path in (metadata_path, rgb_path, depth_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 RGB-D 输入: " + ", ".join(map(str, missing)))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
        raise RuntimeError("RGB-D 图像无法读取或尺寸不一致")
    intrinsics = metadata["color_intrinsics"]
    matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return metadata, rgb, depth, matrix


def _component_box(stats: np.ndarray, label: int, shape: tuple[int, int], padding: int) -> list[int]:
    x, y, width, height = (int(value) for value in stats[label, :4])
    image_height, image_width = shape
    return [
        max(0, x - padding),
        max(0, y - padding),
        min(image_width - 1, x + width - 1 + padding),
        min(image_height - 1, y + height - 1 + padding),
    ]


def generate_masks(
    capture: Path,
    output: Path,
    *,
    workpiece: str,
    config_path: Path,
    sam2_url: str,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mask_config = config["mask"]
    metadata, rgb, depth, matrix = _load_inputs(capture)
    xyz = depth_to_xyz(depth, matrix, float(metadata["depth_scale_m"]))
    valid = (depth > 0) & np.isfinite(xyz).all(axis=2)
    minimum_depth = mask_config.get("minimum_depth_m")
    maximum_depth = mask_config.get("maximum_depth_m")
    if minimum_depth is not None:
        valid &= xyz[:, :, 2] > float(minimum_depth)
    if maximum_depth is not None:
        valid &= xyz[:, :, 2] < float(maximum_depth)

    plane = _fit_table_plane(xyz[valid])
    heights = xyz @ plane[:3] + plane[3]
    roi = workspace_roi_mask(mask_config, depth.shape).astype(bool)
    raw_seed = valid & roi & (heights >= float(mask_config["seed_height_mm"]) * 1e-3)
    maximum_gray = mask_config.get("seed_max_gray")
    if maximum_gray is not None:
        raw_seed &= cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) <= int(maximum_gray)
    cleaned = cv2.morphologyEx(
        raw_seed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    cleaned = cv2.dilate(cleaned, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), connectivity=8
    )
    candidates = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(mask_config["minimum_seed_area_px"])
    ]
    if mask_config["component_selection"] == "largest" and candidates:
        candidates = [max(candidates, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))]

    _ensure_sam2(sam2_url)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "seed_raw.png"), raw_seed.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "seed_cleaned.png"), cleaned.astype(np.uint8) * 255)
    prompt_overlay = rgb.copy()
    mask_overlay = rgb.copy()
    masks: list[np.ndarray] = []
    diagnostics: list[dict] = []
    pixel_columns = np.indices(depth.shape)[1]

    for label in candidates:
        seed_component = labels == label
        positives = farthest_points(seed_component, int(mask_config["positive_points"]))
        if mask_config["sam_box_mode"] == "component":
            box = _component_box(
                stats, label, depth.shape, int(mask_config["sam_box_padding_px"])
            )
        else:
            box = [0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1]

        negatives: list[list[int]] = []
        negative_count = int(mask_config["negative_points"])
        if negative_count:
            x0, y0, x1, y1 = box
            seed_x = int(stats[label, cv2.CC_STAT_LEFT])
            in_box = np.zeros_like(valid)
            in_box[y0:y1 + 1, x0:x1 + 1] = True
            left_of_seed = pixel_columns < max(0, seed_x - 8)
            negatives = farthest_points(
                in_box & left_of_seed & roi & valid, negative_count
            )
            if not negatives:
                negatives = farthest_points(
                    (pixel_columns < rgb.shape[1] * 0.45) & roi & valid & ~cleaned,
                    negative_count,
                )

        cv2.rectangle(prompt_overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        for x, y in positives:
            cv2.circle(prompt_overlay, (x, y), 5, (0, 255, 0), -1)
        for x, y in negatives:
            cv2.circle(prompt_overlay, (x, y), 5, (0, 0, 255), -1)
        mask = prompt_segment(rgb, box, positives, negatives, sam2_url).astype(bool) & roi
        if bool(mask_config["retain_prompt_components"]):
            mask = retain_prompt_components(mask, positives, cleaned)
        if bool(mask_config["require_valid_depth_in_final_mask"]):
            mask &= valid
        accepted = int(np.count_nonzero(mask)) >= int(mask_config["minimum_mask_area_px"])
        diagnostics.append(
            {
                "seed_label": int(label),
                "positives_xy": positives,
                "negatives_xy": negatives,
                "box_xyxy": box,
                "sam2_mask_pixels": int(np.count_nonzero(mask)),
                "accepted": accepted,
            }
        )
        if not accepted:
            continue
        color = np.random.default_rng(label).integers(70, 255, 3, dtype=np.uint8)
        mask_overlay[mask] = (0.52 * mask_overlay[mask] + 0.48 * color).astype(np.uint8)
        masks.append(mask.astype(np.uint8))

    cv2.imwrite(str(output / "sam2_prompt_overlay.png"), prompt_overlay)
    cv2.imwrite(str(output / "sam2_masks_overlay.png"), mask_overlay)
    mask_array = (
        np.asarray(masks, dtype=np.uint8)
        if masks
        else np.empty((0, *rgb.shape[:2]), dtype=np.uint8)
    )
    np.savez_compressed(output / "masks.npz", masks=mask_array)
    payload = {
        "schema_version": 1,
        "workpiece_id": str(workpiece),
        "method": "config_driven_tabletop_geometry_sam2",
        "config": str(config_path),
        "plane": plane.tolist(),
        "table_plane_camera": plane.tolist(),
        "safe_roi": mask_config["safe_roi_polygon_px"],
        "seed_groups": diagnostics,
        "masks": len(masks),
    }
    (output / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SAM2_MASKS={output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", choices=("1", "2"), required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--socket",
        default=os.environ.get("PARM_SAM2_URL", f"unix://{DEFAULT_SOCKET}"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = args.capture_dir.expanduser().resolve()
    output = (args.out or capture / "sam2_masks").expanduser().resolve()
    config_path = (
        args.config or CONFIG_ROOT / f"workpiece_{args.workpiece}.json"
    ).expanduser().resolve()
    generate_masks(
        capture, output, workpiece=args.workpiece,
        config_path=config_path, sam2_url=args.socket,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

