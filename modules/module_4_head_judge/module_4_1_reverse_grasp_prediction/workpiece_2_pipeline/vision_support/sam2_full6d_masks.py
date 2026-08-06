#!/usr/bin/env python3
"""用桌面几何生成 SAM2 正点，并保存不经 RGB-D 二次裁剪的工件 mask。

桌面平面、深度高度和暗度只用于寻找可靠的正点；SAM2 返回的连通域是
后续 ICP 使用的完整掩码，绝不再按灰度或高度把金属工件边缘裁掉。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
LOOP_ROOT = HERE.parents[4]
SAM2_SOCKET = LOOP_ROOT / "tool/sam2-service/runtime/sam2-service.sock"
SAM2_BINARY = LOOP_ROOT / "tool/sam2-service/.venv/bin/sam2-service"
sys.path.insert(0, str(HERE))
from sam2_client import prompt_segment


def depth_to_xyz(depth: np.ndarray, K: np.ndarray, scale_m: float) -> np.ndarray:
    rows, cols = np.indices(depth.shape)
    z = depth.astype(np.float64) * scale_m
    return np.dstack(((cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z))


def fit_table_plane(points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.003)
    if len(cloud.points) < 200:
        raise RuntimeError("有效深度点不足，无法拟合桌面平面")
    o3d.utility.random.seed(17)
    plane, _ = cloud.segment_plane(.004, 3, 1200)
    plane = np.asarray(plane, dtype=float)
    plane /= max(float(np.linalg.norm(plane[:3])), 1e-12)
    if float(plane[:3] @ (-points.mean(axis=0))) < 0.0:
        plane *= -1.0
    return plane


def distributed_positive_points(labels: np.ndarray, label: int, count: int = 3) -> list[list[int]]:
    """与 module_4_2 相同：中心点加最远点采样，生成分散的正点。"""
    rows, cols = np.nonzero(labels == label)
    points = np.column_stack((cols, rows))
    if len(points) == 0:
        return []
    chosen = [int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))]
    nearest = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(1, min(count, len(points))):
        index = int(np.argmax(nearest))
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((points - points[index]) ** 2, axis=1))
    return [[int(value) for value in points[index]] for index in chosen]


def retain_prompt_components(mask: np.ndarray, positives: list[list[int]], seed_mask: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    selected = {
        int(labels[y, x]) for x, y in positives
        if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0] and labels[y, x] > 0
    }
    if not selected and count > 1:
        overlaps = [int(np.count_nonzero((labels == label) & seed_mask)) for label in range(1, count)]
        selected = {1 + int(np.argmax(overlaps))}
    return np.isin(labels, list(selected)) if selected else np.zeros_like(mask, dtype=bool)


def ensure_sam2_service(socket_path: Path) -> None:
    if socket_path.exists():
        return
    if not SAM2_BINARY.is_file():
        raise RuntimeError(f"SAM2 service binary 不存在: {SAM2_BINARY}")
    log_path = socket_path.parent / "sam2-service-workpiece2.log"
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        subprocess.Popen(
            [str(SAM2_BINARY), "--socket", str(socket_path), "--backend", "efficientvit",
             "--queue-capacity", "4", "--request-timeout-s", "90"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    for _ in range(120):
        if socket_path.exists():
            return
        time.sleep(.5)
    raise RuntimeError(f"SAM2 service 未启动: {socket_path}；查看 {log_path}")


def write_prompts(rgb: np.ndarray, raw: np.ndarray, cleaned: np.ndarray,
                  groups: list[dict], output: Path) -> None:
    image = rgb.copy()
    image[raw] = (.45 * image[raw] + np.array([0, 180, 255])).astype(np.uint8)
    image[cleaned] = (.40 * image[cleaned] + np.array([0, 255, 0])).astype(np.uint8)
    for index, group in enumerate(groups, 1):
        for point in group["positives_xy"]:
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(image, "+", (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2, cv2.LINE_AA)
        anchor = group["positives_xy"][0]
        cv2.putText(image, f"SAM2 #{index}", (anchor[0] + 12, max(20, anchor[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 0, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"无法写入正点图: {output}")


def main() -> int:
    workspace = json.loads((PIPELINE_ROOT / "config/tabletop_workspace.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--socket", default=f"unix://{SAM2_SOCKET}")
    parser.add_argument("--roi", nargs=4, type=int, default=workspace["safe_roi_xywh"], metavar=("X", "Y", "W", "H"))
    parser.add_argument("--plane-distance-mm", type=float, default=4.0)
    parser.add_argument("--seed-height-mm", type=float, default=3.0)
    parser.add_argument("--seed-max-gray", type=int, default=105)
    parser.add_argument("--min-seed-area", type=int, default=90)
    parser.add_argument("--min-mask-area", type=int, default=350)
    args = parser.parse_args()
    capture, output = args.capture_dir.expanduser().resolve(), args.out.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
    rgb = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture / "depth_aligned_000.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
        raise RuntimeError("RGB-D 图像无法读取或尺寸不一致")
    expected_size = workspace.get("reference_color_size_wh")
    if expected_size and [rgb.shape[1], rgb.shape[0]] != [int(value) for value in expected_size]:
        raise RuntimeError(f"头部 RGB 分辨率与工作区配置不一致: capture={[rgb.shape[1], rgb.shape[0]]}, expected={expected_size}")
    intr = metadata["color_intrinsics"]
    K = np.array([[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]])
    xyz = depth_to_xyz(depth, K, float(metadata["depth_scale_m"]))
    valid = (depth > 0) & np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > .20) & (xyz[:, :, 2] < 2.50)
    plane = fit_table_plane(xyz[valid])
    heights = xyz @ plane[:3] + plane[3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    x, y, width, height = args.roi
    roi = np.zeros(depth.shape, dtype=bool)
    roi[max(0, y):min(depth.shape[0], y + height), max(0, x):min(depth.shape[1], x + width)] = True
    raw_seed = valid & roi & (heights >= args.seed_height_mm * 1e-3) & (gray <= args.seed_max_gray)
    cleaned = cv2.morphologyEx(raw_seed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cleaned = cv2.dilate(cleaned, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    groups = [{"seed_label": int(label), "positives_xy": distributed_positive_points(labels, label)}
              for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= args.min_seed_area]
    output.mkdir(parents=True)
    cv2.imwrite(str(output / "seed_raw.png"), raw_seed.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "seed_cleaned.png"), cleaned.astype(np.uint8) * 255)
    write_prompts(rgb, raw_seed, cleaned, groups, output / "sam2_positive_prompts.png")
    socket_path = Path(args.socket.removeprefix("unix://"))
    ensure_sam2_service(socket_path)
    masks, diagnostics = [], []
    overlay = rgb.copy()
    for group in groups:
        positives = group["positives_xy"]
        sam_mask = prompt_segment(rgb, [0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1], positives, [], args.socket).astype(bool)
        mask = retain_prompt_components(sam_mask & roi, positives, cleaned)
        accepted = int(np.count_nonzero(mask)) >= args.min_mask_area
        diagnostics.append({**group, "sam2_mask_pixels": int(np.count_nonzero(mask)), "accepted": accepted})
        if not accepted:
            continue
        color = np.random.default_rng(group["seed_label"]).integers(70, 255, 3, dtype=np.uint8)
        overlay[mask] = (.52 * overlay[mask] + .48 * color).astype(np.uint8)
        masks.append(mask.astype(np.uint8))
    if not masks:
        raise RuntimeError("SAM2 未生成可用工件掩码；检查 seed_raw.png、seed_cleaned.png 和 sam2_positive_prompts.png")
    cv2.imwrite(str(output / "sam2_masks_overlay.png"), overlay)
    np.savez_compressed(output / "masks.npz", masks=np.asarray(masks, dtype=np.uint8))
    (output / "meta.json").write_text(json.dumps({
        "method": "sam2_positive_prompt_from_tabletop_geometry_no_rgbd_mask_cut",
        "table_plane_camera": plane.tolist(), "safe_roi_xywh": [int(value) for value in args.roi],
        "seed_height_mm": args.seed_height_mm, "seed_max_gray": args.seed_max_gray,
        "seed_groups": groups, "sam2_groups": diagnostics,
        "artifacts": {"seed_raw": "seed_raw.png", "seed_cleaned": "seed_cleaned.png",
                      "positive_prompts": "sam2_positive_prompts.png", "overlay": "sam2_masks_overlay.png"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SAM2_MASKS={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
