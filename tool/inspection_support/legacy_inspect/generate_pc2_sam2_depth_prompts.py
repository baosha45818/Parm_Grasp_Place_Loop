#!/usr/bin/env python3
"""Batch-generate PC2 SAM2 masks from the URDF-cleaned near-depth region.

Positive points are selected only from broad, interior valid-depth pixels that
are outside the complete projected robot silhouette.  Negative points are
sampled uniformly from an outside ring surrounding that processed region.
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import socket
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .remove_pc2_urdf_gripper_depth import (
    PC2_DIR, load_self_surface_points, matrix, newest_snapshot, projected_gripper_depth,
)

# PC2's camera and left arm pose are fixed.  These prompts explicitly teach
# SAM2 that the arm, closed gripper and supporting tabletop are never target.
FIXED_NEGATIVES_PATH = Path(__file__).with_name("pc2_fixed_negative_points.json")
# Fixed PC2 image area containing the ungrasped workpiece pile (x0, y0, x1, y1).
FAR_PILE_ROI = (240, 280, 430, 450)


def load_fixed_negatives() -> dict[str, list[list[int]]]:
    points = json.loads(FIXED_NEGATIVES_PATH.read_text(encoding="utf-8"))
    if not isinstance(points, dict):
        raise ValueError(f"Invalid fixed-negative prompt file: {FIXED_NEGATIVES_PATH}")
    return {str(category): [[int(x), int(y)] for x, y in category_points]
            for category, category_points in points.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc2-dir", type=Path, default=PC2_DIR)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PC2_DIR / "sam2_depth_prompt_all")
    parser.add_argument("--sam2-url", default="unix:///tmp/sam2-service.sock")
    parser.add_argument("--near-min-m", type=float, default=0.110)
    parser.add_argument("--near-max-m", type=float, default=0.280)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.035)
    parser.add_argument("--mesh-samples", type=int, default=200000)
    parser.add_argument("--projected-exclude-px", type=int, default=7)
    parser.add_argument("--positive-count", type=int, default=6)
    return parser.parse_args()


def sam2(image: np.ndarray, box: list[int], positives: list[list[int]], negatives: list[list[int]], url: str) -> np.ndarray:
    payload = io.BytesIO()
    points = np.asarray(positives + negatives, dtype=np.int32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    np.savez_compressed(payload, image_bgr=image, points=points, labels=labels, box=np.asarray(box, dtype=np.int32))
    endpoint = urlparse(url)
    if endpoint.scheme != "unix" or not endpoint.path:
        raise ValueError("Only unix:///path/to/sam2.sock is supported by this batch tool")

    class UnixConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(endpoint.path)

    connection = UnixConnection("sam2-service", timeout=90)
    try:
        connection.request("POST", "/v1/prompt-segment-sam2", body=payload.getvalue(),
                           headers={"Content-Type": "application/x-sam2-prompt-npz"})
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM2 returned {response.status}: {body.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as result:
        return (result["mask"] > 0).astype(np.uint8)


def usable_components(
    mask: np.ndarray, left_central_min_area: int = 7000
) -> np.ndarray:
    """Keep every substantial neighbouring component, not just the largest one."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    # A held part can be split by the occluding hand.  Retain meaningful pieces
    # instead of discarding all but one, but reject speckles and depth noise.
    min_area = max(350, largest // 25)
    keep = []
    height, width = mask.shape
    for label in np.flatnonzero(areas >= min_area) + 1:
        x, y, w, h, _ = stats[int(label)]
        # The fixed arm enters from the image border.  A held workpiece must
        # remain a self-contained depth island after the projected arm is cut.
        # This rejects residual arm pieces even if URDF/depth calibration is a
        # few pixels off, without discarding the separate workpiece islands.
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        # PC2 has a fixed view: the robot approaches from the left and the
        # held part is on its right.  A small left/central island is therefore
        # residual arm depth; a large one is retained because a real part can
        # bridge behind the gripper and have a left visible lobe.
        if (
            x + w * 0.5 < width * 0.47
            and int(stats[int(label), cv2.CC_STAT_AREA]) < left_central_min_area
        ):
            continue
        keep.append(int(label))
    return np.isin(labels, keep)


def wide_depth_core(mask: np.ndarray, min_half_width_px: float = 10.0) -> np.ndarray:
    """Keep only thick depth interiors for ROI construction.

    A narrow arm-side bridge has no pixel this far from its edge, so it cannot
    extend the ROI into the ungrasped workpiece pile.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    core = distance >= min_half_width_px
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    # Reject isolated distance-transform specks while retaining every broad
    # visible lobe of the held part.
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 50) + 1
    return np.isin(labels, keep)


def positive_points(region: np.ndarray, count: int) -> list[list[int]]:
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[list[int]] = []
    work = distance.copy()
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[y, x] < 8.0:  # never select the narrow residual strips
            break
        points.append([int(x), int(y)])
        cv2.circle(work, (int(x), int(y)), 42, 0.0, -1)
    return points


def fallback_working_area(region: np.ndarray) -> tuple[np.ndarray, list[int] | None]:
    """Select one conservative PC2 working-area point when normal selection is empty."""
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    candidates = distance >= 5.0
    ys, xs = np.where(candidates)
    if not xs.size:
        return np.zeros_like(region, dtype=bool), None
    # The fixed closed gripper presents the held part at this image location.
    target_x, target_y = 590, 505
    selected = int(np.argmin((xs - target_x) ** 2 + (ys - target_y) ** 2))
    x, y = int(xs[selected]), int(ys[selected])
    count, labels, _, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    label = int(labels[y, x])
    if label == 0 or count <= 1:
        return np.zeros_like(region, dtype=bool), None
    return labels == label, [x, y]


def far_depth_negative_points(depth_m: np.ndarray, near_max_m: float, count: int = 3) -> list[list[int]]:
    """Pick far-depth negatives only from the fixed ungrasped-workpiece pile."""
    far = (depth_m >= near_max_m + 0.05) & (depth_m <= 1.50)
    x0, y0, x1, y1 = FAR_PILE_ROI
    pile_roi = np.zeros_like(far, dtype=bool)
    pile_roi[max(0, y0):min(far.shape[0], y1), max(0, x0):min(far.shape[1], x1)] = True
    far &= pile_roi
    distance = cv2.distanceTransform(far.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[list[int]] = []
    work = distance.copy()
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[y, x] < 5.0:
            break
        points.append([int(x), int(y)])
        cv2.circle(work, (int(x), int(y)), 70, 0.0, -1)
    return points


def keep_main_mask_depth_consistent(
    mask: np.ndarray, positives: list[list[int]], depth_m: np.ndarray, depth_region: np.ndarray,
    filtered_near_region: np.ndarray,
) -> np.ndarray:
    """Keep internal mask pixels; depth-filter only external SAM2 expansion.

    The depth test is deliberately applied only outside the filtered near-depth
    ROI.  Internal holes are filled before returning, so a depth hole inside a
    workpiece cannot punch a hole in the final mask.
    """
    if not np.any(mask):
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep_labels = {int(labels[y, x]) for x, y in positives if labels[y, x] != 0}
    if not keep_labels and count > 1:
        keep_labels = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    main_mask = np.isin(labels, list(keep_labels))
    # The fixed left pile is an explicit negative zone.  Only pixels supported
    # by the positive-associated depth core itself may remain there; this
    # rejects SAM2 spillover into ungrasped workpieces even when it is joined by
    # a thin RGB bridge to the held part.
    x0, y0, x1, y1 = FAR_PILE_ROI
    pile_zone = np.zeros_like(main_mask, dtype=bool)
    pile_zone[max(0, y0):min(main_mask.shape[0], y1),
              max(0, x0):min(main_mask.shape[1], x1)] = True
    main_mask &= ~(pile_zone & ~depth_region)
    reference = depth_m[depth_region & (depth_m > 0)]
    if reference.size < 30:
        reference = depth_m[main_mask & (depth_m > 0)]
    if reference.size < 30:
        return fill_mask_holes(main_mask)
    low, high = np.percentile(reference, [2.0, 98.0])
    # The held object can have sloped faces; retain a 3 cm envelope around the
    # robust main-depth span, but discard clearly different-depth intrusions.
    depth_ok = (depth_m <= 0) | ((depth_m >= low - 0.030) & (depth_m <= high + 0.030))
    # Protect exactly the processed near-depth mask itself.  Pixels outside it
    # remain eligible for the main-depth consistency filter; do not dilate this
    # protection zone, otherwise external false positives escape filtering.
    protected_near = filtered_near_region.astype(bool)
    kept = main_mask & (protected_near | depth_ok)
    # Preserve every enclosed SAM2 hole as part of the workpiece mask.
    return (kept | fill_mask_holes(main_mask)).astype(np.uint8)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only enclosed holes; do not bridge disconnected external masks."""
    source = (mask.astype(np.uint8) * 255)
    flood = source.copy()
    flood_mask = np.zeros((source.shape[0] + 2, source.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = (flood == 0) & (source == 0)
    return ((source > 0) | holes).astype(np.uint8)


def ring_points(candidate: np.ndarray, region: np.ndarray, max_points: int) -> list[list[int]]:
    """Return one nearest point from each angular sector around ``region``."""
    ys, xs = np.where(candidate & ~region)
    if not xs.size:
        return []
    cy, cx = np.mean(np.where(region), axis=1)
    angles = (np.arctan2(ys - cy, xs - cx) + 2 * np.pi) % (2 * np.pi)
    radii = (xs - cx) ** 2 + (ys - cy) ** 2
    result: list[list[int]] = []
    # One closest point in each sector makes an evenly distributed prompt ring.
    for index in range(max_points):
        start, end = 2 * np.pi * index / max_points, 2 * np.pi * (index + 1) / max_points
        valid = (angles >= start) & (angles < end)
        if not np.any(valid):
            continue
        candidates = np.flatnonzero(valid)
        chosen = candidates[int(np.argmin(radii[candidates]))]
        point = [int(xs[chosen]), int(ys[chosen])]
        if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 >= 14 ** 2 for old in result):
            result.append(point)
    return result


def exterior_background(depth_mask: np.ndarray) -> np.ndarray:
    """Return only zero-depth background connected to the image border.

    Depth holes inside a part are zero depth too, but are not visually outside
    the depth layer and must never receive a red negative prompt.
    """
    count, labels, _, _ = cv2.connectedComponentsWithStats((~depth_mask).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(depth_mask, dtype=bool)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return np.isin(labels, border_labels[border_labels != 0])


def process(capture: Path, snapshot: dict, points_tool: np.ndarray, args: argparse.Namespace) -> dict:
    rgb = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    depth = np.load(capture / "depth_aligned_000.npy", allow_pickle=False)
    metadata = json.loads((capture / "metadata.json").read_text())
    scale = float(metadata["depth_scale_m_per_unit"])
    K = np.asarray(metadata["color_intrinsics"], dtype=np.float64)
    T_camera_tool = np.linalg.inv(matrix(snapshot["T_body_base_chest_camera"], "camera pose")) @ matrix(
        snapshot["T_body_base_left_hand_transfer_end"], "tool pose")
    rendered = projected_gripper_depth(points_tool, T_camera_tool, K, depth.shape)
    projected = np.isfinite(rendered)
    measured = depth.astype(np.float32) * scale
    depth_matched_self = projected & (measured > 0) & (np.abs(measured - rendered) <= args.depth_tolerance_m)
    kernel = np.ones((2 * args.projected_exclude_px + 1,) * 2, np.uint8)
    projected_exclusion = cv2.dilate(projected.astype(np.uint8), kernel) > 0
    # Raw near-depth is the first usable depth layer after the depth-consistent
    # URDF removal.  Keep it separately: it defines the two negative classes.
    raw_near = (measured >= args.near_min_m) & (measured <= args.near_max_m) & ~depth_matched_self
    # Depth agreement may miss a piece of arm due calibration.  The full URDF
    # silhouette is used only for prompts, so an arm cannot become a positive.
    source_region = usable_components(raw_near & ~projected_exclusion)
    region = wide_depth_core(source_region)
    positives = positive_points(region, args.positive_count)
    fallback_used = False
    if positives:
        # A far workpiece-pile core may also satisfy the near-depth test.  It
        # has no positive point, so it is not part of the currently held part.
        _, core_labels, _, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
        positive_labels = {int(core_labels[y, x]) for x, y in positives if core_labels[y, x] != 0}
        region = np.isin(core_labels, list(positive_labels))
        # Re-expand only from positive-associated cores through original depth.
        # This retains the held part's broad left lobe while cutting the narrow
        # gripper strip and every unprompted workpiece-pile component.
        # Horizontal left/right lobes belong to the held part, while the
        # troublesome gripper strip and workpiece pile are mainly vertical.
        # Use an anisotropic expansion: 45 px horizontal, only 25 px vertical.
        core_neighbourhood = cv2.dilate(
            region.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 51)),
        ) > 0
        roi_region = source_region & core_neighbourhood
    else:
        fallback_region, fallback_point = fallback_working_area(raw_near & ~projected_exclusion)
        if fallback_point is not None:
            region, roi_region, positives, fallback_used = fallback_region, fallback_region, [fallback_point], True
        else:
            roi_region = np.zeros_like(source_region, dtype=bool)
    outside_distance = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)
    height, width = depth.shape
    fixed_negative_groups = load_fixed_negatives()
    fixed_negatives = [point for group in fixed_negative_groups.values() for point in group
                       if 0 <= point[0] < width and 0 <= point[1] < height]
    far_depth_negatives = far_depth_negative_points(measured, args.near_max_m)
    negatives = fixed_negatives + far_depth_negatives
    overlay = rgb.copy()
    if np.any(roi_region):
        contours, _ = cv2.findContours(roi_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 200, 0), 2, cv2.LINE_AA)
    if positives:
        ys, xs = np.where(roi_region)
        box = [max(0, int(xs.min()) - 30), max(0, int(ys.min()) - 30),
               min(rgb.shape[1] - 1, int(xs.max()) + 30), min(rgb.shape[0] - 1, int(ys.max()) + 30)]
        raw_mask = sam2(rgb, box, positives, negatives, args.sam2_url)
        mask = keep_main_mask_depth_consistent(raw_mask, positives, measured, region, roi_region)
        color = overlay.copy()
        color[mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(color, 0.48, overlay, 0.52, 0)
        cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    else:
        mask = np.zeros(depth.shape, np.uint8)
        box = None
    for x, y in positives:
        cv2.circle(overlay, (x, y), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 9, (0, 80, 0), 1, cv2.LINE_AA)
    for x, y in fixed_negatives:
        cv2.drawMarker(overlay, (x, y), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 15, 2, cv2.LINE_AA)
    for x, y in far_depth_negatives:
        cv2.drawMarker(overlay, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2, cv2.LINE_AA)
    text = (f"SAM2: positives={len(positives)}{' fallback' if fallback_used else ''}, "
            f"orange-fixed={len(fixed_negatives)}, purple-pile-far={len(far_depth_negatives)}")
    cv2.putText(overlay, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, .60, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(overlay, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, .60, (0, 0, 0), 1, cv2.LINE_AA)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{capture.name}_sam2_depth_prompt_overlay.png"
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Cannot write {path}")
    return {"capture": capture.name, "overlay": path.name, "positives": positives,
            "fixed_negatives": fixed_negative_groups,
            "far_depth_negatives": far_depth_negatives,
            "fallback_positive": fallback_used,
            "box": box, "region_px": int(region.sum()), "roi_region_px": int(roi_region.sum()),
            "sam2_mask_px": int(mask.sum())}


def main() -> None:
    args = parse_args()
    snapshot_path = args.snapshot or newest_snapshot(args.pc2_dir)
    snapshot = json.loads(snapshot_path.read_text())
    points_tool = load_self_surface_points(Path(snapshot["urdf_root"]), float(snapshot["urdf_active_joint_rad"]),
                                          args.mesh_samples,
                                          [-10.496, -31.557, -4.878, 135.978, -99.147, 5.249, 45.166])
    records = [process(path, snapshot, points_tool, args)
               for path in sorted(args.pc2_dir.glob("rgbd_*")) if path.is_dir()]
    (args.output_dir / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Generated {len(records)} overlays in {args.output_dir}")


if __name__ == "__main__":
    main()
