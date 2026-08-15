"""Depth- and colour-constrained automatic SAM2 prompts for Module 5-2."""
from __future__ import annotations

import cv2
import numpy as np

from tool.inspection_support.sam2_icp import prompt_segment


def _spread(region: np.ndarray, count: int = 3) -> list[list[int]]:
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    points: list[list[int]] = []
    for _ in range(count):
        y, x = np.unravel_index(int(distance.argmax()), distance.shape)
        if distance[y, x] < 4.0:
            break
        points.append([int(x), int(y)])
        cv2.circle(distance, (int(x), int(y)), 38, 0.0, -1)
    return points


def _tail_seed(
    main: np.ndarray,
    band: np.ndarray,
    depth_raw: np.ndarray,
    colour_distance: np.ndarray,
    primary_depth: float,
) -> list[int] | None:
    """Find a dark, depth-consistent tail next to the colour-selected body.

    The first-stage body is intentionally colour biased, so it misses the
    workpiece's dark tail beside the equally dark gripper.  Search only in a
    narrow neighbourhood of that body, reject pixels that look exactly like
    the gripper reference, and require a depth close to the body.  This is a
    *seed* for a second SAM2 call, not a mask expansion by itself.
    """
    near = cv2.dilate(
        main.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (151, 151)),
    ) > 0
    outside_body = cv2.dilate(main.astype(np.uint8), np.ones((11, 11), np.uint8)) == 0
    same_depth = np.abs(depth_raw.astype(np.float32) - primary_depth) <= 60.0
    # 30 is deliberately lower than the green-body threshold.  It retains the
    # dark-grey tail (which differs from the black gripper) while excluding the
    # gripper reference colour itself.
    tail_pixels = near & outside_body & band & same_depth & (colour_distance >= 30.0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        tail_pixels.astype(np.uint8), 8
    )
    touching = cv2.dilate(main.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    body_xy = np.column_stack(np.where(main))[:, ::-1].astype(np.float32)
    body_center = body_xy.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(
        body_xy - body_center, full_matrices=False
    )
    # SVD direction has arbitrary sign.  Use image-left to image-right as the
    # stable convention for this fixed head-camera view, which also gives a
    # positive forward direction for the held part's tail.
    axis = right_vectors[0]
    if axis[0] < 0.0:
        axis = -axis
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)
    choices: list[tuple[float, int]] = []
    for label in range(1, count):
        part = labels == label
        area = int(part.sum())
        if area < 20:
            continue
        contact = int((part & touching).sum())
        if contact == 0:
            continue
        x, y, width, height, _ = stats[label]
        center = np.array([x + width / 2.0, y + height / 2.0], dtype=np.float32)
        forward = float(np.dot(center - body_center, axis))
        sideways = abs(float(np.dot(center - body_center, perpendicular)))
        # A tail must extend past the right/forward end of the colour-selected
        # body, roughly along its long axis.  This rejects the black gripper
        # hardware at the side/back which previously won by contact area.
        if forward < 35.0 or sideways > 80.0:
            continue
        score = forward * 10.0 - sideways * 2.0 + min(area, 800) * 0.2
        choices.append((score, label))
    if not choices:
        return None
    selected = labels == max(choices)[1]
    points = _spread(selected, count=1)
    return points[0] if points else None


def _connected_to(mask: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Keep only SAM2 components that touch the known first-stage body."""
    count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = [label for label in range(1, count) if np.any((labels == label) & anchor)]
    return np.isin(labels, keep)


def segment_workpiece(
    image_bgr: np.ndarray, depth_raw: np.ndarray, depth_scale_m: float, sam2_url: str,
    *, min_depth_m: float = 0.25, max_depth_m: float = 0.40,
) -> tuple[np.ndarray, list[int], list[list[int]], list[list[int]]]:
    """Return depth-constrained SAM2 mask without fixed/manual prompt points."""
    depth_m = depth_raw.astype(np.float32) * float(depth_scale_m)
    band = (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    if int(band.sum()) < 200:
        raise RuntimeError("0.25-0.40m depth band has too few valid pixels")
    h, w = band.shape
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lower = band & (np.indices(band.shape)[0] >= int(h * 0.32))
    luminance = lab[..., 0]
    dark_cut = np.percentile(luminance[lower], 22) if lower.any() else np.percentile(luminance[band], 22)
    arm = lower & (luminance <= dark_cut)
    if int(arm.sum()) < 500:
        arm = band & (luminance <= np.percentile(luminance[band], 22))
    arm_colour = np.median(lab[arm], axis=0)
    delta = np.linalg.norm(lab - arm_colour, axis=2)
    threshold = max(34.0, float(np.percentile(delta[band], 55)))
    different = band & (delta >= threshold)
    different = cv2.morphologyEx(different.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(different.astype(np.uint8), 8)
    near_arm = cv2.dilate(arm.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    choices = []
    for label in range(1, count):
        part = labels == label
        area = int(part.sum())
        if area < 180:
            continue
        x, y, cw, ch, _ = stats[label]
        score = int((part & near_arm).sum()) * 3.0 + min(area, 15000) * .08 - .003 * ((x + cw / 2 - w / 2) ** 2 + (y + ch / 2 - h * .60) ** 2)
        choices.append((score, label))
    if not choices:
        raise RuntimeError("no colour-contrasted workpiece depth component")
    primary = labels == max(choices)[1]
    primary_depth = float(np.median(depth_raw[primary]))
    nearby = cv2.dilate(primary.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    keep = {max(choices)[1]}
    for label in range(1, count):
        part = labels == label
        if label not in keep and int(part.sum()) >= 75 and np.any(part & nearby) and abs(float(np.median(depth_raw[part])) - primary_depth) <= 55:
            keep.add(label)
    core = np.isin(labels, list(keep))
    expanded = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    candidate = band & expanded & (delta >= threshold * .72)
    _, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    main = final_labels == (1 + int(np.argmax(final_stats[1:, cv2.CC_STAT_AREA])))
    positives = _spread(main)
    if not positives:
        raise RuntimeError("workpiece candidate has no interior prompt point")
    ys, xs = np.where(main)
    box = [max(0, int(xs.min()) - 25), max(0, int(ys.min()) - 25), min(w - 1, int(xs.max()) + 25), min(h - 1, int(ys.max()) + 25)]
    ring = cv2.dilate(main.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
    exclusion = cv2.dilate(main.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    negatives = _spread(arm & ring & ~exclusion, 3)
    raw = prompt_segment(image_bgr, box, positives, negatives, sam2_url) > 0
    allowed = cv2.dilate(main.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    first_stage = main | (raw & allowed)

    tail = _tail_seed(main, band, depth_raw, delta, primary_depth)
    if tail is None:
        return first_stage.astype(np.uint8) * 255, box, positives, negatives

    # The first-stage box is fitted tightly to the green body.  Extend only
    # toward the detected tail seed; otherwise SAM2 ignores an out-of-box
    # positive point and cannot recover the tail.
    tail_box = [
        min(box[0], max(0, tail[0] - 55)),
        min(box[1], max(0, tail[1] - 55)),
        max(box[2], min(w - 1, tail[0] + 55)),
        max(box[3], min(h - 1, tail[1] + 55)),
    ]
    second_raw = prompt_segment(
        image_bgr, tail_box, positives + [tail], negatives, sam2_url
    ) > 0
    second_stage = _connected_to(second_raw, first_stage)
    if not np.any(second_stage):
        return first_stage.astype(np.uint8) * 255, box, positives, negatives
    return second_stage.astype(np.uint8) * 255, tail_box, positives + [tail], negatives
