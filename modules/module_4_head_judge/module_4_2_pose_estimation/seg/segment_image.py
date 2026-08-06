#!/usr/bin/env python3
"""Segment one image with the bundled SAM2 model and write binary PNG masks.

SAM2 is prompt-based.  In normal mode, the script estimates the local tabletop
colour, finds large darker regions whose Lab colour differs from that estimate,
then places three evenly spread interior prompts in each region.  The three
prompts for a region are sent to SAM2 together; masks from separate regions are
combined.  Use --point to provide object points manually instead.
Coordinates are normalized, so 0 0 is top-left and 1 1 is bottom-right.

Examples:
    # Segment darker objects that differ from their local tabletop background.
    uv run python my_robot/seg/segment_image.py input.png mask.png

    # Segment an object at 72% across and 56% down the image.
    uv run python my_robot/seg/segment_image.py input.png mask.png --point 0.72 0.56

    # Use a bounding-box prompt, optionally refined by foreground/background points.
    uv run python my_robot/seg/segment_image.py input.png mask.png \\
        --box 0.20 0.15 0.80 0.90 --point 0.50 0.50 --negative-point 0.10 0.10

    # Discover object contours and combine them into one binary mask.
    uv run python my_robot/seg/segment_image.py input.png auto_mask.png --auto

    # Restrict segmentation to the middle 80% of the image.
    uv run python my_robot/seg/segment_image.py input.png mask.png --roi 0.10 0.10 0.90 0.90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# Kept distinct from generic failures so the capture pipeline can stop cleanly
# when the table is empty, while still surfacing camera/model failures.
NO_WORKPIECE_EXIT_CODE = 10


class NoWorkpieceDetected(RuntimeError):
    """The configured tabletop ROI contains no valid workpiece candidate."""


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pipeline.segment_demo import (  # noqa: E402
    DEFAULT_CFG,
    DEFAULT_CKPT,
    _ensure_sam2_on_syspath,
    _get_safe_autocast_context,
    _get_safe_torch_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment one image and write a black/white binary mask")
    parser.add_argument("input", type=Path, help="input RGB image path")
    parser.add_argument(
        "output",
        type=Path,
        help="output binary mask PNG path",
    )
    parser.add_argument(
        "--point",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="normalized foreground point; may be repeated; each point is a separate object",
    )
    parser.add_argument(
        "--negative-point",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="normalized background point; may be repeated",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="normalized bounding box around the target",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="discover contours without prompts and combine them into one binary mask",
    )
    parser.add_argument(
        "--auto-points-per-side",
        type=int,
        default=32,
        help="automatic mode sampling density per image side (default: 32; higher is slower)",
    )
    parser.add_argument(
        "--background-blur-kernel",
        type=int,
        default=101,
        help="odd median-filter kernel used to estimate local tabletop colour (default: 101)",
    )
    parser.add_argument(
        "--outlier-mad-scale",
        type=float,
        default=4.0,
        help="colour-difference threshold in robust MAD units (default: 4.0)",
    )
    parser.add_argument(
        "--object-min-area",
        type=int,
        default=500,
        help="ignore dark colour-outlier regions smaller than this many pixels (default: 500)",
    )
    parser.add_argument(
        "--max-object-mask-area-ratio",
        type=float,
        default=0.15,
        help=(
            "discard an individual SAM2 object mask if it covers more than this fraction "
            "of the ROI (default: 0.15); checked before masks for separate objects are combined"
        ),
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        default=(0.08, 0.20, 0.92, 0.90),
        metavar=("X0", "Y0", "X1", "Y1"),
        help=(
            "normalized region to segment; pixels outside it are zero in the output "
            "(default: 0.08 0.20 0.92 0.90)"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="SAM2 checkpoint path")
    parser.add_argument("--config", default=DEFAULT_CFG, help="SAM2 model config")
    return parser.parse_args()


def _validate_normalized(values: list[float], name: str) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} coordinates must be in [0, 1], got {values}")


def _normalized_points_to_pixels(
    points: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    for point in points:
        _validate_normalized(point, "point")
    normalized = np.asarray(points, dtype=np.float32)
    return normalized * np.asarray([width, height], dtype=np.float32)


def _normalized_roi_to_pixel_bounds(
    roi: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a normalized ROI to non-empty slice bounds."""
    _validate_normalized(list(roi), "roi")
    x0, y0, x1, y1 = roi
    if x0 >= x1 or y0 >= y1:
        raise ValueError("--roi must satisfy X0 < X1 and Y0 < Y1")
    left = int(np.floor(x0 * width))
    top = int(np.floor(y0 * height))
    right = int(np.ceil(x1 * width))
    bottom = int(np.ceil(y1 * height))
    if left >= right or top >= bottom:
        raise ValueError("--roi is empty at this image resolution")
    return left, top, right, bottom


def _move_points_into_roi(
    points: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
    name: str,
) -> np.ndarray:
    """Validate full-image points and translate them to ROI-local coordinates."""
    left, top, right, bottom = roi_bounds
    inside = (
        (points[:, 0] >= left)
        & (points[:, 0] < right)
        & (points[:, 1] >= top)
        & (points[:, 1] < bottom)
    )
    if not np.all(inside):
        raise ValueError(f"{name} must be inside --roi")
    return points - np.asarray([left, top], dtype=np.float32)


def _representative_points(component: np.ndarray, count: int = 3) -> np.ndarray:
    """Choose moderately spread prompts from the safely interior part of a component."""
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    if not np.any(distance > 0):
        return np.empty((0, 2), dtype=np.float32)

    # Keep sampling away from the colour-mask boundary.  Sampling across all
    # foreground pixels can put farthest-point prompts on a thin edge, where a
    # small colour-mask error makes the prompt describe the tabletop instead of
    # the workpiece.
    first = np.unravel_index(np.argmax(distance), distance.shape)
    minimum_interior_distance = max(2.0, float(distance[first]) * 0.5)
    candidates = np.argwhere(distance >= minimum_interior_distance)
    selected = [np.asarray(first, dtype=np.float32)]  # row, column
    while len(selected) < min(count, len(candidates)):
        selected_array = np.asarray(selected)
        separation = np.min(
            np.sum((candidates[:, None, :] - selected_array[None, :, :]) ** 2, axis=2), axis=1
        )
        # Spread prompts only within that safe interior core.
        candidate = candidates[int(np.argmax(separation))].astype(np.float32)
        selected.append(candidate)
    return np.asarray([(column, row) for row, column in selected], dtype=np.float32)


def _find_object_point_groups(
    image_bgr: np.ndarray,
    min_area: int,
    background_blur_kernel: int,
    outlier_mad_scale: float,
) -> list[np.ndarray]:
    """Return three interior prompts for each large, locally-dark colour outlier."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    # Median filtering is robust to isolated workpieces and follows gradual
    # illumination changes across the tabletop better than a fixed grey range.
    local_background = cv2.medianBlur(lab, background_blur_kernel)
    colour_difference = np.linalg.norm(
        lab.astype(np.float32) - local_background.astype(np.float32), axis=2
    )
    median = float(np.median(colour_difference))
    mad = float(np.median(np.abs(colour_difference - median)))
    robust_sigma = max(1.4826 * mad, 1.0)
    threshold = median + outlier_mad_scale * robust_sigma
    # Workpieces are darker than their nearby tabletop.  The signed lightness
    # check avoids selecting bright highlights that also differ in colour.
    darker_than_background = local_background[:, :, 0] > lab[:, :, 0]
    foreground = np.where(
        (colour_difference >= threshold) & darker_than_background, 255, 0
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    groups: list[np.ndarray] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, width, height, _ = stats[label]
        # A lighting or tabletop variation that reaches the ROI edge is not an
        # isolated workpiece. The selected ROI intentionally leaves a margin
        # around the work area, so discard those background-connected regions.
        if x == 0 or y == 0 or x + width == labels.shape[1] or y + height == labels.shape[0]:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        points = _representative_points(component)
        if len(points):
            groups.append(points)
    if not groups:
        raise NoWorkpieceDetected(
            f"no dark Lab-colour outliers found with area >= {min_area}; "
            "lower --outlier-mad-scale or --object-min-area, or pass --point X Y"
        )
    return groups


def _predict_masks_for_points(
    predictor,
    image_rgb: np.ndarray,
    torch,
    device: str,
    positive_point_groups: list[np.ndarray],
    negative_points: np.ndarray | None,
    prompt_box: np.ndarray | None,
    max_object_mask_area_ratio: float,
) -> tuple[np.ndarray, int]:
    """Predict masks per object, discarding implausibly table-sized results."""
    height, width = image_rgb.shape[:2]
    negative_coords = (
        negative_points
        if negative_points is not None
        else np.empty((0, 2), dtype=np.float32)
    )
    combined = np.zeros((height, width), dtype=bool)
    rejected = 0
    max_object_pixels = max_object_mask_area_ratio * height * width
    with torch.inference_mode(), _get_safe_autocast_context(torch, device):
        predictor.set_image(image_rgb)
        for group_index, positive_points in enumerate(positive_point_groups, start=1):
            point_coords = np.vstack((positive_points, negative_coords))
            point_labels = np.asarray([*([1] * len(positive_points)), *([0] * len(negative_coords))], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None if prompt_box is None else prompt_box[None, :],
                multimask_output=True,
            )
            object_mask = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
            object_pixels = int(np.count_nonzero(object_mask))
            if object_pixels > max_object_pixels:
                rejected += 1
                print(
                    f"[warning] rejected object {group_index}/{len(positive_point_groups)}: "
                    f"SAM2 mask covers {object_pixels / (height * width):.1%} of ROI "
                    f"(limit {max_object_mask_area_ratio:.1%})",
                    file=sys.stderr,
                )
                continue
            combined |= object_mask
    if not np.any(combined):
        raise RuntimeError(
            "all SAM2 object masks exceeded --max-object-mask-area-ratio; "
            "raise the limit only if an individual workpiece genuinely occupies that much of the ROI"
        )
    return combined, rejected


def _write_mask(
    combined: np.ndarray,
    output_path: Path,
    full_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
) -> None:
    """Write an original-size mask with only the ROI eligible for foreground."""
    left, top, right, bottom = roi_bounds
    mask = np.zeros(full_shape, dtype=np.uint8)
    mask[top:bottom, left:right] = np.where(combined, 255, 0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mask):
        raise RuntimeError(f"failed to write mask: {output_path}")


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"input image does not exist: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {args.checkpoint}")
    if args.auto and (args.point is not None or args.negative_point is not None or args.box is not None):
        raise ValueError("--auto cannot be combined with --point, --negative-point, or --box")
    if args.auto_points_per_side <= 0:
        raise ValueError("--auto-points-per-side must be positive")
    if args.background_blur_kernel < 3 or args.background_blur_kernel % 2 == 0:
        raise ValueError("--background-blur-kernel must be an odd integer >= 3")
    if args.outlier_mad_scale <= 0:
        raise ValueError("--outlier-mad-scale must be positive")
    if args.object_min_area <= 0:
        raise ValueError("--object-min-area must be positive")
    if not 0.0 < args.max_object_mask_area_ratio <= 1.0:
        raise ValueError("--max-object-mask-area-ratio must be in (0, 1]")
    if args.box is not None:
        _validate_normalized(args.box, "box")
        x0, y0, x1, y1 = args.box
        if x0 >= x1 or y0 >= y1:
            raise ValueError("--box must satisfy X0 < X1 and Y0 < Y1")

    image_bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"could not decode input image: {args.input}")
    height, width = image_bgr.shape[:2]
    roi_bounds = _normalized_roi_to_pixel_bounds(args.roi, width, height)
    left, top, right, bottom = roi_bounds
    roi_bgr = image_bgr[top:bottom, left:right]
    _ensure_sam2_on_syspath()
    import torch
    from sam2.build_sam import build_sam2

    device = _get_safe_torch_device(torch)
    model = build_sam2(args.config, str(args.checkpoint.resolve()), device=device)
    image_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    if args.auto:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        generator = SAM2AutomaticMaskGenerator(model, points_per_side=args.auto_points_per_side)
        with torch.inference_mode(), _get_safe_autocast_context(torch, device):
            annotations = generator.generate(image_rgb)
        combined = np.zeros(image_rgb.shape[:2], dtype=bool)
        for annotation in annotations:
            combined |= np.asarray(annotation["segmentation"], dtype=bool)
        if not np.any(combined):
            raise NoWorkpieceDetected("SAM2 automatic segmentation returned an empty mask")
        _write_mask(
            combined, args.output, (height, width), roi_bounds
        )
        print(
            f"[done] device={device} contours={len(annotations)} roi={args.roi} "
            f"mask={args.output} foreground_pixels={int(np.count_nonzero(combined))}"
        )
        return 0

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    prompt_box = None
    if args.box is not None:
        prompt_box = np.asarray(args.box, dtype=np.float32) * np.asarray(
            [width, height, width, height], dtype=np.float32
        )
        if (
            prompt_box[0] < left
            or prompt_box[1] < top
            or prompt_box[2] > right
            or prompt_box[3] > bottom
        ):
            raise ValueError("--box must be fully inside --roi")
        prompt_box -= np.asarray([left, top, left, top], dtype=np.float32)
    if args.point is not None:
        manual_points = _move_points_into_roi(
            _normalized_points_to_pixels(args.point, width, height), roi_bounds, "--point"
        )
        # Preserve the documented behaviour: repeated --point values are separate objects.
        positive_point_groups = [point[None, :] for point in manual_points]
        point_source = "manual"
    elif args.box is not None:
        x0, y0, x1, y1 = prompt_box
        positive_point_groups = [
            np.asarray([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]], dtype=np.float32)
        ]
        point_source = "box-centre"
    else:
        positive_point_groups = _find_object_point_groups(
            roi_bgr,
            args.object_min_area,
            args.background_blur_kernel,
            args.outlier_mad_scale,
        )
        point_source = f"local Lab outlier (MAD×{args.outlier_mad_scale:g}), 3/region"

    predictor = SAM2ImagePredictor(model)
    negative_points = None
    if args.negative_point is not None:
        negative_points = _move_points_into_roi(
            _normalized_points_to_pixels(args.negative_point, width, height), roi_bounds, "--negative-point"
        )
    combined, rejected_masks = _predict_masks_for_points(
        predictor,
        image_rgb,
        torch,
        device,
        positive_point_groups,
        negative_points,
        prompt_box,
        args.max_object_mask_area_ratio,
    )
    if not np.any(combined):
        raise NoWorkpieceDetected("SAM2 segmentation returned an empty mask")
    _write_mask(combined, args.output, (height, width), roi_bounds)
    print(
        f"[done] device={device} regions={len(positive_point_groups)} "
        f"points={sum(len(points) for points in positive_point_groups)} rejected={rejected_masks} "
        f"source={point_source} roi={args.roi} "
        f"mask={args.output} foreground_pixels={int(np.count_nonzero(combined))}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
    except NoWorkpieceDetected as exc:
        print(f"[no-workpiece] {exc}", file=sys.stderr)
        raise SystemExit(NO_WORKPIECE_EXIT_CODE)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
