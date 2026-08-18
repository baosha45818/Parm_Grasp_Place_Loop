#!/usr/bin/env python3
"""Save an inspectable visualization of a ``/v1/segment`` response.

This calls the resident service, so it exercises the same AMG path as the
traditional picker.  The output intentionally contains both every returned
candidate and their area/quality metadata; it is not a post-processed grasp
visualization.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np

from uds_http_client import segment_full_image_top_n


HERE = Path(__file__).resolve().parent


def _color(index: int) -> tuple[int, int, int]:
    # BGR; deterministic colours make comparison between runs easy.
    return ((53 * index + 47) % 256, (97 * index + 89) % 256, (181 * index + 131) % 256)


def _label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _mask_montage(image_bgr: np.ndarray, masks: np.ndarray, areas: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    tiles: list[np.ndarray] = []
    for index, mask in enumerate(masks, start=1):
        tile = image_bgr.copy()
        color = np.asarray(_color(index), dtype=np.uint8)
        chosen = mask.astype(bool)
        tile[chosen] = (0.35 * tile[chosen] + 0.65 * color).astype(np.uint8)
        tile[~chosen] = (0.18 * tile[~chosen]).astype(np.uint8)
        _label(tile, f"#{index}: {int(areas[index - 1])} px", (12, 25))
        tiles.append(tile)
    if not tiles:
        return image_bgr.copy()
    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    tiles.extend([blank] * (rows * columns - len(tiles)))
    return np.vstack([np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows)])


async def run(args: argparse.Namespace) -> None:
    image = np.load(args.image_npy)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3 BGR image, got {image.shape} {image.dtype}")
    response = await segment_full_image_top_n(
        image, socket_path=args.socket, num_masks=args.num_masks,
        min_mask_area_px=args.min_mask_area_px, timeout_s=args.timeout_s,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "input_bgr.png"), image)

    overlay = image.copy()
    records = []
    for index, mask in enumerate(response.masks, start=1):
        selected = mask.astype(bool)
        color = np.asarray(_color(index), dtype=np.uint8)
        overlay[selected] = (0.55 * overlay[selected] + 0.45 * color).astype(np.uint8)
        x0, y0, x1, y1 = np.rint(response.bboxes[index - 1]).astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _color(index), 2)
        _label(overlay, f"#{index}", (max(0, x0), max(18, y0 - 5)))
        records.append({
            "rank": index,
            "source_index": int(response.source_index[index - 1]),
            "area_px": int(response.area_px[index - 1]),
            "raw_area_px": int(response.raw_area_px[index - 1]),
            "bbox_xyxy": [float(v) for v in response.bboxes[index - 1]],
            "predicted_iou": float(response.predicted_iou[index - 1]),
            "stability_score": float(response.stability_score[index - 1]),
        })
    cv2.imwrite(str(output / "ranked_masks_overlay.png"), overlay)
    cv2.imwrite(str(output / "ranked_masks_montage.png"), _mask_montage(image, response.masks, response.area_px))
    (output / "segment_report.json").write_text(json.dumps({
        "request": {"num_masks": args.num_masks, "min_mask_area_px": args.min_mask_area_px},
        "service_metadata": response.metadata,
        "candidates": records,
    }, indent=2) + "\n")
    print(f"wrote {len(records)} ranked service masks to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-npy", type=Path, required=True, help="saved uint8 BGR HxWx3 .npy image")
    parser.add_argument("--socket", type=Path, default=HERE / "runtime/sam2-service.sock")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--num-masks", type=int, default=32)
    parser.add_argument("--min-mask-area-px", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args = parser.parse_args()
    if args.num_masks < 1 or args.min_mask_area_px < 1:
        parser.error("--num-masks and --min-mask-area-px must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
