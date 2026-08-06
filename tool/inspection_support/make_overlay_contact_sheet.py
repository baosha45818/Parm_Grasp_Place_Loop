#!/usr/bin/env python3
"""Create a labelled contact sheet from RGB-plus-mask overlay PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--glob", default="*_rgb_sam2_mask_overlay.png")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(args.glob))
    if not paths:
        raise RuntimeError("No RGB+SAM2 overlay PNGs found")
    tiles: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        height = round(image.shape[0] * args.tile_width / image.shape[1])
        tile = cv2.resize(image, (args.tile_width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (args.tile_width, 28), (0, 0, 0), -1)
        cv2.putText(tile, path.stem.replace("_rgb_sam2_mask_overlay", ""), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = (len(tiles) + args.columns - 1) // args.columns
    tile_h = tiles[0].shape[0]
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * args.columns - len(tiles)))
    sheet = np.vstack([
        np.hstack(tiles[row * args.columns:(row + 1) * args.columns])
        for row in range(rows)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"wrote {args.output}; overlays={len(paths)}")


if __name__ == "__main__":
    main()
