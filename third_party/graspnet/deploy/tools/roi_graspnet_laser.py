#!/usr/bin/env python3
"""Select an aligned Femto Bolt ROI and print Version5 command arguments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_FEMTO_SERIAL = "CL8L56301LK"


def build_parser() -> argparse.ArgumentParser:
    default_display = os.environ.get("DISPLAY")
    if not default_display and Path("/tmp/.X11-unix/X1").exists():
        default_display = ":1"
    parser = argparse.ArgumentParser(
        description="在奥比中光彩色图上框选与深度图对齐的 GraspNet ROI。"
    )
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--display", default=default_display)
    parser.add_argument("--femto-serial", default=DEFAULT_FEMTO_SERIAL)
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    return parser


def select_roi(args: argparse.Namespace) -> tuple[int, int, int, int]:
    if not args.display:
        raise RuntimeError("ROI 选择需要 X11；请传 --display :1，或直接传 --roi X Y W H")
    os.environ["DISPLAY"] = str(args.display)
    os.environ.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))

    import cv2

    sys.path.insert(0, str(ROOT / "utils"))
    from robot_camera import FemtoBoltCamera

    with FemtoBoltCamera(
        serial=args.femto_serial,
        color_width=args.color_width,
        color_height=args.color_height,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
    ) as camera:
        frame = camera.read()

    if frame.color.shape[:2] != frame.depth.shape:
        raise RuntimeError(
            "Femto 彩色图与深度图尚未对齐："
            f"color={frame.color.shape[:2]} depth={frame.depth.shape}；不能生成可靠 ROI"
        )
    canvas = frame.color.copy()
    cv2.putText(
        canvas,
        "Drag ROI, ENTER/SPACE confirm, C cancel",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    roi = cv2.selectROI("Femto Bolt ROI for GraspNet", canvas, True, False)
    cv2.destroyAllWindows()
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise RuntimeError("ROI 选择已取消，没有运行推理")
    return x, y, width, height


def main() -> None:
    args = build_parser().parse_args()
    roi = tuple(args.roi) if args.roi is not None else select_roi(args)
    print("ROI (x y width height):", *roi)
    print("Version5 参数：--femto-roi", *roi)


if __name__ == "__main__":
    main()
