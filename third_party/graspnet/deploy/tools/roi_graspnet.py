#!/usr/bin/env python3
"""Select and print an aligned D405 ROI for a later version3 GraspNet run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_D405_SERIAL = "260322270242"


def build_parser() -> argparse.ArgumentParser:
    default_display = os.environ.get("DISPLAY")
    if not default_display and Path("/tmp/.X11-unix/X1").exists():
        default_display = ":1"
    parser = argparse.ArgumentParser(
        description="Drag a D405 ROI and print coordinates for a later version3 run."
    )
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--display", default=default_display)
    parser.add_argument("--d405-serial", default=DEFAULT_D405_SERIAL)
    return parser


def select_roi(serial: str, display: str | None) -> tuple[int, int, int, int]:
    if not display:
        raise RuntimeError("ROI selection requires an X11 display; pass --display :1 or pass --roi X Y W H")
    os.environ["DISPLAY"] = display
    os.environ.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))

    import cv2

    sys.path.insert(0, str(ROOT / "utils"))
    from robot_camera import D405Camera

    with D405Camera(serial=serial) as camera:
        frame = camera.read()
    canvas = frame.color.copy()
    cv2.putText(
        canvas,
        "Drag ROI, then ENTER/SPACE to confirm; C to cancel",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    roi = cv2.selectROI("D405 ROI for GraspNet", canvas, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise RuntimeError("ROI selection cancelled; no inference was run")
    return x, y, width, height


def main() -> None:
    args = build_parser().parse_args()
    roi = tuple(args.roi) if args.roi is not None else select_roi(args.d405_serial, args.display)
    print("ROI (x y width height):", *roi)
    print("version3 参数：--roi", *roi)


if __name__ == "__main__":
    main()
