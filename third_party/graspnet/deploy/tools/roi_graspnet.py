#!/usr/bin/env python3
"""Select and print an aligned D405 ROI for a later version3 GraspNet run."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DEFAULT_D405_SERIAL = "260322270242"


def build_parser() -> argparse.ArgumentParser:
    default_display = os.environ.get("DISPLAY")
    if not default_display and Path("/tmp/.X11-unix/X1").exists():
        default_display = ":1"
    parser = argparse.ArgumentParser(
        description="Request one D405 frame from the resident Camera Broker, then drag a GraspNet ROI."
    )
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--display", default=default_display)
    parser.add_argument("--d405-serial", default=DEFAULT_D405_SERIAL,
                        help="deprecated compatibility option; Camera Broker owns D405 selection")
    parser.add_argument("--camera-socket", default=os.environ.get(
        "CAMERA_BROKER_SOCKET", os.environ.get("GRASPNET_V7_CAMERA_SOCKET", "/tmp/graspnet-v7-camera.sock")
    ))
    parser.add_argument("--camera-timeout-s", type=float, default=30.0)
    parser.add_argument("--capture-dir", type=Path, default=None,
                        help="optional Broker-owned output directory; must be below its allow-root")
    return parser


def select_roi(args: argparse.Namespace) -> tuple[int, int, int, int]:
    display = args.display
    if not display:
        raise RuntimeError("ROI selection requires an X11 display; pass --display :1 or pass --roi X Y W H")
    os.environ["DISPLAY"] = display
    os.environ.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))

    import cv2
    import numpy as np

    sys.path.insert(0, str(ROOT / "deploy" / "version7"))
    from camera_client import capture, status

    broker = status(args.camera_socket, float(args.camera_timeout_s))
    if args.capture_dir is None:
        allow_root = Path(str(broker["allow_root"])).resolve()
        output_dir = allow_root / "roi_tool" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        output_dir = args.capture_dir.resolve()
    result = capture(
        args.camera_socket,
        {"camera": "d405", "output_dir": str(output_dir), "frame_count": 1},
        float(args.camera_timeout_s),
    )
    capture_dir = Path(result["capture_dir"])
    canvas = np.asarray(np.load(capture_dir / "color.npy"), dtype=np.uint8).copy()
    print(f"-> D405 frame requested through Camera Broker: {capture_dir}")
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
    roi = tuple(args.roi) if args.roi is not None else select_roi(args)
    print("ROI (x y width height):", *roi)
    print("version3 参数：--roi", *roi)


if __name__ == "__main__":
    main()
