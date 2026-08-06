#!/usr/bin/env python3
"""Independent PC2 RGB-D-band SAM2 replay.

This intentionally does not import the existing inspection/replay prompt or
URDF logic.  Its only candidate rule is the user-facing RGB-D rule in this
file: 0.10--0.25 m, max(R,G,B) in [40, 140], and no blue gripper pixels.
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
from pathlib import Path
import socket
from urllib.parse import urlparse

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-min-m", type=float, default=0.100)
    parser.add_argument("--depth-max-m", type=float, default=0.250)
    parser.add_argument("--rgb-min", type=int, default=40)
    parser.add_argument("--rgb-max", type=int, default=140)
    parser.add_argument("--min-prompt-component-area", type=int, default=500)
    parser.add_argument("--sam2-url", default="unix:///tmp/sam2-service.sock")
    parser.add_argument("--candidate-only", action="store_true")
    return parser.parse_args()


def blue_gripper_mask(rgb_bgr: np.ndarray) -> np.ndarray:
    """Identify visibly blue gripper pixels without relying on any old mask."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # OpenCV hue 95--135 is cyan/blue; saturation avoids deleting grey metal.
    return (hue >= 95) & (hue <= 135) & (saturation >= 50) & (value >= 35)


def component_prompt(candidate: np.ndarray, minimum_area: int) -> tuple[np.ndarray, list[list[int]]]:
    """Keep visibly substantial candidate components and put one point inside each."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected_labels = (np.flatnonzero(areas >= minimum_area) + 1).tolist()
    if not selected_labels and areas.size:
        selected_labels = [1 + int(np.argmax(areas))]
    prompt_mask = np.isin(labels, selected_labels)
    positives: list[list[int]] = []
    for label in selected_labels:
        component = (labels == label).astype(np.uint8)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        positives.append([int(x), int(y)])
    return prompt_mask, positives


def blue_negative_points(blue: np.ndarray, limit: int = 4) -> list[list[int]]:
    """Use centres of substantial blue gripper patches as negative SAM2 cues."""
    count, labels, stats, centres = cv2.connectedComponentsWithStats(blue.astype(np.uint8), 8)
    candidates = sorted(
        range(1, count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True
    )
    points: list[list[int]] = []
    for label in candidates:
        if int(stats[label, cv2.CC_STAT_AREA]) < 50:
            continue
        x, y = centres[label]
        points.append([int(round(x)), int(round(y))])
        if len(points) == limit:
            break
    return points


def sam2_segment(image_bgr: np.ndarray, box: list[int], positives: list[list[int]], negatives: list[list[int]], endpoint_value: str) -> np.ndarray:
    endpoint = urlparse(endpoint_value)
    if endpoint.scheme != "unix":
        raise ValueError("This standalone replay accepts only a local unix:// SAM2 endpoint")
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        image_bgr=image_bgr,
        points=np.asarray(positives + negatives, dtype=np.int32),
        labels=np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32),
        box=np.asarray(box, dtype=np.int32),
    )

    class UnixConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(endpoint.path)

    connection = UnixConnection("sam2-service", timeout=90.0)
    try:
        connection.request(
            "POST", "/v1/prompt-segment-sam2", body=payload.getvalue(),
            headers={"Content-Type": "application/x-sam2-prompt-npz"},
        )
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM2 HTTP {response.status}: {body.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return (archive["mask"] > 0).astype(np.uint8) * 255


def overlay_candidate(rgb: np.ndarray, candidate: np.ndarray, blue: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    image = rgb.copy()
    valid = candidate & np.isfinite(depth_m) & (depth_m > 0)
    if valid.any():
        low, high = np.percentile(depth_m[valid], (2, 98))
        high = max(float(high), float(low) + 1e-4)
        encoded = np.zeros(depth_m.shape, dtype=np.uint8)
        encoded[valid] = np.clip((depth_m[valid] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        colour = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
        image[valid] = (0.35 * image[valid] + 0.65 * colour[valid]).astype(np.uint8)
        label = f"candidate depth {low:.3f}-{high:.3f} m (Turbo)"
    else:
        label = "candidate depth: empty"
    # Blue pixels are explicitly removed; mark their outlines so the removal
    # remains auditable without making them look like retained candidate data.
    contours, _ = cv2.findContours(blue.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
    cv2.putText(image, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "cyan outline: deleted blue gripper pixels", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 0), 1, cv2.LINE_AA)
    return image


def overlay_sam2(rgb: np.ndarray, mask: np.ndarray, box: list[int], positives: list[list[int]], negatives: list[list[int]]) -> np.ndarray:
    image = rgb.copy()
    selected = mask > 0
    image[selected] = (0.35 * image[selected] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    x0, y0, x1, y1 = box
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
    for x, y in positives:
        cv2.circle(image, (x, y), 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 10, (0, 0, 0), 2, cv2.LINE_AA)
    for x, y in negatives:
        cv2.circle(image, (x, y), 8, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 10, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "mask=red ROI=yellow positive=green negative=blue", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "mask=red ROI=yellow positive=green negative=blue", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def process(capture: Path, output: Path, args: argparse.Namespace) -> dict:
    rgb = cv2.imread(str(capture / "rgb_000.png"), cv2.IMREAD_COLOR)
    raw_depth = np.load(capture / "depth_aligned_000.npy", allow_pickle=False)
    if rgb is None or raw_depth.shape != rgb.shape[:2]:
        raise RuntimeError("missing or inconsistent RGB-D capture")
    depth_m = raw_depth.astype(np.float32) * 0.001
    brightness = np.max(rgb, axis=2)
    blue = blue_gripper_mask(rgb)
    candidate = (
        (depth_m >= args.depth_min_m) & (depth_m <= args.depth_max_m)
        & (brightness >= args.rgb_min) & (brightness <= args.rgb_max) & ~blue
    )
    prompt_mask, positives = component_prompt(candidate, args.min_prompt_component_area)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "rgb.png"), rgb)
    np.save(output / "depth.npy", raw_depth)
    cv2.imwrite(str(output / "depth_candidate_mask.png"), candidate.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "blue_gripper_removed_mask.png"), blue.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "depth_candidate_rgb_overlay.png"), overlay_candidate(rgb, candidate, blue, depth_m))
    report = {
        "capture": capture.name,
        "policy": "independent_depth_100_250_rgb40_140_blue_removed",
        "candidate_pixels": int(candidate.sum()),
        "blue_removed_pixels": int(blue.sum()),
        "prompt_component_pixels": int(prompt_mask.sum()),
        "positive_points": positives,
        "negative_points": blue_negative_points(blue),
    }
    if args.candidate_only:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    if not positives:
        report["status"] = "no_prompt_component"
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    ys, xs = np.where(prompt_mask)
    margin = 40
    box = [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin), min(rgb.shape[1] - 1, int(xs.max()) + margin), min(rgb.shape[0] - 1, int(ys.max()) + margin)]
    negatives = report["negative_points"]
    mask = sam2_segment(rgb, box, positives, negatives, args.sam2_url)
    cv2.imwrite(str(output / "sam2_mask.png"), mask)
    cv2.imwrite(str(output / "rgb_sam2_prompt_overlay.png"), overlay_sam2(rgb, mask, box, positives, negatives))
    report.update({"status": "ok", "roi_box_xyxy": box, "sam2_mask_pixels": int((mask > 0).sum())})
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if not (0 < args.depth_min_m < args.depth_max_m and 0 <= args.rgb_min <= args.rgb_max <= 255):
        raise ValueError("invalid depth or RGB range")
    captures = sorted(path for path in args.input_dir.glob("rgbd_*") if path.is_dir())
    if not captures:
        raise FileNotFoundError("no rgbd_* captures found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for capture in captures:
        report = process(capture, args.output_dir / capture.name, args)
        summaries.append(report)
        print(json.dumps(report, ensure_ascii=False))
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
