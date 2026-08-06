#!/usr/bin/env python3
"""Fresh head RGB-D -> SAM2 mask -> reverse ICP -> grasp-point prediction.

Only the calibrated head is moved.  This entry never opens or commands either arm
or gripper.  All capture copies, masks, poses, visualizations and logs are written
under ``reverse_grasp``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
# HERE is .../Parm_Grasp_Place_Loop/modules/module_4_head_judge/
# module_4_1_reverse_grasp_prediction/workpiece_2_pipeline; parent 3 is the
# Parm_Grasp_Place_Loop project root.
ROOT = HERE.parents[3]
VISION = HERE / "vision_support"
BODY_HEAD = HERE / "config/body_head_calibration.json"


def inside_here(path: Path) -> Path:
    path = path.expanduser().resolve()
    try:
        path.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须位于 {HERE}: {path}") from exc
    return path


def move_head(head_host: str) -> dict:
    from parm_sdk import Head
    expected = json.loads(BODY_HEAD.read_text(encoding="utf-8"))["head_pose_deg"]
    pitch, yaw = float(expected["pitch"]), float(expected["yaw"])
    with Head(host=head_host) as head:
        head.move_to(pitch=pitch, yaw=yaw)
        deadline = time.monotonic() + 20.
        while time.monotonic() < deadline:
            state = head.get_state() or {}
            try:
                actual_pitch = -float(state["pitch"]["position"])
                actual_yaw = float(state["yaw"]["position"])
            except (KeyError, TypeError, ValueError):
                time.sleep(.05)
                continue
            if abs(actual_pitch - pitch) <= .5 and abs(actual_yaw - yaw) <= .5:
                time.sleep(1.)
                return {"expected_pitch_yaw_deg": [pitch, yaw], "actual_pitch_yaw_deg": [actual_pitch, actual_yaw]}
            time.sleep(.05)
    raise RuntimeError(f"云台未到标定观察位 pitch={pitch}, yaw={yaw}")


def fuse_static_capture(raw: Path, capture_dir: Path, frame_count: int) -> dict:
    """Fuse consecutive registered depth images at one fixed head pose.

    The RealSense capture helper already median-fuses several sensor frames for
    every saved image.  Here we take several such independent RGB-D snapshots
    and median-fuse them again.  This removes frame-to-frame holes/outliers
    before SAM, point-cloud generation and ICP all consume the same fused depth.
    """
    depth_paths = [raw / f"depth_aligned_{i:03d}.png" for i in range(frame_count)]
    rgb_paths = [raw / f"rgb_{i:03d}.png" for i in range(frame_count)]
    if not all(p.is_file() for p in depth_paths + rgb_paths):
        raise RuntimeError("连续 RGB-D 采集不完整，无法进行多帧融合")
    depths = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in depth_paths]
    if any(depth is None for depth in depths) or len({depth.shape for depth in depths}) != 1:
        raise RuntimeError("连续深度帧无法读取或尺寸不一致")
    stack = np.stack(depths).astype(np.float32)
    stack[stack <= 0] = np.nan
    # Keep invalid pixels invalid; suppress the expected all-invalid-pixel warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused = np.nanmedian(stack, axis=0)
    fused = np.where(np.isfinite(fused), np.rint(fused), 0).astype(depths[0].dtype)
    mid = frame_count // 2
    rgb = cv2.imread(str(rgb_paths[mid]), cv2.IMREAD_COLOR)
    if rgb is None or not cv2.imwrite(str(capture_dir / "rgb_000.png"), rgb):
        raise RuntimeError("无法写入融合采集的 RGB 图像")
    if not cv2.imwrite(str(capture_dir / "depth_aligned_000.png"), fused):
        raise RuntimeError("无法写入融合深度图")
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
        shutil.copy2(rgb_path, capture_dir / f"rgb_snapshot_{i:03d}.png")
        shutil.copy2(depth_path, capture_dir / f"depth_snapshot_{i:03d}.png")
    valid_per_frame = [int(np.count_nonzero(depth)) for depth in depths]
    return {
        "method": "median_of_consecutive_registered_depth_snapshots",
        "snapshot_count": frame_count,
        "rgb_snapshot_used": mid,
        "valid_depth_pixels_per_snapshot": valid_per_frame,
        "valid_depth_pixels_fused": int(np.count_nonzero(fused)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-host", default="127.0.0.1")
    parser.add_argument("--serial", default="261222076212")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--capture-snapshots", type=int, default=5,
                        help="固定观察位连续采集并叠加的 RGB-D 快照数（默认 5）")
    parser.add_argument("--depth-fuse-frames", type=int, default=9,
                        help="每个快照内部的 RealSense 对齐深度中值帧数（默认 9）")
    args = parser.parse_args()
    if args.capture_snapshots < 1 or args.depth_fuse_frames < 1:
        parser.error("--capture-snapshots 与 --depth-fuse-frames 必须至少为 1")
    run = inside_here(args.run_dir) if args.run_dir else HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if run.exists():
        raise FileExistsError(f"运行目录已存在: {run}")
    run.mkdir(parents=True)
    event = {"event": "fresh_head_sam2_full6d_reverse_stl_icp", "started_at": time.time(), "status": "started"}
    try:
        event["head"] = move_head(args.head_host)
        sys.path.insert(0, str(VISION))
        from capture_head_rgbd import capture
        raw = capture(SimpleNamespace(serial=args.serial, output_dir=HERE / "captures", frames=args.capture_snapshots,
                                      width=1280, height=720, fps=30, warmup_frames=30,
                                      timeout_ms=5000, depth_fuse_frames=args.depth_fuse_frames,
                                      preview_min_m=.20, preview_max_m=2.50))
        capture_dir = run / "capture"
        capture_dir.mkdir()
        for name in ("metadata.json",):
            source = raw / name
            if source.is_file():
                shutil.copy2(source, capture_dir / name)
        fusion = fuse_static_capture(raw, capture_dir, args.capture_snapshots)
        metadata_path = capture_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["multiframe_pointcloud_fusion"] = fusion
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        required = [capture_dir / n for n in ("metadata.json", "rgb_000.png", "depth_aligned_000.png")]
        if not all(path.is_file() for path in required):
            raise RuntimeError("新 RGB-D 采集不完整")
        mask_dir = run / "01_sam2_mask"
        icp_dir = run / "02_icp"
        grasp_dir = run / "03_grasp"
        subprocess.run([str(Path(sys.executable)), str(VISION / "sam2_full6d_masks.py"),
                        "--capture-dir", str(capture_dir), "--out", str(mask_dir)],
                       cwd=VISION, check=True)
        subprocess.run([str(Path(sys.executable)), str(HERE / "estimate_fixed_reverse_pose.py"),
                        "--capture-dir", str(capture_dir), "--masks-dir", str(mask_dir),
                        "--output-dir", str(icp_dir)], check=True)
        subprocess.run([str(Path(sys.executable)), str(HERE / "predict_convex_tab_grasp.py"),
                        "--pose", str(icp_dir / "fixed_reverse_pose.json"),
                        "--capture-dir", str(capture_dir), "--masks-dir", str(mask_dir),
                        "--output-dir", str(grasp_dir)], check=True)
        event.update({"status": "complete", "raw_capture_dir": str(raw), "capture_dir": str(capture_dir),
                      "multiframe_pointcloud_fusion": fusion,
                      "sam2_masks": str(mask_dir / "sam2_masks_overlay.png"),
                      "sam2_positive_prompts": str(mask_dir / "sam2_positive_prompts.png"),
                      "fixed_pose": str(icp_dir / "fixed_reverse_pose.json"),
                      "axes_overlay": str(icp_dir / "fixed_reverse_axes_overlay.png"),
                      "grasp_prediction": str(grasp_dir / "convex_tab_grasp_prediction.json"),
                      "grasp_overlay": str(grasp_dir / "convex_tab_grasp_overlay.png"),
                      "finished_at": time.time()})
    except Exception as exc:
        event.update({"status": "failed", "error": str(exc), "finished_at": time.time()})
        raise
    finally:
        (run / "fresh_visual_pipeline.json").write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FRESH_VISUAL_PIPELINE={run / 'fresh_visual_pipeline.json'}")
    print(f"FRESH_FIXED_POSE={event['fixed_pose']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
