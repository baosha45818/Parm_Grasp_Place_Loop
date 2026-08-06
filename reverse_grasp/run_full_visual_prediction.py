#!/usr/bin/env python3
"""以 Module 4-1 的视觉链路为 workpiece_2 生成唯一反抓候选。

输入 RGB-D 采集目录只读。脚本先把所需帧复制到本目录的运行沙盒，再使用与
Module 4-1 相同的 SAM2 分割、受桌面约束的双面 CAD/ICP、唯一候选判定与 3D
可视化；所有新文件均位于 ``reverse_grasp/``。本程序不控制任何硬件。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VISION_ROOT = ROOT / "modules/module_5_left_pick/module_5_1_reverse_grasp/implementation"
STL = ROOT / "tool/workpiece_2/workpiece_2.stl"
ORIENTATION = HERE / "workpiece_2_orientation_config.json"
LOG = HERE / "reverse_grasp_prediction.log.jsonl"


def inside_here(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(f"输出必须位于 {HERE}: {resolved}") from exc
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True,
                        help="Module 4-1 格式的 RGB-D 采集目录；只读")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="本目录内的运行目录；默认按时间创建")
    parser.add_argument("--prediction-file", type=Path, default=None,
                        help="本目录内的预测 JSON；默认 <run-dir>/prediction.json")
    parser.add_argument("--log", type=Path, default=LOG)
    return parser.parse_args()


def require_dependencies() -> None:
    required = [
        STL, ORIENTATION,
        VISION_ROOT / "sam2_tabletop_masks.py",
        VISION_ROOT / "sam2_icp_grasp.py",
        VISION_ROOT / "visualize_grasp_3d.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("完整视觉链路依赖缺失: " + ", ".join(missing))


def validate_capture(capture_dir: Path) -> tuple[Path, Path, Path]:
    metadata = capture_dir / "metadata.json"
    rgb = capture_dir / "rgb_000.png"
    depth = capture_dir / "depth_aligned_000.png"
    missing = [str(path) for path in (metadata, rgb, depth) if not path.is_file()]
    if missing:
        raise FileNotFoundError("采集目录必须包含 metadata.json、rgb_000.png、depth_aligned_000.png: " + ", ".join(missing))
    json.loads(metadata.read_text(encoding="utf-8"))
    return metadata, rgb, depth


def stage_capture(capture_dir: Path, run_dir: Path) -> Path:
    """Copy only the three files consumed by the 4-1 vision scripts."""
    staged = run_dir / "capture"
    staged.mkdir(parents=True, exist_ok=False)
    for source in validate_capture(capture_dir):
        shutil.copy2(source, staged / source.name)
    return staged


def run(script: str, *args: object) -> None:
    subprocess.run(
        [str(Path(sys.executable)), str(VISION_ROOT / script), *(str(value) for value in args)],
        cwd=VISION_ROOT,
        check=True,
    )


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    require_dependencies()
    capture_dir = args.capture_dir.expanduser().resolve()
    validate_capture(capture_dir)
    run_dir = inside_here(args.run_dir) if args.run_dir else HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = inside_here(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction_file = inside_here(args.prediction_file) if args.prediction_file else run_dir / "prediction.json"
    log = inside_here(args.log)

    try:
        staged = stage_capture(capture_dir, run_dir)
        masks_dir = staged / "sam2_masks"
        run("sam2_tabletop_masks.py", "--capture-dir", staged, "--out", masks_dir)
        run("sam2_icp_grasp.py", "--capture-dir", staged, "--stl", STL, "--orientation-config", ORIENTATION)
        run("visualize_grasp_3d.py", "--capture-dir", staged)
        grasp_json = masks_dir / "grasp_poses.json"
        records = json.loads(grasp_json.read_text(encoding="utf-8"))
        if len(records) != 1:
            raise RuntimeError(f"反抓预测必须得到唯一候选，当前得到 {len(records)} 个；不会输出可消费预测。")
        payload = {
            "schema_version": 1,
            "event": "workpiece_2_module_4_1_visual_prediction",
            "execution": "FORBIDDEN_BY_THIS_TOOL",
            "source_capture_dir": str(capture_dir),
            "staged_capture_dir": str(staged),
            "stl": str(STL),
            "orientation_config": str(ORIENTATION),
            "candidate_count": 1,
            "grasp_json": str(grasp_json),
            "grasp_record": records[0],
            "artifacts": {
                "mask_overlay": str(masks_dir / "sam2_masks_overlay.png"),
                "orientation_overlay": str(masks_dir / "orientation_overlay.png"),
                "grasp_overlay": str(masks_dir / "grasp_overlay.png"),
                "grasp_3d": str(masks_dir / "grasp_3d.png"),
                "orientation_diagnostics": str(masks_dir / "orientation_diagnostics.json"),
            },
            "created_at": time.time(),
        }
        write_json_atomic(prediction_file, payload)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": payload["event"], "created_at": payload["created_at"],
                "source_capture_dir": payload["source_capture_dir"], "prediction": str(prediction_file),
                "candidate_count": 1, "grasp_json": str(grasp_json),
            }, ensure_ascii=False) + "\n")
    except Exception as exc:
        failure = {"event": "workpiece_2_module_4_1_visual_prediction_failed", "created_at": time.time(),
                   "source_capture_dir": str(capture_dir), "run_dir": str(run_dir), "error": str(exc)}
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise
    print(f"REVERSE_GRASP_VISUAL_PREDICTION={prediction_file}")
    print(f"REVERSE_GRASP_POSES={grasp_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
