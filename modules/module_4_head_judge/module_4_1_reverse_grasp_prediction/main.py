#!/usr/bin/env python3
"""模块 4-1：从已采集的头部 RGB-D 预测左臂反抓的唯一抓取点。

本模块只负责 SAM2/ICP 抓取点预测和结果交接；不控制云台、机械臂或夹爪。
调用方必须先将云台转到对应观察位并采集 RGB-D。真实反抓由 Module 5-1 执行。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
LOOP_ROOT = MODULE_DIR.parents[2]
VISION_ROOT = (
    LOOP_ROOT
    / "modules/module_5_left_pick/module_5_1_reverse_grasp/implementation"
)
PYTHON = Path(sys.executable)
BODY_HEAD = (
    LOOP_ROOT
    / "modules/module_4_head_judge/module_4_2_pose_estimation/"
    "calibration/head_charuco_13x5_calibration_head.json"
)
sys.path.insert(0, str(LOOP_ROOT))
from modules.module_5_left_pick.module_5_1_reverse_grasp.execution_timing import ExecutionTiming
DEFAULT_SAM2_URL = (
    "unix:///home/dainanjun/Public/Parm_Grasp_Place_Loop/"
    "tool/sam2-service/runtime/sam2-service.sock"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="允许采集真实头部 RGB-D")
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--body-head", type=Path, default=BODY_HEAD)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--sam2-url",
        default=os.environ.get("PARM_SAM2_URL", DEFAULT_SAM2_URL),
        help="驻留 SAM2 服务地址；总控默认继承 PARM_SAM2_URL",
    )
    return parser.parse_args()


def require_paths(body_head: Path) -> None:
    required = [
        body_head,
        VISION_ROOT / "capture_head_rgbd.py",
        VISION_ROOT / "sam2_tabletop_masks.py",
        VISION_ROOT / "sam2_icp_grasp.py",
        VISION_ROOT / "visualize_grasp_3d.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Module 4-1 依赖不存在: " + ", ".join(missing))


def run_visual(timing: ExecutionTiming, stage: str, script: str, *args: object) -> None:
    timing.start(stage)
    try:
        subprocess.run(
            [str(PYTHON), str(VISION_ROOT / script), *(str(arg) for arg in args)],
            cwd=VISION_ROOT,
            check=True,
        )
    except BaseException:
        timing.finish(stage, status="FAILURE")
        raise
    else:
        timing.finish(stage)


def predict(capture_dir: Path, sam2_url: str, timing: ExecutionTiming) -> Path:
    run_visual(
        timing,
        "sam2_mask_generation",
        "sam2_tabletop_masks.py",
        "--capture-dir",
        capture_dir,
        "--socket",
        sam2_url,
    )
    run_visual(timing, "icp_grasp_pose", "sam2_icp_grasp.py", "--capture-dir", capture_dir)
    run_visual(timing, "grasp_visualization", "visualize_grasp_3d.py", "--capture-dir", capture_dir)
    grasp_json = capture_dir / "sam2_masks/grasp_poses.json"
    records = json.loads(grasp_json.read_text(encoding="utf-8"))
    if len(records) != 1:
        raise RuntimeError(
            "Module 4-1 反抓预测必须得到唯一候选，"
            f"当前得到 {len(records)} 个；不会进入 Module 5-1。"
        )
    return grasp_json


def write_prediction(
    prediction_file: Path,
    *,
    capture_dir: Path,
    grasp_json: Path,
    body_head: Path,
) -> None:
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "event": "module_4_1_reverse_grasp_predicted",
        "capture_dir": str(capture_dir),
        "grasp_json": str(grasp_json),
        "candidate_count": 1,
        "body_head_calibration": str(body_head.resolve()),
        "created_at": time.time(),
    }
    temporary = prediction_file.with_suffix(prediction_file.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(prediction_file)


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("Module 4-1 采集真实数据必须显式传入 --execute")
    require_paths(args.body_head)
    capture_dir = args.capture_dir.expanduser().resolve()
    if not (capture_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"不是有效头部 RGB-D 采集目录: {capture_dir}")
    timing = ExecutionTiming()
    timing_path = capture_dir / "reverse_grasp_perception_timing.json"
    outcome = "FAILURE"
    try:
        grasp_json = predict(capture_dir, args.sam2_url, timing)
        timing.start("write_prediction")
        write_prediction(
            args.prediction_file,
            capture_dir=capture_dir,
            grasp_json=grasp_json,
            body_head=args.body_head,
        )
        timing.finish("write_prediction")
        outcome = "SUCCESS"
        print(f"MODULE_4_1_PREDICTION={args.prediction_file.resolve()}")
        print(f"[module-4-1] 唯一反抓候选已预测: {grasp_json}")
        return 0
    finally:
        timing.write(timing_path, outcome)
        print(f"REVERSE_GRASP_PERCEPTION_TIMING={timing_path}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
