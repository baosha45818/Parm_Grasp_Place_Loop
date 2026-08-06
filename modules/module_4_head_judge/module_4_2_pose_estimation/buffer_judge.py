#!/usr/bin/env python3
"""Module 4-2：在已到位的头部观察姿态下估计 Buffer 工件位姿并判定是否需要摆正。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


MODULE_ROOT = Path(__file__).resolve().parent
POSE_ESTIMATOR = MODULE_ROOT / "run_pose_estimation.py"
MODULE_PYTHON = MODULE_ROOT / ".venv/bin/python"
OUTPUT_DIR = MODULE_ROOT.parents[2] / "output/4_2"
Q1_GRASP_YAW_MIN_DEG = 15.0
Q1_GRASP_YAW_MAX_DEG = 75.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--workpiece-index", type=int, default=0)
    parser.add_argument("--serial", default="261222076212")
    parser.add_argument("--roi", nargs=4, type=float, default=(0.08, 0.20, 0.98, 0.90))
    parser.add_argument("--outlier-mad-scale", type=float, default=7.0)
    return parser.parse_args()


def select_workpiece(payload: dict, index: int) -> tuple[int, float]:
    """Return the selected object's body-base quadrant and normalized yaw."""
    workpieces = payload.get("workpieces")
    if not isinstance(workpieces, list) or len(workpieces) != 1:
        count = len(workpieces) if isinstance(workpieces, list) else 0
        raise RuntimeError(f"Buffer 容量为 1，Module 4-2 必须识别到恰好一个工件，实际为 {count}")
    if index != 0:
        raise RuntimeError("Buffer 容量为 1，--workpiece-index 只能为 0")
    workpiece = workpieces[index]
    if not isinstance(workpiece, dict):
        raise RuntimeError(f"workpiece-index={index} 不是有效对象：{workpiece!r}")
    quadrant = workpiece.get("head_direction_base_quadrant")
    if quadrant not in (1, 2, 3, 4):
        raise RuntimeError(f"无效的工件象限：{quadrant!r}")
    try:
        yaw_deg = float(workpiece["head_direction_base_angle_deg"]) % 360.0
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("pose 文件缺少有效的 head_direction_base_angle_deg") from exc
    return int(quadrant), yaw_deg


def main() -> int:
    args = parse_args()
    if args.workpiece_index < 0:
        raise ValueError("--workpiece-index 必须非负")
    if not POSE_ESTIMATOR.is_file():
        raise FileNotFoundError(f"Module 4-2 位姿估计入口不存在：{POSE_ESTIMATOR}")
    if not MODULE_PYTHON.is_file():
        raise FileNotFoundError(
            "Module 4-2 的独立环境不存在，无法加载 SAM2："
            f"{MODULE_PYTHON}"
        )

    before = set(OUTPUT_DIR.glob("*_mask_pose.json"))
    command = [
        # Module 4-2 的 SAM2、torch 和 hydra 都安装在其独立 .venv 中，
        # 不能继承模块化总控的根环境。
        str(MODULE_PYTHON),
        str(POSE_ESTIMATOR),
        # 云台由 Module 8 的 head-buffer-observation 阶段管理；这里仅观测。
        "--skip-head-move",
        "--serial",
        args.serial,
        "--roi",
        *map(str, args.roi),
        "--outlier-mad-scale",
        str(args.outlier_mad_scale),
    ]
    print("[module-4-2] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=MODULE_ROOT, check=True)

    created = sorted(set(OUTPUT_DIR.glob("*_mask_pose.json")) - before)
    if len(created) != 1:
        raise RuntimeError(f"Module 4-2 应只生成一个新的 pose JSON，实际为：{created}")
    pose_file = created[0]
    payload = json.loads(pose_file.read_text(encoding="utf-8"))
    if payload.get("coordinate_frame") != "body_base":
        raise RuntimeError(
            "Module 4-2 位姿必须在 body_base 下，实际为："
            f"{payload.get('coordinate_frame')!r}"
        )
    quadrant, yaw_deg = select_workpiece(payload, args.workpiece_index)
    needs_align = not (
        quadrant == 1
        and Q1_GRASP_YAW_MIN_DEG <= yaw_deg <= Q1_GRASP_YAW_MAX_DEG
    )
    decision = {
        "pose_file": str(pose_file),
        "workpiece_index": args.workpiece_index,
        "quadrant": quadrant,
        "yaw_deg": yaw_deg,
        "needs_align": needs_align,
        "q1_grasp_yaw_range_deg": [Q1_GRASP_YAW_MIN_DEG, Q1_GRASP_YAW_MAX_DEG],
    }
    args.decision_file.parent.mkdir(parents=True, exist_ok=True)
    args.decision_file.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[module-4-2] "
        f"pose={pose_file}, Q{quadrant}, yaw={yaw_deg:.2f}deg, "
        f"needs_align={needs_align}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
