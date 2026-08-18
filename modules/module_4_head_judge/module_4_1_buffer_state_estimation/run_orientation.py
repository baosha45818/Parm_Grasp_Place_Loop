#!/usr/bin/env python3
"""Module 4.1 唯一入口：从一次 RGB-D 同时发布工件正反和 ICP 位姿。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[2]
CONFIG_ROOT = MODULE_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from modules.module_4_head_judge.module_4_1_buffer_state_estimation.deployment_modes import (  # noqa: E402
    apply_mode_environment,
    load_mode,
)


def _run_module(module: str, *args: object) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *(str(arg) for arg in args)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_orientation(
    capture_dir: Path,
    *,
    workpiece: str,
    mode: str,
    sam2_url: str,
    config_path: Path | None = None,
) -> Path:
    capture_dir = capture_dir.expanduser().resolve()
    if not (capture_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"不是有效 RGB-D 采集目录: {capture_dir}")
    deployment = load_mode(mode)
    apply_mode_environment(deployment, workpiece)
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    if str(metadata.get("serial", "")) != deployment.camera_serial:
        raise RuntimeError(
            f"采集相机与 {mode} 标定不匹配: "
            f"capture={metadata.get('serial')!r}, calibration={deployment.camera_serial!r}"
        )
    config_path = (
        config_path or CONFIG_ROOT / f"workpiece_{workpiece}.json"
    ).expanduser().resolve()
    masks_dir = capture_dir / "sam2_masks"
    _run_module(
        "modules.module_4_head_judge.module_4_1_buffer_state_estimation.vision.mask",
        "--workpiece", workpiece,
        "--capture-dir", capture_dir,
        "--out", masks_dir,
        "--config", config_path,
        "--workspace", deployment.workspace_path(workpiece),
        "--socket", sam2_url,
    )
    if workpiece == "1":
        _run_module(
            "modules.module_4_head_judge.module_4_1_buffer_state_estimation.vision.workpiece_1_icp",
            "--capture-dir", capture_dir,
            "--orientation-config", config_path,
        )
    elif workpiece == "2":
        _run_module(
            "modules.module_4_head_judge.module_4_1_buffer_state_estimation.vision.workpiece_2_icp",
            "--capture-dir", capture_dir,
            "--masks-dir", masks_dir,
            "--config", config_path,
        )
    else:
        raise ValueError(f"不支持的工件编号: {workpiece}")
    diagnostics = masks_dir / "orientation_diagnostics.json"
    if not diagnostics.is_file():
        raise RuntimeError(f"视觉流程未生成正反/位姿结果: {diagnostics}")
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpiece", choices=("1", "2"), required=True)
    parser.add_argument("--mode", choices=("xiangxin", "gzhu"), required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--socket",
        default=os.environ.get(
            "PARM_SAM2_URL",
            f"unix://{PROJECT_ROOT / 'tool/sam2-service/runtime/sam2-service.sock'}",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diagnostics = run_orientation(
        args.capture_dir,
        workpiece=args.workpiece,
        mode=args.mode,
        sam2_url=args.socket,
        config_path=args.config,
    )
    print(f"ORIENTATION_DIAGNOSTICS={diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
