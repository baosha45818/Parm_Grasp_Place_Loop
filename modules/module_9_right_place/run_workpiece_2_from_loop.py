#!/usr/bin/env python3
"""总控专用：按工件2 JSON 配置执行右臂孔位放置。

旧调度器会统一附带 ``--arm-host`` 和
``--reset-right-task-before-servo``。工件2的配置启动器不需要这些参数；本文件
只兼容接收它们，并严格转发为经过审核的工件2配置命令。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CONFIG_RUNNER = HERE / "run_place_hole_a_wiggle_config.py"
WORKPIECE_2_CONFIG = HERE / "place_hole_a_wiggle_workpiece_2.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--arm-host", help="旧调度器兼容参数；工件2主机地址由JSON配置固定")
    parser.add_argument(
        "--reset-right-task-before-servo",
        action="store_true",
        help="旧调度器兼容参数；工件2是否重置任务由JSON配置决定",
    )
    args = parser.parse_args()
    if os.environ.get("PARM_LOOP_WORKPIECE") != "2":
        raise RuntimeError(
            "run_workpiece_2_from_loop.py 只能由 --workpiece 2 的总控调用"
        )
    if not args.live or not args.confirm_live:
        raise RuntimeError("总控真实放置必须同时传入 --live --confirm-live")
    for path, label in ((CONFIG_RUNNER, "配置启动器"), (WORKPIECE_2_CONFIG, "工件2放置配置")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    command = [
        sys.executable,
        str(CONFIG_RUNNER),
        "--config", str(WORKPIECE_2_CONFIG),
        "--live",
        "--confirm-live",
    ]
    print("[module-9/workpiece2] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=HERE, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
