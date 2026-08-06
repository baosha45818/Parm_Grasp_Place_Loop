#!/usr/bin/env python3
"""旧版双臂总控入口：保留旧调度器的外部业务脚本调用链。

本文件故意不使用 ``modular_dual_arm_loop.py`` 的 modules 入口替换。
其中右臂正向分支会直接调用旧路径：
``/home/dainanjun/zhuyiming/push_T_ALL/push_to_q1_pipeline.py``。
它仅作为旧流程回归/对照入口，不修改当前模块化总控或旧调度器源码。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


LOOP_ROOT = Path(__file__).resolve().parents[1]
LEGACY_LOOP_DIR = LOOP_ROOT / "tests/loop"


def consume_flag(flag: str) -> bool:
    """Consume an old-wrapper-only flag before legacy argparse sees it."""
    if flag not in sys.argv:
        return False
    sys.argv.remove(flag)
    return True


def main() -> None:
    # Do not let a parent modular run change spawned workers back to modules.
    os.environ.pop("PARM_MODULAR_LOOP_ENTRIES", None)
    os.environ.pop("PARM_MODULAR_REVERSE_PICK", None)

    # ``--no-vis`` is one safe automation mode: no preview also means no
    # interactive y confirmation.  The normal/``--vis`` mode keeps both.
    if consume_flag("--no-vis"):
        os.environ["GRASPNET_VISUALIZATION"] = "0"
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "1"
    elif consume_flag("--vis"):
        os.environ["GRASPNET_VISUALIZATION"] = "1"
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "0"
    else:
        os.environ["GRASPNET_VISUALIZATION"] = "0"
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "1"
    if consume_flag("--auto-confirm-real-grasp"):
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "1"
    elif consume_flag("--no-auto-confirm-real-grasp"):
        os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"] = "0"

    if str(LEGACY_LOOP_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_LOOP_DIR))

    import simple_dual_arm_loop

    print(
        "[old-loop] 使用旧调度器默认业务入口；右臂正向分支将调用 "
        "/home/dainanjun/zhuyiming/push_T_ALL/push_to_q1_pipeline.py。",
        flush=True,
    )
    simple_dual_arm_loop.main()


if __name__ == "__main__":
    main()
