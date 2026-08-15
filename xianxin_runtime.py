"""祥鑫现场的统一运行端点与 SDK 配置。

机器人控制器承载所有硬件 driver；本机只运行规划器、视觉和 Loop 业务进程。
所有入口都允许用已有环境变量或命令行参数覆盖这些默认值。
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARM_HOST = "192.168.3.11"
GRIPPER_HOST = "192.168.3.11"
HEAD_HOST = "192.168.3.22"
FORCE_SENSOR_HOST = "192.168.3.22"
PLANNER_HOST = "127.0.0.1"
PLANNER_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"
HEAD_D435_SERIAL = "261722072445"
LEFT_D405_SERIAL = "260322270600"
SDK_WHEEL = (
    ROOT.parent
    / "260812"
    / "祥鑫现场交付_260812"
    / "parm_sdk-1.0.0+20260811.182400-py3-none-any.whl"
)


def configure_environment() -> None:
    """Set only missing values so explicit operator overrides still win."""
    defaults = {
        "PARM_ARM_HOST": ARM_HOST,
        "PARM_GRIPPER_HOST": GRIPPER_HOST,
        "PARM_HEAD_HOST": HEAD_HOST,
        "PARM_FORCE_SENSOR_HOST": FORCE_SENSOR_HOST,
        "PARM_PLANNER_HOST": PLANNER_HOST,
        "PARM_PLANNER_EXECUTE_URL": PLANNER_EXECUTE_URL,
        "PARM_HEAD_D435_SERIAL": HEAD_D435_SERIAL,
        "PARM_D405_SERIAL": LEFT_D405_SERIAL,
    }
    if SDK_WHEEL.is_file():
        defaults["PARM_SDK_WHEEL"] = str(SDK_WHEEL)
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
