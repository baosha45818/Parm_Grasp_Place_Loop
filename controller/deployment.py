"""由 --mode 统一解析现场网络地址，并导出给所有后续模块子进程。"""
from __future__ import annotations

import argparse
import os


_PROFILES = {
    "xiangxin": {
        "planner_host": "127.0.0.1",
        "arm_host": "192.168.3.11",
        "gripper_host": "192.168.3.11",
        "force_host": "192.168.3.22",
        "head_host": "192.168.3.22",
        # 左腕 D405：Planner-direct 标定链路。
        "d405_serial": "260322270600",
    },
    "gzhu": {
        "planner_host": "127.0.0.1",
        "arm_host": "192.168.88.1",
        "gripper_host": "127.0.0.1",
        "force_host": "127.0.0.1",
        # gzhu 的头部云台服务与左/右夹爪服务均在本机。
        "head_host": "127.0.0.1",
        # 左腕 D405：SDK + bridge 标定链路。
        "d405_serial": "260322270242",
    },
}


def apply(args: argparse.Namespace) -> None:
    """显式参数优先；否则严格采用所选 mode，禁止继承另一现场的旧环境。"""
    profile = _PROFILES[args.mode]
    for field, env_name in (
        ("planner_host", "PARM_PLANNER_HOST"),
        ("arm_host", "PARM_ARM_HOST"),
        ("gripper_host", "PARM_GRIPPER_HOST"),
        ("force_host", "PARM_FORCE_SENSOR_HOST"),
        ("head_host", "PARM_HEAD_HOST"),
        ("d405_serial", "PARM_D405_SERIAL"),
    ):
        value = getattr(args, field)
        if value is None:
            value = profile[field]
            setattr(args, field, value)
        os.environ[env_name] = str(value)
    os.environ["PARM_DEPLOYMENT_MODE"] = args.mode
