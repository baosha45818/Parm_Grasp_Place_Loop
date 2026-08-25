"""由 --mode 统一解析现场网络地址，并导出给所有后续模块子进程。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


_PROFILES = {
    "xiangxin": {
        "planner_host": "127.0.0.1",
        "arm_host": "192.168.3.11",
        "gripper_host": "192.168.3.11",
        "force_host": "192.168.3.22",
        "head_host": "192.168.3.22",
    },
    "gzhu": {
        "planner_host": "127.0.0.1",
        "arm_host": "192.168.4.11",
        "gripper_host": "127.0.0.1",
        "force_host": "127.0.0.1",
        # gzhu 的头部云台服务与左/右夹爪服务均在本机。
        "head_host": "127.0.0.1",
    },
}


# 总控启动时一次性绑定现场标定。反抓交接模块会继承该环境变量；若未设置，
# 它为了兼容独立旧脚本会回退到 xiangxin 标定，容易在 gzhu 现场错误连接
# 261722072445。这里必须与 --mode 原子保持一致。
_HEAD_CALIBRATION_BY_MODE = {
    "gzhu": "gzhu_reverse_observation.json",
    "xiangxin": "xiangxin_reverse_observation.json",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HEAD_CALIBRATION_DIR = _PROJECT_ROOT / "tools/calibration/head_d435"


def apply(args: argparse.Namespace) -> None:
    """显式参数优先；否则严格采用所选 mode，禁止继承另一现场的旧环境。"""
    profile = _PROFILES[args.mode]
    for field, env_name in (
        ("planner_host", "PARM_PLANNER_HOST"),
        ("arm_host", "PARM_ARM_HOST"),
        ("gripper_host", "PARM_GRIPPER_HOST"),
        ("force_host", "PARM_FORCE_SENSOR_HOST"),
        ("head_host", "PARM_HEAD_HOST"),
    ):
        value = getattr(args, field)
        if value is None:
            value = profile[field]
            setattr(args, field, value)
        os.environ[env_name] = str(value)
    os.environ["PARM_DEPLOYMENT_MODE"] = args.mode
    calibration = _HEAD_CALIBRATION_DIR / _HEAD_CALIBRATION_BY_MODE[args.mode]
    if not calibration.is_file():
        raise FileNotFoundError(f"模式 {args.mode} 的头部相机标定不存在：{calibration}")
    os.environ["PARM_BODY_HEAD_CALIBRATION"] = str(calibration)
