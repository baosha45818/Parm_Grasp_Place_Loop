#!/usr/bin/env python3
"""新电脑部署自检（只读，不移动机器人、不修改网络/相机配置）。

用途：把整个项目迁到另一台电脑后，在首次启动总控前确认环境、标定、网卡和
两台 RealSense 相机是否与现场模式匹配。相机按 *序列号* 识别，USB 插口编号
变化不需要改任何项目代码。

建议在项目根目录执行：

    ./.venv/bin/python tools/migration_preflight.py --mode gzhu --strict

全部本机服务已启动后，可再加 ``--check-services --sdk-readiness`` 做完整只读预检。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Profile:
    mode: str
    local_subnet_prefix: str
    arm_host: str
    planner_host: str
    gripper_host: str
    force_host: str
    head_host: str
    d405_serial: str
    d435_serial: str
    head_calibration: str


PROFILES = {
    "gzhu": Profile(
        mode="gzhu",
        local_subnet_prefix="192.168.88.",
        arm_host="192.168.88.1",
        planner_host="127.0.0.1",
        gripper_host="127.0.0.1",
        force_host="127.0.0.1",
        head_host="127.0.0.1",
        d405_serial="260322270242",
        d435_serial="261222076212",
        head_calibration="tools/calibration/head_d435/gzhu_reverse_observation.json",
    ),
    "xiangxin": Profile(
        mode="xiangxin",
        local_subnet_prefix="192.168.3.",
        arm_host="192.168.3.11",
        planner_host="127.0.0.1",
        gripper_host="192.168.3.11",
        force_host="192.168.3.22",
        head_host="192.168.3.22",
        d405_serial="260322270600",
        d435_serial="261722072445",
        head_calibration="tools/calibration/head_d435/xiangxin_reverse_observation.json",
    ),
}


@dataclass
class Check:
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))
        print(f"[{status:4}] {name}: {detail}", flush=True)

    @property
    def has_failure(self) -> bool:
        return any(item.status == "FAIL" for item in self.checks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(PROFILES), default="gzhu")
    parser.add_argument(
        "--project-root", type=Path, default=ROOT,
        help="项目根目录；默认由本脚本所在 tools/ 自动推导",
    )
    parser.add_argument("--skip-cameras", action="store_true", help="跳过 RealSense 序列号检查")
    parser.add_argument("--skip-network", action="store_true", help="跳过本机网卡/到机械臂路由检查")
    parser.add_argument("--check-services", action="store_true", help="检查 Planner 本机端口（服务须已启动）")
    parser.add_argument(
        "--sdk-readiness", action="store_true",
        help="调用 Module 0 的完整只读 SDK 预检；不会运动或使能机器人",
    )
    parser.add_argument("--report", type=Path, help="额外写出 JSON 报告路径")
    parser.add_argument("--strict", action="store_true", help="任一 FAIL 时以退出码 1 结束")
    return parser.parse_args()


def check_project_files(root: Path, profile: Profile, reporter: Reporter) -> None:
    required = (
        ".venv/bin/python",
        "controller/main.py",
        "modules/module_1_left_grasp/main.py",
        "modules/module_4_head_judge/module_4_1_buffer_state_estimation/check_buffer_face.py",
        "modules/module_5_left_pick/module_5_1_front_pick_to_graspuptop/main.py",
        "modules/module_5_left_pick/module_5_2_reverse_pick_to_handoff/main.py",
        "modules/module_3_left_place/standalone_left_place.py",
        "tools/calibration/left_wrist_d405/calibration_wrist1.json",
        profile.head_calibration,
        "tools/sam2-service/start_uds_server.sh",
    )
    missing = [entry for entry in required if not (root / entry).is_file()]
    if missing:
        reporter.add("项目文件", "FAIL", "缺少：" + ", ".join(missing))
    else:
        reporter.add("项目文件", "PASS", f"根目录={root}")

    missing_packages = [
        package for package in ("numpy", "cv2", "open3d", "pyrealsense2", "parm_sdk")
        if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        reporter.add("Python 依赖", "FAIL", "当前 Python 缺少：" + ", ".join(missing_packages))
    else:
        reporter.add("Python 依赖", "PASS", f"解释器={sys.executable}")
    if importlib.util.find_spec("torch") is None:
        reporter.add(
            "SAM2 PyTorch",
            "WARN",
            "当前 .venv 未安装 torch；仅在本机直接启动 SAM2 时需要，独立 sam2-service 环境可不受影响",
        )


def ipv4_addresses() -> list[str]:
    """只读取系统网卡配置；无 ip 命令时返回空列表。"""
    ip_command = shutil.which("ip")
    if not ip_command:
        return []
    try:
        output = subprocess.check_output(
            [ip_command, "-j", "-4", "address", "show"], text=True, stderr=subprocess.DEVNULL, timeout=3
        )
        payload = json.loads(output)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    values: list[str] = []
    for interface in payload:
        for address in interface.get("addr_info", []):
            if address.get("family") == "inet" and isinstance(address.get("local"), str):
                values.append(address["local"])
    return values


def check_network(profile: Profile, reporter: Reporter) -> None:
    addresses = ipv4_addresses()
    if not addresses:
        reporter.add("本机网卡", "WARN", "无法读取 IPv4 地址；请手工确认已接入机械臂网段")
    elif any(value.startswith(profile.local_subnet_prefix) for value in addresses):
        reporter.add("本机网卡", "PASS", f"IPv4={', '.join(addresses)}；已接入 {profile.local_subnet_prefix}0/24")
    else:
        reporter.add(
            "本机网卡", "FAIL",
            f"IPv4={', '.join(addresses)}；模式 {profile.mode} 需要 {profile.local_subnet_prefix}0/24",
        )

    ip_command = shutil.which("ip")
    if not ip_command:
        reporter.add("机械臂路由", "WARN", f"未找到 ip 命令，无法确认到 {profile.arm_host} 的路由")
        return
    try:
        route = subprocess.check_output(
            [ip_command, "route", "get", profile.arm_host], text=True, stderr=subprocess.STDOUT, timeout=3
        ).strip().replace("\n", " ")
    except (OSError, subprocess.SubprocessError) as exc:
        reporter.add("机械臂路由", "FAIL", f"无法路由到 {profile.arm_host}: {exc}")
    else:
        reporter.add("机械臂路由", "PASS", route)


def check_cameras(profile: Profile, reporter: Reporter) -> None:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]

        context = rs.context()
        devices = list(context.query_devices())
        detected = [
            {
                "serial": device.get_info(rs.camera_info.serial_number),
                "name": device.get_info(rs.camera_info.name),
            }
            for device in devices
        ]
    except Exception as exc:  # 库/驱动不存在都应给可操作的提示，而非 traceback。
        reporter.add("RealSense 相机", "FAIL", f"无法枚举相机：{type(exc).__name__}: {exc}")
        return

    serials = {item["serial"] for item in detected}
    expected = {profile.d405_serial: "左腕 D405", profile.d435_serial: "头部 D435"}
    missing = [f"{label}={serial}" for serial, label in expected.items() if serial not in serials]
    description = ", ".join(f"{item['name']}({item['serial']})" for item in detected) or "未发现设备"
    if missing:
        reporter.add("RealSense 相机", "FAIL", f"发现：{description}；缺少：{', '.join(missing)}")
    else:
        reporter.add("RealSense 相机", "PASS", f"发现：{description}")


def tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_services(profile: Profile, reporter: Reporter) -> None:
    ports = (5604, 5609, 5610, 5667)
    unavailable = [str(port) for port in ports if not tcp_open(profile.planner_host, port)]
    if unavailable:
        reporter.add("本机 Planner 服务", "FAIL", "未监听端口：" + ", ".join(unavailable))
    else:
        reporter.add("本机 Planner 服务", "PASS", f"{profile.planner_host}:" + ",".join(map(str, ports)))


def check_sdk_readiness(root: Path, profile: Profile, reporter: Reporter) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from modules.module_0_prepare.preflight_robot_infrastructure import preflight_robot_infrastructure

        preflight_robot_infrastructure(
            planner_host=profile.planner_host,
            arm_host=profile.arm_host,
            gripper_host=profile.gripper_host,
            force_sensor_host=profile.force_host,
        )
    except Exception as exc:
        reporter.add("Module 0 SDK 只读预检", "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        reporter.add("Module 0 SDK 只读预检", "PASS", "Planner、双臂、夹爪、力传感器均可读")


def write_report(path: Path, profile: Profile, reporter: Reporter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": asdict(profile),
        "python": sys.executable,
        "checks": [asdict(item) for item in reporter.checks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] JSON 报告：{path}", flush=True)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    profile = PROFILES[args.mode]
    reporter = Reporter()
    print(f"[INFO] 新电脑部署自检；mode={profile.mode}；只读模式，不会控制机器人", flush=True)
    check_project_files(root, profile, reporter)
    if args.skip_network:
        reporter.add("网络", "SKIP", "由 --skip-network 跳过")
    else:
        check_network(profile, reporter)
    if args.skip_cameras:
        reporter.add("RealSense 相机", "SKIP", "由 --skip-cameras 跳过")
    else:
        check_cameras(profile, reporter)
    if args.check_services:
        check_services(profile, reporter)
    else:
        reporter.add("本机 Planner 服务", "SKIP", "未指定 --check-services")
    if args.sdk_readiness:
        check_sdk_readiness(root, profile, reporter)
    else:
        reporter.add("Module 0 SDK 只读预检", "SKIP", "未指定 --sdk-readiness")
    if args.report:
        write_report(args.report.expanduser(), profile, reporter)

    result = "FAIL" if reporter.has_failure else "PASS"
    print(f"[SUMMARY] {result}；{len(reporter.checks)} 项检查", flush=True)
    return 1 if args.strict and reporter.has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
