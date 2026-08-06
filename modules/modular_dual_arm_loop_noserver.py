#!/usr/bin/env python3
"""Run the modular dual-arm loop with prestarted model services.

SAM2 and GraspNet must already be running and are never managed by this entry.
Camera Broker remains owned by the loop because it arbitrates physical camera
access: the loop starts/restarts it when required and releases it on exit.
Robot preparation and the business loop remain unchanged.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import modular_dual_arm_loop as managed_loop
from modules.module_0_prepare.ensure_camera_broker import (
    ensure_camera_broker,
    stop_camera_broker,
)
from modules.module_0_prepare.ensure_graspnet_service import require_graspnet_service
from modules.module_0_prepare.ensure_sam2_service import (
    require_sam2_service,
    sam2_url,
)
from tool.runtime_log import RuntimeLogCapture


_camera_broker_managed = False


def require_prestarted_sam2() -> dict:
    """Apply the managed entry's backend contract without process ownership."""
    payload = require_sam2_service()
    backend = payload.get("backend")
    if backend != "efficientvit":
        raise RuntimeError(
            f"SAM2 backend mismatch: expected efficientvit, got {backend!r}"
        )
    print(
        "[module-0] SAM2 prestarted and ready: "
        f"{sam2_url().removeprefix('unix://')}",
        flush=True,
    )
    return payload


def ensure_loop_camera_broker() -> dict:
    """Mark Broker ownership before delegating to the production ensure path."""
    global _camera_broker_managed
    _camera_broker_managed = True
    return ensure_camera_broker()


def main() -> None:
    # This flag is inherited by Module 1's Bash/Python subprocesses, preventing
    # the nested grasp flow from silently falling back to ensure/start behavior.
    os.environ["PARM_LOOP_REQUIRE_PRESTARTED_SERVICES"] = "1"
    os.environ["PARM_LOOP_MANAGE_CAMERA_BROKER"] = "1"
    os.environ["PARM_SAM2_AUTO_RECOVER"] = "0"

    # The managed loop resolves these names at runtime.  Replace only its three
    # service ownership functions; scheduling, hardware preflight, preparation,
    # recovery, and all business module bindings stay on the production path.
    managed_loop.ensure_sam2_service = require_prestarted_sam2
    managed_loop.ensure_camera_broker = ensure_loop_camera_broker
    managed_loop.ensure_graspnet_service = require_graspnet_service
    managed_loop.sam2_url = sam2_url

    print(
        "[modular-loop-noserver] SAM2/GraspNet 仅连接预启动服务；"
        "Camera Broker 由总控启动、恢复并在退出时关闭。",
        flush=True,
    )
    managed_loop.main()


if __name__ == "__main__":
    run_dir = (
        ROOT
        / "output/modular_dual_arm_loop_noserver"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    os.environ["PARM_LOOP_RUN_ID"] = run_dir.name
    os.environ["PARM_MODULE_METRICS_DIR"] = str(run_dir)
    os.environ["PARM_MODULE_STATUS_FILE"] = str(
        run_dir / "latest_module_status.json"
    )
    log_path = run_dir / "run.log"
    with RuntimeLogCapture(log_path):
        print(f"[log] 本次完整运行日志: {log_path}", flush=True)
        print(f"[log] 结构化事件: {run_dir / 'events.jsonl'}", flush=True)
        print(f"[log] 耗时与成功率汇总: {run_dir / 'summary.json'}", flush=True)
        print(f"[log] command: {' '.join(sys.argv)}", flush=True)
        try:
            try:
                main()
            except SystemExit:
                raise
            except KeyboardInterrupt:
                print("[log] 收到 Ctrl+C，运行中断。", flush=True)
                raise SystemExit(130)
            except BaseException:
                traceback.print_exc()
                raise SystemExit(1)
        finally:
            if _camera_broker_managed:
                try:
                    stop_camera_broker()
                except BaseException as exc:
                    print(
                        f"[module-0] WARNING: Camera Broker cleanup failed: {exc}",
                        flush=True,
                    )
