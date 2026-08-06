"""启动双臂 Loop，创建调度线程并管理停止与资源释放。"""

from __future__ import annotations

import argparse
import shutil
import signal
import threading
from pathlib import Path

from loop_config import load_config
from schedulers import RuntimeContext, left_scheduler, right_scheduler
from tool.planner_runtime import PlannerRuntime


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PARM 双臂 Grasp-and-Place Loop 骨架")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--live", action="store_true", help="连接规划器并执行规划轨迹")
    parser.add_argument("--cycles", type=int, default=1, help="左手占位流程循环次数")
    return parser.parse_args()


def ensure_config(path: Path) -> None:
    if path.exists():
        return
    example = ROOT / "config.example.json"
    shutil.copyfile(example, path)
    print(f"[prepare] 已从样例创建 {path}")


def main() -> None:
    args = parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    ensure_config(args.config)
    config = load_config(args.config, live=args.live)
    planner = PlannerRuntime(config)
    ctx = RuntimeContext(
        config=config,
        planner=planner,
        max_cycles=args.cycles,
    )

    def request_stop(_signum: int, _frame: object) -> None:
        print("\n[system] 收到停止请求")
        ctx.system.shutdown_requested = True
        ctx.stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        planner.open()
        ctx.system.ready = True
        ctx.system.run_mode = "RUNNING"
        left_thread = threading.Thread(target=left_scheduler, args=(ctx,), name="left-scheduler")
        right_thread = threading.Thread(target=right_scheduler, args=(ctx,), name="right-scheduler")
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
    finally:
        ctx.system.run_mode = "STOPPING"
        planner.close()
    if ctx.system.last_fault:
        raise RuntimeError(ctx.system.last_fault)
    print("[system] Loop 骨架运行结束")


if __name__ == "__main__":
    main()
