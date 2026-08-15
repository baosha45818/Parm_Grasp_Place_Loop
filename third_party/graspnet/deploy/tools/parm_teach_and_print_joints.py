#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np


def _candidate_sdk_entries() -> list[str]:
    home = Path.home()
    candidates: list[str] = []

    env_entry = os.environ.get("PARM_SDK_WHEEL", "").strip()
    if env_entry:
        candidates.append(env_entry)

    sdk_dir = home / "Public" / "parm_sdk_1.0.0"
    for wheel in sorted(sdk_dir.glob("parm_sdk-*.whl"), reverse=True):
        candidates.append(str(wheel))

    candidates.extend(
        [
            str(home / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages"),
            str(home / "qiuguolin" / "graspnet" / ".venv" / "lib" / "python3.10" / "site-packages"),
        ]
    )
    return candidates


def _require_parm_sdk():
    last_exc: Exception | None = None
    tried: list[str] = []
    for entry in _candidate_sdk_entries():
        if not entry:
            continue
        if not Path(entry).exists():
            continue
        tried.append(entry)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        try:
            return importlib.import_module("parm_sdk")
        except Exception as exc:
            last_exc = exc
            for name in list(sys.modules):
                if name == "parm_sdk" or name.startswith("parm_sdk."):
                    sys.modules.pop(name, None)
    raise ImportError(
        "parm_sdk is not importable. Tried:\n  - " + "\n  - ".join(tried)
    ) from last_exc


def _arm_key(arm_id: int) -> str:
    return "left" if int(arm_id) == 1 else "right"


def _require_float_array(raw: Any, *, size: int, field_name: str) -> np.ndarray:
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"{field_name} is not a numeric array: {raw!r}") from exc
    if arr.size < size:
        raise RuntimeError(f"{field_name} has {arr.size} values, expected at least {size}: {raw!r}")
    if not np.all(np.isfinite(arr[:size])):
        raise RuntimeError(f"{field_name} contains NaN or Inf: {arr[:size].tolist()}")
    return arr[:size].copy()


class KeyboardInput:
    def __init__(self, handlers: dict[str, callable], on_quit, use_global: bool = False):
        self.handlers = handlers
        self.on_quit = on_quit
        self.use_global = use_global
        self._listener = None
        self._stdin_thread = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.use_global and sys.stdin.isatty():
            self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
            self._stdin_thread.start()
            print("[keys] 终端读键模式: 焦点放在本终端窗口按键")
            return
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if hasattr(key, "char") and key.char is not None:
                        c = key.char.lower()
                        if c == "q":
                            self.on_quit()
                            return False
                        if c in self.handlers:
                            self.handlers[c]()
                except Exception as exc:
                    print(f"[keys] 键盘处理失败: {exc}")

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            print("[keys] pynput 全局抓键模式")
        except Exception as exc:
            print(f"[keys] pynput 不可用 ({exc}); 退回终端读键")
            self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
            self._stdin_thread.start()

    def _stdin_loop(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    c = sys.stdin.read(1).lower()
                    if c in ("q", "\x03"):
                        self.on_quit()
                        break
                    if c in self.handlers:
                        self.handlers[c]()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.stop()


class TeachAndPrintTool:
    def __init__(self, arm_host: str, arm_id: int, joint_unit: str):
        parm_sdk = _require_parm_sdk()
        self.arm_id = int(arm_id)
        self.arm_host = str(arm_host)
        self.joint_unit = str(joint_unit)
        self.arm = parm_sdk.Arm(host=self.arm_host)
        self.sdk_version = getattr(parm_sdk, "__version__", "unknown")
        # The SDK request socket is shared by keyboard and worker threads.
        self._rpc_lock = threading.RLock()
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

    def _call_arm(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        with self._rpc_lock:
            return getattr(self.arm, method_name)(*args, **kwargs)

    def _raw_state(self) -> dict[str, Any]:
        state = self._call_arm("get_state")
        if state is None:
            raise RuntimeError("Arm.get_state() returned None")
        return state

    def _arm_state(self) -> dict[str, Any]:
        per_arm = (self._raw_state().get("state") or {}).get(_arm_key(self.arm_id))
        if per_arm is None:
            raise RuntimeError(f"arm_id={self.arm_id} state unavailable")
        return per_arm

    def _joint_state_raw(self) -> np.ndarray:
        state = self._arm_state()
        if "position" not in state:
            raise RuntimeError(f"arm state has no position field: {state}")
        return _require_float_array(state["position"], size=7, field_name="state.position")

    def _joint_state_deg(self) -> np.ndarray:
        q = self._joint_state_raw()
        if self.joint_unit == "rad":
            return np.rad2deg(q)
        return q

    def _eef_pose(self) -> Any:
        state = self._arm_state()
        if "eef_pose" not in state:
            raise RuntimeError(f"arm state has no eef_pose field: {state}")
        raw_pose = state["eef_pose"]
        if isinstance(raw_pose, dict):
            return raw_pose
        return _require_float_array(raw_pose, size=6, field_name="state.eef_pose")

    def _is_arm_enabled(self) -> bool:
        try:
            return bool(self._arm_state().get("enabled", False))
        except Exception:
            return False

    def _ensure_enabled(self) -> None:
        if self._is_arm_enabled():
            return
        state = self._raw_state()
        system_state = str(state.get("system_state", "")).strip().lower()
        if system_state in {"estop", "error"}:
            raise RuntimeError(
                f"robot system_state={system_state!r}; release estop or press R to clear an error first"
            )
        self._call_arm("enable")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._is_arm_enabled():
                return
            time.sleep(0.05)
        raise RuntimeError("enable() returned but the arm did not report enabled within 2 seconds")

    def _start_worker(self, label: str, fn) -> None:
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                print(f"[busy] 当前已有操作在执行，忽略 {label}")
                return

            def _wrapped():
                try:
                    fn()
                except Exception as exc:
                    print(f"[error] {label}: {exc}")
                    traceback.print_exc()
                finally:
                    with self._worker_lock:
                        if threading.current_thread() is self._worker_thread:
                            self._worker_thread = None

            self._worker_thread = threading.Thread(target=_wrapped, daemon=True)
            self._worker_thread.start()

    def print_joint_state(self) -> None:
        q_raw = self._joint_state_raw()
        q_deg = self._joint_state_deg()
        pose = self._eef_pose()
        print(f"[state] q_raw({self.joint_unit}) = {np.round(q_raw, 4).tolist()}")
        print(f"[state] q_deg = {np.round(q_deg, 3).tolist()}")
        if isinstance(pose, dict):
            print(f"[state] eef_pose(raw dict) = {pose}")
        else:
            print(f"[state] eef_pose = {np.round(pose, 4).tolist()}")

    def print_joint_state_async(self) -> None:
        self._start_worker("print_joint_state", self.print_joint_state)

    def start_teach_drag(self) -> None:
        def _run():
            self._ensure_enabled()
            print("[teach] start_teach(free=False)")
            self._call_arm("start_teach", self.arm_id, free=False)
            print("[teach] 已进入拖动示教")

        self._start_worker("start_teach_drag", _run)

    def start_teach_free(self) -> None:
        def _run():
            self._ensure_enabled()
            print("[teach] start_teach(free=True)")
            self._call_arm("start_teach", self.arm_id, free=True)
            print("[teach] 已进入 free 示教")

        self._start_worker("start_teach_free", _run)

    def stop_teach(self) -> None:
        def _run():
            print("[teach] stop_teach()")
            self._call_arm("stop_teach", self.arm_id)

        self._start_worker("stop_teach", _run)

    def clear_error(self) -> None:
        self._start_worker("clear_error", lambda: self._call_arm("clear_error"))

    def cancel(self) -> None:
        print("[ctrl] cancel_task()")
        try:
            self._call_arm("cancel_task", self.arm_id)
        except Exception as exc:
            print(f"[warn] cancel_task failed: {exc}")

    def shutdown(self) -> None:
        try:
            self._call_arm("cancel_task", self.arm_id)
        except Exception:
            pass
        with self._worker_lock:
            thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        try:
            self._call_arm("close")
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teach PARM arm and print current joint angles by keyboard hotkeys"
    )
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--arm-id", type=int, choices=(1, 2), default=1, help="1=left, 2=right")
    parser.add_argument(
        "--joint-unit",
        choices=("deg", "rad"),
        default="deg",
        help="unit returned by get_state().position",
    )
    parser.add_argument("--global-keys", action="store_true", help="use pynput global key listener")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    tool = TeachAndPrintTool(
        arm_host=args.arm_host,
        arm_id=args.arm_id,
        joint_unit=args.joint_unit,
    )
    stop_flag = threading.Event()

    print("=== PARM 示教 + 关节角打印工具 ===")
    print(f"arm_host={args.arm_host} arm_id={args.arm_id} sdk_version={tool.sdk_version}")
    print("按键说明:")
    print("  [D] 进入拖动示教 start_teach(free=False)")
    print("  [F] 进入 free 示教 start_teach(free=True)")
    print("  [P] 打印当前 7 维关节角和 eef_pose")
    print("  [X] 停止示教 stop_teach()")
    print("  [C] cancel_task()")
    print("  [R] clear_error()")
    print("  [Q] 退出")

    keys = KeyboardInput(
        handlers={
            "d": tool.start_teach_drag,
            "f": tool.start_teach_free,
            "p": tool.print_joint_state_async,
            "x": tool.stop_teach,
            "c": tool.cancel,
            "r": tool.clear_error,
        },
        on_quit=lambda: (print("\n退出中..."), tool.cancel(), stop_flag.set()),
        use_global=args.global_keys,
    )
    keys.start()

    try:
        while not stop_flag.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备退出...")
        tool.cancel()
        stop_flag.set()
    finally:
        keys.stop()
        tool.shutdown()


if __name__ == "__main__":
    main()
