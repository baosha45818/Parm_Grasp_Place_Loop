"""Mirror one module process's terminal output into output/right_log."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import threading


_LOOP_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if parent.name == "Parm_Grasp_Place_Loop"),
    Path(__file__).resolve().parents[2],
)
_LOG_ROOT = _LOOP_ROOT / "output" / "right_log"


class ConsoleLog:
    """FD-level tee, so output from child commands is logged too."""

    def __init__(self, module: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = _LOG_ROOT / f"{module}_{stamp}.log"
        self._saved_fds: tuple[int, int] | None = None
        self._threads: list[threading.Thread] = []
        self._file = None
        self._write_lock = threading.Lock()

    def _copy_stream(self, read_fd: int, terminal_fd: int) -> None:
        try:
            while chunk := os.read(read_fd, 65536):
                with self._write_lock:
                    os.write(terminal_fd, chunk)
                    self._file.write(chunk)
                    self._file.flush()
        finally:
            os.close(read_fd)

    def __enter__(self) -> Path:
        sys.stdout.flush()
        sys.stderr.flush()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("ab", buffering=0)
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        self._saved_fds = (saved_stdout, saved_stderr)
        os.dup2(stdout_write, 1)
        os.dup2(stderr_write, 2)
        os.close(stdout_write)
        os.close(stderr_write)
        for read_fd, terminal_fd, name in (
            (stdout_read, saved_stdout, "stdout"),
            (stderr_read, saved_stderr, "stderr"),
        ):
            thread = threading.Thread(target=self._copy_stream, args=(read_fd, terminal_fd), name=f"console-log-{name}")
            thread.start()
            self._threads.append(thread)
        print(f"[log] 控制台输出同时保存至 {self.path}", flush=True)
        return self.path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        sys.stdout.flush()
        sys.stderr.flush()
        if self._saved_fds is not None:
            saved_stdout, saved_stderr = self._saved_fds
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
        for thread in self._threads:
            thread.join()
        if self._saved_fds is not None:
            for fd in self._saved_fds:
                os.close(fd)
        if self._file is not None:
            self._file.close()
        return False


def console_log(module: str) -> ConsoleLog:
    return ConsoleLog(module)
