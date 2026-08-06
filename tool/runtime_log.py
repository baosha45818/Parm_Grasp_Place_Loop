"""将当前进程及其子进程的终端输出同时显示并持久化到日志文件。"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading


class RuntimeLogCapture:
    """通过文件描述符转发，确保 subprocess 和 spawn 子进程也写入同一日志。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._original_stdout: int | None = None
        self._original_stderr: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "RuntimeLogCapture":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._original_stdout = os.dup(sys.stdout.fileno())
        self._original_stderr = os.dup(sys.stderr.fileno())
        read_fd, write_fd = os.pipe()
        self._read_fd = read_fd

        def forward() -> None:
            with self.path.open("ab", buffering=0) as log_file:
                while True:
                    try:
                        chunk = os.read(read_fd, 8192)
                    except OSError:
                        break
                    if not chunk:
                        break
                    os.write(self._original_stdout, chunk)
                    log_file.write(chunk)

        self._thread = threading.Thread(
            target=forward, name="runtime-log-forwarder", daemon=True
        )
        self._thread.start()
        os.dup2(write_fd, sys.stdout.fileno())
        os.dup2(write_fd, sys.stderr.fileno())
        os.close(write_fd)
        return self

    def __exit__(self, *_: object) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        if self._original_stdout is not None:
            os.dup2(self._original_stdout, sys.stdout.fileno())
        if self._original_stderr is not None:
            os.dup2(self._original_stderr, sys.stderr.fileno())
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._original_stdout is not None:
            os.close(self._original_stdout)
        if self._original_stderr is not None:
            os.close(self._original_stderr)
        if self._read_fd is not None:
            try:
                os.close(self._read_fd)
            except OSError:
                pass
