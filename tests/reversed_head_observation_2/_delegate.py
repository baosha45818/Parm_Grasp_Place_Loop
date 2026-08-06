"""Small helpers for invoking the field-tested workpiece-1 motion modules."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PYTHON = ROOT / ".venv/bin/python"


def delegate(module: str, defaults: Sequence[str] = ()) -> None:
    """Replace this process with a shared motion module and adapted defaults.

    User arguments are appended after the defaults so an experienced operator
    can explicitly override a default through argparse's normal last-value
    behaviour.
    """
    if not RUNTIME_PYTHON.is_file():
        raise RuntimeError(f"运行环境不存在: {RUNTIME_PYTHON}")
    os.execv(
        str(RUNTIME_PYTHON),
        [str(RUNTIME_PYTHON), "-m", module, *map(str, defaults), *sys.argv[1:]],
    )

