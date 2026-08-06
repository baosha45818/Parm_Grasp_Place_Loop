#!/usr/bin/env python3
"""供 shell 业务模块写入统一状态/耗时记录的极小命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_types import ResultStatus
from modules.module_status import report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("status", choices=tuple(item.value for item in ResultStatus))
    parser.add_argument("message")
    parser.add_argument("--reason-code", default="")
    args = parser.parse_args()
    report(
        args.module,
        ResultStatus(args.status),
        args.message,
        reason_code=args.reason_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
