#!/usr/bin/env python3

"""独立运行 Module 4，输出桌面工件 FRONT、BACK 或 OTHER。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from module import inspect_tabletop_orientation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="读取已有 RGB-D；省略时仅重新采集，不控制云台",
    )
    args = parser.parse_args()

    result = inspect_tabletop_orientation(
        args.capture_dir,
    )
    print(
        json.dumps(
            {
                "orientation": result.orientation.value,
                "capture_dir": str(result.capture_dir),
                "report": str(result.report_path),
                "objects": [
                    {
                        "object_key": item.object_key,
                        "orientation": item.orientation.value,
                        "tilt_recovery_candidate": (
                            item.tilt_recovery_candidate
                        ),
                    }
                    for item in result.objects
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()