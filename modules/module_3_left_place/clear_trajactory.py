#!/usr/bin/env python3
"""清空 Module 3 的全部固定工位轨迹缓存。

默认只展示待删除文件；必须显式传入 ``--yes`` 才执行删除。缓存目录会保留，
下一次 Module 3 真实成功执行四条主路径时会自动重新生成对应轨迹。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TRAJECTORY_ROOT = Path(__file__).resolve().parent / "trajactory"


def parse_args() -> argparse.Namespace:
    """解析清理确认参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认删除 trajactory 内的全部缓存文件和子目录",
    )
    return parser.parse_args()


def cache_entries(root: Path) -> list[Path]:
    """列出缓存根目录的直接内容；不跟随目录外的符号链接。"""
    if not root.exists():
        return []
    if not root.is_dir():
        raise RuntimeError(f"轨迹缓存根路径不是目录: {root}")
    return sorted(root.iterdir(), key=lambda path: path.name)


def remove_entry(path: Path) -> None:
    """删除一个已确认位于缓存根目录下的文件、链接或子目录。"""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise RuntimeError(f"无法识别的缓存条目: {path}")


def main() -> int:
    """列出或清空所有 Module 3 轨迹缓存。"""
    args = parse_args()
    entries = cache_entries(TRAJECTORY_ROOT)
    if not entries:
        print(f"[trajactory] 没有可清理的轨迹缓存：{TRAJECTORY_ROOT}")
        return 0

    print(f"[trajactory] 缓存根目录：{TRAJECTORY_ROOT}")
    for entry in entries:
        print(f"  - {entry.name}")
    if not args.yes:
        print("[trajactory] 未执行删除；确认清空请重新运行并加入 --yes")
        return 0

    for entry in entries:
        remove_entry(entry)
    TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
    print("[trajactory] 已清空全部 Module 3 轨迹缓存；目录已保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
