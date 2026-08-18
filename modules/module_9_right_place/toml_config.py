"""Small TOML reader for this module's commented scalar/array-table profiles."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any


def _without_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if quote:
            if char == quote and not escaped:
                quote = None
            escaped = char == "\\" and not escaped
        elif char in ("'", '"'):
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _value(text: str, *, line_number: int) -> Any:
    normalized = re.sub(r"\btrue\b", "True", text.strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"TOML 第 {line_number} 行的值无效: {text!r}") from exc


def load(path: Path) -> dict[str, Any]:
    """Read the TOML subset used by Module 9 profile files.

    Supported syntax: comments, scalar assignments, arrays, ``[table]`` and
    ``[[array.of.tables]]``. This deliberately rejects unrecognised syntax.
    """
    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取 TOML 配置 {path}: {exc}") from exc
    for line_number, source in enumerate(lines, start=1):
        line = _without_comment(source).strip()
        if not line:
            continue
        array_table = line.startswith("[[") and line.endswith("]]" )
        table = line.startswith("[") and line.endswith("]") and not array_table
        if array_table or table:
            names = line[2:-2] if array_table else line[1:-1]
            parts = names.split(".")
            if not all(part and part.replace("_", "").isalnum() for part in parts):
                raise ValueError(f"TOML 第 {line_number} 行的表名无效: {names!r}")
            parent: dict[str, Any] = root
            for part in parts[:-1]:
                child = parent.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(f"TOML 第 {line_number} 行的表路径冲突")
                parent = child
            if array_table:
                rows = parent.setdefault(parts[-1], [])
                if not isinstance(rows, list):
                    raise ValueError(f"TOML 第 {line_number} 行的数组表路径冲突")
                current = {}
                rows.append(current)
            else:
                child = parent.setdefault(parts[-1], {})
                if not isinstance(child, dict):
                    raise ValueError(f"TOML 第 {line_number} 行的表路径冲突")
                current = child
            continue
        if "=" not in line:
            raise ValueError(f"TOML 第 {line_number} 行必须是键值或表定义")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"TOML 第 {line_number} 行的键无效: {key!r}")
        current[key] = _value(value, line_number=line_number)
    return root
