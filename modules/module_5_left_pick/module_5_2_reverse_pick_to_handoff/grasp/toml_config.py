"""Minimal TOML reader for the checked-in grasp configuration files.

Python 3.10 does not provide ``tomllib`` and this deployment deliberately has
no third-party TOML dependency.  The project configs use only scalar values,
numeric arrays, and one-level tables, which are parsed here explicitly.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def load_toml_mapping(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    section = payload
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if not name or "." in name:
                raise ValueError(f"{path}:{number}: unsupported TOML table")
            section = payload.setdefault(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"{path}:{number}: TOML table conflicts with a value")
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: invalid TOML assignment")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"{path}:{number}: unsupported TOML key")
        try:
            section[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: unsupported TOML value") from exc
    return payload


def load_config_mapping(path: Path) -> dict[str, Any]:
    """Load TOML configs, while retaining xiangxin's existing JSON profile."""
    if path.suffix.lower() == ".toml":
        return load_toml_mapping(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported grasp configuration format: {path}")
