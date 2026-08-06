"""Auditable geometry constants and checks for workpiece 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tool.inspection_support.stl_model import (
    load_stl_triangles_m,
    triangles_in_object_frame,
)


HERE = Path(__file__).resolve().parent
MODEL_CONFIG = HERE / "workpiece_2_config.json"


def load_model_config(path: Path = MODEL_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    triangles = triangles_in_object_frame(
        load_stl_triangles_m(config["stl_path"]),
        R_object_stl=config["R_object_stl"],
        object_origin_stl_m=config["object_origin_stl_m"],
    )
    vertices = triangles.reshape(-1, 3)
    actual_min = vertices.min(axis=0)
    actual_max = vertices.max(axis=0)
    expected_min = np.asarray(config["bounds_object_m"]["min"], dtype=float)
    expected_max = np.asarray(config["bounds_object_m"]["max"], dtype=float)
    if not np.allclose(actual_min, expected_min, atol=1e-7):
        raise RuntimeError(f"工件2模型最小边界变化: {actual_min.tolist()}")
    if not np.allclose(actual_max, expected_max, atol=1e-7):
        raise RuntimeError(f"工件2模型最大边界变化: {actual_max.tolist()}")
    return config


def tail_grasp_point_object_m() -> list[float]:
    config = load_model_config()
    return [float(value) for value in config["right_tail_grasp_point_object_m"]]

