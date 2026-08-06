#!/usr/bin/env python3
"""Record direct-handoff transforms without commanding either robot arm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(__file__).resolve().parents[2] / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable) != PYTHON:
    os.execv(
        str(PYTHON),
        [
            str(PYTHON),
            "-m",
            "tests.reversed_head_observation.configure_handoff",
            *sys.argv[1:],
        ],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tests.reversed_head_observation.handoff_geometry import (
    planner_tf_target_source,
    transform_debug,
    validate_transform,
)
from tests.reversed_head_observation.preflight_handoff import (
    load_icp_object_transform,
    validate_icp_for_motion,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "handoff_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读记录胸前交接目标和右TCP固定抓持关系；不发送运动命令"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser(
        "set-target", help="用ICP姿态和指定body_base中心生成固定工件交接目标"
    )
    target.add_argument("--icp-result", type=Path, required=True)
    target.add_argument("--center-m", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    target.add_argument("--input-config", type=Path, default=DEFAULT_CONFIG)
    target.add_argument("--output-config", type=Path, required=True)
    target.add_argument("--overwrite", action="store_true")

    right = subparsers.add_parser(
        "record-right", help="在右TCP人工对准交接工件后记录固定抓持关系"
    )
    right.add_argument("--input-config", type=Path, required=True)
    right.add_argument("--output-config", type=Path, required=True)
    right.add_argument("--planner-host", default="127.0.0.1")
    right.add_argument(
        "--pregrasp-offset-tcp-m",
        type=float,
        nargs=3,
        required=True,
        metavar=("DX", "DY", "DZ"),
    )
    right.add_argument("--confirm-object-at-handoff", action="store_true")
    right.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("transform_convention") != "p_A = T_A_B @ p_B":
        raise RuntimeError("handoff config transform convention is missing or invalid")
    return config


def write_new_config(path: Path, config: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"output config already exists: {path}; add --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def config_with_handoff_target(
    config: dict[str, Any],
    T_base_object_current: np.ndarray,
    T_base_left_tcp_captured: np.ndarray,
    center_m: list[float],
    icp_result_path: Path,
) -> dict[str, Any]:
    center = np.asarray(center_m, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("--center-m must contain three finite values")
    target = validate_transform(T_base_object_current, "T_base_object_current").copy()
    T_base_left_tcp_captured = validate_transform(
        T_base_left_tcp_captured, "T_base_left_tcp_captured"
    )
    target[:3, 3] = center
    T_left_tcp_object = np.linalg.inv(T_base_left_tcp_captured) @ T_base_object_current
    box_dimensions = np.asarray(config["collision_box_dimensions_m"], dtype=float)
    box_center_object = np.asarray(
        config["collision_box_center_object_m"], dtype=float
    )
    if (
        box_dimensions.shape != (3,)
        or not np.all(np.isfinite(box_dimensions))
        or np.any(box_dimensions <= 0.0)
        or box_center_object.shape != (3,)
        or not np.all(np.isfinite(box_center_object))
    ):
        raise ValueError("invalid workpiece collision-box configuration")
    T_object_box = np.eye(4)
    T_object_box[:3, 3] = box_center_object
    T_left_tcp_box = T_left_tcp_object @ T_object_box
    box_debug = transform_debug(T_left_tcp_box)
    updated = json.loads(json.dumps(config))
    updated["T_base_object_handoff"] = target.tolist()
    updated["handoff_center_body_m"] = center.tolist()
    updated["T_left_tcp_object"] = T_left_tcp_object.tolist()
    updated["attached_collision"] = {
        "id": "held_reverse_workpiece",
        "attached_link": str(config["left_tool_frame"]),
        "box_dimensions_m": box_dimensions.tolist(),
        "box_pose_in_attached_link": {
            "position_m": box_debug["position_m"],
            "orientation_xyzw": box_debug["orientation_xyzw"],
        },
        "touch_links": list(config["collision_touch_links"]),
        "geometry": "conservative_box_around_stl_with_clearance",
    }
    updated.setdefault("calibration_metadata", {})["handoff_target"] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "position_source": "explicit_body_base_center_m",
        "orientation_source": "body_base_icp_result",
        "icp_result": str(icp_result_path),
        "target": transform_debug(target),
    }
    return updated


def config_with_right_relation(
    config: dict[str, Any],
    T_base_right_tcp: np.ndarray,
    pregrasp_offset_tcp_m: list[float],
) -> dict[str, Any]:
    T_base_object_handoff = validate_transform(
        config.get("T_base_object_handoff"), "T_base_object_handoff"
    )
    T_base_right_tcp = validate_transform(T_base_right_tcp, "T_base_right_tcp")
    offset = np.asarray(pregrasp_offset_tcp_m, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("pregrasp offset must contain three finite values")
    if float(np.linalg.norm(offset)) < 0.01:
        raise ValueError("pregrasp offset must be at least 10 mm")
    T_right_tcp_object = np.linalg.inv(T_base_right_tcp) @ T_base_object_handoff
    T_right_tcp_object = validate_transform(
        T_right_tcp_object, "T_right_tcp_object"
    )
    box_center_object = np.asarray(
        config["collision_box_center_object_m"], dtype=float
    )
    if box_center_object.shape != (3,) or not np.all(np.isfinite(box_center_object)):
        raise ValueError("invalid collision_box_center_object_m")
    T_object_box = np.eye(4)
    T_object_box[:3, 3] = box_center_object
    T_right_tcp_box = T_right_tcp_object @ T_object_box
    right_box_debug = transform_debug(T_right_tcp_box)
    updated = json.loads(json.dumps(config))
    updated["T_right_tcp_object"] = T_right_tcp_object.tolist()
    updated["right_pregrasp_offset_tcp_m"] = offset.tolist()
    attached = updated.get("attached_collision")
    if not isinstance(attached, dict):
        raise RuntimeError("set-target must create attached_collision before record-right")
    attached["right_attached_link"] = str(config["right_tool_frame"])
    attached["right_box_pose_in_attached_link"] = {
        "position_m": right_box_debug["position_m"],
        "orientation_xyzw": right_box_debug["orientation_xyzw"],
    }
    updated.setdefault("calibration_metadata", {})["right_grasp_relation"] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumption": "object was physically at T_base_object_handoff",
        "right_tcp_frame": str(config["right_tool_frame"]),
        "T_base_right_tcp_at_recording": T_base_right_tcp.tolist(),
        "T_right_tcp_object": T_right_tcp_object.tolist(),
        "right_pregrasp_offset_tcp_m": offset.tolist(),
    }
    return updated


def read_current_right_tcp(config: dict[str, Any], planner_host: str) -> np.ndarray:
    from parm_sdk import Planner

    right_frame = str(config["right_tool_frame"])
    with Planner(host=planner_host) as planner:
        frames = planner.list_frame_ids()
        if right_frame not in frames or "body_base" not in frames:
            raise RuntimeError(f"Planner TF missing body_base or {right_frame!r}")
        response = planner.lookup_transform(right_frame, "body_base")
    return planner_tf_target_source(response, right_frame, "body_base")


def main() -> int:
    args = parse_args()
    if args.command == "set-target":
        config = load_config(args.input_config)
        T_base_object_current, result = load_icp_object_transform(args.icp_result)
        T_base_left_tcp_captured = validate_icp_for_motion(result, config)
        updated = config_with_handoff_target(
            config,
            T_base_object_current,
            T_base_left_tcp_captured,
            args.center_m,
            args.icp_result,
        )
        write_new_config(args.output_config, updated, args.overwrite)
        print(json.dumps(updated["calibration_metadata"]["handoff_target"], indent=2, ensure_ascii=False))
        print(f"[完成] 已记录胸前工件目标；未运动机器人: {args.output_config}")
        return 0

    if not args.confirm_object_at_handoff:
        raise RuntimeError(
            "record-right requires --confirm-object-at-handoff after visually "
            "confirming the object is physically at the configured handoff pose"
        )
    config = load_config(args.input_config)
    T_base_right_tcp = read_current_right_tcp(config, args.planner_host)
    updated = config_with_right_relation(
        config, T_base_right_tcp, args.pregrasp_offset_tcp_m
    )
    write_new_config(args.output_config, updated, args.overwrite)
    print(json.dumps(updated["calibration_metadata"]["right_grasp_relation"], indent=2, ensure_ascii=False))
    print(f"[完成] 已记录右TCP固定抓持关系；未运动机器人: {args.output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
