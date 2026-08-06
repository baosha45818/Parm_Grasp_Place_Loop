#!/usr/bin/env python3
"""Experimental Version7 flow using the left-gripper URDF nominal TCP.

This entry point keeps Version7's persistent cameras, inference, filtering and
Planner execution unchanged.  Only the ``T_eef_grasp_tcp`` supplied to the
Version5 execution-plan builder is replaced with the nominal closed-jaw TCP
derived from the left AG95 URDF and its distal-phalanx mesh.

The URDF value is intentionally experimental: real execution additionally
requires ``--allow-urdf-tcp-execution`` even when ``--execute-grasp`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOOP_ROOT = HERE.parents[4]
ROOT = Path(
    os.environ.get(
        "V9_GRASPNET_SOURCE_ROOT",
        str(LOOP_ROOT / "third_party/graspnet"),
    )
).resolve()
DEPLOY = ROOT / "deploy"
V7_DIR = DEPLOY / "version7"
for path in (ROOT, DEPLOY, V7_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deploy.version7 import run_persistent_graspnet as v7  # noqa: E402


LEFT_URDF = (
    LOOP_ROOT / "third_party/anchor_urdf/anchor_config/description/urdf/"
    "hand_tool/left/left_hand.urdf.xacro"
)
# At q=0.639806 rad the two distal-phalanx mesh inner surfaces just meet.  The
# resulting midpoint, transformed through joint_transfer_end, is below.  It
# is recomputed offline from the URDF's joint origins/mimic multipliers and
# distal_phalanx_Link.STL; retain the source values in every execution plan.
URDF_CLOSED_JOINT_RAD = 0.6398063553837112
URDF_TCP_DEFAULT_M = np.array([0.245061569, 0.0, 0.0002853], dtype=np.float64)
TCP_STATE = {"xyz_m": URDF_TCP_DEFAULT_M.copy()}


class SetTcpComponent(argparse.Action):
    """Keep the TCP override state synchronized while Version7 parses args."""

    def __call__(self, parser, namespace, value, option_string=None):
        value = float(value)
        if not np.isfinite(value):
            raise argparse.ArgumentError(self, "must be finite")
        setattr(namespace, self.dest, value)
        TCP_STATE["xyz_m"][int(self.const)] = value


def urdf_tcp_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = TCP_STATE["xyz_m"]
    return transform


def load_measured_tcp(path: Path) -> np.ndarray | None:
    """Return the certified pivot TCP only for URDF-vs-measured diagnostics."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("point_is_grasp_tcp") is not True:
            return None
        transform = np.asarray(payload.get("T_eef_grasp_tcp"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return None
        return transform
    except (OSError, ValueError, TypeError):
        return None


def tcp_diagnostics(args) -> dict:
    nominal = urdf_tcp_transform()
    measured = load_measured_tcp(Path(args.tcp_translation_calibration))
    record = {
        "source": "left_AG95_URDF_closed_jaw_mesh_midpoint",
        "urdf_path": str(LEFT_URDF),
        "closed_driver_joint_rad": URDF_CLOSED_JOINT_RAD,
        "T_eef_grasp_tcp": nominal.round(12).tolist(),
        "experimental": True,
    }
    if measured is not None:
        delta_m = nominal[:3, 3] - measured[:3, 3]
        record.update({
            "measured_tcp_path": str(Path(args.tcp_translation_calibration)),
            "T_eef_grasp_tcp_measured": measured.round(12).tolist(),
            "urdf_minus_measured_translation_mm": (delta_m * 1000.0).round(6).tolist(),
            "translation_disagreement_mm": float(np.linalg.norm(delta_m) * 1000.0),
        })
    return record


ORIGINAL_BUILD_PARSER = v7.build_parser
ORIGINAL_LOAD_TCP = v7.version5.load_tcp_translation
ORIGINAL_BUILD_PLAN = v7.version5.build_execution_plan
ORIGINAL_TCP_CERTIFIED = v7.version5.grasp_tcp_is_certified


def build_parser() -> argparse.ArgumentParser:
    parser = ORIGINAL_BUILD_PARSER()
    group = parser.add_argument_group("Version9 experimental URDF TCP")
    group.add_argument("--urdf-tcp-x-m", dest="urdf_tcp_x_m", action=SetTcpComponent, const=0,
                       default=float(URDF_TCP_DEFAULT_M[0]), help="nominal URDF closed-jaw TCP X in EEF (m)")
    group.add_argument("--urdf-tcp-y-m", dest="urdf_tcp_y_m", action=SetTcpComponent, const=1,
                       default=float(URDF_TCP_DEFAULT_M[1]), help="nominal URDF closed-jaw TCP Y in EEF (m)")
    group.add_argument("--urdf-tcp-z-m", dest="urdf_tcp_z_m", action=SetTcpComponent, const=2,
                       default=float(URDF_TCP_DEFAULT_M[2]), help="nominal URDF closed-jaw TCP Z in EEF (m)")
    group.add_argument("--urdf-tcp-max-disagreement-mm", type=float, default=30.0,
                       help="hard-stop real execution when URDF/measured TCP disagreement exceeds this; <=0 disables")
    group.add_argument("--allow-urdf-tcp-execution", action="store_true",
                       help="explicitly authorize real motion with the experimental nominal URDF TCP")
    return parser


def patched_load_tcp_translation(_path: Path) -> np.ndarray:
    """Supply Version5's existing planner code with the URDF nominal TCP."""
    return urdf_tcp_transform()


def patched_grasp_tcp_is_certified(_path: Path) -> bool:
    # Certification is handled by Version9's explicit opt-in and discrepancy
    # gate; do not let Version7 silently fall back to its pivot-only check.
    return True


def patched_build_execution_plan(args, *plan_args, **plan_kwargs):
    plan = ORIGINAL_BUILD_PLAN(args, *plan_args, **plan_kwargs)
    diagnostic = tcp_diagnostics(args)
    plan["tcp_calibration"] = diagnostic
    return plan


def validate_urdf_mode(args) -> None:
    if not LEFT_URDF.is_file():
        raise RuntimeError(f"left-gripper URDF is missing: {LEFT_URDF}")
    if not np.all(np.isfinite(TCP_STATE["xyz_m"])):
        raise RuntimeError("URDF TCP must contain finite coordinates")
    diagnostic = tcp_diagnostics(args)
    difference = diagnostic.get("translation_disagreement_mm")
    limit = float(args.urdf_tcp_max_disagreement_mm)
    if limit < 0 or not np.isfinite(limit):
        raise RuntimeError("--urdf-tcp-max-disagreement-mm must be finite and >= 0")
    if args.execute_grasp:
        if not args.allow_urdf_tcp_execution:
            raise RuntimeError(
                "Version9 uses an experimental URDF TCP; add --allow-urdf-tcp-execution "
                "only after reviewing the plan-only run"
            )
        if difference is None:
            raise RuntimeError("real Version9 execution requires a readable certified measured TCP for comparison")
        if limit > 0 and difference > limit:
            raise RuntimeError(
                f"URDF/measured TCP disagreement is {difference:.2f} mm, above the {limit:.2f} mm limit"
            )
    print("-> Version9 URDF TCP:", np.round(TCP_STATE["xyz_m"] * 1000.0, 3).tolist(), "mm")
    if difference is not None:
        print(f"-> URDF vs measured TCP disagreement: {difference:.3f} mm")


def main() -> int:
    # Version7 owns parsing internally.  Its parser is patched to add Version9
    # options; wrap parse_args once to validate immediately after parsing,
    # before Version7 starts any service or robot operation.
    v7.build_parser = build_parser
    original_parse = argparse.ArgumentParser.parse_args

    def validating_parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        validate_urdf_mode(namespace)
        return namespace

    argparse.ArgumentParser.parse_args = validating_parse
    v7.version5.load_tcp_translation = patched_load_tcp_translation
    v7.version5.grasp_tcp_is_certified = patched_grasp_tcp_is_certified
    v7.version5.build_execution_plan = patched_build_execution_plan
    try:
        return int(v7.main() or 0)
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        v7.version5.load_tcp_translation = ORIGINAL_LOAD_TCP
        v7.version5.grasp_tcp_is_certified = ORIGINAL_TCP_CERTIFIED
        v7.version5.build_execution_plan = ORIGINAL_BUILD_PLAN


if __name__ == "__main__":
    raise SystemExit(main())
