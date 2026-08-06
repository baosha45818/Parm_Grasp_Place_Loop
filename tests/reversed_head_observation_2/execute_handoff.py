#!/usr/bin/env python3
"""Execute the workpiece-2 taught-joint pregrasp and tail grasp."""

from pathlib import Path
import sys

from ._delegate import delegate
HERE = Path(__file__).resolve().parent
GEOMETRY_CONFIRMATION = "--confirm-workpiece-2-grasp-geometry"


if __name__ == "__main__":
    executing = "--execute" in sys.argv[1:]
    confirmed = GEOMETRY_CONFIRMATION in sys.argv[1:]
    if executing and not confirmed:
        raise SystemExit(
            "拒绝执行：请先确认右手预抓取与尾部抓取两组示教关节角及"
            "两段PTP路径安全，再增加 --confirm-workpiece-2-grasp-geometry。"
        )
    if confirmed:
        sys.argv.remove(GEOMETRY_CONFIRMATION)
    delegate(
        "tests.reversed_head_observation_2.execute_icp_tail_handoff",
        (
            "--config", str(HERE / "handoff_config.json"),
            "--attachment-state", str(HERE / "handoff_attachment_state.json"),
        ),
    )
