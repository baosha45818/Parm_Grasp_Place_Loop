#!/usr/bin/env python3
"""Move arm1 to the fixed post-grasp upward joint pose after confirmation."""

from __future__ import annotations

from go_graspnet import build_parser, move_arm


GRASPUP_JOINTS_DEG = [
    -4.081,
    -52.722,
    3.637,
    100.478,
    -10.051,
    53.969,
    8.843,
]


def main() -> None:
    parser = build_parser()
    parser.description = "Move arm1 to the fixed grasp-up joint pose."
    args = parser.parse_args()
    move_arm(args, GRASPUP_JOINTS_DEG)


if __name__ == "__main__":
    main()
