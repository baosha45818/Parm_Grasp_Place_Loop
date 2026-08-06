#!/usr/bin/env python3
"""Move arm1 to the fixed buffer joint pose after confirmation."""

from __future__ import annotations

from go_graspnet import build_parser, move_arm


BUFFER_JOINTS_DEG = [
    49.671,
    -56.037,
    -115.398,
    72.201,
    9.075,
    53.369,
    11.563,
]


def main() -> None:
    parser = build_parser()
    parser.description = "Move arm1 to the fixed buffer joint pose."
    args = parser.parse_args()
    move_arm(args, BUFFER_JOINTS_DEG)


if __name__ == "__main__":
    main()
