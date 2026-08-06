#!/usr/bin/env python3
"""Reuse the verified left-arm reversed-observation joint-path check."""

from ._delegate import delegate


if __name__ == "__main__":
    delegate("tests.reversed_head_observation.check_plan")

