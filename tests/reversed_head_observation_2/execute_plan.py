#!/usr/bin/env python3
"""Reuse the verified left-arm move to the fixed head observation pose."""

from ._delegate import delegate


if __name__ == "__main__":
    delegate("tests.reversed_head_observation.execute_plan")

