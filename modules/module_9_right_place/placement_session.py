"""Shared mutable state for one Hole-A placement attempt.

The placement flow currently owns this state as local variables.  Keeping it
in one explicit object lets motion, sensing, insertion-detection, and
post-insertion components be moved out without hidden closure dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


EventLogger = Callable[..., None]


@dataclass
class PlacementSession:
    """Runtime state shared by the components of one placement attempt."""

    args: Any
    arm: object | None = None
    sensor: object | None = None
    log_file: object | None = None
    log_event: EventLogger | None = None

    servo_started: bool = False
    continuous_servo_stopped: bool = False
    search_started_at: float | None = None
    search_motion_name: str = ""

    force_base: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    torque_base: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    search_axis_index: int = 0
    search_direction_sign: float = -1.0

    release_events_seen: int = 0
    previous_continuous_wrench: tuple[list[float], list[float]] | None = None
    continuous_contact_peak: tuple[list[float], list[float]] | None = None
    continuous_release_frames: int = 0
    continuous_resultant_peak: tuple[float, float] | None = None
    continuous_resultant_release_frames: int = 0
    fx_reversal_peak: float | None = None
    fx_reversal_confirmed_frames: int = 0
    continuous_monitor_tick: int = 0
    continuous_monitor_stride: int = 1
    last_continuous_monitor_output_at: float | None = None

    descend_fx_base: float | None = None
    last_descend_output_at: float | None = None
