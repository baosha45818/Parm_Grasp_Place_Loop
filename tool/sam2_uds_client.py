"""Shared lightweight UDS client facade for robot workflow modules.

Import this module from ``modules`` rather than copying the client into every
workflow.  The implementation remains owned by ``tool/sam2-service`` so the
wire protocol has one authoritative source.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_CLIENT_PATH = Path(__file__).resolve().parent / "sam2-service" / "uds_http_client.py"
_SPEC = importlib.util.spec_from_file_location("_sam2_service_uds_client", _CLIENT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load sam2-service UDS client from {_CLIENT_PATH}")
_IMPLEMENTATION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPLEMENTATION
_SPEC.loader.exec_module(_IMPLEMENTATION)

SegmentationResponse = _IMPLEMENTATION.SegmentationResponse
TrackingResponse = _IMPLEMENTATION.TrackingResponse
FirstFrameResponse = _IMPLEMENTATION.FirstFrameResponse
segment_full_image_top_n = _IMPLEMENTATION.segment_full_image_top_n
select_first_frame_mask = _IMPLEMENTATION.select_first_frame_mask
select_first_frame_candidates = _IMPLEMENTATION.select_first_frame_candidates
initialize_tracking_from_mask = _IMPLEMENTATION.initialize_tracking_from_mask
append_tracking_frame = _IMPLEMENTATION.append_tracking_frame
propagate_tracking = _IMPLEMENTATION.propagate_tracking
close_tracking_stream = _IMPLEMENTATION.close_tracking_stream

__all__ = [
    "SegmentationResponse",
    "TrackingResponse",
    "FirstFrameResponse",
    "segment_full_image_top_n",
    "select_first_frame_mask",
    "select_first_frame_candidates",
    "initialize_tracking_from_mask",
    "append_tracking_frame",
    "propagate_tracking",
    "close_tracking_stream",
]
