"""Client for sam2-service over a local UDS or an explicitly supplied HTTP URL."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


@dataclass(frozen=True)
class SegmentationResponse:
    masks: np.ndarray
    bboxes: np.ndarray
    area_px: np.ndarray
    raw_area_px: np.ndarray
    predicted_iou: np.ndarray
    stability_score: np.ndarray
    source_index: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrackingResponse:
    masks: np.ndarray
    overlays: np.ndarray
    bboxes: np.ndarray
    frame_ids: np.ndarray
    mask_pixels: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FirstFrameResponse:
    mask: np.ndarray
    bbox: np.ndarray
    points: np.ndarray
    labels: np.ndarray
    candidate_masks: np.ndarray
    candidate_bboxes: np.ndarray
    candidate_areas: np.ndarray
    candidate_source_indices: np.ndarray
    candidate_present: np.ndarray
    metadata: dict[str, Any]


def _decode_tracking_response(payload: bytes) -> TrackingResponse:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return TrackingResponse(
            masks=archive["masks"],
            overlays=archive["overlays"],
            bboxes=archive["bboxes"],
            frame_ids=archive["frame_ids"],
            mask_pixels=archive["mask_pixels"],
            metadata=json.loads(str(archive["metadata"])),
        )


def _decode_first_frame_response(payload: bytes) -> FirstFrameResponse:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return FirstFrameResponse(
            mask=archive["mask"],
            bbox=archive["bbox"],
            points=archive["points"],
            labels=archive["labels"],
            candidate_masks=archive["candidate_masks"],
            candidate_bboxes=archive["candidate_bboxes"],
            candidate_areas=archive["candidate_areas"],
            candidate_source_indices=archive["candidate_source_indices"],
            candidate_present=archive["candidate_present"],
            metadata=json.loads(str(archive["metadata"])),
        )


def _encode_rgbd_request(
    image_bgr: np.ndarray, depth_image: np.ndarray, intrinsics: np.ndarray, depth_scale: float,
) -> bytes:
    image = np.asarray(image_bgr)
    depth = np.asarray(depth_image)
    camera = np.asarray(intrinsics)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if depth.shape[:2] != image.shape[:2] or camera.shape != (3, 3) or depth_scale <= 0:
        raise ValueError("depth_image, intrinsics, or depth_scale is invalid")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer, image_bgr=image, depth_image=depth, intrinsics=camera, depth_scale=float(depth_scale)
    )
    return buffer.getvalue()


def _connection(endpoint: str | Path) -> tuple[str, aiohttp.BaseConnector | None]:
    """Return an HTTP base URL and connector for a UDS path or HTTP endpoint.

    A pathname preserves the original local UDS transport.  An ``http://`` or
    ``https://`` value enables the optional TCP listener for LAN clients.
    """
    value = str(endpoint)
    if value.startswith(("http://", "https://")):
        return value.rstrip("/"), None
    return "http://sam2-service", aiohttp.UnixConnector(path=value)


async def _post(
    socket_path: str | Path, path: str, *, params: dict[str, str] | None = None,
    data: bytes = b"", content_type: str | None = None, timeout_s: float = 65.0,
) -> tuple[int, bytes]:
    base_url, connector = _connection(socket_path)
    headers = {} if content_type is None else {"Content-Type": content_type}
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=timeout_s)
    ) as session:
        async with session.post(
            f"{base_url}{path}", params=params, data=data, headers=headers
        ) as response:
            return response.status, await response.read()


async def segment_full_image_top_n(
    image_bgr: np.ndarray,
    *,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    num_masks: int = 1,
    roi: tuple[int, int, int, int] | None = None,
    min_mask_area_px: int = 1,
    timeout_s: float = 65.0,
) -> SegmentationResponse:
    """Submit raw BGR bytes to the local UDS service and decode its NPZ reply."""
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if num_masks < 1 or min_mask_area_px < 1:
        raise ValueError("num_masks and min_mask_area_px must be at least 1")
    height, width = image.shape[:2]
    params = {
        "height": str(height),
        "width": str(width),
        "num_masks": str(num_masks),
        "min_mask_area_px": str(min_mask_area_px),
    }
    if roi is not None:
        params["roi"] = ",".join(str(int(value)) for value in roi)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    base_url, connector = _connection(socket_path)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.post(
            f"{base_url}/v1/segment",
            params=params,
            data=np.ascontiguousarray(image).tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        ) as response:
            payload = await response.read()
            if response.status != 200:
                raise RuntimeError(f"sam2-service returned HTTP {response.status}: {payload.decode(errors='replace')}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return SegmentationResponse(
            masks=archive["masks"],
            bboxes=archive["bboxes"],
            area_px=archive["area_px"],
            raw_area_px=archive["raw_area_px"],
            predicted_iou=archive["predicted_iou"],
            stability_score=archive["stability_score"],
            source_index=archive["source_index"],
            metadata=json.loads(str(archive["metadata"])),
        )


async def select_first_frame_mask(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    timeout_s: float = 65.0,
    **options: Any,
) -> FirstFrameResponse:
    """Run the service's RGB-D first-frame selector with JSON-serializable options."""
    status, body = await _post(
        socket_path, "/v1/first-frame/mask",
        params={"options": json.dumps(options, separators=(",", ":"))},
        data=_encode_rgbd_request(image_bgr, depth_image, intrinsics, depth_scale),
        content_type="application/x-sam2-rgbd-npz", timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return _decode_first_frame_response(body)


async def select_first_frame_candidates(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    num_candidates: int,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    timeout_s: float = 65.0,
    **options: Any,
) -> FirstFrameResponse:
    """Return fixed RGB-D first-frame candidate slots through the UDS service."""
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1")
    status, body = await _post(
        socket_path, "/v1/first-frame/candidates",
        params={
            "num_candidates": str(num_candidates),
            "options": json.dumps(options, separators=(",", ":")),
        },
        data=_encode_rgbd_request(image_bgr, depth_image, intrinsics, depth_scale),
        content_type="application/x-sam2-rgbd-npz", timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return _decode_first_frame_response(body)


async def select_first_frame_candicates_max(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    num_candidates: int = 10,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    timeout_s: float = 90.0,
    **options: Any,
) -> FirstFrameResponse:
    """Call the dense-prompt, 2-mm RGB-D candidate endpoint."""
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1")
    status, body = await _post(
        socket_path, "/v1/first-frame/candicates-max",
        params={
            "num_candidates": str(num_candidates),
            "options": json.dumps(options, separators=(",", ":")),
        },
        data=_encode_rgbd_request(image_bgr, depth_image, intrinsics, depth_scale),
        content_type="application/x-sam2-rgbd-npz", timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return _decode_first_frame_response(body)


async def select_first_frame_candidate_components(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    num_candidates: int = 10,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    timeout_s: float = 90.0,
    **options: Any,
) -> FirstFrameResponse:
    """Call max-style RGB-D candidates with lower size bounds disabled."""
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1")
    status, body = await _post(
        socket_path, "/v1/first-frame/candidate-components",
        params={
            "num_candidates": str(num_candidates),
            "options": json.dumps(options, separators=(",", ":")),
        },
        data=_encode_rgbd_request(image_bgr, depth_image, intrinsics, depth_scale),
        content_type="application/x-sam2-rgbd-npz", timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return _decode_first_frame_response(body)


async def initialize_tracking_from_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    frame_id: int | None = None,
    timeout_s: float = 65.0,
) -> TrackingResponse:
    """Create the service's one active tracking stream from a first-frame mask."""
    image = np.asarray(image_bgr)
    binary_mask = np.asarray(mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if binary_mask.shape != image.shape[:2]:
        raise ValueError("mask must have the same HxW shape as image_bgr")
    payload = io.BytesIO()
    np.savez_compressed(payload, image_bgr=image, mask=binary_mask.astype(np.uint8))
    params = {} if frame_id is None else {"frame_id": str(frame_id)}
    status, body = await _post(
        socket_path, "/v1/tracking/initialize", params=params, data=payload.getvalue(),
        content_type="application/x-sam2-tracking-init-npz", timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return _decode_tracking_response(body)


async def segment_from_points(
    image_bgr: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    *,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    timeout_s: float = 65.0,
) -> np.ndarray:
    """Segment one first-frame object from positive/negative SAM2 point prompts."""
    image = np.asarray(image_bgr)
    prompt_points = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    prompt_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    if len(prompt_points) != len(prompt_labels) or 1 not in prompt_labels:
        raise ValueError("points/labels must match and include at least one positive point")
    height, width = image.shape[:2]
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        image_bgr=image,
        points=prompt_points,
        labels=prompt_labels,
        box=np.asarray([0, 0, width - 1, height - 1], dtype=np.int32),
    )
    status, body = await _post(
        socket_path,
        "/v1/prompt-segment-sam2",
        data=payload.getvalue(),
        content_type="application/x-sam2-prompt-npz",
        timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(
            f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}"
        )
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        mask = np.asarray(archive["mask"], dtype=np.uint8)
    if mask.shape != image.shape[:2] or not np.any(mask):
        raise RuntimeError("SAM2 point prompts produced an empty or invalid mask")
    return (mask > 0).astype(np.uint8)


async def append_tracking_frame(
    image_bgr: np.ndarray,
    *,
    stream_id: str,
    socket_path: str | Path = "/tmp/sam2-service.sock",
    frame_id: int | None = None,
    timeout_s: float = 65.0,
) -> int:
    """Queue one BGR frame for the active tracking stream without inference."""
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    height, width = image.shape[:2]
    params = {"stream_id": stream_id, "height": str(height), "width": str(width)}
    if frame_id is not None:
        params["frame_id"] = str(frame_id)
    status, body = await _post(
        socket_path, "/v1/tracking/append", params=params,
        data=np.ascontiguousarray(image).tobytes(), content_type="application/octet-stream",
        timeout_s=timeout_s,
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return int(json.loads(body)["frame_id"])


async def propagate_tracking(
    *, stream_id: str, socket_path: str | Path = "/tmp/sam2-service.sock", timeout_s: float = 65.0,
) -> TrackingResponse:
    """Propagate all queued frames for one tracking stream."""
    status, body = await _post(
        socket_path, "/v1/tracking/propagate", params={"stream_id": stream_id}, timeout_s=timeout_s
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
    return _decode_tracking_response(body)


async def close_tracking_stream(
    *, stream_id: str, socket_path: str | Path = "/tmp/sam2-service.sock", timeout_s: float = 65.0,
) -> None:
    """Release the active tracking runtime and invalidate its stream id."""
    status, body = await _post(
        socket_path, "/v1/tracking/close", params={"stream_id": stream_id}, timeout_s=timeout_s
    )
    if status != 200:
        raise RuntimeError(f"sam2-service returned HTTP {status}: {body.decode(errors='replace')}")
