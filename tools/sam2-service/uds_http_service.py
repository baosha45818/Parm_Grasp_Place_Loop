"""Local HTTP-over-UDS server for the full-image SAM2 segmentation API.

The request body is raw HxWx3 uint8 BGR bytes.  Shape and selection options
are carried in query parameters so no base64 or JSON image serialization is
needed.  Responses are compressed NumPy ``.npz`` archives.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import stat
from typing import Any
import secrets

from aiohttp import web
import numpy as np

from . import (
    TrackingBatchResult,
    append_tracking_frame,
    close,
    close_tracking,
    initialize_tracking_from_mask,
    initialize_tracking_from_points,
    propagate_tracking,
    segment_full_image_top_n,
    select_first_frame_candidate_components,
    select_first_frame_candicates_max,
    select_first_frame_candidates,
    select_first_frame_mask,
    start,
    start_tracking,
    status,
)


def _log(message: str) -> None:
    print(f"[sam2-service] {message}", flush=True)
from .src.sam2_first_frame_selector import _legacy_select_efficientvit_from_points


@dataclass(frozen=True)
class ServerConfig:
    socket_path: Path
    pid_file: Path | None
    backend: str
    queue_capacity: int
    request_timeout_s: float
    tracking_only: bool = False


@dataclass
class _SegmentJob:
    image_bgr: np.ndarray
    num_masks: int
    roi: tuple[int, int, int, int] | None
    min_mask_area_px: int
    response: asyncio.Future[dict[str, Any]]


def _parse_positive_int(value: str | None, name: str, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise web.HTTPBadRequest(text=f"missing query parameter: {name}")
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=f"{name} must be an integer") from exc
    if parsed < 1:
        raise web.HTTPBadRequest(text=f"{name} must be at least 1")
    return parsed


def _parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None or value == "":
        return None
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise web.HTTPBadRequest(text="roi must be x0,y0,x1,y1") from exc
    if len(values) != 4:
        raise web.HTTPBadRequest(text="roi must be x0,y0,x1,y1")
    return values  # Bounds are validated by segment_full_image_top_n().


async def _decode_request(request: web.Request) -> _SegmentJob:
    if request.content_type != "application/octet-stream":
        raise web.HTTPUnsupportedMediaType(text="use Content-Type: application/octet-stream")
    height = _parse_positive_int(request.query.get("height"), "height")
    width = _parse_positive_int(request.query.get("width"), "width")
    num_masks = _parse_positive_int(request.query.get("num_masks"), "num_masks", default=1)
    min_mask_area_px = _parse_positive_int(
        request.query.get("min_mask_area_px"), "min_mask_area_px", default=1
    )
    payload = await request.read()
    expected_size = height * width * 3
    if len(payload) != expected_size:
        raise web.HTTPBadRequest(
            text=f"BGR body has {len(payload)} bytes; expected {expected_size} for {height}x{width}x3"
        )
    image_bgr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    loop = asyncio.get_running_loop()
    return _SegmentJob(
        image_bgr=image_bgr,
        num_masks=num_masks,
        roi=_parse_roi(request.query.get("roi")),
        min_mask_area_px=min_mask_area_px,
        response=loop.create_future(),
    )


def _encode_response(result: dict[str, Any]) -> bytes:
    candidates = result["masks"]
    if candidates:
        masks = np.stack([item["mask"] for item in candidates], axis=0)
        bboxes = np.asarray([item["bbox"] for item in candidates], dtype=np.float32)
    else:
        masks = np.empty((0, 0, 0), dtype=np.uint8)
        bboxes = np.empty((0, 4), dtype=np.float32)
    metadata = {
        key: result[key]
        for key in (
            "requested_count", "actual_count", "available_count", "min_mask_area_px",
            "roi", "selection_rule", "backend", "resident_model", "timing_ms",
        )
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        masks=masks,
        bboxes=bboxes,
        area_px=np.asarray([item["area_px"] for item in candidates], dtype=np.int64),
        raw_area_px=np.asarray([item["raw_area_px"] for item in candidates], dtype=np.int64),
        predicted_iou=np.asarray([item["predicted_iou"] for item in candidates], dtype=np.float32),
        stability_score=np.asarray([item["stability_score"] for item in candidates], dtype=np.float32),
        source_index=np.asarray([item["source_index"] for item in candidates], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )
    return buffer.getvalue()


def _encode_tracking_response(batch, *, stream_id: str, initialized: bool = False) -> bytes:
    """Serialize tracking frames as a binary NPZ response."""
    frames = tuple(batch.frames)
    if frames:
        masks = np.stack([frame.mask for frame in frames], axis=0)
        overlays = np.stack([frame.overlay for frame in frames], axis=0)
        bboxes = np.asarray(
            [frame.bbox if frame.bbox is not None else (np.nan,) * 4 for frame in frames],
            dtype=np.float32,
        )
    else:
        masks = np.empty((0, 0, 0), dtype=np.uint8)
        overlays = np.empty((0, 0, 0, 3), dtype=np.uint8)
        bboxes = np.empty((0, 4), dtype=np.float32)
    metadata = {
        "stream_id": stream_id,
        "initialized": initialized,
        "propagated_count": int(batch.propagated_count),
        "queued_before": int(batch.queued_before),
        "queued_after": int(batch.queued_after),
        "latency_ms": float(batch.latency_ms),
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        masks=masks,
        overlays=overlays,
        bboxes=bboxes,
        frame_ids=np.asarray([frame.frame_id for frame in frames], dtype=np.int64),
        mask_pixels=np.asarray([frame.mask_pixels for frame in frames], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )
    return buffer.getvalue()


def _parse_optional_frame_id(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        frame_id = int(value)
    except ValueError as exc:
        raise web.HTTPBadRequest(text="frame_id must be an integer") from exc
    if frame_id < 0:
        raise web.HTTPBadRequest(text="frame_id must be non-negative")
    return frame_id


async def _decode_tracking_initialize(request: web.Request) -> tuple[np.ndarray, np.ndarray, int | None]:
    if request.content_type != "application/x-sam2-tracking-init-npz":
        raise web.HTTPUnsupportedMediaType(
            text="use Content-Type: application/x-sam2-tracking-init-npz"
        )
    try:
        with np.load(io.BytesIO(await request.read()), allow_pickle=False) as archive:
            image = np.array(archive["image_bgr"], dtype=np.uint8, copy=True)
            mask = np.array(archive["mask"], dtype=np.uint8, copy=True)
    except (KeyError, ValueError, OSError) as exc:
        raise web.HTTPBadRequest(text=f"invalid tracking initialization NPZ: {exc}") from exc
    if image.ndim != 3 or image.shape[2] != 3:
        raise web.HTTPBadRequest(text=f"image_bgr must be HxWx3, got {image.shape}")
    if mask.shape != image.shape[:2]:
        raise web.HTTPBadRequest(text="mask must have the same HxW shape as image_bgr")
    return image, mask, _parse_optional_frame_id(request.query.get("frame_id"))


async def _prompt_segment(request: web.Request) -> web.Response:
    if request.content_type != "application/x-sam2-prompt-npz":
        raise web.HTTPUnsupportedMediaType(text="use Content-Type: application/x-sam2-prompt-npz")
    try:
        with np.load(io.BytesIO(await request.read()), allow_pickle=False) as archive:
            image = np.array(archive["image_bgr"], dtype=np.uint8, copy=True)
            points = np.array(archive["points"], dtype=np.int32, copy=True).reshape(-1, 2)
            labels = np.array(archive["labels"], dtype=np.int32, copy=True).reshape(-1)
            box = np.array(archive["box"], dtype=np.int32, copy=True).reshape(4)
    except (KeyError, ValueError, OSError) as exc:
        raise web.HTTPBadRequest(text=f"invalid prompt NPZ: {exc}") from exc
    if image.ndim != 3 or image.shape[2] != 3 or len(points) != len(labels):
        raise web.HTTPBadRequest(text="invalid image, points, or labels")
    try:
        result = await asyncio.to_thread(
            _legacy_select_efficientvit_from_points,
            image,
            points=points.tolist(),
            labels=labels.tolist(),
            box=box.tolist(),
            return_formats=("mask",),
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"prompt segmentation failed: {exc}") from exc
    return web.Response(
        body=_encode_first_frame_response(result),
        content_type="application/x-sam2-first-frame-npz",
    )


async def _prompt_segment_sam2(request: web.Request) -> web.Response:
    if request.content_type != "application/x-sam2-prompt-npz":
        raise web.HTTPUnsupportedMediaType(text="use Content-Type: application/x-sam2-prompt-npz")
    try:
        with np.load(io.BytesIO(await request.read()), allow_pickle=False) as archive:
            image = np.array(archive["image_bgr"], dtype=np.uint8, copy=True)
            points = np.array(archive["points"], dtype=np.int32, copy=True).reshape(-1, 2)
            labels = np.array(archive["labels"], dtype=np.int32, copy=True).reshape(-1)
            box = np.array(archive["box"], dtype=np.int32, copy=True).reshape(4)
    except (KeyError, ValueError, OSError) as exc:
        raise web.HTTPBadRequest(text=f"invalid prompt NPZ: {exc}") from exc
    if image.ndim != 3 or image.shape[2] != 3 or len(points) != len(labels) or 1 not in labels:
        raise web.HTTPBadRequest(text="prompt must contain a valid image and at least one positive point")
    try:
        frame = await asyncio.to_thread(
            initialize_tracking_from_points,
            image,
            points=points.tolist(),
            labels=labels.tolist(),
            frame_id=0,
        )
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"SAM2 point segmentation failed: {exc}") from exc
    mask = (np.asarray(frame.mask) > 0).astype(np.uint8)
    x0, y0, x1, y1 = (int(value) for value in box)
    allowed = np.zeros_like(mask)
    allowed[max(0, y0):min(mask.shape[0], y1 + 1), max(0, x0):min(mask.shape[1], x1 + 1)] = 1
    mask &= allowed
    buffer = io.BytesIO()
    np.savez_compressed(buffer, mask=mask, points=points, labels=labels, box=box)
    return web.Response(body=buffer.getvalue(), content_type="application/x-sam2-first-frame-npz")


async def _decode_raw_bgr(request: web.Request) -> tuple[np.ndarray, int | None]:
    if request.content_type != "application/octet-stream":
        raise web.HTTPUnsupportedMediaType(text="use Content-Type: application/octet-stream")
    height = _parse_positive_int(request.query.get("height"), "height")
    width = _parse_positive_int(request.query.get("width"), "width")
    payload = await request.read()
    expected_size = height * width * 3
    if len(payload) != expected_size:
        raise web.HTTPBadRequest(text=f"BGR body has {len(payload)} bytes; expected {expected_size}")
    return (
        np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy(),
        _parse_optional_frame_id(request.query.get("frame_id")),
    )


def _require_active_stream(app: web.Application, request: web.Request) -> str:
    stream_id = request.query.get("stream_id")
    if not stream_id or stream_id != app["tracking_stream_id"]:
        raise web.HTTPConflict(text="unknown or inactive tracking stream_id")
    return stream_id


async def _decode_rgbd_request(request: web.Request) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    if request.content_type != "application/x-sam2-rgbd-npz":
        raise web.HTTPUnsupportedMediaType(text="use Content-Type: application/x-sam2-rgbd-npz")
    try:
        with np.load(io.BytesIO(await request.read()), allow_pickle=False) as archive:
            image_bgr = np.array(archive["image_bgr"], dtype=np.uint8, copy=True)
            depth_image = np.array(archive["depth_image"], copy=True)
            intrinsics = np.array(archive["intrinsics"], dtype=np.float64, copy=True)
            depth_scale = float(archive["depth_scale"])
    except (KeyError, ValueError, OSError) as exc:
        raise web.HTTPBadRequest(text=f"invalid RGB-D NPZ: {exc}") from exc
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise web.HTTPBadRequest(text=f"image_bgr must be HxWx3, got {image_bgr.shape}")
    if depth_image.shape[:2] != image_bgr.shape[:2]:
        raise web.HTTPBadRequest(text="depth_image must be aligned with image_bgr")
    if intrinsics.shape != (3, 3) or depth_scale <= 0:
        raise web.HTTPBadRequest(text="intrinsics must be 3x3 and depth_scale must be positive")
    options_text = request.query.get("options", "{}")
    try:
        options = json.loads(options_text)
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"options must be a JSON object: {exc}") from exc
    if not isinstance(options, dict):
        raise web.HTTPBadRequest(text="options must be a JSON object")
    forbidden = {"return_formats", "return_debug", "num_candidates"}.intersection(options)
    if forbidden:
        raise web.HTTPBadRequest(text=f"options cannot override server-controlled keys: {sorted(forbidden)}")
    return image_bgr, depth_image, intrinsics, depth_scale, options


def _encode_first_frame_response(result: dict[str, Any], *, slots: int | None = None) -> bytes:
    """Return selected RGB-D result and candidate slots in a compact NPZ."""
    candidates = list(result.get("candidates", ()))
    if slots is not None:
        candidates = candidates[:slots]
    present = np.asarray([candidate is not None for candidate in candidates], dtype=bool)
    actual = next((candidate for candidate in candidates if candidate is not None), None)
    mask_shape = np.asarray(result.get("mask", actual["mask"] if actual else np.empty((0, 0)))).shape
    candidate_masks = np.zeros((len(candidates), *mask_shape), dtype=np.uint8)
    candidate_bboxes = np.full((len(candidates), 4), np.nan, dtype=np.float32)
    candidate_areas = np.zeros(len(candidates), dtype=np.int64)
    source_indices = np.full(len(candidates), -1, dtype=np.int64)
    for index, candidate in enumerate(candidates):
        if candidate is None:
            continue
        candidate_masks[index] = np.asarray(candidate["mask"], dtype=np.uint8)
        candidate_bboxes[index] = np.asarray(candidate["bbox"], dtype=np.float32)
        candidate_areas[index] = int(candidate.get("area_px", 0))
        source_indices[index] = int(candidate.get("source_index", -1))
    selected_mask = np.asarray(result.get("mask", np.empty(mask_shape, dtype=np.uint8)), dtype=np.uint8)
    metadata = {
        key: value for key, value in result.items()
        if key not in {"mask", "formats", "candidates", "candidate", "debug"}
        and isinstance(value, (str, int, float, bool, type(None), list, dict))
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        mask=selected_mask,
        bbox=np.asarray(result.get("bbox", (np.nan,) * 4), dtype=np.float32),
        points=np.asarray(result.get("points", ()), dtype=np.int32).reshape(-1, 2),
        labels=np.asarray(result.get("labels", ()), dtype=np.int8),
        candidate_masks=candidate_masks,
        candidate_bboxes=candidate_bboxes,
        candidate_areas=candidate_areas,
        candidate_source_indices=source_indices,
        candidate_present=present,
        metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )
    return buffer.getvalue()


async def _worker(app: web.Application) -> None:
    jobs: asyncio.Queue[_SegmentJob] = app["jobs"]
    while True:
        job = await jobs.get()
        try:
            result = await asyncio.to_thread(
                segment_full_image_top_n,
                job.image_bgr,
                num_masks=job.num_masks,
                roi=job.roi,
                min_mask_area_px=job.min_mask_area_px,
                return_formats=("mask",),
            )
            if not job.response.done():
                job.response.set_result(result)
        except BaseException as exc:
            if not job.response.done():
                job.response.set_exception(exc)
        finally:
            jobs.task_done()


async def _on_startup(app: web.Application) -> None:
    config: ServerConfig = app["config"]
    if not config.tracking_only:
        _log(f"startup 1/3: loading {config.backend} segmentation model")
        await asyncio.to_thread(start, backend=config.backend)
        _log("startup 1/3 complete: segmentation model is resident")
    _log("startup 2/3: loading Efficient-SAM2 tracking model")
    await asyncio.to_thread(start_tracking)
    _log("startup 2/3 complete: tracking model is resident")
    app["worker_task"] = (
        None if config.tracking_only
        else asyncio.create_task(_worker(app), name="sam2-http-uds-worker")
    )
    _log("startup 3/3 complete: request workers are ready")


async def _on_cleanup(app: web.Application) -> None:
    worker: asyncio.Task[None] | None = app.get("worker_task")
    if worker is not None:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    if not app["config"].tracking_only:
        await asyncio.to_thread(close)
    await asyncio.to_thread(close_tracking)


async def _segment(request: web.Request) -> web.Response:
    app = request.app
    job = await _decode_request(request)
    jobs: asyncio.Queue[_SegmentJob] = app["jobs"]
    try:
        jobs.put_nowait(job)
    except asyncio.QueueFull as exc:
        raise web.HTTPTooManyRequests(text="segmentation queue is full") from exc
    timeout_s: float = app["config"].request_timeout_s
    try:
        result = await asyncio.wait_for(asyncio.shield(job.response), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise web.HTTPGatewayTimeout(text="segmentation request timed out") from exc
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"segmentation failed: {exc}") from exc
    return web.Response(
        body=_encode_response(result),
        content_type="application/x-sam2-segmentation-npz",
    )


async def _first_frame_mask(request: web.Request) -> web.Response:
    image_bgr, depth_image, intrinsics, depth_scale, options = await _decode_rgbd_request(request)
    try:
        result = await asyncio.to_thread(
            select_first_frame_mask,
            image_bgr, depth_image, intrinsics, depth_scale,
            return_formats=("mask",), **options,
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"first-frame selection failed: {exc}") from exc
    return web.Response(
        body=_encode_first_frame_response(result),
        content_type="application/x-sam2-first-frame-npz",
    )


async def _first_frame_candidates(request: web.Request) -> web.Response:
    image_bgr, depth_image, intrinsics, depth_scale, options = await _decode_rgbd_request(request)
    num_candidates = _parse_positive_int(request.query.get("num_candidates"), "num_candidates", default=1)
    try:
        result = await asyncio.to_thread(
            select_first_frame_candidates,
            image_bgr, depth_image, intrinsics, depth_scale,
            num_candidates=num_candidates, **options,
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"first-frame candidates failed: {exc}") from exc
    return web.Response(
        body=_encode_first_frame_response(result, slots=num_candidates),
        content_type="application/x-sam2-first-frame-npz",
    )


async def _first_frame_candicates_max(request: web.Request) -> web.Response:
    """RGB-D candidate endpoint with denser prompts and 2-mm support."""
    image_bgr, depth_image, intrinsics, depth_scale, options = await _decode_rgbd_request(request)
    num_candidates = _parse_positive_int(request.query.get("num_candidates"), "num_candidates", default=10)
    try:
        result = await asyncio.to_thread(
            select_first_frame_candicates_max,
            image_bgr, depth_image, intrinsics, depth_scale,
            num_candidates=num_candidates, **options,
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"first-frame candicates-max failed: {exc}") from exc
    return web.Response(
        body=_encode_first_frame_response(result, slots=num_candidates),
        content_type="application/x-sam2-first-frame-npz",
    )


async def _first_frame_candidate_components(request: web.Request) -> web.Response:
    """Max-style RGB-D candidates without lower component-size bounds."""
    image_bgr, depth_image, intrinsics, depth_scale, options = await _decode_rgbd_request(request)
    num_candidates = _parse_positive_int(request.query.get("num_candidates"), "num_candidates", default=10)
    try:
        result = await asyncio.to_thread(
            select_first_frame_candidate_components,
            image_bgr, depth_image, intrinsics, depth_scale,
            num_candidates=num_candidates, **options,
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    except Exception as exc:
        raise web.HTTPInternalServerError(text=f"first-frame candidate-components failed: {exc}") from exc
    return web.Response(
        body=_encode_first_frame_response(result, slots=num_candidates),
        content_type="application/x-sam2-first-frame-npz",
    )


async def _tracking_initialize(request: web.Request) -> web.Response:
    app = request.app
    image_bgr, mask, frame_id = await _decode_tracking_initialize(request)
    async with app["tracking_lock"]:
        if app["tracking_stream_id"] is not None:
            raise web.HTTPConflict(text="a tracking stream is already active")
        try:
            frame = await asyncio.to_thread(
                initialize_tracking_from_mask, image_bgr, mask, frame_id=frame_id
            )
        except Exception as exc:
            raise web.HTTPInternalServerError(text=f"tracking initialization failed: {exc}") from exc
        stream_id = secrets.token_urlsafe(18)
        app["tracking_stream_id"] = stream_id
    batch = TrackingBatchResult(
        frames=(frame,), propagated_count=0, queued_before=0, queued_after=0, latency_ms=0.0
    )
    return web.Response(
        body=_encode_tracking_response(batch, stream_id=stream_id, initialized=True),
        content_type="application/x-sam2-tracking-npz",
    )


async def _tracking_append(request: web.Request) -> web.Response:
    app = request.app
    image_bgr, frame_id = await _decode_raw_bgr(request)
    async with app["tracking_lock"]:
        stream_id = _require_active_stream(app, request)
        try:
            assigned_frame_id = await asyncio.to_thread(
                append_tracking_frame, image_bgr, frame_id=frame_id
            )
        except BufferError as exc:
            raise web.HTTPTooManyRequests(text=str(exc)) from exc
        except Exception as exc:
            raise web.HTTPInternalServerError(text=f"tracking append failed: {exc}") from exc
    return web.json_response({"stream_id": stream_id, "frame_id": assigned_frame_id})


async def _tracking_propagate(request: web.Request) -> web.Response:
    app = request.app
    async with app["tracking_lock"]:
        stream_id = _require_active_stream(app, request)
        try:
            batch = await asyncio.to_thread(propagate_tracking)
        except Exception as exc:
            raise web.HTTPInternalServerError(text=f"tracking propagation failed: {exc}") from exc
    return web.Response(
        body=_encode_tracking_response(batch, stream_id=stream_id),
        content_type="application/x-sam2-tracking-npz",
    )


async def _tracking_close(request: web.Request) -> web.Response:
    app = request.app
    async with app["tracking_lock"]:
        stream_id = _require_active_stream(app, request)
        # The existing tracker API reinitializes state on the next initialize;
        # closing here releases its model promptly for workloads that alternate
        # between tracking sessions.
        await asyncio.to_thread(close_tracking)
        await asyncio.to_thread(start_tracking)
        app["tracking_stream_id"] = None
    return web.json_response({"closed_stream_id": stream_id})


async def _health(request: web.Request) -> web.Response:
    return web.json_response({
        "service": "sam2-service",
        "queue_depth": request.app["jobs"].qsize(),
        "tracking_stream_active": request.app["tracking_stream_id"] is not None,
        **status(),
    })


def create_app(config: ServerConfig) -> web.Application:
    app = web.Application(client_max_size=64 * 1024**2)
    app["config"] = config
    app["jobs"] = asyncio.Queue(maxsize=config.queue_capacity)
    app["tracking_lock"] = asyncio.Lock()
    app["tracking_stream_id"] = None
    app.router.add_get("/healthz", _health)
    app.router.add_post("/v1/segment", _segment)
    app.router.add_post("/v1/prompt-segment", _prompt_segment)
    app.router.add_post("/v1/prompt-segment-sam2", _prompt_segment_sam2)
    app.router.add_post("/v1/first-frame/mask", _first_frame_mask)
    app.router.add_post("/v1/first-frame/candidates", _first_frame_candidates)
    app.router.add_post("/v1/first-frame/candicates-max", _first_frame_candicates_max)
    app.router.add_post("/v1/first-frame/candidate-components", _first_frame_candidate_components)
    app.router.add_post("/v1/tracking/initialize", _tracking_initialize)
    app.router.add_post("/v1/tracking/append", _tracking_append)
    app.router.add_post("/v1/tracking/propagate", _tracking_propagate)
    app.router.add_post("/v1/tracking/close", _tracking_close)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


async def _serve(config: ServerConfig, socket_mode: int) -> None:
    socket_path = config.socket_path
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if not stat.S_ISSOCK(socket_path.stat().st_mode):
            raise RuntimeError(f"refusing to replace non-socket path: {socket_path}")
        socket_path.unlink()
    app = create_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    uds_site = web.UnixSite(runner, str(socket_path))
    await uds_site.start()
    os.chmod(socket_path, socket_mode)
    # 记录本实例实际绑定的目录项。历史重复实例退出时不得删除后来实例
    # 重新创建的同名 socket，否则会出现“进程仍在但客户端无法连接”。
    bound_socket_stat = socket_path.stat()
    bound_socket_identity = (bound_socket_stat.st_dev, bound_socket_stat.st_ino)
    if config.pid_file is not None:
        config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        config.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    _log(f"ready: listening on unix://{socket_path}")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        await stop.wait()
    finally:
        _log("shutdown: stopping resident models")
        await runner.cleanup()
        try:
            current_socket_stat = socket_path.stat()
            current_socket_identity = (current_socket_stat.st_dev, current_socket_stat.st_ino)
            if stat.S_ISSOCK(current_socket_stat.st_mode) \
                    and current_socket_identity == bound_socket_identity:
                socket_path.unlink()
        except FileNotFoundError:
            pass
        if config.pid_file is not None:
            try:
                if config.pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    config.pid_file.unlink()
            except FileNotFoundError:
                pass
        _log("shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 HTTP-over-UDS service")
    parser.add_argument("--socket", default=str(Path(__file__).resolve().parent / "runtime/sam2-service.sock"), help="Unix socket path")
    parser.add_argument("--backend", choices=("sam2", "efficientvit"), default="efficientvit")
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--tracking-only", action="store_true", help="只加载点提示/跟踪SAM2模型")
    parser.add_argument("--socket-mode", default="660", help="Unix permission bits, e.g. 660")
    parser.add_argument("--pid-file", type=Path, help="Write the resident service PID after it is ready")
    args = parser.parse_args()
    if args.queue_capacity < 1 or args.request_timeout_s <= 0:
        parser.error("queue-capacity must be >= 1 and request-timeout-s must be positive")
    try:
        socket_mode = int(args.socket_mode, 8)
    except ValueError as exc:
        parser.error(f"invalid socket mode: {exc}")
    socket_path = Path(args.socket)
    lock_path = Path(f"{socket_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # flock 由内核在进程退出时自动释放，覆盖 GCS、手工脚本和直接执行入口。
    # PID 文件只能用于定位进程，真正消除“检查后同时启动”竞态的是这把锁。
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another sam2-service already owns {lock_path}") from exc
        if args.pid_file is not None:
            args.pid_file.parent.mkdir(parents=True, exist_ok=True)
            args.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
        try:
            asyncio.run(
                _serve(
                    ServerConfig(
                        socket_path=socket_path,
                        pid_file=args.pid_file,
                        backend=args.backend,
                        queue_capacity=args.queue_capacity,
                        request_timeout_s=args.request_timeout_s,
                        tracking_only=args.tracking_only,
                    ),
                    socket_mode,
                )
            )
        finally:
            if args.pid_file is not None:
                try:
                    if args.pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                        args.pid_file.unlink()
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
