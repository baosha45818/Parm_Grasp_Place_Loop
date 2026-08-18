"""Thread-owned, caller-controlled online Efficient-SAM2 tracking runtime."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Iterable, Literal, overload

import cv2
import numpy as np


SERVICE_ROOT = Path(__file__).resolve().parents[1]
EFFICIENT_SAM2_ROOT = SERVICE_ROOT / "3rdparty" / "efficient-sam2"
DEFAULT_CHECKPOINT = SERVICE_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
DEFAULT_BYPASS_CHECKPOINT = EFFICIENT_SAM2_ROOT / "bypass" / "ckpt" / "bypass_bottleneck_base.pth"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


@dataclass(frozen=True)
class TrackingFrameResult:
    frame_id: int
    mask: np.ndarray
    overlay: np.ndarray
    bbox: list[float] | None
    mask_pixels: int


@dataclass(frozen=True)
class TrackingBatchResult:
    frames: tuple[TrackingFrameResult, ...]
    propagated_count: int
    queued_before: int
    queued_after: int
    latency_ms: float

    @property
    def latest(self) -> TrackingFrameResult:
        if not self.frames:
            raise RuntimeError("tracking result is empty")
        return self.frames[-1]

    @property
    def masks(self) -> tuple[np.ndarray, ...]:
        return tuple(frame.mask for frame in self.frames)


def _copy_frame_result(result: TrackingFrameResult) -> TrackingFrameResult:
    return TrackingFrameResult(
        frame_id=result.frame_id,
        mask=np.array(result.mask, copy=True),
        overlay=np.array(result.overlay, copy=True),
        bbox=None if result.bbox is None else list(result.bbox),
        mask_pixels=result.mask_pixels,
    )


def _copy_batch_result(result: TrackingBatchResult) -> TrackingBatchResult:
    return TrackingBatchResult(
        frames=tuple(_copy_frame_result(item) for item in result.frames),
        propagated_count=result.propagated_count,
        queued_before=result.queued_before,
        queued_after=result.queued_after,
        latency_ms=result.latency_ms,
    )


def _bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.array(frame, copy=True)
    foreground = mask > 0
    if foreground.any():
        red = np.asarray([0, 0, 255], dtype=np.float32)
        output[foreground] = np.clip(
            output[foreground].astype(np.float32) * 0.45 + red * 0.55, 0, 255
        ).astype(np.uint8)
        box = _bbox(mask)
        assert box is not None
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    return output


def _normalize_frames(frames: np.ndarray | Iterable[np.ndarray]) -> list[np.ndarray]:
    if isinstance(frames, np.ndarray):
        values = [frames]
    else:
        values = list(frames)
    output = []
    for frame in values:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected BGR HxWx3 frame, got {array.shape}")
        output.append(np.array(array, dtype=np.uint8, copy=True))
    return output


class _TrackerWorker:
    def __init__(self, owner: "EfficientSam2Tracker"):
        self.owner = owner
        self.requests: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="efficient-sam2-tracker", daemon=True)

    def start(self) -> None:
        if self.thread.is_alive():
            return
        self.thread.start()
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError("Efficient-SAM2 tracker startup failed") from self.error

    def call(self, method: str, *args, **kwargs):
        if not self.thread.is_alive():
            raise RuntimeError("tracking worker is not running")
        response: queue.Queue = queue.Queue(maxsize=1)
        self.requests.put((method, args, kwargs, response))
        ok, value = response.get()
        if ok:
            return value
        raise value

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        self.call("_close_runtime")
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            raise TimeoutError("tracking worker did not stop")

    def _run(self) -> None:
        try:
            self.owner._open_runtime()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
            return
        self.ready.set()
        while True:
            method, args, kwargs, response = self.requests.get()
            try:
                value = getattr(self.owner, method)(*args, **kwargs)
                response.put((True, value))
                if method == "_close_runtime":
                    return
            except BaseException as exc:
                response.put((False, exc))


class EfficientSam2Tracker:
    """Keep Efficient-SAM2 resident and propagate only when requested.

    Submitted frames are copied into a bounded pending queue. No inference is
    performed by :meth:`append`, :meth:`append_many`, or :meth:`observe`.
    """

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        queue_capacity: int = 32,
        overflow_policy: Literal["raise", "drop_oldest"] = "raise",
        result_history: int = 32,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        bypass_checkpoint: str | Path = DEFAULT_BYPASS_CHECKPOINT,
        config: str = DEFAULT_CONFIG,
        wb_theta: float = 0.7,
        memory_drop_ratio: float = 0.95,
        memory_stride: int = 1,
        dilate_mask: bool = True,
        dilate_kernel_size: int = 5,
        mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16",
        compile_model: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
        warmup_shape: tuple[int, int] = (720, 1280),
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        if overflow_policy not in ("raise", "drop_oldest"):
            raise ValueError(f"unsupported overflow_policy: {overflow_policy}")
        self.device_request = str(device)
        self.queue_capacity = int(queue_capacity)
        self.overflow_policy = overflow_policy
        self.result_history = max(1, int(result_history))
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.bypass_checkpoint = Path(bypass_checkpoint).expanduser().resolve()
        self.config = str(config)
        self.wb_theta = float(wb_theta)
        self.memory_drop_ratio = float(memory_drop_ratio)
        self.memory_stride = max(1, int(memory_stride))
        self.dilate_mask = bool(dilate_mask)
        self.dilate_kernel_size = max(1, int(dilate_kernel_size))
        if mixed_precision not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"unsupported mixed_precision: {mixed_precision}")
        if compile_mode not in ("default", "reduce-overhead", "max-autotune"):
            raise ValueError(f"unsupported compile_mode: {compile_mode}")
        if len(warmup_shape) != 2 or min(map(int, warmup_shape)) < 1:
            raise ValueError("warmup_shape must be a positive (height, width) pair")
        self.mixed_precision = mixed_precision
        self.compile_model = bool(compile_model)
        self.compile_mode = compile_mode
        self.warmup_shape = tuple(map(int, warmup_shape))

        self._worker: _TrackerWorker | None = None
        self._torch = None
        self._predictor = None
        self._device = None
        self._inference_state: dict | None = None
        self._pending: deque[tuple[int, np.ndarray]] = deque()
        self._results: deque[TrackingFrameResult] = deque(maxlen=self.result_history)
        self._anchor_frame: np.ndarray | None = None
        self._anchor_mask: np.ndarray | None = None
        self._anchor_id: int | None = None
        self._next_frame_id = 0
        self._dropped_frames = 0
        self._propagation_count = 0
        self._last_latency_ms = 0.0
        self._compile_warmup_ms = 0.0

    def start(self) -> dict:
        if self._worker is None:
            self._worker = _TrackerWorker(self)
            self._worker.start()
        return self.status()

    def close(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()

    def __enter__(self) -> "EfficientSam2Tracker":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _call(self, method: str, *args, **kwargs):
        self.start()
        assert self._worker is not None
        return self._worker.call(method, *args, **kwargs)

    def initialize_from_mask(self, image_bgr: np.ndarray, mask: np.ndarray, *, frame_id: int | None = None) -> TrackingFrameResult:
        return self._call("_initialize", np.array(image_bgr, copy=True), "mask", np.array(mask, copy=True), frame_id)

    def initialize_from_points(self, image_bgr: np.ndarray, *, points, labels, frame_id: int | None = None) -> TrackingFrameResult:
        prompt = (np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32))
        return self._call("_initialize", np.array(image_bgr, copy=True), "points", prompt, frame_id)

    def initialize_from_box(self, image_bgr: np.ndarray, *, box_xyxy, frame_id: int | None = None) -> TrackingFrameResult:
        return self._call("_initialize", np.array(image_bgr, copy=True), "box", np.asarray(box_xyxy, dtype=np.float32), frame_id)

    def append(self, image_bgr: np.ndarray, *, frame_id: int | None = None) -> int:
        return self._call("_append", np.array(image_bgr, copy=True), frame_id)

    observe = append

    def append_many(self, images: Iterable[np.ndarray]) -> tuple[int, ...]:
        return self._call("_append_many", _normalize_frames(images))

    def propagate_queued(self) -> TrackingBatchResult:
        return self._call("_propagate", [])

    def propagate_with_frames(self, frames: np.ndarray | Iterable[np.ndarray]) -> TrackingBatchResult:
        return self._call("_propagate", _normalize_frames(frames))

    @overload
    def propagate(self) -> TrackingBatchResult: ...

    @overload
    def propagate(self, frames: np.ndarray | Iterable[np.ndarray]) -> TrackingBatchResult: ...

    def propagate(self, frames=None) -> TrackingBatchResult:
        return self.propagate_queued() if frames is None else self.propagate_with_frames(frames)

    def latest(self) -> TrackingFrameResult | None:
        return self._call("_latest")

    def status(self) -> dict:
        if self._worker is None:
            return {
                "started": False,
                "initialized": False,
                "queue_capacity": self.queue_capacity,
                "queued_frames": 0,
            }
        return self._worker.call("_status")

    def reset(self) -> None:
        self._call("_reset")

    def _open_runtime(self) -> None:
        root_text = str(EFFICIENT_SAM2_ROOT)
        if root_text in sys.path:
            sys.path.remove(root_text)
        sys.path.insert(0, root_text)
        if not EFFICIENT_SAM2_ROOT.is_dir():
            raise FileNotFoundError(f"Efficient-SAM2 source not found: {EFFICIENT_SAM2_ROOT}")
        for path in (self.checkpoint, self.bypass_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {path}")

        import torch
        from bypass.bypass_modeling import build_bypass_model
        from build_model.window_bypass import build_WB_model
        from sam2.build_sam import build_sam2_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("Efficient-SAM2 requires CUDA")
        device = torch.device(self.device_request)
        predictor = build_sam2_video_predictor(self.config, str(self.checkpoint), device=device)
        args = Namespace(
            apply_WB=True, sam2_model="base+", selected_layers=list(range(6, 21)),
            win_sel_layer=[6], fpn_feat_layer=None, scale_layer=None,
            final_global_layer=20, inner_channel=448, WB_all_layer=False,
            Mem_filter=False, disable_WB=False, small_bypass=False,
            prune_memory=True, pool_memory=False, pooling_ks=2,
        )
        predictor.memory_temporal_stride_for_eval = self.memory_stride
        predictor.print_WS = predictor.random_mask = predictor.uniform_mask = False
        predictor.topk_mask = True
        predictor.set_drop_ratio = self.memory_drop_ratio
        predictor.MTP_theta = 0.6
        predictor.disable_WB = False
        predictor.sam2_model = "base+"
        predictor.win_sel_layer = [6]
        predictor.fpn_feat_layer = predictor.scale_layer = None
        predictor.Mem_Frame_Prune = False
        predictor.num_frame_to_prune = 0
        predictor.WB_theta = self.wb_theta
        predictor.obj_theta = 0.0
        predictor.sel_max_iou = False
        predictor.init_memory_info(enable_MeP_info=True)
        build_WB_model(args, predictor, selected_layers=args.selected_layers)
        build_bypass_model(args, predictor, adapter_type="bottleneck", adapter=True, training=False)
        bypass = predictor.image_encoder.trunk.blocks[20].bypass_branch
        bypass.load_state_dict(torch.load(self.bypass_checkpoint, map_location=device, weights_only=True))
        bypass.eval()
        predictor.prune_memory = True
        predictor.dilate_mask = self.dilate_mask
        predictor.dilate_kernel_size = self.dilate_kernel_size
        predictor.eval()
        self._torch, self._predictor, self._device = torch, predictor, device
        if self.compile_model:
            self._compile_and_warmup()

    def _autocast_context(self):
        assert self._torch is not None
        if self.mixed_precision == "fp32":
            return nullcontext()
        dtype = self._torch.bfloat16 if self.mixed_precision == "bf16" else self._torch.float16
        return self._torch.autocast("cuda", dtype=dtype)

    def _compile_and_warmup(self) -> None:
        """Compile hot inference components and materialize kernels before ready."""
        assert self._predictor is not None and self._torch is not None
        started = time.perf_counter()
        modules = (
            self._predictor.memory_encoder,
            self._predictor.sam_prompt_encoder,
            self._predictor.sam_mask_decoder,
        )
        for module in modules:
            module.forward = self._torch.compile(
                module.forward,
                mode=self.compile_mode,
                fullgraph=False,
                dynamic=None,
            )

        height, width = self.warmup_shape
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(2)]
        mask = np.zeros((height, width), dtype=np.uint8)
        y0, y1 = height // 3, max(height // 3 + 1, 2 * height // 3)
        x0, x1 = width // 3, max(width // 3 + 1, 2 * width // 3)
        mask[y0:y1, x0:x1] = 1
        state = self._init_state_from_frames(frames)
        self._propagate_state(
            state,
            start_frame_idx=0,
            frame_count=2,
            kind="mask",
            prompt=mask,
        )
        self._torch.cuda.synchronize(self._device)
        self._compile_warmup_ms = (time.perf_counter() - started) * 1000.0

    def _allocate_id(self, requested: int | None) -> int:
        if requested is None:
            value = self._next_frame_id
            self._next_frame_id += 1
            return value
        value = int(requested)
        self._next_frame_id = max(self._next_frame_id, value + 1)
        return value

    def _validate_shape(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected BGR HxWx3 frame, got {frame.shape}")
        if self._anchor_frame is not None and frame.shape != self._anchor_frame.shape:
            raise ValueError(f"frame shape changed from {self._anchor_frame.shape} to {frame.shape}")

    def _initialize(self, frame, kind: str, prompt, frame_id) -> TrackingFrameResult:
        frame = np.asarray(frame, dtype=np.uint8)
        self._pending.clear()
        self._results.clear()
        self._anchor_frame = self._anchor_mask = None
        self._anchor_id = None
        self._validate_shape(frame)
        fid = self._allocate_id(frame_id)
        state = self._init_state_from_frames([frame])
        self._inference_state = state
        masks = self._propagate_state(state, start_frame_idx=0, frame_count=1, kind=kind, prompt=prompt)
        mask = masks[0]
        self._anchor_frame, self._anchor_mask, self._anchor_id = frame.copy(), mask.copy(), fid
        result = self._pack(fid, frame, mask)
        self._results.append(result)
        return _copy_frame_result(result)

    def _append(self, frame, frame_id) -> int:
        if self._anchor_mask is None:
            raise RuntimeError("tracker is not initialized")
        frame = np.asarray(frame, dtype=np.uint8)
        self._validate_shape(frame)
        if len(self._pending) >= self.queue_capacity:
            if self.overflow_policy == "raise":
                raise BufferError(f"tracking frame queue is full ({self.queue_capacity})")
            self._pending.popleft()
            self._dropped_frames += 1
        fid = self._allocate_id(frame_id)
        self._pending.append((fid, frame.copy()))
        return fid

    def _append_many(self, frames) -> tuple[int, ...]:
        if self.overflow_policy == "raise" and len(self._pending) + len(frames) > self.queue_capacity:
            raise BufferError(
                f"tracking frame queue capacity {self.queue_capacity} cannot hold "
                f"{len(self._pending) + len(frames)} frames"
            )
        return tuple(self._append(frame, None) for frame in frames)

    def _propagate(self, new_frames) -> TrackingBatchResult:
        if self._anchor_frame is None or self._anchor_mask is None or self._anchor_id is None:
            raise RuntimeError("tracker is not initialized")
        queued_before = len(self._pending)
        self._append_many(new_frames)
        if not self._pending:
            current = self._pack(self._anchor_id, self._anchor_frame, self._anchor_mask)
            return TrackingBatchResult((current,), 0, queued_before, 0, 0.0)

        pending = list(self._pending)
        state = self._inference_state
        if state is None:
            raise RuntimeError("tracker inference state is not initialized")
        first_new_idx = int(state["num_frames"])
        started = time.perf_counter()
        new_images = self._preprocess_frames([frame for _, frame in pending])
        state["images"].extend(new_images)
        state["num_frames"] += len(new_images)
        masks = self._propagate_state(
            state,
            start_frame_idx=first_new_idx,
            frame_count=len(new_images),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        results = tuple(
            self._pack(fid, frame, mask)
            for (fid, frame), mask in zip(pending, masks)
        )
        last = results[-1]
        self._anchor_id = last.frame_id
        self._anchor_frame = pending[-1][1].copy()
        self._anchor_mask = last.mask.copy()
        self._pending.clear()
        self._results.extend(results)
        self._propagation_count += 1
        self._last_latency_ms = latency_ms
        return _copy_batch_result(TrackingBatchResult(results, len(results), queued_before, 0, latency_ms))

    def _propagate_state(
        self,
        state: dict,
        *,
        start_frame_idx: int,
        frame_count: int,
        kind: str | None = None,
        prompt=None,
    ) -> list[np.ndarray]:
        assert self._predictor is not None and self._torch is not None and self._device is not None
        masks: list[np.ndarray | None] = [None] * frame_count
        with self._torch.inference_mode(), self._autocast_context():
            if kind == "mask":
                mask = np.asarray(prompt) > 0
                target_shape = (int(state["video_height"]), int(state["video_width"]))
                if mask.shape != target_shape:
                    mask = cv2.resize(
                        mask.astype(np.uint8), target_shape[::-1], interpolation=cv2.INTER_NEAREST
                    ) > 0
                self._predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
            elif kind == "box":
                self._predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=prompt)
            elif kind == "points":
                points, labels = prompt
                self._predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1, points=points, labels=labels,
                )
            elif kind is not None:
                raise ValueError(f"unsupported prompt kind: {kind}")
            for frame_idx, _ids, logits in self._predictor.propagate_in_video(
                state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=frame_count,
            ):
                offset = int(frame_idx) - start_frame_idx
                if 0 <= offset < frame_count:
                    masks[offset] = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
        if any(mask is None for mask in masks):
            raise RuntimeError("Efficient-SAM2 did not return every requested frame mask")
        return [mask for mask in masks if mask is not None]

    def _preprocess_frames(self, frames: list[np.ndarray]) -> list:
        assert self._predictor is not None and self._torch is not None and self._device is not None
        if not frames:
            return []
        torch = self._torch
        image_size = int(self._predictor.image_size)
        resized_rgb = np.stack(
            [
                cv2.cvtColor(
                    cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_BGR2RGB,
                )
                for frame in frames
            ],
            axis=0,
        )
        images = torch.from_numpy(resized_rgb).permute(0, 3, 1, 2).to(
            device=self._device, dtype=torch.float32, non_blocking=True
        )
        images.div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self._device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=self._device)[:, None, None]
        images.sub_(mean).div_(std)
        return list(images.unbind(0))

    def _init_state_from_frames(self, frames: list[np.ndarray]) -> dict:
        """Build the predictor state directly from BGR arrays without JPEG/PIL I/O."""
        assert self._predictor is not None and self._torch is not None and self._device is not None
        video_height, video_width = frames[0].shape[:2]
        images = self._preprocess_frames(frames)
        state = {
            "images": images,
            "num_frames": len(frames),
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": video_height,
            "video_width": video_width,
            "device": self._device,
            "storage_device": self._device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "cached_features_WBP": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        with self._torch.inference_mode(), self._autocast_context():
            self._predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        return state

    def _pack(self, frame_id, frame, mask) -> TrackingFrameResult:
        mask = np.asarray(mask, dtype=np.uint8)
        return TrackingFrameResult(
            frame_id=int(frame_id), mask=mask.copy(), overlay=_overlay(frame, mask),
            bbox=_bbox(mask), mask_pixels=int(np.count_nonzero(mask)),
        )

    def _latest(self):
        return None if not self._results else _copy_frame_result(self._results[-1])

    def _status(self) -> dict:
        return {
            "started": True,
            "initialized": self._anchor_mask is not None,
            "device": str(self._device),
            "queue_capacity": self.queue_capacity,
            "queued_frames": len(self._pending),
            "overflow_policy": self.overflow_policy,
            "dropped_frames": self._dropped_frames,
            "anchor_frame_id": self._anchor_id,
            "next_frame_id": self._next_frame_id,
            "propagation_count": self._propagation_count,
            "last_latency_ms": self._last_latency_ms,
            "state_frames": 0 if self._inference_state is None else self._inference_state["num_frames"],
            "mixed_precision": self.mixed_precision,
            "compile_model": self.compile_model,
            "compile_mode": self.compile_mode if self.compile_model else None,
            "compile_warmup_ms": self._compile_warmup_ms,
        }

    def _reset(self) -> None:
        self._pending.clear()
        self._results.clear()
        self._anchor_frame = self._anchor_mask = None
        self._anchor_id = None
        self._inference_state = None
        self._next_frame_id = 0
        self._dropped_frames = self._propagation_count = 0
        self._last_latency_ms = 0.0

    def _close_runtime(self) -> None:
        self._reset()
        self._predictor = self._torch = self._device = None


_DEFAULT_TRACKER: EfficientSam2Tracker | None = None
_DEFAULT_LOCK = threading.Lock()


def start_tracking(**kwargs) -> dict:
    global _DEFAULT_TRACKER
    with _DEFAULT_LOCK:
        if _DEFAULT_TRACKER is None:
            _DEFAULT_TRACKER = EfficientSam2Tracker(**kwargs)
        elif kwargs:
            raise RuntimeError("default tracker already exists; close it before changing options")
        return _DEFAULT_TRACKER.start()


def get_tracker() -> EfficientSam2Tracker:
    if _DEFAULT_TRACKER is None:
        raise RuntimeError("default tracker has not been started")
    return _DEFAULT_TRACKER


def close_tracking() -> None:
    global _DEFAULT_TRACKER
    with _DEFAULT_LOCK:
        tracker, _DEFAULT_TRACKER = _DEFAULT_TRACKER, None
    if tracker is not None:
        tracker.close()


def append_frame(image_bgr: np.ndarray, *, frame_id: int | None = None) -> int:
    return get_tracker().append(image_bgr, frame_id=frame_id)


def append_frames(images: Iterable[np.ndarray]) -> tuple[int, ...]:
    return get_tracker().append_many(images)


def initialize_tracking_from_mask(
    image_bgr: np.ndarray, mask: np.ndarray, *, frame_id: int | None = None
) -> TrackingFrameResult:
    return get_tracker().initialize_from_mask(image_bgr, mask, frame_id=frame_id)


def initialize_tracking_from_points(
    image_bgr: np.ndarray, *, points, labels, frame_id: int | None = None
) -> TrackingFrameResult:
    return get_tracker().initialize_from_points(
        image_bgr, points=points, labels=labels, frame_id=frame_id
    )


def initialize_tracking_from_box(
    image_bgr: np.ndarray, *, box_xyxy, frame_id: int | None = None
) -> TrackingFrameResult:
    return get_tracker().initialize_from_box(
        image_bgr, box_xyxy=box_xyxy, frame_id=frame_id
    )


def propagate(frames=None) -> TrackingBatchResult:
    return get_tracker().propagate(frames)


__all__ = [
    "EfficientSam2Tracker", "TrackingBatchResult", "TrackingFrameResult",
    "append_frame", "append_frames", "close_tracking", "get_tracker",
    "initialize_tracking_from_box", "initialize_tracking_from_mask",
    "initialize_tracking_from_points", "propagate", "start_tracking",
]
