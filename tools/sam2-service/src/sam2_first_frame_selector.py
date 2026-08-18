"""Small, standalone SAM2 first-frame selector.

This module contains two first-frame paths:

    RGB-D prompts -> EfficientViT proposals -> bbox-area filtering and ranking
    RGB image -> full-image AMG -> original-mask-area ranking

The RGB-D path returns the selected binary mask with its bbox, prompt points
and ranked candidates for downstream tracking.  The full-image path returns
the requested largest AMG proposals without point-cloud preprocessing.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np


DEFAULT_MAX_MASK_AREA_RATIO = 0.04
DEFAULT_MAX_BBOX_SPAN_X_RATIO = 0.36
DEFAULT_MAX_BBOX_SPAN_Y_RATIO = 0.36
DEFAULT_MAX_SINGLE_OBJECT_BBOX_AREA_PX = 17_800
DEFAULT_NORMAL_OBJECT_MASK_AREA_PX = 10_000
DEFAULT_BBOX_AREA_REFERENCE_DEPTH_M = 0.420
DEFAULT_BBOX_AREA_MIN_FACTOR = 0.45
DEFAULT_BBOX_AREA_MAX_FACTOR = 1.1
DEFAULT_SIMILAR_BBOX_AREA_RATIO = 0.10
# A mask shorter than about 70 mm is usually only the exposed part of one of
# the workpieces at the fixed observation distance.  The complete parts seen
# in the calibration frames are about 78--90 mm long.
DEFAULT_OBJECT_LENGTH_RANGE_M = (0.063, 0.095)
DEFAULT_OBJECT_WIDTH_RANGE_M = (0.020, 0.045)
DEFAULT_OBJECT_THICKNESS_RANGE_M = (0.004, 0.020)
MIN_PROMPT_COMPONENT_AREA_PX = 100
MIN_PROMPT_COMPONENT_LARGEST_RATIO = 0.03
SUPPORTED_RETURN_FORMATS = ("mask", "bboxmask", "rotated_bboxmask")
SUPPORTED_BACKENDS = ("sam2", "efficientvit")
SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM2_REPO = SERVICE_ROOT / "3rdparty/sam2"
DEFAULT_SAM2_CKPT = SERVICE_ROOT / "checkpoints/sam2.1_hiera_base_plus.pt"
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_EFFICIENTVIT_REPO = SERVICE_ROOT / "3rdparty/efficientvit"
DEFAULT_EFFICIENTVIT_MODEL = "efficientvit-sam-l0"
DEFAULT_EFFICIENTVIT_WEIGHTS = (
    SERVICE_ROOT / "checkpoints/efficientvit_sam_l0.pt"
)


class _ResidentSegmentationWorker:
    """Own the AMG model and execute all inference on one persistent thread."""

    def __init__(
        self,
        ckpt,
        cfg: str,
        device: str | None,
        amg_options: dict[str, Any],
        *,
        backend: str,
        efficientvit_repo: str | Path | None = None,
    ):
        self.backend = str(backend)
        self.ckpt = str(Path(ckpt).expanduser().resolve())
        self.cfg = str(cfg)
        self.efficientvit_repo = (
            None
            if efficientvit_repo is None
            else str(Path(efficientvit_repo).expanduser().resolve())
        )
        self.device_request = device
        self.amg_options = dict(amg_options)
        self.requests: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.startup_error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="sam2-first-frame", daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.ready.wait()
        if self.startup_error is not None:
            raise RuntimeError("resident segmentation worker startup failed") from self.startup_error

    def call(self, image_bgr: np.ndarray, options: dict[str, Any]):
        response: queue.Queue = queue.Queue(maxsize=1)
        self.requests.put(("select", np.array(image_bgr, copy=True), dict(options), response))
        ok, value = response.get()
        if ok:
            return value
        raise value

    def call_points(self, image_bgr: np.ndarray, points, labels, options):
        response: queue.Queue = queue.Queue(maxsize=1)
        self.requests.put((
            "select_points",
            np.array(image_bgr, copy=True),
            {"points": points, "labels": labels, **dict(options)},
            response,
        ))
        ok, value = response.get()
        if ok:
            return value
        raise value

    def call_restricted_amg(self, image_bgr: np.ndarray, allowed_mask, points, options):
        response: queue.Queue = queue.Queue(maxsize=1)
        self.requests.put((
            "restricted_amg",
            np.array(image_bgr, copy=True),
            {"allowed_mask": np.asarray(allowed_mask, dtype=np.uint8),
             "points": points, **dict(options)},
            response,
        ))
        ok, value = response.get()
        if ok:
            return value
        raise value

    def call_full_image_topn(self, image_bgr: np.ndarray, options: dict[str, Any]):
        """Run automatic mask generation on the complete image once."""
        response: queue.Queue = queue.Queue(maxsize=1)
        self.requests.put((
            "full_image_topn", np.array(image_bgr, copy=True), dict(options), response
        ))
        ok, value = response.get()
        if ok:
            return value
        raise value

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        response: queue.Queue = queue.Queue(maxsize=1)
        self.requests.put(("close", None, None, response))
        response.get()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            raise TimeoutError("SAM2 resident worker did not stop in time")

    def status(self) -> dict[str, Any]:
        return {
            "started": self.thread.is_alive(),
            "backend": self.backend,
            "checkpoint": self.ckpt,
            "config": self.cfg,
            "device": getattr(self, "device", self.device_request),
            "queued_requests": self.requests.qsize(),
        }

    def _run(self) -> None:
        try:
            import torch
            if self.device_request == "cuda" and _safe_torch_device(torch) != "cuda":
                raise RuntimeError(
                    "CUDA was explicitly requested, but PyTorch cannot initialize a CUDA device; "
                    "check the NVIDIA driver/PyTorch CUDA compatibility"
                )
            self.device = self.device_request or _safe_torch_device(torch)
            if self.backend == "sam2":
                _configure_sam2_from_checkpoint(self.ckpt)
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.build_sam import build_sam2

                model = build_sam2(self.cfg, self.ckpt, device=self.device)
                generator_options = {
                    "points_per_side": 16,
                    "points_per_batch": 128,
                    "pred_iou_thresh": 0.65,
                    "stability_score_thresh": 0.85,
                    "crop_n_layers": 0,
                    "min_mask_region_area": 300,
                    "output_mode": "binary_mask",
                    **self.amg_options,
                }
                generator = SAM2AutomaticMaskGenerator(model, **generator_options)
            else:
                repo = Path(self.efficientvit_repo or DEFAULT_EFFICIENTVIT_REPO)
                if not (repo / "efficientvit").is_dir():
                    raise FileNotFoundError(f"EfficientViT repository not found: {repo}")
                if not Path(self.ckpt).is_file():
                    raise FileNotFoundError(f"EfficientViT-SAM checkpoint not found: {self.ckpt}")
                repo_text = str(repo)
                if repo_text not in sys.path:
                    sys.path.insert(0, repo_text)
                from efficientvit.models.efficientvit.sam import (
                    EfficientViTSamAutomaticMaskGenerator,
                )
                from efficientvit.sam_model_zoo import create_efficientvit_sam_model

                model = create_efficientvit_sam_model(
                    self.cfg,
                    pretrained=True,
                    weight_url=self.ckpt,
                ).to(self.device).eval()
                # L0 uses a 512px encoder. A 16x16 no-crop grid keeps the
                # first-frame path fast while preserving the configuration
                # that selected the correct object in the 3.73 live test.
                generator_options = {
                    "points_per_side": 16,
                    "points_per_batch": 128,
                    "pred_iou_thresh": 0.75,
                    "stability_score_thresh": 0.90,
                    "crop_n_layers": 0,
                    "min_mask_region_area": 100,
                    "output_mode": "binary_mask",
                    **self.amg_options,
                }
                generator = EfficientViTSamAutomaticMaskGenerator(model, **generator_options)
            runtime = {
                "torch": torch,
                "device": self.device,
                "autocast_context": (
                    _safe_efficientvit_autocast_context
                    if self.backend == "efficientvit"
                    else _safe_autocast_context
                ),
            }
            # Model loading alone does not initialize the CUDA kernels used by
            # the first real frame.  Pay that one-time cost during start(), so
            # it is reported as preload rather than first-frame latency.
            if self.backend == "efficientvit" and self.device == "cuda":
                warmup_image = np.zeros((512, 512, 3), dtype=np.uint8)
                with torch.inference_mode(), _safe_efficientvit_autocast_context(
                    torch, self.device
                ):
                    generator.predictor.set_image(warmup_image)
                generator.predictor.reset_image()
                torch.cuda.synchronize()
            self.ready.set()
            while True:
                operation, image_bgr, options, response = self.requests.get()
                if operation == "close":
                    response.put((True, None))
                    return
                try:
                    if operation == "select_points":
                        result = _select_efficientvit_from_points_impl(
                            image_bgr,
                            predictor=generator.predictor,
                            runtime=runtime,
                            **options,
                        )
                    elif operation == "full_image_topn":
                        result = _segment_full_image_top_n_impl(
                            image_bgr,
                            generator=generator,
                            runtime=runtime,
                            backend=self.backend,
                            **options,
                        )
                    elif operation == "restricted_amg":
                        result = _select_efficientvit_restricted_amg_impl(
                            image_bgr,
                            generator=generator,
                            runtime=runtime,
                            **options,
                        )
                    else:
                        result = _legacy_select_largest_sam2_mask(
                            image_bgr,
                            ckpt=self.ckpt,
                            cfg=self.cfg,
                            backend=self.backend,
                            _resident_generator=generator,
                            _resident_runtime=runtime,
                            **options,
                        )
                    response.put((True, result))
                except BaseException as exc:
                    response.put((False, exc))
        except BaseException as exc:
            self.startup_error = exc
            self.ready.set()


_RESIDENT_LOCK = threading.RLock()
_RESIDENT_WORKER: _ResidentSegmentationWorker | None = None


def _resolve_runtime_config(ckpt, cfg: str | None) -> tuple[str, str]:
    ckpt = DEFAULT_SAM2_CKPT if ckpt is None else ckpt
    cfg = DEFAULT_SAM2_CFG if cfg is None else cfg
    return str(Path(ckpt).expanduser().resolve()), str(cfg)


def _normalize_backend(backend: str | None) -> str:
    name = "sam2" if backend is None else str(backend).strip().lower()
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; supported: {SUPPORTED_BACKENDS}")
    return name


def start(
    *,
    ckpt=None,
    cfg: str | None = None,
    device: str | None = None,
    amg_options: dict[str, Any] | None = None,
    backend: str = "sam2",
    efficientvit_repo: str | Path = DEFAULT_EFFICIENTVIT_REPO,
    efficientvit_model: str = DEFAULT_EFFICIENTVIT_MODEL,
    efficientvit_weights: str | Path = DEFAULT_EFFICIENTVIT_WEIGHTS,
) -> dict[str, Any]:
    """Load the selected segmentation backend; repeated calls are idempotent."""
    global _RESIDENT_WORKER
    backend = _normalize_backend(backend)
    if backend == "efficientvit":
        resolved_ckpt = str(Path(efficientvit_weights).expanduser().resolve())
        resolved_cfg = str(efficientvit_model)
        resolved_repo = str(Path(efficientvit_repo).expanduser().resolve())
    else:
        resolved_ckpt, resolved_cfg = _resolve_runtime_config(ckpt, cfg)
        resolved_repo = None
    with _RESIDENT_LOCK:
        if _RESIDENT_WORKER is not None:
            same_config = (
                _RESIDENT_WORKER.ckpt == resolved_ckpt
                and _RESIDENT_WORKER.cfg == resolved_cfg
                and _RESIDENT_WORKER.backend == backend
                and _RESIDENT_WORKER.efficientvit_repo == resolved_repo
                and _RESIDENT_WORKER.device_request == device
                and _RESIDENT_WORKER.amg_options == dict(amg_options or {})
            )
            if not same_config:
                raise RuntimeError(
                    "segmentation backend is already started with another configuration; "
                    "call close() before switching checkpoint/config/device"
                )
            return _RESIDENT_WORKER.status()
        worker = _ResidentSegmentationWorker(
            resolved_ckpt,
            resolved_cfg,
            device,
            dict(amg_options or {}),
            backend=backend,
            efficientvit_repo=resolved_repo,
        )
        try:
            worker.start()
        except BaseException:
            _RESIDENT_WORKER = None
            raise
        _RESIDENT_WORKER = worker
        return worker.status()


def close() -> None:
    """Stop the resident worker and release module-owned model references."""
    global _RESIDENT_WORKER
    with _RESIDENT_LOCK:
        worker = _RESIDENT_WORKER
        _RESIDENT_WORKER = None
    if worker is not None:
        worker.close()


def status() -> dict[str, Any]:
    with _RESIDENT_LOCK:
        if _RESIDENT_WORKER is None:
            return {"started": False}
        return _RESIDENT_WORKER.status()


def _legacy_select_efficientvit_from_points(
    image_bgr: np.ndarray,
    *,
    points: list[list[int]],
    labels: list[int],
    box: list[int] | tuple[int, int, int, int] | None = None,
    support_mask: np.ndarray | None = None,
    min_selection_margin: float = 0.08,
    return_formats: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Segment one object using points and optional RGB-D selection evidence.

    ``support_mask`` is only used to rank the three masks returned by one
    EfficientViT decode; it does not trigger another model inference.
    """
    if not points or len(points) != len(labels) or 1 not in labels:
        raise ValueError("points/labels must match and include a positive point")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got {image_bgr.shape}")
    with _RESIDENT_LOCK:
        if _RESIDENT_WORKER is None:
            start(backend="efficientvit")
        worker = _RESIDENT_WORKER
        if worker is None or worker.backend != "efficientvit":
            raise RuntimeError("EfficientViT backend is not resident; call close() first")
        return worker.call_points(image_bgr, points, labels, {
            "box": box,
            "support_mask": None if support_mask is None else np.asarray(support_mask, dtype=np.uint8),
            "min_selection_margin": float(min_selection_margin),
            "return_formats": return_formats,
        })


def _select_efficientvit_restricted_amg(
    image_bgr: np.ndarray,
    *,
    allowed_mask: np.ndarray,
    points: list[list[int]],
    depth_m: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    height_above_plane_m: np.ndarray | None = None,
    min_allowed_ratio: float = 0.70,
    boundary_min_allowed_ratio: float | None = None,
    support_boundary_margin_px: int = 16,
    normal_object_bbox_area_px: float | None = DEFAULT_NORMAL_OBJECT_MASK_AREA_PX,
    bbox_area_min_factor: float = DEFAULT_BBOX_AREA_MIN_FACTOR,
    bbox_area_max_factor: float = DEFAULT_BBOX_AREA_MAX_FACTOR,
    similar_bbox_area_ratio: float = DEFAULT_SIMILAR_BBOX_AREA_RATIO,
    bbox_area_reference_depth_m: float | None = DEFAULT_BBOX_AREA_REFERENCE_DEPTH_M,
    object_length_range_m: tuple[float, float] | None = DEFAULT_OBJECT_LENGTH_RANGE_M,
    object_width_range_m: tuple[float, float] | None = DEFAULT_OBJECT_WIDTH_RANGE_M,
    object_thickness_range_m: tuple[float, float] | None = DEFAULT_OBJECT_THICKNESS_RANGE_M,
    use_direct_predict: bool = True,
    refine_selected: bool = True,
    boundary_refine_all: bool = False,
    max_refine_candidates: int = 3,
    candidate_preselect_topk: int = 10,
    support_crop_padding_px: int = 24,
    target_center: list[int] | tuple[int, int] | None = None,
    return_formats: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run AMG only from points inside a projected non-background mask."""
    h, w = image_bgr.shape[:2]
    allowed = np.asarray(allowed_mask) > 0
    if allowed.shape != (h, w):
        raise ValueError(f"allowed_mask shape {allowed.shape} does not match image {(h, w)}")
    valid_points = [
        [int(x), int(y)] for x, y in points
        if 0 <= int(x) < w and 0 <= int(y) < h and allowed[int(y), int(x)]
    ]
    if not valid_points:
        raise ValueError("restricted AMG requires at least one point inside allowed_mask")
    with _RESIDENT_LOCK:
        if _RESIDENT_WORKER is None:
            start(backend="efficientvit")
        worker = _RESIDENT_WORKER
        if worker is None or worker.backend != "efficientvit":
            raise RuntimeError("EfficientViT backend is not resident; call close() first")
        return worker.call_restricted_amg(image_bgr, allowed, valid_points, {
            "depth_m": depth_m,
            "intrinsics": intrinsics,
            "height_above_plane_m": height_above_plane_m,
            "min_allowed_ratio": float(min_allowed_ratio),
            "boundary_min_allowed_ratio": boundary_min_allowed_ratio,
            "support_boundary_margin_px": int(support_boundary_margin_px),
            "normal_object_bbox_area_px": normal_object_bbox_area_px,
            "bbox_area_min_factor": float(bbox_area_min_factor),
            "bbox_area_max_factor": float(bbox_area_max_factor),
            "similar_bbox_area_ratio": float(similar_bbox_area_ratio),
            "bbox_area_reference_depth_m": bbox_area_reference_depth_m,
            "object_length_range_m": object_length_range_m,
            "object_width_range_m": object_width_range_m,
            "object_thickness_range_m": object_thickness_range_m,
            "use_direct_predict": bool(use_direct_predict),
            "refine_selected": bool(refine_selected),
            "boundary_refine_all": bool(boundary_refine_all),
            "max_refine_candidates": int(max_refine_candidates),
            "candidate_preselect_topk": int(candidate_preselect_topk),
            "support_crop_padding_px": int(support_crop_padding_px),
            "target_center": target_center,
            "return_formats": return_formats,
        })


def _project_points(points: np.ndarray, intrinsics: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape[:2]
    if len(points) == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    points, z = points[valid], z[valid]
    u = np.rint(intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]).astype(np.int32)
    v = np.rint(intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[inside], v[inside]


def _depth_height_above_plane(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    plane: list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    """Return the absolute metric point-to-plane distance for every depth pixel."""
    depth_m = np.asarray(depth_image, dtype=np.float64) * float(depth_scale)
    h, w = depth_m.shape
    K = np.asarray(intrinsics, dtype=np.float64)
    yy, xx = np.mgrid[:h, :w]
    x = (xx - K[0, 2]) * depth_m / K[0, 0]
    y = (yy - K[1, 2]) * depth_m / K[1, 1]
    a, b, c, d = [float(value) for value in plane]
    normal_length = max(1e-12, float(np.sqrt(a * a + b * b + c * c)))
    height = np.abs(a * x + b * y + c * depth_m + d) / normal_length
    height[~np.isfinite(depth_m) | (depth_m <= 1e-6)] = np.nan
    return height


def _robust_mask_3d_extents(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> list[float] | None:
    """Estimate pose-insensitive 3-D extents using PCA and robust percentiles."""
    valid = (
        (np.asarray(mask) > 0)
        & np.isfinite(depth_m)
        & (depth_m > 1e-6)
    )
    yy, xx = np.where(valid)
    if len(xx) < 100:
        return None
    z = depth_m[yy, xx]
    points = np.column_stack((
        (xx - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (yy - intrinsics[1, 2]) * z / intrinsics[1, 1],
        z,
    ))
    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    _, axes = np.linalg.eigh(covariance)
    projected = centered @ axes
    extents = np.percentile(projected, 97.5, axis=0) - np.percentile(
        projected, 2.5, axis=0
    )
    return [float(value) for value in np.sort(extents)[::-1]]


def _coarse_mask_3d_length(
    coarse_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    stride: int,
) -> float | None:
    """Cheap PCA length estimate used only to choose full-resolution top-k."""
    coarse_y, coarse_x = np.where(np.asarray(coarse_mask) > 0)
    yy = np.clip(coarse_y * int(stride), 0, depth_m.shape[0] - 1)
    xx = np.clip(coarse_x * int(stride), 0, depth_m.shape[1] - 1)
    z = depth_m[yy, xx]
    valid = np.isfinite(z) & (z > 1e-6)
    if int(np.count_nonzero(valid)) < 25:
        return None
    xx, yy, z = xx[valid], yy[valid], z[valid]
    points = np.column_stack((
        (xx - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (yy - intrinsics[1, 2]) * z / intrinsics[1, 1],
        z,
    ))
    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    _, axes = np.linalg.eigh(covariance)
    projected = centered @ axes
    extents = np.percentile(projected, 97.5, axis=0) - np.percentile(
        projected, 2.5, axis=0
    )
    return float(np.max(extents))


def _fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the plane without paying multi-thread BLAS startup per hypothesis."""
    try:
        from threadpoolctl import threadpool_limits
        blas_context = threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        blas_context = nullcontext()
    with blas_context:
        return _fit_plane_ransac_impl(
            points,
            distance_threshold=distance_threshold,
            iterations=iterations,
            seed=seed,
        )


def _fit_plane_ransac_impl(
    points: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane with deterministic NumPy RANSAC and PCA refinement."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"plane fitting requires Nx3 points, got {points.shape}")
    threshold = float(distance_threshold)
    if threshold <= 0:
        raise ValueError(f"distance_threshold must be positive, got {threshold}")

    rng = np.random.default_rng(int(seed))
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_error = float("inf")
    for _ in range(max(1, int(iterations))):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-12:
            continue
        normal /= normal_length
        offset = -float(normal @ sample[0])
        distances = np.abs(points @ normal + offset)
        inliers = np.flatnonzero(distances <= threshold)
        count = int(len(inliers))
        error = float(np.mean(distances[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_inliers = inliers
            best_count = count
            best_error = error

    if best_inliers is None or len(best_inliers) < 3:
        raise RuntimeError("RANSAC could not find a valid background plane")

    # Refine the winning hypothesis with total least squares, then recompute
    # the inlier set against the refined plane.
    support = points[best_inliers]
    center = np.mean(support, axis=0)
    covariance = (support - center).T @ (support - center)
    _, axes = np.linalg.eigh(covariance)
    normal = axes[:, 0]
    normal /= max(1e-12, float(np.linalg.norm(normal)))
    offset = -float(normal @ center)
    plane = np.concatenate((normal, [offset]))
    distances = np.abs(points @ normal + offset)
    inliers = np.flatnonzero(distances <= threshold)
    return plane, inliers


def _voxel_downsample_count(points: np.ndarray, voxel_size: float) -> int:
    """Return the occupied voxel count without constructing an Open3D cloud."""
    points = np.asarray(points, dtype=np.float64)
    if not len(points) or float(voxel_size) <= 0:
        return int(len(points))
    voxel_indices = np.floor(points / float(voxel_size)).astype(np.int64)
    return int(len(np.unique(voxel_indices, axis=0)))


def _build_nonplane_amg_prompts(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    image_bgr: np.ndarray | None,
    roi: tuple[float, float, float, float],
    min_depth: float,
    max_depth: float,
    pixel_stride: int,
    plane_distance: float,
    min_object_height_above_plane_m: float,
    candidate_object_height_above_plane_m: float,
    ransac_iterations: int,
    voxel_size: float,
    target_center: tuple[int, int] | None,
    target_window: tuple[int, int] | None,
    nonplane_dilation_px: int,
    min_object_table_contrast: float,
    target_prior_sigma: float,
    min_interior_distance: float,
    max_amg_points: int,
    amg_grid_points_per_side: int,
    max_prompt_distance_px: float | None,
    amg_point_separation: float,
    amg_peak_radius: int,
    retain_all_significant_components: bool = False,
    min_prompt_component_area_px: int = MIN_PROMPT_COMPONENT_AREA_PX,
    min_prompt_component_largest_ratio: float = MIN_PROMPT_COMPONENT_LARGEST_RATIO,
) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    """Build a projected non-background mask and its sparse AMG point grid."""
    import cv2

    started = time.perf_counter()
    depth = np.asarray(depth_image)
    if depth.ndim != 2:
        raise ValueError(f"depth_image must be HxW, got {depth.shape}")
    h, w = depth.shape
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {K.shape}")
    x0, y0, x1, y1 = [float(v) for v in roi]
    ix0, iy0, ix1, iy1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    vv, uu = np.mgrid[0:h:max(1, int(pixel_stride)), 0:w:max(1, int(pixel_stride))]
    metric_depth = depth[vv, uu].astype(np.float64) * float(depth_scale)
    valid = (
        (uu >= ix0) & (uu < ix1) & (vv >= iy0) & (vv < iy1)
        & np.isfinite(metric_depth) & (metric_depth >= min_depth) & (metric_depth <= max_depth)
    )
    u, v, z = uu[valid], vv[valid], metric_depth[valid]
    points = np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
    if len(points) < 3:
        raise RuntimeError("not enough valid depth points for background removal")
    plane, inliers = _fit_plane_ransac(
        points,
        distance_threshold=float(plane_distance),
        iterations=int(ransac_iterations),
    )
    # Orient the plane normal toward increasing camera Z. Objects protruding
    # toward the camera then have negative signed distance. RANSAC non-inliers
    # alone are insufficient: cardboard warping and reflective tape also
    # produce shallow/non-physical non-inliers on either side of the plane.
    plane = np.asarray(plane, dtype=np.float64)
    if plane[2] < 0:
        plane = -plane
    normal_length = max(1e-12, float(np.linalg.norm(plane[:3])))
    signed_distance = (points @ plane[:3] + plane[3]) / normal_length
    foreground_height = -signed_distance
    reliable_foreground_indices = np.flatnonzero(
        foreground_height >= float(min_object_height_above_plane_m)
    )
    candidate_height = min(
        float(min_object_height_above_plane_m),
        float(candidate_object_height_above_plane_m),
    )
    candidate_foreground_indices = np.flatnonzero(foreground_height >= candidate_height)
    # Keep the original sampled non-plane points for the 2-D support mask.
    # Projecting only voxel centroids can punch holes in thin/small objects;
    # voxel downsampling is useful for diagnostics but not for this projection.
    object_points = points[candidate_foreground_indices]
    downsampled_point_count = _voxel_downsample_count(object_points, float(voxel_size))
    projected_u, projected_v = _project_points(object_points, K, (h, w))
    raw_projection = np.zeros((h, w), np.uint8)
    raw_projection[projected_v, projected_u] = 255
    reliable_points = points[reliable_foreground_indices]
    reliable_u, reliable_v = _project_points(reliable_points, K, (h, w))
    reliable_raw_projection = np.zeros((h, w), np.uint8)
    reliable_raw_projection[reliable_v, reliable_u] = 255
    dilation_px = max(1, int(nonplane_dilation_px))
    if dilation_px % 2 == 0:
        dilation_px += 1
    # The expanded mask is useful as a permissive candidate-overlap gate, but
    # prompts should stay closer to pixels with measured foreground depth.
    # Otherwise a large dilation can put prompt points on bare table.
    # Proposal overlap may use a generous 13 px support expansion, but prompt
    # points must stay close to measured non-plane pixels.  A 3 px kernel only
    # reconnects stride-sampled depth pixels; the later distance transform then
    # pulls prompts into the physical object instead of onto the table.
    prompt_dilation_px = min(dilation_px, 3)
    prompt_projection = cv2.dilate(
        raw_projection,
        np.ones((prompt_dilation_px, prompt_dilation_px), np.uint8),
        iterations=1,
    )
    all_projection = cv2.dilate(
        raw_projection,
        np.ones((dilation_px, dilation_px), np.uint8),
        iterations=1,
    )
    reliable_projection = cv2.dilate(
        reliable_raw_projection,
        np.ones((dilation_px, dilation_px), np.uint8),
        iterations=1,
    )
    # A 3--6 mm region is accepted only when its complete 2-D component
    # touches reliable >=6 mm support.  This recovers upside-down, low-profile
    # workpieces without admitting isolated table noise.
    _candidate_count, candidate_labels = cv2.connectedComponents(
        (all_projection > 0).astype(np.uint8), connectivity=8
    )
    touching_ids = np.unique(candidate_labels[reliable_projection > 0])
    touching_ids = touching_ids[touching_ids > 0]
    all_projection = np.where(
        np.isin(candidate_labels, touching_ids), all_projection, 0
    ).astype(np.uint8)
    prompt_projection = np.where(all_projection > 0, prompt_projection, 0).astype(np.uint8)

    table_gray_median: float | None = None
    appearance_threshold: float | None = None
    if image_bgr is not None and float(min_object_table_contrast) > 0:
        color = np.asarray(image_bgr)
        if color.shape[:2] != (h, w) or color.ndim != 3 or color.shape[2] != 3:
            raise ValueError(f"image_bgr must be HxWx3 matching depth, got {color.shape}")
        gray = cv2.cvtColor(color.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
        table_samples = gray[v[inliers], u[inliers]]
        table_gray_median = float(np.median(table_samples))
        appearance_threshold = table_gray_median - float(min_object_table_contrast)
        darker_than_table = (gray.astype(np.float32) <= appearance_threshold).astype(np.uint8)
        darker_than_table = cv2.morphologyEx(
            darker_than_table,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        # Dark edges, holes and textured patches identify each metal part. A
        # moderate dilation reconnects its brighter reflective surfaces while
        # leaving broad table regions excluded.
        appearance_gate_px = max(9, dilation_px + 4)
        if appearance_gate_px % 2 == 0:
            appearance_gate_px += 1
        appearance_gate = cv2.dilate(
            darker_than_table,
            np.ones((appearance_gate_px, appearance_gate_px), np.uint8),
            iterations=1,
        )
        all_projection = np.where(appearance_gate > 0, all_projection, 0).astype(np.uint8)
        prompt_projection = np.where(
            appearance_gate > 0, prompt_projection, 0
        ).astype(np.uint8)

    target_requested = target_center is not None
    anchor_x, anchor_y = (
        [int(v) for v in target_center]
        if target_requested else [w // 2, h // 2]
    )
    if target_window is None:
        # Use the complete configured ROI for plane removal. A target-aware
        # component filter below removes unrelated non-plane structures.
        wx0, wy0, wx1, wy1 = ix0, iy0, ix1, iy1
        allowed = all_projection.copy()
        allowed[:iy0] = 0
        allowed[iy1:] = 0
        allowed[:, :ix0] = 0
        allowed[:, ix1:] = 0
    else:
        if not target_requested:
            raise ValueError("target_window requires target_center")
        window_w, window_h = [int(v) for v in target_window]
        if window_w <= 0 or window_h <= 0:
            raise ValueError(f"target_window must contain positive sizes, got {target_window}")
        wx0, wy0 = max(ix0, anchor_x - window_w // 2), max(iy0, anchor_y - window_h // 2)
        wx1, wy1 = min(ix1, anchor_x + window_w // 2), min(iy1, anchor_y + window_h // 2)
        allowed = np.zeros_like(all_projection)
        allowed[wy0:wy1, wx0:wx1] = all_projection[wy0:wy1, wx0:wx1]
    if not np.any(allowed):
        region_name = "depth ROI" if target_window is None else "target window"
        raise RuntimeError(f"no non-background depth support inside {region_name}")

    # Sparse RealSense outliers can survive the height threshold as tiny
    # disconnected islands. Remove them before both prompt selection and mask
    # scoring; otherwise each island may receive an AMG point on bare table.
    raw_component_count, raw_component_labels, raw_component_stats, _ = (
        cv2.connectedComponentsWithStats((allowed > 0).astype(np.uint8), connectivity=8)
    )
    raw_component_areas = raw_component_stats[1:, cv2.CC_STAT_AREA]
    largest_component_area = int(raw_component_areas.max())
    min_component_area = max(
        int(min_prompt_component_area_px),
        int(np.ceil(largest_component_area * float(min_prompt_component_largest_ratio))),
    )
    kept_component_ids = [
        component_id
        for component_id in range(1, raw_component_count)
        if int(raw_component_stats[component_id, cv2.CC_STAT_AREA]) >= min_component_area
    ]

    # A global ROI is useful for estimating the table plane, but it should not
    # turn every unrelated non-plane structure in the image into an AMG target.
    # With a target anchor retain its nearest significant island; targetless
    # Max mode can instead retain every significant workpiece pile.
    component_distances_px: dict[int, float] = {}
    selected_target_component: int | None = None
    if kept_component_ids:
        for component_id in kept_component_ids:
            component_y, component_x = np.where(raw_component_labels == component_id)
            squared_distance = (
                (component_x.astype(np.float64) - anchor_x) ** 2
                + (component_y.astype(np.float64) - anchor_y) ** 2
            )
            component_distances_px[component_id] = float(np.sqrt(squared_distance.min()))
        if target_requested:
            selected_target_component = min(
                kept_component_ids,
                key=lambda component_id: (
                    component_distances_px[component_id],
                    -int(raw_component_stats[component_id, cv2.CC_STAT_AREA]),
                ),
            )
            kept_component_ids = [selected_target_component]
        elif not retain_all_significant_components:
            selected_target_component = max(
                kept_component_ids,
                key=lambda component_id: int(raw_component_stats[component_id, cv2.CC_STAT_AREA]),
            )
            kept_component_ids = [selected_target_component]
    allowed = np.where(
        np.isin(raw_component_labels, kept_component_ids), allowed, 0
    ).astype(np.uint8)
    if not np.any(allowed):
        raise RuntimeError(
            "non-background depth support contains only small disconnected noise"
        )
    if not target_requested:
        component_y, component_x = np.where(allowed > 0)
        # This is used only to start spatially balanced prompt sampling; it is
        # derived from the support itself and never affects candidate ranking.
        anchor_x, anchor_y = int(np.median(component_x)), int(np.median(component_y))

    prompt_support = np.where(allowed > 0, prompt_projection, 0).astype(np.uint8)
    if not np.any(prompt_support):
        raise RuntimeError("selected non-background region has no measured depth support")
    interior = cv2.distanceTransform(
        (prompt_support > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    yy, xx = np.mgrid[:h, :w]
    center_distance = np.hypot(xx - anchor_x, yy - anchor_y)
    prompt_distance_ok = np.ones((h, w), dtype=bool)
    if max_prompt_distance_px is not None:
        prompt_distance_ok = center_distance <= max(1.0, float(max_prompt_distance_px))
    # Match the original full-image AMG semantics: divide the complete image
    # into a regular NxN lattice and emit at most one point per cell. A cell
    # point is snapped to the nearest reliable workpiece-support pixel inside
    # that same cell, so the grid stays spatially balanced without prompting on
    # bare table.
    grid_side = max(1, int(amg_grid_points_per_side))
    valid_prompt_support = (
        (interior >= float(min_interior_distance))
        & (prompt_support > 0)
        & prompt_distance_ok
    )
    prompt_candidates: list[tuple[int, int]] = []
    for grid_y in range(grid_side):
        cell_y0 = int(np.floor(grid_y * h / grid_side))
        cell_y1 = int(np.floor((grid_y + 1) * h / grid_side))
        center_y = 0.5 * (cell_y0 + cell_y1 - 1)
        for grid_x in range(grid_side):
            cell_x0 = int(np.floor(grid_x * w / grid_side))
            cell_x1 = int(np.floor((grid_x + 1) * w / grid_side))
            cell = valid_prompt_support[cell_y0:cell_y1, cell_x0:cell_x1]
            local_y, local_x = np.where(cell)
            if not len(local_x):
                continue
            global_x = local_x + cell_x0
            global_y = local_y + cell_y0
            center_x = 0.5 * (cell_x0 + cell_x1 - 1)
            distance_sq = (global_x - center_x) ** 2 + (global_y - center_y) ** 2
            near = np.flatnonzero(distance_sq <= float(distance_sq.min()) + 12.0 ** 2)
            best_local = int(max(
                near,
                key=lambda item: (
                    float(interior[global_y[item], global_x[item]]),
                    -float(distance_sq[item]),
                ),
            ))
            prompt_candidates.append(
                (int(global_x[best_local]), int(global_y[best_local]))
            )

    # The full-image 8x8 lattice has ~90 px row spacing at 720p.  Add a local
    # lattice inside the accepted support so low workpieces can receive points
    # between those rows.  Farthest-point selection below still enforces the
    # global max_amg_points limit.
    support_y, support_x = np.where(valid_prompt_support)
    if len(support_x):
        local_spacing = 55
        local_x0, local_x1 = int(support_x.min()), int(support_x.max()) + 1
        local_y0, local_y1 = int(support_y.min()), int(support_y.max()) + 1
        for center_y in range(local_y0 + local_spacing // 2, local_y1, local_spacing):
            for center_x in range(local_x0 + local_spacing // 2, local_x1, local_spacing):
                radius = local_spacing // 2
                x0, x1 = max(local_x0, center_x - radius), min(local_x1, center_x + radius + 1)
                y0, y1 = max(local_y0, center_y - radius), min(local_y1, center_y + radius + 1)
                local_y, local_x = np.where(valid_prompt_support[y0:y1, x0:x1])
                if not len(local_x):
                    continue
                global_x, global_y = local_x + x0, local_y + y0
                distance_sq = (global_x - center_x) ** 2 + (global_y - center_y) ** 2
                near = np.flatnonzero(
                    distance_sq <= float(distance_sq.min()) + 12.0 ** 2
                )
                best = int(max(
                    near,
                    key=lambda item: (
                        float(interior[global_y[item], global_x[item]]),
                        -float(distance_sq[item]),
                    ),
                ))
                prompt_candidates.append((int(global_x[best]), int(global_y[best])))
    prompt_candidates = list(dict.fromkeys(prompt_candidates))

    # Give each retained workpiece pile one interior positive point before
    # distributing the remaining global prompt budget.
    prompts: list[list[int]] = []
    if retain_all_significant_components and not target_requested:
        for component_id in sorted(
            kept_component_ids,
            key=lambda item: int(raw_component_stats[item, cv2.CC_STAT_AREA]),
            reverse=True,
        ):
            if len(prompts) >= int(max_amg_points):
                break
            component_valid = valid_prompt_support & (raw_component_labels == component_id)
            component_y, component_x = np.where(component_valid)
            if not len(component_x):
                continue
            best = int(np.argmax(interior[component_y, component_x]))
            prompts.append([int(component_x[best]), int(component_y[best])])
    remaining = set(prompt_candidates)
    for point in prompts:
        remaining.discard((point[0], point[1]))
    if remaining and not prompts:
        first = min(
            remaining,
            key=lambda point: (
                center_distance[point[1], point[0]], point[1], point[0]
            ),
        )
        prompts.append([first[0], first[1]])
        remaining.remove(first)
    separation_sq = float(amg_point_separation) ** 2
    while remaining and len(prompts) < int(max_amg_points):
        selected_xy = np.asarray(prompts, dtype=np.float64)

        def coverage_key(point: tuple[int, int]) -> tuple[float, float, float]:
            delta = selected_xy - np.asarray(point, dtype=np.float64)
            nearest_sq = float(np.min(np.sum(delta * delta, axis=1)))
            return (
                nearest_sq,
                float(interior[point[1], point[0]]),
                -float(center_distance[point[1], point[0]]),
            )

        next_point = max(remaining, key=coverage_key)
        if coverage_key(next_point)[0] < separation_sq:
            break
        prompts.append([next_point[0], next_point[1]])
        remaining.remove(next_point)
    if not prompts:
        raise RuntimeError("no reliable interior AMG point in non-background region")
    ys, xs = np.where(allowed > 0)
    debug = {
        "plane": [float(v) for v in plane], "input_points": int(len(points)),
        "plane_inliers": int(len(inliers)), "non_plane_points": int(len(object_points)),
        "min_object_height_above_plane_m": float(min_object_height_above_plane_m),
        "candidate_object_height_above_plane_m": float(candidate_height),
        "reliable_non_plane_points": int(len(reliable_points)),
        "candidate_non_plane_points": int(len(object_points)),
        "nonplane_dilation_px": dilation_px,
        "prompt_support_dilation_px": prompt_dilation_px,
        "table_gray_median": table_gray_median,
        "min_object_table_contrast": float(min_object_table_contrast),
        "appearance_gray_threshold": appearance_threshold,
        "max_prompt_distance_px": (
            None if max_prompt_distance_px is None else float(max_prompt_distance_px)
        ),
        "downsampled_non_plane_points": downsampled_point_count,
        "raw_support_components": int(raw_component_count - 1),
        "kept_support_components": int(len(kept_component_ids)),
        "retained_component_ids": [int(component_id) for component_id in kept_component_ids],
        "retain_all_significant_components": bool(retain_all_significant_components),
        "min_support_component_area_px": int(min_component_area),
        "selected_target_component": selected_target_component,
        "selected_target_component_distance_px": (
            None if selected_target_component is None
            else component_distances_px[selected_target_component]
        ),
        "allowed_area_px": int(len(xs)),
        "allowed_bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
        "target_center": None if not target_requested else [anchor_x, anchor_y],
        "target_window": None if target_window is None else [wx0, wy0, wx1, wy1],
        "support_region": [wx0, wy0, wx1, wy1],
        "prompt_sampling": "full_image_regular_amg_grid_snapped_to_support",
        "amg_grid_points_per_side": grid_side,
        "amg_points": int(len(prompts)), "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    return allowed, prompts, debug


def select_first_frame_mask(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    roi: tuple[float, float, float, float] = (0.20, 0.10, 0.88, 0.72),
    min_depth: float = 0.15,
    max_depth: float = 1.0,
    pixel_stride: int = 2,
    plane_distance: float = 0.006,
    min_object_height_above_plane_m: float = 0.006,
    candidate_object_height_above_plane_m: float = 0.003,
    ransac_iterations: int = 120,
    voxel_size: float = 0.005,
    target_center: tuple[int, int] | None = (230, 228),
    target_window: tuple[int, int] | None = None,
    nonplane_dilation_px: int = 7,
    min_object_table_contrast: float = 0.0,
    target_prior_sigma: float = 55.0,
    min_interior_distance: float = 8.0,
    max_amg_points: int = 16,
    amg_grid_points_per_side: int = 8,
    max_prompt_distance_px: float | None = None,
    amg_point_separation: float = 16.0,
    amg_peak_radius: int = 6,
    retain_all_significant_components: bool = False,
    min_prompt_component_area_px: int = MIN_PROMPT_COMPONENT_AREA_PX,
    min_prompt_component_largest_ratio: float = MIN_PROMPT_COMPONENT_LARGEST_RATIO,
    min_allowed_ratio: float = 0.30,
    boundary_min_allowed_ratio: float | None = None,
    support_boundary_margin_px: int = 16,
    normal_object_bbox_area_px: float | None = DEFAULT_NORMAL_OBJECT_MASK_AREA_PX,
    bbox_area_min_factor: float = DEFAULT_BBOX_AREA_MIN_FACTOR,
    bbox_area_max_factor: float = DEFAULT_BBOX_AREA_MAX_FACTOR,
    similar_bbox_area_ratio: float = DEFAULT_SIMILAR_BBOX_AREA_RATIO,
    bbox_area_reference_depth_m: float | None = DEFAULT_BBOX_AREA_REFERENCE_DEPTH_M,
    object_length_range_m: tuple[float, float] | None = DEFAULT_OBJECT_LENGTH_RANGE_M,
    object_width_range_m: tuple[float, float] | None = DEFAULT_OBJECT_WIDTH_RANGE_M,
    object_thickness_range_m: tuple[float, float] | None = DEFAULT_OBJECT_THICKNESS_RANGE_M,
    max_single_object_bbox_area_px: float | None = None,
    use_direct_predict: bool = True,
    refine_selected: bool = True,
    boundary_refine_all: bool = False,
    max_refine_candidates: int = 3,
    candidate_preselect_topk: int = 10,
    support_crop_padding_px: int = 24,
    return_formats: str | list[str] | tuple[str, ...] | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Select the first-frame mask directly from aligned RGB-D input.

    ``target_window=None`` fits the plane over the complete ``roi`` and keeps
    the significant projected non-plane component nearest ``target_center``.
    A ``(width, height)`` window limits proposal support without reducing the
    region used for plane fitting; the nearest significant component is kept.

    ``normal_object_bbox_area_px`` is the expected bounding-box area of one
    workpiece in this fixed observation view.  The box is measured on each
    proposal's largest connected foreground body, so detached speckles cannot
    stretch its box. Only box areas in ``bbox_area_min_factor`` through
    ``bbox_area_max_factor`` are retained.
    ``max_single_object_bbox_area_px`` remains as a compatibility alias;
    ``bbox_area_reference_depth_m`` is kept only so older callers do not break
    and no longer changes the area limits.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
    if float(min_object_height_above_plane_m) <= 0:
        raise ValueError("min_object_height_above_plane_m must be positive")
    if float(candidate_object_height_above_plane_m) <= 0:
        raise ValueError("candidate_object_height_above_plane_m must be positive")
    if tuple(depth_image.shape[:2]) != tuple(image_bgr.shape[:2]):
        raise ValueError("depth_image must already be aligned to image_bgr")
    allowed, prompts, pointcloud_debug = _build_nonplane_amg_prompts(
        depth_image, intrinsics, depth_scale, image_bgr=image_bgr,
        roi=roi, min_depth=min_depth,
        max_depth=max_depth, pixel_stride=pixel_stride, plane_distance=plane_distance,
        min_object_height_above_plane_m=min_object_height_above_plane_m,
        candidate_object_height_above_plane_m=candidate_object_height_above_plane_m,
        ransac_iterations=ransac_iterations, voxel_size=voxel_size,
        target_center=target_center, target_window=target_window,
        nonplane_dilation_px=nonplane_dilation_px,
        min_object_table_contrast=min_object_table_contrast,
        target_prior_sigma=target_prior_sigma, min_interior_distance=min_interior_distance,
        max_amg_points=max_amg_points,
        amg_grid_points_per_side=amg_grid_points_per_side,
        max_prompt_distance_px=max_prompt_distance_px,
        amg_point_separation=amg_point_separation,
        amg_peak_radius=amg_peak_radius,
        retain_all_significant_components=retain_all_significant_components,
        min_prompt_component_area_px=min_prompt_component_area_px,
        min_prompt_component_largest_ratio=min_prompt_component_largest_ratio,
    )
    result = _select_efficientvit_restricted_amg(
        image_bgr, allowed_mask=allowed, points=prompts,
        depth_m=np.asarray(depth_image, dtype=np.float64) * float(depth_scale),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        height_above_plane_m=_depth_height_above_plane(
            depth_image, intrinsics, depth_scale, pointcloud_debug["plane"]
        ),
        min_allowed_ratio=min_allowed_ratio, target_center=target_center,
        boundary_min_allowed_ratio=boundary_min_allowed_ratio,
        support_boundary_margin_px=support_boundary_margin_px,
        normal_object_bbox_area_px=(
            normal_object_bbox_area_px
            if max_single_object_bbox_area_px is None
            else max_single_object_bbox_area_px
        ),
        bbox_area_min_factor=bbox_area_min_factor,
        bbox_area_max_factor=bbox_area_max_factor,
        similar_bbox_area_ratio=similar_bbox_area_ratio,
        bbox_area_reference_depth_m=bbox_area_reference_depth_m,
        object_length_range_m=object_length_range_m,
        object_width_range_m=object_width_range_m,
        object_thickness_range_m=object_thickness_range_m,
        use_direct_predict=use_direct_predict,
        refine_selected=refine_selected,
        boundary_refine_all=boundary_refine_all,
        max_refine_candidates=max_refine_candidates,
        candidate_preselect_topk=candidate_preselect_topk,
        support_crop_padding_px=support_crop_padding_px,
        return_formats=return_formats,
    )
    result["preprocess_timing_ms"] = float(pointcloud_debug["elapsed_ms"])
    if return_debug:
        result["debug"] = {
            "pointcloud": pointcloud_debug,
            "allowed_mask": allowed,
            "amg_points": prompts,
        }
    return result


def select_first_frame_candidates(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    num_candidates: int,
    **kwargs,
) -> dict[str, Any]:
    """Return a fixed number of ranked candidate slots.

    This uses the same single inference as :func:`select_first_frame_mask`, but
    exposes the first ``num_candidates`` ranked proposals. If fewer valid
    proposals exist, the returned ``candidates`` list is padded with ``None``.
    ``actual_count`` reports the number of non-empty slots and
    ``available_count`` reports all candidates that passed the area filter.
    """
    requested_count = int(num_candidates)
    if requested_count < 1:
        raise ValueError("num_candidates must be at least 1")
    options = dict(kwargs)
    options.setdefault("refine_selected", False)
    result = select_first_frame_mask(
        image_bgr, depth_image, intrinsics, depth_scale, **options
    )
    available = list(result["candidates"])
    actual_count = min(requested_count, len(available))
    slots: list[dict[str, Any] | None] = available[:actual_count]
    slots.extend([None] * (requested_count - actual_count))
    return {
        "candidates": slots,
        "requested_count": requested_count,
        "actual_count": actual_count,
        "available_count": len(available),
        "selection_rule": result["selection_rule"],
        "timing_ms": result.get("timing_ms"),
        "preprocess_timing_ms": result.get("preprocess_timing_ms"),
    }


def select_first_frame_candicates_max(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    num_candidates: int,
    **kwargs,
) -> dict[str, Any]:
    """Higher-coverage RGB-D candidate API with a 3-mm support threshold.

    The spelling intentionally matches the public ``/candicates-max`` endpoint.
    It otherwise reuses the standard first-frame candidate implementation.
    """
    options = dict(kwargs)
    # These are endpoint invariants, rather than tunables hidden in client
    # options: retain more low-profile connected workpiece support and sample
    # substantially denser positive AMG prompts.
    options.update({
        "candidate_object_height_above_plane_m": 0.003,
        "max_amg_points": 32,
        "amg_grid_points_per_side": 12,
        "amg_point_separation": 12.0,
        "retain_all_significant_components": True,
        # Reject isolated depth protrusions that are visually table-coloured.
        "min_object_table_contrast": 18.0,
        "candidate_preselect_topk": 32,
        # Only masks whose box reaches the RGB-D support boundary may use the
        # relaxed ratio; interior masks keep the standard 30% requirement.
        "boundary_min_allowed_ratio": 0.20,
        "support_boundary_margin_px": 16,
        "refine_selected": True,
        "boundary_refine_all": True,
        "max_refine_candidates": 10,
        # Keep RGB context around incomplete depth support so a prompt on an
        # upper workpiece can recover its full visible extent.
        "support_crop_padding_px": 100,
        "target_center": None,
        "target_window": None,
        # Retain the upper workpiece whose measured exposed length is about
        # 61 mm in this view; the production default remains unchanged.
        "object_length_range_m": (0.060, 0.095),
        # Constrain the largest connected body's bounding box, not its filled
        # foreground area.  Permit boxes up to 24,000 px in this fixed view.
        "bbox_area_max_factor": 2.40,
    })
    result = select_first_frame_candidates(
        image_bgr, depth_image, intrinsics, depth_scale,
        num_candidates=num_candidates, **options,
    )
    result["endpoint_config"] = {
        "candidate_object_height_above_plane_m": 0.003,
        "max_amg_points": 32,
        "amg_grid_points_per_side": 12,
        "amg_point_separation_px": 12.0,
        "retain_all_significant_components": True,
        "min_object_table_contrast": 18.0,
        "candidate_preselect_topk": 32,
        "boundary_min_allowed_ratio": 0.20,
        "support_boundary_margin_px": 16,
        "boundary_refine_all": True,
        "max_refine_candidates": 10,
        "support_crop_padding_px": 100,
        "target_center": None,
        "object_length_range_m": [0.060, 0.095],
        "bbox_area_max_factor": 2.40,
    }
    return result


def select_first_frame_candidate_components(
    image_bgr: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    *,
    num_candidates: int,
    **kwargs,
) -> dict[str, Any]:
    """Max-style RGB-D candidates without lower-size component filtering."""
    options = dict(kwargs)
    options.update({
        "candidate_object_height_above_plane_m": 0.003,
        "max_amg_points": 32,
        "amg_grid_points_per_side": 12,
        "amg_point_separation": 12.0,
        "retain_all_significant_components": True,
        # Retain every non-plane connected component, including small exposed
        # workpiece fragments.  Upper limits remain to reject merged piles.
        "min_prompt_component_area_px": 1,
        "min_prompt_component_largest_ratio": 0.0,
        "min_object_table_contrast": 18.0,
        "candidate_preselect_topk": 32,
        "boundary_min_allowed_ratio": 0.20,
        "support_boundary_margin_px": 16,
        "refine_selected": True,
        "boundary_refine_all": True,
        "max_refine_candidates": 10,
        "support_crop_padding_px": 100,
        "target_center": None,
        "target_window": None,
        # Disable only lower size bounds.  Keep the established maxima so a
        # connected pile cannot be emitted as one workpiece.
        "bbox_area_min_factor": 1e-6,
        "bbox_area_max_factor": 2.40,
        "object_length_range_m": (0.0, 0.095),
        "object_width_range_m": (0.0, 0.045),
        "object_thickness_range_m": (0.0, 0.020),
    })
    result = select_first_frame_candidates(
        image_bgr, depth_image, intrinsics, depth_scale,
        num_candidates=num_candidates, **options,
    )
    candidates, fully_contained_removed = _remove_fully_contained_mask_candidates(
        [candidate for candidate in result.get("candidates", ()) if candidate is not None]
    )
    if candidates:
        selected = candidates[0]
        selected_mask = np.asarray(selected["mask"], dtype=np.uint8) * 255
        result["candidates"] = candidates
        result["candidate_scores"] = [
            {key: value for key, value in candidate.items() if key != "mask"}
            for candidate in candidates
        ]
        result["valid_candidates"] = len(candidates)
        result["selected_source_index"] = int(selected["source_index"])
        result["score"] = float(selected["score"])
        result["mask"] = selected_mask
        result["bbox"] = _bbox_from_mask(selected_mask)
        result["formats"] = {**result.get("formats", {}), "mask": selected_mask}
    result["fully_contained_candidates_removed"] = fully_contained_removed
    result["endpoint_config"] = {
        "base_endpoint": "/v1/first-frame/candicates-max",
        "candidate_object_height_above_plane_m": 0.003,
        "retain_all_significant_components": True,
        "min_prompt_component_area_px": 1,
        "min_prompt_component_largest_ratio": 0.0,
        "bbox_area_min_factor": 1e-6,
        "bbox_area_max_factor": 2.40,
        "object_length_range_m": [0.0, 0.095],
        "object_width_range_m": [0.0, 0.045],
        "object_thickness_range_m": [0.0, 0.020],
        "remove_fully_contained_masks": True,
    }
    return result


def segment_full_image_top_n(
    image_bgr: np.ndarray,
    *,
    num_masks: int,
    roi: tuple[int, int, int, int] | None = None,
    min_mask_area_px: int = 1,
    return_formats: str | list[str] | tuple[str, ...] | None = None,
    backend: str | None = None,
    ckpt=None,
    cfg: str | None = None,
    efficientvit_repo: str | Path = DEFAULT_EFFICIENTVIT_REPO,
    efficientvit_model: str = DEFAULT_EFFICIENTVIT_MODEL,
    efficientvit_weights: str | Path = DEFAULT_EFFICIENTVIT_WEIGHTS,
) -> dict[str, Any]:
    """Segment the entire RGB image and return up to ``num_masks`` largest masks.

    This API deliberately does *not* use a depth image, point cloud, plane
    removal, or geometric size/span filtering. It performs one automatic
    mask-generator (AMG) pass over ``image_bgr``. When ``roi`` is supplied,
    every proposal is clipped to that ROI and ranked by its retained area;
    otherwise the ROI is the entire image.

    Args:
        image_bgr: HxWx3 uint8-like BGR image.
        num_masks: Maximum number of masks to return.  Fewer are returned when
            AMG produces fewer masks meeting ``min_mask_area_px``.
        roi: Optional pixel-coordinate filter ``(x0, y0, x1, y1)``. The right
            and bottom edges are exclusive. ``None`` (the default) uses the
            complete image.
        min_mask_area_px: Optional lower area threshold only; defaults to 1.
        return_formats: Requested representation for each candidate.  Same
            values as the first-frame APIs: ``mask``, ``bboxmask``, and
            ``rotated_bboxmask``.  Defaults to ``mask``.

    Returns:
        ``masks`` and the identical ``candidates`` list, ordered by descending
        area within ``roi``. Every candidate includes its clipped mask, bbox,
        area, AMG quality scores, and source proposal index.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
    requested_count = int(num_masks)
    if requested_count < 1:
        raise ValueError("num_masks must be at least 1")
    minimum_area = int(min_mask_area_px)
    if minimum_area < 1:
        raise ValueError("min_mask_area_px must be at least 1")
    roi_box = _resolve_pixel_roi(image_bgr.shape[:2], roi)

    with _RESIDENT_LOCK:
        if _RESIDENT_WORKER is None:
            requested_backend = _normalize_backend(backend)
            start(
                ckpt=ckpt,
                cfg=cfg,
                backend=requested_backend,
                efficientvit_repo=efficientvit_repo,
                efficientvit_model=efficientvit_model,
                efficientvit_weights=efficientvit_weights,
            )
        worker = _RESIDENT_WORKER
        if worker is None:
            raise RuntimeError("segmentation resident worker failed to start")
        requested_backend = worker.backend if backend is None else _normalize_backend(backend)
        if worker.backend != requested_backend:
            raise RuntimeError("another segmentation backend is running; call close() first")
        return worker.call_full_image_topn(
            image_bgr,
            {
                "num_masks": requested_count,
                "roi": roi_box,
                "min_mask_area_px": minimum_area,
                "return_formats": return_formats,
            },
        )


def _configure_sam2_from_checkpoint(ckpt: str | Path) -> None:
    """Expose the SAM2 checkout colocated with an explicit my_robot checkpoint.

    This direct selector must also work when the policy's legacy ``my_robot``
    compatibility package is not installed.  The loop always supplies its
    checkpoint explicitly, so its sibling ``external/sam2`` checkout is the
    authoritative import location.
    """
    checkpoint = Path(ckpt).expanduser().resolve()
    sam2_repo = DEFAULT_SAM2_REPO
    if not sam2_repo.is_dir():
        try:
            sam2_repo = checkpoint.parents[2] / "external" / "sam2"
        except IndexError as exc:
            raise FileNotFoundError(f"cannot infer SAM2 repo from checkpoint: {checkpoint}") from exc
    if not sam2_repo.is_dir():
        raise FileNotFoundError(f"SAM2 repo not found next to checkpoint: {sam2_repo}")
    repo_text = str(sam2_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        import hydra  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 dependencies are unavailable in this interpreter. Use "
            "/mnt/data2/shurenbin/.conda/envs/dp/bin/python."
        ) from exc


def _safe_torch_device(torch) -> str:
    try:
        if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
            torch.cuda.get_device_properties(torch.cuda.current_device())
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _safe_autocast_context(torch, device: str):
    if device != "cuda":
        return nullcontext()
    try:
        return torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    except Exception:
        return nullcontext()


def _safe_efficientvit_autocast_context(torch, device: str):
    """Use FP16 on CUDA because EfficientViT AMG exports tensors to NumPy.

    NumPy cannot convert PyTorch BFloat16 tensors. SAM2 can use BF16, but the
    Segment Anything AMG inherited by EfficientViT calls ``tensor.numpy()`` on
    prediction tensors, so this backend must use FP16 (or FP32 on CPU).
    """
    if device != "cuda":
        return nullcontext()
    try:
        return torch.autocast("cuda", dtype=torch.float16)
    except Exception:
        return nullcontext()


def validate_explicit_sam2_runtime(ckpt: str | Path) -> None:
    """Fail early unless an explicit-checkpoint SAM2 runtime is usable.

    Hardware callers should run this check before commanding a robot motion.
    """
    _configure_sam2_from_checkpoint(ckpt)
    import torch  # noqa: F401
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: F401
    from sam2.build_sam import build_sam2  # noqa: F401


def _resolve_norm_box(
    image_shape: tuple[int, int],
    norm_box: tuple[float, float, float, float] | None,
) -> list[float] | None:
    if norm_box is None:
        return None
    h, w = [int(v) for v in image_shape[:2]]
    x0, y0, x1, y1 = [float(v) for v in norm_box]
    box = [x0 * w, y0 * h, x1 * w, y1 * h]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _resolve_pixel_roi(
    image_shape: tuple[int, int],
    roi: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    """Validate a half-open pixel ROI, using the complete image for ``None``."""
    h, w = [int(value) for value in image_shape[:2]]
    if roi is None:
        return (0, 0, w, h)
    if len(roi) != 4:
        raise ValueError("roi must be a four-value (x0, y0, x1, y1) tuple")
    x0, y0, x1, y1 = [int(value) for value in roi]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            f"roi {(x0, y0, x1, y1)} must be inside image bounds (0, 0, {w}, {h})"
        )
    return (x0, y0, x1, y1)


def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Keep only the largest foreground component of an AMG proposal.

    A few isolated pixels can otherwise expand an axis-aligned or rotated bbox
    dramatically.  The first-frame selector represents one grasp target, so
    disconnected noise is not part of the selected target mask.
    """
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros(binary.shape, dtype=np.uint8), 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    return ((labels == largest_label).astype(np.uint8) * 255), int(component_count - 1), largest_area


def _mask_compactness(mask: np.ndarray) -> tuple[float, float]:
    """Return contour solidity and foreground fill ratio of its bounding box."""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    x, y, width, height = cv2.boundingRect(contour)
    bbox_area = max(1, int(width) * int(height))
    return (
        contour_area / max(1.0, hull_area),
        float(np.count_nonzero(binary[y:y + height, x:x + width])) / bbox_area,
    )


def _bbox_mask(image_shape: tuple[int, int], bbox: list[float] | None) -> np.ndarray:
    h, w = [int(v) for v in image_shape[:2]]
    output = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        return output
    x0, y0, x1, y1 = [float(v) for v in bbox]
    ix0 = max(0, min(w, int(np.floor(x0))))
    iy0 = max(0, min(h, int(np.floor(y0))))
    ix1 = max(0, min(w, int(np.ceil(x1))))
    iy1 = max(0, min(h, int(np.ceil(y1))))
    if ix1 > ix0 and iy1 > iy0:
        output[iy0:iy1, ix0:ix1] = 255
    return output


def _rotated_bbox_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[float]], dict[str, Any]]:
    """Return a filled minimum-area rotated rectangle around the mask."""
    import cv2

    mask_bool = np.asarray(mask) > 0
    ys, xs = np.where(mask_bool)
    output = np.zeros(mask_bool.shape[:2], dtype=np.uint8)
    if xs.size == 0:
        return output, [], {"center": None, "size": None, "angle_deg": None}

    points = np.column_stack([xs, ys]).astype(np.float32)
    if points.shape[0] < 3:
        bbox = _bbox_from_mask(mask_bool)
        return _bbox_mask(mask_bool.shape[:2], bbox), [], {
            "center": None,
            "size": None,
            "angle_deg": None,
        }

    rect = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rect).astype(np.float32)
    cv2.fillPoly(output, [np.round(corners).astype(np.int32)], 255)
    center, size, angle = rect
    return output, corners.astype(np.float64).tolist(), {
        "center": [float(center[0]), float(center[1])],
        "size": [float(size[0]), float(size[1])],
        "angle_deg": float(angle),
    }


def _normalize_return_formats(return_formats: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if return_formats is None:
        return ("mask",)
    if isinstance(return_formats, str):
        formats = (return_formats,)
    else:
        formats = tuple(return_formats)
    normalized: list[str] = []
    for value in formats:
        name = str(value).strip().lower()
        if name not in SUPPORTED_RETURN_FORMATS:
            raise ValueError(
                f"unknown return format {value!r}; "
                f"supported formats: {SUPPORTED_RETURN_FORMATS}"
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("return_formats must contain at least one format")
    return tuple(normalized)


def _mask_ratio_inside_box(mask: np.ndarray, box: list[float] | None) -> float:
    if box is None:
        return 0.0
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    roi = np.zeros((h, w), dtype=bool)
    ix0 = max(0, min(w, int(np.floor(x0))))
    iy0 = max(0, min(h, int(np.floor(y0))))
    ix1 = max(0, min(w, int(np.ceil(x1))))
    iy1 = max(0, min(h, int(np.ceil(y1))))
    if ix1 > ix0 and iy1 > iy0:
        roi[iy0:iy1, ix0:ix1] = True
    mask_bool = np.asarray(mask) > 0
    total = int(np.count_nonzero(mask_bool))
    return float(np.count_nonzero(mask_bool & roi)) / float(max(1, total))


def _build_refinement_core(
    mask: np.ndarray,
    anchor_point: list[int] | tuple[int, int] | None,
    *,
    depth_m: np.ndarray | None = None,
    height_above_plane_m: np.ndarray | None = None,
) -> np.ndarray:
    """Keep a small seed-centered core, cutting obvious RGB-D depth seams."""
    import cv2

    original = np.asarray(mask) > 0
    if not np.any(original):
        return original
    work = original.copy()
    h, w = original.shape[:2]
    if depth_m is not None or height_above_plane_m is not None:
        cut = np.zeros_like(original, dtype=bool)
        for values, threshold in (
            (depth_m, 0.015),
            (height_above_plane_m, 0.010),
        ):
            if values is None:
                continue
            field = np.asarray(values, dtype=np.float64)
            if field.shape != original.shape:
                continue
            valid = np.isfinite(field)
            horizontal = (
                valid[:, 1:] & valid[:, :-1]
                & (np.abs(field[:, 1:] - field[:, :-1]) > float(threshold))
            )
            vertical = (
                valid[1:, :] & valid[:-1, :]
                & (np.abs(field[1:, :] - field[:-1, :]) > float(threshold))
            )
            cut[:, 1:] |= horizontal
            cut[:, :-1] |= horizontal
            cut[1:, :] |= vertical
            cut[:-1, :] |= vertical
        if np.any(cut & original):
            work &= ~cv2.dilate(
                (cut & original).astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=1,
            ).astype(bool)
            if not np.any(work):
                work = original.copy()

    # Erosion is used only to find a reliable prompt core; the returned SAM
    # mask is still allowed to grow back to the object's full boundary.
    core = cv2.erode(work.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    if not np.any(core):
        core = work
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        core.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return core
    if anchor_point is not None:
        ax = int(np.clip(round(float(anchor_point[0])), 0, w - 1))
        ay = int(np.clip(round(float(anchor_point[1])), 0, h - 1))
        anchor_label = int(labels[ay, ax])
        if anchor_label > 0:
            return labels == anchor_label
        component_ids = range(1, component_count)
        nearest = min(
            component_ids,
            key=lambda component_id: min(
                ((x - ax) ** 2 + (y - ay) ** 2)
                for y, x in zip(*np.where(labels == component_id))
            ),
        )
        return labels == int(nearest)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == largest


def _sample_prompt_points(
    mask: np.ndarray,
    *,
    anchor_point: list[int] | tuple[int, int] | None = None,
    prompt_core: np.ndarray | None = None,
    contact_negatives: bool = True,
) -> tuple[list[list[int]], list[int]]:
    """Create anchored positive/negative prompts for one candidate mask."""
    import cv2

    mask_bool = np.asarray(mask) > 0
    focus = mask_bool if prompt_core is None else (np.asarray(prompt_core) > 0) & mask_bool
    ys, xs = np.where(focus)
    if xs.size == 0:
        return [], []

    h, w = mask_bool.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    points: list[list[int]] = []
    labels: list[int] = []

    def add(x: float, y: float, label: int, *, allow_mask_negative: bool = False) -> None:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        if label == 1 and not focus[yi, xi]:
            return
        if label == 0 and mask_bool[yi, xi] and not allow_mask_negative:
            return
        point = [xi, yi]
        if point not in points:
            points.append(point)
            labels.append(int(label))

    cx, cy = float(xs.mean()), float(ys.mean())
    if anchor_point is not None:
        add(float(anchor_point[0]), float(anchor_point[1]), 1)
    if not points:
        add(cx, cy, 1)
    if not contact_negatives:
        # A second, maximally interior positive point makes expansion robust
        # when the original AMG anchor landed close to a visible boundary.
        distance = cv2.distanceTransform(focus.astype(np.uint8), cv2.DIST_L2, 5)
        deepest_y, deepest_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        add(float(deepest_x), float(deepest_y), 1)

    # A small ring around the seed core supplies negatives at contact seams.
    # Prefer pixels that belong to the original mask (likely another touching
    # object), then fall back to the outside ring if no such pixels exist.
    ring_size = 9 if contact_negatives else 25
    ring = cv2.dilate(
        focus.astype(np.uint8), np.ones((ring_size, ring_size), np.uint8), iterations=1
    ) > 0
    ring &= ~focus
    contact_ring = ring & mask_bool
    outside_ring = ring & ~mask_bool
    negative_pool = (
        contact_ring if contact_negatives and np.count_nonzero(contact_ring) >= 4
        else outside_ring
    )
    negative_y, negative_x = np.where(negative_pool)
    if len(negative_x):
        order = np.argsort(
            (negative_x.astype(np.float64) - cx) ** 2
            + (negative_y.astype(np.float64) - cy) ** 2
        )
        for index in order[::max(1, len(order) // 6)][:6]:
            add(
                float(negative_x[index]), float(negative_y[index]), 0,
                allow_mask_negative=bool(
                    contact_negatives
                    and contact_ring[negative_y[index], negative_x[index]]
                ),
            )
    return points, labels


def _amg_score(mask: np.ndarray, ann: dict[str, Any]) -> float:
    """Score only for diagnostics/tie-breaking; final selection is by area."""
    h, w = mask.shape[:2]
    area_ratio = float(np.count_nonzero(mask)) / float(max(1, h * w))
    if area_ratio < 0.002 or area_ratio > 0.30:
        return -1e9
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return -1e9
    border_penalty = sum(
        weight * int(touches)
        for weight, touches in (
            (-1.8, xs.min() <= 2),
            (-1.2, xs.max() >= w - 3),
            (-0.4, ys.min() <= 2),
            (-1.2, ys.max() >= h - 3),
        )
    )
    compact_area_bonus = min(area_ratio / 0.018, 1.0)
    return (
        2.0 * float(ann.get("predicted_iou", 0.0))
        + 2.0 * float(ann.get("stability_score", 0.0))
        + 0.8 * compact_area_bonus
        + border_penalty
    )


def _padded_mask_crop(
    mask: np.ndarray,
    *,
    padding: int = 24,
) -> tuple[int, int, int, int]:
    """Return a padded crop around support while preserving image bounds."""
    support = np.asarray(mask) > 0
    h, w = support.shape[:2]
    ys, xs = np.where(support)
    if not len(xs):
        return 0, 0, w, h
    return (
        max(0, int(xs.min()) - int(padding)),
        max(0, int(ys.min()) - int(padding)),
        min(w, int(xs.max()) + 1 + int(padding)),
        min(h, int(ys.max()) + 1 + int(padding)),
    )


def _generate_restricted_annotations_direct(
    image_bgr: np.ndarray,
    points: list[list[int]],
    *,
    generator,
    runtime: dict[str, Any],
    pred_iou_thresh: float,
    stability_score_thresh: float,
    keep_features: bool = False,
    nms_iou_threshold: float | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run one batched decoder call without generic AMG crop/RLE plumbing."""
    from segment_anything.utils.amg import batched_mask_to_box, calculate_stability_score
    from torchvision.ops.boxes import batched_nms

    torch = runtime["torch"]
    device = str(runtime["device"])
    autocast_context = runtime["autocast_context"]
    predictor = generator.predictor
    full_h, full_w = image_bgr.shape[:2]
    if crop_box is None:
        crop_box = (0, 0, full_w, full_h)
    crop_x0, crop_y0, crop_x1, crop_y1 = [int(value) for value in crop_box]
    image_rgb = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1, ::-1].copy()
    encode_t0 = time.perf_counter()
    with torch.inference_mode(), autocast_context(torch, device):
        predictor.set_image(image_rgb)
    encode_ms = (time.perf_counter() - encode_t0) * 1000.0
    try:
        source_point_array = np.asarray(points, dtype=np.float32)
        point_array = source_point_array - np.asarray([crop_x0, crop_y0], dtype=np.float32)
        transformed = predictor.apply_coords(point_array, image_rgb.shape[:2])
        in_points = torch.as_tensor(transformed, dtype=torch.float32, device=predictor.device)
        in_labels = torch.ones(len(points), dtype=torch.int, device=predictor.device)
        decode_t0 = time.perf_counter()
        with torch.inference_mode(), autocast_context(torch, device):
            masks, iou_predictions, _ = predictor.predict_torch(
                in_points[:, None, :], in_labels[:, None],
                multimask_output=True, return_logits=True,
            )
        decode_ms = (time.perf_counter() - decode_t0) * 1000.0

        post_t0 = time.perf_counter()
        masks = masks.flatten(0, 1)
        iou_predictions = iou_predictions.flatten(0, 1)
        source_points = np.repeat(source_point_array, 3, axis=0)
        keep = iou_predictions > float(pred_iou_thresh)
        masks, iou_predictions = masks[keep], iou_predictions[keep]
        source_points = source_points[keep.detach().cpu().numpy()]
        stability = calculate_stability_score(
            masks, predictor.model.mask_threshold, generator.stability_score_offset
        )
        keep = stability >= float(stability_score_thresh)
        masks, iou_predictions, stability = masks[keep], iou_predictions[keep], stability[keep]
        source_points = source_points[keep.detach().cpu().numpy()]
        masks = masks > predictor.model.mask_threshold
        boxes = batched_mask_to_box(masks)
        if len(boxes) and nms_iou_threshold is not None:
            keep_indices = batched_nms(
                boxes.float(), iou_predictions, torch.zeros_like(boxes[:, 0]),
                iou_threshold=float(nms_iou_threshold),
            )
            masks, iou_predictions, stability = (
                masks[keep_indices], iou_predictions[keep_indices], stability[keep_indices]
            )
            source_points = source_points[keep_indices.detach().cpu().numpy()]
        masks_np = masks.detach().cpu().numpy()
        iou_np = iou_predictions.detach().cpu().numpy()
        stability_np = stability.detach().cpu().numpy()
        annotations = [
            {
                "segmentation": np.pad(
                    mask,
                    ((crop_y0, full_h - crop_y1), (crop_x0, full_w - crop_x1)),
                    mode="constant",
                ),
                "predicted_iou": float(iou),
                "stability_score": float(stable),
                "point_coords": [[float(point[0]), float(point[1])]],
            }
            for mask, iou, stable, point in zip(masks_np, iou_np, stability_np, source_points)
        ]
        postprocess_ms = (time.perf_counter() - post_t0) * 1000.0
    finally:
        if not keep_features:
            predictor.reset_image()
    return annotations, {
        "encode": float(encode_ms), "decode": float(decode_ms),
        "postprocess": float(postprocess_ms),
        "crop_box": [crop_x0, crop_y0, crop_x1, crop_y1],
    }


def _refine_selected_mask_with_cached_features(
    selected_mask: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    predictor,
    runtime: dict[str, Any],
    anchor_point: list[int] | tuple[int, int] | None = None,
    depth_m: np.ndarray | None = None,
    height_above_plane_m: np.ndarray | None = None,
    crop_box: tuple[int, int, int, int] | list[int] | None = None,
    split_merged: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine one AMG mask with positive/negative points and a tight box."""
    original_full = np.asarray(selected_mask) > 0
    full_h, full_w = original_full.shape[:2]
    if crop_box is None:
        crop_box = (0, 0, full_w, full_h)
    crop_x0, crop_y0, crop_x1, crop_y1 = [int(value) for value in crop_box]
    original = original_full[crop_y0:crop_y1, crop_x0:crop_x1]
    local_anchor = None
    if anchor_point is not None:
        local_anchor = [
            int(anchor_point[0]) - crop_x0,
            int(anchor_point[1]) - crop_y0,
        ]
    local_depth = None if depth_m is None else np.asarray(depth_m)[crop_y0:crop_y1, crop_x0:crop_x1]
    local_height = (
        None if height_above_plane_m is None
        else np.asarray(height_above_plane_m)[crop_y0:crop_y1, crop_x0:crop_x1]
    )
    prompt_core = _build_refinement_core(
        original,
        local_anchor,
        depth_m=local_depth,
        height_above_plane_m=local_height,
    )
    points, labels = _sample_prompt_points(
        original, anchor_point=local_anchor, prompt_core=prompt_core,
        contact_negatives=bool(split_merged),
    )
    bbox = _bbox_from_mask(original.astype(np.uint8))
    if bbox is None or not points or 1 not in labels:
        return original_full, {"accepted": False, "reason": "insufficient_prompts"}

    torch = runtime["torch"]
    device = str(runtime["device"])
    autocast_context = runtime["autocast_context"]
    coords = predictor.apply_coords(np.asarray(points, dtype=np.float32))
    if not split_merged:
        expansion_padding = 24
        bbox = [
            max(0.0, bbox[0] - expansion_padding),
            max(0.0, bbox[1] - expansion_padding),
            min(float(original.shape[1]), bbox[2] + expansion_padding),
            min(float(original.shape[0]), bbox[3] + expansion_padding),
        ]
    transformed_box = predictor.apply_boxes(np.asarray([bbox], dtype=np.float32))
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=predictor.device)[None]
    labels_t = torch.as_tensor(labels, dtype=torch.int, device=predictor.device)[None]
    box_t = torch.as_tensor(transformed_box, dtype=torch.float32, device=predictor.device)
    started = time.perf_counter()
    with torch.inference_mode(), autocast_context(torch, device):
        masks, predicted_iou, _ = predictor.predict_torch(
            coords_t, labels_t, boxes=box_t,
            multimask_output=True, return_logits=False,
        )
    decode_ms = (time.perf_counter() - started) * 1000.0
    masks_np = masks[0].detach().cpu().numpy()
    iou_np = predicted_iou[0].detach().cpu().numpy()
    allowed = (np.asarray(allowed_mask) > 0)[crop_y0:crop_y1, crop_x0:crop_x1]
    original_area = max(1, int(np.count_nonzero(original)))
    candidates = []
    for index, (raw_mask, model_iou) in enumerate(zip(masks_np, iou_np)):
        component, _, area = _largest_connected_component(raw_mask.astype(np.uint8))
        candidate = component > 0
        if area <= 0:
            continue
        intersection = int(np.count_nonzero(candidate & original))
        union = int(np.count_nonzero(candidate | original))
        consensus_iou = intersection / float(max(1, union))
        allowed_ratio = float(np.count_nonzero(candidate & allowed)) / float(area)
        area_ratio = float(area) / float(original_area)
        positive_hit = float(np.mean([
            candidate[int(y), int(x)] for (x, y), label in zip(points, labels) if label == 1
        ]))
        negative_values = [
            candidate[int(y), int(x)] for (x, y), label in zip(points, labels) if label == 0
        ]
        negative_hit = float(np.mean(negative_values)) if negative_values else 0.0
        solidity, bbox_fill_ratio = _mask_compactness(candidate)
        score = (
            1.20 * float(model_iou) + 1.50 * consensus_iou
            + 0.70 * allowed_ratio + 0.35 * positive_hit
            - 0.60 * negative_hit - 0.35 * abs(np.log(max(1e-6, area_ratio)))
        )
        candidates.append({
            "index": int(index), "mask": candidate, "score": float(score),
            "predicted_iou": float(model_iou), "consensus_iou": float(consensus_iou),
            "allowed_ratio": float(allowed_ratio), "area_ratio": float(area_ratio),
            "positive_hit": positive_hit, "negative_hit": negative_hit,
            "solidity": float(solidity), "bbox_fill_ratio": float(bbox_fill_ratio),
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return original_full, {"accepted": False, "reason": "no_refinement_mask", "decode_ms": decode_ms}
    split_candidates = [
        item for item in candidates
        if split_merged
        and 0.35 <= item["area_ratio"] <= 0.70
        and item["allowed_ratio"] >= 0.80
        and item["positive_hit"] >= 0.50
        and item["negative_hit"] <= 0.35
        and item["solidity"] >= 0.72
        and item["bbox_fill_ratio"] >= 0.40
    ]
    best = max(split_candidates, key=lambda item: item["score"]) if split_candidates else candidates[0]
    spill_ratio = float(np.count_nonzero(best["mask"] & ~original)) / float(max(1, int(np.count_nonzero(best["mask"]))))
    split_accepted = bool(split_candidates)
    accepted = split_accepted or (
        best["consensus_iou"] >= 0.55
        and best["allowed_ratio"] >= 0.80
        and 0.55 <= best["area_ratio"] <= 1.30
        and best["negative_hit"] <= 0.25
        and spill_ratio <= 0.20
    )
    debug_candidates = [{k: v for k, v in item.items() if k != "mask"} for item in candidates]
    refined_full = original_full
    if accepted:
        refined_full = np.zeros_like(original_full)
        refined_full[crop_y0:crop_y1, crop_x0:crop_x1] = best["mask"]
    global_points = [[int(x + crop_x0), int(y + crop_y0)] for x, y in points]
    return refined_full, {
        "accepted": bool(accepted), "split_from_merged": split_accepted,
        "decode_ms": float(decode_ms),
        "spill_ratio": spill_ratio,
        "anchor_point": None if anchor_point is None else [int(v) for v in anchor_point],
        "points": global_points, "labels": labels,
        "crop_box": [crop_x0, crop_y0, crop_x1, crop_y1],
        "selected_candidate": {k: v for k, v in best.items() if k != "mask"},
        "candidates": debug_candidates,
    }


def _rank_mask_area_candidates(
    candidates: list[dict[str, Any]],
    similar_area_ratio: float,
) -> list[dict[str, Any]]:
    """Rank candidates by mask area, using height only for near ties."""
    ranked_candidates: list[dict[str, Any]] = []
    remaining_candidates = list(candidates)
    while remaining_candidates:
        largest_area = max(item["area_px"] for item in remaining_candidates)
        comparable_min = largest_area * (1.0 - float(similar_area_ratio))
        comparable = [
            item for item in remaining_candidates
            if item["area_px"] >= comparable_min
        ]
        comparable.sort(
            key=lambda item: (
                -1.0 if item["candidate_height_above_plane_m"] is None
                else item["candidate_height_above_plane_m"],
                item["area_px"],
                item["segmentation_quality"],
            ),
            reverse=True,
        )
        ranked_candidates.extend(comparable)
        comparable_ids = {id(item) for item in comparable}
        remaining_candidates = [
            item for item in remaining_candidates if id(item) not in comparable_ids
        ]
    return ranked_candidates


def _remove_fully_contained_mask_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Drop a candidate whose entire foreground is covered by another mask."""
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, int]] = []
    masks = [np.asarray(candidate["mask"], dtype=bool) for candidate in candidates]
    areas = [int(np.count_nonzero(mask)) for mask in masks]
    for index, candidate in enumerate(candidates):
        container_index = next((
            other_index
            for other_index, other_mask in enumerate(masks)
            if other_index != index
            and areas[other_index] > areas[index]
            and not np.any(masks[index] & ~other_mask)
        ), None)
        if container_index is None:
            kept.append(candidate)
        else:
            removed.append({
                "source_index": int(candidate["source_index"]),
                "container_source_index": int(candidates[container_index]["source_index"]),
            })
    return kept, removed


def _deduplicate_identical_bboxes(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the highest-ranked proposal for each effectively identical box.

    EfficientViT proposals for one workpiece commonly jitter by a few pixels,
    so an IoU of at least 0.90 is treated as the same bounding rectangle.
    """
    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if bbox is None:
            deduplicated.append(candidate)
            continue
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bbox_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        duplicate = False
        for kept in deduplicated:
            kept_bbox = kept.get("bbox")
            if kept_bbox is None:
                continue
            kx0, ky0, kx1, ky1 = (float(value) for value in kept_bbox)
            intersection = max(0.0, min(x1, kx1) - max(x0, kx0)) * max(
                0.0, min(y1, ky1) - max(y0, ky0)
            )
            kept_area = max(0.0, kx1 - kx0) * max(0.0, ky1 - ky0)
            if intersection / max(1.0, bbox_area + kept_area - intersection) >= 0.90:
                duplicate = True
                break
        if duplicate:
            continue
        deduplicated.append(candidate)
    return deduplicated


def _select_efficientvit_restricted_amg_impl(
    image_bgr: np.ndarray,
    *,
    generator,
    runtime: dict[str, Any],
    allowed_mask: np.ndarray,
    points: list[list[int]],
    depth_m: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    height_above_plane_m: np.ndarray | None = None,
    min_allowed_ratio: float = 0.70,
    boundary_min_allowed_ratio: float | None = None,
    support_boundary_margin_px: int = 16,
    normal_object_bbox_area_px: float | None = DEFAULT_NORMAL_OBJECT_MASK_AREA_PX,
    bbox_area_min_factor: float = DEFAULT_BBOX_AREA_MIN_FACTOR,
    bbox_area_max_factor: float = DEFAULT_BBOX_AREA_MAX_FACTOR,
    similar_bbox_area_ratio: float = DEFAULT_SIMILAR_BBOX_AREA_RATIO,
    bbox_area_reference_depth_m: float | None = DEFAULT_BBOX_AREA_REFERENCE_DEPTH_M,
    object_length_range_m: tuple[float, float] | None = DEFAULT_OBJECT_LENGTH_RANGE_M,
    object_width_range_m: tuple[float, float] | None = DEFAULT_OBJECT_WIDTH_RANGE_M,
    object_thickness_range_m: tuple[float, float] | None = DEFAULT_OBJECT_THICKNESS_RANGE_M,
    use_direct_predict: bool = True,
    refine_selected: bool = True,
    boundary_refine_all: bool = False,
    max_refine_candidates: int = 3,
    candidate_preselect_topk: int = 10,
    support_crop_padding_px: int = 24,
    target_center: tuple[int, int] | None = None,
    return_formats=None,
) -> dict[str, Any]:
    """EfficientViT AMG with a custom point grid constrained by RGB-D."""
    import cv2

    requested_formats = _normalize_return_formats(return_formats)
    torch = runtime["torch"]
    device = str(runtime["device"])
    autocast_context = runtime["autocast_context"]
    h, w = image_bgr.shape[:2]
    allowed = np.asarray(allowed_mask) > 0
    metric_depth = None if depth_m is None else np.asarray(depth_m, dtype=np.float64)
    if metric_depth is not None and metric_depth.shape != (h, w):
        raise ValueError(f"depth_m shape {metric_depth.shape} does not match image {(h, w)}")
    K = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
    if K is not None and K.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {K.shape}")
    plane_height = (
        None if height_above_plane_m is None
        else np.asarray(height_above_plane_m, dtype=np.float64)
    )
    if plane_height is not None and plane_height.shape != (h, w):
        raise ValueError(
            f"height_above_plane_m shape {plane_height.shape} does not match image {(h, w)}"
        )
    if normal_object_bbox_area_px is not None and float(normal_object_bbox_area_px) <= 0:
        raise ValueError("normal_object_mask_area_px must be positive or None")
    if float(bbox_area_min_factor) <= 0:
        raise ValueError("bbox_area_min_factor must be positive")
    if float(bbox_area_max_factor) < float(bbox_area_min_factor):
        raise ValueError("bbox_area_max_factor must be >= bbox_area_min_factor")
    if not 0.0 <= float(similar_bbox_area_ratio) < 1.0:
        raise ValueError("similar_bbox_area_ratio must be in [0, 1)")
    if not 0.0 <= float(min_allowed_ratio) <= 1.0:
        raise ValueError("min_allowed_ratio must be in [0, 1]")
    if boundary_min_allowed_ratio is not None and not (
        0.0 <= float(boundary_min_allowed_ratio) <= float(min_allowed_ratio)
    ):
        raise ValueError("boundary_min_allowed_ratio must be in [0, min_allowed_ratio]")
    if int(candidate_preselect_topk) < 1:
        raise ValueError("candidate_preselect_topk must be at least 1")
    normalized_grid = np.asarray(
        [[float(x) / max(1, w), float(y) / max(1, h)] for x, y in points],
        dtype=np.float32,
    )
    support_bbox = _bbox_from_mask(allowed.astype(np.uint8))
    started = time.perf_counter()
    direct_timing = None
    support_crop = _padded_mask_crop(allowed, padding=max(0, int(support_crop_padding_px)))
    if use_direct_predict:
        annotations, direct_timing = _generate_restricted_annotations_direct(
            image_bgr, points, generator=generator, runtime=runtime,
            pred_iou_thresh=float("-inf"), stability_score_thresh=float("-inf"),
            keep_features=bool(refine_selected),
            nms_iou_threshold=None,
            crop_box=support_crop,
        )
    else:
        original_grids = generator.point_grids
        original_pred_iou_thresh = generator.pred_iou_thresh
        original_stability_thresh = generator.stability_score_thresh
        try:
            generator.point_grids = [normalized_grid]
            generator.pred_iou_thresh = float("-inf")
            generator.stability_score_thresh = float("-inf")
            with torch.inference_mode(), autocast_context(torch, device):
                annotations = generator.generate(image_bgr[..., ::-1].copy())
        finally:
            generator.point_grids = original_grids
            generator.pred_iou_thresh = original_pred_iou_thresh
            generator.stability_score_thresh = original_stability_thresh
    generate_ms = (time.perf_counter() - started) * 1000.0

    use_target_bias = target_center is not None
    cx, cy = (
        (float(target_center[0]), float(target_center[1]))
        if use_target_bias else (0.5 * w, 0.5 * h)
    )
    # The decoder returns three masks per point.  Connected-components and
    # per-mask 3-D reductions at full camera resolution dominate the CPU tail,
    # so cheaply rank all masks first and run those operations only on top-k.
    preselect_t0 = time.perf_counter()
    preliminary = []
    expected_area = normal_object_bbox_area_px
    target_x = int(np.clip(round(cx), 0, w - 1))
    target_y = int(np.clip(round(cy), 0, h - 1))
    image_diagonal = max(1.0, float(np.hypot(w, h)))
    for index, ann in enumerate(annotations):
        raw_mask_bool = np.asarray(ann["segmentation"], dtype=bool)
        raw_area = int(np.count_nonzero(raw_mask_bool))
        if raw_area <= 0:
            continue
        overlap = int(np.count_nonzero(raw_mask_bool & allowed))
        allowed_ratio = overlap / float(raw_area)
        # Estimate the largest connected body on a 1/4-resolution mask.  This
        # is cheap enough for every proposal and prevents fragmented/merged
        # masks from occupying all five full-resolution analysis slots.
        coarse_stride = 4
        _coarse_mask, _coarse_components, coarse_largest_area = (
            _largest_connected_component(
                raw_mask_bool[::coarse_stride, ::coarse_stride].astype(np.uint8)
            )
        )
        estimated_largest_area = int(coarse_largest_area * coarse_stride ** 2)
        estimated_3d_length_m = None
        if metric_depth is not None and K is not None:
            estimated_3d_length_m = _coarse_mask_3d_length(
                _coarse_mask, metric_depth, K, coarse_stride
            )
        annotation_points = ann.get("point_coords") or []
        anchor_point = None
        anchor_distance = image_diagonal
        if annotation_points:
            anchor_point = [
                int(round(float(annotation_points[0][0]))),
                int(round(float(annotation_points[0][1]))),
            ]
            anchor_distance = float(np.hypot(anchor_point[0] - cx, anchor_point[1] - cy))
        area_score = 0.0
        if expected_area is not None:
            area_score = -abs(np.log(max(1.0, raw_area) / float(expected_area)))
        preliminary_score = (
            float(ann.get("predicted_iou", 0.0))
            + 0.60 * float(ann.get("stability_score", 0.0))
            + 0.90 * allowed_ratio
            + 0.35 * area_score
            + (0.20 * (1.0 - min(1.0, anchor_distance / image_diagonal)) if use_target_bias else 0.0)
            + (0.50 * float(raw_mask_bool[target_y, target_x]) if use_target_bias else 0.0)
        )
        preliminary.append({
            "source_index": int(index), "annotation": ann,
            "mask": raw_mask_bool, "raw_area_px": raw_area,
            "raw_allowed_ratio": float(allowed_ratio),
            "estimated_largest_component_area_px": estimated_largest_area,
            "estimated_3d_length_m": estimated_3d_length_m,
            "preliminary_score": float(preliminary_score),
            "anchor_point": anchor_point,
        })
    estimated_min = (
        None if expected_area is None
        else float(expected_area) * float(bbox_area_min_factor)
    )
    estimated_max = (
        None if expected_area is None
        else float(expected_area) * float(bbox_area_max_factor)
    )
    preliminary.sort(
        key=lambda item: (
            object_length_range_m is None
            or (
                item["estimated_3d_length_m"] is not None
                and float(object_length_range_m[0])
                <= item["estimated_3d_length_m"]
                <= float(object_length_range_m[1])
            ),
            estimated_min is None
            or estimated_min
            <= item["estimated_largest_component_area_px"]
            <= estimated_max,
            0.0 if expected_area is None else -abs(
                item["estimated_largest_component_area_px"] - float(expected_area)
            ),
            item["preliminary_score"],
        ),
        reverse=True,
    )
    # Preserve spatial/prompt diversity: three decoder masks from one anchor
    # should not crowd all other workpieces out of top-k.  Take the best mask
    # per anchor first, then fill any remaining slots by global rank.
    expensive_candidates = []
    selected_preliminary_ids: set[int] = set()
    selected_anchors: set[tuple[int, int] | None] = set()
    for item in preliminary:
        anchor = (
            None if item["anchor_point"] is None
            else tuple(int(value) for value in item["anchor_point"])
        )
        if anchor in selected_anchors:
            continue
        expensive_candidates.append(item)
        selected_preliminary_ids.add(id(item))
        selected_anchors.add(anchor)
        if len(expensive_candidates) >= int(candidate_preselect_topk):
            break
    if len(expensive_candidates) < int(candidate_preselect_topk):
        for item in preliminary:
            if id(item) in selected_preliminary_ids:
                continue
            expensive_candidates.append(item)
            if len(expensive_candidates) >= int(candidate_preselect_topk):
                break
    initial_preselected_count = len(expensive_candidates)
    initial_preselected_ids = {id(item) for item in expensive_candidates}
    fallback_candidates = [
        item for item in preliminary if id(item) not in initial_preselected_ids
    ]
    candidates_to_check = expensive_candidates + fallback_candidates
    preselect_ms = (time.perf_counter() - preselect_t0) * 1000.0

    candidates = []
    rejected_candidates = []
    rejected = {
        "empty": int(len(annotations) - len(preliminary)),
        "not_in_preselected_topk": int(len(fallback_candidates)),
        "outside_rgbd_support": 0,
        "invalid_3d_length": 0,
        "mask_area_too_small": 0, "mask_area_too_large": 0,
    }
    expensive_t0 = time.perf_counter()
    checked_candidates = 0
    for preselected in candidates_to_check:
        # Keep the common path bounded by top-k. If every initial candidate is
        # rejected, progressively inspect the remaining ranked proposals and
        # stop as soon as one valid fallback is found.
        if checked_candidates >= initial_preselected_count and candidates:
            break
        checked_candidates += 1
        rejected["not_in_preselected_topk"] = int(
            len(preliminary) - checked_candidates
        )
        index = int(preselected["source_index"])
        ann = preselected["annotation"]
        raw_mask_bool = preselected["mask"]
        raw_area = int(preselected["raw_area_px"])
        raw_bbox = _bbox_from_mask(raw_mask_bool.astype(np.uint8))
        # Box-area filtering is the candidate-size validity rule.  Measure
        # the box on the proposal's main foreground body, so detached pixels
        # cannot stretch a fragmented proposal's union box.
        primary_mask, components, largest_component_area = (
            _largest_connected_component(raw_mask_bool.astype(np.uint8))
        )
        mask_bool = primary_mask > 0
        area = int(largest_component_area)
        candidate_bbox = _bbox_from_mask(primary_mask)
        if candidate_bbox is None:
            rejected["empty"] += 1
            continue
        raw_bbox_area = (
            0.0 if raw_bbox is None
            else float(max(0.0, raw_bbox[2] - raw_bbox[0]))
            * float(max(0.0, raw_bbox[3] - raw_bbox[1]))
        )
        bbox_area = (
            float(max(0.0, candidate_bbox[2] - candidate_bbox[0]))
            * float(max(0.0, candidate_bbox[3] - candidate_bbox[1]))
        )
        touches_support_boundary = False
        if support_bbox is not None:
            sx0, sy0, sx1, sy1 = (float(value) for value in support_bbox)
            margin = float(max(0, int(support_boundary_margin_px)))
            touches_support_boundary = bool(
                candidate_bbox[0] <= sx0 + margin
                or candidate_bbox[1] <= sy0 + margin
                or candidate_bbox[2] >= sx1 - margin
                or candidate_bbox[3] >= sy1 - margin
            )
        required_allowed_ratio = float(min_allowed_ratio)
        if boundary_min_allowed_ratio is not None and touches_support_boundary:
            required_allowed_ratio = float(boundary_min_allowed_ratio)
        overlap = int(np.count_nonzero(mask_bool & allowed))
        allowed_ratio = overlap / float(area)
        ys, xs = np.where(mask_bool)
        center_distance = float(np.hypot(float(xs.mean()) - cx, float(ys.mean()) - cy))
        center_score = np.exp(-center_distance / max(1.0, 0.20 * np.hypot(w, h)))
        target_hit = float(mask_bool[int(np.clip(round(cy), 0, h - 1)), int(np.clip(round(cx), 0, w - 1))])
        candidate_depth_m = None
        candidate_height_m = None
        expected_bbox_area = normal_object_bbox_area_px
        if metric_depth is not None:
            depth_values = metric_depth[
                mask_bool & np.isfinite(metric_depth) & (metric_depth > 1e-6)
            ]
            if depth_values.size:
                candidate_depth_m = float(np.median(depth_values))
        if plane_height is not None:
            height_values = plane_height[
                mask_bool & np.isfinite(plane_height) & (plane_height >= 0.0)
            ]
            if height_values.size:
                candidate_height_m = float(np.median(height_values))
        bbox_area_min = (
            None if expected_bbox_area is None
            else float(expected_bbox_area) * float(bbox_area_min_factor)
        )
        bbox_area_max = (
            None if expected_bbox_area is None
            else float(expected_bbox_area) * float(bbox_area_max_factor)
        )
        if allowed_ratio < required_allowed_ratio:
            rejected["outside_rgbd_support"] += 1
            rejected_candidates.append({
                "source_index": int(index), "reason": "outside_rgbd_support",
                "area_px": int(area), "bbox": candidate_bbox,
                "bbox_area_px": bbox_area, "raw_area_px": int(raw_area),
                "allowed_ratio": float(allowed_ratio),
                "min_allowed_ratio": required_allowed_ratio,
                "boundary_relaxed_support": touches_support_boundary,
                "candidate_height_above_plane_m": candidate_height_m,
            })
            continue
        if bbox_area_min is not None and bbox_area < bbox_area_min:
            rejected["mask_area_too_small"] += 1
            rejected_candidates.append({
                "source_index": int(index), "reason": "mask_area_too_small",
                "area_px": int(area), "bbox": candidate_bbox,
                "bbox_area_px": bbox_area,
                "raw_area_px": int(raw_area),
                "raw_bbox": raw_bbox, "raw_bbox_area_px": raw_bbox_area,
                "largest_component_area_px": int(largest_component_area),
                "candidate_depth_m": candidate_depth_m,
                "normal_bbox_area_px": expected_bbox_area,
                "bbox_area_min_px": bbox_area_min,
                "bbox_area_max_px": bbox_area_max,
            })
            continue
        if bbox_area_max is not None and bbox_area > bbox_area_max:
            rejected["mask_area_too_large"] += 1
            rejected_candidates.append({
                "source_index": int(index), "reason": "mask_area_too_large",
                "area_px": int(area), "bbox": candidate_bbox,
                "bbox_area_px": bbox_area,
                "raw_area_px": int(raw_area),
                "raw_bbox": raw_bbox, "raw_bbox_area_px": raw_bbox_area,
                "largest_component_area_px": int(largest_component_area),
                "candidate_depth_m": candidate_depth_m,
                "normal_bbox_area_px": expected_bbox_area,
                "bbox_area_min_px": bbox_area_min,
                "bbox_area_max_px": bbox_area_max,
            })
            continue
        object_extents_m = None
        if metric_depth is not None and K is not None:
            object_extents_m = _robust_mask_3d_extents(mask_bool, metric_depth, K)
        if object_extents_m is not None and object_length_range_m is not None:
            min_length, max_length = [float(value) for value in object_length_range_m]
            measured_length = float(object_extents_m[0])
            if not min_length <= measured_length <= max_length:
                rejected["invalid_3d_length"] += 1
                rejected_candidates.append({
                    "source_index": int(index), "reason": "invalid_3d_length",
                    "area_px": int(area), "bbox": candidate_bbox,
                    "bbox_area_px": bbox_area,
                    "allowed_ratio": float(allowed_ratio),
                    "object_3d_extents_m": object_extents_m,
                    "object_length_range_m": [min_length, max_length],
                })
                continue
        score = (
            float(ann.get("predicted_iou", 0.0))
            + 0.60 * float(ann.get("stability_score", 0.0))
            + 0.90 * allowed_ratio
            + (0.25 * float(center_score) if use_target_bias else 0.0)
            + (1.20 * target_hit if use_target_bias else 0.0)
            - 0.04 * max(0, components - 1)
        )
        segmentation_quality = (
            float(ann.get("predicted_iou", 0.0))
            + 0.60 * float(ann.get("stability_score", 0.0))
        )
        object_footprint_m2 = (
            None if object_extents_m is None
            else float(object_extents_m[0] * object_extents_m[1])
        )
        mask_solidity, bbox_fill_ratio = _mask_compactness(mask_bool)
        merged_suspect = bool(mask_solidity < 0.78 and bbox_fill_ratio < 0.55)
        anchor_point = preselected["anchor_point"]
        candidates.append({
            "source_index": int(index), "mask": mask_bool, "area_px": int(area),
            "raw_area_px": int(raw_area),
            "score": float(score), "predicted_iou": float(ann.get("predicted_iou", 0.0)),
            "stability_score": float(ann.get("stability_score", 0.0)),
            "allowed_ratio": float(allowed_ratio), "center_distance_px": center_distance,
            "required_allowed_ratio": required_allowed_ratio,
            "boundary_relaxed_support": touches_support_boundary,
            "target_center_hit": target_hit, "bbox": candidate_bbox,
            "bbox_area_px": bbox_area,
            "raw_bbox": raw_bbox, "raw_bbox_area_px": raw_bbox_area,
            "largest_component_area_px": int(largest_component_area),
            "anchor_point": anchor_point,
            "candidate_depth_m": candidate_depth_m,
            "candidate_height_above_plane_m": candidate_height_m,
            "normal_bbox_area_px": expected_bbox_area,
            "bbox_area_min_px": bbox_area_min,
            "bbox_area_max_px": bbox_area_max,
            "object_3d_extents_m": object_extents_m,
            "components": int(components),
            "segmentation_quality": float(segmentation_quality),
            "object_footprint_m2": object_footprint_m2,
            "mask_solidity": float(mask_solidity),
            "bbox_fill_ratio": float(bbox_fill_ratio),
            "merged_suspect": merged_suspect,
            "preliminary_score": float(preselected["preliminary_score"]),
        })
    expensive_candidate_ms = (time.perf_counter() - expensive_t0) * 1000.0
    if not candidates:
        raise RuntimeError(
            f"restricted AMG produced no valid non-plane mask; raw={len(annotations)} "
            f"rejected={rejected}"
        )
    # Rank by actual SAM mask area. Candidates within the configured fraction
    # of the largest remaining area form one comparable group; only within
    # that group does physical height above the table take precedence.
    candidates = _rank_mask_area_candidates(candidates, similar_bbox_area_ratio)
    area_valid_candidates = len(candidates)
    refinement = {"accepted": False, "reason": "disabled_or_uncached"}
    refinement_candidates: list[dict[str, Any]] = []
    refinement_ms = 0.0
    if use_direct_predict and refine_selected and candidates:
        refinement_t0 = time.perf_counter()
        try:
            # The feature map is already resident from the batched AMG call;
            # refining three masks only adds three decoder calls, not three
            # image encodes. Refine the largest candidates before final rank.
            for candidate in candidates[:max(1, int(max_refine_candidates))]:
                refined_mask, candidate_refinement = (
                    _refine_selected_mask_with_cached_features(
                        candidate["mask"], allowed,
                        predictor=generator.predictor, runtime=runtime,
                        anchor_point=candidate.get("anchor_point"),
                        depth_m=metric_depth,
                        height_above_plane_m=plane_height,
                        crop_box=support_crop,
                        split_merged=bool(boundary_refine_all or candidate.get("merged_suspect", False)),
                    )
                )
                refined_area = int(np.count_nonzero(refined_mask))
                refined_bbox = _bbox_from_mask(np.asarray(refined_mask, dtype=np.uint8))
                refined_bbox_area = (
                    0.0 if refined_bbox is None else
                    float(refined_bbox[2] - refined_bbox[0])
                    * float(refined_bbox[3] - refined_bbox[1])
                )
                min_area = candidate.get("bbox_area_min_px")
                max_area = candidate.get("bbox_area_max_px")
                accepted = bool(candidate_refinement.get("accepted", False))
                split_from_merged = bool(
                    candidate_refinement.get("split_from_merged", False)
                )
                if accepted and not split_from_merged and (
                    (min_area is not None and refined_bbox_area < float(min_area))
                    or (max_area is not None and refined_bbox_area > float(max_area))
                ):
                    accepted = False
                    candidate_refinement = {
                        **candidate_refinement,
                        "accepted": False,
                        "reason": "refined_bbox_outside_area_limits",
                    }
                if (
                    accepted
                    and metric_depth is not None
                    and K is not None
                    and object_length_range_m is not None
                ):
                    refined_extents = _robust_mask_3d_extents(
                        np.asarray(refined_mask, dtype=bool), metric_depth, K
                    )
                    min_length, max_length = [
                        float(value) for value in object_length_range_m
                    ]
                    if (
                        refined_extents is None
                        or not min_length <= float(refined_extents[0]) <= max_length
                    ):
                        accepted = False
                        candidate_refinement = {
                            **candidate_refinement,
                            "accepted": False,
                            "reason": "refined_mask_invalid_3d_length",
                            "refined_object_3d_extents_m": refined_extents,
                        }
                if accepted:
                    candidate["mask"] = np.asarray(refined_mask, dtype=bool)
                    candidate["area_px"] = refined_area
                    candidate["bbox"] = refined_bbox
                    candidate["bbox_area_px"] = refined_bbox_area
                    if metric_depth is not None:
                        values = metric_depth[
                            candidate["mask"]
                            & np.isfinite(metric_depth)
                            & (metric_depth > 1e-6)
                        ]
                        candidate["candidate_depth_m"] = (
                            None if not values.size else float(np.median(values))
                        )
                    if plane_height is not None:
                        values = plane_height[
                            candidate["mask"]
                            & np.isfinite(plane_height)
                            & (plane_height >= 0.0)
                        ]
                        candidate["candidate_height_above_plane_m"] = (
                            None if not values.size else float(np.median(values))
                        )
                    if metric_depth is not None and K is not None:
                        extents = _robust_mask_3d_extents(
                            candidate["mask"], metric_depth, K
                        )
                        candidate["object_3d_extents_m"] = extents
                        candidate["object_footprint_m2"] = (
                            None if extents is None
                            else float(extents[0] * extents[1])
                        )
                        if extents is not None and object_length_range_m is not None:
                            min_length, max_length = [
                                float(value) for value in object_length_range_m
                            ]
                            candidate["geometry_rejected"] = not (
                                min_length <= float(extents[0]) <= max_length
                            )
                    solidity, fill_ratio = _mask_compactness(candidate["mask"])
                    candidate["mask_solidity"] = float(solidity)
                    candidate["bbox_fill_ratio"] = float(fill_ratio)
                candidate["merged_rejected"] = bool(
                    candidate.get("merged_suspect", False) and not split_from_merged
                )
                candidate["refinement"] = candidate_refinement
                refinement_candidates.append({
                    "source_index": int(candidate["source_index"]),
                    "accepted": bool(accepted),
                    "area_px": int(refined_area if accepted else candidate["area_px"]),
                    "decode_ms": candidate_refinement.get("decode_ms"),
                    "reason": candidate_refinement.get("reason"),
                })
            non_merged_candidates = [
                candidate for candidate in candidates
                if not candidate.get("merged_rejected", False)
                and not candidate.get("geometry_rejected", False)
            ]
            if non_merged_candidates:
                candidates = non_merged_candidates
            candidates = _rank_mask_area_candidates(candidates, similar_bbox_area_ratio)
            candidates = _deduplicate_identical_bboxes(candidates)
            selected = candidates[0]
            refinement = selected.get(
                "refinement", {"accepted": False, "reason": "not_refined"}
            )
        finally:
            generator.predictor.reset_image()
            refinement_ms = (time.perf_counter() - refinement_t0) * 1000.0
    else:
        selected = candidates[0]
    # Boundary refinement can change boxes, so apply the near-identical-box
    # rule again just before exposing the ranked candidate list.
    candidates = _deduplicate_identical_bboxes(candidates)
    selected = candidates[0]
    selected_mask = selected["mask"]
    mask = np.asarray(selected_mask, dtype=np.uint8) * 255
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        raise RuntimeError("restricted AMG selected an empty mask")
    rotated_mask, rotated_bbox, rotated_params = _rotated_bbox_mask(mask)
    formats = {}
    if "mask" in requested_formats:
        formats["mask"] = mask
    if "bboxmask" in requested_formats:
        formats["bboxmask"] = _bbox_mask(mask.shape, bbox)
    if "rotated_bboxmask" in requested_formats:
        formats["rotated_bboxmask"] = rotated_mask
    candidate_summary = [{k: v for k, v in item.items() if k != "mask"} for item in candidates]
    candidate_results = [
        {
            **summary,
            "mask": np.asarray(item["mask"], dtype=np.uint8) * 255,
            "selected": int(item["source_index"]) == int(selected["source_index"]),
        }
        for item, summary in zip(candidates, candidate_summary)
    ]
    return {
        **formats, "formats": formats, "return_formats": list(requested_formats),
        "mask": mask, "bbox": bbox, "rotated_bbox": rotated_bbox,
        "rotated_bbox_params": rotated_params, "points": points,
        "score": float(selected["score"]), "backend": "efficientvit_restricted_amg",
        "raw_annotations": int(len(annotations)),
        "preselected_candidates": int(initial_preselected_count),
        "checked_candidates": int(checked_candidates),
        "area_valid_candidates": int(area_valid_candidates),
        "valid_candidates": int(len(candidates)),
        "selected_source_index": int(selected["source_index"]),
        "selection_rule": "largest_mask_area_with_height_tiebreak_for_similar_areas",
        "refinement": refinement,
        "refinement_candidates": refinement_candidates,
        "candidate_scores": candidate_summary,
        "candidates": candidate_results,
        "rejected": rejected,
        "rejected_candidates": rejected_candidates,
        "normal_object_mask_area_px": normal_object_bbox_area_px,
        "bbox_area_min_factor": float(bbox_area_min_factor),
        "bbox_area_max_factor": float(bbox_area_max_factor),
        "similar_bbox_area_ratio": float(similar_bbox_area_ratio),
        "bbox_area_reference_depth_m": bbox_area_reference_depth_m,
        "timing_ms": {
            "generate": generate_ms,
            "direct_predict": direct_timing,
            "candidate_preselect": float(preselect_ms),
            "topk_connected_components_and_3d": float(expensive_candidate_ms),
            "refinement": float(refinement_ms),
            "total": (time.perf_counter() - started) * 1000.0,
        },
    }


def _select_efficientvit_from_points_impl(
    image_bgr: np.ndarray,
    *,
    predictor,
    runtime: dict[str, Any],
    points: list[list[int]],
    labels: list[int],
    box=None,
    support_mask=None,
    min_selection_margin: float = 0.08,
    return_formats=None,
) -> dict[str, Any]:
    requested_formats = _normalize_return_formats(return_formats)
    torch = runtime["torch"]
    device = str(runtime["device"])
    autocast_context = runtime["autocast_context"]
    started = time.perf_counter()
    image_rgb = image_bgr[..., ::-1].copy()
    encode_t0 = time.perf_counter()
    with torch.inference_mode(), autocast_context(torch, device):
        predictor.set_image(image_rgb)
    encode_ms = (time.perf_counter() - encode_t0) * 1000.0
    decode_t0 = time.perf_counter()
    with torch.inference_mode(), autocast_context(torch, device):
        masks, scores, _logits = predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            box=None if box is None else np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0
    model_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    h, w = image_bgr.shape[:2]
    support = None if support_mask is None else np.asarray(support_mask) > 0
    if support is not None and support.shape != (h, w):
        raise ValueError(f"support_mask shape {support.shape} does not match image {(h, w)}")
    box_values = None if box is None else [int(round(v)) for v in box]
    candidate_debug = []
    selection_scores = []
    positive_points = [p for p, label in zip(points, labels) if int(label) == 1]
    negative_points = [p for p, label in zip(points, labels) if int(label) == 0]
    for index, raw_mask in enumerate(masks):
        candidate, component_count, area = _largest_connected_component(np.asarray(raw_mask))
        candidate = candidate > 0
        support_precision = 0.0
        support_recall = 0.0
        if support is not None and area:
            overlap = int(np.count_nonzero(candidate & support))
            support_precision = overlap / float(max(1, area))
            support_recall = overlap / float(max(1, np.count_nonzero(support)))
        positive_hit = float(np.mean([candidate[int(p[1]), int(p[0])] for p in positive_points]))
        negative_hit = float(np.mean([candidate[int(p[1]), int(p[0])] for p in negative_points])) if negative_points else 0.0
        box_spill = 0.0
        if box_values is not None and area:
            x0, y0, x1, y1 = box_values
            inside = np.zeros((h, w), dtype=bool)
            inside[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
            box_spill = float(np.count_nonzero(candidate & ~inside)) / float(area)
        area_ratio = area / float(max(1, h * w))
        geometry_penalty = max(0.0, area_ratio - 0.08) * 5.0 + 0.08 * max(0, component_count - 1)
        selection_score = (
            float(model_scores[index])
            + 0.85 * support_precision
            + 0.20 * min(1.0, support_recall)
            + 0.35 * positive_hit
            - 0.90 * negative_hit
            - 0.65 * box_spill
            - geometry_penalty
        )
        selection_scores.append(selection_score)
        candidate_debug.append({
            "index": int(index), "model_score": float(model_scores[index]),
            "selection_score": float(selection_score), "area_px": int(area),
            "components": int(component_count), "support_precision": float(support_precision),
            "support_recall": float(support_recall), "positive_hit": positive_hit,
            "negative_hit": negative_hit, "box_spill": box_spill,
        })
    order = np.argsort(np.asarray(selection_scores))[::-1]
    best = int(order[0])
    runner_up = int(order[1]) if len(order) > 1 else best
    margin = float(selection_scores[best] - selection_scores[runner_up]) if len(order) > 1 else float("inf")
    best_mask = np.asarray(masks[best]) > 0
    runner_up_mask = np.asarray(masks[runner_up]) > 0
    union = int(np.count_nonzero(best_mask | runner_up_mask))
    candidate_consensus_iou = (
        float(np.count_nonzero(best_mask & runner_up_mask)) / float(max(1, union))
    )
    # A small score margin is safe when both leading candidates describe the
    # same object. Reject only genuinely different, similarly ranked masks.
    if margin < float(min_selection_margin) and candidate_consensus_iou < 0.85:
        raise RuntimeError(
            f"ambiguous EfficientViT masks: selection margin {margin:.3f} "
            f"< {float(min_selection_margin):.3f}, consensus IoU "
            f"{candidate_consensus_iou:.3f} < 0.850"
        )
    mask, _components, _area = _largest_connected_component(np.asarray(masks[best]))
    mask = (mask > 0).astype(np.uint8) * 255
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        raise RuntimeError("EfficientViT point prompt produced an empty mask")
    rotated_mask, rotated_bbox, rotated_params = _rotated_bbox_mask(mask)
    formats = {}
    if "mask" in requested_formats:
        formats["mask"] = mask
    if "bboxmask" in requested_formats:
        formats["bboxmask"] = _bbox_mask(mask.shape, bbox)
    if "rotated_bboxmask" in requested_formats:
        formats["rotated_bboxmask"] = rotated_mask
    return {
        **formats,
        "formats": formats,
        "return_formats": list(requested_formats),
        "bbox": bbox,
        "rotated_bbox": rotated_bbox,
        "rotated_bbox_params": rotated_params,
        "points": [list(map(int, point)) for point in points],
        "labels": [int(v) for v in labels],
        "score": float(model_scores[best]),
        "selection_score": float(selection_scores[best]),
        "selection_margin": margin,
        "candidate_consensus_iou": candidate_consensus_iou,
        "selected_candidate": best,
        "candidate_scores": candidate_debug,
        "prompt_box": box_values,
        "mask": mask,
        "backend": "efficientvit_point_prompt",
        "timing_ms": {
            "image_encode": encode_ms,
            "prompt_decode": decode_ms,
            "total": (time.perf_counter() - started) * 1000.0,
        },
    }


def _segment_full_image_top_n_impl(
    image_bgr: np.ndarray,
    *,
    num_masks: int,
    roi: tuple[int, int, int, int],
    min_mask_area_px: int,
    return_formats: str | list[str] | tuple[str, ...] | None,
    generator,
    runtime: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    """Resident-worker implementation for :func:`segment_full_image_top_n`."""
    requested_formats = _normalize_return_formats(return_formats)
    torch = runtime["torch"]
    device = str(runtime["device"])
    get_autocast_context = runtime["autocast_context"]
    started = time.perf_counter()
    image_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    generate_t0 = time.perf_counter()
    with torch.inference_mode(), get_autocast_context(torch, device):
        annotations = generator.generate(image_rgb)
    generate_ms = (time.perf_counter() - generate_t0) * 1000.0

    candidates: list[dict[str, Any]] = []
    x0, y0, x1, y1 = roi
    for source_index, annotation in enumerate(annotations):
        # Do not reduce to the largest connected component here: the requested
        # filtering is still purely image based; no point-cloud or other
        # object-selection preprocessing is involved.
        original_mask_bool = np.asarray(annotation["segmentation"], dtype=bool)
        raw_area_px = int(np.count_nonzero(original_mask_bool))
        # Preserve image-sized masks for downstream tracking, while retaining
        # only pixels in the caller's ROI for filtering and area ranking.
        mask_bool = np.zeros_like(original_mask_bool, dtype=bool)
        mask_bool[y0:y1, x0:x1] = original_mask_bool[y0:y1, x0:x1]
        area_px = int(np.count_nonzero(mask_bool))
        if area_px < min_mask_area_px:
            continue
        bbox = _bbox_from_mask(mask_bool)
        if bbox is None:
            continue
        mask = mask_bool.astype(np.uint8) * 255
        rotated_mask, rotated_bbox, rotated_bbox_params = _rotated_bbox_mask(mask)
        formats: dict[str, np.ndarray] = {}
        if "mask" in requested_formats:
            formats["mask"] = mask
        if "bboxmask" in requested_formats:
            formats["bboxmask"] = _bbox_mask(mask.shape, bbox)
        if "rotated_bboxmask" in requested_formats:
            formats["rotated_bboxmask"] = rotated_mask
        candidates.append(
            {
                **formats,
                "formats": formats,
                "return_formats": list(requested_formats),
                "source_index": int(source_index),
                "bbox": [float(value) for value in bbox],
                "rotated_bbox": rotated_bbox,
                "rotated_bbox_params": rotated_bbox_params,
                "area_px": area_px,
                "area_ratio": float(area_px) / float(mask_bool.size),
                "raw_area_px": raw_area_px,
                "roi_mask_ratio": float(area_px) / float(max(1, raw_area_px)),
                "predicted_iou": float(annotation.get("predicted_iou", 0.0)),
                "stability_score": float(annotation.get("stability_score", 0.0)),
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(item["area_px"]),
            -float(item["predicted_iou"]),
            -float(item["stability_score"]),
            int(item["source_index"]),
        )
    )
    selected = candidates[:num_masks]
    return {
        "masks": selected,
        "candidates": selected,
        "requested_count": int(num_masks),
        "actual_count": len(selected),
        "available_count": len(candidates),
        "min_mask_area_px": int(min_mask_area_px),
        "roi": [int(value) for value in roi],
        "selection_rule": "full_image_amg_roi_mask_area_descending",
        "return_formats": list(requested_formats),
        "backend": backend,
        "resident_model": True,
        "timing_ms": {
            "generate": float(generate_ms),
            "total": float((time.perf_counter() - started) * 1000.0),
        },
    }


def _legacy_select_largest_sam2_mask(
    image_bgr: np.ndarray,
    *,
    ckpt=None,
    cfg: str | None = None,
    candidate_roi_norm: tuple[float, float, float, float] | None = None,
    candidate_roi_min_mask_ratio: float = 0.60,
    topk: int = 8,
    return_formats: str | list[str] | tuple[str, ...] | None = None,
    return_debug: bool = False,
    backend: str | None = None,
    efficientvit_repo: str | Path = DEFAULT_EFFICIENTVIT_REPO,
    efficientvit_model: str = DEFAULT_EFFICIENTVIT_MODEL,
    efficientvit_weights: str | Path = DEFAULT_EFFICIENTVIT_WEIGHTS,
    _resident_generator=None,
    _resident_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and select the largest valid SAM2 AMG mask.

    Args:
        image_bgr: Input image in HxWx3 BGR format.
        ckpt: SAM2 checkpoint path.
        cfg: SAM2 model config name/path.
        candidate_roi_norm: Optional ``(x0, y0, x1, y1)`` ROI in [0, 1].
        candidate_roi_min_mask_ratio: Required fraction of mask pixels inside ROI.
        topk: Maximum number of returned candidates kept for diagnostics.
        return_formats: One or more of ``mask``, ``bboxmask`` and
            ``rotated_bboxmask``. Defaults to ``("mask",)``.
        return_debug: Include every raw AMG mask and its accept/reject reason.
            Defaults to ``False`` because raw masks can consume substantial memory.
        backend: ``"sam2"`` or ``"efficientvit"``. If omitted while a backend
            is resident, the active backend is reused.

    Returns:
        A dictionary containing ``mask`` (0/255 uint8), ``bbox``, ``points``,
        ``labels``, ``candidate`` and selection statistics.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape HxWx3, got {image_bgr.shape}")

    # Public calls automatically start and use the resident worker. The worker
    # re-enters with its private generator/runtime arguments to run inference.
    if _resident_generator is None:
        with _RESIDENT_LOCK:
            if _RESIDENT_WORKER is None:
                requested_backend = _normalize_backend(backend)
                start(
                    ckpt=ckpt,
                    cfg=cfg,
                    backend=requested_backend,
                    efficientvit_repo=efficientvit_repo,
                    efficientvit_model=efficientvit_model,
                    efficientvit_weights=efficientvit_weights,
                )
            worker = _RESIDENT_WORKER
            if worker is None:
                raise RuntimeError("SAM2 resident worker failed to start")
            requested_backend = worker.backend if backend is None else _normalize_backend(backend)
            if worker.backend != requested_backend:
                raise RuntimeError("another segmentation backend is running; call close() first")
            if worker.backend == "sam2":
                resolved_ckpt, resolved_cfg = _resolve_runtime_config(ckpt, cfg)
                if worker.ckpt != resolved_ckpt or worker.cfg != resolved_cfg:
                    raise RuntimeError(
                        "SAM2 is running with another checkpoint/config; "
                        "call close() before this inference"
                    )
            # Keep lifecycle lock until the response arrives, so close() cannot
            # enqueue shutdown ahead of this request and strand the caller.
            return worker.call(
                image_bgr,
                {
                    "candidate_roi_norm": candidate_roi_norm,
                    "candidate_roi_min_mask_ratio": candidate_roi_min_mask_ratio,
                    "topk": topk,
                    "return_formats": return_formats,
                    "return_debug": return_debug,
                },
            )
    requested_formats = _normalize_return_formats(return_formats)

    resident = _resident_generator is not None
    if resident:
        if not _resident_runtime:
            raise ValueError("_resident_runtime is required with _resident_generator")
        torch = _resident_runtime["torch"]
        device = str(_resident_runtime["device"])
        get_autocast_context = _resident_runtime["autocast_context"]
        generator = _resident_generator
    elif ckpt is None or cfg is None:
        # Preserve the original API behavior for callers that rely on their
        # installed my_robot defaults.
        from my_robot.pipeline.segment_demo import (
            DEFAULT_CFG,
            DEFAULT_CKPT,
            _ensure_sam2_on_syspath,
            _get_safe_autocast_context,
            _get_safe_torch_device,
        )
        ckpt = DEFAULT_CKPT if ckpt is None else ckpt
        cfg = DEFAULT_CFG if cfg is None else cfg
        _ensure_sam2_on_syspath()
        get_device = _get_safe_torch_device
        get_autocast_context = _get_safe_autocast_context
    else:
        _configure_sam2_from_checkpoint(ckpt)
        get_device = _safe_torch_device
        get_autocast_context = _safe_autocast_context
    if not resident:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

    started = time.perf_counter()
    image_rgb = image_bgr[..., ::-1].copy()
    h, w = image_rgb.shape[:2]
    roi_box = _resolve_norm_box((h, w), candidate_roi_norm)
    if not resident:
        device = get_device(torch)
    print(f"[sam2-first-frame] device={device} image={w}x{h}", flush=True)
    model_t0 = time.perf_counter()
    if resident:
        model_load_ms = 0.0
    else:
        sam2 = build_sam2(cfg, str(ckpt), device=device)
        model_load_ms = (time.perf_counter() - model_t0) * 1000.0
        generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=16,
            points_per_batch=128,
            pred_iou_thresh=0.65,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            min_mask_region_area=300,
            output_mode="binary_mask",
        )

    # This starts immediately before SAM2 consumes the image. AMG returns its
    # candidate list as one batch, so this is the earliest observable point at
    # which a candidate is available to this Python caller.
    image_candidate_t0 = time.perf_counter()
    generate_t0 = time.perf_counter()
    with torch.inference_mode(), get_autocast_context(torch, device):
        annotations = generator.generate(image_rgb)
    generate_ms = (time.perf_counter() - generate_t0) * 1000.0
    first_amg_candidate_ms = (time.perf_counter() - image_candidate_t0) * 1000.0

    candidates: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []

    for ann_index, ann in enumerate(annotations):
        raw_mask = np.asarray(ann["segmentation"], dtype=np.uint8)
        raw_area_px = int(np.count_nonzero(raw_mask > 0))
        mask, component_count, largest_component_area_px = _largest_connected_component(raw_mask)
        debug_item: dict[str, Any] = {
            "index": int(ann_index),
            "mask": mask,
            "accepted": False,
            "reject_reason": None,
            "predicted_iou": float(ann.get("predicted_iou", 0.0)),
            "stability_score": float(ann.get("stability_score", 0.0)),
            "raw_area_px": raw_area_px,
            "component_count": component_count,
            "largest_component_area_px": largest_component_area_px,
            "discarded_component_area_px": int(raw_area_px - largest_component_area_px),
        }

        def finish_debug(reason: str | None = None, *, accepted: bool = False) -> None:
            if return_debug:
                debug_item["accepted"] = bool(accepted)
                debug_item["reject_reason"] = reason
                debug_records.append(debug_item)

        score = _amg_score(mask, ann)
        if score <= -1e8:
            finish_debug("invalid_or_empty_mask")
            continue
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            finish_debug("empty_mask_bbox")
            continue
        area_px = int(np.count_nonzero(mask > 0))
        area_ratio = float(area_px) / float(max(1, h * w))
        span_x = float(bbox[2] - bbox[0]) / float(max(1, w))
        span_y = float(bbox[3] - bbox[1]) / float(max(1, h))
        debug_item.update(
            {
                "bbox": [float(v) for v in bbox],
                "amg_score": float(score),
                "area_px": int(area_px),
                "area_ratio": float(area_ratio),
                "bbox_span_x_ratio": float(span_x),
                "bbox_span_y_ratio": float(span_y),
            }
        )
        if (
            area_ratio > DEFAULT_MAX_MASK_AREA_RATIO
            or span_x > DEFAULT_MAX_BBOX_SPAN_X_RATIO
            or span_y > DEFAULT_MAX_BBOX_SPAN_Y_RATIO
        ):
            print(
                "[sam2-first-frame] reject oversized/merged mask: "
                f"area_ratio={area_ratio:.3f} bbox_span=({span_x:.3f},{span_y:.3f})"
            )
            finish_debug("oversized_or_merged_mask")
            continue
        roi_ratio = _mask_ratio_inside_box(mask, roi_box)
        debug_item["roi_mask_ratio"] = float(roi_ratio)
        if roi_box is not None and roi_ratio < float(candidate_roi_min_mask_ratio):
            finish_debug("roi_mask_ratio_too_low")
            continue
        points, labels = _sample_prompt_points(mask)
        candidates.append(
            {
                "source_index": int(ann_index),
                "mask": (mask > 0).astype(np.uint8) * 255,
                "box": [float(v) for v in bbox],
                "points": points,
                "labels": labels,
                "amg_score": float(score),
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
                "area_px": area_px,
                "area_ratio": area_ratio,
                "raw_area_px": raw_area_px,
                "component_count": component_count,
                "discarded_component_area_px": int(raw_area_px - area_px),
                "bbox_span_x_ratio": span_x,
                "bbox_span_y_ratio": span_y,
                "roi_mask_ratio": float(roi_ratio),
            }
        )
        finish_debug(None, accepted=True)

    if not candidates:
        raise RuntimeError("SAM2 AMG 没有生成有效首帧候选")

    candidates.sort(
        key=lambda item: (
            -int(item["area_px"]),
            -float(item["predicted_iou"]),
            -float(item["stability_score"]),
        )
    )
    selected = candidates[0]
    if return_debug:
        selected_source_index = int(selected["source_index"])
        for record in debug_records:
            record["selected"] = int(record["index"]) == selected_source_index
    original_mask = np.asarray(selected["mask"], dtype=np.uint8)
    selected_bbox = list(selected["box"])
    rotated_mask, rotated_bbox, rotated_bbox_params = _rotated_bbox_mask(original_mask)
    format_outputs: dict[str, np.ndarray] = {}
    if "mask" in requested_formats:
        format_outputs["mask"] = original_mask
    if "bboxmask" in requested_formats:
        format_outputs["bboxmask"] = _bbox_mask(original_mask.shape[:2], selected_bbox)
    if "rotated_bboxmask" in requested_formats:
        format_outputs["rotated_bboxmask"] = rotated_mask

    result = {
        **format_outputs,
        "formats": format_outputs,
        "return_formats": list(requested_formats),
        "bbox": selected_bbox,
        "rotated_bbox": rotated_bbox,
        "rotated_bbox_params": rotated_bbox_params,
        "points": [list(point) for point in selected["points"]],
        "labels": [int(label) for label in selected["labels"]],
        "candidate": selected,
        "candidates": candidates[: max(1, int(topk))],
        "raw_candidate_count": int(len(annotations)),
        "valid_candidate_count": int(len(candidates)),
        "timing_ms": {
            "model_load": float(model_load_ms),
            "generate": float(generate_ms),
            "total": float((time.perf_counter() - started) * 1000.0),
        },
    }
    if return_debug:
        result["debug"] = {
            "raw_annotations": debug_records,
            "raw_annotation_count": int(len(annotations)),
            "valid_candidate_count": int(len(candidates)),
            "selected_source_index": int(selected["source_index"]),
            "roi_box": None if roi_box is None else [float(v) for v in roi_box],
        }
    result["timing_ms"]["image_to_first_amg_candidate"] = float(first_amg_candidate_ms)
    result["timing_ms"]["image_to_selected_candidate"] = float(
        (time.perf_counter() - image_candidate_t0) * 1000.0
    )
    result["resident_model"] = bool(resident)
    result["backend"] = _normalize_backend(backend)
    return result


__all__ = [
    "start",
    "close",
    "status",
    "select_first_frame_mask",
    "select_first_frame_candidates",
    "segment_full_image_top_n",
    "validate_explicit_sam2_runtime",
    "SUPPORTED_BACKENDS",
]
