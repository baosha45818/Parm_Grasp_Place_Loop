"""SAM2-based segmentation of a recorded demo's video.

For each selected camera in the demo (wrist by default), this script:
  1. Lets the user browse frames and choose a seed frame where the workpiece
     is visible.
  2. Shows that seed frame in an OpenCV window for the user to click on the
     workpiece (left-click = foreground, right-click = background, 'r' = reset,
     ENTER = run SAM2, 'q' = quit).
  3. Runs SAM2 video predictor to propagate the mask through every frame.
  4. Saves the per-frame binary masks in a separate mask root, then plays an
     overlay review before accepting the episode.

Output layout (added to the mask root):
    <mask_root>/ep_0/
        wrist/masks/000000.png ...     # uint8 0/255, same H x W as RGB
        scene/masks/000000.png ...     # only when --cameras scene is selected
        seg_clicks.json                # seed frame + clicks per camera, for reproducibility

Requirements: run in an environment that has SAM2 installed.

Run:
    cd /home/haotianyu/Hao_projects/policy
    DISPLAY=:1 XAUTHORITY=$HOME/.Xauthority \\
    /mnt/data2/shurenbin/.conda/envs/dp/bin/python \\
        my_robot/pipeline/segment_demo.py \\
        --demo my_robot/pipeline/demos/demo_001

Run the next unprocessed episode under a prepared input root:
    python my_robot/pipeline/segment_demo.py \\
        --input-root /mnt/data1/zhuyiming/openvla-prepared-necessary \\
        --output-mask-root /mnt/data1/zhuyiming/openvla-prepared-necessary-masks

If you want non-interactive (re-segment with same clicks):
    --reuse-clicks   # reads seg_clicks.json from previous run

If you want automatic seeding:
    --auto-mode hybrid
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_CKPT = _REPO_ROOT / "models/sam2.1_hiera_base_plus.pt"
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# The vendored repository root is placed on sys.path; it contains the
# ``sam2`` package and its Hydra configuration package.
DEFAULT_SAM2_REPO = _REPO_ROOT / "vendor"
DEFAULT_YOLO_WEIGHTS = (
    _REPO_ROOT / "ultralytics" / "process_data" / "runs" / "extract_yolo_run1" / "weights" / "best.pt"
)
DEFAULT_ULTRALYTICS_REPO = _REPO_ROOT / "ultralytics"
DEFAULT_T_E_WRIST = _REPO_ROOT / "my_robot" / "calibration" / "T_e_wrist_cam.json"
DEFAULT_T_WORLD_SCENE = _REPO_ROOT / "my_robot" / "calibration" / "T_world_scene_cam.json"
DEFAULT_WRIST_TARGET_X = 0.73
DEFAULT_WRIST_TARGET_Y = 0.56
DEFAULT_WRIST_TARGET_RADIUS_PX = 28
# Estimated from 80 wrist seed frames near gripper-close: stable fingertip
# image locations for the left / right gripper jaws in the wrist camera.
DEFAULT_WRIST_GRIPPER_LEFT_TIP_X = 0.414
DEFAULT_WRIST_GRIPPER_LEFT_TIP_Y = 0.513
DEFAULT_WRIST_GRIPPER_RIGHT_TIP_X = 0.980
DEFAULT_WRIST_GRIPPER_RIGHT_TIP_Y = 0.506

_LEFT_ARROW_KEYS = {81, 2424832, 65361}
_RIGHT_ARROW_KEYS = {83, 2555904, 65363}
_ENTER_KEYS = {10, 13}


def _ensure_sam2_on_syspath() -> None:
    sam2_repo = DEFAULT_SAM2_REPO.resolve()
    if not sam2_repo.exists():
        raise SystemExit(f"SAM2 repo not found: {sam2_repo}")
    sam2_repo_str = str(sam2_repo)
    if sam2_repo_str not in sys.path:
        sys.path.insert(0, sam2_repo_str)
    try:
        import hydra  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "SAM2 dependencies are missing in the current interpreter. "
            "Please use /mnt/data2/shurenbin/.conda/envs/dp/bin/python "
            "or pass --python to the wrapper."
        ) from e


def _get_safe_torch_device(torch) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            return "cpu"
        device_idx = int(torch.cuda.current_device())
        if device_idx < 0 or device_idx >= device_count:
            return "cpu"
        torch.cuda.get_device_properties(device_idx)
        return "cuda"
    except Exception:
        return "cpu"


def _get_safe_autocast_context(torch, device: str):
    if device != "cuda":
        return nullcontext()
    try:
        if torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    except Exception:
        pass
    try:
        return torch.autocast("cuda", dtype=torch.float16)
    except Exception:
        return nullcontext()


# -------------------------------------------------- click capture UI ----------

def list_rgb_frames(rgb_dir: Path) -> list[Path]:
    files = sorted(rgb_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no rgb frames found in {rgb_dir}")
    return files


def choose_seed_frame_interactively(frame_paths: list[Path], role: str) -> int | None:
    """Browse frames and choose a seed frame for segmentation."""
    index = 0
    step_small = 1
    win = (
        f"segment_demo: choose seed for '{role}' "
        "(LEFT/RIGHT=prev/next, ENTER=select, q=quit)"
    )

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame = cv2.imread(str(frame_paths[index]), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit(f"failed to read frame: {frame_paths[index]}")

            vis = frame.copy()
            cv2.putText(
                vis,
                (
                    f"{role} seed frame {index + 1}/{len(frame_paths)}  "
                    "LEFT/RIGHT prev/next  ENTER select  q/ESC quit"
                ),
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"file={frame_paths[index].name}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win, vis)
            k = cv2.waitKeyEx(20)
            if k in _LEFT_ARROW_KEYS:
                index = max(0, index - step_small)
            elif k in _RIGHT_ARROW_KEYS:
                index = min(len(frame_paths) - 1, index + step_small)
            elif k in _ENTER_KEYS:
                return index
            elif k in (ord("q"), 27):
                raise SystemExit("aborted by user")
    finally:
        cv2.destroyWindow(win)


def collect_clicks_interactively(seed_frame_bgr: np.ndarray, role: str, seed_frame_idx: int
                                  ) -> tuple[list, list] | None:
    """Open a window, let user place positive/negative clicks on the workpiece.

    Returns (positive_xy_list, negative_xy_list). Pressing q / ESC aborts.
    """
    pos: list[list[int]] = []
    neg: list[list[int]] = []
    win = (
        f"segment_demo: click WORKPIECE in '{role}' seed={seed_frame_idx:06d} "
        "(LMB=fg, RMB=bg, r=reset, ENTER=confirm, q=quit)"
    )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pos.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            neg.append([x, y])

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = seed_frame_bgr.copy()
        for p in pos:
            cv2.circle(vis, tuple(p), 8, (0, 255, 0), -1)
            cv2.circle(vis, tuple(p), 8, (0, 0, 0), 1)
        for p in neg:
            cv2.circle(vis, tuple(p), 8, (0, 0, 255), -1)
            cv2.circle(vis, tuple(p), 8, (0, 0, 0), 1)
        cv2.putText(vis, f"{role} seed={seed_frame_idx:06d}: +{len(pos)} pos, -{len(neg)} neg",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            vis,
            "ENTER confirm  r reset  q/ESC quit",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        cv2.imshow(win, vis)

        k = cv2.waitKeyEx(20)
        if k == ord("r"):
            pos.clear()
            neg.clear()
        elif k in _ENTER_KEYS:
            if not pos:
                continue
            cv2.destroyWindow(win)
            return pos, neg
        elif k in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("aborted by user")


# -------------------------------------------------- auto seeding --------------

def _load_source_h5_from_demo(demo_dir: Path, explicit_source_h5: Path | None) -> Path | None:
    if explicit_source_h5 is not None:
        return explicit_source_h5
    meta_path = demo_dir / "source_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    episode_h5 = meta.get("episode_h5")
    if not episode_h5:
        return None
    source_h5 = Path(str(episode_h5))
    return source_h5 if source_h5.is_file() else None


def _choose_auto_seed_frame(
    source_h5: Path | None,
    n_frames: int,
    *,
    close_threshold: float,
    after_close_offset: int,
) -> int:
    if source_h5 is None:
        return max(0, min(n_frames - 1, int(round(0.75 * max(0, n_frames - 1)))))

    import h5py

    with h5py.File(source_h5, "r") as root:
        key = "observation/gripper_position"
        if key not in root:
            return max(0, min(n_frames - 1, int(round(0.75 * max(0, n_frames - 1)))))
        grip = np.asarray(root[key][:], dtype=np.float32).reshape(-1)

    close_candidates = np.where(grip > float(close_threshold))[0]
    if len(close_candidates) == 0:
        return max(0, min(n_frames - 1, int(round(0.75 * max(0, n_frames - 1)))))
    seed_frame = int(close_candidates[0]) + int(after_close_offset)
    return max(0, min(n_frames - 1, seed_frame))


def _get_first_close_frame(source_h5: Path | None, *, close_threshold: float) -> int | None:
    if source_h5 is None:
        return None

    import h5py

    with h5py.File(source_h5, "r") as root:
        key = "observation/gripper_position"
        if key not in root:
            return None
        grip = np.asarray(root[key][:], dtype=np.float32).reshape(-1)
    close_candidates = np.where(grip > float(close_threshold))[0]
    if len(close_candidates) == 0:
        return None
    return int(close_candidates[0])


def _choose_lifted_scene_seed_frame(
    source_h5: Path | None,
    n_frames: int,
    *,
    close_threshold: float,
    after_close_offset: int,
    lift_delta_m: float,
    lift_search_frames: int,
) -> int:
    close_frame = _get_first_close_frame(source_h5, close_threshold=close_threshold)
    if source_h5 is None or close_frame is None:
        return _choose_auto_seed_frame(
            source_h5,
            n_frames,
            close_threshold=close_threshold,
            after_close_offset=after_close_offset,
        )

    import h5py

    with h5py.File(source_h5, "r") as root:
        key = "observation/end_effector_pose_6dof"
        if key not in root:
            return _choose_auto_seed_frame(
                source_h5,
                n_frames,
                close_threshold=close_threshold,
                after_close_offset=after_close_offset,
            )
        ee = np.asarray(root[key][:], dtype=np.float32)

    base_z = float(ee[min(close_frame, len(ee) - 1), 2])
    start = min(len(ee) - 1, max(0, close_frame + int(after_close_offset)))
    end = min(len(ee), start + max(1, int(lift_search_frames)))
    for idx in range(start, end):
        if float(ee[idx, 2]) >= base_z + float(lift_delta_m):
            return max(0, min(n_frames - 1, idx))
    return max(0, min(n_frames - 1, start))


def _ensure_ultralytics_importable(repo_dir: Path | None) -> None:
    if repo_dir is None:
        return
    resolved = str(repo_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _detect_best_yolo_box(
    frame_paths: list[Path],
    *,
    source_h5: Path | None,
    weights_path: Path,
    repo_dir: Path | None,
    conf_thresh: float,
    imgsz: int,
    class_id: int | None,
    device: str | None,
    close_threshold: float,
    close_window_before: int,
    close_window_after: int,
) -> tuple[int, list[float]] | None:
    if not frame_paths:
        return None
    if not weights_path.is_file():
        raise SystemExit(f"YOLO weights not found: {weights_path}")

    _ensure_ultralytics_importable(repo_dir)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "failed to import ultralytics; pass a valid --ultralytics-repo-dir or install ultralytics"
        ) from exc

    model = YOLO(str(weights_path))
    close_frame = _get_first_close_frame(source_h5, close_threshold=close_threshold)
    if close_frame is not None:
        start_idx = max(0, int(close_frame) - int(close_window_before))
        end_idx = min(len(frame_paths), int(close_frame) + int(close_window_after) + 1)
        candidate_indices = list(range(start_idx, end_idx))
    else:
        candidate_indices = list(range(len(frame_paths)))

    candidates: list[tuple[int, float, list[float]]] = []
    has_cv2_cuda = hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
    for frame_idx in candidate_indices:
        image_bgr = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        predict_kwargs = {
            "source": [image_bgr],
            "conf": float(conf_thresh),
            "imgsz": int(imgsz),
            "device": device or ("cuda:0" if has_cv2_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"),
            "verbose": False,
        }
        if class_id is not None:
            predict_kwargs["classes"] = [int(class_id)]
        results = model.predict(**predict_kwargs)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        if xyxy.shape[0] == 0:
            continue
        best_idx = int(np.argmax(confs))
        score = float(confs[best_idx])
        box = xyxy[best_idx].astype(np.float32).tolist()
        candidates.append((frame_idx, score, box))

    if not candidates:
        return None

    if close_frame is not None:
        frame_idx, _, box = min(
            candidates,
            key=lambda item: (
                abs(item[0] - close_frame),
                -item[0],
                -item[1],
            ),
        )
    else:
        _, frame_idx, box = max(candidates, key=lambda item: item[1])
    return int(frame_idx), [float(v) for v in box]


def _detect_best_single_object_scene_box(
    frame_paths: list[Path],
    *,
    source_h5: Path | None,
    weights_path: Path,
    repo_dir: Path | None,
    conf_thresh: float,
    imgsz: int,
    class_id: int | None,
    device: str | None,
    close_threshold: float,
    close_window_before: int,
) -> tuple[int, list[float]] | None:
    """Find a scene box for simple single-object episodes.

    Unlike the generic YOLO path, this intentionally looks in the early part
    of the episode, before the grasp starts to happen. For the single-object
    episodes after ~140, the workpiece is usually cleanly visible on the table
    at the beginning, while the old close-frame-centered logic seeds too late.
    """
    if not frame_paths:
        return None
    if not weights_path.is_file():
        raise SystemExit(f"YOLO weights not found: {weights_path}")

    _ensure_ultralytics_importable(repo_dir)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "failed to import ultralytics; pass a valid --ultralytics-repo-dir or install ultralytics"
        ) from exc

    model = YOLO(str(weights_path))
    close_frame = _get_first_close_frame(source_h5, close_threshold=close_threshold)
    if close_frame is not None:
        early_end = max(1, int(close_frame) - max(2, int(close_window_before)))
    else:
        early_end = max(1, int(round(0.45 * len(frame_paths))))
    candidate_indices = list(range(min(len(frame_paths), early_end)))
    if not candidate_indices:
        candidate_indices = [0]

    candidates: list[tuple[int, float, float, list[float]]] = []
    has_cv2_cuda = hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
    for frame_idx in candidate_indices:
        image_bgr = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        predict_kwargs = {
            "source": [image_bgr],
            "conf": float(conf_thresh),
            "imgsz": int(imgsz),
            "device": device or ("cuda:0" if has_cv2_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"),
            "verbose": False,
        }
        if class_id is not None:
            predict_kwargs["classes"] = [int(class_id)]
        results = model.predict(**predict_kwargs)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        if xyxy.shape[0] == 0:
            continue
        best_idx = int(np.argmax(confs))
        score = float(confs[best_idx])
        box = xyxy[best_idx].astype(np.float32).tolist()
        x0, y0, x1, y1 = [float(v) for v in box]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        candidates.append((frame_idx, score, area, [float(v) for v in box]))

    if not candidates:
        return None

    best_score = max(score for _, score, _, _ in candidates)
    shortlisted = [
        item for item in candidates
        if item[1] >= max(float(conf_thresh), best_score - 0.05)
    ]
    if not shortlisted:
        shortlisted = candidates

    frame_idx, _, _, box = sorted(
        shortlisted,
        key=lambda item: (-item[2], item[0], -item[1]),
    )[0]
    return int(frame_idx), [float(v) for v in box]


def _score_amg_mask(
    mask: np.ndarray,
    ann: dict,
    *,
    image_shape: tuple[int, int],
    target_xy: tuple[float, float],
    target_radius_px: float,
    require_target: bool,
) -> float:
    h, w = image_shape
    area = float(ann.get("area", 0.0))
    if area <= 0:
        return -1e9
    area_ratio = area / float(h * w)
    if area_ratio < 0.002 or area_ratio > 0.30:
        return -1e9

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return -1e9

    cx = float(xs.mean())
    cy = float(ys.mean())
    dist = np.hypot(cx - target_xy[0], cy - target_xy[1])
    dist_norm = dist / max(1.0, np.hypot(w, h))

    touches_left = float(xs.min() <= 2)
    touches_right = float(xs.max() >= w - 3)
    touches_top = float(ys.min() <= 2)
    touches_bottom = float(ys.max() >= h - 3)
    target_x = int(np.clip(round(target_xy[0]), 0, w - 1))
    target_y = int(np.clip(round(target_xy[1]), 0, h - 1))
    target_hit = float(mask[target_y, target_x] > 0)

    yy, xx = np.ogrid[:h, :w]
    target_disk = ((xx - target_xy[0]) ** 2 + (yy - target_xy[1]) ** 2) <= float(target_radius_px) ** 2
    target_overlap = float(np.count_nonzero((mask > 0) & target_disk))
    target_overlap_ratio = target_overlap / max(1.0, float(np.count_nonzero(target_disk)))

    if require_target and target_hit <= 0.0 and target_overlap_ratio <= 0.02:
        return -1e9

    score = 0.0
    score += 2.0 * float(ann.get("predicted_iou", 0.0))
    score += 2.0 * float(ann.get("stability_score", 0.0))
    score += 6.0 * target_hit
    score += 4.0 * target_overlap_ratio
    score -= 4.0 * dist_norm
    score -= 2.0 * touches_bottom
    score -= 1.5 * touches_left
    score -= 1.5 * touches_right
    score -= 0.5 * touches_top
    return score


def _generate_amg_seed_mask(
    image_bgr: np.ndarray,
    *,
    ckpt: str,
    cfg: str,
    role: str,
    wrist_target_xy_norm: tuple[float, float],
    wrist_target_radius_px: int,
) -> tuple[np.ndarray, list[float]] | None:
    _ensure_sam2_on_syspath()
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    image_rgb = image_bgr[..., ::-1].copy()
    h, w = image_rgb.shape[:2]
    if role == "wrist":
        target_xy = (
            float(wrist_target_xy_norm[0]) * w,
            float(wrist_target_xy_norm[1]) * h,
        )
        target_radius_px = int(wrist_target_radius_px)
        require_target = True

        # For wrist, a positive point inside the grasp center is a better cue
        # for the whole workpiece than AMG's local region proposals.
        point_generated = _generate_point_seed_mask(
            image_bgr,
            ckpt=ckpt,
            cfg=cfg,
            target_xy=target_xy,
            target_radius_px=target_radius_px,
        )
        if point_generated is not None:
            return point_generated
    else:
        target_xy = (0.50 * w, 0.60 * h)
        target_radius_px = max(16, int(round(0.04 * max(h, w))))
        require_target = False

    device = _get_safe_torch_device(torch)
    sam2 = build_sam2(cfg, ckpt, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        min_mask_region_area=300,
        output_mode="binary_mask",
    )

    with torch.inference_mode(), _get_safe_autocast_context(torch, device):
        anns = mask_generator.generate(image_rgb)

    best_score = None
    best_mask = None
    best_box = None
    for ann in anns:
        mask = ann["segmentation"].astype(np.uint8)
        score = _score_amg_mask(
            mask,
            ann,
            image_shape=(h, w),
            target_xy=target_xy,
            target_radius_px=target_radius_px,
            require_target=require_target,
        )
        if best_score is None or score > best_score:
            best_score = score
            x, y, bw, bh = ann["bbox"]
            best_box = [float(x), float(y), float(x + bw), float(y + bh)]
            best_mask = mask

    if best_mask is None or best_box is None:
        return None
    return best_mask, best_box


def _pick_best_mask_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    image_shape: tuple[int, int],
    target_xy: tuple[float, float],
    target_radius_px: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[np.ndarray, list[float]] | None:
    h, w = image_shape
    target_x = int(np.clip(round(target_xy[0]), 0, w - 1))
    target_y = int(np.clip(round(target_xy[1]), 0, h - 1))
    yy, xx = np.ogrid[:h, :w]
    target_disk = ((xx - target_xy[0]) ** 2 + (yy - target_xy[1]) ** 2) <= float(target_radius_px) ** 2

    best_idx = None
    best_key = None
    for idx, mask in enumerate(masks):
        mask_u8 = mask.astype(np.uint8)
        area = int(mask_u8.sum())
        if area <= 0:
            continue
        area_ratio = area / float(h * w)
        if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
            continue
        mask_box = _mask_to_xyxy(mask_u8)
        if mask_box is None:
            continue
        x0, y0, x1, y1 = mask_box
        touches_border = int(x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2)
        target_hit = int(mask_u8[target_y, target_x] > 0)
        target_overlap = float(np.count_nonzero((mask_u8 > 0) & target_disk))
        target_overlap_ratio = target_overlap / max(1.0, float(np.count_nonzero(target_disk)))
        if target_hit <= 0 and target_overlap_ratio <= 0.02:
            continue

        # Prefer masks that keep the target point but also cover a full object.
        key = (
            target_hit,
            target_overlap_ratio,
            -touches_border,
            area,
            float(scores[idx]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx

    if best_idx is None:
        return None

    best_mask = masks[best_idx].astype(np.uint8)
    best_box = _mask_to_xyxy(best_mask)
    if best_box is None:
        return None
    return best_mask, best_box


def _build_wrist_prompt_points(
    image_shape: tuple[int, int],
    target_xy: tuple[float, float],
    target_radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    tx, ty = float(target_xy[0]), float(target_xy[1])
    r = max(12.0, float(target_radius_px))

    # For the wrist view, use a cross-shaped positive prompt template:
    # stretch points along the long axis of the part, but also add short-axis
    # support so SAM2 grows the whole part body instead of only a thin strip.
    pos_points = [
        [tx, ty],
        [tx, ty - 1.0 * r],
        [tx, ty - 2.1 * r],
        [tx, ty + 1.0 * r],
        [tx, ty + 2.1 * r],
        [tx - 0.9 * r, ty],
        [tx + 0.9 * r, ty],
        [tx - 0.9 * r, ty - 1.0 * r],
        [tx + 0.9 * r, ty - 1.0 * r],
        [tx - 0.9 * r, ty + 1.0 * r],
        [tx + 0.9 * r, ty + 1.0 * r],
    ]
    neg_points = [
        # Keep negative prompts, but move them farther out so they suppress
        # the gripper/background without eating into the object's edge.
        [tx - 2.3 * r, ty + 0.1 * r],
        [tx + 2.3 * r, ty + 0.1 * r],
        [tx - 2.0 * r, ty + 2.1 * r],
        [tx + 2.0 * r, ty + 2.1 * r],
        [tx - 1.6 * r, ty + 3.0 * r],
        [tx + 1.6 * r, ty + 3.0 * r],
    ]

    all_points = []
    all_labels = []
    for x, y in pos_points:
        if 0 <= x < w and 0 <= y < h:
            all_points.append([float(x), float(y)])
            all_labels.append(1)
    for x, y in neg_points:
        if 0 <= x < w and 0 <= y < h:
            all_points.append([float(x), float(y)])
            all_labels.append(0)

    if not all_points:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.asarray(all_points, dtype=np.float32), np.asarray(all_labels, dtype=np.int32)


def _clip_point_to_image(point_xy: tuple[float, float], image_shape: tuple[int, int]) -> tuple[float, float]:
    h, w = image_shape
    return (
        float(np.clip(point_xy[0], 0.0, float(w - 1))),
        float(np.clip(point_xy[1], 0.0, float(h - 1))),
    )


def _refine_gripper_tip_from_dark_patch(
    image_bgr: np.ndarray,
    *,
    tip_xy: tuple[float, float],
    window_hw: tuple[int, int] = (70, 80),
) -> tuple[float, float]:
    h, w = image_bgr.shape[:2]
    cx, cy = tip_xy
    half_w, half_h = int(window_hw[0]), int(window_hw[1])
    x0 = max(0, int(round(cx)) - half_w)
    x1 = min(w, int(round(cx)) + half_w)
    y0 = max(0, int(round(cy)) - half_h)
    y1 = min(h, int(round(cy)) + half_h)
    if x1 <= x0 or y1 <= y0:
        return _clip_point_to_image(tip_xy, (h, w))

    roi = image_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    # Gripper fingers are matte black in wrist view; combine low value with
    # low-to-mid saturation to avoid bright metal highlights.
    mask = ((v < 95) & (s < 140)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    best_key = None
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < 50:
            continue
        ys, xs = np.where(labels == idx)
        if len(xs) == 0:
            continue
        top_y = int(ys.min())
        top_band = xs[ys <= top_y + 4]
        if len(top_band) == 0:
            continue
        tip_x = float(top_band.mean()) + x0
        tip_y = float(top_y) + y0
        dist = float(np.hypot(tip_x - cx, tip_y - cy))
        key = (-area, dist)
        if best_key is None or key < best_key:
            best_key = key
            best = (tip_x, tip_y)

    if best is None:
        return _clip_point_to_image(tip_xy, (h, w))
    return _clip_point_to_image(best, (h, w))


def _estimate_wrist_gripper_tips(
    image_bgr: np.ndarray,
    *,
    left_tip_xy_norm: tuple[float, float] = (
        DEFAULT_WRIST_GRIPPER_LEFT_TIP_X,
        DEFAULT_WRIST_GRIPPER_LEFT_TIP_Y,
    ),
    right_tip_xy_norm: tuple[float, float] = (
        DEFAULT_WRIST_GRIPPER_RIGHT_TIP_X,
        DEFAULT_WRIST_GRIPPER_RIGHT_TIP_Y,
    ),
) -> tuple[tuple[float, float], tuple[float, float]]:
    h, w = image_bgr.shape[:2]
    left_guess = (left_tip_xy_norm[0] * w, left_tip_xy_norm[1] * h)
    right_guess = (right_tip_xy_norm[0] * w, right_tip_xy_norm[1] * h)
    left_tip = _refine_gripper_tip_from_dark_patch(image_bgr, tip_xy=left_guess)
    right_tip = _refine_gripper_tip_from_dark_patch(image_bgr, tip_xy=right_guess)
    return left_tip, right_tip


def _build_wrist_gripper_axis_prompt_points(
    image_shape: tuple[int, int],
    *,
    left_tip_xy: tuple[float, float],
    right_tip_xy: tuple[float, float],
    target_xy: tuple[float, float],
    target_radius_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    left = np.asarray(left_tip_xy, dtype=np.float32)
    right = np.asarray(right_tip_xy, dtype=np.float32)
    center = 0.5 * (left + right)
    axis = right - left
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-3:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    axis_unit = axis / axis_norm
    normal = np.asarray([-axis_unit[1], axis_unit[0]], dtype=np.float32)
    target_vec = np.asarray(target_xy, dtype=np.float32) - center
    if float(np.dot(normal, target_vec)) < 0.0:
        normal = -normal

    r = max(12.0, float(target_radius_px))
    axis_side = max(0.18 * axis_norm, 0.8 * r)
    pos_points = [
        center + normal * (0.6 * r),
        center + normal * (1.4 * r),
        center + normal * (2.2 * r),
        center + normal * (1.2 * r) - axis_unit * axis_side,
        center + normal * (1.2 * r) + axis_unit * axis_side,
        center + normal * (2.0 * r) - axis_unit * axis_side,
        center + normal * (2.0 * r) + axis_unit * axis_side,
    ]
    neg_points = [
        center - normal * (0.8 * r),
        left - normal * (0.4 * r),
        right - normal * (0.4 * r),
        left + axis_unit * (0.6 * r),
        right - axis_unit * (0.6 * r),
        center + normal * (1.4 * r) - axis_unit * (2.1 * axis_side),
        center + normal * (1.4 * r) + axis_unit * (2.1 * axis_side),
    ]

    all_points = []
    all_labels = []
    for point in pos_points:
        x, y = point.tolist()
        if 0 <= x < w and 0 <= y < h:
            all_points.append([float(x), float(y)])
            all_labels.append(1)
    for point in neg_points:
        x, y = point.tolist()
        if 0 <= x < w and 0 <= y < h:
            all_points.append([float(x), float(y)])
            all_labels.append(0)
    if not all_points:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.asarray(all_points, dtype=np.float32), np.asarray(all_labels, dtype=np.int32)


def _generate_wrist_gripper_axis_seed_mask(
    image_bgr: np.ndarray,
    *,
    ckpt: str,
    cfg: str,
    wrist_target_xy_norm: tuple[float, float],
    wrist_target_radius_px: int,
) -> tuple[np.ndarray, list[float], dict] | None:
    _ensure_sam2_on_syspath()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image_rgb = image_bgr[..., ::-1].copy()
    h, w = image_rgb.shape[:2]
    target_xy = (
        float(wrist_target_xy_norm[0]) * w,
        float(wrist_target_xy_norm[1]) * h,
    )
    left_tip_xy, right_tip_xy = _estimate_wrist_gripper_tips(image_bgr)
    point_coords, point_labels = _build_wrist_gripper_axis_prompt_points(
        (h, w),
        left_tip_xy=left_tip_xy,
        right_tip_xy=right_tip_xy,
        target_xy=target_xy,
        target_radius_px=wrist_target_radius_px,
    )
    if point_coords.shape[0] == 0:
        return None

    device = _get_safe_torch_device(torch)
    sam2 = build_sam2(cfg, ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    with torch.inference_mode(), _get_safe_autocast_context(torch, device):
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    picked = _pick_best_mask_candidate(
        masks,
        scores,
        image_shape=(h, w),
        target_xy=target_xy,
        target_radius_px=float(wrist_target_radius_px),
        min_area_ratio=0.003,
        max_area_ratio=0.20,
    )
    if picked is None:
        return None
    seed_mask, best_box = picked
    debug = {
        "left_tip_xy": [float(left_tip_xy[0]), float(left_tip_xy[1])],
        "right_tip_xy": [float(right_tip_xy[0]), float(right_tip_xy[1])],
    }
    return seed_mask, best_box, debug


def _generate_point_seed_mask(
    image_bgr: np.ndarray,
    *,
    ckpt: str,
    cfg: str,
    target_xy: tuple[float, float],
    target_radius_px: int,
) -> tuple[np.ndarray, list[float]] | None:
    _ensure_sam2_on_syspath()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image_rgb = image_bgr[..., ::-1].copy()
    h, w = image_rgb.shape[:2]
    point_coords, point_labels = _build_wrist_prompt_points(
        (h, w),
        target_xy,
        target_radius_px,
    )
    if point_coords.shape[0] == 0:
        return None

    device = _get_safe_torch_device(torch)
    sam2 = build_sam2(cfg, ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    with torch.inference_mode(), _get_safe_autocast_context(torch, device):
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    return _pick_best_mask_candidate(
        masks,
        scores,
        image_shape=(h, w),
        target_xy=target_xy,
        target_radius_px=float(target_radius_px),
        min_area_ratio=0.003,
        max_area_ratio=0.20,
    )


def _mask_to_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _expand_xyxy(box_xyxy: list[float], image_shape: tuple[int, int], scale: float) -> list[float]:
    h, w = image_shape
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(2.0, (x1 - x0) * float(scale))
    bh = max(2.0, (y1 - y0) * float(scale))
    out = [
        max(0.0, cx - 0.5 * bw),
        max(0.0, cy - 0.5 * bh),
        min(float(w - 1), cx + 0.5 * bw),
        min(float(h - 1), cy + 0.5 * bh),
    ]
    return out


def _load_transform_matrix(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["T"], dtype=np.float64)


def _transform_points(points: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    return points @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def _project_world_points_to_scene(points_world: np.ndarray, T_world_scene: np.ndarray, K: np.ndarray) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    T_scene_world = np.linalg.inv(T_world_scene)
    pts_scene = _transform_points(points_world, T_scene_world)
    valid = np.isfinite(pts_scene).all(axis=1) & (pts_scene[:, 2] > 1e-4)
    pts_scene = pts_scene[valid]
    if pts_scene.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    uv = np.empty((pts_scene.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pts_scene[:, 0] / pts_scene[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pts_scene[:, 1] / pts_scene[:, 2] + K[1, 2]
    valid = np.isfinite(uv).all(axis=1)
    return uv[valid]


def _estimate_scene_box_from_wrist_projection(
    demo_dir: Path,
    source_h5: Path | None,
    *,
    frame_idx: int,
    T_e_wrist_path: Path,
    T_world_scene_path: Path,
) -> list[float] | None:
    if source_h5 is None:
        return None

    wrist_mask_path = demo_dir / "wrist" / "masks" / f"{frame_idx:06d}.png"
    wrist_mask = cv2.imread(str(wrist_mask_path), cv2.IMREAD_GRAYSCALE)
    if wrist_mask is None:
        return None

    import h5py
    from my_robot.arm.rokae_bridge import pose_to_T

    T_e_wrist = _load_transform_matrix(T_e_wrist_path)
    T_world_scene = _load_transform_matrix(T_world_scene_path)

    with h5py.File(source_h5, "r") as root:
        if "observation/exterior_camera_info_1_left/k" not in root:
            return None
        K_scene = np.asarray(
            root["observation/exterior_camera_info_1_left/k"][:],
            dtype=np.float64,
        ).reshape(3, 3)

        if "observation/wrist_pointcloud_xyz" in root:
            wrist_xyz = np.asarray(root["observation/wrist_pointcloud_xyz"][frame_idx], dtype=np.float64)
            valid = (
                (wrist_mask > 0)
                & np.isfinite(wrist_xyz).all(axis=2)
                & (wrist_xyz[..., 2] > 0.05)
                & (wrist_xyz[..., 2] < 0.5)
            )
            points_cam = wrist_xyz[valid]
        else:
            return None

        if points_cam.shape[0] == 0:
            return None

        # Trim some outliers from the propagated wrist mask before projection.
        q_lo = np.quantile(points_cam, 0.1, axis=0)
        q_hi = np.quantile(points_cam, 0.9, axis=0)
        keep = np.all((points_cam >= q_lo) & (points_cam <= q_hi), axis=1)
        points_cam = points_cam[keep]
        if points_cam.shape[0] == 0:
            return None

        if "observation/end_effector_pose_6dof" not in root:
            return None
        T_w_e = pose_to_T(np.asarray(root["observation/end_effector_pose_6dof"][frame_idx], dtype=np.float64))

    T_w_wrist = T_w_e @ T_e_wrist
    points_world = _transform_points(points_cam, T_w_wrist)
    uv = _project_world_points_to_scene(points_world, T_world_scene, K_scene)
    if uv.shape[0] == 0:
        return None

    q_lo = np.quantile(uv, 0.05, axis=0)
    q_hi = np.quantile(uv, 0.95, axis=0)
    x0, y0 = q_lo.tolist()
    x1, y1 = q_hi.tolist()
    if x1 <= x0 or y1 <= y0:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def _generate_seed_mask_from_box(
    image_bgr: np.ndarray,
    *,
    box_xyxy: list[float],
    ckpt: str,
    cfg: str,
    role: str,
) -> tuple[np.ndarray, list[float]] | None:
    _ensure_sam2_on_syspath()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image_rgb = image_bgr[..., ::-1].copy()
    h, w = image_rgb.shape[:2]
    expand_scale = 1.8 if role == "scene" else 1.4
    prompt_box = np.asarray(
        _expand_xyxy(box_xyxy, (h, w), scale=expand_scale),
        dtype=np.float32,
    )

    device = _get_safe_torch_device(torch)
    sam2 = build_sam2(cfg, ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    with torch.inference_mode(), _get_safe_autocast_context(torch, device):
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=prompt_box[None, :],
            multimask_output=True,
        )

    best_idx = None
    best_key = None
    for idx, mask in enumerate(masks):
        area = int(mask.sum())
        if area <= 0:
            continue
        area_ratio = area / float(h * w)
        if area_ratio < 0.0005 or area_ratio > 0.08:
            continue
        mask_box = _mask_to_xyxy(mask.astype(np.uint8))
        if mask_box is None:
            continue
        x0, y0, x1, y1 = mask_box
        touch_border = int(x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2)
        key = (
            -touch_border,
            area,
            float(scores[idx]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx

    if best_idx is None:
        return None

    best_mask = masks[best_idx].astype(np.uint8)
    best_box = _mask_to_xyxy(best_mask)
    if best_box is None:
        return None
    return best_mask, best_box


def _select_auto_prompt(
    frame_paths: list[Path],
    *,
    role: str,
    demo_dir: Path,
    source_h5: Path | None,
    auto_mode: str,
    ckpt: str,
    cfg: str,
    yolo_weights: Path,
    ultralytics_repo_dir: Path | None,
    yolo_conf_thresh: float,
    yolo_imgsz: int,
    yolo_class_id: int | None,
    yolo_device: str | None,
    gripper_close_threshold: float,
    after_close_offset: int,
    close_window_before: int,
    close_window_after: int,
    T_e_wrist_path: Path,
    T_world_scene_path: Path,
    scene_lift_delta_m: float,
    scene_lift_search_frames: int,
    wrist_target_xy_norm: tuple[float, float],
    wrist_target_radius_px: int,
) -> dict | None:
    if auto_mode == "hybrid":
        role_mode = "wrist_project_box" if role == "scene" else "gripper_axis"
    elif auto_mode == "single_object":
        role_mode = "single_object_scene_box" if role == "scene" else "gripper_axis"
    else:
        role_mode = auto_mode

    if role_mode == "single_object_scene_box":
        detected = _detect_best_single_object_scene_box(
            frame_paths,
            source_h5=source_h5,
            weights_path=yolo_weights,
            repo_dir=ultralytics_repo_dir,
            conf_thresh=yolo_conf_thresh,
            imgsz=yolo_imgsz,
            class_id=yolo_class_id,
            device=yolo_device,
            close_threshold=gripper_close_threshold,
            close_window_before=close_window_before,
        )
        if detected is None:
            role_mode = "wrist_project_box"
        else:
            seed_frame, box = detected
            seed_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
            if seed_image is None:
                raise SystemExit(f"failed to read seed frame {frame_paths[seed_frame]}")
            generated = _generate_seed_mask_from_box(
                seed_image,
                box_xyxy=box,
                ckpt=ckpt,
                cfg=cfg,
                role=role,
            )
            if generated is not None:
                seed_mask, refined_box = generated
                seed_mask_path = demo_dir / role / "seed_mask.png"
                cv2.imwrite(str(seed_mask_path), seed_mask * 255)
                return {
                    "seed_frame": int(seed_frame),
                    "pos": [],
                    "neg": [],
                    "box": [float(v) for v in refined_box],
                    "detected_box": [float(v) for v in box],
                    "seed_mask_path": str(seed_mask_path.relative_to(demo_dir)),
                    "auto_mode": "single_object_scene_box",
                }
            return {
                "seed_frame": int(seed_frame),
                "pos": [],
                "neg": [],
                "box": [float(v) for v in box],
                "auto_mode": "single_object_scene_box",
            }

    if role_mode == "wrist_project_box":
        seed_frame = _choose_lifted_scene_seed_frame(
            source_h5,
            len(frame_paths),
            close_threshold=gripper_close_threshold,
            after_close_offset=after_close_offset,
            lift_delta_m=scene_lift_delta_m,
            lift_search_frames=scene_lift_search_frames,
        )
        projected_box = _estimate_scene_box_from_wrist_projection(
            demo_dir,
            source_h5,
            frame_idx=seed_frame,
            T_e_wrist_path=T_e_wrist_path,
            T_world_scene_path=T_world_scene_path,
        )
        if projected_box is not None:
            seed_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
            if seed_image is None:
                raise SystemExit(f"failed to read seed frame {frame_paths[seed_frame]}")
            generated = _generate_seed_mask_from_box(
                seed_image,
                box_xyxy=projected_box,
                ckpt=ckpt,
                cfg=cfg,
                role=role,
            )
            if generated is not None:
                seed_mask, refined_box = generated
                seed_mask_path = demo_dir / role / "seed_mask.png"
                cv2.imwrite(str(seed_mask_path), seed_mask * 255)
                return {
                    "seed_frame": int(seed_frame),
                    "pos": [],
                    "neg": [],
                    "box": [float(v) for v in refined_box],
                    "projected_box": [float(v) for v in projected_box],
                    "seed_mask_path": str(seed_mask_path.relative_to(demo_dir)),
                    "auto_mode": "wrist_project_box",
                }
        if auto_mode == "hybrid":
            role_mode = "yolo_box"
        else:
            raise SystemExit(f"wrist-projected scene seed failed for role '{role}'")

    if role_mode == "yolo_box":
        detected = _detect_best_yolo_box(
            frame_paths,
            source_h5=source_h5,
            weights_path=yolo_weights,
            repo_dir=ultralytics_repo_dir,
            conf_thresh=yolo_conf_thresh,
            imgsz=yolo_imgsz,
            class_id=yolo_class_id,
            device=yolo_device,
            close_threshold=gripper_close_threshold,
            close_window_before=close_window_before,
            close_window_after=close_window_after,
        )
        if detected is None:
            if auto_mode == "hybrid":
                role_mode = "amg_center"
            else:
                raise SystemExit(f"YOLO failed to find a box for role '{role}'")
        else:
            seed_frame, box = detected
            seed_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
            if seed_image is None:
                raise SystemExit(f"failed to read seed frame {frame_paths[seed_frame]}")
            generated = _generate_seed_mask_from_box(
                seed_image,
                box_xyxy=box,
                ckpt=ckpt,
                cfg=cfg,
                role=role,
            )
            if generated is not None:
                seed_mask, refined_box = generated
                seed_mask_path = demo_dir / role / "seed_mask.png"
                cv2.imwrite(str(seed_mask_path), seed_mask * 255)
                return {
                    "seed_frame": int(seed_frame),
                    "pos": [],
                    "neg": [],
                    "box": [float(v) for v in refined_box],
                    "detected_box": [float(v) for v in box],
                    "seed_mask_path": str(seed_mask_path.relative_to(demo_dir)),
                    "auto_mode": "yolo_box",
                }
            return {
                "seed_frame": int(seed_frame),
                "pos": [],
                "neg": [],
                "box": [float(v) for v in box],
                "auto_mode": "yolo_box",
            }

    if role_mode == "gripper_axis":
        seed_frame = _choose_auto_seed_frame(
            source_h5,
            len(frame_paths),
            close_threshold=gripper_close_threshold,
            after_close_offset=after_close_offset,
        )
        seed_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
        if seed_image is None:
            raise SystemExit(f"failed to read seed frame {frame_paths[seed_frame]}")
        generated = _generate_wrist_gripper_axis_seed_mask(
            seed_image,
            ckpt=ckpt,
            cfg=cfg,
            wrist_target_xy_norm=wrist_target_xy_norm,
            wrist_target_radius_px=wrist_target_radius_px,
        )
        if generated is None:
            if auto_mode == "hybrid":
                role_mode = "amg_center"
            else:
                raise SystemExit(f"gripper-axis seed generation failed for role '{role}'")
        else:
            seed_mask, box, debug = generated
            seed_mask_path = demo_dir / role / "seed_mask.png"
            cv2.imwrite(str(seed_mask_path), seed_mask * 255)
            return {
                "seed_frame": int(seed_frame),
                "pos": [],
                "neg": [],
                "box": [float(v) for v in box],
                "seed_mask_path": str(seed_mask_path.relative_to(demo_dir)),
                "auto_mode": "gripper_axis",
                "wrist_target_xy_norm": list(wrist_target_xy_norm) if role == "wrist" else None,
                "gripper_axis_debug": debug,
            }

    if role_mode == "amg_center":
        seed_frame = _choose_auto_seed_frame(
            source_h5,
            len(frame_paths),
            close_threshold=gripper_close_threshold,
            after_close_offset=after_close_offset,
        )
        seed_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
        if seed_image is None:
            raise SystemExit(f"failed to read seed frame {frame_paths[seed_frame]}")
        generated = _generate_amg_seed_mask(
            seed_image,
            ckpt=ckpt,
            cfg=cfg,
            role=role,
            wrist_target_xy_norm=wrist_target_xy_norm,
            wrist_target_radius_px=wrist_target_radius_px,
        )
        if generated is None:
            raise SystemExit(f"SAM2 automatic mask generation failed for role '{role}'")
        seed_mask, box = generated
        seed_mask_path = demo_dir / role / "seed_mask.png"
        cv2.imwrite(str(seed_mask_path), seed_mask * 255)
        return {
            "seed_frame": int(seed_frame),
            "pos": [],
            "neg": [],
            "box": [float(v) for v in box],
            "seed_mask_path": str(seed_mask_path.relative_to(demo_dir)),
            "auto_mode": "amg_center",
            "wrist_target_xy_norm": list(wrist_target_xy_norm) if role == "wrist" else None,
        }

    raise SystemExit(f"unknown auto mode: {auto_mode}")


# -------------------------------------------------- SAM2 wrapper --------------

def run_sam2_on_frames(
    frames_paths: list[Path],
    pos_xy: list,
    neg_xy: list,
    *,
    ckpt: str,
    cfg: str,
    seed_frame: int,
    box_xyxy: list[float] | None = None,
    seed_mask: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Build SAM2 video predictor, seed with clicks, propagate, return per-frame
    binary masks {frame_idx: uint8 array}.

    Frames are passed as a list of file paths in their original directory; SAM2
    will load them itself. We rely on the caller to provide frames as JPGs
    named NNNNN.jpg (SAM2 limitation), which is handled by `_stage_frames_as_jpg`.
    """
    _ensure_sam2_on_syspath()
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = _get_safe_torch_device(torch)
    print(f"[sam2] device={device}, frames={len(frames_paths)}")

    pos_arr = np.asarray(pos_xy, dtype=np.float32) if pos_xy else np.zeros((0, 2), np.float32)
    neg_arr = np.asarray(neg_xy, dtype=np.float32) if neg_xy else np.zeros((0, 2), np.float32)
    has_points = (len(pos_arr) + len(neg_arr)) > 0
    point_coords = np.concatenate([pos_arr, neg_arr], axis=0) if has_points else None
    point_labels = (
        np.concatenate(
            [
                np.ones(len(pos_arr), np.int32),
                np.zeros(len(neg_arr), np.int32),
            ]
        )
        if has_points
        else None
    )

    frames_dir = frames_paths[0].parent
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)

    masks: dict[int, np.ndarray] = {}
    with torch.inference_mode(), _get_safe_autocast_context(torch, device):
        state = predictor.init_state(video_path=str(frames_dir))
        if seed_mask is not None:
            _, _, mask_logits = predictor.add_new_mask(
                inference_state=state,
                frame_idx=seed_frame,
                obj_id=1,
                mask=seed_mask.astype(bool),
            )
        else:
            _, _, mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seed_frame,
                obj_id=1,
                points=point_coords,
                labels=point_labels,
                box=box_xyxy,
            )
        seed_mask = (mask_logits[0, 0].cpu().numpy() > 0).astype(np.uint8)
        print(
            f"[sam2] seed frame={seed_frame} "
            f"seed mask pixels: {int(seed_mask.sum())} / {seed_mask.size}"
        )

        for fidx, oids, mlogits in predictor.propagate_in_video(state):
            m = (mlogits[0, 0] > 0).cpu().numpy().astype(np.uint8)
            masks[int(fidx)] = m
        if seed_frame > 0:
            for fidx, oids, mlogits in predictor.propagate_in_video(state, reverse=True):
                m = (mlogits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                masks.setdefault(int(fidx), m)
    return masks


def _stage_frames_as_jpg(src_dir: Path, dst_dir: Path) -> list[Path]:
    """Copy/convert PNG frames from src_dir into dst_dir as NNNNN.jpg files.

    SAM2's init_state reads frames from a directory of JPG files named in
    a strict 5-digit lexicographic order. Our recorder writes 6-digit PNGs.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(src_dir.glob("*.png"))
    out_paths = []
    for i, src in enumerate(src_files):
        # PIL handles PNG -> JPG conversion robustly
        img = Image.open(src).convert("RGB")
        out = dst_dir / f"{i:05d}.jpg"
        img.save(out, quality=92)
        out_paths.append(out)
    return out_paths


def _render_overlay_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    alpha: float = 0.45,
) -> np.ndarray:
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = frame.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool].astype(np.float32)
        + alpha * np.asarray([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    cv2.putText(
        overlay,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    return overlay


def _review_masks_interactively(
    rgb_dir: Path,
    masks_dir: Path,
    *,
    role: str,
    episode_name: str,
    fps: float = 8.0,
) -> str:
    """Play mask overlay once, then ask whether to accept, retry, or quit."""
    frame_paths = sorted(rgb_dir.glob("*.png"))
    if not frame_paths:
        print(f"[seg] WARNING: no rgb frames for review: {rgb_dir}")
        return "retry"

    delay_ms = max(1, int(round(1000.0 / max(1.0, fps))))
    win = (
        f"review {episode_name}/{role}: "
        "ENTER accept+next, s accept+stop, r retry, q discard+quit"
    )
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    last_vis = None
    try:
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            mask = cv2.imread(str(masks_dir / frame_path.name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            last_vis = _render_overlay_frame(
                frame,
                mask,
                label=f"{episode_name}/{role} frame {idx:06d}",
            )
            cv2.imshow(win, last_vis)
            key = cv2.waitKeyEx(delay_ms)
            if key in _ENTER_KEYS:
                return "accept_next"
            if key == ord("s"):
                return "accept_stop"
            if key == ord("r"):
                return "retry"
            if key in (ord("q"), 27):
                return "quit"

        if last_vis is None:
            return "retry"

        while True:
            prompt = last_vis.copy()
            cv2.putText(
                prompt,
                "ENTER accept+next   s accept+stop   r retry   q discard+quit",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win, prompt)
            key = cv2.waitKeyEx(20)
            if key in _ENTER_KEYS:
                return "accept_next"
            if key == ord("s"):
                return "accept_stop"
            if key == ord("r"):
                return "retry"
            if key in (ord("q"), 27):
                return "quit"
    finally:
        cv2.destroyWindow(win)


# ------------------------------------------------ demo discovery -------------

def _ep_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("ep_"):
        try:
            return int(name[3:]), name
        except ValueError:
            pass
    return 10**12, name


def _has_camera_rgb_dir(demo_dir: Path, cameras: list[str]) -> bool:
    return any((demo_dir / role / "rgb").is_dir() for role in cameras)


def _mask_episode_dir(mask_root: Path, demo_dir: Path) -> Path:
    return mask_root / demo_dir.name


def _camera_masks_complete(demo_dir: Path, mask_demo_dir: Path, role: str) -> bool:
    rgb_dir = demo_dir / role / "rgb"
    masks_dir = mask_demo_dir / role / "masks"
    if not rgb_dir.is_dir():
        return True

    frame_paths = sorted(rgb_dir.glob("*.png"))
    if not frame_paths:
        return True
    if not masks_dir.is_dir():
        return False

    for frame_path in frame_paths:
        if not (masks_dir / frame_path.name).is_file():
            return False
    return True


def _demo_complete_for_cameras(demo_dir: Path, mask_root: Path, cameras: list[str]) -> bool:
    mask_demo_dir = _mask_episode_dir(mask_root, demo_dir)
    return all(_camera_masks_complete(demo_dir, mask_demo_dir, role) for role in cameras)


def _find_next_unprocessed_demo(prepared_root: Path, mask_root: Path, cameras: list[str]) -> Path:
    if not prepared_root.is_dir():
        raise SystemExit(f"input root not found: {prepared_root}")

    episode_dirs = sorted(
        [path for path in prepared_root.iterdir() if path.is_dir() and path.name.startswith("ep_")],
        key=_ep_sort_key,
    )
    if not episode_dirs:
        raise SystemExit(f"no ep_* directories found under input root: {prepared_root}")

    for episode_dir in episode_dirs:
        if not _has_camera_rgb_dir(episode_dir, cameras):
            print(f"[seg] skip {episode_dir.name}: no rgb dir for cameras={cameras}")
            continue
        if not _demo_complete_for_cameras(episode_dir, mask_root, cameras):
            return episode_dir

    raise SystemExit(
        f"all episodes under {prepared_root} already have masks in {mask_root} for cameras={cameras}"
    )


def _derive_mask_root(
    demo_arg: Path | None,
    prepared_root_arg: Path | None,
    mask_root_arg: Path | None,
) -> Path:
    if mask_root_arg is not None:
        return mask_root_arg

    prepared_root = prepared_root_arg
    if prepared_root is None and demo_arg is not None:
        prepared_root = demo_arg.parent if demo_arg.name.startswith("ep_") else demo_arg
    if prepared_root is None:
        raise SystemExit(
            "pass --input-root/--demo or explicitly pass --output-mask-root/--mask-root"
        )

    return prepared_root.parent / f"{prepared_root.name}-masks"


def _resolve_demo_dir(
    demo_arg: Path | None,
    prepared_root_arg: Path | None,
    mask_root: Path,
    cameras: list[str],
) -> Path:
    if prepared_root_arg is not None:
        demo_dir = _find_next_unprocessed_demo(prepared_root_arg, mask_root, cameras)
        print(f"[seg] selected next unprocessed demo: {demo_dir}")
        return demo_dir

    if demo_arg is None:
        raise SystemExit("pass --input-root to auto-select an episode, or --demo for one episode")
    if not demo_arg.exists():
        raise SystemExit(f"demo/input-root not found: {demo_arg}")

    if _has_camera_rgb_dir(demo_arg, cameras):
        return demo_arg

    episode_dirs = [path for path in demo_arg.iterdir() if path.is_dir() and path.name.startswith("ep_")]
    if episode_dirs:
        demo_dir = _find_next_unprocessed_demo(demo_arg, mask_root, cameras)
        print(f"[seg] selected next unprocessed demo: {demo_dir}")
        return demo_dir

    raise SystemExit(
        f"{demo_arg} is neither an episode dir nor an input root with ep_* dirs"
    )


# ------------------------------------------------------ main ------------------

def _clear_mask_pngs(masks_dir: Path) -> None:
    if not masks_dir.is_dir():
        return
    for path in masks_dir.glob("*.png"):
        path.unlink()


def _save_masks(
    masks: dict[int, np.ndarray],
    *,
    rgb_dir: Path,
    masks_dir: Path,
    n_frames: int,
) -> int:
    masks_dir.mkdir(parents=True, exist_ok=True)
    n_nonzero = 0
    for i in range(n_frames):
        m = masks.get(i)
        if m is None:
            print(f"[seg] WARNING: no mask for frame {i}")
            frame = cv2.imread(str(rgb_dir / f"{i:06d}.png"), cv2.IMREAD_COLOR)
            shape = frame.shape[:2] if frame is not None else (480, 640)
            m = np.zeros(shape, dtype=np.uint8)
        cv2.imwrite(str(masks_dir / f"{i:06d}.png"), m * 255)
        if int(m.sum()) > 0:
            n_nonzero += 1
    return n_nonzero


def _load_seed_mask(record: dict, mask_demo_dir: Path, demo_dir: Path) -> np.ndarray | None:
    seed_mask_rel = record.get("seed_mask_path")
    if not seed_mask_rel:
        return None
    seed_mask_path = mask_demo_dir / str(seed_mask_rel)
    if not seed_mask_path.is_file():
        seed_mask_path = demo_dir / str(seed_mask_rel)
    seed_mask_img = cv2.imread(str(seed_mask_path), cv2.IMREAD_GRAYSCALE)
    if seed_mask_img is None:
        raise SystemExit(f"failed to read seed mask: {seed_mask_path}")
    return (seed_mask_img > 0).astype(np.uint8)


def _process_demo(args, demo_dir: Path) -> str:
    mask_demo_dir = _mask_episode_dir(args.mask_root, demo_dir)
    mask_demo_dir.mkdir(parents=True, exist_ok=True)
    print(f"[seg] mask output dir: {mask_demo_dir}")

    source_meta_path = demo_dir / "source_meta.json"
    if source_meta_path.is_file():
        shutil.copy2(source_meta_path, mask_demo_dir / "source_meta.json")

    clicks_path = mask_demo_dir / "seg_clicks.json"
    legacy_clicks_path = demo_dir / "seg_clicks.json"
    source_h5 = _load_source_h5_from_demo(demo_dir, args.source_h5)
    saved_clicks = {}
    if args.reuse_clicks:
        read_clicks_path = clicks_path if clicks_path.exists() else legacy_clicks_path
        if not read_clicks_path.exists():
            raise SystemExit(f"--reuse-clicks but {clicks_path} doesn't exist")
        saved_clicks = json.loads(read_clicks_path.read_text(encoding="utf-8"))
        print(f"[seg] reusing clicks from {read_clicks_path}")

    new_clicks: dict[str, dict] = {}
    stop_after_accept = False

    staging_root = Path(tempfile.mkdtemp(prefix="sam2_stage_"))
    print(f"[seg] staging frames in {staging_root}")

    try:
        for role in args.cameras:
            cam_dir = demo_dir / role
            rgb_dir = cam_dir / "rgb"
            mask_cam_dir = mask_demo_dir / role
            masks_dir = mask_cam_dir / "masks"
            if not rgb_dir.exists():
                print(f"[seg] WARNING: no rgb frames for '{role}', skipping")
                continue

            force_manual = False
            while True:
                print()
                print(f"=== episode: {demo_dir.name} camera: {role} ===")

                if args.reuse_clicks and role in saved_clicks and not force_manual:
                    record = dict(saved_clicks[role])
                    seed_frame = int(record.get("seed_frame", 0))
                    pos = record.get("pos", [])
                    neg = record.get("neg", [])
                    box = record.get("box")
                    seed_mask = _load_seed_mask(record, mask_demo_dir, demo_dir)
                    print(f"[seg] using saved seed for role={role} seed_frame={seed_frame}")
                else:
                    frame_paths = list_rgb_frames(rgb_dir)
                    if args.auto_mode != "manual" and not force_manual:
                        record = _select_auto_prompt(
                            frame_paths,
                            role=role,
                            demo_dir=demo_dir,
                            source_h5=source_h5,
                            auto_mode=args.auto_mode,
                            ckpt=str(args.ckpt),
                            cfg=args.cfg,
                            yolo_weights=args.yolo_weights,
                            ultralytics_repo_dir=args.ultralytics_repo_dir,
                            yolo_conf_thresh=args.yolo_conf_thresh,
                            yolo_imgsz=args.yolo_imgsz,
                            yolo_class_id=args.yolo_class_id,
                            yolo_device=args.yolo_device,
                            gripper_close_threshold=args.gripper_close_threshold,
                            after_close_offset=args.after_close_offset,
                            close_window_before=args.close_window_before,
                            close_window_after=args.close_window_after,
                            T_e_wrist_path=args.T_e_wrist,
                            T_world_scene_path=args.T_world_scene,
                            scene_lift_delta_m=args.scene_lift_delta_m,
                            scene_lift_search_frames=args.scene_lift_search_frames,
                            wrist_target_xy_norm=(args.wrist_target_x, args.wrist_target_y),
                            wrist_target_radius_px=args.wrist_target_radius_px,
                        )
                        if record is None:
                            print(f"[seg] skipped '{role}'")
                            break
                        seed_frame = int(record["seed_frame"])
                        pos = record.get("pos", [])
                        neg = record.get("neg", [])
                        box = record.get("box")
                        seed_mask = _load_seed_mask(record, mask_demo_dir, demo_dir)
                        print(
                            f"[seg] auto seed role={role} mode={record.get('auto_mode')} "
                            f"frame={seed_frame} box={record.get('box')}"
                        )
                    else:
                        seed_frame = choose_seed_frame_interactively(frame_paths, role)
                        if seed_frame is None:
                            print(f"[seg] skipped '{role}'")
                            break
                        seed_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
                        if seed_image is None:
                            print(f"[seg] WARNING: can't read seed frame {seed_frame} of '{role}', skipping")
                            break
                        clicks = collect_clicks_interactively(seed_image, role, seed_frame)
                        if clicks is None:
                            print(f"[seg] skipped '{role}'")
                            break
                        pos, neg = clicks
                        box = None
                        seed_mask = None
                        record = {
                            "seed_frame": int(seed_frame),
                            "pos": pos,
                            "neg": neg,
                        }

                staged_dir = staging_root / role
                staged_paths = _stage_frames_as_jpg(rgb_dir, staged_dir)
                print(f"[seg] staged {len(staged_paths)} JPG frames")

                masks = run_sam2_on_frames(
                    staged_paths,
                    pos,
                    neg,
                    ckpt=str(args.ckpt),
                    cfg=args.cfg,
                    seed_frame=seed_frame,
                    box_xyxy=box,
                    seed_mask=seed_mask,
                )

                n_nonzero = _save_masks(
                    masks,
                    rgb_dir=rgb_dir,
                    masks_dir=masks_dir,
                    n_frames=len(staged_paths),
                )
                print(f"[seg] '{role}': frames-with-mask = {n_nonzero}/{len(staged_paths)}")

                review = _review_masks_interactively(
                    rgb_dir,
                    masks_dir,
                    role=role,
                    episode_name=demo_dir.name,
                )
                if review == "retry":
                    print(f"[seg] retrying {demo_dir.name}/{role}; discarded current masks")
                    _clear_mask_pngs(masks_dir)
                    force_manual = True
                    continue
                if review == "quit":
                    print(f"[seg] quit; discarded unaccepted masks for {demo_dir.name}/{role}")
                    _clear_mask_pngs(masks_dir)
                    return "stop"

                new_clicks[role] = record
                stop_after_accept = review == "accept_stop"
                print(f"[seg] accepted {demo_dir.name}/{role}")
                break

        merged = {**saved_clicks, **new_clicks}
        if merged:
            clicks_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            print(f"[seg] saved clicks to {clicks_path}")

    finally:
        if not args.keep_staging:
            shutil.rmtree(staging_root, ignore_errors=True)
        cv2.destroyAllWindows()

    return "stop" if stop_after_accept else "continue"


def _auto_select_mode(demo_arg: Path | None, output_root_arg: Path | None, cameras: list[str]) -> bool:
    if output_root_arg is not None:
        return True
    if demo_arg is None or not demo_arg.exists() or _has_camera_rgb_dir(demo_arg, cameras):
        return False
    return any(path.is_dir() and path.name.startswith("ep_") for path in demo_arg.iterdir())


def _normalize_cli_args(args) -> tuple[Path | None, Path | None]:
    prepared_root = args.input_root
    if args.output_root is not None:
        if prepared_root is not None and prepared_root != args.output_root:
            raise SystemExit("pass only one of --input-root or legacy --output-root")
        prepared_root = args.output_root

    mask_root = args.output_mask_root
    if args.mask_root is not None:
        if mask_root is not None and mask_root != args.mask_root:
            raise SystemExit("pass only one of --output-mask-root or --mask-root")
        mask_root = args.mask_root

    return prepared_root, mask_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", type=Path, default=None,
                   help="episode dir to segment, or an input root containing ep_* dirs")
    p.add_argument("--input-root", type=Path, default=None,
                   help="input prepared root containing ep_* dirs; segment the first episode missing masks")
    p.add_argument("--output-mask-root", type=Path, default=None,
                   help="output root for masks/clicks; default is <input-root>-masks")
    p.add_argument("--output-root", type=Path, default=None,
                   help="legacy alias of --input-root")
    p.add_argument("--mask-root", type=Path, default=None,
                   help="legacy alias of --output-mask-root")
    p.add_argument("--cameras", nargs="+", default=["wrist"])
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--cfg", default=DEFAULT_CFG)
    p.add_argument(
        "--auto-mode",
        choices=["manual", "hybrid", "yolo_box", "amg_center", "gripper_axis", "single_object"],
        default="manual",
        help="automatic seeding strategy; 'hybrid' uses wrist-projected scene seeding and gripper-axis SAM2 prompts for wrist; 'single_object' prefers an early scene box plus gripper-axis SAM2 prompts for wrist",
    )
    p.add_argument("--source-h5", type=Path, default=None)
    p.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument("--ultralytics-repo-dir", type=Path, default=DEFAULT_ULTRALYTICS_REPO)
    p.add_argument("--yolo-conf-thresh", type=float, default=0.10)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--yolo-class-id", type=int, default=0)
    p.add_argument("--yolo-device", default=None)
    p.add_argument("--gripper-close-threshold", type=float, default=0.5)
    p.add_argument("--after-close-offset", type=int, default=0)
    p.add_argument("--close-window-before", type=int, default=12)
    p.add_argument("--close-window-after", type=int, default=8)
    p.add_argument("--T-e-wrist", type=Path, default=DEFAULT_T_E_WRIST)
    p.add_argument("--T-world-scene", type=Path, default=DEFAULT_T_WORLD_SCENE)
    p.add_argument("--scene-lift-delta-m", type=float, default=0.02)
    p.add_argument("--scene-lift-search-frames", type=int, default=30)
    p.add_argument("--wrist-target-x", type=float, default=DEFAULT_WRIST_TARGET_X)
    p.add_argument("--wrist-target-y", type=float, default=DEFAULT_WRIST_TARGET_Y)
    p.add_argument("--wrist-target-radius-px", type=int, default=DEFAULT_WRIST_TARGET_RADIUS_PX)
    p.add_argument("--reuse-clicks", action="store_true",
                   help="re-segment using clicks saved in seg_clicks.json (no UI)")
    p.add_argument("--keep-staging", action="store_true",
                   help="don't delete the temp dir of staged JPGs (debug)")
    args = p.parse_args()

    prepared_root, explicit_mask_root = _normalize_cli_args(args)
    args.mask_root = _derive_mask_root(args.demo, prepared_root, explicit_mask_root)
    auto_select = _auto_select_mode(args.demo, prepared_root, args.cameras)
    processed_any = False

    while True:
        try:
            demo_dir = _resolve_demo_dir(args.demo, prepared_root, args.mask_root, args.cameras)
        except SystemExit as exc:
            if processed_any and "all episodes" in str(exc):
                print(f"[seg] {exc}")
                break
            raise

        result = _process_demo(args, demo_dir)
        processed_any = True
        if result == "stop" or not auto_select:
            break

    print("[seg] DONE")


if __name__ == "__main__":
    main()
