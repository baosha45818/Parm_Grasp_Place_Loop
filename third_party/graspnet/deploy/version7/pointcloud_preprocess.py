"""CPU-only RGB-D to GraspNet point-cloud preparation for Version7."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # Keep offline/raw replay usable where OpenCV is unavailable.
    cv2 = None


def prepare_capture(raw_dir: Path, output_dir: Path, *, roi_xywh: list[int], z_min: float, z_max: float,
                    x_abs_max: float, y_abs_max: float, color_profile_path: str | None = None) -> dict:
    """Build one camera-frame cloud; intentionally contains no CUDA/robot calls."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    depth = np.asarray(np.load(raw_dir / "depth.npy"))
    K = np.asarray(np.load(raw_dir / "depth_intrinsics.npy"), dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3): raise RuntimeError(f"invalid RGB-D capture: {raw_dir}")
    x, y, width, height = (int(v) for v in roi_xywh)
    h, w = depth.shape
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > w or y + height > h:
        raise RuntimeError(f"ROI {roi_xywh} is outside {w}x{h} depth image")
    started = time.monotonic()
    scale = float(meta["depth_scale_m"])
    if not np.isfinite(scale) or scale <= 0: raise RuntimeError("invalid depth scale")
    # The GraspNet input is restricted to ROI anyway.  Projecting only this crop
    # avoids undistorting/projecting the remaining camera frame every cycle.
    roi_depth = depth[y:y + height, x:x + width]
    z = roi_depth.astype(np.float32) * scale
    distortion_path = raw_dir / "depth_distortion.npy"
    if distortion_path.exists():
        if cv2 is None:
            raise RuntimeError("depth distortion was captured but OpenCV is unavailable for undistortion")
        distortion = np.asarray(np.load(distortion_path), dtype=np.float64).reshape(-1)
        grid_u, grid_v = np.meshgrid(np.arange(x, x + width), np.arange(y, y + height))
        pixels = np.stack((grid_u, grid_v), axis=-1).astype(np.float32).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(pixels, K, distortion).reshape(height, width, 2)
        cloud = np.stack((normalized[:, :, 0] * z, normalized[:, :, 1] * z, z), axis=-1)
        projection_model = "undistortPoints"
    else:
        grid_u, grid_v = np.meshgrid(np.arange(x, x + width, dtype=np.float32), np.arange(y, y + height, dtype=np.float32))
        cloud = np.stack(((grid_u - K[0, 2]) * z / K[0, 0], (grid_v - K[1, 2]) * z / K[1, 1], z), axis=-1)
        projection_model = "pinhole"
    mask = roi_depth > 0
    points = cloud[mask]
    workspace = ((points[:, 2] >= z_min) & (points[:, 2] <= z_max) &
                 (np.abs(points[:, 0]) <= x_abs_max) & (np.abs(points[:, 1]) <= y_abs_max))
    points = points[workspace]
    if len(points) < 100: raise RuntimeError(f"only {len(points)} valid points after CPU preprocessing")
    color = np.asarray(Image.open(raw_dir / "color.png").convert("RGB"), dtype=np.float32) / 255.0
    colors = color[y:y + height, x:x + width][mask][workspace] if color.shape[:2] == depth.shape else np.zeros_like(points)
    color_record = {"enabled": False, "points_before_color_filter": int(len(points)), "points_after_color_filter": int(len(points))}
    if color_profile_path is not None:
        profile_path = Path(color_profile_path).resolve()
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("camera") not in (None, meta.get("camera_name")):
            raise RuntimeError(
                f"color profile camera={profile.get('camera')!r} does not match capture camera={meta.get('camera_name')!r}"
            )
        if not bool(profile.get("enabled", False)):
            color_record.update({"profile_path": str(profile_path), "enabled": False})
        else:
            if color.shape[:2] != depth.shape:
                raise RuntimeError("enabled color profile requires aligned RGB and depth images of the same size")
            if cv2 is None:
                raise RuntimeError("enabled color profile requires OpenCV for RGB-to-Lab conversion")
            samples = np.asarray(profile.get("object_samples_rgb", []), dtype=np.uint8)
            if samples.ndim != 2 or samples.shape[1:] != (3,) or len(samples) == 0:
                raise RuntimeError(f"invalid object_samples_rgb in color profile: {profile_path}")
            max_distance = float(profile.get("max_lab_distance", 0.0))
            lightness = profile.get("lab_l_range", [0, 255])
            if not np.isfinite(max_distance) or max_distance <= 0.0 or len(lightness) != 2:
                raise RuntimeError(f"invalid Lab thresholds in color profile: {profile_path}")
            l_min, l_max = (float(value) for value in lightness)
            if not (0.0 <= l_min <= l_max <= 255.0):
                raise RuntimeError(f"invalid lab_l_range in color profile: {profile_path}")
            rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            sample_lab = cv2.cvtColor(samples.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            nearest_distance = np.linalg.norm(lab[:, None, :] - sample_lab[None, :, :], axis=2).min(axis=1)
            seed_mask = (nearest_distance <= max_distance) & (lab[:, 0] >= l_min) & (lab[:, 0] <= l_max)
            mode = str(profile.get("mode", "object_sample_only"))
            if mode in {"seeded_non_table_components", "exclude_table_like"}:
                table_samples = np.asarray(profile.get("table_samples_rgb", []), dtype=np.uint8)
                table_distance = float(profile.get("max_table_lab_distance", 0.0))
                min_component_seeds = int(profile.get("min_component_seed_points", 1))
                close_kernel_px = int(profile.get("component_close_kernel_px", 0))
                if (table_samples.ndim != 2 or table_samples.shape[1:] != (3,) or len(table_samples) == 0
                        or not np.isfinite(table_distance) or table_distance <= 0.0
                        or min_component_seeds < 1 or close_kernel_px < 0):
                    raise RuntimeError(f"invalid seeded non-table thresholds in color profile: {profile_path}")
                table_lab = cv2.cvtColor(table_samples.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
                nearest_table_distance = np.linalg.norm(lab[:, None, :] - table_lab[None, :, :], axis=2).min(axis=1)
                non_table_mask = nearest_table_distance > table_distance
                if mode == "exclude_table_like":
                    # Table color is the stable observation.  Do not require
                    # a point to resemble a fixed workpiece color: rust,
                    # oil, and specular highlights are all retained unless
                    # they are positively table-like.
                    color_mask = non_table_mask
                    color_mode_record = {
                        "mode": mode,
                        "non_table_points": int(np.count_nonzero(non_table_mask)),
                        "max_table_lab_distance": table_distance,
                    }
                else:
                    valid_image = np.zeros((height, width), dtype=bool)
                    valid_image[mask] = workspace
                    seed_image = np.zeros((height, width), dtype=np.uint8)
                    candidate_image = np.zeros((height, width), dtype=np.uint8)
                    seed_image[valid_image] = seed_mask.astype(np.uint8)
                    candidate_image[valid_image] = non_table_mask.astype(np.uint8)
                    if close_kernel_px >= 2:
                        kernel = np.ones((close_kernel_px, close_kernel_px), dtype=np.uint8)
                        candidate_image = cv2.morphologyEx(candidate_image, cv2.MORPH_CLOSE, kernel)
                        candidate_image &= valid_image.astype(np.uint8)
                    component_count, labels = cv2.connectedComponents(candidate_image, connectivity=8)
                    seed_labels = labels[seed_image.astype(bool)]
                    seed_counts = np.bincount(seed_labels, minlength=component_count)
                    keep_labels = np.flatnonzero(seed_counts >= min_component_seeds)
                    keep_labels = keep_labels[keep_labels != 0]
                    color_mask = np.isin(labels[valid_image], keep_labels)
                    color_mode_record = {
                        "mode": mode,
                        "seed_points": int(np.count_nonzero(seed_mask)),
                        "non_table_points": int(np.count_nonzero(non_table_mask)),
                        "kept_components": int(len(keep_labels)),
                        "max_table_lab_distance": table_distance,
                    }
            elif mode == "object_sample_only":
                color_mask = seed_mask
                color_mode_record = {"mode": mode, "seed_points": int(np.count_nonzero(seed_mask))}
            else:
                raise RuntimeError(f"unsupported color profile mode={mode!r}: {profile_path}")
            points, colors = points[color_mask], colors[color_mask]
            if len(points) < 100:
                raise RuntimeError(
                    f"color profile retained only {len(points)} points; inspect or widen {profile_path}"
                )
            color_record = {
                "enabled": True,
                "profile_path": str(profile_path),
                "profile_name": profile.get("name", profile_path.stem),
                "points_before_color_filter": int(len(color_mask)),
                "points_after_color_filter": int(len(points)),
                "max_lab_distance": max_distance,
                "lab_l_range": [l_min, l_max],
                **color_mode_record,
            }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "point_cloud.npy", points.astype(np.float32))
    np.save(output_dir / "point_colors.npy", colors.astype(np.float32))
    np.save(output_dir / "depth_intrinsics.npy", K)
    if distortion_path.exists():
        np.save(output_dir / "depth_distortion.npy", np.load(distortion_path))
    np.save(output_dir / "color_intrinsics.npy", np.load(raw_dir / "color_intrinsics.npy"))
    shutil.copy2(raw_dir / "color.png", output_dir / "color.png")
    output_meta = dict(meta)
    output_meta.update({"roi_xywh": [x, y, width, height], "num_points": int(len(points)),
                        "workspace_limits": {"z_min": z_min, "z_max": z_max, "x_abs_max": x_abs_max, "y_abs_max": y_abs_max},
                        "cpu_preprocessed": True, "depth_projection_model": projection_model,
                        "color_filter": color_record})
    (output_dir / "meta.json").write_text(json.dumps(output_meta, indent=2), encoding="utf-8")
    if color_profile_path is not None:
        shutil.copy2(Path(color_profile_path), output_dir / "color_profile.json")
    return {"asset_dir": str(output_dir), "num_points": int(len(points)), "color_filter": color_record,
            "elapsed_s": time.monotonic() - started}
