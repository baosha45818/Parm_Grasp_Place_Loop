#!/usr/bin/env python3
"""单独运行模块2：规划观察、交互式SAM2分割、点数判断和工件ICP位姿估计。"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import os
from pathlib import Path
import select
import socket
import sys
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"模块2运行环境不存在: {PYTHON}")
if Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from loop_config import LoopConfig
from loop_types import FaceSide, Holding, LeftArmState, ObjectPose, ObservationClass
from tools.inspection_support.mask_depth import (
    mask_depth_diagnostics,
    mask_depth_to_points_cam,
    remove_radius_outliers_np,
    valid_median_depth,
    voxel_downsample_np,
)
from tools.inspection_support.registration import make_pca_initial_guesses
from tools.inspection_support.robot_camera import FemtoBoltCamera
from tools.inspection_support.stl_model import load_stl_triangles_m, sample_surface_points
from tools.inspection_support.geometry import invert_T
from tools.planner_runtime import PlannerRuntime
from .implementation import (
    _load_base_camera_transform,
    _transform_from_planner_response,
)


CONFIG_PATH = Path(__file__).with_name("inspect_config.json")
# SAM2 runs as a separately deployed service.  The resident service defaults to
# HTTP-over-UDS, while HTTP(S) endpoints remain available for remote deployments.
DEFAULT_SAM2_URL = os.environ.get(
    "SAM2_SERVICE_URL", "unix:///tmp/sam2-service.sock"
)
GRASPUP_JOINTS_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]
DEFAULT_TARGET_OBJECT_XYZ_M = [0.517668, 0.018779, -0.200000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立测试左手模块2 inspect")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--execute-url", default="http://127.0.0.1:5667/middle/waypoint/execute")
    parser.add_argument(
        "--sam2-url", default=DEFAULT_SAM2_URL,
        help=("SAM2 C/S 服务地址（默认 unix:///tmp/sam2-service.sock）；"
              "可填写 HTTP(S) 接口，默认值可由 SAM2_SERVICE_URL 环境变量覆盖"),
    )
    parser.add_argument(
        "--auto-gripper-prompt", action=argparse.BooleanOptionalAction, default=True,
        help="先用深度约束的RGB轮廓自动生成夹爪中心十字、ROI和SAM2正点（默认开启）",
    )
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--point-threshold", type=int, default=None, help="pc1低于该点数时转pc2")
    parser.add_argument("--skip-motion", action="store_true", help="不移动机械臂，只在当前位置拍摄")
    parser.add_argument("--graspup-lift-m", type=float, default=0.05,
                        help="回到graspup后沿body_base +Z继续上移的距离，默认0.05m")
    parser.add_argument("--place-descent-m", type=float, default=0.18,
                        help="到达工件目标位姿后沿body_base -Z下降距离，默认0.18m")
    parser.add_argument("--target-xy-radius-m", type=float, default=0.05,
                        help="工件目标xy允许圆半径，默认0.05m；采样圆心、半径一半和外圈")
    parser.add_argument("--gripper-host", default="127.0.0.1",
                        help="左夹爪服务地址，默认127.0.0.1")
    parser.add_argument("--target-object-xyz", type=float, nargs=3,
                        default=DEFAULT_TARGET_OBJECT_XYZ_M, metavar=("X", "Y", "Z"),
                        help="工件原点在body_base下的目标位置，单位m")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/module_2_inspect")
    return parser.parse_args()


def matrix_to_object_pose(matrix: np.ndarray) -> ObjectPose:
    return ObjectPose(
        frame_id="body_base",
        position=np.asarray(matrix[:3, 3], dtype=float).tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def target_object_pose_for_yaw(target_xyz: list[float], yaw_deg: float) -> np.ndarray:
    target = np.eye(4, dtype=np.float64)
    # Rotate object +X from body_base +X toward body_base +Y.
    # Object +Y follows the right-handed frame and object +Z remains body_base +Z.
    target[:3, :3] = Rotation.from_euler("z", float(yaw_deg), degrees=True).as_matrix()
    target[:3, 3] = np.asarray(target_xyz, dtype=np.float64)
    return target


def prompt_segment(
    image: np.ndarray,
    box: list[int],
    positives: list[list[int]],
    negatives: list[list[int]],
    sam2_url: str,
) -> np.ndarray:
    if not positives:
        raise ValueError("SAM2分割至少需要一个手动正点")
    points = np.asarray(positives + negatives, dtype=np.int32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    payload = io.BytesIO()
    np.savez_compressed(payload, image_bgr=image, points=points, labels=labels, box=np.asarray(box))

    endpoint = urlparse(sam2_url)
    if endpoint.scheme == "unix":
        if not endpoint.path:
            raise ValueError(f"--sam2-url 必须包含 Unix Socket 路径: {sam2_url!r}")

        class UnixConnection(http.client.HTTPConnection):
            def connect(self) -> None:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(endpoint.path)

        connection = UnixConnection("sam2-service", timeout=90.0)
        request_path = "/v1/prompt-segment-sam2"
    elif endpoint.scheme in {"http", "https"} and endpoint.hostname:
        connection_class = (
            http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(endpoint.hostname, port=endpoint.port, timeout=90.0)
        request_path = (endpoint.path or "/") + (f"?{endpoint.query}" if endpoint.query else "")
    else:
        raise ValueError(
            f"--sam2-url 必须是 unix:// Socket 地址或有效 HTTP(S) 地址: {sam2_url!r}"
        )
    try:
        connection.request(
            "POST", request_path, body=payload.getvalue(),
            headers={"Content-Type": "application/x-sam2-prompt-npz"},
        )
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"SAM2返回HTTP {response.status}: {body.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return (archive["mask"] > 0).astype(np.uint8) * 255


def prompt_segment_with_auto_negatives(
    image: np.ndarray, box: list[int], positives: list[list[int]], sam2_url: str,
) -> tuple[np.ndarray, list[list[int]]]:
    """Use detached first-pass mask fragments as conservative SAM2 negative points."""
    first_mask = prompt_segment(image, box, positives, [], sam2_url)
    count, labels, stats, centers = cv2.connectedComponentsWithStats((first_mask > 0).astype(np.uint8))
    primary_label = int(labels[positives[0][1], positives[0][0]])
    if primary_label == 0 and count > 1:
        primary_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    negatives = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label != primary_label and area >= 8:
            negatives.append([int(round(centers[label][0])), int(round(centers[label][1]))])
    negatives = negatives[:3]
    if not negatives:
        return first_mask, negatives
    return prompt_segment(image, box, positives, negatives, sam2_url), negatives


def propose_gripper_prompt(
    image: np.ndarray, depth: np.ndarray, depth_scale: float, cfg: dict,
) -> dict | None:
    """Create a conservative SAM2 prompt from two depth-constrained RGB components."""
    height, width = depth.shape
    fixed_box = cfg.get("auto_prompt_roi_xyxy")
    fixed_positive = cfg.get("auto_prompt_positive_xy")
    if fixed_box is not None and fixed_positive is not None:
        box = [int(value) for value in fixed_box]
        positive = [int(value) for value in fixed_positive]
        if (
            len(box) == 4 and len(positive) == 2
            and 0 <= box[0] < box[2] < width and 0 <= box[1] < box[3] < height
            and box[0] <= positive[0] <= box[2] and box[1] <= positive[1] <= box[3]
        ):
            return {
                "box": box,
                "positives": [positive],
                "negatives": [],
                "cross": positive,
                "depth_near_limit_m": float(cfg.get("auto_prompt_depth_limit_m", 0.5)),
                "gap_px": None,
                "strategy": "configured_head_observation_roi",
            }
    depth_m = depth.astype(np.float32) * float(depth_scale)
    valid = (depth_m >= float(cfg["min_depth_m"])) & (depth_m <= float(cfg["max_depth_m"]))
    samples = depth_m[valid]
    if samples.size < max(1_000, height * width // 200):
        return None
    # Keep the near-depth layer before inspecting RGB contours, rejecting table/background.
    near_limit = min(float(np.percentile(samples, 90)), float(np.percentile(samples, 10)) + 0.18)
    foreground = (valid & (depth_m <= near_limit)).astype(np.uint8) * 255
    foreground[: int(height * 0.08), :] = 0
    foreground[int(height * 0.92):, :] = 0
    foreground[:, : int(width * 0.04)] = 0
    foreground[:, int(width * 0.96):] = 0
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    rgb_edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 150)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    min_area = max(180, int(height * width * 0.0008))
    components = []
    for label in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area < min_area:
            continue
        component = labels == label
        components.append({"label": label, "x": x, "y": y, "w": w, "h": h, "area": area,
                           "cx": float(centers[label][0]), "cy": float(centers[label][1]),
                           "edge_density": float(np.mean(rgb_edges[component] > 0))})
    left = [item for item in components if item["cx"] < width * 0.48]
    right = [item for item in components if item["cx"] > width * 0.52]
    if left and right:
        # Select the two inward-facing foreground components; RGB edges break ties.
        left_finger = max(left, key=lambda item: (item["x"] + item["w"], item["edge_density"], item["area"]))
        right_finger = min(right, key=lambda item: (item["x"], -item["edge_density"], -item["area"]))
        left_inner, right_inner = left_finger["x"] + left_finger["w"] - 1, right_finger["x"]
        gap = right_inner - left_inner
        if max(20, width // 50) <= gap <= int(width * 0.65):
            cross_x = int(round((left_inner + right_inner) / 2))
            cross_y = int(round((left_finger["cy"] + right_finger["cy"]) / 2))
            y0, y1 = max(0, cross_y - max(30, gap // 2)), min(height, cross_y + max(30, gap // 2) + 1)
            x0, x1 = left_inner + max(3, gap // 5), right_inner - max(3, gap // 5)
            candidate = foreground[y0:y1, x0:x1] > 0 if x1 > x0 else np.empty((0, 0), dtype=bool)
            ys, xs = np.where(candidate)
            if xs.size:
                closest = int(np.argmin((xs + x0 - cross_x) ** 2 + (ys + y0 - cross_y) ** 2))
                positive = [int(xs[closest] + x0), int(ys[closest] + y0)]
                pad_x, pad_y = max(24, gap // 3), max(30, gap // 2)
                return {"box": [max(0, left_inner - pad_x), max(0, cross_y - pad_y),
                                min(width - 1, right_inner + pad_x), min(height - 1, cross_y + pad_y)],
                        "positives": [positive], "negatives": [], "cross": [cross_x, cross_y],
                        "depth_near_limit_m": near_limit, "gap_px": gap, "strategy": "two_fingers"}

    # In the chest-camera view, two fingers can merge with the held workpiece
    # into one depth component. Use its deepest interior pixel as a stable
    # foreground positive prompt; SAM2 then separates its RGB boundary.
    candidates = [item for item in components if item["area"] >= min_area * 4
                  and width * 0.12 <= item["cx"] <= width * 0.72]
    if not candidates:
        return None
    target = max(candidates, key=lambda item: item["area"] * (1.0 + item["edge_density"]))
    target_mask = (labels == target["label"]).astype(np.uint8)
    distance = cv2.distanceTransform(target_mask, cv2.DIST_L2, 5)
    positive_y, positive_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    pad = 24
    return {"box": [max(0, target["x"] - pad), max(0, target["y"] - pad),
                    min(width - 1, target["x"] + target["w"] - 1 + pad),
                    min(height - 1, target["y"] + target["h"] - 1 + pad)],
            "positives": [[int(positive_x), int(positive_y)]], "negatives": [],
            "cross": [int(positive_x), int(positive_y)], "depth_near_limit_m": near_limit,
            "gap_px": None, "strategy": "merged_gripper_workpiece"}


def propose_pc2_depth_layer_prompt(
    depth: np.ndarray, depth_scale: float, cfg: dict,
) -> dict | None:
    """Generate a PC2-only SAM2 prompt from the fixed-pose depth-layer rule.

    PC1 deliberately continues to use ``propose_gripper_prompt`` unchanged.
    PC2 uses the largest connected component of its near valid-depth layer.
    """
    height, width = depth.shape
    options = dict(cfg.get("pc2_auto_prompt", {}))
    depth_m = depth.astype(np.float32) * float(depth_scale)
    valid = (depth_m >= float(cfg["min_depth_m"])) & (depth_m <= float(cfg["max_depth_m"]))
    samples = depth_m[valid]
    if samples.size < max(1_000, height * width // 200):
        return None
    lower = float(options.get(
        "near_min_m", np.percentile(samples, float(options.get("near_percentile", 2.0)))
    ))
    near_limit = float(options.get(
        "near_max_m",
        min(
            float(np.percentile(samples, float(options.get("far_percentile", 90.0)))),
            lower + float(options.get("near_layer_width_m", 0.18)),
        ),
    ))
    if not np.isfinite(lower) or not np.isfinite(near_limit) or near_limit <= lower:
        raise ValueError("pc2_auto_prompt near-depth range is invalid")
    foreground = valid & (depth_m >= lower) & (depth_m <= near_limit)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    min_area = max(int(options.get("min_component_area_px", 500)), int(height * width * 0.0005))
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(areas))
    if int(areas[largest_index]) < min_area:
        return None
    largest_label = largest_index + 1
    candidate = (labels == largest_label).astype(np.uint8)
    x, y, component_width, component_height, _ = [
        int(value) for value in stats[largest_label]
    ]
    distance = cv2.distanceTransform(candidate, cv2.DIST_L2, 5)
    positive_y, positive_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    pad = int(options.get("box_padding_px", 24))
    box = [max(0, x - pad), max(0, y - pad),
           min(width - 1, x + component_width - 1 + pad),
           min(height - 1, y + component_height - 1 + pad)]
    return {
        "box": box,
        "positives": [[int(positive_x), int(positive_y)]],
        "negatives": [],
        "cross": [int(positive_x), int(positive_y)],
        "depth_near_min_m": lower,
        "depth_near_limit_m": near_limit,
        "gap_px": None,
        "strategy": "pc2_near_depth_largest_component",
        "candidate_mask": candidate * 255,
    }


def annotate(
    image: np.ndarray, view_name: str, sam2_url: str, auto_prompt: dict | None = None,
) -> tuple[np.ndarray, list[int], list[list[int]], list[list[int]]]:
    box: list[int] = list(auto_prompt["box"]) if auto_prompt else []
    positives: list[list[int]] = list(auto_prompt["positives"]) if auto_prompt else []
    negatives: list[list[int]] = list(auto_prompt.get("negatives", [])) if auto_prompt else []
    dragging = False
    start = (0, 0)
    window = f"module2-{view_name}"
    preview_mask: np.ndarray | None = None

    def mouse(event, x, y, _flags, _param):
        nonlocal dragging, start, box
        if preview_mask is not None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(box) != 4:
                dragging, start = True, (x, y)
            elif box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                positives.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            box = [min(start[0], x), min(start[1], y), max(start[0], x), max(start[1], y)]
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            box = [min(start[0], x), min(start[1], y), max(start[0], x), max(start[1], y)]
        elif event == cv2.EVENT_RBUTTONDOWN and len(box) == 4:
            negatives.append([x, y])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    print(
        f"[{view_name}] 在CV窗口拖ROI并标正负点；所有确认在本终端输入。\n"
        "  s+回车=开始分割  u+回车=撤销点  r+回车=全部重标  q+回车=退出"
    )
    try:
        while True:
            if preview_mask is None:
                canvas = image.copy()
                if len(box) == 4:
                    cv2.rectangle(canvas, box[:2], box[2:], (0, 255, 0), 2)
                if auto_prompt:
                    cross = auto_prompt["cross"]
                    cv2.drawMarker(canvas, cross, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                    cv2.putText(canvas, "auto gripper center", (cross[0] + 10, max(20, cross[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                for point in positives:
                    cv2.circle(canvas, point, 6, (0, 255, 0), -1)
                for point in negatives:
                    cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            else:
                canvas = image.copy()
                canvas[preview_mask > 0] = (
                    canvas[preview_mask > 0] * 0.45 + np.array([0, 0, 255]) * 0.55
                )
                canvas = canvas.astype(np.uint8)
                for point in positives:
                    cv2.circle(canvas, point, 6, (0, 255, 0), -1)
                for point in negatives:
                    cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.imshow(window, canvas)
            cv2.waitKey(30)
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                continue
            command = sys.stdin.readline().strip().lower()
            if preview_mask is not None:
                if command in {"y", "yes", "1"}:
                    return preview_mask, box, positives, negatives
                if command in {"r", "redo", "0"}:
                    preview_mask = None
                    auto_prompt = None
                    box, positives, negatives = [], [], []
                    print(f"[{view_name}] 已清空，请在CV窗口重新标注；完成后在终端输入 s")
                    continue
                if command in {"q", "quit"}:
                    raise KeyboardInterrupt
                print(f"[{view_name}] 预览确认请输入 y、r 或 q")
                continue

            if command in {"s", "segment", ""}:
                if len(box) != 4 or box[2] - box[0] <= 5 or box[3] - box[1] <= 5:
                    print(f"[{view_name}] 请先左键拖出有效ROI")
                    continue
                if not positives:
                    print(f"[{view_name}] ROI完成后，请在工件上左键添加至少一个正点")
                    continue
                print(f"[{view_name}] 开始SAM2分割: 正点={len(positives)}, 负点={len(negatives)}")
                if auto_prompt and not negatives:
                    preview_mask, generated_negatives = prompt_segment_with_auto_negatives(
                        image, box, positives, sam2_url
                    )
                    negatives.extend(generated_negatives)
                    if generated_negatives:
                        print(f"[{view_name}] 已从第一轮孤立mask自动生成负点: {generated_negatives}")
                else:
                    preview_mask = prompt_segment(image, box, positives, negatives, sam2_url)
                print(
                    f"[{view_name}] SAM2 mask pixels={np.count_nonzero(preview_mask)}；"
                    "请在本终端输入 y接受 / r重标 / q退出"
                )
            elif command in {"r", "reset"}:
                auto_prompt = None
                box, positives, negatives = [], [], []
                print(f"[{view_name}] 已清空ROI和所有点")
            elif command in {"u", "undo"}:
                if negatives:
                    negatives.pop()
                elif positives:
                    positives.pop()
                print(f"[{view_name}] 当前正点={len(positives)}，负点={len(negatives)}")
            elif command in {"q", "quit"}:
                raise KeyboardInterrupt
            else:
                print(f"[{view_name}] 未知命令；请输入 s、u、r 或 q")
    finally:
        cv2.destroyWindow(window)


def capture_view(
    camera: FemtoBoltCamera, cfg: dict, name: str, output: Path, sam2_url: str,
    auto_gripper_prompt: bool,
) -> tuple[np.ndarray, dict]:
    frames = [camera.read() for _ in range(int(cfg["capture_frames"]))]
    frame = frames[-1]
    if cfg.get("depth_aggregation") == "valid_median":
        depth, depth_support = valid_median_depth(
            [item.depth for item in frames], frame.depth_scale,
            float(cfg["min_depth_m"]), float(cfg["max_depth_m"]),
        )
    else:
        depth = np.median(np.stack([item.depth for item in frames]), axis=0).astype(np.uint16)
        depth_support = np.count_nonzero(
            np.stack([item.depth for item in frames]), axis=0
        ).astype(np.uint16)
    if depth.shape != frame.color.shape[:2]:
        raise RuntimeError(
            f"RGB/depth未对齐: camera={frame.camera_name}, "
            f"rgb={frame.color.shape[:2]}, depth={depth.shape}；"
            "不能直接用RGB上的SAM2 mask索引未对齐深度"
        )
    if auto_gripper_prompt:
        # PC1 keeps the existing gripper/RGB-contour strategy exactly as-is.
        # PC2 uses the separately configured fixed-pose depth-layer strategy.
        auto_prompt = (
            propose_pc2_depth_layer_prompt(depth, frame.depth_scale, cfg)
            if name == "pc2"
            else propose_gripper_prompt(frame.color, depth, frame.depth_scale, cfg)
        )
    else:
        auto_prompt = None
    if auto_prompt:
        print(f"[{name}] 自动提示策略={auto_prompt.get('strategy')}: "
              f"gap={auto_prompt['gap_px']}px, "
              f"深度近景阈值={auto_prompt['depth_near_limit_m']:.3f}m；"
              "黄色十字是提示中心，绿色点是SAM2正点")
    elif auto_gripper_prompt:
        print(f"[{name}] 未找到可信的双夹爪深度/RGB候选，切换为手动标注")
    mask, box, positives, negatives = annotate(frame.color, name, sam2_url, auto_prompt)
    mask_depth = depth[mask > 0]
    mask_depth_nonzero = mask_depth[mask_depth > 0]
    raw_depth_range = None
    if mask_depth_nonzero.size:
        raw_depth_range = {
            "min": float(np.min(mask_depth_nonzero)),
            "median": float(np.median(mask_depth_nonzero)),
            "max": float(np.max(mask_depth_nonzero)),
        }
    print(
        f"[{name}] depth scale={frame.depth_scale:.6g} m/unit，"
        f"mask内原始深度(min/median/max)={raw_depth_range}"
    )
    diagnostics = mask_depth_diagnostics(
        mask, depth, frame.depth_scale, min_depth_m=float(cfg["min_depth_m"]),
        max_depth_m=float(cfg["max_depth_m"]), erode_px=int(cfg["erode_px"]),
    )
    points = mask_depth_to_points_cam(
        mask, depth, frame.depth_intrinsics, frame.depth_scale,
        min_depth_m=float(cfg["min_depth_m"]), max_depth_m=float(cfg["max_depth_m"]),
        erode_px=int(cfg["erode_px"]),
    )
    raw_point_count = int(len(points))
    points = remove_radius_outliers_np(
        points, float(cfg.get("outlier_mad_scale", 4.0))
    )
    filtered_point_count = int(len(points))
    points = voxel_downsample_np(points, float(cfg["observed_voxel_m"]))
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / f"{name}_rgb.png"), frame.color)
    if auto_prompt and "candidate_mask" in auto_prompt:
        cv2.imwrite(str(output / f"{name}_auto_depth_candidate.png"), auto_prompt["candidate_mask"])
    cv2.imwrite(str(output / f"{name}_mask.png"), mask)
    cv2.imwrite(str(output / f"{name}_valid_depth.png"), diagnostics["valid_mask_u8"])
    support_visual = np.clip(
        depth_support.astype(np.float32) * (255.0 / max(int(cfg["capture_frames"]), 1)),
        0, 255,
    ).astype(np.uint8)
    cv2.imwrite(str(output / f"{name}_depth_support.png"), support_visual)
    np.save(output / f"{name}_depth.npy", depth)
    np.save(output / f"{name}_depth_support.npy", depth_support)
    np.save(output / f"{name}_points_camera.npy", points)
    report = {"roi_xyxy": box, "positive_points": positives, "negative_points": negatives,
              "mask_pixels": int(np.count_nonzero(mask)),
              "depth_scale_m_per_unit": float(frame.depth_scale),
              "camera_intrinsics": frame.depth_intrinsics.tolist(),
              "mask_raw_depth": raw_depth_range,
              "raw_point_count": raw_point_count,
              "outlier_filtered_point_count": filtered_point_count,
              "point_filter": {
                  "erode_px": int(cfg["erode_px"]),
                  "outlier_mad_scale": float(cfg.get("outlier_mad_scale", 4.0)),
                  "voxel_m": float(cfg["observed_voxel_m"]),
              },
              "depth_aggregation": str(cfg.get("depth_aggregation", "raw_median")),
              "capture_frames": int(cfg["capture_frames"]),
              "depth_support_nonzero_px": int(np.count_nonzero(depth_support)),
              "valid_point_count": int(len(points)), "depth_valid_ratio": float(diagnostics["depth_valid_ratio"])}
    (output / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[{name}] 有效点云数量={len(points)}，depth有效比例={report['depth_valid_ratio']:.3f}")
    return points, report


def save_pose_visualization(
    run_dir: Path,
    view_name: str,
    cad_points: np.ndarray,
    T_camera_object: np.ndarray,
    T_output_object: np.ndarray,
    output_frame: str,
    camera_intrinsics: np.ndarray,
    output_filename: str = "pose_estimation_visualization.png",
) -> Path:
    """Overlay the estimated STL pose and object axes on the captured RGB image."""
    image_path = run_dir / f"{view_name}_rgb.png"
    mask_path = run_dir / f"{view_name}_mask.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise RuntimeError(f"位姿可视化缺少图像或mask: {image_path}, {mask_path}")

    left = image.copy()
    red = np.zeros_like(left)
    red[:, :, 2] = 255
    selected = mask > 0
    left[selected] = (left[selected] * 0.55 + red[selected] * 0.45).astype(np.uint8)
    cv2.putText(left, "SAM2 mask", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    right = image.copy()
    K = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)

    def project(points_object: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_object, dtype=np.float64).reshape(-1, 3)
        points_camera = (T_camera_object[:3, :3] @ points.T).T + T_camera_object[:3, 3]
        valid = points_camera[:, 2] > 1e-4
        pixels_h = (K @ points_camera[valid].T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        return np.rint(pixels).astype(np.int32), valid

    stride = max(1, len(cad_points) // 4000)
    cad_pixels, _ = project(cad_points[::stride])
    height, width = right.shape[:2]
    inside = (
        (cad_pixels[:, 0] >= 0) & (cad_pixels[:, 0] < width)
        & (cad_pixels[:, 1] >= 0) & (cad_pixels[:, 1] < height)
    )
    for u, v in cad_pixels[inside]:
        cv2.circle(right, (int(u), int(v)), 1, (0, 255, 255), -1)

    extent = np.ptp(np.asarray(cad_points, dtype=np.float64), axis=0)
    axis_length = max(0.03, min(0.12, float(np.max(extent)) * 0.55))
    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
        [0.0, 0.0, -axis_length],
    ])
    axis_pixels, axis_valid = project(axis_points)
    if np.all(axis_valid):
        origin = tuple(axis_pixels[0])
        axes = [
            (axis_pixels[1], (0, 0, 255), "+X"),
            (axis_pixels[2], (0, 255, 0), "+Y"),
            (axis_pixels[3], (255, 0, 255), "FRONT(+Z)"),
            (axis_pixels[4], (160, 160, 160), "-Z"),
        ]
        for endpoint, color, label in axes:
            endpoint_xy = tuple(endpoint)
            cv2.arrowedLine(right, origin, endpoint_xy, color, 4, tipLength=0.18)
            cv2.putText(right, label, endpoint_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    camera_rpy_deg = Rotation.from_matrix(T_camera_object[:3, :3]).as_euler("xyz", degrees=True)
    output_rpy_deg = Rotation.from_matrix(T_output_object[:3, :3]).as_euler("xyz", degrees=True)
    text_lines = [
        "Estimated STL overlay (yellow)",
        f"camera xyz={np.round(T_camera_object[:3, 3], 4).tolist()} m",
        f"camera rpy={np.round(camera_rpy_deg, 1).tolist()} deg",
        f"{output_frame} xyz={np.round(T_output_object[:3, 3], 4).tolist()} m",
        f"{output_frame} rpy={np.round(output_rpy_deg, 1).tolist()} deg",
        "axes: X=red Y=green FRONT(+Z)=magenta -Z=gray",
    ]
    for index, line in enumerate(text_lines):
        y = 35 + index * 30
        cv2.putText(right, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(right, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    combined = np.concatenate([left, right], axis=1)
    output_path = run_dir / output_filename
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"位姿可视化保存失败: {output_path}")
    return output_path


def _projected_mask_overlap(
    cad: np.ndarray,
    transformation: np.ndarray,
    mask: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> float:
    """Return the fraction of sampled CAD points projected inside the SAM2 mask."""
    points = (
        transformation[:3, :3] @ np.asarray(cad, dtype=np.float64).T
    ).T + transformation[:3, 3]
    valid_depth = points[:, 2] > 1e-4
    projected = (np.asarray(camera_intrinsics, dtype=np.float64) @ points[valid_depth].T).T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
    height, width = mask.shape[:2]
    inside_image = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    hits = int(np.count_nonzero(
        np.asarray(mask, dtype=bool)[pixels[inside_image, 1], pixels[inside_image, 0]]
    ))
    return float(hits / max(1, len(cad)))


def icp_pose(
    cad: np.ndarray,
    observed: np.ndarray,
    cfg: dict,
    *,
    selection_mask: np.ndarray | None = None,
    camera_intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    cad_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
    obs_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    radius = float(cfg.get("normal_radius_m", float(cfg["observed_voxel_m"]) * 4.0))
    cad_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    obs_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    guesses = make_pca_initial_guesses(cad, observed)
    methods = [
        ("point_to_plane", o3d.pipelines.registration.TransformationEstimationPointToPlane()),
        ("point_to_point", o3d.pipelines.registration.TransformationEstimationPointToPoint()),
    ]
    attempts = []
    quality_gate_enabled = bool(cfg.get("quality_gate_enabled", True))
    global_best = None
    all_results = []
    for method_name, estimator in methods:
        best = None
        for init_name, initial in guesses:
            result = o3d.pipelines.registration.registration_icp(
                cad_cloud, obs_cloud, float(cfg["icp_max_corr_m"]), initial, estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(cfg["icp_max_iters"])),
            )
            item = {"method": method_name, "initialization": init_name,
                    "fitness": float(result.fitness), "rmse_m": float(result.inlier_rmse)}
            transformation = np.asarray(result.transformation)
            if selection_mask is not None and camera_intrinsics is not None:
                item["mask_overlap_ratio"] = _projected_mask_overlap(
                    cad, transformation, selection_mask, camera_intrinsics
                )
            attempts.append(item)
            all_results.append((item, transformation))
            if best is None or (item["fitness"], -item["rmse_m"]) > (best[0]["fitness"], -best[0]["rmse_m"]):
                best = (item, transformation)
        if best and (
            global_best is None
            or (best[0]["fitness"], -best[0]["rmse_m"])
            > (global_best[0]["fitness"], -global_best[0]["rmse_m"])
        ):
            global_best = best
        passed = bool(
            best
            and best[0]["fitness"] >= float(cfg["quality_min_fitness"])
            and best[0]["rmse_m"] <= float(cfg["quality_max_rmse_m"])
        )
        if quality_gate_enabled and passed:
            return best[1], {
                **best[0], "attempts": attempts,
                "quality_gate_enabled": True, "quality_gate_passed": True,
            }
        if quality_gate_enabled:
            print(f"[ICP] {method_name}未通过质量门限，尝试下一方法")
    if not quality_gate_enabled and global_best is not None:
        if selection_mask is not None and camera_intrinsics is not None:
            max_fitness = max(item["fitness"] for item, _ in all_results)
            margin = float(cfg.get("icp_selection_fitness_margin", 0.03))
            eligible = [
                candidate for candidate in all_results
                if candidate[0]["fitness"] >= max_fitness - margin
            ]
            item, transformation = max(
                eligible,
                key=lambda candidate: (
                    candidate[0].get("mask_overlap_ratio", 0.0),
                    -candidate[0]["rmse_m"],
                    candidate[0]["fitness"],
                ),
            )
            item = {
                **item,
                "selection_policy": "max_mask_overlap_within_fitness_margin",
                "selection_max_fitness": float(max_fitness),
                "selection_fitness_margin": margin,
            }
        else:
            item, transformation = global_best
        passed = bool(
            item["fitness"] >= float(cfg["quality_min_fitness"])
            and item["rmse_m"] <= float(cfg["quality_max_rmse_m"])
        )
        if not passed:
            print(
                "[ICP] 警告：最佳结果未通过原质量门限，但质量门限已关闭，"
                "继续输出候选位姿"
            )
        return transformation, {
            **item,
            "attempts": attempts,
            "quality_gate_enabled": False,
            "quality_gate_passed": passed,
            "configured_quality_min_fitness": float(cfg["quality_min_fitness"]),
            "configured_quality_max_rmse_m": float(cfg["quality_max_rmse_m"]),
        }
    raise RuntimeError(f"Point-to-plane和Point-to-point均未通过质量门限: {attempts}")


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.graspup_lift_m) or args.graspup_lift_m <= 0.0:
        raise ValueError("--graspup-lift-m 必须是大于0的有限米制数值")
    if not np.isfinite(args.place_descent_m) or args.place_descent_m <= 0.0:
        raise ValueError("--place-descent-m 必须是大于0的有限米制数值")
    if not np.isfinite(args.target_xy_radius_m) or args.target_xy_radius_m <= 0.0:
        raise ValueError("--target-xy-radius-m 必须是大于0的有限米制数值")
    if args.target_object_xyz is not None and not np.all(
        np.isfinite(np.asarray(args.target_object_xyz, dtype=float))
    ):
        raise ValueError("--target-object-xyz 必须包含3个有限米制数值")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    threshold = int(args.point_threshold or cfg["face_visible_min_points"])
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    planner_cfg = LoopConfig(
        planner_host=args.planner_host, execute_url=args.execute_url, base_frame="body_base",
        left_group="left_arm", right_group="right_arm", left_tool_frame="left_hand_transfer_end",
        right_tool_frame="right_hand_transfer_end", max_velocity=args.speed,
        max_acceleration=args.speed, idle_poll_seconds=0.2, live=not args.skip_motion, poses={},
    )
    planner = PlannerRuntime(planner_cfg)
    state = LeftArmState(holding=Holding.SINGLE, grasp_stable=True)
    try:
        planner.open(required_frames={"body_base", str(cfg["planner_bridge_base_frame"])})
        print(f"[SAM2] 使用远程C/S服务: {args.sam2_url}")
        with FemtoBoltCamera(
            serial=str(cfg["camera_serial"]),
            color_width=int(cfg["color_width"]), color_height=int(cfg["color_height"]),
            depth_width=int(cfg["depth_width"]), depth_height=int(cfg["depth_height"]),
            fps=30, warmup_frames=int(cfg["warmup_frames"]),
        ) as camera:
            if not args.skip_motion:
                input("确认夹爪已夹住一个工件且路径安全，按回车移动到graspup...")
                planner.move_left_joints(GRASPUP_JOINTS_DEG)
                input("已到graspup；确认到pc1路径安全，按回车移动到pc1...")
                planner.move_left_joints([float(v) for v in cfg["pc_joints_deg"]])
            points, report = capture_view(
                camera, cfg, "pc1", run_dir, args.sam2_url, args.auto_gripper_prompt
            )
            if len(points) < threshold:
                state.observation_class = ObservationClass.OBLIQUE
                state.face_side = FaceSide.NOT_APPLICABLE
                if args.skip_motion:
                    state.observation_pose_used = "CURRENT"
                    print(
                        f"[判断] 当前位点云数 {len(points)} < {threshold}，判为侧着；"
                        "--skip-motion 模式不移动到pc2，直接使用当前点云估计位姿"
                    )
                else:
                    state.observation_pose_used = "PC_2"
                    print(f"[判断] pc1点数 {len(points)} < {threshold}，判为侧着，转pc2")
                    input("确认路径安全，按回车移动到pc2...")
                    planner.move_left_joints([float(v) for v in cfg["pc2_joints_deg"]])
                    points, report = capture_view(
                        camera, cfg, "pc2", run_dir, args.sam2_url, args.auto_gripper_prompt
                    )
            else:
                state.observation_class = ObservationClass.FACE_VISIBLE
                state.observation_pose_used = "PC"
                print(f"[判断] pc1点数 {len(points)} >= {threshold}，判为正面或反面视角")

        triangles = load_stl_triangles_m(Path(cfg["stl_path"]))
        cad = sample_surface_points(triangles, int(cfg["cad_sample_points"]), int(cfg["seed"]))
        T_camera_object, metrics = icp_pose(cad, points, cfg)
        if state.observation_class is ObservationClass.FACE_VISIBLE:
            front_normal = T_camera_object[:3, :3] @ np.array([0.0, 0.0, 1.0])
            state.face_side = FaceSide.FRONT if -front_normal[2] >= 0 else FaceSide.BACK
            state.orientation_confidence = abs(float(front_normal[2]))
        # --skip-motion 只禁止机械臂动作，不应降低位姿输出坐标系。固定胸前
        # Femto 使用直接标定的 body_base <- SDK base <- camera 变换。
        T_output_object = _load_base_camera_transform(cfg, planner) @ T_camera_object
        output_frame = "body_base"
        state.object_pose = ObjectPose(
            frame_id=output_frame,
            position=T_output_object[:3, 3].tolist(),
            orientation=Rotation.from_matrix(T_output_object[:3, :3]).as_quat().tolist(),
        )
        state.pose_confidence = float(metrics["fitness"])
        state.pose_quality = {**metrics, "valid_point_count": int(len(points)),
                              "T_camera_object": T_camera_object.tolist()}
        visualization_view = "pc2" if state.observation_pose_used == "PC_2" else "pc1"
        visualization_path = save_pose_visualization(
            run_dir,
            visualization_view,
            cad,
            T_camera_object,
            T_output_object,
            output_frame,
            np.asarray(report["camera_intrinsics"], dtype=np.float64),
        )
        result = {"observation_class": state.observation_class.value, "face_side": state.face_side.value,
                  "observation_pose_used": state.observation_pose_used,
                  "object_pose": {"frame_id": state.object_pose.frame_id,
                                  "position": state.object_pose.position,
                                  "orientation_xyzw": state.object_pose.orientation},
                  "T_camera_object": T_camera_object.tolist(), "registration": metrics,
                  "point_report": report,
                  "pose_visualization": str(visualization_path)}
        print(f"[位姿可视化] {visualization_path}")

        if not args.skip_motion and args.target_object_xyz is not None:
            tool_response = planner.lookup_transform("body_base", "left_hand_transfer_end")
            if tool_response is None:
                raise RuntimeError("规划器没有返回当前左臂末端位姿")
            T_body_tool_current = _transform_from_planner_response(tool_response)
            T_tool_object = invert_T(T_body_tool_current) @ T_output_object
            target_xyz = np.asarray(args.target_object_xyz, dtype=float)
            candidate_yaws_deg = [20, 30, 40, 50, 60, 70]
            xy_samples = [{
                "sample_id": "center",
                "radius_m": 0.0,
                "direction_deg": None,
                "xyz": target_xyz.copy(),
            }]
            for radius_m, ring_name in (
                (float(args.target_xy_radius_m) * 0.5, "inner"),
                (float(args.target_xy_radius_m), "outer"),
            ):
                for direction_deg in range(0, 360, 45):
                    angle_rad = np.deg2rad(float(direction_deg))
                    sampled_xyz = target_xyz.copy()
                    sampled_xyz[0] += radius_m * np.cos(angle_rad)
                    sampled_xyz[1] += radius_m * np.sin(angle_rad)
                    xy_samples.append({
                        "sample_id": f"{ring_name}_{direction_deg}deg",
                        "radius_m": radius_m,
                        "direction_deg": direction_deg,
                        "xyz": sampled_xyz,
                    })

            candidate_targets: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
            candidate_records = []
            for xy_sample in xy_samples:
                for yaw_deg in candidate_yaws_deg:
                    object_target = target_object_pose_for_yaw(
                        xy_sample["xyz"].tolist(), yaw_deg
                    )
                    tool_target = object_target @ invert_T(T_tool_object)
                    key = (str(xy_sample["sample_id"]), yaw_deg)
                    candidate_targets[key] = (object_target, tool_target)
                    candidate_records.append({
                        "sample_id": xy_sample["sample_id"],
                        "xy_radius_m": xy_sample["radius_m"],
                        "xy_direction_deg": xy_sample["direction_deg"],
                        "yaw_deg": yaw_deg,
                        "target_object_pose": {
                            "frame_id": "body_base",
                            "position": object_target[:3, 3].tolist(),
                            "orientation_xyzw": Rotation.from_matrix(
                                object_target[:3, :3]
                            ).as_quat().tolist(),
                        },
                        "target_tool_pose": {
                            "frame_id": "body_base",
                            "position": tool_target[:3, 3].tolist(),
                            "orientation_xyzw": Rotation.from_matrix(
                                tool_target[:3, :3]
                            ).as_quat().tolist(),
                        },
                    })
            result["placement"] = {
                "front_axis_in_stl": "+Z",
                "front_target_axis_in_body_base": "+Z",
                "yaw_policy": "try_20_30_40_50_60_70_choose_nearest_to_45",
                "xy_policy": "center_then_8_inner_ring_then_8_outer_ring_choose_nearest_center",
                "target_xy_center_m": target_xyz[:2].tolist(),
                "target_xy_radius_m": float(args.target_xy_radius_m),
                "graspup_lift_m": float(args.graspup_lift_m),
                "place_descent_m": float(args.place_descent_m),
                "T_tool_object": T_tool_object.tolist(),
                "candidate_targets": candidate_records,
                "planning_attempts": [],
                "selected_yaw_deg": None,
            }
            print("\n=== 最终位置/姿态候选（此时仅计算，尚未规划或执行） ===")
            print(f"工件目标xy圆心及固定z (m) = {target_xyz.tolist()}")
            print(
                f"xy采样：圆心 + 半径{args.target_xy_radius_m * 0.5:.3f}m的8点 + "
                f"半径{args.target_xy_radius_m:.3f}m的8点"
            )
            print("候选yaw (deg) = [20, 30, 40, 50, 60, 70]；工件+Z均对齐body_base +Z")
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[完成] 结果保存在 {run_dir}")

        if not args.skip_motion and args.target_object_xyz is not None:
            input("已计算工具-工件关系；确认路径安全，按回车回到graspup...")
            planner.move_left_joints(GRASPUP_JOINTS_DEG)

            graspup_response = planner.lookup_transform("body_base", "left_hand_transfer_end")
            if graspup_response is None:
                raise RuntimeError("规划器没有返回graspup处的左臂末端位姿")
            T_body_tool_graspup = _transform_from_planner_response(graspup_response)
            T_body_tool_above = T_body_tool_graspup.copy()
            T_body_tool_above[2, 3] += float(args.graspup_lift_m)
            input(
                f"确认沿body_base +Z上移{args.graspup_lift_m:.3f}m路径安全，按回车规划LIN并执行..."
            )
            planner.move_cartesian_pose(
                "left", matrix_to_object_pose(T_body_tool_above), planner="lin"
            )

            print("\n=== 分层测试xy/yaw最终PTP候选（只规划，不执行） ===")
            planning_attempts = []
            successful_candidates = []
            tested_radius_m = None
            for radius_m in (0.0, float(args.target_xy_radius_m) * 0.5,
                             float(args.target_xy_radius_m)):
                radius_successes = []
                ring_samples = [
                    sample for sample in xy_samples
                    if abs(float(sample["radius_m"]) - radius_m) < 1e-9
                ]
                print(f"\n>>> 测试xy半径层 {radius_m:.3f}m，共{len(ring_samples)}个位置")
                for xy_sample in ring_samples:
                    sample_id = str(xy_sample["sample_id"])
                    object_xyz = np.asarray(xy_sample["xyz"], dtype=float)
                    for yaw_deg in candidate_yaws_deg:
                        object_target, tool_target = candidate_targets[(sample_id, yaw_deg)]
                        tool_quaternion = Rotation.from_matrix(
                            tool_target[:3, :3]
                        ).as_quat()
                        tool_rpy_deg = Rotation.from_matrix(
                            tool_target[:3, :3]
                        ).as_euler("xyz", degrees=True)
                        print(
                            f"\n--- 候选位置={sample_id}, 工件xyz="
                            f"{np.round(object_xyz, 6).tolist()}, yaw={yaw_deg}deg ---"
                        )
                        print("frame_id = body_base")
                        print("ik_frame = left_hand_transfer_end")
                        print(
                            "末端xyz (m) = "
                            f"{np.array2string(tool_target[:3, 3], precision=6, separator=', ')}"
                        )
                        print(
                            "末端RPY xyz (deg) = "
                            f"{np.array2string(tool_rpy_deg, precision=3, separator=', ')}"
                        )
                        print(
                            "末端quaternion xyzw = "
                            f"{np.array2string(tool_quaternion, precision=6, separator=', ')}"
                        )
                        try:
                            planned = planner.plan_cartesian_pose(
                                "left", matrix_to_object_pose(tool_target), planner="ptp"
                            )
                        except Exception as exc:
                            error_text = f"{type(exc).__name__}: {exc}"
                            planning_attempts.append({
                                "sample_id": sample_id,
                                "xy_radius_m": radius_m,
                                "object_xyz_m": object_xyz.tolist(),
                                "yaw_deg": yaw_deg,
                                "success": False,
                                "error": error_text,
                            })
                            print(
                                f"[候选 {sample_id}/{yaw_deg}deg] 规划失败: {error_text}"
                            )
                        else:
                            if planned is None:
                                raise RuntimeError("live模式下规划器返回了空规划结果")
                            planning_attempts.append({
                                "sample_id": sample_id,
                                "xy_radius_m": radius_m,
                                "object_xyz_m": object_xyz.tolist(),
                                "yaw_deg": yaw_deg,
                                "success": True,
                                "error": "",
                            })
                            candidate = (
                                radius_m,
                                yaw_deg,
                                sample_id,
                                object_target,
                                tool_target,
                                planned,
                            )
                            radius_successes.append(candidate)
                            print(
                                f"[候选 {sample_id}/{yaw_deg}deg] 规划成功（尚未执行）"
                            )
                if radius_successes:
                    successful_candidates = radius_successes
                    tested_radius_m = radius_m
                    print(
                        f"半径层{radius_m:.3f}m已有可行解，"
                        "按离圆心最近规则不再测试更远半径层。"
                    )
                    break

            result["placement"]["planning_attempts"] = planning_attempts
            if not successful_candidates:
                (run_dir / "result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                raise RuntimeError(
                    "圆心、半径一半内圈和允许半径外圈的全部采样候选均规划失败，"
                    "未执行最终运动"
                )

            (selected_radius_m, selected_yaw_deg, selected_sample_id,
             T_body_object_target, T_body_tool_target, selected_plan) = min(
                successful_candidates,
                key=lambda item: (item[0], abs(item[1] - 45), item[1], item[2]),
            )
            target_tool_quaternion = Rotation.from_matrix(
                T_body_tool_target[:3, :3]
            ).as_quat()
            target_tool_rpy_deg = Rotation.from_matrix(
                T_body_tool_target[:3, :3]
            ).as_euler("xyz", degrees=True)
            result["placement"]["selected_yaw_deg"] = selected_yaw_deg
            result["placement"]["selected_sample_id"] = selected_sample_id
            result["placement"]["selected_xy_radius_m"] = selected_radius_m
            result["placement"]["target_object_pose"] = {
                "frame_id": "body_base",
                "position": T_body_object_target[:3, 3].tolist(),
                "orientation_xyzw": Rotation.from_matrix(
                    T_body_object_target[:3, :3]
                ).as_quat().tolist(),
            }
            result["placement"]["target_tool_pose"] = {
                "frame_id": "body_base",
                "position": T_body_tool_target[:3, 3].tolist(),
                "orientation_xyzw": target_tool_quaternion.tolist(),
            }
            (run_dir / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

            successful_labels = [f"{item[2]}/{item[1]}deg" for item in successful_candidates]
            print(f"\n最近可行半径层={tested_radius_m:.3f}m")
            print(f"该层规划成功候选={successful_labels}")
            print(
                f"最终选择位置={selected_sample_id}, 工件xyz="
                f"{np.round(T_body_object_target[:3, 3], 6).tolist()}, "
                f"yaw={selected_yaw_deg}deg"
            )
            print("=== 即将执行选中的最终PTP轨迹 ===")
            print(
                "xyz (m) = "
                f"{np.array2string(T_body_tool_target[:3, 3], precision=6, separator=', ')}"
            )
            print(
                "RPY xyz (deg) = "
                f"{np.array2string(target_tool_rpy_deg, precision=3, separator=', ')}"
            )
            print(
                "quaternion xyzw = "
                f"{np.array2string(target_tool_quaternion, precision=6, separator=', ')}"
            )
            input("确认选中轨迹及目标区域安全，按回车执行该PTP轨迹...")
            planner.execute_planned_result(selected_plan)
            print(
                f"[完成] 工件已规划到目标位置：yaw={selected_yaw_deg}deg，"
                "+Z对齐body_base +Z"
            )

            T_body_tool_release = T_body_tool_target.copy()
            T_body_tool_release[2, 3] -= float(args.place_descent_m)
            release_rpy_deg = Rotation.from_matrix(
                T_body_tool_release[:3, :3]
            ).as_euler("xyz", degrees=True)
            release_quaternion = Rotation.from_matrix(
                T_body_tool_release[:3, :3]
            ).as_quat()
            print("\n=== 即将下降到松爪位姿 ===")
            print("planner = lin")
            print("frame_id = body_base")
            print("ik_frame = left_hand_transfer_end")
            print(
                "xyz (m) = "
                f"{np.array2string(T_body_tool_release[:3, 3], precision=6, separator=', ')}"
            )
            print(
                "RPY xyz (deg) = "
                f"{np.array2string(release_rpy_deg, precision=3, separator=', ')}"
            )
            print(
                "quaternion xyzw = "
                f"{np.array2string(release_quaternion, precision=6, separator=', ')}"
            )
            input(
                f"确认沿body_base -Z下降{args.place_descent_m:.3f}m路径安全，"
                "按回车规划LIN并执行..."
            )
            planner.move_cartesian_pose(
                "left", matrix_to_object_pose(T_body_tool_release), planner="lin"
            )

            print(">>> 下降到位，正在同步打开左夹爪到100%...")
            from parm_sdk import Gripper

            with Gripper(host=args.gripper_host) as gripper:
                hand_state = gripper.get_state().get(1)
                if hand_state is None:
                    raise RuntimeError("左夹爪hand=1没有状态帧或未连接")
                if hand_state.get("initialized") is not True:
                    raise RuntimeError("左夹爪尚未初始化，禁止发送打开指令")
                response = gripper.move(
                    1,
                    100.0,
                    force=100.0,
                    mode="sync",
                    preempt=True,
                    timeout=60.0,
                )
            final_position = (response.get("data") or {}).get("position")
            print(f"[完成] 左夹爪已打开，最终开度={final_position}%")
        elif not args.skip_motion:
            print("[完成] 未提供--target-object-xyz：仅完成模块2识别，未执行graspup或放置运动。")
    finally:
        planner.close()


if __name__ == "__main__":
    main()
