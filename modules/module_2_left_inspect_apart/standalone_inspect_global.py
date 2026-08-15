#!/usr/bin/env python3
"""通用左臂 Inspect：所有工件共用 PC1 -> PC2、SAM2 和 ICP 流程。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import http.client
import io
import json
import math
import os
from pathlib import Path
import select
import socket
import sys
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loop_config import LoopConfig
from loop_types import ObjectPose
from modules.stage_result import write_stage_result
from modules.module_0_prepare.ensure_sam2_service import require_sam2_service
from tool.inspection_support.mask_depth import (
    mask_depth_diagnostics,
    mask_depth_to_points_cam,
    remove_radius_outliers_np,
    voxel_downsample_np,
)
from tool.inspection_support.registration import (
    make_pca_initial_guesses,
    register_cad_to_observed_candidates,
)
from tool.camera_broker import BrokerFemtoCamera
from tool.inspection_support.stl_model import (
    load_stl_triangles_m,
    sample_surface_points,
    triangles_in_object_frame,
)
from tool.planner_runtime import PlannerRuntime
from tool.runtime_log import RuntimeLogCapture
from tool.inspection_support.legacy_inspect.capture_selected_rgbd import capture_robot_state
from tool.inspection_support.legacy_inspect.generate_pc2_sam2_depth_prompts import (
    far_depth_negative_points,
    load_fixed_negatives,
    positive_points,
    usable_components,
    wide_depth_core,
)
from tool.inspection_support.legacy_inspect.remove_pc1_urdf_gripper_depth import tool_matrix
from tool.inspection_support.legacy_inspect.remove_pc2_urdf_gripper_depth import (
    load_self_surface_points,
    matrix,
    projected_gripper_depth,
)


CONFIG_PATH = Path(__file__).with_name("inspect_config.json")
DEFAULT_OUTPUT_DIR = ROOT / "output/module_2_left_inspect_apart"
PC1_FIXED_NEGATIVES_PATH = Path(__file__).with_name("pc1_fixed_negative_points.json")
WORKPIECE_2_STL_PATH = ROOT / "tool/workpiece_2/workpiece_2.stl"
WORKPIECE_2_R_OBJECT_STL = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
DEFAULT_URDF_DESCRIPTION_ROOT = Path(
    "/home/dainanjun/Public/plucky_hjt_parm/welding_config/description"
)
DEFAULT_LEFT_ARM_KINEMATICS = Path(
    "/home/dainanjun/Public/plucky_hjt_parm/arm_configs/arm_kinematics/left.yaml"
)
GRIPPER_HOLD_STABLE_S = 0.5
GRIPPER_HOLD_POLL_S = 0.1
GRIPPER_HOLD_TIMEOUT_S = 2.0


def load_object_triangles(cfg: dict) -> np.ndarray:
    triangles = load_stl_triangles_m(Path(cfg["stl_path"]))
    if cfg.get("R_object_stl") is None:
        return triangles
    return triangles_in_object_frame(
        triangles,
        R_object_stl=cfg["R_object_stl"],
        object_origin_stl_m=cfg.get("object_origin_stl_m", [0.0, 0.0, 0.0]),
    )


class ICPRegistrationError(RuntimeError):
    """Carry rejected ICP metrics into the PC fallback log."""

    def __init__(self, message: str, metrics: dict) -> None:
        super().__init__(message)
        self.metrics = metrics


def _projected_mask_consistency(
    cad: np.ndarray, transformation: np.ndarray, selection_mask: np.ndarray,
    camera_intrinsics: np.ndarray, dilation_px: int,
) -> dict[str, float | int]:
    mask = np.asarray(selection_mask) > 0
    height, width = mask.shape
    transformed = (transformation[:3, :3] @ cad.T).T + transformation[:3, 3]
    transformed = transformed[transformed[:, 2] > 1e-6]
    if not len(transformed):
        return {"projected_cad_pixels": 0, "mask_near_cad_ratio": 0.0,
                "projected_iou": 0.0, "mask_to_cad_median_px": float("inf")}
    projected = (camera_intrinsics @ transformed.T).T
    u = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int32)
    v = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    rendered = np.zeros((height, width), dtype=np.uint8)
    rendered[v[inside], u[inside]] = 1
    if not np.any(rendered) or not np.any(mask):
        return {"projected_cad_pixels": int(np.count_nonzero(rendered)), "mask_near_cad_ratio": 0.0,
                "projected_iou": 0.0, "mask_to_cad_median_px": float("inf")}
    kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), np.uint8)
    dilated = cv2.dilate(rendered, kernel) > 0
    distance = cv2.distanceTransform((~(rendered > 0)).astype(np.uint8), cv2.DIST_L2, 3)
    mask_distances = distance[mask]
    union = int(np.count_nonzero(dilated | mask))
    return {
        "projected_cad_pixels": int(np.count_nonzero(rendered)),
        "mask_near_cad_ratio": float(np.mean(mask_distances <= dilation_px)),
        "projected_iou": float(np.count_nonzero(dilated & mask) / union) if union else 0.0,
        "mask_to_cad_median_px": float(np.median(mask_distances)),
    }


def _icp_pose(
    cad: np.ndarray, observed: np.ndarray, cfg: dict, selection_mask: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """统一执行 Point-to-Plane/Point-to-Point ICP 和二维投影质量门限。"""
    cad_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
    obs_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
    radius = float(cfg["observed_voxel_m"]) * 4.0
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    cad_cloud.estimate_normals(search)
    obs_cloud.estimate_normals(search)
    attempts: list[dict] = []
    dilation_px = int(cfg["icp_projection_dilation_px"])
    for method_name, estimator in (
        ("point_to_plane", o3d.pipelines.registration.TransformationEstimationPointToPlane()),
        ("point_to_point", o3d.pipelines.registration.TransformationEstimationPointToPoint()),
    ):
        accepted: list[tuple[dict, np.ndarray]] = []
        for init_name, initial in make_pca_initial_guesses(cad, observed):
            result = o3d.pipelines.registration.registration_icp(
                cad_cloud, obs_cloud, float(cfg["icp_max_corr_m"]), initial, estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(cfg["icp_max_iters"])),
            )
            item = {"method": method_name, "initialization": init_name,
                    "fitness": float(result.fitness), "rmse_m": float(result.inlier_rmse)}
            item["passed_3d"] = item["fitness"] >= float(cfg["quality_min_fitness"]) and item["rmse_m"] <= float(cfg["quality_max_rmse_m"])
            item.update(_projected_mask_consistency(cad, np.asarray(result.transformation), selection_mask, camera_intrinsics, dilation_px))
            item["passed_2d"] = (
                item["mask_near_cad_ratio"] >= float(cfg["icp_min_mask_near_cad_ratio"])
                and item["projected_iou"] >= float(cfg["icp_min_projected_iou"])
                and item["mask_to_cad_median_px"] <= float(cfg["icp_max_mask_to_cad_median_px"])
            )
            item["quality_passed"] = bool(item["passed_3d"] and item["passed_2d"])
            attempts.append(item)
            if item["quality_passed"]:
                accepted.append((item, np.asarray(result.transformation)))
            else:
                print(f"[ICP] 拒绝 {method_name}/{init_name}: fitness={item['fitness']:.4f}, rmse={item['rmse_m'] * 1000:.2f}mm, 二维通过={item['passed_2d']}", flush=True)
        if accepted:
            best, transform = max(accepted, key=lambda pair: (pair[0]["mask_near_cad_ratio"], pair[0]["projected_iou"], pair[0]["fitness"], -pair[0]["rmse_m"]))
            print(f"[ICP] 选中 {method_name}/{best['initialization']}；其余候选已完成检查", flush=True)
            return transform, {**best, "projection_validation_enabled": True, "attempts": attempts}
        print(f"[ICP] {method_name}所有初值均被拒绝，继续尝试下一方法", flush=True)
    raise ICPRegistrationError("Point-to-plane和Point-to-point均未通过三维/二维质量门限", {"projection_validation_enabled": True, "attempts": attempts})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workpiece",
        choices=("1", "2"),
        required=True,
        help="仅选择真实工件对应的 STL 与固定坐标轴变换；不改变统一视觉流程",
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"))
    parser.add_argument("--gripper-hand", type=int, default=1)
    parser.add_argument("--execute-url", default="http://127.0.0.1:5667/middle/waypoint/execute")
    parser.add_argument("--sam2-url", default=os.environ.get(
        "PARM_SAM2_URL",
        "unix:///home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service/runtime/sam2-service.sock",
    ))
    parser.add_argument(
        "--camera-broker-socket",
        default=os.environ.get("CAMERA_BROKER_SOCKET", "/tmp/graspnet-v7-camera.sock"),
        help="Module 0 Camera Broker Unix socket; Module 2 never opens Femto directly",
    )
    parser.add_argument(
        "--urdf-description-root",
        type=Path,
        default=DEFAULT_URDF_DESCRIPTION_ROOT,
        help="用于自遮挡投影的 URDF description 根目录",
    )
    parser.add_argument(
        "--left-arm-kinematics",
        type=Path,
        default=DEFAULT_LEFT_ARM_KINEMATICS,
        help="新 URDF 中 left_calib_origin 使用的左臂标定 YAML",
    )
    parser.add_argument(
        "--urdf-active-joint-max-rad",
        type=float,
        default=0.8,
        help="夹爪完全闭合时 URDF 主动关节角；Robotiq 2F-85 默认 0.8rad",
    )
    left_scale = float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0"))
    parser.add_argument("--preset-speed", type=float, default=0.2 * left_scale)
    parser.add_argument("--motion-speed", type=float, default=0.1 * left_scale)
    parser.add_argument("--point-threshold", type=int, default=700)
    parser.add_argument(
        "--pc1-near-min-m", type=float, default=0.100,
        help="PC1 自动提示近深度下限（默认 0.100 m）",
    )
    parser.add_argument(
        "--pc1-near-max-m", type=float, default=0.300,
        help="PC1 自动提示近深度上限（默认 0.300 m）",
    )
    parser.add_argument(
        "--pc1-workpiece2-positive-inset-px", type=float, default=25.0,
        help="工作件2在PC1主候选内缩取正点的最小边界距离（默认25px）",
    )
    parser.add_argument(
        "--pc1-workpiece2-tail-second-sam2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="工作件2的PC1在主候选未被首轮SAM2覆盖时启用尾端局部SAM2（默认开启）",
    )
    parser.add_argument(
        "--pc1-workpiece2-tail-roi-margin-px", type=int, default=55,
        help="工作件2 PC1尾端局部SAM2 ROI扩展量（默认55px）",
    )
    parser.add_argument("--pc2-near-min-m", type=float, default=0.110)
    parser.add_argument("--pc2-near-max-m", type=float, default=0.280)
    parser.add_argument("--auto-positive-count", type=int, default=6)
    parser.add_argument(
        "--colour-difference-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "当棕色、银色自动提示均未通过时，以同帧 URDF 确认机械臂的 "
            "LAB 相对色差生成额外提示；默认开启"
        ),
    )
    parser.add_argument(
        "--pc1-candidate-opening-kernel-px", type=int, default=0,
        help="仅在 PC1 首轮提示候选上执行的开运算核大小；0 为关闭",
    )
    parser.add_argument(
        "--pc1-roi-margin-px", type=int, default=50,
        help="兼容旧参数名：全局自动 SAM2 ROI 在可信候选包围盒外的余量（默认 50px）",
    )
    parser.add_argument(
        "--global-urdf-component-overlap-ratio", type=float, default=0.55,
        help="候选区域被判为机械臂所需的最小完整 URDF 投影重叠率（默认 0.55）",
    )
    parser.add_argument(
        "--global-urdf-component-deep-overlap-ratio", type=float, default=0.35,
        help="候选区域被判为机械臂所需的 URDF 投影深处重叠率（默认 0.35）",
    )
    parser.add_argument(
        "--global-urdf-component-interior-px", type=int, default=8,
        help="距完整 URDF 投影边界至少该距离才算投影深处（默认 8px）",
    )
    parser.add_argument(
        "--global-anchor-contact-max-px", type=float, default=80.0,
        help="优先从距夹爪完整投影该距离内选择工件主体（默认 80px）",
    )
    parser.add_argument(
        "--global-rejected-negative-count", type=int, default=8,
        help="最多把多少个被拒绝候选作为额外 SAM2 负点（默认 8）",
    )
    parser.add_argument(
        "--pc1-negative-depth-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "PC1 的负点若靠近可信工件支持区且深度相近则取消，"
            "避免固定坐标落到工件上（默认开启）"
        ),
    )
    parser.add_argument(
        "--pc1-negative-depth-guard-margin-m", type=float, default=0.020,
        help="PC1 负点与正点支持深度分位区间的安全余量（默认 20mm）",
    )
    parser.add_argument(
        "--pc1-negative-depth-guard-distance-px", type=float, default=60.0,
        help="PC1 负点距可信工件支持区不超过该距离时才启用深度取消（默认 60px）",
    )
    parser.add_argument(
        "--pc1-negative-depth-guard-positive-radius-px", type=int, default=31,
        help="PC1 负点深度保护统计每个正点周围的半径（默认 31px）",
    )
    parser.add_argument(
        "--global-close-fragment-depth-diff-m", type=float, default=0.035,
        help="近距离碎片相对工件主体深度四分位区间的最大距离（默认 35mm）",
    )
    parser.add_argument(
        "--global-brown-urdf-front-black-max-rgb", type=int, default=45,
        help="brown 自动候选中，URDF 投影内且近黑的前景像素直接剔除（默认 45）",
    )
    parser.add_argument(
        "--global-exclude-urdf-front-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="自动候选中剔除 URDF 投影内、实测比 URDF 更近至少 15mm 的像素（默认关闭）",
    )
    parser.add_argument("--pc2-axial-tail-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pc2-axial-tail-max-behind-m", type=float, default=0.060)
    parser.add_argument(
        "--global-mask-max-deep-urdf-ratio", type=float, default=0.35,
        help="自动最终 mask 位于 URDF 投影深处的最大比例（默认 0.35）",
    )
    parser.add_argument(
        "--global-mask-max-outside-roi-ratio", type=float, default=0.10,
        help="自动最终 mask 超出可信候选 ROI 的最大比例（默认 0.10）",
    )
    parser.add_argument(
        "--pc1-left-gripper-colour-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "PC1 最终 mask 禁止进入夹爪左侧易误检区，只有棕色或银色金属像素例外；"
            "默认开启；若门控会破坏正点工件连通域，则自动回退到门控前 mask"
        ),
    )
    parser.add_argument("--pc1-urdf-barrier-signed-depth-m", type=float, default=0.120)
    parser.add_argument("--pc1-urdf-barrier-min-area-px", type=int, default=200)
    parser.add_argument("--pc1-workpiece2-expand-px", type=int, default=8)
    parser.add_argument("--pc1-workpiece2-expand-color-diff", type=int, default=30)
    parser.add_argument("--pc1-workpiece2-expand-depth-diff-m", type=float, default=0.015)
    parser.add_argument(
        "--workpiece2-auto-near-min-m", type=float, default=0.100,
        help="兼容旧参数名：统一自动 SAM2 首轮深度下限（默认 0.100m）",
    )
    parser.add_argument(
        "--workpiece2-auto-near-max-m", type=float, default=0.280,
        help="兼容旧参数名：统一自动 SAM2 首轮深度上限（默认 0.280m）",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("auto", "manual", "auto-then-manual"),
        default=None,
        help=(
            "SAM2 提示方式；auto=仅自动，manual=手动ROI/正负点，"
            "auto-then-manual=自动提示失败后手动；全局版统一默认auto"
        ),
    )
    parser.add_argument(
        "--pc2-gripper-box-margin-px", type=int, default=50,
        help="PC2 ROI 朝最近夹爪投影单边扩展的余量（默认 50px）",
    )
    parser.add_argument(
        "--urdf-self-depth-tolerance-m", type=float, default=0.035,
        help="URDF 与实测深度的绝对一致性阈值（默认 35mm）",
    )
    parser.add_argument(
        "--urdf-self-dilation-kernel-px", type=int, default=3,
        help="深度确认后的 URDF 自遮挡膨胀核边长；必须为正奇数（全局版默认 3）",
    )
    parser.add_argument(
        "--urdf-near-black-max-rgb", type=int, default=None,
        help="统一近黑色判定的 RGB 三通道最大值（默认 35；小于0时关闭）",
    )
    parser.add_argument(
        "--urdf-near-black-depth-tolerance-m", type=float, default=0.250,
        help="仅用于首轮候选过滤时、URDF 投影内近黑色像素的深度容差（默认 250mm）",
    )
    parser.add_argument(
        "--urdf-far-black-depth-tolerance-m", type=float, default=0.300,
        help="URDF 预测深度较远时近黑色像素的最大深度容差（默认 300mm）",
    )
    parser.add_argument(
        "--urdf-far-black-depth-start-m", type=float, default=0.250,
        help="达到该 URDF 预测深度后，近黑色像素使用最大容差（默认 0.250m）",
    )
    parser.add_argument(
        "--urdf-projected-nearer-exclusion-m", type=float, default=0.125,
        help="首轮候选中 URDF 预测比实测更远至少该值时直接排除（默认 125mm）",
    )
    parser.add_argument(
        "--mask-positive-depth-tolerance-m", type=float, default=0.050,
        help="SAM2 后仅剔除比正点中位深度更远的像素，容差默认 50mm",
    )
    parser.add_argument(
        "--workpiece2-extended-mask-depth-tolerance-m",
        type=float,
        default=0.120,
        help="工件2扩展模式在正点后方保留的最大深度，默认120mm",
    )
    parser.add_argument(
        "--manual-icp-nearest-depth-front-m", type=float, default=0.030,
        help="手动 SAM2 mask 转 ICP 点云时，距 mask 最近深度向相机侧保留的范围（默认 30mm）",
    )
    parser.add_argument(
        "--manual-icp-nearest-depth-back-m", type=float, default=0.100,
        help="手动 SAM2 mask 转 ICP 点云时，距 mask 最近深度背向相机侧保留的范围（默认 100mm）",
    )
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-handoff-file", type=Path)
    parser.add_argument("--skip-motion", action="store_true")
    args = parser.parse_args()
    if args.prompt_mode is None:
        # 所有观察位和工件都使用同一套自动提示逻辑。
        args.prompt_mode = "auto"
    if args.urdf_near_black_max_rgb is None:
        args.urdf_near_black_max_rgb = 35
    if args.manual_icp_nearest_depth_front_m < 0:
        raise ValueError("--manual-icp-nearest-depth-front-m 不能小于 0")
    if args.manual_icp_nearest_depth_back_m <= 0:
        raise ValueError("--manual-icp-nearest-depth-back-m 必须大于 0")
    # ``capture_robot_state`` writes a degree-normalized snapshot and expects
    # this field even though it is not otherwise part of Apart's CLI.
    args.joint_unit = "deg"
    return args


def _planner_matrix(response: dict) -> np.ndarray:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [translation.get(axis) for axis in "xyz"]
    xyzw = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in xyz + xyzw):
        raise RuntimeError(f"规划器 TF 返回格式无效: {response}")
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def _body_camera_matrix(cfg: dict) -> np.ndarray:
    alignment = json.loads(
        Path(cfg["sdk_base_alignment_path"]).read_text(encoding="utf-8")
    )
    chest = json.loads(
        Path(cfg["chest_calibration_path"]).read_text(encoding="utf-8")
    )
    if chest.get("status") != "ACCEPTED":
        raise RuntimeError(f"胸前 Femto 标定未通过: {chest.get('status')!r}")
    T_body_sdk = np.asarray(alignment["T_body_base_sdk_base"], dtype=float)
    T_sdk_camera = np.asarray(chest["T_base_laser"], dtype=float)
    for name, matrix in (
        ("T_body_base_sdk_base", T_body_sdk),
        ("T_base_laser", T_sdk_camera),
    ):
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"{name} 不是有限 4x4 变换")
    return T_body_sdk @ T_sdk_camera


def _sam2_prompt(
    image: np.ndarray,
    box: list[int],
    positives: list[list[int]],
    negatives: list[list[int]],
    endpoint_value: str,
) -> np.ndarray:
    endpoint = urlparse(endpoint_value)
    points = np.asarray(positives + negatives, dtype=np.int32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    payload = io.BytesIO()
    np.savez_compressed(
        payload,
        image_bgr=image,
        points=points,
        labels=labels,
        box=np.asarray(box, dtype=np.int32),
    )

    if endpoint.scheme == "unix":
        class UnixConnection(http.client.HTTPConnection):
            def connect(self) -> None:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(endpoint.path)

        connection: http.client.HTTPConnection = UnixConnection(
            "sam2-service", timeout=90.0
        )
        request_path = "/v1/prompt-segment-sam2"
    elif endpoint.scheme in {"http", "https"} and endpoint.hostname:
        connection_type = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            endpoint.hostname, port=endpoint.port, timeout=90.0
        )
        request_path = endpoint.path or "/v1/prompt-segment-sam2"
    elif hypothesis == "colour_difference":
        # This is a fallback specifically intended to reject residual gripper
        # pixels, so require stronger robot exclusion than brown/silver.
        minimum_support_coverage = 0.55
        minimum_plausible_coverage = 0.40
        maximum_unexplained_robot_ratio = 0.20
    else:
        raise ValueError(f"无效 SAM2 地址: {endpoint_value!r}")
    try:
        connection.request(
            "POST",
            request_path,
            body=payload.getvalue(),
            headers={"Content-Type": "application/x-sam2-prompt-npz"},
        )
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(
                f"SAM2 返回 HTTP {response.status}: {body.decode(errors='replace')}"
            )
    finally:
        connection.close()
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return (archive["mask"] > 0).astype(np.uint8) * 255


def _pc1_fixed_negative_points(rgb: np.ndarray) -> list[list[int]]:
    """Load dark fixed PC1 negatives valid for this particular RGB frame."""
    spec = json.loads(PC1_FIXED_NEGATIVES_PATH.read_text(encoding="utf-8"))
    height, width = rgb.shape[:2]
    configured_width, configured_height = spec["camera_resolution"]
    if (width, height) != (configured_width, configured_height):
        raise RuntimeError(
            "PC1 fixed negative prompts require "
            f"{configured_width}x{configured_height} RGB, got {width}x{height}"
        )
    points = [[int(x), int(y)] for x, y in spec["negative_points"]]
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise RuntimeError("PC1 fixed negative prompt is outside the image")
    # Fixed prompts belong to the dark robot/gripper connection only.  In a
    # changed pose, a point landing on a bright object/background must not
    # become a false negative prompt.
    return [point for point in points if int(np.max(rgb[point[1], point[0]])) < 40]


def _main_mask_with_filled_holes(
    mask: np.ndarray, prompt_positives: list[list[int]] | None = None,
    fill_holes: bool = True,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Keep the target component(s) and fill only their enclosed holes.

    PC1 has one compact target, so it retains the largest component.  PC2 can
    show two target lobes separated by the gripper; when automatic positives
    land on both, retain each positive-supported component instead.
    """
    foreground = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return np.zeros_like(foreground, dtype=np.uint8), {
            "operation": "target_components_then_fill_enclosed_holes",
            "raw_mask_pixels": int(foreground.sum()),
            "retained_component_pixels_before_hole_fill": 0,
            "filled_hole_pixels": 0,
            "final_mask_pixels": 0,
        }
    supported_labels = {
        int(labels[y, x])
        for x, y in prompt_positives or []
        if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0] and labels[y, x]
    }
    if supported_labels:
        main = np.isin(labels, list(supported_labels)).astype(np.uint8)
        selection = "components_containing_positive_prompts"
    else:
        main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        main = (labels == main_label).astype(np.uint8)
        selection = "largest_connected_component"
    if not fill_holes:
        cleaned = main.astype(np.uint8) * 255
        return cleaned, {
            "operation": "target_components_without_hole_fill",
            "component_selection": selection,
            "retained_component_count": int(len(supported_labels) or 1),
            "raw_mask_pixels": int(foreground.sum()),
            "retained_component_pixels_before_hole_fill": int(main.sum()),
            "filled_hole_pixels": 0,
            "final_mask_pixels": int((cleaned > 0).sum()),
        }
    padded = cv2.copyMakeBorder(main, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    external = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(external, flood_mask, (0, 0), 1)
    holes = ((external == 0) & (padded == 0)).astype(np.uint8)[1:-1, 1:-1]
    cleaned = np.maximum(main, holes).astype(np.uint8) * 255
    return cleaned, {
        "operation": "target_components_then_fill_enclosed_holes",
        "component_selection": selection,
        "retained_component_count": int(len(supported_labels) or 1),
        "raw_mask_pixels": int(foreground.sum()),
        "retained_component_pixels_before_hole_fill": int(main.sum()),
        "filled_hole_pixels": int(holes.sum()),
        "final_mask_pixels": int((cleaned > 0).sum()),
    }


def _recover_pc2_axial_tail(
    main_mask: np.ndarray, raw_mask: np.ndarray, depth: np.ndarray, scale: float,
    positives: list[list[int]], self_pixels: np.ndarray, max_behind_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recover only a shallow, end-on continuation of the visible PC2 part."""
    main = main_mask > 0
    ys, xs = np.where(main)
    if xs.size < 200:
        return main_mask, {"accepted_pixels": 0, "reason": "insufficient_main"}
    xy = np.column_stack((xs, ys)).astype(np.float64)
    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    axis = vh[0]
    along_main = (xy - center) @ axis
    span = float(np.ptp(along_main))
    if span < 30.0:
        return main_mask, {"accepted_pixels": 0, "reason": "short_main"}
    source, _ = _filter_mask_by_positive_depth(
        raw_mask, depth, scale, positives, max_behind_m, 0.05,
        preserve_invalid_depth=False,
    )
    source = (source > 0) & ~self_pixels & ~main
    sy, sx = np.where(source)
    if not sx.size:
        return main_mask, {"accepted_pixels": 0, "reason": "no_source"}
    delta = np.column_stack((sx, sy)).astype(np.float64) - center
    along = delta @ axis
    perp = np.abs(delta[:, 0] * axis[1] - delta[:, 1] * axis[0])
    tail = np.zeros_like(main)
    tail[sy, sx] = (
        (perp <= max(35.0, 0.28 * span))
        & ((along < along_main.min() - 4.0) | (along > along_main.max() + 4.0))
    )
    near = cv2.dilate(main.astype(np.uint8), np.ones((101, 101), np.uint8)) > 0
    tail &= near
    count, labels, stats, _ = cv2.connectedComponentsWithStats(tail.astype(np.uint8), 8)
    keep = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= 200]
    tail = np.isin(labels, keep)
    merged = (main | tail).astype(np.uint8) * 255
    return merged, {"accepted_pixels": int(tail.sum()), "component_count": len(keep), "axis": axis.tolist(), "span_px": span}


def _largest_mask_components(
    mask: np.ndarray, maximum_components: int = 3, minimum_area: int = 500,
) -> tuple[np.ndarray, dict[str, object]]:
    """Retain a few largest SAM components after severe positive-component collapse."""
    foreground = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    candidates = [
        label for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(minimum_area)
    ]
    candidates.sort(
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True
    )
    kept = candidates[:int(maximum_components)]
    recovered = np.isin(labels, kept).astype(np.uint8) * 255
    recovered, fill_report = _fill_small_enclosed_holes(recovered)
    return recovered, {
        "operation": "largest_sam_components_after_positive_component_collapse",
        "kept_labels": [int(label) for label in kept],
        "kept_areas": [int(stats[label, cv2.CC_STAT_AREA]) for label in kept],
        "maximum_components": int(maximum_components),
        "minimum_area": int(minimum_area),
        "final_mask_pixels": int(np.count_nonzero(recovered)),
        "hole_fill": fill_report,
    }


def _recover_pc1_workpiece2_tail_with_sam2(
    raw_mask: np.ndarray,
    auto: dict,
    rgb: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    """Ask SAM2 once more for a large trusted PC1 lobe missed by its first pass.

    This is deliberately a local *semantic* recovery: RGB-D supplies only the
    trusted target area and validates the returned pixels; it never paints the
    missing tail into the final mask on its own.
    """
    support = auto["debug_masks"]["prompt_support"].astype(bool)
    raw = raw_mask > 0
    # First preference: an automatically generated interior positive that
    # SAM2 itself missed.  It is stronger evidence than a large generic RGB-D
    # island, and a small ROI around it is markedly less ambiguous for a long
    # stamping tail than the full PC1 box.
    uncovered_positives = [
        [int(x), int(y)] for x, y in auto["positives"]
        if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] and not raw[y, x]
    ]
    if uncovered_positives:
        seed = uncovered_positives[0]
        other_points = sorted(
            (point for point in auto["positives"] if point != seed),
            key=lambda point: (point[0] - seed[0]) ** 2 + (point[1] - seed[1]) ** 2,
        )
        local_positives = [seed] + other_points[:1]
        x, y = seed
        box = [
            max(0, x - 150), max(0, y - 120),
            min(raw.shape[1] - 1, x + 150), min(raw.shape[0] - 1, y + 160),
        ]
        negatives = [
            point for point in auto["negatives"]
            if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        ]
        second_raw = _sam2_prompt(rgb, box, local_positives, negatives, args.sam2_url)
        second_main, _ = _main_mask_with_filled_holes(
            second_raw, local_positives, fill_holes=False,
        )
        allowed = cv2.dilate(
            support.astype(np.uint8), np.ones((201, 201), np.uint8)
        ) > 0
        accepted = (second_main > 0) & allowed
        added = accepted & ~raw
        point_debug = np.zeros_like(raw, dtype=bool)
        point_debug[seed[1], seed[0]] = True
        debug = {
            "pc1_workpiece2_tail_uncovered_positive": point_debug,
            "pc1_workpiece2_tail_second_raw": second_raw > 0,
            "pc1_workpiece2_tail_accepted": accepted,
        }
        report: dict[str, object] = {
            "attempted": True,
            "mode": "uncovered_positive_local_sam2",
            "uncovered_positives": uncovered_positives,
            "target_box": box,
            "positives": local_positives,
            "negatives": negatives,
            "second_raw_pixels": int(np.count_nonzero(second_raw)),
            "accepted_pixels": int(np.count_nonzero(accepted)),
            "new_pixels": int(np.count_nonzero(added)),
        }
        if np.count_nonzero(added) >= 500:
            report["accepted"] = True
            return (raw | accepted).astype(np.uint8) * 255, report, debug
        report["accepted"] = False
        report["reason"] = "uncovered_positive_local_sam2_added_too_little"
        return raw_mask, report, debug
    near_raw = cv2.dilate(raw.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    missing = support & ~near_raw
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), 8)
    candidates = [
        label for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 800
    ]
    debug: dict[str, np.ndarray] = {"pc1_workpiece2_tail_missing_support": missing}
    if not candidates:
        return raw_mask, {"attempted": False, "reason": "no_large_uncovered_support"}, debug
    # Restrict the second ROI to the first-pass object's major axis.  A large
    # cable above the gripper otherwise looks like a tempting missed lobe.
    first_main, _ = _main_mask_with_filled_holes(raw_mask, auto["positives"])
    first_y, first_x = np.where(first_main > 0)
    if first_x.size < 100:
        return raw_mask, {"attempted": False, "reason": "first_sam2_has_no_positive_component"}, debug
    first_xy = np.column_stack((first_x, first_y)).astype(np.float64)
    first_center = np.mean(first_xy, axis=0)
    centered = first_xy - first_center
    eigenvalues, eigenvectors = np.linalg.eigh(
        centered.T @ centered / max(1, len(first_xy) - 1)
    )
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor_extent = float(np.sqrt(max(1.0, np.min(eigenvalues))))
    major_extent = float(np.sqrt(max(1.0, np.max(eigenvalues))))
    aligned: list[tuple[int, float, float]] = []
    for candidate_label in candidates:
        ys, xs = np.where(labels == candidate_label)
        centroid = np.array([float(np.mean(xs)), float(np.mean(ys))])
        delta = centroid - first_center
        axial = float(abs(np.dot(delta, major_axis)))
        perpendicular = float(
            np.linalg.norm(delta - np.dot(delta, major_axis) * major_axis)
        )
        if axial >= max(35.0, 0.35 * major_extent) and perpendicular <= max(55.0, 1.5 * minor_extent):
            aligned.append((candidate_label, axial, perpendicular))
    if not aligned:
        return raw_mask, {
            "attempted": False, "reason": "no_axially_aligned_uncovered_support",
            "candidate_count": len(candidates),
        }, debug
    label, axial_offset, perpendicular_offset = max(
        aligned, key=lambda item: int(stats[item[0], cv2.CC_STAT_AREA])
    )
    target = labels == label
    distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)
    inset = target & (distance >= float(args.pc1_workpiece2_positive_inset_px))
    if not inset.any():
        inset = _prompt_interior(target)
    positives = _spatially_uniform_positive_points(inset, min(3, int(args.auto_positive_count)))
    if not positives:
        return raw_mask, {"attempted": False, "reason": "no_tail_interior_points"}, debug
    ys, xs = np.where(target)
    margin = int(args.pc1_workpiece2_tail_roi_margin_px)
    box = [
        max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
        min(raw.shape[1] - 1, int(xs.max()) + margin),
        min(raw.shape[0] - 1, int(ys.max()) + margin),
    ]
    negatives = [
        point for point in auto["negatives"]
        if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
    ]
    second_raw = _sam2_prompt(rgb, box, positives, negatives, args.sam2_url)
    second_main, _ = _main_mask_with_filled_holes(second_raw, positives)
    allowed = cv2.dilate(support.astype(np.uint8), np.ones((49, 49), np.uint8)) > 0
    accepted = (second_main > 0) & allowed
    coverage = float(np.count_nonzero(accepted & target) / max(1, np.count_nonzero(target)))
    added = accepted & ~raw
    debug.update({
        "pc1_workpiece2_tail_target": target,
        "pc1_workpiece2_tail_second_raw": second_raw > 0,
        "pc1_workpiece2_tail_accepted": accepted,
    })
    report: dict[str, object] = {
        "attempted": True, "target_pixels": int(np.count_nonzero(target)),
        "target_axial_offset_px": axial_offset,
        "target_perpendicular_offset_px": perpendicular_offset,
        "target_box": box, "positives": positives, "negatives": negatives,
        "second_raw_pixels": int(np.count_nonzero(second_raw)),
        "accepted_pixels": int(np.count_nonzero(accepted)),
        "new_pixels": int(np.count_nonzero(added)),
        "target_coverage": coverage,
    }
    # At least a visible fraction of the intended lobe must actually be
    # explained by the local SAM2 result; otherwise retain the first pass.
    if coverage < 0.12 or np.count_nonzero(added) < 500:
        report["accepted"] = False
        report["reason"] = "insufficient_tail_semantic_coverage"
        return raw_mask, report, debug
    report["accepted"] = True
    return (raw | accepted).astype(np.uint8) * 255, report, debug


def _filter_mask_by_positive_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    scale: float,
    positives: list[list[int]],
    tolerance_m: float,
    minimum_depth_m: float,
    preserve_invalid_depth: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Drop only far-depth mask pixels; retain all valid nearer object surface."""
    measured = depth.astype(np.float32) * float(scale)
    positive_depths = [
        float(measured[y, x])
        for x, y in positives
        if 0 <= x < measured.shape[1]
        and 0 <= y < measured.shape[0]
        and np.isfinite(measured[y, x])
        and measured[y, x] > 0
    ]
    if not positive_depths:
        raise RuntimeError("没有有效正点深度，不能执行 mask 深度一致性过滤")
    reference_m = float(np.median(positive_depths))
    foreground = mask > 0
    valid_depth = np.isfinite(measured) & (measured >= minimum_depth_m)
    # A held workpiece can have a much nearer lobe than its prompt core.  That
    # lobe must be retained; only a depth clearly behind the prompt majority
    # is evidence of a different object/background surface.
    depth_consistent = measured <= reference_m + tolerance_m
    retained_invalid = foreground & ~valid_depth & bool(preserve_invalid_depth)
    filtered = (
        foreground & ((valid_depth & depth_consistent) | retained_invalid)
    ).astype(np.uint8) * 255
    return filtered, {
        "reference_positive_depth_m": reference_m,
        "positive_depth_count": len(positive_depths),
        "tolerance_m": float(tolerance_m),
        "minimum_depth_m": float(minimum_depth_m),
        "input_mask_pixels": int(foreground.sum()),
        "removed_invalid_depth_pixels": int(np.count_nonzero(foreground & ~valid_depth)),
        "preserve_invalid_depth_for_semantic_mask": bool(preserve_invalid_depth),
        "retained_invalid_depth_pixels": int(np.count_nonzero(retained_invalid)),
        "removed_depth_mismatch_pixels": int(
            np.count_nonzero(foreground & valid_depth & ~depth_consistent)
        ),
        "final_mask_pixels": int(np.count_nonzero(filtered)),
    }


def _fill_small_enclosed_holes(
    mask: np.ndarray, maximum_hole_pixels: int | None = None
) -> tuple[np.ndarray, dict[str, int | None]]:
    """Restore background islands fully enclosed by a final mask.

    Border-connected background stays empty, so this cannot regrow a removed
    arm outside the workpiece silhouette.  With ``maximum_hole_pixels=None``
    every internal hole is filled for the final workpiece mask.
    """
    foreground = (mask > 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(foreground, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    exterior = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(exterior, flood_mask, (0, 0), 1)
    enclosed = ((exterior == 0) & (padded == 0)).astype(np.uint8)[1:-1, 1:-1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, 8)
    restored = np.zeros_like(enclosed, dtype=np.uint8)
    for label in range(1, count):
        if maximum_hole_pixels is None or int(stats[label, cv2.CC_STAT_AREA]) <= maximum_hole_pixels:
            restored[labels == label] = 1
    filled = np.maximum(foreground, restored).astype(np.uint8) * 255
    return filled, {
        "maximum_hole_pixels": maximum_hole_pixels,
        "enclosed_hole_count": int(max(count - 1, 0)),
        "restored_hole_pixels": int(restored.sum()),
        "final_mask_pixels": int((filled > 0).sum()),
    }


def _open_and_remove_distant_small_components(
    mask: np.ndarray,
    keep_only_main_component: bool = False,
    opening_kernel_px: int = 13,
    maximum_small_component_pixels: int = 4000,
    maximum_distance_to_main_px: float = 25.0,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Break narrow robot ties, then discard small components away from the part.

    The largest post-opening component is the held workpiece.  This removes
    detached gripper leftovers without using the prompt itself as a blanket
    exemption from URDF filtering.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_kernel_px, opening_kernel_px)
    )
    opened = cv2.morphologyEx(
        (mask > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel
    ) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8), {
            "opening_kernel_px": opening_kernel_px,
            "removed_by_opening_pixels": int(np.count_nonzero(mask)),
            "removed_distant_small_component_pixels": 0,
            "final_mask_pixels": 0,
        }
    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = labels == main_label
    if keep_only_main_component:
        cleaned = main.astype(np.uint8) * 255
        return cleaned, {
            "opening_kernel_px": opening_kernel_px,
            "component_selection": "largest_component_only",
            "removed_by_opening_pixels": int(np.count_nonzero(mask) - opened.sum()),
            "removed_distant_small_component_pixels": int(opened.sum() - main.sum()),
            "final_mask_pixels": int(main.sum()),
        }
    distance = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)
    distant_small = np.zeros_like(opened, dtype=bool)
    for label in range(1, count):
        region = labels == label
        if (
            label != main_label
            and int(stats[label, cv2.CC_STAT_AREA]) < maximum_small_component_pixels
            and float(distance[region].min()) > maximum_distance_to_main_px
        ):
            distant_small |= region
    cleaned = (opened & ~distant_small).astype(np.uint8) * 255
    return cleaned, {
        "opening_kernel_px": opening_kernel_px,
        "removed_by_opening_pixels": int(np.count_nonzero(mask) - opened.sum()),
        "removed_distant_small_component_pixels": int(distant_small.sum()),
        "maximum_small_component_pixels": maximum_small_component_pixels,
        "maximum_distance_to_main_px": maximum_distance_to_main_px,
        "final_mask_pixels": int(np.count_nonzero(cleaned)),
    }


def _filter_mask_by_urdf_silhouette(
    mask: np.ndarray, rendered_depth: np.ndarray, protected_depth_prompt: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove raw projected robot pixels while preserving depth-prompt object cues."""
    silhouette = cv2.morphologyEx(
        np.isfinite(rendered_depth).astype(np.uint8), cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
    ) > 0
    removed = (mask > 0) & silhouette & ~protected_depth_prompt
    return ((mask > 0) & ~removed).astype(np.uint8) * 255, {
        "urdf_silhouette_pixels": int(silhouette.sum()),
        "removed_mask_pixels": int(removed.sum()),
    }


def _exclude_urdf_silhouette_positive_points(
    positives: list[list[int]], rendered_depth: np.ndarray
) -> tuple[list[list[int]], list[list[int]]]:
    """Do not let an exact closed URDF-projection pixel cue SAM2.

    This is deliberately an exact silhouette test, not a spatial-margin rule:
    points immediately beside the robot remain valid object prompts.
    """
    silhouette = np.isfinite(rendered_depth)
    kept, rejected = [], []
    for point in positives:
        x, y = point
        (rejected if silhouette[y, x] else kept).append(point)
    return kept, rejected


def _prompt_components_containing_points(
    prompt_mask: np.ndarray, points: list[list[int]]
) -> np.ndarray:
    """Protect only prompt components that still have a valid object cue."""
    _, labels = cv2.connectedComponents((prompt_mask > 0).astype(np.uint8), 8)
    supported = {
        int(labels[y, x])
        for x, y in points
        if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0] and labels[y, x]
    }
    return np.isin(labels, list(supported)) if supported else np.zeros_like(prompt_mask, bool)


def _snapshot_and_points(
    name: str, args: argparse.Namespace, cfg: dict
) -> tuple[dict, np.ndarray]:
    """Build the live URDF state used by the SAM2 C/S automatic prompt flow."""
    state = capture_robot_state(args)
    if state.get("status") != "ok":
        raise RuntimeError(
            f"cannot read live robot state for {name}: {state.get('error')}"
        )
    opening = float((state.get("left_gripper_state") or {})["position"])
    active = float(args.urdf_active_joint_max_rad) * (1.0 - opening / 100.0)
    joints = [float(value) for value in state["left_arm_joint_position_deg"]]
    if name == "pc2":
        joints = [float(value) for value in cfg[f"{name}_joints_deg"]]
    snapshot = {
        # Apart maintains this calibration directly rather than relying on a
        # snapshot under the original module's output directory.
        "T_body_base_chest_camera": _body_camera_matrix(cfg).tolist(),
        "T_body_base_left_hand_transfer_end": tool_matrix(
            state["T_body_base_left_hand_transfer_end"]
        ),
        "urdf_active_joint_rad": active,
        "urdf_description_root": str(args.urdf_description_root),
        "left_arm_kinematics": str(args.left_arm_kinematics),
    }
    points = load_self_surface_points(
        args.urdf_description_root,
        active,
        200000,
        joints,
        arm_kinematics_path=args.left_arm_kinematics,
    )
    return snapshot, points


def _projected_self(
    depth: np.ndarray,
    color: np.ndarray,
    intrinsics: np.ndarray,
    scale: float,
    view: str,
    args: argparse.Namespace,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshot, points = _snapshot_and_points(view, args, cfg)
    T_camera_tool = np.linalg.inv(
        matrix(snapshot["T_body_base_chest_camera"], "camera pose")
    ) @ matrix(snapshot["T_body_base_left_hand_transfer_end"], "tool pose")
    rendered = projected_gripper_depth(
        points, T_camera_tool, np.asarray(intrinsics, dtype=float), depth.shape
    )
    # Close the URDF's rasterised projection before using it to remove depth.
    # Newly closed pixels inherit the nearest rendered surface depth, so they
    # can participate in the same measured-vs-URDF consistency test instead
    # of being a purely 2-D deletion.
    projected = np.isfinite(rendered)
    closed_projected = cv2.morphologyEx(
        projected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
    ) > 0
    newly_closed = closed_projected & ~projected
    if np.any(newly_closed):
        _, nearest = ndimage.distance_transform_edt(
            ~projected, return_indices=True
        )
        rendered = rendered.copy()
        rendered[newly_closed] = rendered[
            nearest[0][newly_closed], nearest[1][newly_closed]
        ]
    measured = depth.astype(np.float32) * float(scale)
    valid_projected = np.isfinite(rendered) & (measured > 0)
    depth_error = np.abs(measured - rendered)
    depth_matched_pixels = valid_projected & (
        depth_error <= args.urdf_self_depth_tolerance_m
    )
    near_black_relaxed_pixels = np.zeros(depth.shape, dtype=bool)
    if args.urdf_near_black_max_rgb >= 0:
        near_black = np.max(color, axis=2) <= args.urdf_near_black_max_rgb
        # Dark robot materials can report a much less reliable measured depth.
        # For the farther part of the arm, grow the tolerance smoothly from
        # the ordinary black-material allowance to 300 mm.  The expected
        # (URDF-rendered) depth, rather than the noisy measured depth, defines
        # "farther" here.
        ramp_start_m = max(0.0, args.urdf_far_black_depth_start_m - 0.050)
        ramp_fraction = np.clip(
            (rendered - ramp_start_m) / max(1e-6, args.urdf_far_black_depth_start_m - ramp_start_m),
            0.0,
            1.0,
        )
        near_black_tolerance_m = (
            args.urdf_near_black_depth_tolerance_m
            + ramp_fraction * (
                args.urdf_far_black_depth_tolerance_m
                - args.urdf_near_black_depth_tolerance_m
            )
        )
        near_black_relaxed_pixels = (
            valid_projected
            & near_black
            & (depth_error <= near_black_tolerance_m)
            & ~depth_matched_pixels
        )
        depth_matched_pixels |= near_black_relaxed_pixels
    # Do not use a signed URDF-vs-measured depth mismatch as a 2-D exclusion.
    # It can remove held workpiece surfaces when the rendered URDF depth is
    # locally misregistered.
    projected_nearer_mismatch = np.zeros(depth.shape, dtype=bool)
    depth_matched_pixels |= projected_nearer_mismatch
    dilation_kernel_px = int(args.urdf_self_dilation_kernel_px)
    self_pixels = cv2.dilate(
        depth_matched_pixels.astype(np.uint8),
        np.ones((dilation_kernel_px, dilation_kernel_px), np.uint8),
    ) > 0
    return measured, rendered, depth_matched_pixels, near_black_relaxed_pixels, self_pixels


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _spatially_uniform_positive_points(region: np.ndarray, count: int) -> list[list[int]]:
    """Cover a valid-depth region with farthest-point sampling in image space.

    Unlike a distance-transform peak, every valid pixel is eligible.  Thus a
    long or thin workpiece-2 lobe receives a prompt when it is spatially far
    from the already selected prompts, rather than losing to a thicker lobe.
    """
    ys, xs = np.where(region)
    if not xs.size:
        return []
    center_x = (float(xs.min()) + float(xs.max())) * 0.5
    center_y = (float(ys.min()) + float(ys.max())) * 0.5
    first = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    selected = [first]
    nearest_sq = (xs - xs[first]).astype(np.float64) ** 2 + (ys - ys[first]).astype(np.float64) ** 2
    for _ in range(1, min(int(count), int(xs.size))):
        candidate = int(np.argmax(nearest_sq))
        if nearest_sq[candidate] <= 0.0:
            break
        selected.append(candidate)
        distance_sq = (xs - xs[candidate]).astype(np.float64) ** 2 + (ys - ys[candidate]).astype(np.float64) ** 2
        nearest_sq = np.minimum(nearest_sq, distance_sq)
    return [[int(xs[index]), int(ys[index])] for index in selected]


def _prompt_interior(support: np.ndarray) -> np.ndarray:
    """Prefer a safely interior depth region for workpiece-2 positive prompts."""
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    for inset_px in (10.0, 6.0, 3.0):
        interior = distance >= inset_px
        if interior.any():
            return interior
    return support


def _pc2_prompt_interior(roi: np.ndarray) -> np.ndarray:
    """Keep PC2 positive prompts away from the qualified ROI boundary."""
    distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    for inset_px in (10.0, 6.0, 3.0):
        interior = distance >= inset_px
        if interior.any():
            return interior
    return roi


def _expand_depth_candidate(
    support: np.ndarray,
    measured: np.ndarray,
    rgb: np.ndarray,
    self_pixels: np.ndarray,
    near_min_m: float,
    near_max_m: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Grow a workpiece-2 depth seed only across locally coherent RGB-D."""
    grown = support.copy()
    expandable = (
        (measured > 0)
        & (measured >= near_min_m - 0.030)
        & (measured <= near_max_m + 0.030)
        & ~self_pixels
    )
    rgb_i16 = rgb.astype(np.int16)
    height, width = support.shape
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for _ in range(int(args.pc1_workpiece2_expand_px)):
        frontier = expandable & ~grown
        if not frontier.any():
            break
        accepted = np.zeros_like(grown)
        for dy, dx in offsets:
            neighbor = np.roll(grown, shift=(dy, dx), axis=(0, 1))
            neighbor_depth = np.roll(measured, shift=(dy, dx), axis=(0, 1))
            neighbor_rgb = np.roll(rgb_i16, shift=(dy, dx), axis=(0, 1))
            if dy > 0:
                neighbor[:dy] = False
            elif dy < 0:
                neighbor[dy:] = False
            if dx > 0:
                neighbor[:, :dx] = False
            elif dx < 0:
                neighbor[:, dx:] = False
            colour_difference = np.max(np.abs(rgb_i16 - neighbor_rgb), axis=2)
            locally_coherent = (
                neighbor
                & (colour_difference <= args.pc1_workpiece2_expand_color_diff)
                & (np.abs(measured - neighbor_depth) <= args.pc1_workpiece2_expand_depth_diff_m)
            )
            accepted |= frontier & locally_coherent
        if not accepted.any():
            break
        grown |= accepted
    return grown


def _outer_dark_cleanup(
    mask: np.ndarray,
    rgb: np.ndarray,
    depth_candidate: np.ndarray,
    prompt_positives: list[list[int]] | None = None,
    dark_max_rgb: int = 30,
    candidate_margin_px: int = 16,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove only very-dark SAM2 pixels outside the trusted depth footprint.

    Workpiece 2 itself contains black material, so colour alone must never
    delete a dark interior surface.  The dilated pre-SAM2 depth candidate is
    its protection zone; only dark pixels outside that zone are removed.
    """
    size = 2 * int(candidate_margin_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    protected = cv2.dilate(depth_candidate.astype(np.uint8), kernel) > 0
    brightness = np.max(rgb, axis=2)
    foreground = mask > 0
    removed = foreground & (brightness <= int(dark_max_rgb)) & ~protected
    after_dark = foreground & ~removed
    if prompt_positives:
        filled, selection_report = _main_mask_with_filled_holes(
            after_dark.astype(np.uint8) * 255, prompt_positives
        )
        retained_pixels = int(selection_report["retained_component_pixels_before_hole_fill"])
        removed_non_main = int(after_dark.sum() - retained_pixels)
    else:
        main = _largest_component(after_dark)
        filled, fill_report = _fill_small_enclosed_holes(main.astype(np.uint8) * 255)
        retained_pixels = int(main.sum())
        removed_non_main = int(after_dark.sum() - retained_pixels)
    if prompt_positives:
        # _main_mask_with_filled_holes already fills every enclosed hole.
        fill_report = {
            "restored_hole_pixels": int((filled > 0).sum() - retained_pixels),
        }
    return filled, {
        "dark_max_rgb": int(dark_max_rgb),
        "candidate_margin_px": int(candidate_margin_px),
        "removed_outer_dark_pixels": int(removed.sum()),
        "removed_non_main_pixels": removed_non_main,
        "filled_hole_pixels": int(fill_report["restored_hole_pixels"]),
        "final_mask_pixels": int((filled > 0).sum()),
    }


def _pc1_left_gripper_colour_gate(
    mask: np.ndarray,
    rgb: np.ndarray,
    view: str,
    workpiece: str | int,
    enabled: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Suppress the fixed PC1 left-gripper patch unless it looks like metal.

    The PC1 observation pose is calibrated at 1280x720.  The rectangular
    camera/gripper block at its upper-left repeatedly joins the held part in a
    SAM2 component.  Workpiece 1 permits brown metal there; workpiece 2
    permits neutral silver metal.  Everything else is rejected.  This is
    deliberately a small, view-specific safety patch; all other pixels keep
    the existing pipeline.
    """
    foreground = mask > 0
    report: dict[str, object] = {
        "operation": "pc1_left_gripper_colour_gate",
        "enabled": bool(enabled),
        "applied": False,
        "removed_pixels": 0,
    }
    if not enabled or view.lower() != "pc1":
        report["operation"] = "skipped_non_pc1_or_disabled"
        return mask, report
    height, width = foreground.shape
    if (width, height) != (1280, 720):
        report["operation"] = "skipped_unexpected_resolution"
        report["resolution"] = [int(width), int(height)]
        return mask, report

    exclusion = np.zeros_like(foreground, dtype=bool)
    exclusion[70:281, 330:571] = True

    b, g, r = [rgb[:, :, channel].astype(np.int16) for channel in range(3)]
    maximum = np.maximum(np.maximum(b, g), r)
    minimum = np.minimum(np.minimum(b, g), r)
    brown_metal = (
        (r >= g - 2)
        & (r >= b + 5)
        & (maximum >= 30)
        & (maximum <= 150)
    )
    silver_metal = (
        ((maximum - minimum) <= 12)
        & (maximum >= 35)
        & (maximum <= 180)
    )
    material = brown_metal if str(workpiece) == "1" else silver_metal
    removed = foreground & exclusion & ~material
    cleaned = foreground & ~removed
    report.update({
        "applied": True,
        "region_xyxy": [330, 70, 570, 280],
        "workpiece": str(workpiece),
        "allowed_material": "brown" if str(workpiece) == "1" else "silver",
        "selected_pixels_in_region": int(np.count_nonzero(foreground & exclusion)),
        "brown_pixels_kept_in_region": int(
            np.count_nonzero(foreground & exclusion & brown_metal)
        ),
        "silver_pixels_kept_in_region": int(
            np.count_nonzero(foreground & exclusion & silver_metal)
        ),
        "removed_pixels": int(np.count_nonzero(removed)),
        "final_mask_pixels": int(np.count_nonzero(cleaned)),
    })
    return cleaned.astype(np.uint8) * 255, report


def _pc1_left_gripper_colour_gate_with_quality_fallback(
    mask: np.ndarray,
    rgb: np.ndarray,
    view: str,
    workpiece: str | int,
    prompt_positives: list[list[int]],
    enabled: bool = True,
    minimum_retention_ratio: float = 0.85,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the PC1 colour gate only when it preserves the prompted object.

    The fixed exclusion rectangle can overlap a grey/underexposed portion of
    a real workpiece.  Reject its edit if it removes too much mask, deletes a
    positive cue, or splits a positive-supported connected component.
    """
    original = (mask > 0).astype(np.uint8) * 255
    gated, report = _pc1_left_gripper_colour_gate(
        original, rgb, view, workpiece, enabled=enabled
    )
    before_pixels = int(np.count_nonzero(original))
    after_pixels = int(np.count_nonzero(gated))
    report["quality_fallback_enabled"] = True
    report["minimum_retention_ratio"] = float(minimum_retention_ratio)
    report["retention_ratio"] = float(after_pixels / max(1, before_pixels))
    if not report.get("applied"):
        report["quality_fallback_used"] = False
        return gated, report

    before_count, before_labels = cv2.connectedComponents((original > 0).astype(np.uint8), 8)
    after_count, after_labels = cv2.connectedComponents((gated > 0).astype(np.uint8), 8)
    del before_count, after_count
    positive_labels = {
        int(before_labels[y, x])
        for x, y in prompt_positives
        if 0 <= x < original.shape[1] and 0 <= y < original.shape[0]
        and before_labels[y, x] > 0
    }
    positive_points_retained = all(
        0 <= x < original.shape[1]
        and 0 <= y < original.shape[0]
        and gated[y, x] > 0
        for x, y in prompt_positives
    )
    component_reports: list[dict[str, object]] = []
    component_ok = True
    for label in positive_labels:
        component = before_labels == label
        remaining = gated[component] > 0
        surviving_labels = np.unique(after_labels[component][remaining])
        surviving_labels = surviving_labels[surviving_labels > 0]
        retention = float(np.count_nonzero(remaining) / max(1, np.count_nonzero(component)))
        intact = retention >= minimum_retention_ratio and len(surviving_labels) == 1
        component_ok &= intact
        component_reports.append({
            "label": int(label),
            "retention_ratio": retention,
            "surviving_component_count": int(len(surviving_labels)),
            "intact": bool(intact),
        })
    accepted = bool(
        report["retention_ratio"] >= minimum_retention_ratio
        and positive_points_retained
        and component_ok
    )
    report.update({
        "positive_points_retained": bool(positive_points_retained),
        "positive_component_checks": component_reports,
        "quality_fallback_used": not accepted,
        "gate_accepted": accepted,
    })
    if not accepted:
        report["operation"] = "pc1_left_gripper_colour_gate_rejected_quality_fallback"
        report["final_mask_pixels"] = before_pixels
        return original, report
    return gated, report


def _auto_rgbd_prompt(
    measured: np.ndarray,
    self_pixels: np.ndarray,
    rgb: np.ndarray,
    args: argparse.Namespace,
    rendered_depth: np.ndarray | None = None,
    hypothesis: str = "brown",
    extended_depth_override: bool | None = None,
    view: str | None = None,
) -> dict | None:
    """Build one brown- or silver-metal RGB-D prompt hypothesis.

    Keep all visible RGB-D islands for diagnosis, but use only the spatially
    coherent, depth-consistent group to form the actual SAM2 ROI and cues.
    """
    if hypothesis == "brown":
        seed_brightness_min, seed_brightness_max = 30, 130
        point_brightness_min, point_brightness_max = 35, 100
    elif hypothesis == "silver":
        seed_brightness_min, seed_brightness_max = 90, 220
        point_brightness_min, point_brightness_max = 105, 220
    elif hypothesis == "colour_difference":
        # Both workpieces are consistently brighter than the depth-confirmed
        # black gripper, but their chroma is not stable enough for one fixed
        # brown/silver threshold.  Build a per-frame relative LAB gate.
        seed_brightness_min, seed_brightness_max = 0, 255
        point_brightness_min, point_brightness_max = 0, 255
    else:
        raise ValueError(f"unknown automatic prompt hypothesis: {hypothesis}")
    brightness = np.max(rgb, axis=2)
    low = float(args.workpiece2_auto_near_min_m)
    # Workpiece 2 can extend far behind the contact point in an edge-on view;
    # workpiece 1 stays in the near band to avoid admitting table clutter.
    extended_depth = (
        str(args.workpiece) == "2"
        if extended_depth_override is None
        else bool(extended_depth_override)
    )
    high = (
        max(float(args.workpiece2_auto_near_max_m), 0.75)
        if extended_depth
        else float(args.workpiece2_auto_near_max_m)
    )
    depth_band = (measured >= low) & (measured <= high)
    colour_difference = None
    if hypothesis == "colour_difference":
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
        robot_reference = self_pixels & (measured > 0)
        if np.count_nonzero(robot_reference) < 200:
            return None
        robot_lab = np.median(lab[robot_reference], axis=0)
        colour_difference = np.linalg.norm(lab - robot_lab, axis=2)
        # Empirical replay set: the closest successful workpiece/robot median
        # separation is DeltaE~30.  Leave margin for pixel-level variation.
        relative_colour = (
            (lab[:, :, 0] >= robot_lab[0] + 15.0)
            & (colour_difference >= 20.0)
        )
        seed = depth_band & relative_colour
    else:
        relative_colour = None
        seed = (
            depth_band
            & (brightness >= seed_brightness_min)
            & (brightness <= seed_brightness_max)
        )
    if not seed.any():
        return None
    grown = seed.copy()
    missing = ~np.isfinite(measured) | (measured <= 0)
    # Do not erase the URDF silhouette at the semantic-prompt stage.  A held
    # workpiece can genuinely project deep into the gripper and can even have
    # nearly the same measured depth there.  Whole-component URDF evidence
    # below rejects isolated robot islands; the ICP mask later removes only
    # depth-confirmed robot pixels.
    expandable = missing | (
        (measured >= low - 0.030) & (measured <= high + 0.030)
    )
    rgb_i16 = rgb.astype(np.int16)
    for _ in range(10):
        frontier = (cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~grown & expandable
        accepted = np.zeros_like(grown)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
            neighbour_rgb = np.roll(np.roll(rgb_i16, dy, axis=0), dx, axis=1)
            neighbour_depth = np.roll(np.roll(measured, dy, axis=0), dx, axis=1)
            neighbour_missing = np.roll(np.roll(missing, dy, axis=0), dx, axis=1)
            if dy > 0: neighbour[:dy] = False
            elif dy < 0: neighbour[dy:] = False
            if dx > 0: neighbour[:, :dx] = False
            elif dx < 0: neighbour[:, dx:] = False
            colour_ok = np.max(np.abs(rgb_i16 - neighbour_rgb), axis=2) <= 30
            depth_ok = missing | neighbour_missing | (np.abs(measured - neighbour_depth) <= 0.015)
            accepted |= frontier & neighbour & colour_ok & depth_ok
        if not accepted.any(): break
        grown |= accepted
    grown_before_final_colour = grown.copy()
    if hypothesis == "colour_difference":
        grown &= relative_colour
    else:
        grown &= (brightness >= seed_brightness_min) & (brightness <= seed_brightness_max)
    grown_after_final_colour = grown.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown.astype(np.uint8), 8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 200) + 1
    candidate = np.isin(labels, keep)
    if not candidate.any(): return None
    candidate_after_area_filter = candidate.copy()
    projected = (
        np.isfinite(rendered_depth)
        if rendered_depth is not None
        else np.zeros_like(candidate, dtype=bool)
    )
    dark_urdf_front_veto = np.zeros_like(candidate, dtype=bool)
    urdf_front_candidate_veto = np.zeros_like(candidate, dtype=bool)
    if rendered_depth is not None:
        object_in_front_before_black_veto = (
            projected
            & np.isfinite(measured)
            & (measured > 0)
            & (measured <= np.asarray(rendered_depth) - 0.015)
        )
        if bool(args.global_exclude_urdf_front_candidates):
            urdf_front_candidate_veto = (
                candidate & object_in_front_before_black_veto
            )
        elif hypothesis == "brown":
            dark_urdf_front_veto = (
                candidate
                & object_in_front_before_black_veto
                & (brightness <= int(args.global_brown_urdf_front_black_max_rgb))
            )
        candidate &= ~(dark_urdf_front_veto | urdf_front_candidate_veto)
        candidate_after_urdf_pixel_veto = candidate.copy()
        # The veto can split an RGB-D island.  Re-label after it so point
        # selection and component association cannot bridge across black arm.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), 8
        )
        keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 200) + 1
        candidate = np.isin(labels, keep)
        if not candidate.any():
            return None
    else:
        candidate_after_urdf_pixel_veto = candidate.copy()
    # The calibrated fixed negatives describe PC1 only.  Reusing them in PC2
    # can put negative prompts directly on a rotated workpiece/tail.
    negatives = _pc1_fixed_negative_points(rgb) if view == "pc1" else []
    forbidden = np.zeros_like(candidate, dtype=np.uint8)
    for x, y in negatives: cv2.circle(forbidden, (x, y), 42, 1, -1)
    eligible = (
        candidate
        & (measured > 0)
        & (brightness >= point_brightness_min)
        & (brightness <= point_brightness_max)
        & ~(forbidden > 0)
    )
    if view == "pc2":
        eligible &= ~(self_pixels & (brightness <= 90))
    if hypothesis == "colour_difference":
        eligible &= relative_colour
    distance = cv2.distanceTransform(candidate.astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_projected = cv2.distanceTransform(
        (~projected).astype(np.uint8), cv2.DIST_L2, 5
    )
    points: dict[int, list[int]] = {}
    areas = np.bincount(labels.ravel())
    for label in keep:
        point_region = (labels == label) & eligible
        if extended_depth and projected.any():
            contact_region = point_region & (distance_to_projected <= 100.0)
            if contact_region.any():
                point_region = contact_region
        ys, xs = np.where(point_region)
        if xs.size:
            i = int(np.argmax(distance[ys, xs]))
            points[int(label)] = [int(xs[i]), int(ys[i])]
    if not points: return None
    # Use the complete URDF projection only to classify whole RGB-D islands.
    # It must not erase individual final-mask pixels: a genuine workpiece lobe
    # can be visible inside a slightly misregistered gripper silhouette.  A
    # reflective robot remnant, however, lies both mostly inside the complete
    # silhouette and well away from its boundary.
    ids = list(points)
    valid_projected_depth = projected & np.isfinite(measured) & (measured > 0)
    object_in_front = (
        valid_projected_depth
        & (measured <= np.asarray(rendered_depth) - 0.015)
        if rendered_depth is not None
        else np.zeros_like(candidate, dtype=bool)
    )
    projected_interior_distance = cv2.distanceTransform(
        projected.astype(np.uint8), cv2.DIST_L2, 5
    )
    deep_projected = projected_interior_distance >= float(
        args.global_urdf_component_interior_px
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    signatures: dict[int, dict[str, object]] = {}
    for label in ids:
        component = labels == label
        valid = component & (measured > 0)
        values = measured[valid]
        colours = lab[component]
        x, y, width, height = stats[label, :4]
        component_area = int(areas[label])
        overlap_ratio = float(np.count_nonzero(component & projected) / component_area)
        deep_overlap_ratio = float(
            np.count_nonzero(component & deep_projected) / component_area
        )
        object_in_front_ratio = float(
            np.count_nonzero(component & object_in_front) / component_area
        )
        robot_veto = bool(
            overlap_ratio >= float(args.global_urdf_component_overlap_ratio)
            and deep_overlap_ratio
            >= float(args.global_urdf_component_deep_overlap_ratio)
            and object_in_front_ratio < 0.08
        )
        component_projected_distances = distance_to_projected[component]
        contact_distance_px = (
            float(np.min(component_projected_distances))
            if projected.any() and component_projected_distances.size
            else float("inf")
        )
        signatures[label] = {
            "depth_median_m": float(np.median(values)),
            "depth_q25_m": float(np.quantile(values, 0.25)),
            "depth_q75_m": float(np.quantile(values, 0.75)),
            "lab_median": np.median(colours, axis=0),
            "bbox_diagonal_px": float(np.hypot(width, height)),
            "area_px": component_area,
            "urdf_overlap_ratio": overlap_ratio,
            "urdf_deep_overlap_ratio": deep_overlap_ratio,
            "urdf_contact_distance_px": contact_distance_px,
            "object_in_front_of_urdf_ratio": object_in_front_ratio,
            "robot_veto": robot_veto,
        }
    object_ids = [label for label in ids if not signatures[label]["robot_veto"]]
    if not object_ids:
        return None
    contact_ids = [
        label
        for label in object_ids
        if float(signatures[label]["urdf_contact_distance_px"])
        <= float(args.global_anchor_contact_max_px)
    ]
    pc2_exposed_seed = np.zeros_like(candidate, dtype=bool)
    pc2_gripper_center = None
    pc2_side_axis = None
    pc2_exposed_ids: list[int] = []
    if view == "pc2" and str(args.workpiece) == "1" and projected.any():
        guard = cv2.dilate(projected.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
        pc2_exposed_seed = candidate & (~guard | object_in_front)
        distance_to_candidate = cv2.distanceTransform((~candidate).astype(np.uint8), cv2.DIST_L2, 5)
        local_gripper_projection = projected & (distance_to_candidate <= 80.0)
        if np.count_nonzero(local_gripper_projection) < 200:
            local_gripper_projection = projected
        py, px = np.where(local_gripper_projection)
        projected_xy = np.column_stack((px, py)).astype(np.float64)
        pc2_gripper_center = np.mean(projected_xy, axis=0)
        projected_centered = projected_xy - pc2_gripper_center
        projected_covariance = projected_centered.T @ projected_centered / max(1, len(projected_xy) - 1)
        projected_values, projected_vectors = np.linalg.eigh(projected_covariance)
        gripper_axis = projected_vectors[:, int(np.argmax(projected_values))]
        pc2_side_axis = np.array([-gripper_axis[1], gripper_axis[0]])
        pc2_exposed_ids = [label for label in object_ids if np.count_nonzero((labels == label) & pc2_exposed_seed) >= 300]
        anchor_pool = pc2_exposed_ids or contact_ids or object_ids
    else:
        anchor_pool = contact_ids or object_ids
    anchor_label = max(anchor_pool, key=lambda label: int(np.count_nonzero((labels == label) & pc2_exposed_seed)) if label in pc2_exposed_ids else int(areas[label]))
    associated = {anchor_label}
    if view == "pc2" and str(args.workpiece) == "1" and projected.any():
        anchor_for_side = labels == anchor_label
        distance_to_anchor_for_side = cv2.distanceTransform((~anchor_for_side).astype(np.uint8), cv2.DIST_L2, 5)
        local_gripper = projected & (distance_to_anchor_for_side <= 80.0)
        if np.count_nonzero(local_gripper) >= 100:
            gy, gx = np.where(local_gripper)
            pc2_gripper_center = np.array([float(np.median(gx)), float(np.median(gy))])
            pc2_side_axis = np.array([1.0, 0.0])
    pc2_anchor_side = None
    if pc2_side_axis is not None and pc2_gripper_center is not None:
        ay, ax = np.where((labels == anchor_label) & pc2_exposed_seed)
        if ax.size:
            pc2_anchor_side = float(np.dot(np.array([float(np.mean(ax)), float(np.mean(ay))]) - pc2_gripper_center, pc2_side_axis))
    component_decisions: list[dict[str, object]] = []
    # Compare every island directly with the main anchor.  Letting an accepted
    # speck become a new anchor creates a bridge from the workpiece, through a
    # screw highlight, and eventually into a distant arm component.
    anchor = labels == anchor_label
    to_anchor = cv2.distanceTransform((~anchor).astype(np.uint8), cv2.DIST_L2, 5)
    anchor_signature = signatures[anchor_label]
    anchor_scale_px = float(anchor_signature["bbox_diagonal_px"])
    anchor_lab = np.asarray(anchor_signature["lab_median"])
    anchor_lightness = float(anchor_lab[0])
    anchor_chroma = anchor_lab[1:]
    anchor_depth_low_m = float(anchor_signature["depth_q25_m"])
    anchor_depth_high_m = float(anchor_signature["depth_q75_m"])
    anchor_y, anchor_x = np.where(anchor)
    anchor_xy = np.column_stack((anchor_x, anchor_y)).astype(np.float64)
    anchor_centroid = np.mean(anchor_xy, axis=0)
    centered_anchor = anchor_xy - anchor_centroid
    covariance = centered_anchor.T @ centered_anchor / max(1, len(anchor_xy) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    anisotropy = float(
        np.sqrt(max(eigenvalues) / max(1e-6, min(eigenvalues)))
    )
    maximum_extended_gap_px = min(240.0, max(140.0, 0.55 * anchor_scale_px))
    for label in points:
        if label == anchor_label:
            continue
        component = labels == label
        gap_px = float(to_anchor[component].min())
        candidate_signature = signatures[label]
        candidate_depth_m = float(candidate_signature["depth_median_m"])
        depth_interval_delta_m = max(
            anchor_depth_low_m - candidate_depth_m,
            candidate_depth_m - anchor_depth_high_m,
            0.0,
        )
        candidate_chroma = np.asarray(candidate_signature["lab_median"])[1:]
        chroma_delta = float(np.linalg.norm(candidate_chroma - anchor_chroma))
        candidate_y, candidate_x = np.where(component)
        candidate_centroid = np.array(
            [float(np.mean(candidate_x)), float(np.mean(candidate_y))]
        )
        centroid_delta = candidate_centroid - anchor_centroid
        axial_offset_px = float(abs(np.dot(centroid_delta, major_axis)))
        perpendicular_offset_px = float(
            np.linalg.norm(centroid_delta - np.dot(centroid_delta, major_axis) * major_axis)
        )
        directional_corridor = bool(
            anisotropy < 1.35
            or perpendicular_offset_px <= max(45.0, 0.22 * anchor_scale_px)
        )
        robot_veto = bool(candidate_signature["robot_veto"])
        close_fragment = (
            not robot_veto
            and not extended_depth
            and gap_px <= 32.0
            and depth_interval_delta_m
            <= float(args.global_close_fragment_depth_diff_m)
            and chroma_delta <= 12.0
        )
        same_material_lobe = (
            not robot_veto
            and not extended_depth
            and gap_px > 32.0
            and gap_px <= maximum_extended_gap_px
            and depth_interval_delta_m <= 0.035
            and chroma_delta <= 10.0
            and directional_corridor
        )
        accepted = (close_fragment or same_material_lobe) and not (
            view == "pc2"
            and str(args.workpiece) == "1"
            and pc2_side_axis is not None
        )
        component_decisions.append({
            "label": int(label),
            "area_px": int(areas[label]),
            "gap_to_anchor_px": gap_px,
            "depth_to_anchor_iqr_m": depth_interval_delta_m,
            "lab_chroma_delta": chroma_delta,
            "urdf_overlap_ratio": float(candidate_signature["urdf_overlap_ratio"]),
            "urdf_deep_overlap_ratio": float(candidate_signature["urdf_deep_overlap_ratio"]),
            "urdf_contact_distance_px": float(candidate_signature["urdf_contact_distance_px"]),
            "object_in_front_of_urdf_ratio": float(
                candidate_signature["object_in_front_of_urdf_ratio"]
            ),
            "robot_veto": robot_veto,
            "axial_offset_px": axial_offset_px,
            "perpendicular_offset_px": perpendicular_offset_px,
            "anchor_anisotropy": anisotropy,
            "directional_corridor": directional_corridor,
            "maximum_extended_gap_px": maximum_extended_gap_px,
            "maximum_close_depth_delta_m": float(
                args.global_close_fragment_depth_diff_m
            ),
            "accepted": accepted,
            "rule": (
                "close_fragment" if close_fragment
                else "same_material_lobe" if same_material_lobe
                else "rejected"
            ),
        })
        if accepted:
            associated.add(label)
    if view == "pc2":
        remote_tail_candidates = []
        for label in ids:
            if label in associated or bool(signatures[label]["robot_veto"]):
                continue
            component = labels == label
            if view == "pc2" and str(args.workpiece) == "1":
                visible_component = component & pc2_exposed_seed
                if np.count_nonzero(visible_component) < 300:
                    continue
                component = visible_component
            ys, xs = np.where(component)
            centroid = np.array([float(np.mean(xs)), float(np.mean(ys))])
            delta = centroid - anchor_centroid
            axial = float(abs(np.dot(delta, major_axis)))
            perpendicular = float(np.linalg.norm(delta - np.dot(delta, major_axis) * major_axis))
            signature = signatures[label]
            candidate_side = float(np.dot(centroid - pc2_gripper_center, pc2_side_axis)) if pc2_side_axis is not None and pc2_gripper_center is not None else 0.0
            vertical_offset_px = float(abs(centroid[1] - anchor_centroid[1]))
            # The PC2 camera can see the opposite lobe below-left rather than
            # at exactly the same image row.  Permit that rotated layout, but
            # only close to the projected gripper; colour/depth checks below
            # still reject bright background and unrelated table parts.
            same_height_corridor = bool(
                vertical_offset_px <= max(190.0, 1.20 * anchor_scale_px)
                and float(signature["urdf_contact_distance_px"]) <= 80.0
            )
            opposite_gripper_side = bool(pc2_anchor_side is not None and pc2_anchor_side * candidate_side < 0.0 and abs(candidate_side) >= 20.0)
            candidate_lab = np.asarray(signature["lab_median"])
            lightness_delta = abs(float(candidate_lab[0]) - anchor_lightness)
            if (opposite_gripper_side and same_height_corridor and int(signature["area_px"]) >= 400 and abs(float(signature["depth_median_m"]) - float(anchor_signature["depth_median_m"])) <= 0.060 and lightness_delta <= 70.0 and float(np.linalg.norm(candidate_lab[1:] - anchor_chroma)) <= 10.0):
                remote_tail_candidates.append(label)
        if remote_tail_candidates:
            associated.add(max(remote_tail_candidates, key=lambda label: int(areas[label])))
    pc1_excluded_lobes: list[int] = []
    pc1_retained_front_lobes: list[int] = []
    if view == "pc1":
        # Most secondary islands deep inside the PC1 gripper projection are
        # robot highlights.  A brown workpiece lobe can nevertheless be split
        # from its main body by the projected finger while remaining clearly
        # in front of the rendered robot.  Preserve that narrowly evidenced
        # case; otherwise the true lobe becomes a SAM2 negative before the
        # model ever sees it (for example a perforated workpiece-1 head).
        pc1_decisions = {
            int(decision["label"]): decision for decision in component_decisions
        }
        pc1_retained_front_lobes = [
            label for label in associated if label != anchor_label
            and str(args.workpiece) == "1"
            and float(signatures[label]["urdf_deep_overlap_ratio"]) >= 0.35
            and float(signatures[label]["urdf_deep_overlap_ratio"]) <= 0.80
            and float(signatures[label]["object_in_front_of_urdf_ratio"])
            >= 0.55
            and float(pc1_decisions[label]["gap_to_anchor_px"]) <= 20.0
            and float(pc1_decisions[label]["lab_chroma_delta"]) <= 5.0
        ]
        pc1_excluded_lobes = [
            label for label in associated if label != anchor_label
            and float(signatures[label]["urdf_deep_overlap_ratio"]) >= 0.35
            and label not in pc1_retained_front_lobes
        ]
        associated.difference_update(pc1_excluded_lobes)
    prompt_support = np.isin(labels, list(associated))
    if view == "pc2" and str(args.workpiece) == "1" and pc2_exposed_seed.any():
        prompt_support &= pc2_exposed_seed
    # A prompt on every RGB-D speck lets one bad island force SAM2 to retain a
    # robot component.  Keep the anchor, then choose spatially diverse cues so
    # a small opposite-side tail is preferred over redundant nearby fragments.
    positive_labels = [anchor_label]
    if view == "pc1":
        # PC1 coverage points must all remain inside its one clean body anchor.
        # Workpiece 2 is a thin, reflective stamping.  Keeping its points at
        # least 25px inside the trusted candidate avoids putting a cue on a
        # hole, the gripper-contact boundary, or the table transition; all of
        # those can make SAM2 select only a local high-contrast patch.
        # The grown anchor can contain a few depth holes, but using the
        # brightness-qualified ``eligible`` subset here can collapse a broad
        # silver part to one tiny island.  Require a valid point depth only;
        # preserve the full anchor geometry for spatially diverse prompts.
        anchor_eligible = anchor & (measured > 0)
        if str(args.workpiece) == "2":
            anchor_distance = cv2.distanceTransform(
                anchor_eligible.astype(np.uint8), cv2.DIST_L2, 5
            )
            anchor_interior = anchor_eligible & (
                anchor_distance >= float(args.pc1_workpiece2_positive_inset_px)
            )
            if not anchor_interior.any():
                anchor_interior = _prompt_interior(anchor_eligible)
        else:
            anchor_interior = _prompt_interior(anchor_eligible)
        positive_budget = int(args.auto_positive_count)
        retained_front_label = (
            max(pc1_retained_front_lobes, key=lambda label: int(areas[label]))
            if pc1_retained_front_lobes else None
        )
        retained_front_count = (
            min(2, max(1, positive_budget // 3))
            if retained_front_label is not None and positive_budget >= 2 else 0
        )
        positives = _spatially_uniform_positive_points(
            anchor_interior, max(1, positive_budget - retained_front_count)
        )
        if retained_front_label is not None:
            retained_front_region = (
                (labels == retained_front_label) & (measured > 0)
            )
            retained_front_interior = _prompt_interior(retained_front_region)
            positives.extend(_spatially_uniform_positive_points(
                retained_front_interior, retained_front_count
            ))
            positive_labels.append(retained_front_label)
        if not positives:
            positives = [points[anchor_label]]
    else:
        if view == "pc2" and str(args.workpiece) == "1" and pc2_exposed_seed.any():
            # Once the component is accepted from its exposed RGB-D evidence,
            # distribute prompts over the whole component.  Restricting point
            # placement back to ``pc2_exposed_seed`` clusters every cue in the
            # one portion lying outside the URDF guard.
            trusted_anchor = (
                anchor
                & (measured > 0)
                & ~(self_pixels & (brightness <= 90))
            )
            x_grid = np.arange(trusted_anchor.shape[1])[None, :]
            split_x = float(pc2_gripper_center[0]) if pc2_gripper_center is not None else float(anchor_centroid[0])
            left_region = trusted_anchor & (x_grid <= split_x - 12.0)
            right_region = trusted_anchor & (x_grid >= split_x + 12.0)
            left_valid = np.count_nonzero(left_region) >= 300
            right_valid = np.count_nonzero(right_region) >= 300
            positives = []
            if left_valid and right_valid:
                left_count = max(1, int(args.auto_positive_count) - 2)
                positives.extend(_spatially_uniform_positive_points(_prompt_interior(left_region), left_count))
                positives.extend(_spatially_uniform_positive_points(_prompt_interior(right_region), min(2, int(args.auto_positive_count))))
            else:
                # A perforated stamping can have a single thick patch that
                # survives the generic 10px interior preference, concentrating
                # every point in one corner.  A fixed 3px inset remains safely
                # inside the component while covering its full extent.
                anchor_point_distance = cv2.distanceTransform(
                    trusted_anchor.astype(np.uint8), cv2.DIST_L2, 5
                )
                anchor_point_region = trusted_anchor & (
                    anchor_point_distance >= 3.0
                )
                if not anchor_point_region.any():
                    anchor_point_region = trusted_anchor
                positives = _spatially_uniform_positive_points(
                    anchor_point_region, int(args.auto_positive_count)
                )
            if not positives:
                positives = [points[anchor_label]]
            positive_labels = [anchor_label]
            tail_labels = [label for label in associated if label != anchor_label]
            if tail_labels and not (left_valid and right_valid):
                tail_label = max(tail_labels, key=lambda label: int(np.count_nonzero((labels == label) & pc2_exposed_seed)))
                tail_interior = _prompt_interior((labels == tail_label) & pc2_exposed_seed & (measured > 0))
                positives.extend(_spatially_uniform_positive_points(tail_interior, min(2, int(args.auto_positive_count))))
                positive_labels.append(tail_label)

        else:
            remaining = set(associated) - {anchor_label}
            while remaining and len(positive_labels) < int(args.auto_positive_count):
                selected_xy = np.asarray([points[label] for label in positive_labels], dtype=float)
                next_label = max(remaining, key=lambda label: float(np.min(np.sum((selected_xy - np.asarray(points[label], dtype=float)) ** 2, axis=1))))
                positive_labels.append(next_label)
                remaining.remove(next_label)
            positives = [points[label] for label in positive_labels]
    positive_to_urdf_depth_gaps_m: list[float] = []
    if projected.any():
        _, nearest_projected = ndimage.distance_transform_edt(
            ~projected, return_indices=True
        )
        for x, y in positives:
            nearest_y = int(nearest_projected[0, y, x])
            nearest_x = int(nearest_projected[1, y, x])
            measured_depth_m = float(measured[y, x])
            rendered_depth_m = float(rendered_depth[nearest_y, nearest_x])
            if measured_depth_m > 0.0 and np.isfinite(rendered_depth_m):
                positive_to_urdf_depth_gaps_m.append(
                    measured_depth_m - rendered_depth_m
                )
    ys, xs = np.where(prompt_support)
    margin = 40 if view == "pc2" and str(args.workpiece) == "1" and pc2_exposed_seed.any() else int(args.pc1_roi_margin_px)
    box = [
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(candidate.shape[1] - 1, int(xs.max()) + margin),
        min(candidate.shape[0] - 1, int(ys.max()) + margin),
    ]
    if view == "pc2" and str(args.workpiece) == "1":
        # Two valid lobes on opposite gripper sides can make SAM2 bridge
        # through the intervening arm.  Add only depth-confirmed, near-black
        # robot pixels as negatives; unlike the old PC1 fixed points these are
        # frame-specific and cannot land on a bright workpiece/tail.
        inside_box = np.zeros_like(candidate, dtype=bool)
        inside_box[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
        protected_support = cv2.dilate(
            prompt_support.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ).astype(bool)
        confirmed_robot_negative_region = (
            inside_box
            & self_pixels
            & (brightness <= 90)
            & ~protected_support
        )
        negatives.extend(
            _spatially_uniform_positive_points(
                _prompt_interior(confirmed_robot_negative_region), 4
            )
        )
    # Rejected islands inside/near the final ROI are useful SAM2 negatives.
    # This is especially important for reflective robot holes that survived
    # measured-vs-rendered depth matching.  Keep the fixed negatives and add
    # only a small, spatially distinct set so genuine object detail is not
    # overwhelmed by negative cues.
    rejected_labels = [label for label in ids if label not in associated]
    rejected_labels.sort(
        key=lambda label: (
            not bool(signatures[label]["robot_veto"]),
            -int(areas[label]),
        )
    )
    distance_to_prompt_support = cv2.distanceTransform(
        (~prompt_support).astype(np.uint8), cv2.DIST_L2, 5
    )
    extra_negatives: list[list[int]] = []
    for label in rejected_labels:
        if (
            view == "pc2"
            and str(args.workpiece) == "1"
            and not bool(signatures[label]["robot_veto"])
        ):
            continue
        if (
            view == "pc1"
            and label in pc1_excluded_lobes
            and float(np.min(distance_to_prompt_support[labels == label])) <= 35.0
        ):
            # A tiny island split from a retained object lobe by the projected
            # finger is ambiguous: do not promote it to object support, but do
            # not place a negative cue at the physical workpiece junction.
            continue
        point = points[label]
        x, y = point
        if not (box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < 20 ** 2 for px, py in extra_negatives):
            continue
        extra_negatives.append(point)
        if len(extra_negatives) >= int(args.global_rejected_negative_count):
            break
    negatives = negatives + extra_negatives
    negative_depth_guard: dict[str, object] = {
        "operation": "skipped_not_pc1",
        "input_negative_count": len(negatives),
        "kept_negative_count": len(negatives),
        "removed_negative_points": [],
    }
    if view == "pc1" and bool(args.pc1_negative_depth_guard):
        # PC1's legacy fixed negatives are image coordinates.  A changed grasp
        # can put one on a dark hole or shadow of the held workpiece.  Use the
        # already-associated prompt support (rather than individual positive
        # pixels) to reject only nearby negatives whose measured depth belongs
        # to the same robust depth interval as the prospective object.
        positive_neighborhood = np.zeros_like(prompt_support, dtype=np.uint8)
        for x, y in positives:
            cv2.circle(
                positive_neighborhood,
                (int(x), int(y)),
                int(args.pc1_negative_depth_guard_positive_radius_px),
                1,
                -1,
            )
        # Do not pool the complete support: an accepted remote lobe can have
        # a very different depth and make the protection interval uselessly
        # wide.  Only use support local to an actual positive prompt.
        support_neighborhood = prompt_support & (positive_neighborhood > 0)
        support_depths = measured[
            support_neighborhood & np.isfinite(measured) & (measured > 0)
        ]
        negative_depth_guard["operation"] = (
            "remove_nearby_negative_within_prompt_support_depth_interval"
        )
        if len(support_depths) >= 20:
            depth_low_m, depth_high_m = np.quantile(support_depths, [0.05, 0.95])
            margin_m = float(args.pc1_negative_depth_guard_margin_m)
            max_distance_px = float(args.pc1_negative_depth_guard_distance_px)
            kept_negatives: list[list[int]] = []
            removed_negative_points: list[dict[str, float | int | list[int]]] = []
            for x, y in negatives:
                depth_m = float(measured[y, x])
                distance_px = float(distance_to_prompt_support[y, x])
                same_depth = (
                    np.isfinite(depth_m)
                    and depth_m > 0.0
                    and float(depth_low_m) - margin_m <= depth_m
                    <= float(depth_high_m) + margin_m
                )
                if same_depth and distance_px <= max_distance_px:
                    removed_negative_points.append({
                        "point": [int(x), int(y)],
                        "depth_m": depth_m,
                        "distance_to_prompt_support_px": distance_px,
                    })
                else:
                    kept_negatives.append([int(x), int(y)])
            negatives = kept_negatives
            negative_depth_guard.update({
                "support_depth_q05_m": float(depth_low_m),
                "support_depth_q95_m": float(depth_high_m),
                "margin_m": margin_m,
                "max_distance_px": max_distance_px,
                "removed_negative_points": removed_negative_points,
                "kept_negative_count": len(negatives),
            })
        else:
            negative_depth_guard["reason"] = "insufficient_valid_prompt_support_depth"
    plausible_contact_labels = contact_ids or object_ids
    plausible_contact_support = np.isin(labels, plausible_contact_labels)
    quality_support = prompt_support
    if extended_depth and projected.any():
        quality_support = prompt_support & (distance_to_projected <= 120.0)
    return {
        "box": box,
        "positives": positives, "negatives": negatives,
        "view": view,
        "strategy": f"global_dual_{hypothesis}_urdf_depth_order_contact_roi",
        "hypothesis": hypothesis,
        "extended_depth": extended_depth,
        "positive_to_nearest_urdf_depth_gaps_m": positive_to_urdf_depth_gaps_m,
        "negative_depth_guard": negative_depth_guard,
        "component_selection": {
            "anchor_label": int(anchor_label),
            "candidate_component_count": len(ids),
            "associated_component_count": len(associated),
            "positive_component_count": len(positive_labels),
            "associated_labels": sorted(int(label) for label in associated),
            "positive_labels": [int(label) for label in positive_labels],
            "pc1_excluded_deep_urdf_lobes": [int(label) for label in pc1_excluded_lobes],
            "pc1_retained_object_in_front_lobes": [
                int(label) for label in pc1_retained_front_lobes
            ],
            "robot_veto_labels": [
                int(label) for label in ids if signatures[label]["robot_veto"]
            ],
            "extra_negative_labels": [
                int(label) for label in rejected_labels if points[label] in extra_negatives
            ],
            "roi_margin_px": margin,
            "decisions": component_decisions,
        },
        "debug_masks": {
            # Keep every semantic-prompt stage observable.  These are depth
            # candidates only; SAM2 still receives RGB + box + point prompts.
            "depth_band": depth_band,
            "initial_depth_seed": seed,
            "grown_before_final_colour": grown_before_final_colour,
            "grown_after_final_colour": grown_after_final_colour,
            "candidate_after_area_filter": candidate_after_area_filter,
            "candidate_after_urdf_pixel_veto": candidate_after_urdf_pixel_veto,
            "processed_near": candidate,
            "expanded_depth_candidate": candidate,
            "plausible_contact_support": plausible_contact_support,
            "object_in_front_of_urdf": object_in_front,
            "brown_urdf_front_black_veto": dark_urdf_front_veto,
            "urdf_front_candidate_veto": urdf_front_candidate_veto,
            "anchor_component": anchor,
            "prompt_support": prompt_support,
            "quality_support": quality_support,
            "prompt_core": prompt_support,
        },
    }


def _expand_workpiece2_main_mask(
    mask: np.ndarray, measured: np.ndarray, rgb: np.ndarray, self_pixels: np.ndarray, depth_support: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Repair a SAM2 boundary only through adjacent RGB/depth-consistent pixels."""
    grown = mask.astype(bool).copy()
    values = measured[grown & depth_support & (measured > 0)]
    if values.size < 20:
        return grown, {"added_pixels": 0, "trusted_main_depth_pixels": int(values.size)}
    low, high = np.quantile(values, [0.05, 0.95])
    low, high = max(0.0, float(low) - 0.015), float(high) + 0.015
    expandable = (measured > 0) & (measured >= low) & (measured <= high) & ~self_pixels
    rgb_i16, original = rgb.astype(np.int16), grown.copy()
    for _ in range(16):
        frontier = (cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~grown & expandable
        accepted = np.zeros_like(grown)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
            neighbour_rgb = np.roll(np.roll(rgb_i16, dy, axis=0), dx, axis=1)
            if dy > 0: neighbour[:dy] = False
            elif dy < 0: neighbour[dy:] = False
            if dx > 0: neighbour[:, :dx] = False
            elif dx < 0: neighbour[:, dx:] = False
            accepted |= frontier & neighbour & (np.max(np.abs(rgb_i16 - neighbour_rgb), axis=2) <= 30)
        if not accepted.any(): break
        grown |= accepted
    return grown, {"added_pixels": int((grown & ~original).sum()), "trusted_main_depth_pixels": int(values.size), "main_depth_low_m": low, "main_depth_high_m": high, "passes": 16, "colour_difference": 30}


def _postprocess_pc2_workpiece1_candidate(
    raw_mask: np.ndarray,
    auto: dict,
    measured: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    rgb: np.ndarray,
    self_pixels: np.ndarray,
    args: argparse.Namespace,
    min_depth_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the complete PC2/workpiece-1 mask cleanup to one hypothesis.

    Hypotheses are compared only after this identical constrained cleanup.  In
    particular, a useful silver head is allowed to grow through compatible
    RGB-D pixels before its quality is compared with brown/colour-difference.
    """
    mask = raw_mask.copy()
    initial_pixels = int(np.count_nonzero(mask))
    mask, hole_fill = _fill_small_enclosed_holes(mask)
    prompt_support = auto["debug_masks"].get(
        "prompt_support", auto["debug_masks"]["processed_near"]
    )
    mask, expansion = _expand_workpiece2_main_mask(
        mask, measured, rgb, self_pixels, prompt_support,
    )
    before_depth = mask.copy()
    tolerance_m = max(float(args.mask_positive_depth_tolerance_m), 0.060)
    mask, depth_filter = _filter_mask_by_positive_depth(
        mask,
        depth,
        depth_scale,
        auto["positives"],
        tolerance_m,
        float(min_depth_m),
        preserve_invalid_depth=True,
    )
    mask, component_cleanup = _main_mask_with_filled_holes(
        mask, auto["positives"],
    )
    axial_tail = {"accepted_pixels": 0, "reason": "disabled"}
    if bool(args.pc2_axial_tail_recovery):
        mask, axial_tail = _recover_pc2_axial_tail(
            mask,
            raw_mask,
            depth,
            depth_scale,
            auto["positives"],
            self_pixels,
            float(args.pc2_axial_tail_max_behind_m),
        )
    before_depth_pixels = int(np.count_nonzero(before_depth))
    depth_filter["operation"] = (
        "remove_farther_than_positive_median_then_keep_positive_components"
    )
    depth_filter["component_cleanup"] = component_cleanup
    depth_filter["hypothesis"] = str(auto.get("hypothesis", "brown"))
    depth_filter["effective_tolerance_m"] = tolerance_m
    depth_filter["postprocess_retention_ratio"] = float(
        np.count_nonzero(mask) / max(1, before_depth_pixels)
    )
    depth_filter["adaptive_wide_depth_retry"] = None
    depth_filter["largest_component_collapse_recovery"] = None
    depth_filter["axial_tail_recovery"] = axial_tail
    return mask, {
        "mask_postprocess": {
            "operation": "pc2_workpiece1_defer_component_selection_until_after_rgbd_expansion",
            "initial_mask_pixels": initial_pixels,
            "initial_positive_component_retention_ratio": 1.0,
            "initial_component_collapse_recovery": None,
        },
        "mask_urdf_hole_fill": hole_fill,
        "mask_rgb_cleanup": {
            "operation": "pc2_workpiece1_defer_dark_component_cleanup_until_after_rgbd"
        },
        "mask_rgbd_expansion": expansion,
        "mask_depth_filter": depth_filter,
        "mask_urdf_filter": {"operation": "disabled_global_urdf_used_only_before_sam2"},
        "mask_opening_cleanup": {"operation": "disabled_global"},
    }


def _expand_pc1_workpiece2_mask_conservatively(
    mask: np.ndarray,
    measured: np.ndarray,
    rgb: np.ndarray,
    self_pixels: np.ndarray,
    prompt_support: np.ndarray,
    positives: list[list[int]],
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Fill only the local RGB-D continuation of a PC1 workpiece-2 SAM mask.

    Unlike the generic W2 expansion, this cannot start a new island or walk
    through a near-black cable: every added pixel must be locally adjacent to
    the already accepted mask and lie in the RGB-D prompt envelope.
    """
    original = mask > 0
    valid_seed = original & (measured > 0) & np.isfinite(measured)
    values = measured[valid_seed]
    if values.size < 50:
        return mask, {"operation": "skipped_pc1_workpiece2_insufficient_seed_depth"}
    low, high = np.quantile(values, [0.03, 0.97])
    low, high = max(0.05, float(low) - 0.025), float(high) + 0.035
    brightness = np.max(rgb, axis=2)
    # The trusted prompt envelope may include a thin far tail; the raw SAM
    # envelope supplies a small allowance around a boundary SAM missed.
    support_near = cv2.dilate(prompt_support.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    raw_near = cv2.dilate(original.astype(np.uint8), np.ones((101, 101), np.uint8)) > 0
    expandable = (
        support_near
        & raw_near
        & (measured >= low)
        & (measured <= high)
        & (brightness >= 60)
        & ~self_pixels
    )
    grown = original.copy()
    # A PC1 interior prompt is evidence for the object even if the first SAM2
    # pass missed that local lobe.  Admit it as a *local* seed only when its
    # depth agrees with the other positive prompts, then subject its expansion
    # to exactly the same RGB-D continuity checks as the main mask.
    positive_depths = [
        float(measured[y, x]) for x, y in positives
        if 0 <= x < measured.shape[1] and 0 <= y < measured.shape[0]
        and np.isfinite(measured[y, x]) and measured[y, x] > 0
    ]
    positive_reference = float(np.median(positive_depths)) if positive_depths else 0.0
    admitted_positive_seeds: list[list[int]] = []
    for x, y in positives:
        if not (0 <= x < measured.shape[1] and 0 <= y < measured.shape[0]):
            continue
        point_depth = float(measured[y, x])
        if (
            point_depth <= 0.0
            or abs(point_depth - positive_reference) > 0.040
            or not expandable[y, x]
        ):
            continue
        local = np.zeros_like(grown, dtype=np.uint8)
        cv2.circle(local, (x, y), 8, 1, -1)
        local_seed = (
            (local > 0) & expandable
            & (np.abs(measured - point_depth) <= 0.020)
        )
        if local_seed.any():
            grown |= local_seed
            admitted_positive_seeds.append([int(x), int(y)])
    rgb_i16 = rgb.astype(np.int16)
    for _ in range(24):
        frontier = (
            cv2.dilate(grown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ) & ~grown & expandable
        accepted = np.zeros_like(grown)
        for dy, dx in (
            (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ):
            neighbour = np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
            neighbour_rgb = np.roll(np.roll(rgb_i16, dy, axis=0), dx, axis=1)
            neighbour_depth = np.roll(np.roll(measured, dy, axis=0), dx, axis=1)
            if dy > 0:
                neighbour[:dy] = False
            elif dy < 0:
                neighbour[dy:] = False
            if dx > 0:
                neighbour[:, :dx] = False
            elif dx < 0:
                neighbour[dx:] = False
            colour_ok = np.max(np.abs(rgb_i16 - neighbour_rgb), axis=2) <= 30
            depth_ok = np.abs(measured - neighbour_depth) <= 0.020
            accepted |= frontier & neighbour & colour_ok & depth_ok
        if not accepted.any():
            break
        grown |= accepted
    return grown.astype(np.uint8) * 255, {
        "operation": "pc1_workpiece2_local_rgbd_continuation",
        "added_pixels": int(np.count_nonzero(grown & ~original)),
        "seed_depth_low_m": low,
        "seed_depth_high_m": high,
        "minimum_brightness": 60,
        "neighbour_colour_difference": 30,
        "neighbour_depth_difference_m": 0.020,
        "passes": 24,
        "positive_depth_reference_m": positive_reference,
        "positive_seed_depth_agreement_m": 0.040,
        "admitted_uncovered_positive_seeds": admitted_positive_seeds,
    }


def _automatic_hypothesis_quality(
    auto: dict,
    raw_mask: np.ndarray,
    self_pixels: np.ndarray,
) -> dict[str, float | int | bool | str]:
    """Score one SAM2 hypothesis without treating 2-D URDF overlap as fatal."""
    foreground = raw_mask > 0
    foreground_pixels = int(np.count_nonzero(foreground))
    support = auto["debug_masks"].get(
        "quality_support", auto["debug_masks"]["prompt_support"]
    ).astype(bool)
    plausible = auto["debug_masks"].get(
        "plausible_contact_support", support
    ).astype(bool)
    if bool(auto.get("extended_depth", False)):
        plausible = plausible & cv2.dilate(
            support.astype(np.uint8), np.ones((41, 41), np.uint8)
        ).astype(bool)
    protected_contact = cv2.dilate(
        support.astype(np.uint8), np.ones((21, 21), np.uint8)
    ) > 0
    unexplained_robot = foreground & self_pixels & ~protected_contact
    support_pixels = int(np.count_nonzero(support))
    plausible_pixels = int(np.count_nonzero(plausible))
    support_coverage = float(
        np.count_nonzero(foreground & support) / max(1, support_pixels)
    )
    plausible_coverage = float(
        np.count_nonzero(foreground & plausible) / max(1, plausible_pixels)
    )
    unexplained_robot_ratio = float(
        np.count_nonzero(unexplained_robot) / max(1, foreground_pixels)
    )
    height, width = foreground.shape
    image_pixels = max(1, height * width)
    raw_image_fraction = float(foreground_pixels / image_pixels)
    foreground_y, foreground_x = np.where(foreground)
    if foreground_x.size:
        raw_bbox = [
            int(foreground_x.min()), int(foreground_y.min()),
            int(foreground_x.max()), int(foreground_y.max()),
        ]
        raw_bbox_fraction = float(
            (raw_bbox[2] - raw_bbox[0] + 1)
            * (raw_bbox[3] - raw_bbox[1] + 1)
            / image_pixels
        )
        edge_margin = 5
        touched_edges = {
            "left": bool(raw_bbox[0] <= edge_margin),
            "top": bool(raw_bbox[1] <= edge_margin),
            "right": bool(raw_bbox[2] >= width - 1 - edge_margin),
            "bottom": bool(raw_bbox[3] >= height - 1 - edge_margin),
        }
    else:
        raw_bbox = [0, 0, 0, 0]
        raw_bbox_fraction = 0.0
        touched_edges = {name: False for name in ("left", "top", "right", "bottom")}
    touched_edge_count = sum(int(value) for value in touched_edges.values())
    positive_depth_gaps = [
        float(value)
        for value in auto.get("positive_to_nearest_urdf_depth_gaps_m", [])
        if np.isfinite(value)
    ]
    minimum_positive_depth_gap_m = (
        min(positive_depth_gaps) if positive_depth_gaps else None
    )
    hypothesis = str(auto.get("hypothesis", "unknown"))
    if hypothesis == "brown":
        if bool(auto.get("extended_depth", False)):
            minimum_support_coverage = 0.20
            minimum_plausible_coverage = 0.20
            maximum_unexplained_robot_ratio = 0.10
        else:
            minimum_support_coverage = 0.53
            minimum_plausible_coverage = 0.35
            maximum_unexplained_robot_ratio = 0.35
    else:
        # A bright-metal hypothesis is more likely to absorb the white table,
        # so require stronger RGB-D agreement than for dark/underexposed metal.
        minimum_support_coverage = 0.55
        minimum_plausible_coverage = 0.40
        maximum_unexplained_robot_ratio = 0.35
    oversized_border_background = bool(
        hypothesis == "silver"
        and raw_image_fraction >= 0.08
        and raw_bbox_fraction >= 0.35
        and touched_edge_count >= 2
    )
    far_behind_urdf_background = bool(
        hypothesis == "silver"
        and minimum_positive_depth_gap_m is not None
        and minimum_positive_depth_gap_m > 0.120
        and raw_image_fraction >= 0.08
        and raw_bbox_fraction >= 0.30
    )
    # In PC2 the held workpiece is observed around the gripper contact area,
    # away from the image boundary.  A component entering continuously from
    # the top edge is the robot arm itself; without this veto a dark-metal
    # branch can narrowly pass its coverage threshold and beat the genuinely
    # useful (but incomplete) silver workpiece mask.
    pc2_top_border_robot_veto = bool(
        auto.get("view") == "pc2"
        and touched_edges["top"]
        and raw_bbox[1] <= edge_margin
    )
    scene_background_veto = bool(
        oversized_border_background
        or far_behind_urdf_background
        or pc2_top_border_robot_veto
    )
    valid = bool(
        foreground_pixels >= 500
        and support_coverage >= minimum_support_coverage
        and plausible_coverage >= minimum_plausible_coverage
        and unexplained_robot_ratio <= maximum_unexplained_robot_ratio
        and not scene_background_veto
    )
    score = float(
        2.0 * support_coverage
        + 1.5 * plausible_coverage
        - 2.0 * unexplained_robot_ratio
        - (5.0 if scene_background_veto else 0.0)
    )
    return {
        "hypothesis": hypothesis,
        "valid": valid,
        "score": score,
        "raw_mask_pixels": foreground_pixels,
        "support_pixels": support_pixels,
        "plausible_contact_pixels": plausible_pixels,
        "support_coverage": support_coverage,
        "plausible_contact_coverage": plausible_coverage,
        "minimum_support_coverage": minimum_support_coverage,
        "minimum_plausible_contact_coverage": minimum_plausible_coverage,
        "maximum_unexplained_robot_ratio": maximum_unexplained_robot_ratio,
        "unexplained_robot_pixels": int(np.count_nonzero(unexplained_robot)),
        "unexplained_robot_ratio": unexplained_robot_ratio,
        "raw_image_fraction": raw_image_fraction,
        "raw_bbox_xyxy": raw_bbox,
        "raw_bbox_fraction": raw_bbox_fraction,
        "touched_image_edges": touched_edges,
        "touched_image_edge_count": touched_edge_count,
        "positive_to_nearest_urdf_depth_gaps_m": positive_depth_gaps,
        "minimum_positive_to_urdf_depth_gap_m": minimum_positive_depth_gap_m,
        "oversized_border_background": oversized_border_background,
        "far_behind_urdf_background": far_behind_urdf_background,
        "pc2_top_border_robot_veto": pc2_top_border_robot_veto,
        "scene_background_veto": scene_background_veto,
    }


def _automatic_icp_mask(
    semantic_mask: np.ndarray,
    measured: np.ndarray,
    rendered: np.ndarray,
    self_pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Remove only depth-confirmed robot pixels from the automatic ICP mask."""
    projected = np.isfinite(rendered)
    object_in_front = (
        projected
        & (measured > 0)
        & (measured <= rendered - 0.015)
    )
    robot_confirmed = self_pixels & ~object_in_front
    semantic = semantic_mask > 0
    icp_mask = (semantic & ~robot_confirmed).astype(np.uint8) * 255
    return icp_mask, {
        "operation": "semantic_mask_minus_depth_confirmed_robot_for_icp_only",
        "semantic_mask_pixels": int(np.count_nonzero(semantic)),
        "removed_robot_pixels": int(np.count_nonzero(semantic & robot_confirmed)),
        "protected_object_in_front_pixels": int(
            np.count_nonzero(semantic & object_in_front)
        ),
        "icp_mask_pixels": int(np.count_nonzero(icp_mask)),
    }


def _pc1_prompt(
    measured: np.ndarray,
    rendered: np.ndarray,
    self_pixels: np.ndarray,
    args: argparse.Namespace,
    cfg: dict,
) -> dict | None:
    near_min_m = (
        float(cfg["min_depth_m"])
        if args.pc1_near_min_m is None
        else float(args.pc1_near_min_m)
    )
    near = (
        (measured >= near_min_m)
        & (measured <= args.pc1_near_max_m)
        & ~self_pixels
    )
    candidate = near
    if args.pc1_candidate_opening_kernel_px > 1:
        size = int(args.pc1_candidate_opening_kernel_px)
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        candidate = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    # PC1 is intentionally simple: retain the largest 8-connected remaining
    # depth candidate.  RGB is used only by the preceding URDF black-material
    # depth matching; it does not independently delete depth candidates.
    support = _largest_component(candidate)
    if str(getattr(args, "workpiece", "1")) == "2":
        support = _expand_depth_candidate(
            support, measured, args.rgb_for_prompt, self_pixels,
            near_min_m, float(args.pc1_near_max_m), args,
        )
    # Workpiece 2 uses image-space farthest-point coverage over a safely
    # interior part of its valid depth footprint, instead of the wide-core
    # heuristic used for workpiece 1.  It deliberately has no fixed-radius
    # suppression.
    if str(getattr(args, "workpiece", "1")) == "2":
        prompt_interior = _prompt_interior(support)
        positives = _spatially_uniform_positive_points(prompt_interior, args.auto_positive_count)
        # A near-black point is characteristic of the residual robot material
        # in these PC1 workpiece-2 views; do not turn it into a positive SAM2
        # prompt.  Do not refill the quota from elsewhere after cancellation.
        positives = [
            point for point in positives
            if 30 <= int(np.max(args.rgb_for_prompt[point[1], point[0]])) <= 128
        ]
        if not positives:
            return None
        ys, xs = np.where(support)
        return {
            "box": [
                max(0, int(xs.min()) - 140),
                max(0, int(ys.min()) - 140),
                min(support.shape[1] - 1, int(xs.max()) + 140),
                min(support.shape[0] - 1, int(ys.max()) + 140),
            ],
            "positives": positives,
            "negatives": _pc1_fixed_negative_points(args.rgb_for_prompt),
            "strategy": "pc1_workpiece2_interior_valid_depth_farthest_point_coverage",
            "near_depth_range_m": [near_min_m, float(args.pc1_near_max_m)],
            "debug_masks": {
                "processed_near": support,
                "expanded_depth_candidate": support,
                "prompt_core": prompt_interior,
            },
        }

    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    core = wide_depth_core(support, min_half_width_px=10.0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    if count <= 1:
        return None
    # The held part often has a broad plate and a narrow/holed long lobe.  A
    # single "best" core makes every prompt fall onto the broad plate.  Seed
    # every substantial core once, then fill the remaining budget with the
    # usual well-spaced interior points across their union.
    eligible = [
        label for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 50
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA])
        * float(distance[labels == label].max()) ** 2,
        reverse=True,
    )
    selected_cores = eligible[:2]
    region = np.isin(labels, selected_cores)
    positives: list[list[int]] = []
    # The two best separated lobes get an explicit seed before density-based
    # filling.  Keeping the scope to two avoids a small residual arm fragment
    # becoming a third positive target.
    for label in selected_cores:
        work = np.where(labels == label, distance, 0.0)
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[y, x] >= 8.0:
            positives.append([int(x), int(y)])
    for point in positive_points(region, args.auto_positive_count):
        if len(positives) >= args.auto_positive_count:
            break
        if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 >= 42 ** 2 for old in positives):
            positives.append(point)
    if not positives:
        return None
    ys, xs = np.where(support)
    return {
        "box": [
            max(0, int(xs.min()) - int(args.pc1_roi_margin_px)),
            max(0, int(ys.min()) - int(args.pc1_roi_margin_px)),
            min(support.shape[1] - 1, int(xs.max()) + int(args.pc1_roi_margin_px)),
            min(support.shape[0] - 1, int(ys.max()) + int(args.pc1_roi_margin_px)),
        ],
        "positives": positives,
        "negatives": _pc1_fixed_negative_points(args.rgb_for_prompt),
        "strategy": "pc1_depth_largest_component_multicore",
        "near_depth_range_m": [near_min_m, float(args.pc1_near_max_m)],
        "debug_masks": {
            "processed_near": support,
            "prompt_core": region,
        },
    }


def _extend_pc2_box_toward_gripper(
    box: list[int], core: np.ndarray, self_pixels: np.ndarray, margin_px: int = 30
) -> tuple[list[int], dict[str, object]]:
    """Extend exactly one PC2 SAM2-box edge towards the projected gripper.

    Depth-derived ROI remains the trusted source of positive prompts.  The box
    is widened only on the dominant direction from that core to its closest
    projected robot pixel, giving SAM2 visual context for the object portion
    adjacent to the gripper without opening the whole tabletop.
    """
    ys, xs = np.where(core)
    self_ys, self_xs = np.where(self_pixels)
    original = list(box)
    if not xs.size or not self_xs.size:
        return box, {"applied": False, "original_box": original}
    center_x, center_y = float(xs.mean()), float(ys.mean())
    squared_distance = (self_xs - center_x) ** 2 + (self_ys - center_y) ** 2
    closest = int(np.argmin(squared_distance))
    gripper_x, gripper_y = int(self_xs[closest]), int(self_ys[closest])
    delta_x, delta_y = gripper_x - center_x, gripper_y - center_y
    height, width = core.shape
    extended = list(box)
    if abs(delta_x) >= abs(delta_y):
        if delta_x >= 0:
            extended[2] = min(width - 1, max(extended[2] + margin_px, gripper_x + margin_px))
            edge = "right"
        else:
            extended[0] = max(0, min(extended[0] - margin_px, gripper_x - margin_px))
            edge = "left"
    elif delta_y >= 0:
        extended[3] = min(height - 1, max(extended[3] + margin_px, gripper_y + margin_px))
        edge = "bottom"
    else:
        extended[1] = max(0, min(extended[1] - margin_px, gripper_y - margin_px))
        edge = "top"
    return extended, {
        "applied": extended != original,
        "edge": edge,
        "original_box": original,
        "extended_box": extended,
        "closest_gripper_pixel": [gripper_x, gripper_y],
        "margin_px": margin_px,
    }


def _pc2_component_positive_points(
    core: np.ndarray, requested_count: int
) -> list[list[int]]:
    """Spread PC2 prompts across lobes and cap narrow lobes at two points."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    candidates: list[tuple[int, list[list[int]]]] = []
    for label in range(1, count):
        region = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        radius = float(distance.max())
        # Narrow/small lobes need only one or two cues.  Otherwise greedy
        # point placement overweights residual gripper geometry.
        capacity = 1 if area < 1000 or radius < 12.0 else 2 if (
            area < 6000 or radius < 25.0
        ) else 3
        points = positive_points(region, capacity)
        if points:
            candidates.append((area, points))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[int]] = []
    max_capacity = max((len(points) for _, points in candidates), default=0)
    # Assign one point per lobe before assigning each lobe a second point.
    for index in range(max_capacity):
        for _, points in candidates:
            if index < len(points) and len(selected) < requested_count:
                selected.append(points[index])
    return selected


def _pc2_left_right_core_points(core: np.ndarray) -> list[list[int]]:
    """Return one interior point from each horizontal half of the core."""
    ys, xs = np.where(core)
    if not xs.size:
        return []
    midpoint = (float(xs.min()) + float(xs.max())) * 0.5
    columns = np.indices(core.shape)[1]
    points: list[list[int]] = []
    for half in (core & (columns <= midpoint), core & (columns > midpoint)):
        selected = positive_points(half, 1)
        if selected:
            points.append(selected[0])
    return points


def _pc2_prompt(
    measured: np.ndarray,
    rendered: np.ndarray,
    self_pixels: np.ndarray,
    args: argparse.Namespace,
) -> dict | None:
    raw = (
        (measured >= args.pc2_near_min_m)
        & (measured <= args.pc2_near_max_m)
        & ~self_pixels
    )
    # ``raw`` has already removed depth-confirmed URDF pixels.  Do not subtract
    # the whole projected silhouette here: at a grasp contact it overlaps the
    # real workpiece and previously cut the target into thin residual lobes.
    # The component filter below still rejects small/border arm fragments.
    source = usable_components(raw, left_central_min_area=6000)
    core = wide_depth_core(source)
    # Use the established core-component allocation, but reserve cues for
    # the left and right halves before filling the remaining prompt budget.
    positives = _pc2_left_right_core_points(core)
    for point in _pc2_component_positive_points(core, args.auto_positive_count):
        if len(positives) >= args.auto_positive_count:
            break
        if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 >= 42 ** 2 for old in positives):
            positives.append(point)
    if not positives:
        return None
    neighbourhood = cv2.dilate(
        core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 51))
    ) > 0
    roi = source & neighbourhood
    ys, xs = np.where(roi)
    if not xs.size:
        return None
    box = [
        max(0, int(xs.min()) - 30),
        max(0, int(ys.min()) - 30),
        min(roi.shape[1] - 1, int(xs.max()) + 30),
        min(roi.shape[0] - 1, int(ys.max()) + 30),
    ]
    box, gripper_extension = _extend_pc2_box_toward_gripper(
        box, core, self_pixels, args.pc2_gripper_box_margin_px
    )
    fixed = [point for group in load_fixed_negatives().values() for point in group]
    fixed = [
        point for point in fixed
        if not (
            0 <= point[0] < raw.shape[1]
            and 0 <= point[1] < raw.shape[0]
            and (raw[point[1], point[0]] or core[point[1], point[0]])
        )
    ]
    negatives = fixed + far_depth_negative_points(measured, args.pc2_near_max_m)
    return {
        "box": box,
        "positives": positives,
        "negatives": negatives,
        "strategy": "pc2_current_depth_prompt",
        "debug_masks": {
            "processed_near": raw,
            "prompt_core": core,
            "prompt_roi": roi,
        },
        "gripper_box_extension": gripper_extension,
    }


def _manual_sam2_prompt(color: np.ndarray, view: str, endpoint: str) -> dict:
    """Use the same terminal-confirmed manual interaction as inspect_and_place."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("手动 SAM2 提示需要图形桌面；请在有 DISPLAY 的终端运行，或加 --prompt-mode auto")

    height, width = color.shape[:2]
    box: list[int] = []
    positives: list[list[int]] = []
    negatives: list[list[int]] = []
    dragging = False
    start = (0, 0)
    window = f"module2-{view}"
    preview_mask: np.ndarray | None = None

    def on_mouse(event: int, px: int, py: int, _flags: int, _param: object) -> None:
        nonlocal dragging, start, box
        if preview_mask is not None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(box) != 4:
                dragging, start = True, (px, py)
            elif box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                positives.append([int(px), int(py)])
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            box = [min(start[0], px), min(start[1], py), max(start[0], px), max(start[1], py)]
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            box = [min(start[0], px), min(start[1], py), max(start[0], px), max(start[1], py)]
        elif event == cv2.EVENT_RBUTTONDOWN and len(box) == 4:
            negatives.append([int(px), int(py)])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(
        f"[{view}] 在CV窗口拖ROI并标正负点；所有确认在本终端输入。\n"
        "  s+回车=开始分割  u+回车=撤销点  r+回车=全部重标  q+回车=退出",
        flush=True,
    )
    try:
        while True:
            if preview_mask is None:
                canvas = color.copy()
                if len(box) == 4:
                    cv2.rectangle(canvas, box[:2], box[2:], (0, 255, 0), 2)
                for point in positives:
                    cv2.circle(canvas, point, 6, (0, 255, 0), -1)
                for point in negatives:
                    cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            else:
                canvas = color.copy()
                canvas[preview_mask > 0] = (
                    canvas[preview_mask > 0] * 0.45 + np.array([0, 0, 255]) * 0.55
                )
                canvas = canvas.astype(np.uint8)
            cv2.imshow(window, canvas)
            cv2.waitKey(30)
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                continue
            command = sys.stdin.readline().strip().lower()
            if preview_mask is not None:
                if command in {"y", "yes", "1"}:
                    break
                if command in {"r", "redo", "0"}:
                    preview_mask = None
                    box, positives, negatives = [], [], []
                    print(f"[{view}] 已清空，请在CV窗口重新标注；完成后在终端输入 s", flush=True)
                    continue
                if command in {"q", "quit"}:
                    raise RuntimeError(f"{view}: 用户取消手动 SAM2 提示")
                print(f"[{view}] 预览确认请输入 y、r 或 q", flush=True)
                continue
            if command in {"s", "segment", ""}:
                if len(box) != 4 or box[2] - box[0] <= 5 or box[3] - box[1] <= 5:
                    print(f"[{view}] 请先左键拖出有效ROI", flush=True)
                    continue
                if not positives:
                    print(f"[{view}] ROI完成后，请在工件上左键添加至少一个正点", flush=True)
                    continue
                print(f"[{view}] 开始SAM2分割: 正点={len(positives)}, 负点={len(negatives)}", flush=True)
                preview_mask = _sam2_prompt(color, box, positives, negatives, endpoint)
                print(
                    f"[{view}] SAM2 mask pixels={np.count_nonzero(preview_mask)}；"
                    "请在本终端输入 y接受 / r重标 / q退出",
                    flush=True,
                )
            elif command in {"r", "reset"}:
                box, positives, negatives = [], [], []
                print(f"[{view}] 已清空ROI和所有点", flush=True)
            elif command in {"u", "undo"}:
                if negatives:
                    negatives.pop()
                elif positives:
                    positives.pop()
                print(f"[{view}] 当前正点={len(positives)}，负点={len(negatives)}", flush=True)
            elif command in {"q", "quit"}:
                raise RuntimeError(f"{view}: 用户取消手动 SAM2 提示")
            else:
                print(f"[{view}] 未知命令；请输入 s、u、r 或 q", flush=True)
    finally:
        cv2.destroyWindow(window)

    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
    empty = np.zeros((height, width), dtype=bool)
    return {
        "box": box,
        "positives": positives,
        "negatives": negatives,
        "strategy": "manual_terminal_confirmed_roi_positive_negative",
        "accepted_mask": preview_mask,
        "debug_masks": {
            "processed_near": empty,
            "prompt_core": roi_mask,
            "prompt_roi": roi_mask,
        },
    }


def _capture_and_register(
    camera: BrokerFemtoCamera,
    cfg: dict,
    args: argparse.Namespace,
    run_dir: Path,
    view: str,
) -> tuple[np.ndarray, dict, dict]:
    # ``inspect_femto_v1`` owns the existing median-of-five policy.  The
    # broker returns one immutable, already validated aggregate frame, so no
    # module can contend for the physical Femto SDK handle.
    frame = camera.capture(purpose=view)
    depth = frame.depth
    if depth.shape != frame.color.shape[:2]:
        raise RuntimeError(
            f"RGB/depth 未对齐: rgb={frame.color.shape[:2]}, depth={depth.shape}"
        )
    # Every view uses the URDF only before SAM2 to keep depth-confirmed robot
    # pixels out of automatically generated prompts.  The full projected
    # silhouette never edits the accepted SAM2 mask afterwards.
    measured, rendered, depth_matched_pixels, near_black_relaxed_pixels, self_pixels = _projected_self(
        depth, frame.color, frame.depth_intrinsics, frame.depth_scale, view, args, cfg
    )
    # Persist raw RGB-D evidence before any automatic prompt/SAM2 operation.
    # A later candidate or service failure must never erase PC2 offline data.
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / f"{view}_rgb.png"), frame.color)
    np.save(run_dir / f"{view}_depth.npy", depth)
    np.save(run_dir / f"{view}_urdf_rendered_depth_m.npy", rendered)
    cv2.imwrite(str(run_dir / f"{view}_urdf_self_removed.png"), self_pixels.astype(np.uint8) * 255)
    cv2.imwrite(str(run_dir / f"{view}_urdf_projected_mask.png"), np.isfinite(rendered).astype(np.uint8) * 255)

    # 不区分 PC1/PC2。只有调用方明确选择
    # manual，或 auto-then-manual 的自动阶段失败，才进入人工提示。
    using_manual_prompt = args.prompt_mode == "manual"
    dual_hypothesis_report: dict[str, object] = {
        "operation": "skipped_manual_prompt"
    }
    automatic_raw_masks: dict[str, np.ndarray] = {}
    automatic_postprocessed_masks: dict[str, np.ndarray] = {}

    def persist_automatic_prompt_failure() -> None:
        """Keep offline evidence even when no automatic prompt is selectable."""
        run_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(run_dir / f"{view}_rgb.png"), frame.color)
        np.save(run_dir / f"{view}_depth.npy", depth)
        np.save(run_dir / f"{view}_urdf_rendered_depth_m.npy", rendered)
        cv2.imwrite(
            str(run_dir / f"{view}_urdf_self_removed.png"),
            self_pixels.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(run_dir / f"{view}_urdf_projected_mask.png"),
            np.isfinite(rendered).astype(np.uint8) * 255,
        )
        for hypothesis, raw_mask in automatic_raw_masks.items():
            cv2.imwrite(
                str(run_dir / f"{view}_sam2_{hypothesis}_raw_mask.png"), raw_mask
            )
        (run_dir / f"{view}_automatic_prompt_failure.json").write_text(
            json.dumps(
                {
                    "view": view,
                    "reason": "no_selectable_automatic_prompt",
                    "camera_intrinsics": np.asarray(
                        frame.depth_intrinsics, dtype=float
                    ).tolist(),
                    "depth_scale": float(frame.depth_scale),
                    "automatic_prompt_report": dual_hypothesis_report,
                    "saved_raw_sam2_hypotheses": sorted(automatic_raw_masks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if using_manual_prompt:
        auto = _manual_sam2_prompt(frame.color, view, args.sam2_url)
    else:
        # 每个视角都实际运行棕色/银色两套 SAM2。已知工件类型只决定
        # 主假设；主假设质量不合格时允许另一套接管。
        args.rgb_for_prompt = frame.color
        automatic_candidates: dict[str, tuple[dict, np.ndarray, dict]] = {}
        for hypothesis in ("brown", "silver"):
            candidate_auto = _auto_rgbd_prompt(
                measured,
                self_pixels,
                frame.color,
                args,
                rendered,
                hypothesis=hypothesis,
                extended_depth_override=(False if str(args.workpiece) == "2" and hypothesis == "silver" else None),
                view=view,
            )
            if candidate_auto is None:
                continue
            candidate_raw = _sam2_prompt(
                frame.color,
                candidate_auto["box"],
                candidate_auto["positives"],
                candidate_auto["negatives"],
                args.sam2_url,
            )
            quality = _automatic_hypothesis_quality(
                candidate_auto, candidate_raw, self_pixels
            )
            automatic_candidates[hypothesis] = (
                candidate_auto, candidate_raw, quality
            )
            automatic_raw_masks[hypothesis] = candidate_raw
        primary_hypothesis = "brown" if str(args.workpiece) == "1" else "silver"
        fallback_hypothesis = "silver" if primary_hypothesis == "brown" else "brown"
        selected_hypothesis = None
        selection_reason = "no_valid_hypothesis"
        for hypothesis in (primary_hypothesis, fallback_hypothesis):
            item = automatic_candidates.get(hypothesis)
            if item is not None and bool(item[2]["valid"]):
                selected_hypothesis = hypothesis
                selection_reason = (
                    "valid_primary" if hypothesis == primary_hypothesis
                    else "valid_alternate"
                )
                break
        # A nominally silver workpiece can be strongly underexposed.  If the
        # strict silver hypothesis fails (often by selecting the white table),
        # allow a conservative dark-metal fallback only when its coverage and
        # robot-leakage evidence are adequate and it outscores silver.
        if selected_hypothesis is None and primary_hypothesis == "silver":
            brown_item = automatic_candidates.get("brown")
            silver_item = automatic_candidates.get("silver")
            if brown_item is not None:
                brown_quality = brown_item[2]
                silver_score = (
                    float(silver_item[2]["score"])
                    if silver_item is not None else float("-inf")
                )
                if (
                    float(brown_quality["support_coverage"]) >= 0.60
                    and float(brown_quality["plausible_contact_coverage"]) >= 0.20
                    and float(brown_quality["unexplained_robot_ratio"]) <= 0.10
                    and float(brown_quality["score"]) > silver_score
                ):
                    selected_hypothesis = "brown"
                    selection_reason = "underexposed_silver_dark_metal_fallback"
        if selected_hypothesis is None and automatic_candidates:
            best_hypothesis, best_item = max(
                automatic_candidates.items(),
                key=lambda item: float(item[1][2]["score"]),
            )
            if bool(best_item[2]["valid"]):
                selected_hypothesis = best_hypothesis
        near_silver_fallback: dict[str, object] = {"attempted": False}
        if selected_hypothesis is None and primary_hypothesis == "silver":
            near_silver_fallback["attempted"] = True
            near_auto = _auto_rgbd_prompt(
                measured,
                self_pixels,
                frame.color,
                args,
                rendered,
                hypothesis="silver",
                extended_depth_override=False,
                view=view,
            )
            if near_auto is not None:
                near_raw = _sam2_prompt(
                    frame.color,
                    near_auto["box"],
                    near_auto["positives"],
                    near_auto["negatives"],
                    args.sam2_url,
                )
                near_quality = _automatic_hypothesis_quality(
                    near_auto, near_raw, self_pixels
                )
                near_silver_fallback["quality"] = near_quality
                automatic_raw_masks["silver_near_fallback"] = near_raw
                if bool(near_quality["valid"]):
                    automatic_candidates["silver"] = (
                        near_auto, near_raw, near_quality
                    )
                    selected_hypothesis = "silver"
                    selection_reason = "near_depth_silver_fallback_after_extended_failure"
        colour_difference_fallback: dict[str, object] = {"attempted": False}
        compare_pc2_after_postprocess = bool(
            view == "pc2" and str(args.workpiece) == "1"
        )
        if (
            selected_hypothesis is None or compare_pc2_after_postprocess
        ) and bool(args.colour_difference_fallback):
            colour_difference_fallback["attempted"] = True
            contrast_auto = _auto_rgbd_prompt(
                measured,
                self_pixels,
                frame.color,
                args,
                rendered,
                hypothesis="colour_difference",
                view=view,
            )
            if contrast_auto is not None:
                contrast_raw = _sam2_prompt(
                    frame.color,
                    contrast_auto["box"],
                    contrast_auto["positives"],
                    contrast_auto["negatives"],
                    args.sam2_url,
                )
                contrast_quality = _automatic_hypothesis_quality(
                    contrast_auto, contrast_raw, self_pixels
                )
                automatic_candidates["colour_difference"] = (
                    contrast_auto, contrast_raw, contrast_quality
                )
                automatic_raw_masks["colour_difference"] = contrast_raw
                colour_difference_fallback["quality"] = contrast_quality
                if bool(contrast_quality["valid"]):
                    selected_hypothesis = "colour_difference"
                    selection_reason = "relative_lab_colour_difference_fallback"
            else:
                colour_difference_fallback["reason"] = "no_reliable_relative_colour_candidate"

        # PC2/workpiece-1 can be split by the gripper into several lobes.  A
        # raw SAM2 mask (especially silver) may cover only the large head but
        # become the best hypothesis after RGB-D continuation.  Therefore all
        # available hypotheses must receive the same constrained postprocess
        # before branch selection.
        postprocess_hypotheses: dict[str, dict[str, object]] = {}
        if compare_pc2_after_postprocess:
            selected_hypothesis = None
            selection_reason = "no_valid_postprocessed_hypothesis"
            for name, (candidate_auto, candidate_raw, raw_quality) in automatic_candidates.items():
                candidate_mask, candidate_reports = _postprocess_pc2_workpiece1_candidate(
                    candidate_raw,
                    candidate_auto,
                    measured,
                    depth,
                    frame.depth_scale,
                    frame.color,
                    self_pixels,
                    args,
                    float(cfg["min_depth_m"]),
                )
                post_quality = _automatic_hypothesis_quality(
                    candidate_auto, candidate_mask, self_pixels
                )
                candidate_auto["pc2_postprocessed_candidate_mask"] = candidate_mask
                candidate_auto["pc2_postprocessed_candidate_reports"] = candidate_reports
                candidate_auto["pc2_postprocessed_candidate_quality"] = post_quality
                automatic_postprocessed_masks[name] = candidate_mask
                postprocess_hypotheses[name] = {
                    "raw_quality": raw_quality,
                    "postprocess_quality": post_quality,
                    "postprocess": candidate_reports,
                }
            valid_postprocessed = [
                (name, item)
                for name, item in automatic_candidates.items()
                if bool(item[0]["pc2_postprocessed_candidate_quality"]["valid"])
            ]
            if valid_postprocessed:
                selected_hypothesis, _ = max(
                    valid_postprocessed,
                    key=lambda item: float(
                        item[1][0]["pc2_postprocessed_candidate_quality"]["score"]
                    ),
                )
                selection_reason = "best_valid_after_pc2_postprocess"
        if selected_hypothesis is None and automatic_candidates:
            deferred_candidates = [
                (name, item)
                for name, item in automatic_candidates.items()
                if int(item[2]["raw_mask_pixels"]) >= 500
                and not bool(item[2]["scene_background_veto"])
            ]
            if deferred_candidates:
                selected_hypothesis, selected_item = max(
                    deferred_candidates,
                    key=lambda item: float(
                        item[1][0].get(
                            "pc2_postprocessed_candidate_quality", item[1][2]
                        )["score"]
                    ),
                )
                selected_item[0]["defer_quality_until_postprocess"] = True
                selection_reason = (
                    "defer_best_pc2_postprocessed_quality"
                    if compare_pc2_after_postprocess
                    else "defer_automatic_quality_until_postprocess"
                )
        dual_hypothesis_report = {
            "operation": "run_brown_and_silver_then_relative_colour_fallback",
            "workpiece": str(args.workpiece),
            "primary_hypothesis": primary_hypothesis,
            "selected_hypothesis": selected_hypothesis,
            "selection_reason": selection_reason,
            "near_silver_fallback": near_silver_fallback,
            "colour_difference_fallback": colour_difference_fallback,
            "postprocess_hypotheses": postprocess_hypotheses,
            "hypotheses": {
                name: item[2] for name, item in automatic_candidates.items()
            },
        }
        if selected_hypothesis is None:
            auto = None
        else:
            auto, selected_raw, _ = automatic_candidates[selected_hypothesis]
            auto["accepted_mask"] = selected_raw
            auto["dual_hypothesis"] = dual_hypothesis_report
    if auto is None:
        persist_automatic_prompt_failure()
        if args.prompt_mode != "auto-then-manual":
            raise RuntimeError(f"{view}: no reliable automatic processed-depth prompt")
        print(
            f"[{view}] 自动提示失败，切换到手动 ROI/正负点标注。",
            flush=True,
        )
        auto = _manual_sam2_prompt(frame.color, view, args.sam2_url)
        using_manual_prompt = True
    # 人工模式统一使用相同语义：人工确认的 SAM2 原始 mask
    # 是二维最终结果，后续只对生成 ICP 点云施加最近深度带。
    using_global_manual_mask = using_manual_prompt
    box = auto["box"]
    positives = auto["positives"]
    # 正点已经由统一的 depth-confirmed self mask 约束；不再根据 PC 编号
    # 额外使用完整 URDF 投影轮廓删点。
    urdf_silhouette_positive_points_removed: list[list[int]] = []
    negatives = auto["negatives"]
    # SAM2 requires one positive.  If every automatic point is on the URDF,
    # use one only as a first-pass seed; it receives no prompt protection and
    # is removed by the subsequent URDF filter before re-sampling.
    first_pass_sam2_positives = positives or [auto["positives"][0]]
    raw_sam2_mask = auto.get("accepted_mask")
    if raw_sam2_mask is None:
        raw_sam2_mask = _sam2_prompt(
            frame.color, box, first_pass_sam2_positives, negatives, args.sam2_url
        )
    pc1_workpiece2_tail_recovery: dict[str, object] = {
        "attempted": False, "reason": "not_pc1_workpiece2"
    }
    if (
        not using_manual_prompt
        and view == "pc1"
        and str(args.workpiece) == "2"
        and bool(args.pc1_workpiece2_tail_second_sam2)
    ):
        raw_sam2_mask, pc1_workpiece2_tail_recovery, tail_debug = (
            _recover_pc1_workpiece2_tail_with_sam2(
                raw_sam2_mask, auto, frame.color, args,
            )
        )
        auto["debug_masks"].update(tail_debug)
    auto["pc1_workpiece2_tail_second_sam2"] = pc1_workpiece2_tail_recovery
    mask_rgbd_expansion: dict[str, float | int | str] = {"operation": "not_applied"}
    if using_global_manual_mask:
        # 与 inspect_and_place 一致：人工接受的 SAM2 mask 绝不再被二维
        # 规则删改，避免手工标出的工件区域被 URDF 或正点深度筛掉。
        mask = raw_sam2_mask.copy()
        mask_postprocess = {
            "operation": "skipped_manual_raw_sam2_mask_is_final",
            "raw_mask_pixels": int(np.count_nonzero(raw_sam2_mask)),
            "final_mask_pixels": int(np.count_nonzero(mask)),
        }
        mask_depth_filter = {
            "operation": "skipped_manual_raw_sam2_mask_is_final",
            "final_mask_pixels": int(np.count_nonzero(mask)),
        }
        mask_urdf_filter = {
            "operation": "skipped_manual_raw_sam2_mask_is_final",
            "removed_mask_pixels": 0,
        }
        mask_opening_cleanup = {
            "operation": "skipped_manual_raw_sam2_mask_is_final",
            "final_mask_pixels": int(np.count_nonzero(mask)),
        }
        mask_urdf_hole_fill = {
            "operation": "skipped_manual_raw_sam2_mask_is_final",
            "final_mask_pixels": int(np.count_nonzero(mask)),
        }
    else:
        pc1_workpiece2_semantic_mask = view == "pc1" and str(args.workpiece) == "2"
        pc2_workpiece1_expand_before_components = bool(
            view == "pc2" and str(args.workpiece) == "1"
        )
        if pc2_workpiece1_expand_before_components:
            # PC2/W1 can show the long part as separated lobes around the
            # gripper. Keep raw SAM2 lobes until RGB-D continuation and depth
            # filtering validate their colour/depth continuity; component
            # selection remains mandatory, but occurs afterwards.
            mask = raw_sam2_mask.copy()
            mask_postprocess = {
                "operation": "pc2_workpiece1_defer_component_selection_until_after_rgbd_expansion",
                "initial_mask_pixels": int(np.count_nonzero(mask)),
            }
        else:
            mask, mask_postprocess = _main_mask_with_filled_holes(
                raw_sam2_mask, positives, fill_holes=not pc1_workpiece2_semantic_mask,
            )
        initial_component_recovery = None
        initial_retention_ratio = float(
            np.count_nonzero(mask) / max(1, np.count_nonzero(raw_sam2_mask))
        )
        if bool(auto.get("extended_depth", False)) and initial_retention_ratio < 0.20:
            recovered, recovery_report = _largest_mask_components(raw_sam2_mask)
            if np.count_nonzero(recovered) > np.count_nonzero(mask):
                mask = recovered
                initial_component_recovery = recovery_report
        mask_postprocess["initial_positive_component_retention_ratio"] = (
            initial_retention_ratio
        )
        mask_postprocess["initial_component_collapse_recovery"] = (
            initial_component_recovery
        )
        mask_depth_filter = {
            "operation": "disabled_global_no_post_sam2_depth_filter",
            "final_mask_pixels": int(np.count_nonzero(mask)),
        }
        mask_urdf_filter = {
            "operation": "disabled_global_urdf_used_only_before_sam2"
        }
        mask_opening_cleanup = {"operation": "disabled_global"}
        if pc1_workpiece2_semantic_mask:
            mask_urdf_hole_fill = {
                "operation": "skipped_pc1_workpiece2_avoid_large_semantic_hole_fill",
                "final_mask_pixels": int(np.count_nonzero(mask)),
            }
        else:
            mask, mask_urdf_hole_fill = _fill_small_enclosed_holes(mask)
        prompt_support = auto["debug_masks"].get(
            "prompt_support", auto["debug_masks"]["processed_near"]
        )
        extended_semantic_mask = bool(auto.get("extended_depth", False))
        if extended_semantic_mask:
            mask_rgb_cleanup = {
                "operation": "skipped_extended_depth_semantic_components",
                "final_mask_pixels": int(np.count_nonzero(mask)),
            }
        elif not pc2_workpiece1_expand_before_components:
            mask, mask_rgb_cleanup = _outer_dark_cleanup(
                mask, frame.color, prompt_support, positives,
            )
        else:
            mask_rgb_cleanup = {"operation": "pc2_workpiece1_defer_dark_component_cleanup_until_after_rgbd"}
        if not using_manual_prompt:
            if view == "pc1" and str(args.workpiece) == "2":
                mask, mask_rgbd_expansion = _expand_pc1_workpiece2_mask_conservatively(
                    mask, measured, frame.color, self_pixels, prompt_support, positives,
                )
            else:
                mask, mask_rgbd_expansion = _expand_workpiece2_main_mask(
                    mask, measured, frame.color, self_pixels, prompt_support,
                )
            if not extended_semantic_mask and not pc2_workpiece1_expand_before_components:
                mask, _ = _outer_dark_cleanup(
                    mask, frame.color, prompt_support, positives,
                )
            mask_before_depth_filter = mask.copy()
            selected_hypothesis = str(auto.get("hypothesis", "brown"))
            depth_tolerance_m = float(args.mask_positive_depth_tolerance_m)
            if extended_semantic_mask:
                # The prompt may search as far as 0.75m to discover a distant
                # tail, but the accepted semantic mask must not keep arbitrary
                # table/background depth.  Nearer surfaces remain protected;
                # only pixels more than this distance behind the prompt depth
                # are removed.
                depth_tolerance_m = float(
                    args.workpiece2_extended_mask_depth_tolerance_m
                )
            elif (
                str(args.workpiece) == "2"
                or selected_hypothesis == "silver"
                or (view == "pc2" and str(args.workpiece) == "1")
            ):
                # The silver stamping has a much larger visible depth span in
                # side views than the flatter brown part.  Component retention
                # still prevents this wider band from reviving disconnected
                # table objects.
                depth_tolerance_m = max(depth_tolerance_m, 0.060)
            mask, mask_depth_filter = _filter_mask_by_positive_depth(
                mask,
                depth,
                frame.depth_scale,
                positives,
                depth_tolerance_m,
                float(cfg["min_depth_m"]),
                preserve_invalid_depth=True,
            )
            if extended_semantic_mask:
                depth_component_cleanup = {
                    "operation": "skipped_extended_depth_semantic_mask_keeps_components",
                    "final_mask_pixels": int(np.count_nonzero(mask)),
                }
            else:
                mask, depth_component_cleanup = _main_mask_with_filled_holes(
                    mask, positives,
                )
            axial_tail_recovery = {"accepted_pixels": 0, "reason": "not_pc2"}
            if view == "pc2" and bool(args.pc2_axial_tail_recovery):
                mask, axial_tail_recovery = _recover_pc2_axial_tail(
                    mask, raw_sam2_mask, depth, frame.depth_scale, positives,
                    self_pixels, float(args.pc2_axial_tail_max_behind_m),
                )
            pre_depth_pixels = int(np.count_nonzero(mask_before_depth_filter))
            retention_ratio = float(
                np.count_nonzero(mask) / max(1, pre_depth_pixels)
            )
            # A small prompt-depth mismatch must not collapse a steeply tilted
            # part to one bolt/edge.  Retry only on obvious collapse, and still
            # keep positive-supported components after the wider depth band.
            adaptive_retry = None
            if retention_ratio < 0.30 and depth_tolerance_m < 0.060:
                retry_mask, retry_report = _filter_mask_by_positive_depth(
                    mask_before_depth_filter,
                    depth,
                    frame.depth_scale,
                    positives,
                    0.060,
                    float(cfg["min_depth_m"]),
                    preserve_invalid_depth=True,
                )
                retry_mask, retry_cleanup = _main_mask_with_filled_holes(
                    retry_mask, positives,
                )
                retry_ratio = float(
                    np.count_nonzero(retry_mask) / max(1, pre_depth_pixels)
                )
                adaptive_retry = {
                    "triggered": True,
                    "initial_retention_ratio": retention_ratio,
                    "retry_tolerance_m": 0.060,
                    "retry_retention_ratio": retry_ratio,
                    "filter": retry_report,
                    "component_cleanup": retry_cleanup,
                }
                if retry_ratio > retention_ratio:
                    mask = retry_mask
                    mask_depth_filter = retry_report
                    depth_component_cleanup = retry_cleanup
                    depth_tolerance_m = 0.060
                    retention_ratio = retry_ratio
            largest_recovery = None
            if retention_ratio < 0.30:
                largest_source, _ = _filter_mask_by_positive_depth(
                    mask_before_depth_filter,
                    depth,
                    frame.depth_scale,
                    positives,
                    depth_tolerance_m,
                    float(cfg["min_depth_m"]),
                    preserve_invalid_depth=True,
                )
                largest_mask, largest_report = _main_mask_with_filled_holes(
                    largest_source, None,
                )
                largest_pixels = int(np.count_nonzero(largest_mask))
                support_overlap = int(
                    np.count_nonzero((largest_mask > 0) & prompt_support)
                )
                largest_recovery = {
                    "triggered": True,
                    "largest_pixels": largest_pixels,
                    "support_overlap_pixels": support_overlap,
                    "cleanup": largest_report,
                }
                if (
                    largest_pixels > 1.5 * int(np.count_nonzero(mask))
                    and support_overlap >= 500
                ):
                    mask = largest_mask
                    depth_component_cleanup = largest_report
                    retention_ratio = float(
                        largest_pixels / max(1, pre_depth_pixels)
                    )
                    largest_recovery["accepted"] = True
                else:
                    largest_recovery["accepted"] = False
            mask_depth_filter["operation"] = (
                "extended_depth_remove_far_background_keep_components"
                if extended_semantic_mask
                else "remove_farther_than_positive_median_then_keep_positive_components"
            )
            mask_depth_filter["component_cleanup"] = depth_component_cleanup
            mask_depth_filter["hypothesis"] = selected_hypothesis
            mask_depth_filter["effective_tolerance_m"] = depth_tolerance_m
            mask_depth_filter["postprocess_retention_ratio"] = retention_ratio
            mask_depth_filter["adaptive_wide_depth_retry"] = adaptive_retry
            mask_depth_filter["largest_component_collapse_recovery"] = largest_recovery
            mask_depth_filter["axial_tail_recovery"] = axial_tail_recovery
        mask_rgb_cleanup["operation"] = (
            "global_remove_outer_rgb_le30_protect_depth_candidate_then_largest_component_fill_holes"
        )
    pc1_left_gripper_colour_gate: dict[str, object] = {
        "operation": "skipped_manual_mask",
        "applied": False,
        "removed_pixels": 0,
    }
    if not using_global_manual_mask:
        mask, pc1_left_gripper_colour_gate = _pc1_left_gripper_colour_gate_with_quality_fallback(
            mask,
            frame.color,
            view,
            args.workpiece,
            positives,
            enabled=bool(args.pc1_left_gripper_colour_gate),
        )

    deferred_postprocess_quality: dict[str, object] = {"operation": "not_deferred"}
    if (
        not using_global_manual_mask
        and bool(auto.get("defer_quality_until_postprocess", False))
    ):
        final_quality = _automatic_hypothesis_quality(auto, mask, self_pixels)
        deferred_postprocess_quality = {
            "operation": "automatic_quality_after_constrained_postprocess",
            "quality": final_quality,
        }
        if not bool(final_quality["valid"]):
            raise RuntimeError(
                f"{view}: automatic mask did not pass quality after constrained postprocess"
            )

    automatic_mask_validation: dict[str, object] = {
        "operation": "skipped_manual_mask"
    }
    if not using_global_manual_mask:
        selected_mask = mask > 0
        selected_pixels = int(np.count_nonzero(selected_mask))
        projected = np.isfinite(rendered)
        object_in_front = (
            projected
            & (measured > 0)
            & (measured <= rendered - 0.015)
        )
        robot_confirmed = self_pixels & ~object_in_front
        robot_confirmed_pixels = int(
            np.count_nonzero(selected_mask & robot_confirmed)
        )
        object_in_front_pixels = int(
            np.count_nonzero(selected_mask & object_in_front)
        )
        x0, y0, x1, y1 = [int(value) for value in box]
        inside_roi = np.zeros_like(selected_mask, dtype=bool)
        inside_roi[max(0, y0):min(selected_mask.shape[0], y1 + 1),
                   max(0, x0):min(selected_mask.shape[1], x1 + 1)] = True
        outside_roi_pixels = int(np.count_nonzero(selected_mask & ~inside_roi))
        robot_confirmed_ratio = robot_confirmed_pixels / max(1, selected_pixels)
        outside_roi_ratio = outside_roi_pixels / max(1, selected_pixels)
        automatic_mask_validation = {
            "operation": "depth_ordered_robot_leakage_or_roi_escape",
            "mask_pixels": selected_pixels,
            "robot_confirmed_pixels": robot_confirmed_pixels,
            "robot_confirmed_ratio": robot_confirmed_ratio,
            "object_in_front_of_urdf_pixels": object_in_front_pixels,
            "outside_roi_pixels": outside_roi_pixels,
            "outside_roi_ratio": outside_roi_ratio,
            "maximum_robot_confirmed_ratio": float(args.global_mask_max_deep_urdf_ratio),
            "maximum_outside_roi_ratio": float(args.global_mask_max_outside_roi_ratio),
        }
        if robot_confirmed_ratio > float(args.global_mask_max_deep_urdf_ratio):
            raise RuntimeError(
                f"{view}: automatic SAM2 mask contains too many depth-confirmed robot pixels "
                f"({robot_confirmed_ratio:.3f} > {args.global_mask_max_deep_urdf_ratio:.3f})"
            )
        if outside_roi_ratio > float(args.global_mask_max_outside_roi_ratio):
            raise RuntimeError(
                f"{view}: automatic SAM2 mask escaped trusted RGB-D ROI "
                f"({outside_roi_ratio:.3f} > {args.global_mask_max_outside_roi_ratio:.3f})"
            )
    sam2_second_pass: dict[str, object] = {"applied": False, "operation": "disabled"}
    point_cloud_mask = mask
    icp_point_depth_filter: dict[str, object] = {"operation": "not_applied"}
    if using_global_manual_mask:
        depth_m = depth.astype(np.float32) * float(frame.depth_scale)
        mask_depths_m = depth_m[(mask > 0) & (depth_m >= 0.005)]
        if not len(mask_depths_m):
            raise RuntimeError(f"{view}: 手动 SAM2 mask 内没有有效深度，无法生成 ICP 点云")
        nearest_depth_m = float(np.min(mask_depths_m))
        lower_depth_m = max(
            1e-6, nearest_depth_m - float(args.manual_icp_nearest_depth_front_m)
        )
        upper_depth_m = nearest_depth_m + float(args.manual_icp_nearest_depth_back_m)
        point_cloud_mask = (
            (mask > 0)
            & (depth_m >= lower_depth_m)
            & (depth_m <= upper_depth_m)
        ).astype(np.uint8) * 255
        icp_point_depth_filter = {
            "operation": "manual_nearest_mask_depth_band_for_3d_icp_only",
            "nearest_mask_depth_m": nearest_depth_m,
            "lower_depth_m": lower_depth_m,
            "upper_depth_m": upper_depth_m,
            "front_margin_m": float(args.manual_icp_nearest_depth_front_m),
            "back_margin_m": float(args.manual_icp_nearest_depth_back_m),
            "raw_sam2_mask_pixels": int(np.count_nonzero(mask)),
            "icp_depth_cluster_pixels": int(np.count_nonzero(point_cloud_mask)),
        }
        print(
            f"[{view}] 手动 mask 保持原样；三维 ICP 点云按 mask 最近深度 "
            f"{nearest_depth_m:.3f}m 取[{lower_depth_m:.3f}, {upper_depth_m:.3f}]m，"
            f"像素={np.count_nonzero(point_cloud_mask)}",
            flush=True,
        )
    else:
        # 二维 SAM2 语义轮廓允许进入夹爪投影；ICP 只使用没有被实测
        # 深度确认成机器人的像素。若工件实测表面比 URDF 更靠近相机，
        # object_in_front 会保护该像素，不会因为二维投影重叠而删除。
        point_cloud_mask, icp_point_depth_filter = _automatic_icp_mask(
            mask, measured, rendered, self_pixels
        )
    point_min_depth_m = 1e-6 if using_global_manual_mask else float(cfg["min_depth_m"])
    point_max_depth_m = 10.0 if using_global_manual_mask else float(cfg["max_depth_m"])
    point_erode_px = 0 if using_global_manual_mask else int(cfg["erode_px"])
    diagnostics = mask_depth_diagnostics(
        point_cloud_mask,
        depth,
        frame.depth_scale,
        min_depth_m=point_min_depth_m,
        max_depth_m=point_max_depth_m,
        erode_px=point_erode_px,
    )
    points = mask_depth_to_points_cam(
        point_cloud_mask,
        depth,
        frame.depth_intrinsics,
        frame.depth_scale,
        min_depth_m=point_min_depth_m,
        max_depth_m=point_max_depth_m,
        erode_px=point_erode_px,
    )
    if not using_global_manual_mask:
        points = remove_radius_outliers_np(points)
    points = voxel_downsample_np(points, float(cfg["observed_voxel_m"]))
    required_points = max(args.point_threshold, int(cfg["min_registration_points"]))
    roi_fallback: dict[str, object] = {
        "attempted": False,
        "operation": "disabled_global_no_view_specific_fallback",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    overlay = frame.color.copy()
    overlay[mask > 0] = (
        0.45 * overlay[mask > 0] + 0.55 * np.array([0, 0, 255])
    ).astype(np.uint8)
    cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    for x, y in positives:
        cv2.circle(overlay, (x, y), 7, (0, 255, 0), -1)
    for x, y in negatives:
        cv2.drawMarker(overlay, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
    cv2.imwrite(str(run_dir / f"{view}_rgb.png"), frame.color)
    cv2.imwrite(str(run_dir / f"{view}_sam2_raw_mask.png"), raw_sam2_mask)
    for hypothesis, hypothesis_mask in automatic_raw_masks.items():
        cv2.imwrite(
            str(run_dir / f"{view}_sam2_{hypothesis}_raw_mask.png"),
            hypothesis_mask,
        )
    for hypothesis, hypothesis_mask in automatic_postprocessed_masks.items():
        cv2.imwrite(
            str(run_dir / f"{view}_{hypothesis}_postprocessed_mask.png"),
            hypothesis_mask,
        )
        hypothesis_overlay = frame.color.copy()
        hypothesis_foreground = hypothesis_mask > 0
        hypothesis_overlay[hypothesis_foreground] = (
            0.45 * hypothesis_overlay[hypothesis_foreground]
            + 0.55 * np.array([0, 0, 255])
        ).astype(np.uint8)
        cv2.imwrite(
            str(run_dir / f"{view}_{hypothesis}_postprocessed_overlay.png"),
            hypothesis_overlay,
        )
    cv2.imwrite(str(run_dir / f"{view}_mask.png"), mask)
    if not using_manual_prompt:
        cv2.imwrite(str(run_dir / f"{view}_main_mask_rgbd_expanded.png"), mask)
    cv2.imwrite(str(run_dir / f"{view}_valid_depth.png"), diagnostics["valid_mask_u8"])
    cv2.imwrite(str(run_dir / f"{view}_icp_depth.png"), diagnostics["valid_mask_u8"])
    cv2.imwrite(str(run_dir / f"{view}_overlay.png"), overlay)
    cv2.imwrite(
        str(run_dir / f"{view}_depth_prompt.png"),
        auto["debug_masks"]["prompt_core"].astype(np.uint8) * 255,
    )
    for debug_name, debug_mask in auto["debug_masks"].items():
        cv2.imwrite(
            str(run_dir / f"{view}_{debug_name}.png"),
            debug_mask.astype(np.uint8) * 255,
        )
    # These overlays are diagnostic only: ``processed_near`` is the depth-band
    # candidate after the URDF self-pixel subtraction, while ``self_pixels``
    # is the depth-confirmed part of the projected URDF.  Keeping them
    # separate makes a registration/depth mismatch immediately visible.
    processed_overlay = frame.color.copy()
    processed_near = auto["debug_masks"]["processed_near"]
    processed_overlay[processed_near] = (
        0.45 * processed_overlay[processed_near]
        + 0.55 * np.array([0, 165, 255])  # BGR: orange
    ).astype(np.uint8)
    cv2.imwrite(
        str(run_dir / f"{view}_processed_near_overlay.png"), processed_overlay
    )
    self_overlay = frame.color.copy()
    self_overlay[self_pixels] = (
        0.45 * self_overlay[self_pixels]
        + 0.55 * np.array([255, 255, 0])  # BGR: cyan
    ).astype(np.uint8)
    cv2.imwrite(str(run_dir / f"{view}_urdf_self_overlay.png"), self_overlay)
    projected_overlay = frame.color.copy()
    projected_pixels = np.isfinite(rendered)
    projected_overlay[projected_pixels] = (
        0.45 * projected_overlay[projected_pixels]
        + 0.55 * np.array([255, 0, 255])  # BGR: magenta
    ).astype(np.uint8)
    cv2.imwrite(
        str(run_dir / f"{view}_urdf_projected_overlay.png"), projected_overlay
    )
    cv2.imwrite(
        str(run_dir / f"{view}_urdf_self_removed.png"),
        self_pixels.astype(np.uint8) * 255,
    )
    # Persist the geometric inputs, so an offline review can distinguish an
    # uncovered robot pixel from a projected pixel that failed depth matching.
    cv2.imwrite(
        str(run_dir / f"{view}_urdf_projected_mask.png"),
        np.isfinite(rendered).astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(run_dir / f"{view}_urdf_depth_matched.png"),
        depth_matched_pixels.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(run_dir / f"{view}_urdf_near_black_relaxed.png"),
        near_black_relaxed_pixels.astype(np.uint8) * 255,
    )
    np.save(run_dir / f"{view}_urdf_rendered_depth_m.npy", rendered)
    # Colour is recorded as a diagnostic, not used as a deletion rule: the
    # held metal part can itself contain genuinely near-black pixels.
    dark_pixels = np.max(frame.color, axis=2) <= 20
    urdf_neighbourhood = cv2.dilate(
        np.isfinite(rendered).astype(np.uint8), np.ones((5, 5), np.uint8)
    ) > 0
    cv2.imwrite(
        str(run_dir / f"{view}_near_black_pixels.png"),
        dark_pixels.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(run_dir / f"{view}_near_black_in_urdf_neighborhood.png"),
        (dark_pixels & urdf_neighbourhood).astype(np.uint8) * 255,
    )
    np.save(run_dir / f"{view}_depth.npy", depth)
    np.save(run_dir / f"{view}_points_camera.npy", points)

    if len(points) < required_points:
        report = {
            "valid_point_count": int(len(points)),
            "depth_valid_ratio": float(diagnostics["depth_valid_ratio"]),
            "box": box,
            "positive_points": positives,
            "negative_points": negatives,
            "auto_strategy": auto["strategy"],
            "sam2_policy": "global_rgbd",
            "urdf_self_dilation_kernel_px": int(
                args.urdf_self_dilation_kernel_px
            ),
        }
        if "component_selection" in auto:
            report["component_selection"] = auto["component_selection"]
        if "gripper_box_extension" in auto:
            report["gripper_box_extension"] = auto["gripper_box_extension"]
        report["mask_postprocess"] = mask_postprocess
        report["mask_depth_filter"] = mask_depth_filter
        report["prompt_roi_fallback"] = roi_fallback
        report["mask_urdf_filter"] = mask_urdf_filter
        report["mask_opening_cleanup"] = mask_opening_cleanup
        report["mask_urdf_hole_fill"] = mask_urdf_hole_fill
        report["mask_rgbd_expansion"] = mask_rgbd_expansion
        report["pc1_workpiece2_tail_second_sam2"] = pc1_workpiece2_tail_recovery
        report["pc1_left_gripper_colour_gate"] = pc1_left_gripper_colour_gate
        report["automatic_mask_validation"] = automatic_mask_validation
        report["dual_sam2_hypothesis"] = dual_hypothesis_report
        report["sam2_second_pass"] = sam2_second_pass
        report["mask_source"] = (
            "sam2_raw_mask_direct" if using_global_manual_mask else "automatic_postprocessed_mask"
        )
        report["icp_point_depth_filter"] = icp_point_depth_filter
        report["urdf_silhouette_positive_points_removed"] = urdf_silhouette_positive_points_removed
        return points, {}, report
    triangles = load_object_triangles(cfg)
    cad = sample_surface_points(
        triangles, int(cfg["cad_sample_points"]), int(cfg["seed"])
    )
    T_camera_object, metrics = _icp_pose(
        cad, points, cfg, mask, np.asarray(frame.depth_intrinsics, dtype=float)
    )
    report = {
        **metrics,
        "valid_point_count": int(len(points)),
        "depth_valid_ratio": float(diagnostics["depth_valid_ratio"]),
        "camera_intrinsics": np.asarray(frame.depth_intrinsics, dtype=float).tolist(),
        "box": box,
        "positive_points": positives,
        "negative_points": negatives,
        "auto_strategy": auto["strategy"],
        "sam2_policy": "global_rgbd",
        "urdf_self_dilation_kernel_px": int(
            args.urdf_self_dilation_kernel_px
        ),
    }
    if "component_selection" in auto:
        report["component_selection"] = auto["component_selection"]
    if "gripper_box_extension" in auto:
        report["gripper_box_extension"] = auto["gripper_box_extension"]
    report["mask_postprocess"] = mask_postprocess
    report["mask_depth_filter"] = mask_depth_filter
    report["prompt_roi_fallback"] = roi_fallback
    report["mask_urdf_filter"] = mask_urdf_filter
    report["mask_opening_cleanup"] = mask_opening_cleanup
    report["mask_urdf_hole_fill"] = mask_urdf_hole_fill
    report["mask_rgbd_expansion"] = mask_rgbd_expansion
    report["pc1_workpiece2_tail_second_sam2"] = pc1_workpiece2_tail_recovery
    report["pc1_left_gripper_colour_gate"] = pc1_left_gripper_colour_gate
    report["automatic_mask_validation"] = automatic_mask_validation
    report["dual_sam2_hypothesis"] = dual_hypothesis_report
    report["sam2_second_pass"] = sam2_second_pass
    report["mask_source"] = (
        "sam2_raw_mask_direct" if using_global_manual_mask else "automatic_postprocessed_mask"
    )
    report["icp_point_depth_filter"] = icp_point_depth_filter
    report["urdf_silhouette_positive_points_removed"] = urdf_silhouette_positive_points_removed
    return points, np.asarray(T_camera_object, dtype=float), report


def _pose(matrix: np.ndarray) -> ObjectPose:
    return ObjectPose(
        frame_id="body_base",
        position=matrix[:3, 3].tolist(),
        orientation=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
    )


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.asarray(transform[:3, :3], dtype=float) @ np.asarray(points, dtype=float).T
    ).T + np.asarray(transform[:3, 3], dtype=float)


def _plot_pose_estimation(
    path: Path,
    observed_points: np.ndarray,
    cad_points: np.ndarray,
    T_frame_object: np.ndarray,
    *,
    frame_name: str,
    seed: int,
) -> None:
    """Render the registered CAD and the predicted workpiece XYZ axes."""
    rng = np.random.default_rng(seed)

    def sample(points: np.ndarray, limit: int) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) <= limit:
            return points
        return points[rng.choice(len(points), size=limit, replace=False)]

    observed = sample(observed_points, 8000)
    registered_cad = sample(_transform_points(T_frame_object, cad_points), 8000)
    origin = np.asarray(T_frame_object[:3, 3], dtype=float)
    axis_length_m = max(0.02, min(0.06, float(np.ptp(registered_cad, axis=0).max()) * 0.45))

    figure = plt.figure(figsize=(9, 7), dpi=150)
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(
        observed[:, 0], observed[:, 1], observed[:, 2],
        s=0.6, c="#6f7680", alpha=0.38, label="observed cloud",
    )
    axes.scatter(
        registered_cad[:, 0], registered_cad[:, 1], registered_cad[:, 2],
        s=0.7, c="#e07a32", alpha=0.65, label="registered CAD",
    )
    for index, (color, label) in enumerate((("#d62728", "X"), ("#2ca02c", "Y"), ("#1f77b4", "Z"))):
        direction = T_frame_object[:3, index] * axis_length_m
        axes.quiver(*origin, *direction, color=color, linewidth=2.2, arrow_length_ratio=0.14)
        endpoint = origin + direction * 1.08
        axes.text(*endpoint, label, color=color, fontsize=11, weight="bold")
    axes.scatter(*origin, s=28, c="#111111", label="object origin")
    axes.set_title(f"Module 2 ICP pose in {frame_name} (RGB=object XYZ axes)")
    axes.set_xlabel(f"{frame_name} X (m)")
    axes.set_ylabel(f"{frame_name} Y (m)")
    axes.set_zlabel(f"{frame_name} Z (m)")
    axes.legend(loc="upper right", markerscale=5)

    all_points = np.vstack((observed, registered_cad, origin[None, :]))
    center = all_points.mean(axis=0)
    half_extent = max(float(np.ptp(all_points, axis=0).max()) * 0.6, axis_length_m * 1.5)
    axes.set_xlim(center[0] - half_extent, center[0] + half_extent)
    axes.set_ylim(center[1] - half_extent, center[1] + half_extent)
    axes.set_zlim(center[2] - half_extent, center[2] + half_extent)
    axes.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _save_pose_overlay(
    run_dir: Path,
    view: str,
    cad_points: np.ndarray,
    T_camera_object: np.ndarray,
    T_body_object: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> Path:
    """Save the readable RGB overlay used by the original Inspect implementation."""
    image_path = run_dir / f"{view}_rgb.png"
    mask_path = run_dir / f"{view}_mask.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise RuntimeError(f"位姿可视化缺少图像或掩码: {image_path}, {mask_path}")

    mask_panel = image.copy()
    red = np.zeros_like(mask_panel)
    red[:, :, 2] = 255
    selected = mask > 0
    mask_panel[selected] = (
        mask_panel[selected] * 0.55 + red[selected] * 0.45
    ).astype(np.uint8)
    cv2.putText(
        mask_panel, "SAM2 mask", (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2,
    )

    pose_panel = image.copy()
    K = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)

    def project(points_object: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_object = np.asarray(points_object, dtype=np.float64).reshape(-1, 3)
        points_camera = (
            T_camera_object[:3, :3] @ points_object.T
        ).T + T_camera_object[:3, 3]
        valid = points_camera[:, 2] > 1e-4
        pixels_h = (K @ points_camera[valid].T).T
        return np.rint(pixels_h[:, :2] / pixels_h[:, 2:3]).astype(np.int32), valid

    stride = max(1, len(cad_points) // 4000)
    cad_pixels, _ = project(cad_points[::stride])
    height, width = pose_panel.shape[:2]
    inside = (
        (cad_pixels[:, 0] >= 0) & (cad_pixels[:, 0] < width)
        & (cad_pixels[:, 1] >= 0) & (cad_pixels[:, 1] < height)
    )
    for u, v in cad_pixels[inside]:
        cv2.circle(pose_panel, (int(u), int(v)), 1, (0, 255, 255), -1)

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
            cv2.arrowedLine(pose_panel, origin, endpoint_xy, color, 4, tipLength=0.18)
            cv2.putText(
                pose_panel, label, endpoint_xy, cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2,
            )

    camera_rpy_deg = Rotation.from_matrix(T_camera_object[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    body_rpy_deg = Rotation.from_matrix(T_body_object[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    text_lines = [
        "Estimated STL overlay (yellow)",
        f"camera xyz={np.round(T_camera_object[:3, 3], 4).tolist()} m",
        f"camera rpy={np.round(camera_rpy_deg, 1).tolist()} deg",
        f"body_base xyz={np.round(T_body_object[:3, 3], 4).tolist()} m",
        f"body_base rpy={np.round(body_rpy_deg, 1).tolist()} deg",
        "axes: X=red Y=green FRONT(+Z)=magenta -Z=gray",
    ]
    for index, line in enumerate(text_lines):
        y = 35 + index * 30
        cv2.putText(pose_panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(pose_panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    output_path = run_dir / "pose_estimation_visualization.png"
    combined = np.concatenate([mask_panel, pose_panel], axis=1)
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"位姿可视化保存失败: {output_path}")
    return output_path


def _validate_args(args: argparse.Namespace) -> None:
    if not (args.urdf_description_root / "urdf").is_dir():
        raise ValueError(
            f"--urdf-description-root 不包含 urdf 目录: {args.urdf_description_root}"
        )
    if not args.left_arm_kinematics.is_file():
        raise ValueError(f"--left-arm-kinematics 不存在: {args.left_arm_kinematics}")
    if args.urdf_active_joint_max_rad <= 0.0:
        raise ValueError("--urdf-active-joint-max-rad 必须大于 0")
    for name in ("preset_speed", "motion_speed"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须在 (0, 1] 内")
    if args.point_threshold <= 0:
        raise ValueError("--point-threshold 必须大于 0")
    for prefix in ("pc1", "pc2"):
        near_min_value = getattr(args, f"{prefix}_near_min_m")
        if prefix == "pc1" and near_min_value is None:
            continue
        near_min = float(near_min_value)
        near_max = float(getattr(args, f"{prefix}_near_max_m"))
        if not 0.0 < near_min < near_max:
            raise ValueError(f"--{prefix}-near-min-m 必须大于 0 且小于 max")
    if not 0.0 < args.workpiece2_auto_near_min_m < args.workpiece2_auto_near_max_m:
        raise ValueError("--workpiece2-auto-near-min-m 必须大于 0 且小于 --workpiece2-auto-near-max-m")
    if args.auto_positive_count <= 0:
        raise ValueError("--auto-positive-count 必须大于 0")
    if args.pc1_candidate_opening_kernel_px < 0:
        raise ValueError("--pc1-candidate-opening-kernel-px 不能为负")
    if args.pc1_roi_margin_px < 0:
        raise ValueError("--pc1-roi-margin-px 不能为负")
    for name in (
        "global_urdf_component_overlap_ratio",
        "global_urdf_component_deep_overlap_ratio",
        "global_mask_max_deep_urdf_ratio",
        "global_mask_max_outside_roi_ratio",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须在 [0, 1] 内")
    if args.global_urdf_component_interior_px <= 0:
        raise ValueError("--global-urdf-component-interior-px 必须大于 0")
    if args.global_anchor_contact_max_px < 0.0:
        raise ValueError("--global-anchor-contact-max-px 不能为负")
    if args.global_rejected_negative_count < 0:
        raise ValueError("--global-rejected-negative-count 不能为负")
    if args.global_close_fragment_depth_diff_m <= 0.0:
        raise ValueError("--global-close-fragment-depth-diff-m 必须大于 0")
    if args.pc1_urdf_barrier_signed_depth_m <= 0.0:
        raise ValueError("--pc1-urdf-barrier-signed-depth-m 必须大于 0")
    if args.pc1_urdf_barrier_min_area_px <= 0:
        raise ValueError("--pc1-urdf-barrier-min-area-px 必须大于 0")
    if args.pc2_gripper_box_margin_px < 0:
        raise ValueError("--pc2-gripper-box-margin-px 不能为负")
    if not 0.0 < args.urdf_self_depth_tolerance_m <= 0.035:
        raise ValueError("--urdf-self-depth-tolerance-m 必须在 (0, 0.035] 内")
    if (
        args.urdf_self_dilation_kernel_px <= 0
        or args.urdf_self_dilation_kernel_px % 2 == 0
    ):
        raise ValueError("--urdf-self-dilation-kernel-px 必须为正奇数")
    if args.urdf_near_black_max_rgb > 255:
        raise ValueError("--urdf-near-black-max-rgb 必须不大于 255")
    if not 0.0 < args.urdf_near_black_depth_tolerance_m <= 0.300:
        raise ValueError("--urdf-near-black-depth-tolerance-m 必须在 (0, 0.300] 内")
    if not 0.0 < args.urdf_far_black_depth_tolerance_m <= 0.300:
        raise ValueError("--urdf-far-black-depth-tolerance-m 必须在 (0, 0.300] 内")
    if args.urdf_far_black_depth_tolerance_m < args.urdf_near_black_depth_tolerance_m:
        raise ValueError("--urdf-far-black-depth-tolerance-m 不能小于近黑色容差")
    if args.urdf_far_black_depth_start_m <= 0.0:
        raise ValueError("--urdf-far-black-depth-start-m 必须大于 0")
    if args.urdf_projected_nearer_exclusion_m <= 0.0:
        raise ValueError("--urdf-projected-nearer-exclusion-m 必须大于 0")
    if args.mask_positive_depth_tolerance_m <= 0.0:
        raise ValueError("--mask-positive-depth-tolerance-m 必须大于 0")
    if not 0.0 < args.workpiece2_extended_mask_depth_tolerance_m <= 0.300:
        raise ValueError(
            "--workpiece2-extended-mask-depth-tolerance-m 必须在 (0, 0.300] 内"
        )


def main() -> int:
    args = parse_args()
    _validate_args(args)
    print(
        "[inspect/global] 所有 PC 视角和工件统一使用 RGB-D/SAM2 逻辑；"
        f"URDF自遮挡膨胀核={args.urdf_self_dilation_kernel_px}x"
        f"{args.urdf_self_dilation_kernel_px}。",
        flush=True,
    )
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.workpiece == "2":
        if not WORKPIECE_2_STL_PATH.is_file():
            raise FileNotFoundError(f"工件2 STL不存在：{WORKPIECE_2_STL_PATH}")
        cfg.update({
            "stl_path": str(WORKPIECE_2_STL_PATH),
            "R_object_stl": WORKPIECE_2_R_OBJECT_STL,
            "object_origin_stl_m": [0.0, 0.0, 0.0],
        })
        print(
            "[inspect/workpiece2] 使用工件2坐标系 X=-X_stl、Y=-Y_stl、Z=+Z_stl；"
            "按 PC1 -> PC2 回退。",
            flush=True,
        )
    # Fail before any planner/gripper action if the resident Broker no longer
    # matches this module's calibrated capture contract.
    BrokerFemtoCamera(args.camera_broker_socket).assert_compatible(cfg)
    require_sam2_service()
    weight_handoff = None
    if args.weight_handoff_file is not None:
        weight_handoff = json.loads(
            args.weight_handoff_file.read_text(encoding="utf-8")
        )
        weight_check = weight_handoff.get("weight_check") or {}
        if (
            weight_handoff.get("weight_check_enabled") is not True
            or weight_check.get("classification") not in {"SINGLE", "MULTIPLE"}
        ):
            raise RuntimeError(
                "GraspNet handoff 未确认 SINGLE 或 MULTIPLE 工件，禁止进入独立 Inspect"
            )
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PARM_PLANNING_RECORD_DIR", str(run_dir / "planning_trajectories")
    )
    planner = PlannerRuntime(
        LoopConfig(
            planner_host=args.planner_host,
            execute_url=args.execute_url,
            base_frame="body_base",
            left_group="left_arm",
            right_group="right_arm",
            left_tool_frame="left_hand_transfer_end",
            right_tool_frame="right_hand_transfer_end",
            max_velocity=args.preset_speed,
            max_acceleration=args.preset_speed,
            idle_poll_seconds=0.2,
            live=not args.skip_motion,
            poses={},
        )
    )
    try:
        planner.open(required_frames={
            "body_base", "left_hand_transfer_end", "left_eef",
        })
        from parm_sdk import Gripper

        # Module 1 已在同姿态 pregrasp 完成单件/多件重量判断。之后不同
        # 手腕姿态下不再用跨姿态力差估重；只读新夹爪驱动的 gripping 状态。
        def check_left_gripper_holding(checkpoint: str) -> dict[str, object]:
            try:
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    deadline = time.monotonic() + GRIPPER_HOLD_TIMEOUT_S
                    stable_since: float | None = None
                    stable_samples = 0
                    state = None
                    while time.monotonic() < deadline:
                        candidate = gripper.get_state().get(args.gripper_hand)
                        if not isinstance(candidate, dict):
                            stable_since = None
                            stable_samples = 0
                        else:
                            state = candidate
                            motion_state = str(candidate.get("motion_state", "")).lower()
                            if motion_state == "gripping":
                                stable_since = stable_since or time.monotonic()
                                stable_samples += 1
                                if time.monotonic() - stable_since >= GRIPPER_HOLD_STABLE_S:
                                    return {
                                        "checkpoint": checkpoint,
                                        "source": "gripper_state",
                                        "motion_state": motion_state,
                                        "position_percent": candidate.get("position"),
                                        "initialized": candidate.get("initialized"),
                                        "stable_duration_s": GRIPPER_HOLD_STABLE_S,
                                        "stable_samples": stable_samples,
                                        "passed": True,
                                    }
                            else:
                                stable_since = None
                                stable_samples = 0
                        time.sleep(GRIPPER_HOLD_POLL_S)
                if not isinstance(state, dict):
                    raise RuntimeError(f"未收到 hand={args.gripper_hand} 夹爪状态: {state}")
                motion_state = state.get("motion_state")
                return {
                    "checkpoint": checkpoint,
                    "source": "gripper_state",
                    "motion_state": motion_state,
                    "position_percent": state.get("position"),
                    "initialized": state.get("initialized"),
                    "stable_duration_s": 0.0,
                    "stable_samples": stable_samples,
                    "passed": False,
                }
            except Exception as exc:
                return {
                    "checkpoint": checkpoint,
                    "source": "gripper_state",
                    "motion_state": None,
                    "passed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }

        holding_recheck = None
        if not args.skip_motion:
            holding_recheck = check_left_gripper_holding("Module 2开始（graspup）")
            print(
                "[inspect运动前持件复检] "
                f"motion_state={holding_recheck.get('motion_state')} "
                f"passed={holding_recheck['passed']}",
                flush=True,
            )
            if not holding_recheck["passed"]:
                planner.set_next_plan_context("left_graspup", "left_graspup")
                planner.move_left_joints(
                    [float(value) for value in cfg["graspup_joints_deg"]]
                )
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    gripper.move(
                        args.gripper_hand, 100.0, force=100.0,
                        speed=80.0,
                        mode="sync", preempt=False, timeout=60.0,
                    )
                write_stage_result(
                    args.result_file,
                    {
                        "stage": "left_inspect",
                        "outcome": "held_object_dropped",
                        "holding_recheck": holding_recheck,
                        "returned_to_graspup": True,
                        "gripper_final_position_percent": 100.0,
                    },
                )
                print(
                    "[inspect工件掉落恢复] 已回 graspup 并打开左夹爪；"
                    "返回总控从 Module 1 重新开始。",
                    flush=True,
                )
                return 0
        with Gripper(host=args.gripper_host, status_port=7702) as gripper:
            state = gripper.get_state().get(args.gripper_hand)
        if not isinstance(state, dict) or state.get("position") is None:
            raise RuntimeError(f"无法读取左夹爪开度: {state}")
        opening = float(state["position"])

        # 所有工件固定使用同一条观察回退链：PC1 的点云、SAM2 或 ICP 任一
        # 环节不合格时才移到 PC2；夹爪开度不再决定观察位，也不再存在 PC3。
        view = None
        points = np.empty((0, 3), dtype=float)
        T_camera_object = np.eye(4, dtype=float)
        report: dict[str, object] = {}
        fallback_attempts: list[dict[str, object]] = []
        previous_view = "graspup"
        required_points = max(
            int(args.point_threshold), int(cfg["min_registration_points"])
        )
        for candidate_view, joint_key in (
            ("pc1", "pc_joints_deg"),
            ("pc2", "pc2_joints_deg"),
        ):
            print(
                f"[inspect] 尝试 {candidate_view.upper()}"
                + ("（手动 SAM2 标注）" if args.prompt_mode == "manual" else ""),
                flush=True,
            )
            if not args.skip_motion:
                planner.set_next_plan_context(
                    f"left_{previous_view}_observation_pose",
                    f"left_{candidate_view}_observation_pose",
                )
                planner.move_left_joints(
                    [float(value) for value in cfg[joint_key]], planner="rrtstar"
                )
            try:
                with BrokerFemtoCamera(args.camera_broker_socket) as camera:
                    candidate_points, candidate_transform, candidate_report = (
                        _capture_and_register(
                            camera, cfg, args, run_dir, candidate_view
                        )
                    )
            except RuntimeError as exc:
                if "用户取消手动 SAM2 提示" in str(exc):
                    raise
                fallback_attempts.append({
                    "view": candidate_view,
                    "outcome": "registration_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(
                    f"[inspect] {candidate_view.upper()} ICP/分割未通过：{exc}",
                    flush=True,
                )
                previous_view = candidate_view
                continue
            if len(candidate_points) < required_points:
                fallback_attempts.append({
                    "view": candidate_view,
                    "outcome": "insufficient_point_cloud",
                    "valid_point_count": int(len(candidate_points)),
                    "required_point_count": required_points,
                })
                print(
                    f"[inspect] {candidate_view.upper()} 点云不足："
                    f"{len(candidate_points)} < {required_points}",
                    flush=True,
                )
                previous_view = candidate_view
                continue
            view = candidate_view
            points = candidate_points
            T_camera_object = candidate_transform
            report = candidate_report
            fallback_attempts.append({
                "view": candidate_view,
                "outcome": "accepted",
                "valid_point_count": int(len(candidate_points)),
            })
            break
        fallback_path = run_dir / "pc1_pc2_fallback.json"
        fallback_path.write_text(
            json.dumps(fallback_attempts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if view is None:
            # PC1/PC2 都已完成采集但没有可信位姿时，这是一次可恢复的
            # 视觉失败，不应让左臂带着工件停在 PC2。安全回到 graspup、
            # 松开夹爪，并把结构化结果交给总控从 Module 1 重抓。
            final_position = None
            if not args.skip_motion:
                planner.set_next_plan_context(
                    f"left_{previous_view}_observation_pose", "left_graspup"
                )
                planner.move_left_joints(
                    [float(value) for value in cfg["graspup_joints_deg"]],
                    planner="rrtstar",
                )
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    response = gripper.move(
                        args.gripper_hand, 100.0, force=100.0,
                        speed=80.0,
                        mode="sync", preempt=False, timeout=60.0,
                    )
                    final_state = gripper.get_state().get(args.gripper_hand)
                if not isinstance(final_state, dict) or final_state.get("position") is None:
                    raise RuntimeError(
                        "PC1/PC2 ICP失败恢复时无法确认左夹爪最终开度: "
                        f"response={response}, state={final_state}"
                    )
                final_position = float(final_state["position"])
            result = {
                "stage": "left_inspect",
                "outcome": "pc1_pc2_registration_failed_recovered",
                "observation_pose": previous_view.upper(),
                "gripper_opening_percent": opening,
                "fallback_attempts": fallback_attempts,
                "fallback_log": str(fallback_path),
                "returned_to_graspup": not args.skip_motion,
                "gripper_final_position_percent": final_position,
            }
            write_stage_result(args.result_file, result)
            print(
                "[inspect恢复] PC1、PC2均未获得可信位姿；"
                "已回 graspup 并打开左夹爪，返回总控重新抓取。"
                f"详情: {fallback_path}",
                flush=True,
            )
            return 0

        if len(points) < max(args.point_threshold, int(cfg["min_registration_points"])):
            if not args.skip_motion:
                planner.set_next_plan_context(
                    f"left_{view}_observation_pose", "left_graspup"
                )
                planner.move_left_joints([float(v) for v in cfg["graspup_joints_deg"]])
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    gripper.move(
                        args.gripper_hand, 100.0, force=100.0,
                        speed=80.0,
                        mode="sync", preempt=False, timeout=60.0,
                    )
            write_stage_result(
                args.result_file,
                {
                    "stage": "left_inspect",
                    "outcome": "insufficient_point_cloud_discarded",
                    "observation_pose": view.upper(),
                    "gripper_opening_percent": opening,
                    "point_report": report,
                    "returned_to_graspup": not args.skip_motion,
                },
            )
            return 0

        T_body_camera = _body_camera_matrix(cfg)
        T_body_object = T_body_camera @ T_camera_object
        tool_response = planner.lookup_transform(
            "body_base", "left_hand_transfer_end"
        )
        if tool_response is None:
            raise RuntimeError("规划器没有返回 left_hand_transfer_end TF")
        T_body_tool = _planner_matrix(tool_response)
        T_tool_object = np.linalg.inv(T_body_tool) @ T_body_object
        eef_response = planner.lookup_transform("body_base", "left_eef")
        if eef_response is None:
            raise RuntimeError("规划器没有返回 left_eef TF")
        # Keep this TCP snapshot paired with the ICP result.  Module 3 uses
        # the rigid object orientation in left_eef, while T_tool_object stays
        # the transfer-end relation used for Cartesian placement planning.
        T_body_left_eef = _planner_matrix(eef_response)
        T_left_eef_object = np.linalg.inv(T_body_left_eef) @ T_body_object
        triangles = load_object_triangles(cfg)
        cad_for_visualization = sample_surface_points(
            triangles, min(int(cfg["cad_sample_points"]), 12000), int(cfg["seed"])
        )
        pose_visualization = _save_pose_overlay(
            run_dir,
            view,
            cad_for_visualization,
            T_camera_object,
            T_body_object,
            np.asarray(report["camera_intrinsics"], dtype=float),
        )
        camera_visualization = run_dir / "debug_pose_estimation_camera_3d.png"
        body_visualization = run_dir / "debug_pose_estimation_body_base_3d.png"
        _plot_pose_estimation(
            camera_visualization,
            points,
            cad_for_visualization,
            T_camera_object,
            frame_name="camera",
            seed=int(cfg["seed"]),
        )
        _plot_pose_estimation(
            body_visualization,
            _transform_points(T_body_camera, points),
            cad_for_visualization,
            T_body_object,
            frame_name="body_base",
            seed=int(cfg["seed"]),
        )
        print(
            "[inspect] 工件位姿可视化: "
            f"{pose_visualization}（左侧SAM2掩码；右侧STL与工件坐标轴）\n"
            "[inspect] 工件位姿三维调试图: "
            f"camera={camera_visualization}, body_base={body_visualization}",
            flush=True,
        )

        object_z_camera = T_camera_object[:3, 2]
        object_z_body = T_body_object[:3, 2]
        object_z_body /= np.linalg.norm(object_z_body)
        positive_angle = float(np.degrees(np.arccos(np.clip(object_z_body[2], -1.0, 1.0))))
        negative_angle = float(np.degrees(np.arccos(np.clip(-object_z_body[2], -1.0, 1.0))))
        print(
            f"[inspect] object +Z(camera)={np.round(object_z_camera, 6).tolist()}, "
            f"+Z->body_base +Z={positive_angle:.3f}deg, "
            f"+Z->body_base -Z={negative_angle:.3f}deg"
        )

        if not args.skip_motion:
            planner.config = replace(
                planner.config,
                max_velocity=float(args.preset_speed),
                max_acceleration=float(args.preset_speed),
            )
            planner.set_next_plan_context(
                f"left_{view}_observation_pose", "left_graspup"
            )
            planner.move_left_joints([float(v) for v in cfg["graspup_joints_deg"]])
            # 回到 graspup 后仅检查夹爪是否仍报告 gripping；重量分类只在
            # Module 1 的同姿态 pregrasp 检查执行。
            final_holding_recheck = check_left_gripper_holding(
                "Module 2结束（graspup）"
            )
            print(
                "[inspect结束持件复检] "
                f"motion_state={final_holding_recheck.get('motion_state')} "
                f"passed={final_holding_recheck['passed']} "
                f"reason={final_holding_recheck.get('reason', '-')}",
                flush=True,
            )
            if not final_holding_recheck["passed"]:
                with Gripper(host=args.gripper_host, status_port=7702) as gripper:
                    gripper.move(
                        args.gripper_hand, 100.0, force=100.0,
                        speed=80.0,
                        mode="sync", preempt=False, timeout=60.0,
                    )
                write_stage_result(
                    args.result_file,
                    {
                        "stage": "left_inspect",
                        "outcome": "held_object_dropped",
                        "holding_recheck": final_holding_recheck,
                        "returned_to_graspup": True,
                        "gripper_final_position_percent": 100.0,
                    },
                )
                print(
                    "[inspect结束掉件恢复] 已回 graspup 并打开左夹爪；"
                    "返回总控从 Module 1 重新开始。",
                    flush=True,
                )
                return 0
        result = {
            "stage": "left_inspect",
            "outcome": "success",
            "observation_pose": view.upper(),
            "gripper_opening_percent": opening,
            "T_camera_object": T_camera_object.tolist(),
            "T_body_base_object": T_body_object.tolist(),
            "T_tool_object": T_tool_object.tolist(),
            "T_body_left_eef_at_inspect": T_body_left_eef.tolist(),
            "T_left_eef_object": T_left_eef_object.tolist(),
            "pose_visualizations": {
                "overlay": str(pose_visualization),
                "camera": str(camera_visualization),
                "body_base": str(body_visualization),
                "axis_convention": "object +X=red, +Y=green, FRONT(+Z)=magenta, -Z=gray",
            },
            "object_pose": {
                "frame_id": "body_base",
                "position": _pose(T_body_object).position,
                "orientation_xyzw": _pose(T_body_object).orientation,
            },
            "left_state": {
                "current_module": "2",
                "holding": (
                    str((weight_handoff or {}).get("weight_check", {}).get("classification", "UNKNOWN"))
                    if weight_handoff is not None else "SINGLE"
                ),
                "grasp_stable": True,
                "observation_class": "POSE_ESTIMATED",
                "observation_pose_used": view.upper(),
                "object_pose": {
                    "frame_id": "body_base",
                    "position": _pose(T_body_object).position,
                    "orientation_xyzw": _pose(T_body_object).orientation,
                },
                "pose_confidence": float(report["fitness"]),
            },
            "object_positive_z_in_camera": object_z_camera.tolist(),
            "object_positive_z_to_body_base_positive_z_deg": positive_angle,
            "object_positive_z_to_body_base_negative_z_deg": negative_angle,
            "registration": report,
            "graspnet_weight_handoff": weight_handoff,
            "returned_to_graspup": not args.skip_motion,
            "run_dir": str(run_dir),
        }
        write_stage_result(args.result_file, result)
        (run_dir / "result.json").write_text(
            json.dumps({"schema_version": 1, **result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[inspect] 已回 graspup；结果={args.result_file}")
        return 0
    finally:
        planner.close()


if __name__ == "__main__":
    # 独立运行也要留下完整终端记录；由总控启动时会同时被总控日志收集。
    run_root = DEFAULT_OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    log_path = run_root / "run.log"
    with RuntimeLogCapture(log_path):
        print(f"[log] 本次 Inspect 日志: {log_path}", flush=True)
        print(f"[log] command: {' '.join(sys.argv)}", flush=True)
        try:
            raise SystemExit(main())
        except SystemExit:
            raise
        except KeyboardInterrupt:
            print("[log] 收到 Ctrl+C，Inspect 中断。", flush=True)
            raise SystemExit(130)
        except BaseException:
            import traceback

            traceback.print_exc()
            raise SystemExit(1)
