"""实现胸前 Femto Bolt 的工件视角分类、正反判断和六自由度位姿估计。"""

from __future__ import annotations

import http.client
import io
import json
from pathlib import Path
import socket
from time import time
from urllib.parse import urlencode

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from loop_types import (
    FaceSide,
    Holding,
    LeftArmState,
    ObjectPose,
    ObservationClass,
    ResultStatus,
    make_result,
)
from tools.planner_runtime import PlannerRuntime


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("inspect_config.json")
from tools.inspection_support.mask_depth import (
    mask_depth_diagnostics,
    mask_depth_to_points_cam,
    remove_radius_outliers_np,
    voxel_downsample_np,
)
from tools.inspection_support.registration import (
    register_cad_to_observed_candidates,
)
from tools.inspection_support.stl_model import (
    load_stl_triangles_m,
    sample_surface_points,
)
from tools.inspection_support.robot_camera import FemtoBoltCamera


def _reset_inspection_state(left: LeftArmState) -> None:
    left.observation_class = ObservationClass.UNKNOWN
    left.face_side = FaceSide.UNKNOWN
    left.grasp_region = "UNKNOWN"
    left.object_pose = None
    left.pose_confidence = 0.0
    left.orientation_confidence = 0.0
    left.observation_pose_used = ""
    left.pose_quality.clear()


def _transform_from_planner_response(response: dict) -> np.ndarray:
    data = response.get("data") or response
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [translation.get(axis) for axis in "xyz"]
    xyzw = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in xyz + xyzw):
        raise RuntimeError(f"规划器 TF 返回格式无效: {response}")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(xyzw, dtype=float)).as_matrix()
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def _load_base_camera_transform(cfg: dict, planner: PlannerRuntime) -> np.ndarray:
    # 首选直接标定得到的 body_base <- SDK base。胸前 Femto 固定在机身上，
    # 因而 T_body_base_camera = T_body_base_sdk_base @ T_sdk_base_camera，
    # 不依赖头部 D435 当前的云台角。
    alignment_path_value = str(cfg.get("sdk_base_alignment_path") or "").strip()
    if alignment_path_value:
        alignment_path = Path(alignment_path_value)
        chest_path = Path(cfg["chest_calibration_path"])
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        chest = json.loads(chest_path.read_text(encoding="utf-8"))
        if chest.get("status") != "ACCEPTED":
            raise RuntimeError(f"胸前Femto标定未通过验收: {chest.get('status')}")
        if alignment.get("transform_convention") != "p_A = T_A_B @ p_B":
            raise RuntimeError(
                "SDK base对齐文件的变换约定缺失或不匹配: "
                f"{alignment.get('transform_convention')!r}"
            )
        T_body_base_sdk_base = np.asarray(
            alignment["T_body_base_sdk_base"], dtype=float
        )
        T_sdk_base_camera = np.asarray(chest["T_base_laser"], dtype=float)
        for label, matrix in (
            ("T_body_base_sdk_base", T_body_base_sdk_base),
            ("T_base_laser", T_sdk_base_camera),
        ):
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise RuntimeError(f"{label} 必须是有限的 4x4 矩阵")
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
                raise RuntimeError(f"{label} 不是合法齐次变换")
        return T_body_base_sdk_base @ T_sdk_base_camera

    camera_frame = str(cfg.get("planner_camera_frame") or "").strip()
    if camera_frame:
        response = planner.lookup_transform(planner.config.base_frame, camera_frame)
        if response is None:
            raise RuntimeError("dry-run 不会生成可写入 State 的 base 系工件位姿")
        return _transform_from_planner_response(response)

    chest_path = Path(cfg["chest_calibration_path"])
    bridge_path = Path(cfg["planner_tf_bridge_path"])
    chest = json.loads(chest_path.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    if chest.get("status") != "ACCEPTED":
        raise RuntimeError(f"胸前Femto标定未通过验收: {chest.get('status')}")
    if bridge.get("status") != "ACCEPTED":
        raise RuntimeError(f"规划器 TF Bridge 未通过验收: {bridge.get('status')}")
    T_old_base_chest = np.asarray(chest["T_base_laser"], dtype=float)
    T_planner_arm_base_old_base = np.asarray(bridge["T_planner_base_base"], dtype=float)
    for label, matrix in (
        ("T_base_laser", T_old_base_chest),
        ("T_planner_base_base", T_planner_arm_base_old_base),
    ):
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"{label} 必须是有限的 4x4 矩阵")

    bridge_base_frame = str(cfg["planner_bridge_base_frame"])
    response = planner.lookup_transform(planner.config.base_frame, bridge_base_frame)
    if response is None:
        raise RuntimeError("dry-run 不会生成可写入 State 的 body_base 系工件位姿")
    T_body_base_arm_base = _transform_from_planner_response(response)
    return T_body_base_arm_base @ T_planner_arm_base_old_base @ T_old_base_chest


def _capture_cloud(camera: FemtoBoltCamera, cfg: dict) -> tuple[np.ndarray, dict]:
    frames = [camera.read() for _ in range(int(cfg["capture_frames"]))]
    color = frames[-1].color
    depth = np.median(np.stack([frame.depth for frame in frames]), axis=0).astype(np.uint16)
    if depth.shape != color.shape[:2]:
        raise RuntimeError(
            f"Femto RGB/depth未对齐: rgb={color.shape[:2]}, depth={depth.shape}；"
            "模块2禁止用缩放代替相机几何对齐"
        )
    roi = tuple(int(value) for value in cfg["roi_xywh"])
    class UnixConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(str(cfg["sam2_socket"]))

    params = urlencode({
        "height": color.shape[0],
        "width": color.shape[1],
        "num_masks": 1,
        "min_mask_area_px": int(cfg["min_mask_area_px"]),
        "roi": ",".join(str(value) for value in roi),
    })
    connection = UnixConnection("localhost", timeout=65.0)
    try:
        connection.request(
            "POST",
            f"/v1/segment?{params}",
            body=np.ascontiguousarray(color).tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        reply = connection.getresponse()
        payload = reply.read()
        if reply.status != 200:
            raise RuntimeError(f"SAM2 服务返回 HTTP {reply.status}: {payload.decode(errors='replace')}")
    finally:
        connection.close()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        masks = archive["masks"]
    if len(masks) != 1:
        raise RuntimeError("SAM2 没有返回唯一工件掩码")
    mask = (masks[0] > 0).astype(np.uint8) * 255
    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
    x, y, width, height = roi
    roi_mask[y:y + height, x:x + width] = 255
    mask = cv2.bitwise_and(mask, roi_mask)
    diagnostics = mask_depth_diagnostics(
        mask,
        depth,
        frames[-1].depth_scale,
        min_depth_m=float(cfg["min_depth_m"]),
        max_depth_m=float(cfg["max_depth_m"]),
        erode_px=int(cfg["erode_px"]),
    )
    points = mask_depth_to_points_cam(
        mask,
        depth,
        frames[-1].depth_intrinsics,
        frames[-1].depth_scale,
        min_depth_m=float(cfg["min_depth_m"]),
        max_depth_m=float(cfg["max_depth_m"]),
        erode_px=int(cfg["erode_px"]),
    )
    points = remove_radius_outliers_np(points)
    points = voxel_downsample_np(points, float(cfg["observed_voxel_m"]))
    diagnostics["valid_point_count"] = int(len(points))
    return points, diagnostics


def module_2_inspect(left: LeftArmState, planner: PlannerRuntime):
    started = time()
    left.current_module = "2"
    _reset_inspection_state(left)
    if left.holding is not Holding.SINGLE or not left.grasp_stable:
        return make_result(ResultStatus.RETRY, "2", started, "左手不是稳定的单工件抓取状态")

    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        stl_path = Path(cfg["stl_path"])
        if not stl_path.is_file():
            raise RuntimeError(f"工件 STL 不存在: {stl_path}")
        planner.move_left_joints([float(value) for value in cfg["pc_joints_deg"]])
        with FemtoBoltCamera(
            serial=str(cfg["camera_serial"]),
            color_width=int(cfg["color_width"]),
            color_height=int(cfg["color_height"]),
            depth_width=int(cfg["depth_width"]),
            depth_height=int(cfg["depth_height"]),
            fps=30,
            warmup_frames=int(cfg["warmup_frames"]),
        ) as camera:
            points_camera, diagnostics = _capture_cloud(camera, cfg)
            if len(points_camera) < int(cfg["face_visible_min_points"]):
                left.observation_class = ObservationClass.OBLIQUE
                left.face_side = FaceSide.NOT_APPLICABLE
                left.observation_pose_used = "PC_2"
                planner.move_left_joints([float(value) for value in cfg["pc2_joints_deg"]])
                points_camera, diagnostics = _capture_cloud(camera, cfg)
            else:
                left.observation_class = ObservationClass.FACE_VISIBLE
                left.observation_pose_used = "PC"

        if len(points_camera) < int(cfg["min_registration_points"]):
            raise ValueError(f"有效工件点云过少: {len(points_camera)}")
        triangles = load_stl_triangles_m(stl_path)
        cad = sample_surface_points(
            triangles,
            n_points=int(cfg["cad_sample_points"]),
            seed=int(cfg["seed"]),
        )
        candidates = register_cad_to_observed_candidates(
            cad,
            points_camera,
            max_corr_m=float(cfg["icp_max_corr_m"]),
            max_iters=int(cfg["icp_max_iters"]),
            max_cad_points=int(cfg["icp_max_points"]),
            max_obs_points=int(cfg["icp_max_points"]),
            seed=int(cfg["seed"]),
        )
        best = candidates[0]
        if best.rmse_m > float(cfg["quality_max_rmse_m"]) or best.fitness < float(cfg["quality_min_fitness"]):
            raise ValueError(f"点云配准质量不足: fitness={best.fitness:.3f}, rmse={best.rmse_m:.4f}m")

        T_camera_object = np.asarray(best.T_target_source, dtype=float)
        if left.observation_class is ObservationClass.FACE_VISIBLE:
            front_normal_camera = T_camera_object[:3, :3] @ np.array([0.0, 0.0, 1.0])
            score = float(np.clip(-front_normal_camera[2], -1.0, 1.0))
            left.face_side = FaceSide.FRONT if score >= 0.0 else FaceSide.BACK
            left.orientation_confidence = abs(score)
            if left.orientation_confidence < float(cfg["min_face_alignment"]):
                raise ValueError("正反面方向置信度不足")

        T_base_camera = _load_base_camera_transform(cfg, planner)
        T_base_object = T_base_camera @ T_camera_object
        quaternion = Rotation.from_matrix(T_base_object[:3, :3]).as_quat()
        left.object_pose = ObjectPose(
            frame_id=planner.config.base_frame,
            position=T_base_object[:3, 3].tolist(),
            orientation=quaternion.tolist(),
        )
        left.pose_confidence = float(best.fitness)
        left.pose_quality = {
            "fitness": float(best.fitness),
            "rmse_m": float(best.rmse_m),
            "inlier_count": int(best.n_corr),
            "valid_point_count": int(len(points_camera)),
            "valid_depth_ratio": float(diagnostics["depth_valid_ratio"]),
            "T_camera_object": T_camera_object.tolist(),
        }
        return make_result(ResultStatus.SUCCESS, "2", started, "工件分类和 base 系位姿估计完成")
    except ValueError as exc:
        left.last_error = str(exc)
        return make_result(ResultStatus.RETRY, "2", started, str(exc))
    except Exception as exc:
        left.last_error = str(exc)
        return make_result(ResultStatus.FAULT, "2", started, str(exc))
