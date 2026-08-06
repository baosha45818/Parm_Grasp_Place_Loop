"""Run GraspNet directly on robot-mounted RGB-D cameras."""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from robot_camera import (
    DEFAULT_D405_SERIAL,
    DEFAULT_FEMTO_BOLT_SERIAL,
    D405Camera,
    FemtoBoltCamera,
    list_orbbec_devices,
    list_realsense_devices,
)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True, help="Model checkpoint path")
parser.add_argument(
    "--camera",
    required=True,
    choices=["d405", "femto_bolt"],
    help="Robot camera to use",
)
parser.add_argument("--camera_serial", default=None, help="Override camera serial number")
parser.add_argument("--num_point", type=int, default=20000, help="Point Number [default: 20000]")
parser.add_argument("--num_view", type=int, default=300, help="View Number [default: 300]")
parser.add_argument(
    "--collision_thresh",
    type=float,
    default=0.01,
    help="Collision Threshold in collision detection [default: 0.01]",
)
parser.add_argument(
    "--voxel_size",
    type=float,
    default=0.01,
    help="Voxel Size to process point clouds before collision detection [default: 0.01]",
)
parser.add_argument("--d405_width", type=int, default=640, help="D405 color/depth width")
parser.add_argument("--d405_height", type=int, default=480, help="D405 color/depth height")
parser.add_argument("--femto_color_width", type=int, default=1280, help="Femto Bolt color width")
parser.add_argument("--femto_color_height", type=int, default=720, help="Femto Bolt color height")
parser.add_argument("--femto_depth_width", type=int, default=640, help="Femto Bolt depth width")
parser.add_argument("--femto_depth_height", type=int, default=480, help="Femto Bolt depth height")
cfgs = parser.parse_args()


def get_net():
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, checkpoint["epoch"]))
    net.eval()
    return net


def _sample_points(cloud_masked, color_masked):
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return cloud_masked[idxs], color_masked[idxs]


def _project_depth_colors_to_points(color, cloud, color_intrinsics, rotation, translation):
    points = cloud.reshape(-1, 3)
    colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    valid = points[:, 2] > 0
    if not np.any(valid):
        return colors.reshape(cloud.shape)

    points_valid = points[valid]
    points_color = (rotation @ points_valid.T).T + translation.reshape(1, 3)
    z = points_color[:, 2]
    keep = z > 1e-6
    if not np.any(keep):
        return colors.reshape(cloud.shape)

    valid_indices = np.flatnonzero(valid)[keep]
    points_color = points_color[keep]
    u = np.round(
        points_color[:, 0] * color_intrinsics[0, 0] / points_color[:, 2] + color_intrinsics[0, 2]
    ).astype(np.int32)
    v = np.round(
        points_color[:, 1] * color_intrinsics[1, 1] / points_color[:, 2] + color_intrinsics[1, 2]
    ).astype(np.int32)

    in_bounds = (
        (u >= 0)
        & (u < color.shape[1])
        & (v >= 0)
        & (v < color.shape[0])
    )
    if np.any(in_bounds):
        colors[valid_indices[in_bounds]] = color[v[in_bounds], u[in_bounds]].astype(np.float32) / 255.0
    return colors.reshape(cloud.shape)


def build_cloud_and_colors(frame):
    color = frame.color
    depth = frame.depth
    depth_h, depth_w = depth.shape
    camera = CameraInfo(
        depth_w,
        depth_h,
        frame.depth_intrinsics[0, 0],
        frame.depth_intrinsics[1, 1],
        frame.depth_intrinsics[0, 2],
        frame.depth_intrinsics[1, 2],
        1.0 / frame.depth_scale,
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = depth > 0

    if color.shape[:2] == depth.shape:
        color_float = color.astype(np.float32) / 255.0
    elif (
        frame.depth_to_color_rotation is not None
        and frame.depth_to_color_translation is not None
    ):
        color_float = _project_depth_colors_to_points(
            color,
            cloud,
            frame.color_intrinsics,
            frame.depth_to_color_rotation,
            frame.depth_to_color_translation,
        )
    else:
        color_float = np.zeros((depth_h, depth_w, 3), dtype=np.float32)

    cloud_masked = cloud[mask]
    color_masked = color_float[mask]
    cloud_sampled, color_sampled = _sample_points(cloud_masked, color_masked)

    vis_cloud = o3d.geometry.PointCloud()
    vis_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    vis_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    end_points = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    end_points["point_clouds"] = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points["cloud_colors"] = color_sampled
    return end_points, vis_cloud


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)


def collision_detection(gg, cloud):
    detector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = detector.detect(
        gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
    )
    return gg[~collision_mask]


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def make_camera():
    if cfgs.camera == "d405":
        return D405Camera(
            serial=cfgs.camera_serial or DEFAULT_D405_SERIAL,
            width=cfgs.d405_width,
            height=cfgs.d405_height,
        )
    return FemtoBoltCamera(
        serial=cfgs.camera_serial or DEFAULT_FEMTO_BOLT_SERIAL,
        color_width=cfgs.femto_color_width,
        color_height=cfgs.femto_color_height,
        depth_width=cfgs.femto_depth_width,
        depth_height=cfgs.femto_depth_height,
    )


def print_available_devices():
    print("RealSense devices:", list_realsense_devices())
    print("Orbbec devices:", list_orbbec_devices())


def demo():
    print_available_devices()
    net = get_net()
    with make_camera() as camera:
        frame = camera.read()
    print("-> captured frame from %s (%s)" % (frame.camera_name, frame.serial))
    end_points, cloud = build_cloud_and_colors(frame)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)


if __name__ == "__main__":
    demo()
