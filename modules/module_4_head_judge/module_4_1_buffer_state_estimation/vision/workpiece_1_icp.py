#!/usr/bin/env python3
"""Classify workpiece face with two table-constrained CAD/ICP hypotheses.

The CAD +Z face is the validated physical positive face (defined once in
config).  Each SAM2 mask is registered twice: the positive and reverse faces
towards the camera.  Point-to-plane ICP may only translate and yaw about the
table normal, so it cannot silently rotate one face hypothesis into the other.
A dense CAD z-buffer is
then compared with the observed aligned depth; points on the CAD peg and wide
head receive extra weight.  Ambiguous score pairs remain ``unknown``.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
ROOT = MODULE_ROOT.parents[2]
BODY_HEAD_CALIBRATION = (
    Path(os.environ.get(
        'PARM_BODY_HEAD_CALIBRATION',
        ROOT / 'tools/calibration/head_d435/xiangxin_reverse_observation.json',
    )).expanduser().resolve()
)
sys.path[:0] = [str(ROOT), str(HERE)]
from tools.inspection_support.mask_depth import mask_depth_to_points_cam, voxel_downsample_np, remove_radius_outliers_np
from tools.inspection_support.stl_model import load_stl_triangles_m, sample_surface_points
from tools.icp import tabletop_constrained_point_to_plane_icp
from .geometry import workspace_roi_mask


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def table_normal(points):
    """Fit the actual tabletop, never an individual workpiece surface."""
    o3d.utility.random.seed(17)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.004)
    plane, idx = cloud.segment_plane(.0025, 3, 800)
    plane = np.asarray(plane, dtype=float)
    center = np.asarray(points)[idx].mean(0)
    if np.dot(plane[:3], -center) < 0:
        plane *= -1
    return normalize(plane[:3])


def axis_angle(axis, angle):
    x, y, z = normalize(axis); c, s = np.cos(angle), np.sin(angle); C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s], [y*x*C+z*s, c+y*y*C, y*z*C-x*s], [z*x*C-y*s, z*y*C-x*s, z*z*C+c]])


def transform(T, p):
    return (T[:3, :3] @ np.asarray(p).T).T + T[:3, 3]


def planar_basis(points, normal):
    q = np.asarray(points) - np.mean(points, axis=0)
    q -= (q @ normal)[:, None] * normal
    _, vecs = np.linalg.eigh(np.cov(q.T))
    x = vecs[:, -1] - normal * np.dot(vecs[:, -1], normal)
    x = normalize(x)
    return x, normalize(np.cross(normal, x))


def constrained_icp(source, target, initial, normal, max_iters=45, max_corr=.009):
    """Point-to-plane ICP whose rotation is yaw-only about the table normal."""
    return tabletop_constrained_point_to_plane_icp(
        source, target, initial, normal, max_iterations=max_iters, max_corr_m=max_corr
    )


def face_initials(cad, observed, table_n, cad_up):
    """Four yaw starts for each requested face; face normal remains fixed."""
    obs_x, obs_y = planar_basis(observed, table_n); cad_center = cad.mean(0); obs_center = observed.mean(0)
    starts = []
    for side, source_n in (('positive', cad_up), ('reverse', -cad_up)):
        src_x = normalize(np.array([1., 0., 0.]) - source_n*np.dot([1., 0., 0.], source_n))
        src_y = normalize(np.cross(source_n, src_x)); src_basis = np.column_stack((src_x, src_y, source_n))
        for q in range(4):
            theta = q*np.pi/2; tx = np.cos(theta)*obs_x + np.sin(theta)*obs_y; ty = normalize(np.cross(table_n, tx))
            R = np.column_stack((tx, ty, table_n)) @ src_basis.T
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = obs_center - R@cad_center
            starts.append((side, T))
    return starts


def depth_score(T, cad_render, feature, depth, mask, K, scale):
    """Visible-surface depth error from a dense CAD point z-buffer/splat."""
    p = transform(T, cad_render); good = p[:, 2] > .1; p, feature = p[good], feature[good]
    uv = np.column_stack((K[0,0]*p[:,0]/p[:,2]+K[0,2], K[1,1]*p[:,1]/p[:,2]+K[1,2]))
    inside = (uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0])
    uv, p, feature = uv[inside], p[inside], feature[inside]
    tree = cKDTree(uv)
    yy, xx = np.nonzero(mask & (depth > 0)); z = depth[yy,xx].astype(float)*scale
    if len(z) > 14000:
        take = np.linspace(0, len(z)-1, 14000).astype(int); yy, xx, z = yy[take], xx[take], z[take]
    pix = np.column_stack((xx, yy)); d2, ind = tree.query(pix, k=1, distance_upper_bound=2.5)
    keep = np.isfinite(d2)
    if keep.sum() < 80:
        return float('inf'), {'match_fraction': float(keep.mean()), 'depth_mae_m': None}
    residual = np.abs(p[ind[keep], 2]-z[keep]); weights = np.where(feature[ind[keep]], 3.0, 1.0)
    mae = float(np.average(np.minimum(residual, .020), weights=weights))
    # Missing predicted pixels are evidence too, but lightly weighted for occlusion.
    score = mae + .004*float(1-keep.mean())
    return score, {'match_fraction': float(keep.mean()), 'depth_mae_m': mae, 'weighted_feature_fraction': float(np.mean(feature[ind[keep]]))}


def classify(cad_icp, cad_render, feature, observed, depth, mask, K, scale, cad_up, normal):
    trials = []
    for side, initial in face_initials(cad_icp, observed, normal, cad_up):
        T, rmse, fitness = constrained_icp(cad_icp, observed, initial, normal)
        _score, render = depth_score(T, cad_render, feature, depth, mask, K, scale)
        trials.append({'side': side, 'T': T, 'icp_method': 'point_to_plane',
                       'icp_rmse_m': rmse, 'icp_fitness': fitness, 'render': render})
    quality_key = lambda x: (x['icp_fitness'], x['render']['match_fraction'], -x['icp_rmse_m'])
    best = {side: max((x for x in trials if x['side']==side), key=quality_key) for side in ('positive','reverse')}
    side = max(best, key=lambda name: quality_key(best[name]))
    return side, best


def _project(points, K, shape):
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > .1)
    points = points[valid]
    uv = np.column_stack((
        K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
        K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
    ))
    uv = np.rint(uv).astype(np.int32)
    height, width = shape[:2]
    inside = ((0 <= uv[:, 0]) & (uv[:, 0] < width)
              & (0 <= uv[:, 1]) & (uv[:, 1] < height))
    return uv[inside]


def draw_icp_alignment(image, mask, K, T, cad_render, side, quality):
    """Overlay the selected aligned CAD, observed mask contour, and object axes."""
    canvas = image.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    aligned = transform(T, cad_render)
    stride = max(1, len(aligned) // 7000)
    pixels = _project(aligned[::stride], K, canvas.shape)
    cad_layer = np.zeros(mask.shape, dtype=np.uint8)
    if len(pixels):
        cad_layer[pixels[:, 1], pixels[:, 0]] = 255
        cad_layer = cv2.dilate(cad_layer, np.ones((3, 3), np.uint8))
        selected = cad_layer > 0
        color = np.array((255, 0, 255), dtype=np.float64)
        canvas[selected] = (
            .35 * canvas[selected].astype(np.float64) + .65 * color
        ).astype(np.uint8)

    origin = np.asarray(T[:3, 3], dtype=float)
    axes = np.asarray([origin, *(origin + T[:3, :3][:, axis] * .04 for axis in range(3))])
    projected = _project(axes, K, canvas.shape)
    # _project preserves all points here when the pose lies inside the image.
    if len(projected) == 4:
        start = tuple(projected[0])
        for endpoint, color, label in zip(
            projected[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), ("X", "Y", "Z")
        ):
            end = tuple(endpoint)
            cv2.arrowedLine(canvas, start, end, color, 3, cv2.LINE_AA, tipLength=.18)
            cv2.putText(canvas, label, end, cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    label = (
        f"{side} point-to-plane ICP  fitness={quality['icp_fitness']:.3f}  "
        f"rmse={quality['icp_rmse_m'] * 1000.0:.2f}mm  "
        f"depth={quality['render']['match_fraction']:.3f}"
    )
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 920), 42), (0, 0, 0), -1)
    cv2.putText(canvas, label, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, aligned


def write_icp_alignment_3d(path, observed, aligned_cad, table_n, side, quality):
    """Save a headless 3-D view of observed points and the selected aligned CAD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sample(points, maximum):
        points = np.asarray(points, dtype=float)
        if len(points) <= maximum:
            return points
        return points[np.linspace(0, len(points) - 1, maximum).astype(int)]

    observed_plot = sample(observed, 3500)
    cad_plot = sample(aligned_cad, 5000)
    both = np.vstack((observed_plot, cad_plot))
    center = (both.min(axis=0) + both.max(axis=0)) * .5
    radius = max(float(np.ptp(both, axis=0).max()) * .55, .03)
    fig = plt.figure(figsize=(8.5, 7.0), dpi=150)
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(observed_plot[:, 0], observed_plot[:, 1], observed_plot[:, 2],
                 s=2, c="#20c997", alpha=.65, label="observed depth")
    axis.scatter(cad_plot[:, 0], cad_plot[:, 1], cad_plot[:, 2],
                 s=2, c="#d633ff", alpha=.42, label="aligned CAD")
    cloud_center = observed_plot.mean(axis=0)
    axis.quiver(*cloud_center, *normalize(table_n), length=.05,
                color="#0066ff", linewidth=2, label="table normal")
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("camera X / m")
    axis.set_ylabel("camera Y / m")
    axis.set_zlabel("camera Z / m")
    axis.set_title(
        f"{side} constrained point-to-plane ICP\n"
        f"fitness={quality['icp_fitness']:.3f}, "
        f"RMSE={quality['icp_rmse_m'] * 1000.0:.2f} mm"
    )
    axis.legend(loc="upper right")
    axis.view_init(elev=24, azim=-62)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--stl', type=Path, default=ROOT/'tools/workpiece_1/welding_up_side_1118_only_2_small_screws_removed.stl')
    p.add_argument('--orientation-config', type=Path, default=MODULE_ROOT/'configs/workpiece_1.json')
    a = p.parse_args(); d = a.capture_dir
    meta = json.loads((d/'metadata.json').read_text()); rgb = cv2.imread(str(d/'rgb_000.png')); depth = cv2.imread(str(d/'depth_aligned_000.png'), cv2.IMREAD_UNCHANGED)
    masks = np.load(d/'sam2_masks/masks.npz')['masks']; k = meta['color_intrinsics']; K = np.array([[k['fx'],0,k['ppx']],[0,k['fy'],k['ppy']],[0,0,1.]])
    tris = load_stl_triangles_m(a.stl); cad_icp = sample_surface_points(tris, 14000, 3); cad_render = sample_surface_points(tris, 90000, 4)
    # The deployment mode binds this calibration to the captured camera serial.
    orientation = json.loads(a.orientation_config.read_text())
    body_head = json.loads(BODY_HEAD_CALIBRATION.read_text())
    transform_key = next((key for key in ('T_body_base_head_camera','T_base_head_camera','T') if key in body_head),None)
    if transform_key is None:
        raise RuntimeError(f'头部标定缺少 body_base <- head_camera 变换: {BODY_HEAD_CALIBRATION}')
    T_body_base_camera = np.asarray(body_head[transform_key], dtype=float)
    workspace_path = Path(os.environ.get('PARM_HEAD_TABLETOP_WORKSPACE', a.orientation_config)).expanduser().resolve()
    workspace_payload = json.loads(workspace_path.read_text())
    workspace = workspace_payload.get('mask', workspace_payload)
    safe_roi = workspace_roi_mask(workspace, depth.shape).astype(bool)
    # A per-object RANSAC can fit the workpiece itself and flip its normal.
    # Estimate this once from the complete scene, where the tabletop dominates.
    scene_points = mask_depth_to_points_cam(np.ones(depth.shape, dtype=bool), depth, K, float(meta['depth_scale_m']), .35, 1.2, 1)
    scene_normal = table_normal(scene_points)
    cad_up = normalize(orientation['positive_face_normal_stl']); feature = (cad_render @ cad_up >= np.percentile(cad_render @ cad_up, 88))
    diagnostics = []
    alignment_overlay = d/'sam2_masks/icp_alignment_overlay.png'
    orient = rgb.copy(); alignment = rgb.copy(); alignment_paths = []
    for i, raw_m in enumerate(masks):
        m = raw_m.astype(bool) & safe_roi
        if np.count_nonzero(m) < 350:
            diagnostics.append({'object_key':f'sam2-{i}', 'classification':'rejected_outside_safe_roi'})
            continue
        pts = voxel_downsample_np(remove_radius_outliers_np(mask_depth_to_points_cam(m, depth, K, float(meta['depth_scale_m']), .35, 1.2, 1)), .003)
        if len(pts) < 120: continue
        side, best = classify(cad_icp, cad_render, feature, pts, depth, m.astype(bool), K, float(meta['depth_scale_m']), cad_up, scene_normal)
        reverse_quality = best['reverse']
        reverse_match = float(reverse_quality['render']['match_fraction'])
        reverse_fitness = float(reverse_quality['icp_fitness'])
        geometry_ok = (
            reverse_fitness >= float(orientation['minimum_reverse_icp_fitness'])
            and reverse_match >= float(orientation['minimum_reverse_depth_match_fraction'])
        )
        accepted_side = side if side != 'reverse' or geometry_ok else 'rejected_geometry'
        selected_T = best[side]['T']
        T_body_base_object = T_body_base_camera @ selected_T
        # Keep the winning ICP pose in the public diagnostic contract.  The
        # Buffer recovery path consumes this exact estimate to re-grasp a part
        # which has been corrected to its front side; recomputing a centre from
        # the mask would discard the CAD/ICP registration we just validated.
        suffix = '' if len(masks) == 1 else f'_{i:03d}'
        alignment, aligned_cad = draw_icp_alignment(
            alignment, m, K, selected_T, cad_render, side, best[side]
        )
        alignment_3d_path = d/f'sam2_masks/icp_alignment_3d{suffix}.png'
        write_icp_alignment_3d(
            alignment_3d_path, pts, aligned_cad, scene_normal, side, best[side]
        )
        alignment_paths.append(str(alignment_3d_path))
        diag = {'object_key': f'sam2-{i}', 'classification': accepted_side,
                'table_normal_camera': scene_normal.round(9).tolist(),
                'selected_hypothesis': side,
                'pose': {
                    'valid': accepted_side in {'positive', 'reverse'},
                    'orientation_valid': accepted_side in {'positive', 'reverse'},
                    'method': 'tabletop_front_back_multi_yaw_constrained_point_to_plane_icp',
                    'T_head_camera_workpiece': selected_T.round(9).tolist(),
                    'T_body_base_workpiece': T_body_base_object.round(9).tolist(),
                    'calibration': str(BODY_HEAD_CALIBRATION),
                    'calibrated_head_pose_deg': body_head.get('head_pose_deg'),
                    'visualization_3d': str(alignment_3d_path),
                },
                'reverse_geometry_quality': {'icp_fitness': reverse_fitness, 'minimum_icp_fitness': orientation['minimum_reverse_icp_fitness'], 'depth_match_fraction': reverse_match, 'minimum_depth_match_fraction': orientation['minimum_reverse_depth_match_fraction'], 'accepted': geometry_ok},
                'visualization': {
                    'icp_alignment_overlay': str(alignment_overlay),
                    'icp_alignment_3d': str(alignment_3d_path),
                },
                'positive': {**{k:v for k,v in best['positive'].items() if k!='T'}, 'T_camera_object': best['positive']['T'].round(9).tolist()},
                'reverse': {**{k:v for k,v in best['reverse'].items() if k!='T'}, 'T_camera_object': best['reverse']['T'].round(9).tolist()}}
        diagnostics.append(diag)
        color = {'positive': (0,180,0), 'reverse': (0,0,220), 'unknown': (0,180,255), 'rejected_geometry': (120,120,120)}[accepted_side]
        tint = np.zeros_like(orient); tint[:]=color; orient[m>0] = (.55*orient[m>0]+.45*tint[m>0]).astype(np.uint8)
        ys, xs = np.nonzero(m); cv2.putText(orient, f"{accepted_side} f={best[side]['icp_fitness']:.2f} d={best[side]['render']['match_fraction']:.2f}", (int(xs.mean()),int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
    overlay = d/'sam2_masks/orientation_overlay.png'
    diagnostics_path = d/'sam2_masks/orientation_diagnostics.json'
    cv2.imwrite(str(overlay), orient)
    cv2.imwrite(str(alignment_overlay), alignment)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({'objects': len(diagnostics), 'diagnostics': str(diagnostics_path),
                      'orientation_overlay': str(overlay),
                      'icp_alignment_overlay': str(alignment_overlay),
                      'icp_alignment_3d': alignment_paths}))


if __name__ == '__main__': main()
