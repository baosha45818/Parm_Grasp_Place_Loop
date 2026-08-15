#!/usr/bin/env python3
"""Classify workpiece face with two table-constrained CAD/ICP hypotheses.

The CAD +Z face is the validated physical positive face (defined once in
config).  Each SAM2 mask is registered twice: the positive and reverse faces
towards the camera.  ICP may only translate and yaw about the table normal, so it cannot
silently rotate one face hypothesis into the other.  A dense CAD z-buffer is
then compared with the observed aligned depth; points on the CAD peg and wide
head receive extra weight.  Ambiguous score pairs remain ``unknown``.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
from tool.inspection_support.mask_depth import mask_depth_to_points_cam, voxel_downsample_np, remove_radius_outliers_np
from tool.inspection_support.stl_model import load_stl_triangles_m, sample_surface_points
from detect_reverse_workpieces import widest_head_in_stl, make_grasp_frame, draw_grasp_overlay, pose_record


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


def signed_rotation_about_axis(rotation, axis):
    """Signed angle of a rotation around ``axis``."""
    rotation = np.asarray(rotation, dtype=float)
    skew_vector = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) * .5
    return float(np.arctan2(np.dot(normalize(axis), skew_vector), (np.trace(rotation) - 1.) * .5))


def transform(T, p):
    return (T[:3, :3] @ np.asarray(p).T).T + T[:3, 3]


def planar_basis(points, normal):
    q = np.asarray(points) - np.mean(points, axis=0)
    q -= (q @ normal)[:, None] * normal
    _, vecs = np.linalg.eigh(np.cov(q.T))
    x = vecs[:, -1] - normal * np.dot(vecs[:, -1], normal)
    x = normalize(x)
    return x, normalize(np.cross(normal, x))


def constrained_icp(source, target, initial, normal, max_iters=45, max_corr=.009,
                    yaw_reference_rotation=None, yaw_min_rad=None, yaw_max_rad=None):
    """Point-to-plane ICP constrained to tabletop yaw and camera-frame translation."""
    tree = cKDTree(target); T = initial.copy(); normal = normalize(normal)
    reference_rotation = None if yaw_reference_rotation is None else np.asarray(yaw_reference_rotation, dtype=float)
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    target_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(.012, max_corr * 2.0), max_nn=40)
    )
    target_normals = np.asarray(target_cloud.normals)
    target_normals /= np.maximum(np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-12)
    for _ in range(max_iters):
        sw = transform(T, source); dist, near = tree.query(sw, k=1, workers=-1)
        keep = dist < max_corr
        if keep.sum() < 30:
            keep[np.argsort(dist)[:min(250, len(dist))]] = True
        cutoff = np.quantile(dist[keep], .82); keep &= dist <= cutoff
        a, b, b_normals = sw[keep], target[near[keep]], target_normals[near[keep]]
        residual = np.einsum('ij,ij->i', b_normals, a - b)
        # d(theta) rotates each source point about the table normal.  The
        # remaining three variables are translation in the camera frame.
        yaw_jacobian = np.einsum('ij,ij->i', b_normals, np.cross(normal, a))
        jacobian = np.column_stack((yaw_jacobian, b_normals))
        delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        theta = float(np.clip(delta[0], -.05, .05))
        translation = np.clip(delta[1:], -.008, .008)
        dT = np.eye(4); dT[:3, :3] = axis_angle(normal, theta); dT[:3, 3] = translation
        T = dT @ T
        if reference_rotation is not None and yaw_min_rad is not None and yaw_max_rad is not None:
            relative_yaw = signed_rotation_about_axis(T[:3, :3] @ reference_rotation.T, normal)
            bounded_yaw = float(np.clip(relative_yaw, yaw_min_rad, yaw_max_rad))
            if bounded_yaw != relative_yaw:
                T[:3, :3] = axis_angle(normal, bounded_yaw - relative_yaw) @ T[:3, :3]
        if abs(theta) < 2e-5 and np.linalg.norm(translation) < 2e-6:
            break
    final = transform(T, source); dist, near = tree.query(final, k=1, workers=-1); inlier = dist < max_corr
    if not inlier.any():
        return T, float('inf'), float(inlier.mean())
    plane_residual = np.einsum(
        'ij,ij->i', target_normals[near[inlier]], final[inlier] - target[near[inlier]]
    )
    return T, float(np.sqrt(np.mean(plane_residual ** 2))), float(inlier.mean())


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


def wide_head_outward_direction(vertices, head_center, long_axis):
    """Unit STL direction from the wide head centre towards its outside end."""
    v = np.asarray(vertices, dtype=float); origin = v.mean(axis=0); u = (v-origin) @ long_axis
    head_u = float((np.asarray(head_center)-origin) @ long_axis)
    inward = long_axis if abs(head_u-u.min()) < abs(head_u-u.max()) else -long_axis
    return -inward


def wide_head_grasp_point(head_center, outward_direction, outward_m, inward_m, head_towards_camera):
    """Offset from the wide-head centre, moving inward only when its end faces the camera."""
    direction = -outward_direction if head_towards_camera else outward_direction
    distance = inward_m if head_towards_camera else outward_m
    return np.asarray(head_center, dtype=float) + direction * float(distance)


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
        trials.append({'side': side, 'T': T, 'icp_rmse_m': rmse, 'icp_fitness': fitness, 'render': render})
    quality_key = lambda x: (x['icp_fitness'], x['render']['match_fraction'], -x['icp_rmse_m'])
    best = {side: max((x for x in trials if x['side']==side), key=quality_key) for side in ('positive','reverse')}
    side = max(best, key=lambda name: quality_key(best[name]))
    return side, best


def main():
    p = argparse.ArgumentParser(); p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--stl', type=Path, default=ROOT/'tool/welding_up_side_1118_only_2_small_screws_removed.stl')
    p.add_argument('--orientation-config', type=Path, default=MODULE_ROOT/'config/cad_orientation.json')
    a = p.parse_args(); d = a.capture_dir
    meta = json.loads((d/'metadata.json').read_text()); rgb = cv2.imread(str(d/'rgb_000.png')); depth = cv2.imread(str(d/'depth_aligned_000.png'), cv2.IMREAD_UNCHANGED)
    masks = np.load(d/'sam2_masks/masks.npz')['masks']; k = meta['color_intrinsics']; K = np.array([[k['fx'],0,k['ppx']],[0,k['fy'],k['ppy']],[0,0,1.]])
    tris = load_stl_triangles_m(a.stl); cad_icp = sample_surface_points(tris, 14000, 3); cad_render = sample_surface_points(tris, 90000, 4)
    # This is a fixed project calibration, not a command-line hyperparameter.
    orientation = json.loads(a.orientation_config.read_text())
    workspace = json.loads((MODULE_ROOT/'config/tabletop_workspace.json').read_text())
    rx, ry, rw, rh = workspace['safe_roi_xywh']; safe_roi = np.zeros(depth.shape, dtype=bool); safe_roi[ry:ry+rh,rx:rx+rw] = True
    # A per-object RANSAC can fit the workpiece itself and flip its normal.
    # Estimate this once from the complete scene, where the tabletop dominates.
    scene_points = mask_depth_to_points_cam(np.ones(depth.shape, dtype=bool), depth, K, float(meta['depth_scale_m']), .35, 1.2, 1)
    scene_normal = table_normal(scene_points)
    cad_up = normalize(orientation['positive_face_normal_stl']); feature = (cad_render @ cad_up >= np.percentile(cad_render @ cad_up, 88))
    vertices = tris.reshape(-1,3)
    raw_head, long_axis, width, _ = widest_head_in_stl(vertices)
    if orientation.get('grasp_point_stl_m') is not None:
        raw_head = np.asarray(orientation['grasp_point_stl_m'], dtype=float)
    if orientation.get('grasp_width_axis_stl') is not None:
        width = normalize(np.asarray(orientation['grasp_width_axis_stl'], dtype=float))
    if orientation.get('grasp_outward_axis_stl') is not None:
        head_outward_stl = normalize(np.asarray(orientation['grasp_outward_axis_stl'], dtype=float))
    else:
        head_outward_stl = wide_head_outward_direction(vertices, raw_head, long_axis)
    records, diagnostics = [], []
    orient = rgb.copy()
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
        diag = {'object_key': f'sam2-{i}', 'classification': accepted_side,
                'reverse_geometry_quality': {'icp_fitness': reverse_fitness, 'minimum_icp_fitness': orientation['minimum_reverse_icp_fitness'], 'depth_match_fraction': reverse_match, 'minimum_depth_match_fraction': orientation['minimum_reverse_depth_match_fraction'], 'accepted': geometry_ok},
                'positive': {k:v for k,v in best['positive'].items() if k!='T'}, 'reverse': {k:v for k,v in best['reverse'].items() if k!='T'}}
        diagnostics.append(diag)
        color = {'positive': (0,180,0), 'reverse': (0,0,220), 'unknown': (0,180,255), 'rejected_geometry': (120,120,120)}[accepted_side]
        tint = np.zeros_like(orient); tint[:]=color; orient[m>0] = (.55*orient[m>0]+.45*tint[m>0]).astype(np.uint8)
        ys, xs = np.nonzero(m); cv2.putText(orient, f"{accepted_side} f={best[side]['icp_fitness']:.2f} d={best[side]['render']['match_fraction']:.2f}", (int(xs.mean()),int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        if accepted_side != 'reverse': continue
        T = best[side]['T']; rotation, translation = T[:3,:3], T[:3,3]
        object_center = rotation @ vertices.mean(axis=0) + translation
        object_to_camera = normalize(-object_center)
        # The CAD direction runs from the wide-head centre to its outer end.
        # A positive dot means that end is physically closer to the camera;
        # place the jaws inside the head in that case.  Otherwise retain the
        # established 5-mm outward placement.
        head_towards_camera = bool(np.dot(rotation @ head_outward_stl, object_to_camera) > 0.0)
        inward_offset_m = float(orientation['wide_head_inward_grasp_offset_m'])
        outward_offset_m = float(orientation['wide_head_outward_grasp_offset_m'])
        head = wide_head_grasp_point(raw_head, head_outward_stl, outward_offset_m, inward_offset_m, head_towards_camera)
        center = rotation @ head + translation; R = make_grasp_frame(rotation@width, scene_normal)
        # Version7/GraspNet convention: +X is the approach axis.  Retreating
        # by -X is therefore a vertical, above-table pregrasp for this frame.
        records.append({'object_side':side, 'object_key':f'sam2-{i}', 'icp_fitness':best[side]['icp_fitness'], 'depth_match_fraction':best[side]['render']['match_fraction'], 'icp_rmse_m':best[side]['icp_rmse_m'], 'wide_head_outward_grasp_offset_m':outward_offset_m, 'wide_head_inward_grasp_offset_m':inward_offset_m, 'wide_head_towards_camera':head_towards_camera, 'wide_head_grasp_offset_direction':'inward' if head_towards_camera else 'outward', 'wide_head_grasp_offset_m':inward_offset_m if head_towards_camera else outward_offset_m, 'grasp_frame_convention':'version7_graspnet:+X_approach_down,+Y_jaw_opening,+Z_side', 'grasp_pose':pose_record(R,center), 'pregrasp_pose':pose_record(R,center-R[:,0]*.08)})
    # One workpiece can receive overlapping SAM2 prompts.  Associate duplicate
    # reverse masks by their CAD/TCP centres and retain the stronger face score.
    unique = []
    for record in records:
        center = np.asarray(record['grasp_pose']['translation_m'], dtype=float)
        same = next((j for j, old in enumerate(unique) if np.linalg.norm(center-np.asarray(old['grasp_pose']['translation_m'], dtype=float)) < .025), None)
        if same is None or (record['icp_fitness'], record['depth_match_fraction'], -record['icp_rmse_m']) > (unique[same]['icp_fitness'], unique[same]['depth_match_fraction'], -unique[same]['icp_rmse_m']):
            if same is None: unique.append(record)
            else: unique[same] = record
    records = unique
    out = d/'sam2_masks/grasp_overlay.png'; cv2.imwrite(str(out), draw_grasp_overlay(rgb, K, records, .04)); cv2.imwrite(str(d/'sam2_masks/orientation_overlay.png'), orient)
    (d/'sam2_masks/grasp_poses.json').write_text(json.dumps(records,indent=2)); (d/'sam2_masks/orientation_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps({'reverse':len(records), 'overlay':str(out), 'orientation_overlay':str(d/'sam2_masks/orientation_overlay.png')}))


if __name__ == '__main__': main()
