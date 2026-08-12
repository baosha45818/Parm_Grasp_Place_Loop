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

ROOT = Path(__file__).resolve().parents[4]
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


def transform(T, p):
    return (T[:3, :3] @ np.asarray(p).T).T + T[:3, 3]


def project_to_pixel(K, point_cam):
    """Project one camera-frame point to floating-point image coordinates."""
    point = np.asarray(point_cam, dtype=float)
    if point.shape != (3,) or point[2] <= 1e-12:
        raise ValueError(f'cannot project invalid camera point: {point!r}')
    return np.array([
        K[0, 0] * point[0] / point[2] + K[0, 2],
        K[1, 1] * point[1] / point[2] + K[1, 2],
    ])

def planar_basis(points, normal):
    q = np.asarray(points) - np.mean(points, axis=0)
    q -= (q @ normal)[:, None] * normal
    _, vecs = np.linalg.eigh(np.cov(q.T))
    x = vecs[:, -1] - normal * np.dot(vecs[:, -1], normal)
    x = normalize(x)
    return x, normalize(np.cross(normal, x))


def constrained_icp(source, target, initial, normal, max_iters=45, max_corr=.009):
    """ICP whose incremental rotation is yaw-only around the detected table."""
    tree = cKDTree(target); T = initial.copy(); normal = normalize(normal)
    ex, ey = planar_basis(target, normal)
    for _ in range(max_iters):
        sw = transform(T, source); dist, near = tree.query(sw, k=1, workers=-1)
        keep = dist < max_corr
        if keep.sum() < 30:
            keep[np.argsort(dist)[:min(250, len(dist))]] = True
        cutoff = np.quantile(dist[keep], .82); keep &= dist <= cutoff
        a, b = sw[keep], target[near[keep]]
        ac, bc = a.mean(0), b.mean(0)
        A = np.column_stack(((a-ac)@ex, (a-ac)@ey)); B = np.column_stack(((b-bc)@ex, (b-bc)@ey))
        H = A.T @ B
        theta = np.arctan2(H[0, 1]-H[1, 0], H[0, 0]+H[1, 1])
        Q = axis_angle(normal, theta); t = bc - Q @ ac
        dT = np.eye(4); dT[:3, :3] = Q; dT[:3, 3] = t
        T = dT @ T
        if abs(theta) < 2e-5 and np.linalg.norm(t) < 2e-6:
            break
    final = transform(T, source); dist, _ = tree.query(final, k=1, workers=-1); inlier = dist < max_corr
    return T, float(np.sqrt(np.mean(dist[inlier]**2))) if inlier.any() else float('inf'), float(inlier.mean())


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


def classify(cad_icp, cad_render, feature, observed, depth, mask, K, scale, cad_up, min_margin_m, normal):
    trials = []
    for side, initial in face_initials(cad_icp, observed, normal, cad_up):
        T, rmse, fitness = constrained_icp(cad_icp, observed, initial, normal)
        score, render = depth_score(T, cad_render, feature, depth, mask, K, scale)
        trials.append({'side': side, 'T': T, 'icp_rmse_m': rmse, 'icp_fitness': fitness, 'render_score_m': score, 'render': render})
    best = {side: min((x for x in trials if x['side']==side), key=lambda x:x['render_score_m']) for side in ('positive','reverse')}
    up, down = best['positive'], best['reverse']; margin = abs(up['render_score_m']-down['render_score_m'])
    # A fixed, labelled-scene-calibrated physical margin is preferable to a
    # percentage of the score: the latter incorrectly rejected the known
    # reverse-side part when both hypotheses had a high common depth offset.
    threshold = float(min_margin_m)
    side = 'unknown' if not np.isfinite(margin) or margin < threshold else min(best, key=lambda s: best[s]['render_score_m'])
    return side, best, margin, threshold


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
        side, best, margin, threshold = classify(cad_icp, cad_render, feature, pts, depth, m.astype(bool), K, float(meta['depth_scale_m']), cad_up, orientation['minimum_score_margin_m'], scene_normal)
        reverse_quality = best['reverse']
        reverse_match = float(reverse_quality['render']['match_fraction'])
        reverse_fitness = float(reverse_quality['icp_fitness'])
        geometry_ok = (
            reverse_fitness >= float(orientation['minimum_reverse_icp_fitness'])
            and reverse_match >= float(orientation['minimum_reverse_depth_match_fraction'])
        )
        accepted_side = side if side != 'reverse' or geometry_ok else 'rejected_geometry'
        # Keep the winning ICP pose in the public diagnostic contract.  The
        # Buffer recovery path consumes this exact estimate to re-grasp a part
        # which has been corrected to its front side; recomputing a centre from
        # the mask would discard the CAD/ICP registration we just validated.
        diag = {'object_key': f'sam2-{i}', 'classification': accepted_side, 'score_margin_m': margin, 'required_margin_m': threshold,
                'table_normal_camera': scene_normal.round(9).tolist(),
                'reverse_geometry_quality': {'icp_fitness': reverse_fitness, 'minimum_icp_fitness': orientation['minimum_reverse_icp_fitness'], 'depth_match_fraction': reverse_match, 'minimum_depth_match_fraction': orientation['minimum_reverse_depth_match_fraction'], 'accepted': geometry_ok},
                'positive': {**{k:v for k,v in best['positive'].items() if k!='T'}, 'T_camera_object': best['positive']['T'].round(9).tolist()},
                'reverse': {**{k:v for k,v in best['reverse'].items() if k!='T'}, 'T_camera_object': best['reverse']['T'].round(9).tolist()}}
        diagnostics.append(diag)
        color = {'positive': (0,180,0), 'reverse': (0,0,220), 'unknown': (0,180,255), 'rejected_geometry': (120,120,120)}[accepted_side]
        tint = np.zeros_like(orient); tint[:]=color; orient[m>0] = (.55*orient[m>0]+.45*tint[m>0]).astype(np.uint8)
        ys, xs = np.nonzero(m); cv2.putText(orient, f'{accepted_side} d={margin*1000:.1f}mm', (int(xs.mean()),int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        if accepted_side != 'reverse': continue
        T = best[side]['T']; rotation, translation = T[:3,:3], T[:3,3]
        object_center = rotation @ vertices.mean(axis=0) + translation
        object_to_camera = normalize(-object_center)
        # The CAD direction runs from the wide-head centre to its outer end.
        # A positive dot means that end is physically closer to the camera;
        # use the configured inward displacement (zero means the head centre).
        # The opposite direction uses its independently configured displacement.
        head_towards_camera = bool(np.dot(rotation @ head_outward_stl, object_to_camera) > 0.0)
        inward_offset_m = float(orientation['wide_head_inward_grasp_offset_m'])
        outward_offset_m = float(orientation['wide_head_outward_grasp_offset_m'])
        head_center_camera = rotation @ raw_head + translation
        wide_head_image_left = bool(
            project_to_pixel(K, head_center_camera)[0]
            < project_to_pixel(K, object_center)[0]
        )
        lateral_offset_m = float(
            orientation['wide_head_image_left_grasp_lateral_offset_m']
            if wide_head_image_left
            else orientation['wide_head_image_right_grasp_lateral_offset_m']
        )
        head = wide_head_grasp_point(raw_head, head_outward_stl, outward_offset_m, inward_offset_m, head_towards_camera)
        R = make_grasp_frame(rotation@width, scene_normal)
        # Apply the tuned side bias in the grasp/workpiece frame, never in the
        # camera frame.  Grasp +Y is the jaw-opening (workpiece-width) axis, so
        # the chosen physical side follows the part when its yaw changes.
        center = rotation @ head + translation + R[:, 1] * lateral_offset_m
        # Version7/GraspNet convention: +X is the approach axis.  Retreating
        # by -X is therefore a vertical, above-table pregrasp for this frame.
        records.append({'object_side':side, 'object_key':f'sam2-{i}', 'orientation_score_margin_m':margin, 'wide_head_outward_grasp_offset_m':outward_offset_m, 'wide_head_inward_grasp_offset_m':inward_offset_m, 'grasp_lateral_offset_m':lateral_offset_m, 'grasp_lateral_offset_axis':'grasp_frame_+Y', 'wide_head_image_direction':'left' if wide_head_image_left else 'right', 'wide_head_towards_camera':head_towards_camera, 'wide_head_grasp_offset_direction':'inward' if head_towards_camera else 'outward', 'wide_head_grasp_offset_m':inward_offset_m if head_towards_camera else outward_offset_m, 'grasp_frame_convention':'version7_graspnet:+X_approach_down,+Y_jaw_opening,+Z_side', 'grasp_pose':pose_record(R,center), 'pregrasp_pose':pose_record(R,center-R[:,0]*.08)})
    # One workpiece can receive overlapping SAM2 prompts.  Associate duplicate
    # reverse masks by their CAD/TCP centres and retain the stronger face score.
    unique = []
    for record in records:
        center = np.asarray(record['grasp_pose']['translation_m'], dtype=float)
        same = next((j for j, old in enumerate(unique) if np.linalg.norm(center-np.asarray(old['grasp_pose']['translation_m'], dtype=float)) < .025), None)
        if same is None or record['orientation_score_margin_m'] > unique[same]['orientation_score_margin_m']:
            if same is None: unique.append(record)
            else: unique[same] = record
    records = unique
    out = d/'sam2_masks/grasp_overlay.png'; cv2.imwrite(str(out), draw_grasp_overlay(rgb, K, records, .04)); cv2.imwrite(str(d/'sam2_masks/orientation_overlay.png'), orient)
    (d/'sam2_masks/grasp_poses.json').write_text(json.dumps(records,indent=2)); (d/'sam2_masks/orientation_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps({'reverse':len(records), 'overlay':str(out), 'orientation_overlay':str(d/'sam2_masks/orientation_overlay.png')}))


if __name__ == '__main__': main()
