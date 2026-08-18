"""CAD render-to-depth scoring used by Module 4 pose/orientation estimation."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def depth_score(matrix, cad_render, feature, depth, mask, intrinsics, scale):
    points = (matrix[:3, :3] @ np.asarray(cad_render).T).T + matrix[:3, 3]
    valid = points[:, 2] > .1
    points, feature = points[valid], feature[valid]
    pixels = np.column_stack((
        intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2],
        intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2],
    ))
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < depth.shape[1]) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < depth.shape[0]))
    pixels, points, feature = pixels[inside], points[inside], feature[inside]
    tree = cKDTree(pixels)
    yy, xx = np.nonzero(mask & (depth > 0))
    observed_depth = depth[yy, xx].astype(float) * scale
    if len(observed_depth) > 14000:
        take = np.linspace(0, len(observed_depth) - 1, 14000).astype(int)
        yy, xx, observed_depth = yy[take], xx[take], observed_depth[take]
    distance, index = tree.query(np.column_stack((xx, yy)), k=1, distance_upper_bound=2.5)
    keep = np.isfinite(distance)
    if keep.sum() < 80:
        return float("inf"), {"match_fraction": float(keep.mean()), "depth_mae_m": None}
    residual = np.abs(points[index[keep], 2] - observed_depth[keep])
    weights = np.where(feature[index[keep]], 3.0, 1.0)
    mae = float(np.average(np.minimum(residual, .020), weights=weights))
    return mae + .004 * float(1 - keep.mean()), {
        "match_fraction": float(keep.mean()),
        "depth_mae_m": mae,
        "weighted_feature_fraction": float(np.mean(feature[index[keep]])),
    }
