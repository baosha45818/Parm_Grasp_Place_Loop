"""定义模块 3-1、3-2、3-3 共用的左臂放置策略数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlaceBranchProfile:
    module_id: str
    classification: str
    placement_branch: str
    target_argument: str
    default_target_xyz_m: tuple[float, float, float]
    target_z_candidates_m: tuple[float, ...]
    release_z_candidates_m: tuple[float, ...]
