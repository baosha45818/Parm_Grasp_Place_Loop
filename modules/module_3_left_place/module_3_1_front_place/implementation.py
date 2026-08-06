"""模块 3-1：定义正面工件放置的目标、候选高度和松爪高度。"""

from modules.module_3_left_place.branch_profile import PlaceBranchProfile
from modules.module_3_left_place.placement_config import load_workpiece_parameters


WORKPIECE_1 = load_workpiece_parameters("1")


PROFILE = PlaceBranchProfile(
    module_id="3-1",
    classification="FRONT",
    placement_branch="BRANCH_1_FRONT",
    target_argument="front_target_object_xyz",
    default_target_xyz_m=tuple(WORKPIECE_1.FRONT_TARGET_OBJECT_XYZ_M),
    target_z_candidates_m=tuple(WORKPIECE_1.FRONT_TARGET_Z_CANDIDATES_M),
    release_z_candidates_m=tuple(WORKPIECE_1.FRONT_RELEASE_Z_CANDIDATES_M),
)
