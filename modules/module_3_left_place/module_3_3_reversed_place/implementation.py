"""模块 3-3：定义反面工件放置的翻面目标和松爪高度。"""

from modules.module_3_left_place.branch_profile import PlaceBranchProfile
from modules.module_3_left_place.placement_config import load_workpiece_parameters


WORKPIECE_1 = load_workpiece_parameters("1")


PROFILE = PlaceBranchProfile(
    module_id="3-3",
    classification="REVERSED",
    placement_branch="BRANCH_1_REVERSED",
    target_argument="reversed_target_object_xyz",
    default_target_xyz_m=tuple(WORKPIECE_1.REVERSED_TARGET_OBJECT_XYZ_M),
    target_z_candidates_m=tuple(WORKPIECE_1.REVERSED_TARGET_Z_CANDIDATES_M),
    release_z_candidates_m=tuple(WORKPIECE_1.REVERSED_RELEASE_Z_CANDIDATES_M),
)
