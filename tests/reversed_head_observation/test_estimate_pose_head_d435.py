import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.reversed_head_observation.estimate_pose_head_d435 import (
    CAMERA_FRAME,
    choose_output_transform,
    pose_error_deg,
)
from tests.reversed_head_observation.derive_head_alignment import (
    nominal_body_head_cam,
    propagate_extrinsic,
)
from tool.inspection_support.sam2_icp import (
    _projected_mask_overlap,
)
from tool.inspection_support.robot_camera import neutral_roi_balance_bgr
from tool.inspection_support.mask_depth import valid_median_depth


class HeadD435PoseTests(unittest.TestCase):
    def test_projected_mask_overlap_rewards_projected_cad_inside_mask(self):
        cad = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255
        intrinsics = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(
            _projected_mask_overlap(cad, np.eye(4), mask, intrinsics), 0.5
        )

    def test_head_kinematic_propagation_is_identity_at_same_pose(self):
        reference = np.eye(4)
        reference[:3, 3] = [0.1, 0.2, 0.3]
        actual = propagate_extrinsic(reference, 38.5, 0.0, 38.5, 0.0)
        np.testing.assert_allclose(actual, reference, atol=1e-10)

    def test_nominal_head_cam_changes_translation_when_gimbal_rotates(self):
        reference = nominal_body_head_cam(38.5, 0.0)
        target = nominal_body_head_cam(2.0, 19.0)
        self.assertGreater(float(np.linalg.norm(target[:3, 3] - reference[:3, 3])), 0.01)

    def test_nominal_zero_pose_matches_deployed_tf(self):
        actual = nominal_body_head_cam(0.0, 0.0)
        np.testing.assert_allclose(
            actual[:3, 3], [0.0829899729, 0.0038827509, 0.227498], atol=1e-8
        )

    def test_valid_median_does_not_let_zero_frames_erase_depth(self):
        frames = [
            np.array([[0, 900]], dtype=np.uint16),
            np.array([[260, 900]], dtype=np.uint16),
            np.array([[0, 900]], dtype=np.uint16),
            np.array([[262, 900]], dtype=np.uint16),
            np.array([[0, 900]], dtype=np.uint16),
        ]
        depth, support = valid_median_depth(frames, 0.001, 0.05, 0.8)
        self.assertEqual(int(depth[0, 0]), 261)
        self.assertEqual(int(support[0, 0]), 2)
        self.assertEqual(int(depth[0, 1]), 0)

    def test_neutral_roi_balance_reduces_green_cast(self):
        image = np.full((20, 20, 3), [100, 120, 50], dtype=np.uint8)
        corrected, gains = neutral_roi_balance_bgr(image, (0, 0, 20, 20))
        self.assertGreater(gains[2], 2.0)
        self.assertLessEqual(int(corrected[..., 1].mean()) - int(corrected[..., 2].mean()), 1)

    def test_pose_error_uses_largest_axis_error(self):
        self.assertEqual(
            pose_error_deg({"pitch": 2.2, "yaw": 18.0}, {"pitch": 2.0, "yaw": 19.0}),
            1.0,
        )

    def test_camera_output_does_not_use_fixed_calibration(self):
        pose = np.eye(4)
        output, frame, info = choose_output_transform(
            "camera", pose, {"pitch": 2.0, "yaw": 19.0}, Path("/missing"), 1.0
        )
        np.testing.assert_array_equal(output, pose)
        self.assertEqual(frame, CAMERA_FRAME)
        self.assertEqual(info["status"], "camera_frame_only")

    def test_body_output_rejects_wrong_head_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            path.write_text(json.dumps({
                "T_body_base_head_camera": np.eye(4).tolist(),
                "head_pose_deg": {"pitch": 38.5, "yaw": 0.0},
            }))
            with self.assertRaisesRegex(RuntimeError, "只在.*38.500"):
                choose_output_transform(
                    "body_base", np.eye(4), {"pitch": 2.0, "yaw": 19.0}, path, 1.0
                )

    def test_body_output_uses_matching_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            T_body_camera = np.eye(4)
            T_body_camera[0, 3] = 0.25
            path.write_text(json.dumps({
                "T_body_base_head_camera": T_body_camera.tolist(),
                "head_pose_deg": {"pitch": 38.5, "yaw": 0.0},
            }))
            T_camera_object = np.eye(4)
            T_camera_object[2, 3] = 0.4
            output, frame, info = choose_output_transform(
                "body_base", T_camera_object,
                {"pitch": 38.6, "yaw": 0.1}, path, 0.5,
            )
            np.testing.assert_allclose(output, T_body_camera @ T_camera_object)
            self.assertEqual(frame, "body_base")
            self.assertEqual(info["status"], "calibrated_fixed_head_pose")


if __name__ == "__main__":
    unittest.main()
