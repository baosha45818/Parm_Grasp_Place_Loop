"""Pure policy tests for the workpiece-2 orientation/pose adapter."""

import unittest

import numpy as np

from modules.module_4_head_judge.module_4_1_buffer_state_estimation.vision import (
    workpiece_2_icp as state,
)


class Workpiece2StateTests(unittest.TestCase):
    def test_duplicate_sam_masks_collapse_to_one_object(self):
        first = np.zeros((20, 20), dtype=np.uint8)
        first[3:15, 4:16] = 1
        duplicate = first.copy()
        duplicate[3, 4] = 0
        distinct = np.zeros_like(first)
        distinct[16:19, 16:19] = 1
        result = state._deduplicate_masks(np.asarray([first, duplicate, distinct]))
        self.assertEqual(len(result), 2)

    def test_strong_single_face_can_supply_orientation(self):
        record = {
            "orientation": "OTHER",
            "object_side": "ambiguous_face_full_6dof",
            "face_hypotheses": {
                "BACK": {"fitness": 0.95, "rmse_mm": 4.0, "depth_score_mm": 5.5},
            },
            "pose_confidence": {
                "level": "LOW",
                "recommended_for_grasp": False,
                "low_reasons": ["front_back_orientation_ambiguous"],
                "checks": {"orientation": "OTHER"},
            },
        }
        config = {
            "single_face_minimum_fitness": 0.93,
            "single_face_maximum_rmse_mm": 4.8,
            "single_face_maximum_depth_score_mm": 6.25,
        }
        state._promote_single_face(record, config)
        self.assertEqual(record["orientation"], "BACK")
        self.assertTrue(record["pose_confidence"]["recommended_for_grasp"])

    def test_weak_single_face_remains_ambiguous(self):
        record = {
            "orientation": "OTHER",
            "face_hypotheses": {
                "FRONT": {"fitness": 0.80, "rmse_mm": 4.0, "depth_score_mm": 5.5},
            },
            "pose_confidence": {"low_reasons": [], "checks": {}},
        }
        config = {
            "single_face_minimum_fitness": 0.93,
            "single_face_maximum_rmse_mm": 4.8,
            "single_face_maximum_depth_score_mm": 6.25,
        }
        state._promote_single_face(record, config)
        self.assertEqual(record["orientation"], "OTHER")


if __name__ == "__main__":
    unittest.main()
