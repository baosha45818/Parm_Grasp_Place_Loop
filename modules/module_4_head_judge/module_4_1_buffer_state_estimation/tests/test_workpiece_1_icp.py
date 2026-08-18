"""Regression tests for workpiece-1 point-to-plane registration and masks."""

import unittest

import cv2
import numpy as np

from modules.module_4_head_judge.module_4_1_buffer_state_estimation.vision.mask import (
    _interior_prompt_region,
    _largest_connected_component,
)
from tools.icp import tabletop_constrained_point_to_plane_icp


class Workpiece1PointToPlaneTests(unittest.TestCase):
    def test_constrained_point_to_plane_recovers_yaw_and_translation(self):
        rng = np.random.default_rng(7)
        count = 2500
        source = rng.uniform(
            [-0.045, -0.025, -0.006], [0.045, 0.025, 0.006], (count, 3)
        )
        face = rng.integers(0, 6, count)
        dimensions = np.array([0.045, 0.025, 0.006])
        axis = face // 2
        source[np.arange(count), axis] = dimensions[axis] * np.where(
            face % 2, 1.0, -1.0
        )
        expected_yaw = np.deg2rad(3.5)
        rotation = np.array(
            [
                [np.cos(expected_yaw), -np.sin(expected_yaw), 0.0],
                [np.sin(expected_yaw), np.cos(expected_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_translation = np.array([0.0025, -0.0015, 0.61])
        target = (rotation @ source.T).T + expected_translation
        target += rng.normal(0.0, 0.00015, target.shape)
        initial = np.eye(4)
        initial[:3, 3] = target.mean(axis=0) - source.mean(axis=0)

        transform, rmse, fitness = tabletop_constrained_point_to_plane_icp(
            source, target, initial, [0.0, 0.0, 1.0], max_corr_m=0.012
        )

        actual_yaw = np.arctan2(transform[1, 0], transform[0, 0])
        self.assertAlmostEqual(actual_yaw, expected_yaw, delta=np.deg2rad(0.15))
        np.testing.assert_allclose(
            transform[:3, 3], expected_translation, atol=0.00025
        )
        self.assertLess(rmse, 0.0005)
        self.assertGreater(fitness, 0.98)
        np.testing.assert_allclose(transform[:3, 2], [0.0, 0.0, 1.0], atol=1e-10)

    def test_final_mask_keeps_only_largest_connected_component(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[2:12, 3:15] = True
        mask[20:23, 20:24] = True
        selected = _largest_connected_component(mask)
        self.assertEqual(int(selected.sum()), 120)
        self.assertFalse(selected[21, 21])

    def test_positive_prompt_region_rejects_thin_attached_branch(self):
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:31, 8:40] = True
        mask[3:10, 23:25] = True
        interior = _interior_prompt_region(mask, 4.0)
        self.assertTrue(interior[20, 20])
        self.assertFalse(np.any(interior[3:10, 23:25]))
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        self.assertGreaterEqual(float(distance[interior].min()), 4.0)


if __name__ == "__main__":
    unittest.main()
