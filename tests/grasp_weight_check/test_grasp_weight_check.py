from __future__ import annotations

import unittest

from tests.grasp_weight_check import (
    ForceDataError,
    GraspClass,
    WeightThresholds,
    classify_grasp_weight,
    collect_force_window,
    summarize_force,
    verify_grasp_after_lift,
)


G = 9.80665
THRESHOLDS = WeightThresholds(
    empty_max_g=30.0,
    single_min_g=80.0,
    single_max_g=140.0,
    multiple_min_g=180.0,
    max_axis_mad_n=0.10,
)


def window_for_mass(mass_g: float):
    baseline = summarize_force([[1.0, -2.0, 3.0]] * 5)
    delta_n = mass_g / 1000.0 * G
    loaded = summarize_force([[1.0, -2.0, 3.0 + delta_n]] * 5)
    return baseline, loaded


class ScriptedSensor:
    def __init__(self, frames):
        self.frames = iter(frames)

    def get_state(self):
        return next(self.frames, None)


class GraspWeightClassificationTests(unittest.TestCase):
    def test_empty(self):
        result = classify_grasp_weight(*window_for_mass(15.0), THRESHOLDS)
        self.assertEqual(result.grasp_class, GraspClass.EMPTY)

    def test_single(self):
        result = classify_grasp_weight(*window_for_mass(110.0), THRESHOLDS)
        self.assertEqual(result.grasp_class, GraspClass.SINGLE)
        self.assertAlmostEqual(result.mass_g, 110.0, places=6)

    def test_multiple(self):
        result = classify_grasp_weight(*window_for_mass(220.0), THRESHOLDS)
        self.assertEqual(result.grasp_class, GraspClass.MULTIPLE)

    def test_threshold_gap_is_unknown(self):
        result = classify_grasp_weight(*window_for_mass(160.0), THRESHOLDS)
        self.assertEqual(result.grasp_class, GraspClass.UNKNOWN)

    def test_vector_norm_is_independent_of_sensor_axis(self):
        baseline = summarize_force([[0.0, 0.0, 0.0]] * 5)
        # 100 g 对应的增量分布在 x/y 两轴，模长仍为 0.980665 N。
        loaded = summarize_force([[0.588399, 0.784532, 0.0]] * 5)
        result = classify_grasp_weight(baseline, loaded, THRESHOLDS)
        self.assertEqual(result.grasp_class, GraspClass.SINGLE)
        self.assertAlmostEqual(result.mass_g, 100.0, places=3)

    def test_unstable_window_is_unknown(self):
        baseline = summarize_force([[0.0, 0.0, 0.0]] * 5)
        loaded = summarize_force(
            [[0.0, 0.0, 0.8], [0.0, 0.0, 1.2]] * 3
        )
        result = classify_grasp_weight(baseline, loaded, THRESHOLDS)
        self.assertEqual(result.grasp_class, GraspClass.UNKNOWN)
        self.assertIn("采样不稳定", result.reason)

    def test_median_rejects_one_force_spike(self):
        window = summarize_force(
            [[1.0, 2.0, 3.0]] * 4 + [[100.0, -100.0, 80.0]]
        )
        self.assertEqual(window.median_force_n, (1.0, 2.0, 3.0))


class ForceCollectionTests(unittest.TestCase):
    def test_duplicate_timestamp_and_invalid_frame_are_not_counted(self):
        sensor = ScriptedSensor(
            [
                {"timestamp": 1, "filtered_force": [1, 2, 3]},
                {"timestamp": 1, "filtered_force": [9, 9, 9]},
                {"timestamp": 2, "filtered_force": []},
                {"timestamp": 3, "filtered_force": [2, 3, 4]},
                {"timestamp": 4, "filtered_force": [3, 4, 5]},
            ]
        )
        window = collect_force_window(
            sensor, sample_count=3, timeout_s=0.2, poll_interval_s=0
        )
        self.assertEqual(window.sample_count, 3)
        self.assertEqual(window.median_force_n, (2.0, 3.0, 4.0))

    def test_timeout_without_frames(self):
        sensor = ScriptedSensor([])
        with self.assertRaises(ForceDataError):
            collect_force_window(
                sensor, sample_count=3, timeout_s=0.01, poll_interval_s=0.001
            )

    def test_rejects_overlapping_thresholds(self):
        with self.assertRaises(ValueError):
            WeightThresholds(30, 20, 100, 150)

    def test_verify_after_lift_collects_and_classifies(self):
        baseline = summarize_force([[1.0, -2.0, 3.0]] * 5)
        delta_n = 0.110 * G
        sensor = ScriptedSensor(
            [
                {
                    "timestamp": index,
                    "filtered_force": [1.0, -2.0, 3.0 + delta_n],
                }
                for index in range(3)
            ]
        )
        result = verify_grasp_after_lift(
            sensor,
            baseline,
            THRESHOLDS,
            settle_s=0,
            sample_count=3,
            timeout_s=0.2,
            poll_interval_s=0,
            sleep=lambda _: None,
        )
        self.assertEqual(result.grasp_class, GraspClass.SINGLE)


if __name__ == "__main__":
    unittest.main()
