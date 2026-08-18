"""Unit tests for the flattened Module 4-1 Buffer face release policy."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parent.parent / "check_buffer_face.py"
SPEC = importlib.util.spec_from_file_location("check_buffer_face", MODULE_PATH)
checker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checker)


class ClassifyTests(unittest.TestCase):
    gate = {
        "minimum_positive_icp_fitness": 0.75,
        "minimum_positive_depth_match_fraction": 0.70,
    }

    @classmethod
    def positive_diagnostic(cls, **overrides):
        diagnostic = {
            "classification": "positive",
            "score_margin_m": 0.0005,
            "positive": {
                "icp_fitness": 0.80,
                "render": {"match_fraction": 0.80},
            },
        }
        diagnostic.update(overrides)
        return diagnostic

    @staticmethod
    def module4_result(orientation, diagnostic, *, count=1):
        item = SimpleNamespace(
            orientation=orientation,
            diagnostics=diagnostic,
            reason="test result",
        )
        objects = tuple(item for _ in range(count))
        return SimpleNamespace(objects=objects)

    def test_positive_is_the_only_pass(self):
        decision, _ = checker._classify(
            self.module4_result(checker.ObjectOrientation.FRONT, self.positive_diagnostic()),
            self.gate,
        )
        self.assertEqual(decision, "FRONT")

    def test_reverse_stops_the_flow(self):
        decision, _ = checker._classify(
            self.module4_result(checker.ObjectOrientation.BACK, {"classification": "reverse"}),
            self.gate,
        )
        self.assertEqual(decision, "BACK")

    def test_workpiece_2_releases_face_and_pose_from_same_result(self):
        for orientation, expected in (
            (checker.ObjectOrientation.FRONT, "FRONT"),
            (checker.ObjectOrientation.BACK, "BACK"),
        ):
            diagnostic = {
                "pose": {"valid": True, "orientation_valid": True},
                "pose_confidence": {"low_reasons": []},
            }
            decision, reason = checker._classify(
                self.module4_result(orientation, diagnostic),
                self.gate,
                workpiece_id="2",
            )
            self.assertEqual(decision, expected)
            self.assertIn("位姿已同时发布", reason)

    def test_workpiece_2_weak_pose_is_uncertain_and_recapturable(self):
        diagnostic = {
            "classification": "unknown",
            "pose": {"valid": False, "orientation_valid": False},
            "pose_confidence": {"low_reasons": ["high_depth_score"]},
        }
        result = self.module4_result(checker.ObjectOrientation.OTHER, diagnostic)
        self.assertEqual(
            checker._classify(result, self.gate, workpiece_id="2")[0],
            "UNCERTAIN",
        )
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            result, self.gate, workpiece_id="2"
        ))

    def test_ambiguous_and_multiple_objects_stop_the_flow(self):
        self.assertEqual(
            checker._classify(
                self.module4_result(checker.ObjectOrientation.OTHER, {"classification": "unknown"}), self.gate
            )[0],
            "UNCERTAIN",
        )
        self.assertEqual(checker._classify(SimpleNamespace(objects=()), self.gate)[0], "UNCERTAIN")
        self.assertEqual(
            checker._classify(
                self.module4_result(checker.ObjectOrientation.FRONT, self.positive_diagnostic(), count=2), self.gate
            )[0],
            "UNCERTAIN",
        )

    def test_positive_with_bad_or_missing_quality_stops_the_flow(self):
        low_fitness = self.positive_diagnostic(
            positive={"icp_fitness": 0.74, "render": {"match_fraction": 0.80}}
        )
        low_match = self.positive_diagnostic(
            positive={"icp_fitness": 0.80, "render": {"match_fraction": 0.69}}
        )
        for diagnostic in (low_fitness, low_match, {"classification": "positive"}):
            self.assertEqual(
                checker._classify(
                    self.module4_result(checker.ObjectOrientation.FRONT, diagnostic), self.gate
                )[0],
                "UNCERTAIN",
            )

    def test_positive_failure_with_good_reverse_falls_back_to_back(self):
        diagnostic = self.positive_diagnostic(
            score_margin_m=0.000351,
            positive={"icp_fitness": 0.619, "render": {"match_fraction": 0.558}},
            reverse={"icp_fitness": 0.828, "render": {"match_fraction": 0.975}},
        )
        decision, reason = checker._classify(
            self.module4_result(checker.ObjectOrientation.FRONT, diagnostic), self.gate
        )
        self.assertEqual(decision, "BACK")
        self.assertIn("按安全反抓回退处理", reason)

    def test_ambiguous_score_with_good_reverse_falls_back_to_back(self):
        diagnostic = {
            "classification": "unknown",
            "positive": {"icp_fitness": 0.619, "render": {"match_fraction": 0.558}},
            "reverse": {"icp_fitness": 0.828, "render": {"match_fraction": 0.975}},
        }
        decision, _ = checker._classify(
            self.module4_result(checker.ObjectOrientation.OTHER, diagnostic), self.gate
        )
        self.assertEqual(decision, "BACK")

    def test_reverse_fallback_is_not_used_when_reverse_is_weak(self):
        base = self.positive_diagnostic(
            score_margin_m=0.000351,
            positive={"icp_fitness": 0.619, "render": {"match_fraction": 0.558}},
            reverse={"icp_fitness": 0.828, "render": {"match_fraction": 0.975}},
        )
        for diagnostic in ({**base, "reverse": {"icp_fitness": 0.70, "render": {"match_fraction": 0.975}}},):
            self.assertEqual(
                checker._classify(
                    self.module4_result(checker.ObjectOrientation.FRONT, diagnostic), self.gate
                )[0],
                "UNCERTAIN",
            )

    def test_both_face_quality_failures_are_recapturable(self):
        both_failed = {
            "classification": "rejected_geometry",
            "score_margin_m": 0.0005,
            "positive": {"icp_fitness": 0.60, "render": {"match_fraction": 0.60}},
            "reverse_geometry_quality": {"accepted": False},
        }
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            self.module4_result(checker.ObjectOrientation.OTHER, both_failed), self.gate
        ))
        self.assertFalse(checker._all_objects_both_face_quality_rejected(
            self.module4_result(checker.ObjectOrientation.OTHER, {
                **both_failed, "reverse_geometry_quality": {"accepted": True},
            }), self.gate
        ))
        # Module 4 emits ``unknown`` when the face-score margin is below its
        # decision threshold.  If both hypotheses are weak this must request a
        # fresh capture, not return exit 21 immediately.
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            self.module4_result(checker.ObjectOrientation.OTHER, {
                **both_failed, "classification": "unknown",
            }), self.gate
        ))
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            self.module4_result(checker.ObjectOrientation.OTHER, {
                **both_failed, "classification": "unknown",
                "reverse_geometry_quality": {"accepted": True},
            }), self.gate
        ))

    def test_multiple_rejected_objects_are_recapturable(self):
        rejected = {
            "classification": "rejected_geometry",
            "score_margin_m": 0.0005,
            "positive": {"icp_fitness": 0.60, "render": {"match_fraction": 0.60}},
            "reverse_geometry_quality": {"accepted": False},
        }
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            self.module4_result(checker.ObjectOrientation.OTHER, rejected, count=2), self.gate
        ))

    def test_zero_or_multiple_objects_are_recapturable_regardless_of_face_class(self):
        reliable_positive = self.positive_diagnostic()
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            SimpleNamespace(objects=()), self.gate
        ))
        self.assertTrue(checker._all_objects_both_face_quality_rejected(
            self.module4_result(checker.ObjectOrientation.FRONT, reliable_positive, count=2), self.gate
        ))
