#!/usr/bin/env python3
"""Module 1 v2 正面夹爪对称姿态软偏好的纯函数回归测试。"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import main as module1


def front_plan() -> dict:
    """构造开合轴与工件 X 正交的最小正面 Planner 计划。"""
    rotation = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()
    return {
        "face_side": "front",
        "eef_rotation_base": rotation.tolist(),
        "eef_axis_angle_rad": Rotation.from_matrix(rotation).as_rotvec().tolist(),
        "object_negative_x_axis_base": [1.0, 0.0, 0.0],
        "pregrasp_grasp_tcp_base_m": [0.1, 0.2, 0.3],
        "grasp_tcp_base_m": [0.1, 0.2, 0.2],
        "grasp_center_base_m": [0.1, 0.2, 0.2],
        "predicted_grasp_tcp_base_m": [0.1, 0.2, 0.2],
        "compensated_grasp_center_base_m": [0.1, 0.2, 0.2],
        "planner_direct_grasp_center": True,
    }


def reachable(candidate: dict, pregrasp: int, descent: int) -> tuple[dict, dict]:
    return candidate, {
        "pregrasp_waypoints": pregrasp,
        "lin_descent_waypoints": descent,
    }


class FrontCameraPreferenceTests(unittest.TestCase):
    def test_symmetry_preserves_primary_jaw_perpendicularity(self) -> None:
        variants = module1.parallel_jaw_variants(front_plan())
        self.assertEqual([roll for roll, _ in variants], [0.0, 180.0])
        diagnostics = [
            module1.front_transfer_camera_alignment(plan, np.array([0.0, 0.0, 1.0]))
            for _, plan in variants
        ]
        self.assertAlmostEqual(
            diagnostics[0]["jaw_opening_to_object_x_perpendicular_error_deg"], 0.0,
            places=7,
        )
        self.assertAlmostEqual(
            diagnostics[1]["jaw_opening_to_object_x_perpendicular_error_deg"], 0.0,
            places=7,
        )
        self.assertAlmostEqual(diagnostics[0]["alignment_dot"], 1.0, places=7)
        self.assertAlmostEqual(diagnostics[1]["alignment_dot"], -1.0, places=7)

    def test_front_selects_direction_preference_before_planner(self) -> None:
        selected, report = module1.preferred_front_parallel_jaw_variant(
            front_plan(), np.array([0.0, 0.0, 1.0])
        )
        symmetry_order, roll, plan = selected
        self.assertEqual(symmetry_order, 0)
        self.assertEqual(roll, 0.0)
        self.assertEqual(report["planner_preflight_variant_count"], 1)
        self.assertTrue(
            plan["front_transfer_camera_alignment"][
                "selected_before_planner_preflight"
            ]
        )

    def test_front_can_select_180_before_planner(self) -> None:
        selected, _ = module1.preferred_front_parallel_jaw_variant(
            front_plan(), np.array([0.0, 0.0, -1.0])
        )
        symmetry_order, roll, _ = selected
        self.assertEqual(symmetry_order, 1)
        self.assertEqual(roll, 180.0)

    def test_front_uses_only_reachable_symmetry(self) -> None:
        (_, roll, plan), _ = module1.preferred_front_parallel_jaw_variant(
            front_plan(), np.array([0.0, 0.0, -1.0])
        )
        candidate = {"rank": 2, "roll": roll, "plan": plan}
        chosen, _ = module1.choose_reachable_parallel_jaw_symmetry([
            reachable(candidate, pregrasp=10, descent=5),
        ])
        self.assertIs(chosen, candidate)

    def test_unreachable_top1_advances_to_top2_without_opposite_symmetry(self) -> None:
        candidates = []
        for semantic_rank in (1, 2):
            (symmetry_order, roll, plan), _ = (
                module1.preferred_front_parallel_jaw_variant(
                    front_plan(), np.array([0.0, 0.0, 1.0])
                )
            )
            candidates.append({
                "rank": (semantic_rank - 1) * 2 + symmetry_order + 1,
                "semantic_candidate_rank": semantic_rank,
                "source_grasp_index": semantic_rank,
                "grasp_source": "test",
                "score": 1.0 / semantic_rank,
                "parallel_jaw_symmetry_rotation_deg": roll,
                "plan": plan,
            })

        def fake_planner(_pa: object, candidate: dict):
            if candidate["semantic_candidate_rank"] == 1:
                raise module1.PairedTrajectoryUnavailable("Top1 unreachable")
            trajectory = {"trajectory": [{"joint": 0}, {"joint": 1}]}
            return trajectory, trajectory, 1, []

        with patch.object(
            module1, "paired_pregrasp_and_descent", side_effect=fake_planner
        ):
            reachable_candidates, records = module1.paired_preflight(
                None, candidates, max_inflight=2
            )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [item["semantic_candidate_rank"] for item in reachable_candidates], [2]
        )
        selected, policy = module1.select_reachable_candidate(
            reachable_candidates, failed_grasp_retry_index=0
        )
        self.assertEqual(selected["semantic_candidate_rank"], 2)
        self.assertEqual(policy, "highest_score_first_attempt")
        self.assertFalse(records[0]["reachable"])
        self.assertTrue(records[1]["reachable"])

    def test_back_keeps_original_shortest_path_policy(self) -> None:
        long_candidate = {"rank": 1, "plan": {"face_side": "back"}}
        short_candidate = {"rank": 2, "plan": {"face_side": "back"}}
        chosen, _ = module1.choose_reachable_parallel_jaw_symmetry([
            reachable(long_candidate, pregrasp=30, descent=10),
            reachable(short_candidate, pregrasp=8, descent=4),
        ])
        self.assertIs(chosen, short_candidate)

    def test_alignment_does_not_mutate_input_plan(self) -> None:
        plan = front_plan()
        before = copy.deepcopy(plan)
        module1.front_transfer_camera_alignment(plan, np.array([0.0, 0.0, 1.0]))
        self.assertEqual(plan, before)


if __name__ == "__main__":
    unittest.main()
