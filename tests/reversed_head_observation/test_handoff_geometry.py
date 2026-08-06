import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from loop_config import LoopConfig
from loop_types import ObjectPose
from tests.reversed_head_observation.handoff_attachment import (
    attachment_request_applied,
    write_attachment_request,
)
from tests.reversed_head_observation.configure_handoff import (
    config_with_handoff_target,
    config_with_right_relation,
)
from tests.reversed_head_observation.execute_handoff import require_pose_close
from tests.reversed_head_observation.execute_taught_tail_handoff import (
    parse_args as parse_handoff_args,
    pregrasp_offset_body,
    taught_handoff_object_transform,
)
from tests.reversed_head_observation.move_object_center import (
    captured_left_tcp,
    centre_target,
    load_result,
    result_target_center,
    redundant_handoff_candidates,
    handoff_axis_errors,
)
from tests.reversed_head_observation.handoff_geometry import (
    compute_right_tail_grasp_targets,
    compute_handoff_targets,
    planner_tf_target_source,
    transform_error,
    validate_transform,
)
from tests.reversed_head_observation.preflight_handoff import (
    load_icp_object_transform,
    trajectory_summary,
    validate_icp_for_motion,
)
from tool.planner_runtime import PlannerRuntime


def translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


class HandoffGeometryTests(unittest.TestCase):
    def test_taught_object_tilt_defaults_to_fixed_handoff_orientation(self):
        with patch.object(
            sys,
            "argv",
            ["execute_taught_tail_handoff", "--result", "result.json"],
        ):
            args = parse_handoff_args()
        self.assertEqual(args.taught_object_tilt_deg, 0.0)
        self.assertEqual(args.arrival_translation_tolerance_mm, 1.0)
        self.assertEqual(args.final_grasp_motion, "lin")
        self.assertEqual(args.right_grasp_tolerance_deg, 0.5)

    def test_zero_tilt_taught_object_matches_fixed_handoff_orientation(self):
        taught = taught_handoff_object_transform([0.42, 0.10, 0.25], 0.0)
        np.testing.assert_allclose(
            taught[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )
        np.testing.assert_allclose(taught[:3, 3], [0.42, 0.10, 0.25])

    def test_legacy_taught_object_tilt_remains_available(self):
        fixed = taught_handoff_object_transform([0.42, 0.10, 0.25], 0.0)
        legacy = taught_handoff_object_transform([0.42, 0.10, 0.25], 5.0)
        _, rotation_deg = transform_error(fixed, legacy)
        self.assertAlmostEqual(rotation_deg, 5.0)

    def test_object_minus_x_pregrasp_follows_live_workpiece_axis(self):
        grasp = translation(0.4, -0.1, 0.3)
        object_pose = translation(0.42, 0.1, 0.25)
        object_pose[:3, :3] = Rotation.from_euler(
            "z", 90.0, degrees=True
        ).as_matrix()
        offset, metadata = pregrasp_offset_body(
            grasp, object_pose, 0.05, "object-minus-x"
        )
        np.testing.assert_allclose(offset, [0.0, -0.05, 0.0], atol=1e-9)
        self.assertEqual(metadata["frame"], "live workpiece")
        self.assertEqual(metadata["axis"], "-X")

    def test_default_pregrasp_mode_keeps_tool_minus_z_behavior(self):
        grasp = translation(0.4, -0.1, 0.3)
        object_pose = translation(0.42, 0.1, 0.25)
        object_pose[:3, :3] = Rotation.from_euler(
            "z", 90.0, degrees=True
        ).as_matrix()
        offset, metadata = pregrasp_offset_body(
            grasp, object_pose, 0.05, "tool-minus-z"
        )
        np.testing.assert_allclose(offset, [0.0, 0.0, -0.05], atol=1e-9)
        self.assertEqual(metadata["frame"], "right_hand_transfer_end")
        self.assertEqual(metadata["axis"], "-Z")

    def test_redundant_candidates_constrain_center_front_and_tail(self):
        current = translation(0.2, 0.1, 0.3)
        relation = translation(0.0, 0.0, 0.18)
        center = np.array([0.5, 0.0, -0.2])
        candidates = redundant_handoff_candidates(current, relation, center)
        self.assertEqual(len(candidates), 56)
        self.assertEqual(candidates[0]["name"], "exact/front_up_tail_right")
        np.testing.assert_allclose(
            candidates[0]["T_base_object_target"][:3, 3], center
        )
        np.testing.assert_allclose(
            candidates[0]["T_base_object_target"][:3, 0], [0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            candidates[0]["T_base_object_target"][:3, 2], [0.0, 0.0, 1.0]
        )
        self.assertTrue(all(c["center_offset_m"][2] >= 0.0 for c in candidates))
        self.assertTrue(all(
            np.linalg.norm(c["center_offset_m"]) <= 0.020001 for c in candidates
        ))
        self.assertTrue(all(c["front_up_error_deg"] <= 5.0001 for c in candidates))
        self.assertTrue(all(c["tail_to_right_error_deg"] <= 15.0001 for c in candidates))
        front_error, tail_error = handoff_axis_errors(
            candidates[0]["T_base_object_target"]
        )
        self.assertAlmostEqual(front_error, 0.0)
        self.assertAlmostEqual(tail_error, 0.0)
        for candidate in candidates:
            reconstructed = (
                candidate["T_base_left_tcp_target"] @ relation
            )
            np.testing.assert_allclose(
                reconstructed, candidate["T_base_object_target"], atol=1e-9
            )

    def test_result_target_center_uses_placement_position(self):
        center = result_target_center({
            "placement": {
                "target_object_pose": {"position": [0.5, 0.02, -0.2]},
            }
        })
        np.testing.assert_allclose(center, [0.5, 0.02, -0.2])

    def test_result_target_center_defaults_to_fixed_handoff(self):
        np.testing.assert_allclose(
            result_target_center({}), [0.42, 0.20, 0.12]
        )

    def test_body_result_can_build_relation_from_tcp_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({
                "object_pose": {
                    "frame_id": "body_base",
                    "position_m": [0.2, 0.1, 0.4],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "left_tcp_snapshot": {
                    "frame_id": "body_base",
                    "T_base_left_tcp": translation(0.2, 0.1, 0.3).tolist(),
                },
            }))
            _, _, relation = load_result(path)
            np.testing.assert_allclose(relation, translation(0, 0, 0.1), atol=1e-9)

    def test_capture_tcp_reconstructed_from_recorded_relation(self):
        tcp = translation(0.2, 0.1, 0.3)
        relation = translation(0.0, 0.0, 0.15)
        obj = tcp @ relation
        np.testing.assert_allclose(captured_left_tcp(obj, relation), tcp, atol=1e-9)

    def test_left_target_places_object_at_handoff_pose(self):
        T_base_object_current = translation(0.3, 0.1, 0.4)
        T_base_left_current = translation(0.2, 0.1, 0.4)
        T_base_object_handoff = translation(0.4, 0.0, 0.3)
        T_right_object = translation(0.0, 0.0, 0.08)
        targets = compute_handoff_targets(
            T_base_object_current,
            T_base_left_current,
            T_base_object_handoff,
            T_right_object,
            [-0.05, 0.0, 0.0],
        )
        reconstructed = (
            targets["T_base_left_tcp_handoff"] @ targets["T_left_tcp_object"]
        )
        np.testing.assert_allclose(reconstructed, T_base_object_handoff, atol=1e-9)

    def test_right_target_preserves_calibrated_tcp_object_relation(self):
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        T_base_object_handoff = translation(0.4, 0.0, 0.3) @ rotation
        T_right_object = translation(0.0, 0.0, 0.08)
        targets = compute_handoff_targets(
            np.eye(4), np.eye(4), T_base_object_handoff, T_right_object,
            [-0.05, 0.0, 0.0],
        )
        reconstructed = targets["T_base_right_tcp_grasp"] @ T_right_object
        np.testing.assert_allclose(reconstructed, T_base_object_handoff, atol=1e-9)

    def test_pregrasp_offset_is_expressed_in_grasp_tcp_frame(self):
        T_base_object_handoff = np.eye(4)
        T_base_object_handoff[:3, :3] = Rotation.from_euler(
            "z", 90, degrees=True
        ).as_matrix()
        targets = compute_handoff_targets(
            np.eye(4), np.eye(4), T_base_object_handoff, np.eye(4),
            [-0.05, 0.0, 0.0],
        )
        delta = (
            targets["T_base_right_tcp_pregrasp"][:3, 3]
            - targets["T_base_right_tcp_grasp"][:3, 3]
        )
        np.testing.assert_allclose(delta, [0.0, -0.05, 0.0], atol=1e-9)

    def test_tail_grasp_places_right_pinch_point_and_approaches_from_tail(self):
        T_base_object = np.eye(4)
        # Object tail +X is physically toward the right arm at body_base +Y.
        T_base_object[:3, :3] = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        T_base_object[:3, 3] = [0.42, 0.20, 0.12]
        grasp_object = [0.034, 0.0, 0.01065]
        pinch_tcp = [0.0, 0.0, 0.1089]
        targets = compute_right_tail_grasp_targets(
            T_base_object, grasp_object, pinch_tcp, 0.08
        )
        grasp = targets["T_base_right_tcp_grasp"]
        reconstructed_pinch = grasp[:3, 3] + grasp[:3, :3] @ pinch_tcp
        expected_pinch = (
            T_base_object[:3, 3]
            + T_base_object[:3, :3] @ np.asarray(grasp_object)
        )
        np.testing.assert_allclose(reconstructed_pinch, expected_pinch, atol=1e-9)
        # TCP +Z approaches inward, opposite the object's tail axis.
        np.testing.assert_allclose(
            grasp[:3, 2], -T_base_object[:3, 0], atol=1e-9
        )
        # TCP +X is the jaw-opening axis and clamps across object width.
        np.testing.assert_allclose(
            grasp[:3, 0], T_base_object[:3, 1], atol=1e-9
        )
        outside_delta = (
            targets["T_base_right_tcp_pregrasp"][:3, 3] - grasp[:3, 3]
        )
        np.testing.assert_allclose(
            outside_delta, 0.08 * T_base_object[:3, 0], atol=1e-9
        )

    def test_oblique_tail_pregrasp_stays_outside_object_tail(self):
        T_base_object = np.eye(4)
        for mode, expected_vertical_sign in (
            ("tail_above_30", 1.0),
            ("tail_below_30", -1.0),
        ):
            targets = compute_right_tail_grasp_targets(
                T_base_object,
                [0.034, 0.0, 0.01065],
                [0.0, 0.0, 0.1089],
                0.08,
                approach_mode=mode,
            )
            delta = (
                targets["T_base_right_tcp_pregrasp"][:3, 3]
                - targets["T_base_right_tcp_grasp"][:3, 3]
            )
            self.assertGreater(float(delta @ T_base_object[:3, 0]), 0.0)
            self.assertEqual(float(np.sign(delta @ T_base_object[:3, 2])),
                             expected_vertical_sign)

    def test_body_side_slide_starts_toward_object_front(self):
        targets = compute_right_tail_grasp_targets(
            np.eye(4),
            [0.034, 0.0, 0.01065],
            [0.0, 0.0, 0.1089],
            0.08,
            approach_mode="slide_from_body_30",
        )
        delta = (
            targets["T_base_right_tcp_pregrasp"][:3, 3]
            - targets["T_base_right_tcp_grasp"][:3, 3]
        )
        self.assertLess(float(delta @ np.asarray([1.0, 0.0, 0.0])), 0.0)

    def test_camera_frame_icp_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({
                "object_pose": {
                    "frame_id": "head_d435i_color_optical_frame",
                    "position_m": [0, 0, 0.3],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "T_output_object": np.eye(4).tolist(),
            }))
            with self.assertRaisesRegex(RuntimeError, "expected 'body_base'"):
                load_icp_object_transform(path)

    def test_planner_tf_payload_is_inverted(self):
        serialized = {
            "data": {
                "source_frame": "left_tcp",
                "target_frame": "body_base",
                "translation": {"x": -0.2, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        }
        T_base_left = planner_tf_target_source(serialized, "left_tcp", "body_base")
        np.testing.assert_allclose(T_base_left, translation(0.2, 0.0, 0.0))

    def test_invalid_rotation_is_rejected(self):
        matrix = np.eye(4)
        matrix[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_transform(matrix, "bad")

    def test_transform_error_reports_translation_and_rotation(self):
        moved = translation(0.003, 0.0, 0.0)
        moved[:3, :3] = Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
        distance_m, angle_deg = transform_error(np.eye(4), moved)
        self.assertAlmostEqual(distance_m, 0.003)
        self.assertAlmostEqual(angle_deg, 2.0)

    def test_motion_validation_requires_synchronized_tcp_and_good_icp(self):
        config = {
            "expected_head_pose_deg": {"pitch": 2.0, "yaw": 19.0},
            "left_tool_frame": "left_hand_transfer_end",
            "motion_quality_min_fitness": 0.33,
            "motion_quality_max_rmse_m": 0.007,
            "motion_min_registration_points": 300,
        }
        result = {
            "transform_info": {
                "status": "calibrated_fixed_head_pose",
                "calibration_head_pose_deg": {"pitch": 2.0, "yaw": 19.0},
            },
            "head_pose_command_deg": {"pitch": 2.0, "yaw": 19.0},
            "registration": {"fitness": 0.9, "rmse_m": 0.004},
            "point_report": {"valid_point_count": 1000},
            "left_tcp_snapshot": {
                "frame_id": "body_base",
                "tcp_frame": "left_hand_transfer_end",
                "T_base_left_tcp": np.eye(4).tolist(),
            },
        }
        np.testing.assert_allclose(validate_icp_for_motion(result, config), np.eye(4))
        result["registration"]["fitness"] = 0.2
        with self.assertRaisesRegex(RuntimeError, "fitness"):
            validate_icp_for_motion(result, config)

    def test_trajectory_summary_rejects_wrong_arm(self):
        response = {
            "data": {
                "arm_id": 2,
                "trajectory": [[0.0] * 7],
                "sample_frequency": 10.0,
            }
        }
        with self.assertRaisesRegex(RuntimeError, "expected 1"):
            trajectory_summary(response, 1)

    def test_configure_target_changes_only_object_center(self):
        current = translation(0.1, 0.2, 0.3)
        current[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        config = {"transform_convention": "p_A = T_A_B @ p_B"}
        updated = config_with_handoff_target(
            {
                **config,
                "left_tool_frame": "left_hand_transfer_end",
                "collision_box_dimensions_m": [0.102, 0.057, 0.025],
                "collision_box_center_object_m": [0.0, 0.0, 0.01],
                "collision_touch_links": ["left_finger", "right_finger"],
            },
            current,
            np.eye(4),
            [0.4, 0.0, 0.25],
            Path("result.json"),
        )
        target = np.asarray(updated["T_base_object_handoff"])
        np.testing.assert_allclose(target[:3, 3], [0.4, 0.0, 0.25])
        np.testing.assert_allclose(target[:3, :3], current[:3, :3])
        self.assertEqual(
            updated["attached_collision"]["attached_link"],
            "left_hand_transfer_end",
        )
        self.assertEqual(
            updated["attached_collision"]["touch_links"],
            ["left_finger", "right_finger"],
        )

    def test_recorded_right_relation_reconstructs_handoff_object(self):
        config = {
            "transform_convention": "p_A = T_A_B @ p_B",
            "right_tool_frame": "right_hand_transfer_end",
            "T_base_object_handoff": translation(0.4, 0.0, 0.25).tolist(),
            "collision_box_center_object_m": [0.0, 0.0, 0.01],
            "attached_collision": {
                "id": "held_reverse_workpiece",
                "attached_link": "left_hand_transfer_end",
            },
        }
        T_base_right = translation(0.35, 0.0, 0.25)
        updated = config_with_right_relation(config, T_base_right, [-0.05, 0, 0])
        reconstructed = T_base_right @ np.asarray(updated["T_right_tcp_object"])
        np.testing.assert_allclose(
            reconstructed, np.asarray(config["T_base_object_handoff"]), atol=1e-9
        )
        self.assertEqual(
            updated["attached_collision"]["right_attached_link"],
            "right_hand_transfer_end",
        )

    def test_attachment_owner_handshake_requires_matching_request_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({
                "requested_owner": "left",
                "request_id": None,
                "applied_owner": None,
                "applied_request_id": None,
            }))
            request_id = write_attachment_request(path, "right")
            self.assertFalse(attachment_request_applied(path, "right", request_id))
            state = json.loads(path.read_text())
            state["applied_owner"] = "right"
            state["applied_request_id"] = request_id
            path.write_text(json.dumps(state))
            self.assertTrue(attachment_request_applied(path, "right", request_id))

    def test_execution_pose_gate_rejects_excess_error(self):
        with self.assertRaisesRegex(RuntimeError, "did not reach target"):
            require_pose_close(
                translation(0.004, 0.0, 0.0),
                np.eye(4),
                translation_tolerance_mm=3.0,
                rotation_tolerance_deg=2.0,
                label="test pose",
            )

    def test_center_move_preserves_object_orientation_and_changes_only_center(self):
        current_object = translation(0.15, 0.22, -0.31)
        current_object[:3, :3] = Rotation.from_euler(
            "xyz", [15, -20, 35], degrees=True
        ).as_matrix()
        current_left = translation(0.10, 0.20, -0.30)
        relation, target_object, target_left = centre_target(
            current_object, current_left, [0.25, 0.0, -0.20]
        )
        np.testing.assert_allclose(target_object[:3, 3], [0.25, 0.0, -0.20])
        np.testing.assert_allclose(
            target_object[:3, :3], current_object[:3, :3], atol=1e-9
        )
        np.testing.assert_allclose(target_left @ relation, target_object, atol=1e-9)

    def test_right_pregrasp_and_grasp_are_one_ordered_plan(self):
        class FakePlanner:
            def __init__(self):
                self.waypoints = None

            def plan_trajectory(self, waypoints, **kwargs):
                self.waypoints = waypoints
                return {
                    "data": {
                        "arm_id": 2,
                        "trajectory": [[0.0] * 7, [0.1] * 7],
                        "sample_frequency": 10.0,
                    }
                }

        config = LoopConfig(
            planner_host="127.0.0.1",
            execute_url="http://127.0.0.1:5667/middle/waypoint/execute",
            base_frame="body_base",
            left_group="left_arm",
            right_group="right_arm",
            left_tool_frame="left_hand_transfer_end",
            right_tool_frame="right_hand_transfer_end",
            max_velocity=0.05,
            max_acceleration=0.05,
            idle_poll_seconds=0.2,
            live=True,
            poses={},
        )
        runtime = PlannerRuntime(config)
        fake = FakePlanner()
        runtime._planner = fake
        pose = ObjectPose("body_base", [0.3, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0])
        response = runtime.plan_cartesian_sequence(
            "right", [("rrtstar", pose), ("lin", pose)]
        )
        self.assertEqual(response["data"]["arm_id"], 2)
        self.assertEqual([item["planner"] for item in fake.waypoints], ["rrtstar", "lin"])
        self.assertEqual(
            [item["ik_frame"] for item in fake.waypoints],
            ["right_hand_transfer_end", "right_hand_transfer_end"],
        )


if __name__ == "__main__":
    unittest.main()
