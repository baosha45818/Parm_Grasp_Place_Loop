import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from tool.inspection_support.stl_model import (
    load_stl_triangles_m,
    triangles_in_object_frame,
)
from tests.reversed_head_observation_2.workpiece_model import (
    HERE,
    load_model_config,
    tail_grasp_point_object_m,
)
from tests.reversed_head_observation_2.execute_first_workpiece_place import (
    load_config as load_first_workpiece_place_config,
)
from tests.reversed_head_observation_2.execute_full_process import (
    parse_args as parse_full_process_args,
)
from tests.reversed_head_observation.move_object_center import (
    parse_args as parse_move_object_center_args,
)
from tests.reversed_head_observation_2.execute_icp_tail_handoff import (
    DEFAULT_HANDOFF_AXIS_TOLERANCE_DEG,
    DEFAULT_POST_RELEASE_CLEARANCE_MINUS_X_M,
    DEFAULT_POST_RELEASE_CLEARANCE_PLUS_Z_M,
    DEFAULT_RIGHT_PREGRASP_JOINTS_DEG,
    DEFAULT_RIGHT_GRASP_JOINTS_DEG,
    DEFAULT_RIGHT_PLACE_READY_JOINTS_DEG,
    advance_grasp_along_object_x,
    build_post_release_clearance_target,
    parse_args as parse_tail_handoff_args,
    validate_live_object,
)
from tests.reversed_head_observation.execute_taught_tail_handoff import (
    taught_right_tcp_from_joints,
)
from modules.workpiece_2_handoff_profile import (
    load_workpiece_2_handoff_profile,
    workpiece_2_right_prepare_joints_deg,
)
from modules.module_5_left_pick.module_5_2_handoff_prepare.main import (
    DEFAULT_ALIGNMENT,
    HANDOFF_HEAD_PITCH_DEG,
    HANDOFF_HEAD_YAW_DEG,
    publish_handoff_result,
)
from modules.module_8_right_grasp.module_8_2_left_handoff.main import (
    WORKPIECE_1_HANDOFF_CENTER_BODY_M,
    WORKPIECE_1_RIGHT_GRASP_JOINTS_DEG,
    WORKPIECE_1_RIGHT_PLACE_READY_JOINTS_DEG,
    execute_movel_handoff,
    execute_workpiece_2_handoff,
)
from modules.module_8_right_grasp.module_8_2_left_handoff.workpiece_2_runtime.execute_icp_tail_handoff import (
    current_arm_joint_error_deg,
)


class Workpiece2AdaptationTests(unittest.TestCase):
    def test_production_high_handoff_profile_is_one_consistent_contract(self):
        profile = load_workpiece_2_handoff_profile()
        self.assertEqual(profile["workpiece"], "2")
        self.assertEqual(
            profile["profile_id"],
            "workpiece_2_reverse_grasp_handoff_toml_v1",
        )
        np.testing.assert_allclose(
            profile["handoff"]["center_body_m"], [0.42, 0.10, 0.25]
        )
        np.testing.assert_allclose(
            profile["right_handoff"]["prepare_joints_deg"],
            profile["right_handoff"]["pregrasp_joints_deg"],
        )
        np.testing.assert_allclose(
            profile["left_return_joints_deg"],
            profile["left_observation"]["prepare_joints_deg"],
        )
        np.testing.assert_allclose(
            workpiece_2_right_prepare_joints_deg(),
            DEFAULT_RIGHT_PREGRASP_JOINTS_DEG,
        )

    def test_module_5_2_handoff_contract_identifies_workpiece_and_profile(self):
        profile = load_workpiece_2_handoff_profile()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "handoff.json"
            result = Path(directory) / "result.json"
            result.write_text("{}", encoding="utf-8")
            publish_handoff_result(
                destination,
                result,
                workpiece="2",
                profile_id=profile["profile_id"],
                handoff_center_body_m=profile["handoff"]["center_body_m"],
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["event"], "module_5_2_left_handoff_ready")
        self.assertEqual(payload["workpiece"], "2")
        self.assertEqual(payload["profile_id"], profile["profile_id"])
        np.testing.assert_allclose(
            payload["handoff_center_body_m"],
            profile["handoff"]["center_body_m"],
        )

    def test_workpiece_1_observation_uses_the_shared_high_profile(self):
        profile = load_workpiece_2_handoff_profile()
        head_pose = profile["left_observation"]["head_pose_deg"]
        self.assertEqual(HANDOFF_HEAD_PITCH_DEG, head_pose["pitch"])
        self.assertEqual(HANDOFF_HEAD_YAW_DEG, head_pose["yaw"])
        self.assertEqual(DEFAULT_ALIGNMENT, profile["left_observation"]["alignment"])

    def test_workpiece_1_uses_updated_high_handoff_taught_pose(self):
        profile = load_workpiece_2_handoff_profile()
        captured_argv = []

        def capture_handoff_command():
            captured_argv.extend(sys.argv)
            return 0

        args = SimpleNamespace(
            planner_host="planner",
            gripper_host="gripper",
            arm_host="arm",
            speed_multiplier=3.0,
            require_confirmation=False,
            right_place_ready_file=None,
        )
        with patch(
            "tests.reversed_head_observation.execute_taught_tail_handoff.main",
            side_effect=capture_handoff_command,
        ):
            self.assertEqual(execute_movel_handoff(Path("result.json"), args), 0)

        grasp_start = captured_argv.index("--right-grasp-joints-deg") + 1
        high_joints = [
            float(value) for value in captured_argv[grasp_start:grasp_start + 7]
        ]
        high_tcp = taught_right_tcp_from_joints(high_joints)
        high_center = np.asarray(WORKPIECE_1_HANDOFF_CENTER_BODY_M)
        np.testing.assert_allclose(
            high_joints,
            WORKPIECE_1_RIGHT_GRASP_JOINTS_DEG,
        )
        self.assertEqual(
            captured_argv[captured_argv.index("--final-grasp-motion") + 1],
            "taught-joints",
        )
        self.assertEqual(
            float(
                captured_argv[
                    captured_argv.index("--right-grasp-tolerance-deg") + 1
                ]
            ),
            0.5,
        )
        np.testing.assert_allclose(
            high_tcp[:3, 3],
            # Current workpiece-1 high-handoff taught pose.  This value must
            # track Module 8-2's live taught joint target, not the previous
            # high-handoff teaching.
            [0.3931181508, -0.1248361212, 0.2948360274],
            atol=1e-7,
        )
        self.assertEqual(
            captured_argv[captured_argv.index("--pregrasp-axis") + 1],
            "object-minus-x",
        )
        self.assertEqual(
            float(captured_argv[captured_argv.index("--pregrasp-distance-m") + 1]),
            0.05,
        )
        self.assertEqual(
            float(
                captured_argv[
                    captured_argv.index("--arrival-translation-tolerance-mm") + 1
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(
                captured_argv[
                    captured_argv.index("--taught-object-tilt-deg") + 1
                ]
            ),
            0.0,
        )
        center_start = captured_argv.index("--expected-center-m") + 1
        np.testing.assert_allclose(
            [float(value) for value in captured_argv[center_start:center_start + 3]],
            high_center,
        )
        return_start = captured_argv.index("--left-return-joints-deg") + 1
        np.testing.assert_allclose(
            [float(value) for value in captured_argv[return_start:return_start + 7]],
            profile["left_return_joints_deg"],
        )
        ready_start = captured_argv.index("--right-place-ready-joints-deg") + 1
        np.testing.assert_allclose(
            [float(value) for value in captured_argv[ready_start:ready_start + 7]],
            WORKPIECE_1_RIGHT_PLACE_READY_JOINTS_DEG,
        )
        np.testing.assert_allclose(
            WORKPIECE_1_RIGHT_PLACE_READY_JOINTS_DEG,
            [22.1484, 10.9122, 42.3263, 107.9434, -97.3484, -25.6449, -24.1054],
        )

    def test_module_8_2_forwards_the_complete_high_handoff_profile(self):
        profile = load_workpiece_2_handoff_profile()
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text("{}", encoding="utf-8")
            marker = Path(directory) / "ready.json"
            args = SimpleNamespace(
                planner_host="planner",
                gripper_host="gripper",
                arm_host="arm",
                speed_multiplier=3.0,
                require_confirmation=False,
                right_place_ready_file=marker,
            )
            with patch(
                "modules.module_8_right_grasp.module_8_2_left_handoff.main.subprocess.run"
            ) as run_mock:
                execute_workpiece_2_handoff(result, args)
            command = run_mock.call_args.args[0]
            ready_payload = json.loads(marker.read_text(encoding="utf-8"))
        self.assertIn("--speed-multiplier", command)
        self.assertEqual(command[command.index("--speed-multiplier") + 1], "3.0")
        for flag, expected in (
            ("--right-pregrasp-joints-deg", profile["right_handoff"]["pregrasp_joints_deg"]),
            ("--right-grasp-joints-deg", profile["right_handoff"]["grasp_joints_deg"]),
            ("--right-place-ready-joints-deg", profile["right_handoff"]["place_ready_joints_deg"]),
            ("--left-return-joints-deg", profile["left_return_joints_deg"]),
        ):
            start = command.index(flag) + 1
            np.testing.assert_allclose(
                [float(value) for value in command[start:start + 7]], expected
            )
        self.assertEqual(ready_payload["workpiece"], "2")
        self.assertEqual(ready_payload["profile_id"], profile["profile_id"])

    def test_loop_validates_workpiece_2_profile_before_right_motion(self):
        profile = load_workpiece_2_handoff_profile()
        loop_dir = HERE.parent / "loop"
        sys.path.insert(0, str(loop_dir))
        try:
            loop = importlib.import_module("simple_dual_arm_loop")
        finally:
            sys.path.remove(str(loop_dir))
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text("{}", encoding="utf-8")
            contract = Path(directory) / "handoff.json"
            contract.write_text(
                json.dumps(
                    {
                        "event": "module_5_2_left_handoff_ready",
                        "workpiece": "2",
                        "profile_id": profile["profile_id"],
                        "handoff_center_body_m": profile["handoff"][
                            "center_body_m"
                        ],
                        "left_gripper_holds_object": True,
                        "result": str(result),
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "PARM_MODULAR_LOOP_ENTRIES": "1",
                    "PARM_LOOP_WORKPIECE": "2",
                },
                clear=False,
            ):
                validated_result, workpiece = (
                    loop.validate_reverse_handoff_contract(contract)
                )
                np.testing.assert_allclose(
                    loop.reverse_handoff_prepare_joints_deg(),
                    profile["right_handoff"]["prepare_joints_deg"],
                )
        self.assertEqual(validated_result, result.resolve())
        self.assertEqual(workpiece, "2")

    def test_modular_workpiece_1_keeps_its_own_right_entry_and_validates_center(self):
        profile = load_workpiece_2_handoff_profile()
        loop_dir = HERE.parent / "loop"
        sys.path.insert(0, str(loop_dir))
        try:
            loop = importlib.import_module("simple_dual_arm_loop")
        finally:
            sys.path.remove(str(loop_dir))
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text("{}", encoding="utf-8")
            contract = Path(directory) / "handoff.json"
            contract.write_text(
                json.dumps(
                    {
                        "event": "module_5_2_left_handoff_ready",
                        "workpiece": "1",
                        "handoff_center_body_m": profile["handoff"]["center_body_m"],
                        "left_gripper_holds_object": True,
                        "result": str(result),
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "PARM_MODULAR_LOOP_ENTRIES": "1",
                    "PARM_LOOP_WORKPIECE": "1",
                },
                clear=False,
            ):
                validated_result, workpiece = loop.validate_reverse_handoff_contract(
                    contract
                )
                np.testing.assert_allclose(
                    loop.reverse_handoff_prepare_joints_deg(),
                    loop.RIGHT_REVERSE_HANDOFF_PREP_JOINTS_DEG,
                )
        self.assertEqual(validated_result, result.resolve())
        self.assertEqual(workpiece, "1")

    def test_workpiece_2_pregrasp_entry_can_skip_zero_length_plan(self):
        target = load_workpiece_2_handoff_profile()["right_handoff"][
            "pregrasp_joints_deg"
        ]
        with patch("parm_sdk.Arm") as arm_class:
            arm_class.return_value.__enter__.return_value.get_state.return_value = {
                "state": {"right": {"position": list(target)}}
            }
            error = current_arm_joint_error_deg("arm", 2, list(target))
        self.assertAlmostEqual(error, 0.0)

    def test_documented_axes_are_applied_to_raw_stl(self):
        config = load_model_config()
        rotation = np.asarray(config["R_object_stl"], dtype=float)
        np.testing.assert_allclose(rotation @ [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotation @ [0.0, -1.0, 0.0], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_transformed_model_has_expected_center_and_size(self):
        config = load_model_config()
        triangles = triangles_in_object_frame(
            load_stl_triangles_m(config["stl_path"]),
            R_object_stl=config["R_object_stl"],
            object_origin_stl_m=config["object_origin_stl_m"],
        )
        vertices = triangles.reshape(-1, 3)
        np.testing.assert_allclose((vertices.min(0) + vertices.max(0)) / 2.0, 0.0, atol=1e-8)
        np.testing.assert_allclose(
            vertices.max(0) - vertices.min(0),
            [0.111, 0.087519776, 0.062130112],
            atol=1e-7,
        )

    def test_collision_box_encloses_model_with_clearance(self):
        model = load_model_config()
        handoff = json.loads((HERE / "handoff_config.json").read_text(encoding="utf-8"))
        model_min = np.asarray(model["bounds_object_m"]["min"])
        model_max = np.asarray(model["bounds_object_m"]["max"])
        box_size = np.asarray(handoff["collision_box_dimensions_m"])
        box_center = np.asarray(handoff["collision_box_center_object_m"])
        self.assertTrue(np.all(box_center - box_size / 2.0 <= model_min))
        self.assertTrue(np.all(box_center + box_size / 2.0 >= model_max))
        self.assertTrue(np.all(box_size - (model_max - model_min) >= 0.00399))

    def test_right_grasp_seed_is_15mm_inboard_on_positive_x_end(self):
        config = load_model_config()
        point = np.asarray(tail_grasp_point_object_m())
        bounds = config["bounds_object_m"]
        self.assertAlmostEqual(float(point[0]), float(bounds["max"][0]) - 0.015)
        self.assertTrue(np.all(point >= np.asarray(bounds["min"])))
        self.assertTrue(np.all(point <= np.asarray(bounds["max"])))

    def test_auto_mask_is_disabled_until_calibrated(self):
        config = json.loads(
            (HERE / "head_d435_auto_mask_config.json").read_text(encoding="utf-8")
        )
        self.assertIs(config["enabled"], False)

    def test_reverse_grasp_seed_uses_left_hold_end_in_raw_stl(self):
        config = json.loads(
            (HERE / "reverse_grasp_config.json").read_text(encoding="utf-8")
        )
        # Raw STL +X maps to documented workpiece -X, the left-hand end.
        np.testing.assert_allclose(config["grasp_point_stl_m"], [0.0405, 0.0, 0.0])
        np.testing.assert_allclose(config["grasp_width_axis_stl"], [0.0, 1.0, 0.0])

    def test_first_workpiece_place_mapping_is_explicitly_authorized(self):
        config = load_first_workpiece_place_config(
            HERE / "first_workpiece_place_config.json"
        )
        self.assertEqual(config["strategy"], "reuse_first_workpiece_module9_force_servo")
        self.assertEqual(len(config["right_place_ready_joints_deg"]), 7)
        self.assertEqual(len(config["module9_initial_right_joints_deg"]), 7)

    def test_new_right_place_ready_is_shared_with_placement(self):
        config = load_first_workpiece_place_config(
            HERE / "first_workpiece_place_config.json"
        )
        np.testing.assert_allclose(
            config["right_place_ready_joints_deg"],
            DEFAULT_RIGHT_PLACE_READY_JOINTS_DEG,
        )

    def test_right_taught_pregrasp_and_grasp_joint_defaults(self):
        np.testing.assert_allclose(DEFAULT_RIGHT_PREGRASP_JOINTS_DEG, [
            173.9519, 50.8063, -79.5218, 85.0397,
            -25.3112, 4.5552, 79.6344,
        ])
        np.testing.assert_allclose(DEFAULT_RIGHT_GRASP_JOINTS_DEG, [
            162.5467, 59.2095, -70.8701, 76.9661,
            -21.2146, 0.4807, 80.0175,
        ])
        pregrasp = taught_right_tcp_from_joints(DEFAULT_RIGHT_PREGRASP_JOINTS_DEG)
        grasp = taught_right_tcp_from_joints(DEFAULT_RIGHT_GRASP_JOINTS_DEG)
        self.assertGreater(float(np.linalg.norm(grasp[:3, 3] - pregrasp[:3, 3])), 0.005)

    def test_tail_advance_uses_live_object_positive_x(self):
        object_pose = np.eye(4)
        object_pose[:3, :3] = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        grasp = np.eye(4)
        grasp[:3, 3] = [0.4, 0.1, 0.2]
        advanced = advance_grasp_along_object_x(grasp, object_pose, 0.005)
        np.testing.assert_allclose(
            advanced[:3, 3] - grasp[:3, 3], [0.0, 0.005, 0.0], atol=1e-9
        )

    def test_post_release_clearance_is_object_minus_x_plus_z_30mm(self):
        tcp = np.eye(4)
        tcp[:3, 3] = [0.35, 0.15, 0.20]
        object_pose = np.eye(4)
        object_pose[:3, :3] = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        object_pose[:3, 3] = [0.42, 0.20, 0.12]
        relation = np.linalg.inv(tcp) @ object_pose
        clearance = build_post_release_clearance_target(
            tcp,
            relation,
            DEFAULT_POST_RELEASE_CLEARANCE_MINUS_X_M,
            DEFAULT_POST_RELEASE_CLEARANCE_PLUS_Z_M,
        )
        delta_object = object_pose[:3, :3].T @ (
            clearance[:3, 3] - tcp[:3, 3]
        )
        np.testing.assert_allclose(delta_object, [-0.03, 0.0, 0.03], atol=1e-9)
        np.testing.assert_allclose(clearance[:3, :3], tcp[:3, :3], atol=1e-9)

    def test_right_handoff_rejects_15deg_yaw_candidate(self):
        object_pose = np.eye(4)
        yaw_deg = 15.0
        yaw_rad = np.deg2rad(yaw_deg)
        object_pose[:3, :3] = np.asarray([
            [-np.sin(yaw_rad), -np.cos(yaw_rad), 0.0],
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ])
        object_pose[:3, 3] = [0.42, 0.20, 0.12]
        args = SimpleNamespace(
            expected_center_m=[0.42, 0.20, 0.12],
            center_tolerance_mm=25.0,
            axis_tolerance_deg=DEFAULT_HANDOFF_AXIS_TOLERANCE_DEG,
        )
        with self.assertRaisesRegex(RuntimeError, "交接姿态不正确"):
            validate_live_object(args, object_pose)

    def test_workpiece2_left_handoff_requests_fixed_orientation_cartesian_rrtstar(self):
        with patch(
            "sys.argv",
            [
                "move_object_center.py",
                "--result", "/tmp/result.json",
                "--cartesian-planner", "rrtstar",
                "--fixed-handoff-orientation",
            ],
        ):
            args = parse_move_object_center_args()
        self.assertEqual(args.cartesian_planner, "rrtstar")
        self.assertTrue(args.fixed_handoff_orientation)
        self.assertFalse(args.exact_handoff_pose)

    def test_full_process_accepts_already_left_held_resume_mode(self):
        with patch(
            "sys.argv",
            ["execute_full_process.py", "--handoff-center-m", "0.42", "0.20", "0.12", "--already-left-held"],
        ):
            args = parse_full_process_args()
        self.assertTrue(args.already_left_held)
        self.assertEqual(args.handoff_center_m, [0.42, 0.20, 0.12])

    def test_tail_handoff_accepts_after_release_resume_mode(self):
        with patch(
            "sys.argv",
            [
                "execute_icp_tail_handoff.py",
                "--result", "/tmp/result.json",
                "--execute",
                "--resume-after-left-release",
            ],
        ):
            args = parse_tail_handoff_args()
        self.assertTrue(args.resume_after_left_release)
        np.testing.assert_allclose(
            args.right_pregrasp_joints_deg, DEFAULT_RIGHT_PREGRASP_JOINTS_DEG
        )
        np.testing.assert_allclose(
            args.right_grasp_joints_deg, DEFAULT_RIGHT_GRASP_JOINTS_DEG
        )
        self.assertAlmostEqual(args.post_release_clearance_minus_x_m, 0.03)
        self.assertAlmostEqual(args.post_release_clearance_plus_z_m, 0.03)
        self.assertAlmostEqual(args.axis_tolerance_deg, 2.0)


if __name__ == "__main__":
    unittest.main()
