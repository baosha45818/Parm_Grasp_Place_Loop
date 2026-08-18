import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.deployment_modes import (
    apply_mode_environment,
    load_mode,
)
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.profiles import (
    load_reverse_grasp_profile,
    load_reverse_ready_profile,
)
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.workpiece_1.execute_confirmed_grasp import (
    build_candidates as build_workpiece_1_candidates,
)
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.workpiece_2.execute_convex_tab_grasp import (
    build_targets as build_workpiece_2_targets,
)


class ReverseGraspProfileTest(unittest.TestCase):
    def test_default_profiles_select_distinct_adapters(self) -> None:
        first = load_reverse_grasp_profile("1")
        second = load_reverse_grasp_profile("2")
        self.assertEqual(first.adapter, "unique_back_prediction")
        self.assertEqual(second.adapter, "convex_tab_prediction")
        self.assertTrue(first.perception["orientation_config"].is_file())
        self.assertGreaterEqual(second.execution["planning_inflight"], 1)
        self.assertEqual(second.execution["grasp_down_offset_m"], -0.005)

    def test_profile_cannot_be_used_for_another_workpiece(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "wrong.json"
            profile.write_text(
                '{"schema_version":1,"workpiece":"2","adapter":"convex_tab_prediction"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "工件编号"):
                load_reverse_grasp_profile("1", profile)

    def test_reverse_ready_preserves_deployed_motion_values(self) -> None:
        """重构只能搬迁配置，不能改变预备位、到位门限或重试次数。"""
        expected_joints = [
            14.281, -29.886, -85.633, 102.504, 37.378, 56.016, 9.241,
        ]
        for workpiece in ("1", "2"):
            ready = load_reverse_ready_profile(workpiece)
            self.assertEqual(ready["ready_joints_deg"], expected_joints)
            self.assertEqual(ready["arrival_tolerance_deg"], 0.5)
            self.assertEqual(ready["arrival_timeout_s"], 90.0)
            self.assertEqual(ready["post_close_lift_m"], 0.03)
            self.assertEqual(ready["max_grasp_attempts"], 5)


class ReverseGraspDeploymentModeTest(unittest.TestCase):
    EXPECTED = {
        "gzhu": {
            "serial": "261222076212",
            "arm": "192.168.88.1",
            "gripper": "127.0.0.1",
            "force": "127.0.0.1",
            "head": "127.0.0.1",
            "pitch_sign": "-1",
            "calibration_kind": "sdk_bridge",
            "body_z_down_m": 0.020,
            "head_pose_deg": {"pitch": 38.5, "yaw": 0.0},
            "grasp_offsets_m": (0.005, 0.010),
            "uses_fixed_width_axis": False,
            "sizes": {"1": ("1280", "720"), "2": ("1280", "720")},
        },
        "xiangxin": {
            "serial": "261722072445",
            "arm": "192.168.3.11",
            "gripper": "192.168.3.11",
            "force": "192.168.3.22",
            "head": "192.168.3.22",
            "pitch_sign": "1",
            "calibration_kind": "planner_direct",
            "body_z_down_m": 0.005,
            "head_pose_deg": {"pitch": 45.0, "yaw": 0.0},
            "grasp_offsets_m": (0.0, 0.0),
            "uses_fixed_width_axis": True,
            "sizes": {"1": ("848", "480"), "2": ("1280", "720")},
        },
    }

    def test_mode_atomically_selects_hardware_calibration_and_roi(self) -> None:
        for mode_name, expected in self.EXPECTED.items():
            mode = load_mode(mode_name)
            self.assertEqual(mode.camera_serial, expected["serial"])
            self.assertEqual(mode.calibration_kind, expected["calibration_kind"])
            self.assertEqual(mode.workpiece_1_body_z_down_m, expected["body_z_down_m"])
            calibration = json.loads(
                mode.body_head_calibration.read_text(encoding="utf-8")
            )
            self.assertEqual(calibration["head_pose_deg"], expected["head_pose_deg"])
            grasp_config = json.loads(
                mode.workpiece_1_grasp_config.read_text(encoding="utf-8")
            )
            self.assertEqual(
                (
                    grasp_config["wide_head_outward_grasp_offset_m"],
                    grasp_config["wide_head_inward_grasp_offset_m"],
                ),
                expected["grasp_offsets_m"],
            )
            self.assertEqual(
                "grasp_width_axis_stl" in grasp_config,
                expected["uses_fixed_width_axis"],
            )
            for workpiece in ("1", "2"):
                apply_mode_environment(mode, workpiece)
                self.assertEqual(os.environ["PARM_HEAD_D435_SERIAL"], expected["serial"])
                self.assertEqual(os.environ["PARM_ARM_HOST"], expected["arm"])
                self.assertEqual(os.environ["PARM_GRIPPER_HOST"], expected["gripper"])
                self.assertEqual(os.environ["PARM_FORCE_SENSOR_HOST"], expected["force"])
                self.assertEqual(os.environ["PARM_HEAD_HOST"], expected["head"])
                self.assertEqual(
                    os.environ["PARM_HEAD_FEEDBACK_PITCH_SIGN"],
                    expected["pitch_sign"],
                )
                self.assertEqual(
                    (
                        os.environ["PARM_HEAD_D435_WIDTH"],
                        os.environ["PARM_HEAD_D435_HEIGHT"],
                    ),
                    expected["sizes"][workpiece],
                )
                expected_roi = (
                    mode.tabletop_workspace
                    if workpiece == "1"
                    else mode.workpiece_2_tabletop_workspace
                )
                self.assertEqual(
                    Path(os.environ["PARM_HEAD_TABLETOP_WORKSPACE"]), expected_roi
                )
                self.assertEqual(
                    Path(os.environ["PARM_BODY_HEAD_CALIBRATION"]),
                    mode.body_head_calibration,
                )

    def test_calibration_files_match_their_authoritative_sources(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_paths = {
            "gzhu": (
                Path("/home/dainanjun/Public/clean/modules/module_5_left_pick/")
                / "module_5_1_reverse_grasp/config/body_head_calibration.json",
                "T",
            ),
            "xiangxin": (
                root
                / "modules/module_4_head_judge/module_4_1_buffer_state_estimation/"
                "calibration/head_charuco_13x5_calibration_head.json",
                "T_base_head_camera",
            ),
        }
        for mode_name, (source_path, source_key) in source_paths.items():
            # clean/ 是部署机上的重构前只读快照；缺失时跳过跨快照断言。
            if not source_path.is_file():
                continue
            mode = load_mode(mode_name)
            deployed = json.loads(mode.body_head_calibration.read_text(encoding="utf-8"))
            source = json.loads(source_path.read_text(encoding="utf-8"))
            np.testing.assert_array_equal(
                np.asarray(deployed["T_body_base_head_camera"]),
                np.asarray(source[source_key]),
            )


class ReverseGraspGeometryTest(unittest.TestCase):
    def test_workpiece_1_candidate_targets_keep_original_transform_order(self) -> None:
        camera_grasp = np.eye(4)
        camera_grasp[:3, :3] = Rotation.from_euler(
            "xyz", [0.2, -0.1, 0.3]
        ).as_matrix()
        camera_grasp[:3, 3] = [0.1, -0.2, 0.5]
        body_head = np.eye(4)
        body_head[:3, :3] = Rotation.from_euler(
            "xyz", [-0.3, 0.2, 0.1]
        ).as_matrix()
        body_head[:3, 3] = [0.02, 0.03, 0.04]

        candidates = build_workpiece_1_candidates(
            camera_grasp, body_head, [90.0, 15.0, -15.0], 0.03, 0.005
        )
        self.assertEqual(
            [candidate.approach_deg for candidate in candidates],
            [90.0, 15.0, -15.0],
        )
        for candidate in candidates:
            tilt = np.eye(4)
            tilt[:3, :3] = Rotation.from_rotvec(
                [
                    0.0,
                    np.deg2rad(
                        np.copysign(
                            90.0 - abs(candidate.approach_deg),
                            candidate.approach_deg,
                        )
                    ),
                    0.0,
                ]
            ).as_matrix()
            expected_camera = camera_grasp @ tilt
            expected_target = body_head @ expected_camera
            expected_target[2, 3] -= 0.005
            expected_pregrasp_camera = expected_camera.copy()
            expected_pregrasp_camera[:3, 3] -= expected_camera[:3, 0] * 0.03
            expected_pregrasp = body_head @ expected_pregrasp_camera
            expected_pregrasp[2, 3] -= 0.005
            np.testing.assert_array_equal(candidate.body_grasp, expected_target)
            np.testing.assert_array_equal(candidate.body_pregrasp, expected_pregrasp)

    def test_workpiece_2_pregrasp_keeps_direct_left_eef_geometry(self) -> None:
        camera_tcp = np.eye(4)
        camera_tcp[:3, 3] = [0.1, 0.2, 0.3]
        body_head = np.eye(4)
        body_head[:3, 3] = [0.01, -0.02, 0.04]
        target, pregrasp = build_workpiece_2_targets(
            camera_tcp, body_head, 0.06
        )
        expected_target = body_head @ camera_tcp
        expected_camera_pregrasp = camera_tcp.copy()
        expected_camera_pregrasp[:3, 3] -= camera_tcp[:3, 0] * 0.06
        np.testing.assert_array_equal(target, expected_target)
        np.testing.assert_array_equal(
            pregrasp, body_head @ expected_camera_pregrasp
        )


if __name__ == "__main__":
    unittest.main()
