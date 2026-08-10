"""验证拆分后模块协议，不连接相机、规划器、夹爪或机械臂。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from modules.module_2_left_inspect_apart import run_inspect_then_place
from modules.module_3_left_place.standalone_left_place import (
    GRASPUP_ABOVE_JOINTS_DEG,
    GRASPUP_ABOVE_PLANNING_TIMEOUT_S,
    parse_args as parse_left_place_args,
    resolve_place_profile,
    select_face_candidate_order,
)
from modules.module_3_left_place.placement_config import load_workpiece_parameters
from modules.stage_result import (
    read_stage_result,
    validate_inspection_success,
    write_stage_result,
)


ROOT = Path(__file__).resolve().parents[1]
WORKPIECE_1 = load_workpiece_parameters("1")
WORKPIECE_2 = load_workpiece_parameters("2")


class ApartModuleContractTest(unittest.TestCase):
    def test_inspection_result_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inspect.json"
            payload = {
                "stage": "left_inspect",
                "outcome": "success",
                "T_body_base_object": [[1, 0, 0, 0]] * 4,
                "T_tool_object": [[1, 0, 0, 0]] * 4,
                "T_body_left_eef_at_inspect": [[1, 0, 0, 0]] * 4,
                "T_left_eef_object": [[1, 0, 0, 0]] * 4,
            }
            write_stage_result(path, payload)
            loaded = read_stage_result(path, expected_stage="left_inspect")
            validate_inspection_success(loaded)
            self.assertIn("T_body_base_object", loaded)
            self.assertIn("T_left_eef_object", loaded)

    def test_wrong_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_stage_result(path, {
                "stage": "left_place",
                "outcome": "success",
            })
            with self.assertRaises(RuntimeError):
                read_stage_result(path, expected_stage="left_inspect")

    def test_workpiece_candidate_policies(self) -> None:
        self.assertEqual(WORKPIECE_1.XY_RADIUS_FACTORS, [0.0])
        self.assertEqual(WORKPIECE_2.XY_RADIUS_FACTORS, [0.0])
        self.assertEqual(WORKPIECE_1.FRONT_YAW_CANDIDATES_DEG[0], 45.0)
        self.assertEqual(
            WORKPIECE_1.REVERSED_YAW_CANDIDATES_DEG[:5],
            [0.0, -30.0, 30.0, -60.0, 60.0],
        )

    def test_module_3_mapping_is_front_reversed_only(self) -> None:
        expected = {
            ("FRONT", "BRANCH_1_FRONT"): "3-1",
            ("REVERSED", "BRANCH_1_REVERSED"): "3-3",
        }
        for arguments, module_id in expected.items():
            self.assertEqual(resolve_place_profile(*arguments).module_id, module_id)
        with self.assertRaises(RuntimeError):
            resolve_place_profile("UNKNOWN", "BRANCH_1_FRONT")

    def test_face_candidate_order_uses_object_plus_z_vs_left_eef_x(self) -> None:
        front, order, minus_x_angle, plus_x_angle = select_face_candidate_order(
            [-1.0, 0.0, 0.0], threshold_deg=40.0, workpiece="1",
        )
        self.assertEqual((front, order), ("front_then_reversed", "front_then_reversed"))
        self.assertAlmostEqual(minus_x_angle, 0.0)
        self.assertAlmostEqual(plus_x_angle, 180.0)

        reversed_mode, order, minus_x_angle, plus_x_angle = select_face_candidate_order(
            [1.0, 0.0, 0.0], threshold_deg=40.0, workpiece="1",
        )
        self.assertEqual((reversed_mode, order), ("reversed_then_front", "reversed_then_front"))
        self.assertAlmostEqual(minus_x_angle, 180.0)
        self.assertAlmostEqual(plus_x_angle, 0.0)

        alternating, order, _, _ = select_face_candidate_order(
            [0.0, 1.0, 0.0], threshold_deg=40.0, workpiece="1",
        )
        self.assertEqual((alternating, order), ("alternating", "reversed_only"))

    def test_original_candidate_cli_aliases_are_supported(self) -> None:
        argv = [
            "standalone_left_place.py",
            "--inspection-result", "/tmp/inspect.json",
            "--result-file", "/tmp/place.json",
            "--branch1-front-target-z-candidates-m", "-0.42", "-0.41",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_left_place_args()
        self.assertEqual(args.front_target_z_candidates_m, [-0.42, -0.41])
        self.assertEqual(len(GRASPUP_ABOVE_JOINTS_DEG), 7)
        self.assertEqual(GRASPUP_ABOVE_PLANNING_TIMEOUT_S, 10.0)

    def test_wrapper_forwards_weight_and_place_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "weight.json"
            weight.write_text("{}", encoding="utf-8")
            argv = [
                "run_inspect_then_place.py",
                "--workpiece", "1",
                "--output-dir", str(root / "output"),
                "--weight-handoff-file", str(weight),
                "--auto-confirm-motion",
                "--skip-motion",
                "--branch1-front-target-z-candidates-m", "-0.42", "-0.41",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    run_inspect_then_place.subprocess,
                    "run",
                ) as mocked_run,
            ):
                def write_mock_inspection_result(command, **_kwargs):
                    if Path(command[1]).name != "standalone_inspect.py":
                        return None
                    result_path = Path(command[command.index("--result-file") + 1])
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text("{}", encoding="utf-8")
                    return None

                mocked_run.side_effect = write_mock_inspection_result
                self.assertEqual(run_inspect_then_place.main(), 0)
            commands = [call.args[0] for call in mocked_run.call_args_list]
            self.assertEqual(len(commands), 2)
            inspect_command, place_command = commands
            self.assertIn("--weight-handoff-file", inspect_command)
            self.assertIn("--weight-handoff-file", place_command)
            self.assertIn("--auto-confirm-motion", place_command)
            self.assertIn("--branch1-front-target-z-candidates-m", place_command)
            self.assertIn("-0.42", place_command)

    def test_registration_recovery_skips_left_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "left_place_ready.json"
            permit = root / "left_place_permit"
            argv = [
                "run_inspect_then_place.py",
                "--workpiece", "2",
                "--output-dir", str(root / "output"),
                "--placement-ready-file", str(ready),
                "--placement-permit-file", str(permit),
                "--skip-motion",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    run_inspect_then_place.subprocess, "run",
                ) as mocked_run,
            ):
                def write_recovered_result(command, **_kwargs):
                    result_path = Path(command[command.index("--result-file") + 1])
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(
                        json.dumps({
                            "stage": "left_inspect",
                            "outcome": "pc1_pc2_registration_failed_recovered",
                        }),
                        encoding="utf-8",
                    )

                mocked_run.side_effect = write_recovered_result
                self.assertEqual(run_inspect_then_place.main(), 0)
            self.assertEqual(mocked_run.call_count, 1)
            self.assertEqual(
                json.loads(ready.read_text(encoding="utf-8"))["outcome"],
                "pc1_pc2_registration_failed_recovered",
            )

    def test_new_modules_do_not_import_frozen_inspect(self) -> None:
        files = [
            ROOT / "modules/module_2_left_inspect_apart/standalone_inspect.py",
            ROOT / "modules/module_2_left_inspect_apart/run_inspect_then_place.py",
            ROOT / "modules/module_3_left_place/standalone_left_place.py",
        ]
        forbidden = "modules." + "module_2_left_inspect"
        for path in files:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(f"from {forbidden}", source)
            self.assertNotIn(f"import {forbidden}", source)


if __name__ == "__main__":
    unittest.main()
