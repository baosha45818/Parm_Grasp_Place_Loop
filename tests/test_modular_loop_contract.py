#!/usr/bin/env python3
"""验证模块化总控的入口映射，不连接机械臂或相机。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"
OLD_LOOP = ROOT / "tests/loop/simple_dual_arm_loop.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import modular_dual_arm_loop
from modules.module_4_head_judge import ObjectOrientation
from modules.module_5_left_pick import run_head_judge_then_pick


class ModularLoopContractTest(unittest.TestCase):
    def test_graspnet_visualization_flag_reaches_version7_environment(self) -> None:
        argv = ["modular_dual_arm_loop_noserver.py", "--left-only", "--vis"]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {}, clear=False
        ):
            modular_dual_arm_loop.extract_graspnet_visualization_argument()
            self.assertNotIn("--vis", sys.argv)
            self.assertIn("--left-only", sys.argv)
            self.assertEqual(os.environ["GRASPNET_VISUALIZATION"], "1")
            self.assertEqual(
                os.environ["GRASPNET_AUTO_CONFIRM_REAL_GRASP"], "0"
            )

    def test_help_does_not_require_workpiece_selector(self) -> None:
        argv = ["modular_dual_arm_loop.py", "--help"]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("PARM_LOOP_WORKPIECE", None)
            modular_dual_arm_loop.extract_workpiece_argument()
            self.assertEqual(os.environ["PARM_LOOP_WORKPIECE"], "1")

    def test_unreported_worker_exit_is_detected(self) -> None:
        scheduler = modular_dual_arm_loop.load_scheduler()

        class ProcessState:
            def __init__(self, alive: bool, exitcode: int | None):
                self._alive = alive
                self.exitcode = exitcode

            def is_alive(self) -> bool:
                return self._alive

        left_done = SimpleNamespace(is_set=lambda: False)
        alive = ProcessState(True, None)
        crashed = ProcessState(False, -9)
        self.assertIn(
            "left worker 非正常退出",
            scheduler.unexpected_worker_exit(crashed, alive, left_done),
        )
        self.assertIn(
            "right worker 提前退出",
            scheduler.unexpected_worker_exit(alive, crashed, left_done),
        )

    def test_bounded_left_worker_normal_exit_is_allowed(self) -> None:
        scheduler = modular_dual_arm_loop.load_scheduler()
        left = SimpleNamespace(is_alive=lambda: False, exitcode=0)
        right = SimpleNamespace(is_alive=lambda: True, exitcode=None)
        left_done = SimpleNamespace(is_set=lambda: True)
        self.assertIsNone(
            scheduler.unexpected_worker_exit(left, right, left_done)
        )

    def test_scheduler_does_not_reference_frozen_inspect(self) -> None:
        source = OLD_LOOP.read_text(encoding="utf-8")
        self.assertNotIn("modules/module_2_left_inspect/", source)
        self.assertEqual(set(modular_dual_arm_loop.load_scheduler().LEFT_FLOW_SCRIPTS), {"apart"})

    def test_all_scheduler_business_entries_are_in_modules(self) -> None:
        scheduler = modular_dual_arm_loop.load_scheduler()
        modular_dual_arm_loop.configure_module_entries(scheduler)

        entries = [
            scheduler.GRASPNET_FLOW,
            *scheduler.LEFT_FLOW_SCRIPTS.values(),
            scheduler.LEFT_PICK,
            scheduler.LEFT_HANDOFF_PREP,
            scheduler.RIGHT_PUSH_TO_Q1,
            scheduler.RIGHT_LEFT_HANDOFF,
            scheduler.RIGHT_PLACE_WIGGLE,
            scheduler.RIGHT_FINISH,
        ]
        for entry in entries:
            path = Path(entry).resolve()
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.is_relative_to(MODULES.resolve()), path)

        self.assertEqual(
            Path(scheduler.GRASPNET_FLOW_WORKDIR).resolve(),
            (MODULES / "module_1_left_grasp/module_1_3_icp_grasp").resolve(),
        )

        self.assertEqual(
            scheduler.LEFT_FLOW_SCRIPTS,
            {
                "apart": (
                    MODULES
                    / "module_2_left_inspect_apart/run_inspect_then_place.py"
                )
            },
        )

    def test_fixed_arguments_select_apart_and_module_idle_pose(self) -> None:
        argv = ["modular_dual_arm_loop.py", "--planner-host", "127.0.0.1"]
        with mock.patch.object(sys, "argv", argv):
            modular_dual_arm_loop.inject_fixed_module_arguments()
            self.assertEqual(sys.argv.count("--left-flow"), 1)
            self.assertEqual(
                sys.argv[sys.argv.index("--left-flow") + 1],
                "apart",
            )
            self.assertEqual(sys.argv.count("--right-idle-pose"), 1)
            idle_pose = Path(
                sys.argv[sys.argv.index("--right-idle-pose") + 1]
            )
            self.assertEqual(
                idle_pose.resolve(),
                modular_dual_arm_loop.RIGHT_IDLE_POSE.resolve(),
            )

    def test_reverse_pick_uses_module_4_1_then_executes_prediction(
        self,
    ) -> None:
        capture_dir = Path("/tmp/module_4_capture")
        result = SimpleNamespace(
            orientation=ObjectOrientation.BACK,
            capture_dir=capture_dir,
            objects=(object(),),
            report_path=capture_dir / "sam2_masks/tabletop_orientation_result.json",
        )
        argv = [
            "run_head_judge_then_pick.py",
            "--execute",
            "--planner-host",
            "planner.test",
            "--arm-host",
            "arm.test",
        ]
        def fake_run(command, **_kwargs):
            if Path(command[1]).resolve() == run_head_judge_then_pick.RGBD_CAPTURE:
                capture_root = Path(command[command.index("--output-dir") + 1])
                generated_capture_dir = capture_root / "head_rgbd_mock"
                generated_capture_dir.mkdir(parents=True, exist_ok=True)
                (generated_capture_dir / "metadata.json").write_text(
                    "{}", encoding="utf-8"
                )
                result.capture_dir = generated_capture_dir
                result.report_path = (
                    generated_capture_dir
                    / "sam2_masks/tabletop_orientation_result.json"
                )
            if Path(command[1]).resolve() == run_head_judge_then_pick.REVERSE_PREDICTION:
                prediction_file = Path(command[command.index("--prediction-file") + 1])
                prediction_file.write_text(
                    json.dumps({"capture_dir": str(result.capture_dir)}), encoding="utf-8"
                )

        with mock.patch.object(sys, "argv", argv), mock.patch(
            "modules.module_4_head_judge.inspect_tabletop_orientation",
            return_value=result,
        ) as judge, mock.patch("subprocess.run", side_effect=fake_run) as run:
            self.assertEqual(run_head_judge_then_pick.main(), 0)

        self.assertEqual(judge.call_args.args, (result.capture_dir,))
        self.assertEqual(len(run.call_args_list), 4)
        head_command = run.call_args_list[0].args[0]
        capture_command = run.call_args_list[1].args[0]
        predictor_command = run.call_args_list[2].args[0]
        executor_command = run.call_args_list[3].args[0]
        self.assertEqual(Path(head_command[1]).resolve(), run_head_judge_then_pick.HEAD_MOTION)
        self.assertEqual(Path(capture_command[1]).resolve(), run_head_judge_then_pick.RGBD_CAPTURE)
        self.assertEqual(Path(predictor_command[1]).resolve(), run_head_judge_then_pick.REVERSE_PREDICTION)
        self.assertEqual(Path(executor_command[1]).resolve(), run_head_judge_then_pick.LEFT_PICK_EXECUTION)
        self.assertEqual(executor_command[executor_command.index("--planner-host") + 1], "planner.test")

    def test_reverse_pick_retries_zero_candidates_before_execution(self) -> None:
        result = SimpleNamespace(
            orientation=ObjectOrientation.BACK,
            capture_dir=Path("/tmp/unused"),
            objects=(object(),),
            report_path=Path("/tmp/unused/report.json"),
        )
        argv = [
            "run_head_judge_then_pick.py", "--execute",
            "--prediction-retry-seconds", "0",
            "--max-prediction-attempts", "2",
        ]
        prediction_attempts = 0

        def fake_run(command, **_kwargs):
            nonlocal prediction_attempts
            script = Path(command[1]).resolve()
            if script == run_head_judge_then_pick.RGBD_CAPTURE:
                root = Path(command[command.index("--output-dir") + 1])
                capture = root / "head_rgbd_mock"
                capture.mkdir(parents=True, exist_ok=True)
                (capture / "metadata.json").write_text("{}", encoding="utf-8")
            if script == run_head_judge_then_pick.REVERSE_PREDICTION:
                prediction_attempts += 1
                capture = Path(command[command.index("--capture-dir") + 1])
                grasp_json = capture / "sam2_masks/grasp_poses.json"
                grasp_json.parent.mkdir(parents=True, exist_ok=True)
                if prediction_attempts == 1:
                    grasp_json.write_text("[]", encoding="utf-8")
                    raise subprocess.CalledProcessError(1, command)
                grasp_json.write_text("[{}]", encoding="utf-8")
                prediction = Path(command[command.index("--prediction-file") + 1])
                prediction.write_text(
                    json.dumps({"capture_dir": str(capture)}), encoding="utf-8"
                )
                result.capture_dir = capture

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            sys, "argv", argv
        ), mock.patch.object(
            run_head_judge_then_pick, "OUTPUT_ROOT", Path(temporary)
        ), mock.patch(
            "modules.module_4_head_judge.inspect_tabletop_orientation",
            return_value=result,
        ), mock.patch("subprocess.run", side_effect=fake_run) as run:
            self.assertEqual(run_head_judge_then_pick.main(), 0)

        capture_calls = [
            call for call in run.call_args_list
            if Path(call.args[0][1]).resolve() == run_head_judge_then_pick.RGBD_CAPTURE
        ]
        execution_calls = [
            call for call in run.call_args_list
            if Path(call.args[0][1]).resolve() == run_head_judge_then_pick.LEFT_PICK_EXECUTION
        ]
        self.assertEqual(prediction_attempts, 2)
        self.assertEqual(len(capture_calls), 2)
        self.assertEqual(len(execution_calls), 1)

    def test_right_pipeline_uses_replaced_module_paths(self) -> None:
        pipeline = (
            MODULES
            / "module_8_right_grasp/module_8_1_buffer_grasp/workpiece_pipeline.py"
        )
        source = pipeline.read_text(encoding="utf-8")
        self.assertTrue(pipeline.is_file())
        self.assertIn(
            "modules/module_4_head_judge/module_4_2_pose_estimation", source
        )
        self.assertIn("modules/module_7_right_align", source)
        self.assertFalse(
            (MODULES / "module_8_right_grasp/run_align_then_buffer_grasp.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
