"""Regression tests for Module 0's non-actuating infrastructure preflight."""

from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.module_0_prepare import preflight_robot_infrastructure as subject


class _Planner:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def list_frame_ids(self, **_kwargs):
        return [
            "body_base",
            "left_eef",
            "left_hand_transfer_end",
            "right_hand_transfer_end",
        ]


class _Arm:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def get_state(self):
        return {
            "system_state": "running",
            "estop_reason": None,
            "state": {
                "left": {"position": [0.0] * 7, "enabled": True, "task_state": "idle"},
                "right": {"position": [0.0] * 7, "enabled": True, "task_state": "idle"},
            },
        }


class _Gripper:
    init_kwargs: list[dict] = []

    def __init__(self, **_kwargs):
        type(self).init_kwargs.append(_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def get_state(self):
        return {
            1: {"initialized": True, "motion_state": "arrived", "position": 100.0},
            2: {"initialized": True, "motion_state": "arrived", "position": 100.0},
        }


class RobotInfrastructurePreflightTests(unittest.TestCase):
    def setUp(self):
        _Gripper.init_kwargs.clear()

    def test_missing_left_eef_is_a_hard_preflight_failure(self):
        class PlannerWithoutLeftEef(_Planner):
            def list_frame_ids(self, **_kwargs):
                return [
                    "body_base",
                    "left_hand_transfer_end",
                    "right_hand_transfer_end",
                ]

        fake_sdk = SimpleNamespace(
            Planner=PlannerWithoutLeftEef, Arm=_Arm, Gripper=_Gripper
        )
        with patch.dict(sys.modules, {"parm_sdk": fake_sdk}):
            with self.assertRaisesRegex(RuntimeError, "left_eef"):
                subject.preflight_robot_infrastructure(
                    planner_host="planner",
                    arm_host="arm",
                    gripper_host="gripper",
                )

    def test_success_reads_only_required_state_contracts(self):
        fake_sdk = SimpleNamespace(Planner=_Planner, Arm=_Arm, Gripper=_Gripper)
        force = {"force_n": [0.0, 0.0, 9.8], "torque_nm": [0.0, 0.0, 0.0]}
        with patch.dict(sys.modules, {"parm_sdk": fake_sdk}), \
             patch.object(subject, "_check_execute_endpoint"), \
             patch.object(subject, "_check_tcp_endpoint") as check_tcp, \
             patch.object(subject, "_sdk_version", return_value="test-sdk"), \
             patch.object(subject, "_read_force_sensor", return_value=force) as read_force:
            report = subject.preflight_robot_infrastructure(
                planner_host="planner", arm_host="arm", gripper_host="gripper"
            )

        self.assertEqual(report["sdk_version"], "test-sdk")
        self.assertEqual(report["planner"]["frame_count"], 4)
        self.assertEqual(
            report["planner"]["graspnet_ports"],
            (5604, 5609, 5610),
        )
        self.assertEqual(
            [call.args[1] for call in check_tcp.call_args_list],
            [5604, 5609, 5610],
        )
        self.assertEqual(read_force.call_count, 2)
        self.assertEqual(_Gripper.init_kwargs, [{"host": "gripper", "status_port": 7702}])

    def test_left_only_does_not_require_right_side_state(self):
        class LeftOnlyPlanner(_Planner):
            def list_frame_ids(self, **_kwargs):
                return ["body_base", "left_eef", "left_hand_transfer_end"]

        class LeftOnlyArm(_Arm):
            def get_state(self):
                return {
                    "system_state": "running",
                    "estop_reason": None,
                    "state": {
                        "left": {
                            "position": [0.0] * 7,
                            "enabled": True,
                            "task_state": "idle",
                        },
                        # Deliberately malformed: left-only preflight must not
                        # inspect or reject the inactive right arm.
                        "right": {"position": None},
                    },
                }

        class LeftOnlyGripper(_Gripper):
            def get_state(self):
                return {
                    1: {
                        "initialized": True,
                        "motion_state": "arrived",
                        "position": 100.0,
                    },
                    2: {"motion_state": "dropped"},
                }

        fake_sdk = SimpleNamespace(
            Planner=LeftOnlyPlanner, Arm=LeftOnlyArm, Gripper=LeftOnlyGripper
        )
        force = {"force_n": [0.0, 0.0, 9.8], "torque_nm": [0.0, 0.0, 0.0]}
        with patch.dict(sys.modules, {"parm_sdk": fake_sdk}), \
             patch.object(subject, "_check_execute_endpoint"), \
             patch.object(subject, "_check_tcp_endpoint"), \
             patch.object(subject, "_read_force_sensor", return_value=force) as read_force:
            report = subject.preflight_robot_infrastructure(
                planner_host="planner",
                arm_host="arm",
                gripper_host="gripper",
                force_sensor_host="gripper",
                left_only=True,
            )

        self.assertIn("left", report["arm"])
        self.assertNotIn("right", report["arm"])
        self.assertIn("left", report["gripper"])
        self.assertNotIn("right", report["gripper"])
        self.assertEqual(set(report["force_sensors"]), {"host", "left"})
        read_force.assert_called_once_with("gripper", 1)
        self.assertNotIn(
            "right_hand_transfer_end", report["planner"]["required_frames"]
        )

    def test_estop_system_state_is_a_hard_preflight_failure(self):
        with self.assertRaisesRegex(RuntimeError, "急停锁存"):
            subject._validate_arm_state({
                "system_state": "estop",
                "estop_reason": "HEARTBEAT_TIMEOUT",
            })

    def test_running_state_with_stale_estop_reason_is_allowed(self):
        summary = subject._validate_arm_state({
            "system_state": "RUNNING",
            "estop_reason": "HEARTBEAT_TIMEOUT",
            "state": {
                "left": {"position": [0.0] * 7, "enabled": True, "task_state": "idle"},
                "right": {"position": [0.0] * 7, "enabled": True, "task_state": "idle"},
            },
        })
        self.assertEqual(summary["system_state"], "RUNNING")

    def test_new_driver_arrived_state_is_usable_after_init(self):
        summary = subject._validate_gripper_state({
            1: {"initialized": False, "motion_state": "arrived", "position": 100.0},
            2: {"initialized": True, "motion_state": "arrived", "position": 100.0},
        })
        self.assertFalse(summary["left"]["initialized"])
        self.assertEqual(summary["left"]["motion_state"], "arrived")


if __name__ == "__main__":
    unittest.main()
