from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.reversed_head_observation.check_plan import (
    GOAL_JOINTS_DEG,
    HEAD_TARGET_DEG,
    START_JOINTS_DEG,
    analyze_response,
    find_joint_limit_violations,
    joint_waypoint,
)
from tests.reversed_head_observation.execute_plan import (
    TaskBusyError,
    analyze_target_response,
    build_execute_payload,
    confirmed,
    execute_trajectory,
    head_target_reached,
    left_gripper_holds_object,
    open_left_gripper_after_failed_hold,
    reverse_executed_phase,
)


class PlanValidationTests(unittest.TestCase):
    def response(self, trajectory):
        return {
            "success": True,
            "message": "ok",
            "data": {
                "arm_id": 1,
                "trajectory": trajectory,
                "sample_frequency": 10.0,
            },
        }

    def test_waypoint_uses_left_arm_degrees(self):
        waypoint = joint_waypoint("start", START_JOINTS_DEG)
        self.assertEqual(waypoint["group"], "left_arm")
        self.assertEqual(waypoint["joint_values"], START_JOINTS_DEG)

    def test_accepts_trajectory_through_start_to_goal(self):
        result = analyze_response(
            self.response([[0.0] * 7, START_JOINTS_DEG, GOAL_JOINTS_DEG])
        )
        self.assertEqual(result["tested_segment_start_index"], 1)
        self.assertEqual(result["tested_segment_points"], 2)

    def test_rejects_missing_recorded_start(self):
        with self.assertRaisesRegex(RuntimeError, "没有经过记录起点"):
            analyze_response(self.response([[0.0] * 7, GOAL_JOINTS_DEG]))

    def test_rejects_wrong_goal(self):
        with self.assertRaisesRegex(RuntimeError, "未到达观察位"):
            analyze_response(self.response([START_JOINTS_DEG, [0.0] * 7]))

    def test_rejects_failed_planner_response(self):
        with self.assertRaisesRegex(RuntimeError, "规划器报告失败"):
            analyze_response({"success": False, "message": "collision"})

    def test_current_recorded_poses_are_within_joint_limits(self):
        self.assertEqual(find_joint_limit_violations(), [])

    def test_builds_direct_sdk_execute_payload(self):
        response = self.response([START_JOINTS_DEG, GOAL_JOINTS_DEG])
        payload = build_execute_payload(response)
        self.assertEqual(payload["arm_id"], 1)
        self.assertEqual(payload["sample_frequency"], 10.0)
        self.assertEqual(payload["trajectory"][-1], GOAL_JOINTS_DEG)

    def test_confirmation_accepts_y(self):
        self.assertTrue(confirmed("y"))
        self.assertTrue(confirmed("Y"))
        self.assertFalse(confirmed("EXECUTE"))

    def test_task_busy_error_is_runtime_error(self):
        self.assertTrue(issubclass(TaskBusyError, RuntimeError))

    def test_observation_hold_gate_accepts_settled_or_active_grasp(self):
        self.assertTrue(
            left_gripper_holds_object(
                {"motion_state": "gripping", "position": 20.1}, 20.0
            )
        )
        self.assertTrue(
            left_gripper_holds_object(
                {"motion_state": "arrived", "position": 55.0}, 20.0
            )
        )
        self.assertFalse(
            left_gripper_holds_object(
                {"motion_state": "gripping", "position": 20.0}, 20.0
            )
        )
        self.assertFalse(left_gripper_holds_object(None, 20.0))

    def test_failed_observation_replays_exact_trajectory_in_reverse(self):
        payload = {
            "arm_id": 1,
            "trajectory": [[0.0] * 7, [1.0] * 7, [2.0] * 7],
            "sample_frequency": 10.0,
        }
        with patch(
            "tests.reversed_head_observation.execute_plan.execute_trajectory"
        ) as execute, patch(
            "tests.reversed_head_observation.execute_plan.wait_for_left_joint_target"
        ) as wait:
            reverse_executed_phase(
                "arm", payload, tolerance_deg=0.5, label="test"
            )

        reversed_payload = execute.call_args.args[1]
        self.assertEqual(
            reversed_payload["trajectory"],
            [[2.0] * 7, [1.0] * 7, [0.0] * 7],
        )
        wait.assert_called_once_with("arm", [0.0] * 7, tolerance_deg=0.5)
        self.assertEqual(payload["trajectory"][0], [0.0] * 7)

    def test_failed_observation_opens_gripper_after_rollback(self):
        class FakeGripper:
            calls = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def move(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {
                    "status": "success",
                    "data": {
                        "devices": {
                            "left_gripper": {"device_state": "arrived"}
                        }
                    },
                }

        with patch("parm_sdk.Gripper", FakeGripper):
            response = open_left_gripper_after_failed_hold("gripper")

        self.assertEqual(response["status"], "success")
        args, kwargs = FakeGripper.calls[-1]
        self.assertEqual(args, (1, 100.0))
        self.assertEqual(kwargs["mode"], "sync")

    def test_first_phase_accepts_any_current_start_and_reaches_middle(self):
        result = analyze_target_response(
            self.response([[0.0] * 7, START_JOINTS_DEG]),
            START_JOINTS_DEG,
            0.5,
        )
        self.assertEqual(result["goal_error_deg"], 0.0)

    def test_head_feedback_uses_current_sdk_pitch_sign(self):
        state = {
            "pitch": {"position": HEAD_TARGET_DEG["pitch"]},
            "yaw": {"position": HEAD_TARGET_DEG["yaw"]},
        }
        self.assertTrue(head_target_reached(state))
        state["yaw"]["position"] = 0.0
        self.assertFalse(head_target_reached(state))


if __name__ == "__main__":
    unittest.main()
