from __future__ import annotations

import unittest

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

    def test_first_phase_accepts_any_current_start_and_reaches_middle(self):
        result = analyze_target_response(
            self.response([[0.0] * 7, START_JOINTS_DEG]),
            START_JOINTS_DEG,
            0.5,
        )
        self.assertEqual(result["goal_error_deg"], 0.0)

    def test_head_feedback_accounts_for_sdk_pitch_sign(self):
        state = {
            "pitch": {"position": -HEAD_TARGET_DEG["pitch"]},
            "yaw": {"position": HEAD_TARGET_DEG["yaw"]},
        }
        self.assertTrue(head_target_reached(state))
        state["yaw"]["position"] = 0.0
        self.assertFalse(head_target_reached(state))


if __name__ == "__main__":
    unittest.main()
