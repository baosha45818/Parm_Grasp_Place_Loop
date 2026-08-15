#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_0_prepare import ensure_sam2_service as sam2
from modules.module_8_right_grasp.module_8_2_left_handoff import gripper_compat


class _FakeGripper:
    last_move = None
    state = {
        1: {
            "initialized": False,
            "motion_state": "arrived",
            "position": 100.0,
        }
    }

    def __init__(self, *, host: str, status_port: int = 7702):
        self.host = host
        self.status_port = status_port

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_state(self):
        return self.state

    def move(self, hand, position, **_kwargs):
        type(self).last_move = {
            "hand": hand,
            "position": position,
            **_kwargs,
        }
        self.state[hand] = {
            "initialized": False,
            "motion_state": "arrived",
            "position": position,
        }
        return {"status": "success", "data": {"position": position}}


class RuntimeServiceRecoveryTest(unittest.TestCase):
    def test_module_8_accepts_schema3_gripper_without_initialized_flag(self):
        fake_sdk = SimpleNamespace(Gripper=_FakeGripper)
        with mock.patch.dict(sys.modules, {"parm_sdk": fake_sdk}):
            state = gripper_compat.read_gripper_state("127.0.0.1", 1, "left")
            response = gripper_compat.move_gripper(
                "127.0.0.1", 1, 80.0, 60.0, "left"
            )
        self.assertFalse(state["initialized"])
        self.assertEqual(response["status"], "success")

    def test_module_8_left_release_preempts_active_grip(self):
        _FakeGripper.state[1] = {
            "initialized": False,
            "motion_state": "gripping",
            "position": 30.0,
        }
        _FakeGripper.last_move = None
        fake_sdk = SimpleNamespace(Gripper=_FakeGripper)
        with mock.patch.dict(sys.modules, {"parm_sdk": fake_sdk}):
            response = gripper_compat.release_left_gripper(
                "127.0.0.1", 100.0, 60.0
            )
        self.assertEqual(response["status"], "success")
        self.assertEqual(_FakeGripper.last_move["mode"], "async")
        self.assertTrue(_FakeGripper.last_move["preempt"])

    def test_require_sam2_recovers_missing_socket(self):
        recovered = {"service": "sam2-service", "started": True}
        with mock.patch.dict(
            "os.environ", {"PARM_SAM2_AUTO_RECOVER": "1"}
        ), mock.patch.object(
            sam2, "_health", side_effect=FileNotFoundError("missing")
        ), mock.patch.object(
            sam2, "ensure_sam2_service", return_value=recovered
        ) as ensure:
            self.assertEqual(sam2.require_sam2_service(), recovered)
        ensure.assert_called_once_with()

    def test_require_sam2_noserver_mode_does_not_start_process(self):
        with mock.patch.dict(
            "os.environ", {"PARM_SAM2_AUTO_RECOVER": "0"}
        ), mock.patch.object(
            sam2, "_health", side_effect=FileNotFoundError("missing")
        ), mock.patch.object(sam2, "ensure_sam2_service") as ensure:
            with self.assertRaises(FileNotFoundError):
                sam2.require_sam2_service()
        ensure.assert_not_called()

    def test_require_sam2_does_not_restart_healthy_service(self):
        healthy = {"service": "sam2-service", "started": True}
        with mock.patch.object(sam2, "_health", return_value=healthy), mock.patch.object(
            sam2, "ensure_sam2_service"
        ) as ensure:
            self.assertEqual(sam2.require_sam2_service(), healthy)
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
