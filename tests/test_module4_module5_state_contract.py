"""Module 4-1 正反+位姿报告到 Module 5-1 的纯文件接口测试。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.workpiece_1.execute_prediction import (
    load_prediction,
)
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.workpiece_2.execute_convex_tab_grasp import (
    validate_pose_quality,
)
from modules.module_5_left_pick.reverse_pick_to_handoff.grasp.workpiece_2.predict_convex_tab_grasp import (
    pose_from_module4_result,
)


class Module4ToModule5ContractTests(unittest.TestCase):
    def test_workpiece_1_executor_accepts_new_buffer_state_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            masks = capture / "sam2_masks"
            masks.mkdir(parents=True)
            (capture / "metadata.json").write_text("{}", encoding="utf-8")
            diagnostics = masks / "orientation_diagnostics.json"
            diagnostics.write_text("[]", encoding="utf-8")
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            report = root / "state.json"
            report.write_text(json.dumps({
                "event": "module_4_1_buffer_state_estimated",
                "workpiece_id": "1",
                "decision": "BACK",
                "objects": [{}],
                "capture_dir": str(capture),
                "orientation_diagnostics": str(diagnostics),
                "body_head_calibration": str(calibration),
            }), encoding="utf-8")
            _, loaded_capture, loaded_diagnostics, loaded_calibration = load_prediction(report)
            self.assertEqual(loaded_capture, capture.resolve())
            self.assertEqual(loaded_diagnostics, diagnostics.resolve())
            self.assertEqual(loaded_calibration, calibration.resolve())

    def test_workpiece_2_adapter_extracts_the_same_icp_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            transform = np.eye(4)
            transform[:3, 3] = [0.1, -0.2, 0.6]
            report = root / "state.json"
            report.write_text(json.dumps({
                "event": "module_4_1_buffer_state_estimated",
                "workpiece_id": "2",
                "decision": "BACK",
                "capture_dir": str(capture),
                "objects": [{"diagnostics": {
                    "classification": "reverse",
                    "table_normal_camera": [0.0, 0.0, -1.0],
                    "pose": {
                        "valid": True,
                        "orientation_valid": True,
                        "T_head_camera_workpiece": transform.tolist(),
                    },
                    "icp": {"fitness": 0.95, "rmse_mm": 4.0, "depth_score_mm": 5.0},
                    "pose_confidence": {"recommended_for_grasp": True},
                }}],
            }), encoding="utf-8")
            pose, loaded_capture = pose_from_module4_result(report)
            np.testing.assert_array_equal(pose["T_camera_object"], transform)
            self.assertEqual(loaded_capture, capture.resolve())
            self.assertTrue(validate_pose_quality(pose)["accepted"])


if __name__ == "__main__":
    unittest.main()
