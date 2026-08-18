"""Module 4-1 双现场相机/标定原子绑定测试。"""

import os
import json
import unittest

from modules.module_4_head_judge.module_4_1_buffer_state_estimation.deployment_modes import (
    apply_mode_environment,
    load_mode,
)


class DeploymentModeTests(unittest.TestCase):
    def test_modes_bind_distinct_camera_and_calibration(self) -> None:
        xiangxin = load_mode("xiangxin")
        gzhu = load_mode("gzhu")
        self.assertNotEqual(xiangxin.camera_serial, gzhu.camera_serial)
        self.assertNotEqual(xiangxin.calibration_path, gzhu.calibration_path)
        self.assertEqual(xiangxin.camera_serial, "261722072445")
        self.assertEqual(gzhu.camera_serial, "261222076212")

    def test_workpiece_capture_profiles_are_published_to_children(self) -> None:
        expected = {
            ("xiangxin", "1"): ("848", "480"),
            ("xiangxin", "2"): ("1280", "720"),
            ("gzhu", "1"): ("1280", "720"),
            ("gzhu", "2"): ("1280", "720"),
        }
        for (mode_name, workpiece), size in expected.items():
            mode = load_mode(mode_name)
            apply_mode_environment(mode, workpiece)
            self.assertEqual(os.environ["PARM_HEAD_D435_SERIAL"], mode.camera_serial)
            self.assertEqual(
                (os.environ["PARM_HEAD_D435_WIDTH"], os.environ["PARM_HEAD_D435_HEIGHT"]),
                size,
            )
            self.assertEqual(
                os.environ["PARM_BODY_HEAD_CALIBRATION"], str(mode.calibration_path)
            )

    def test_each_mode_and_workpiece_has_its_own_workspace(self) -> None:
        for workpiece in ("1", "2"):
            xiangxin = load_mode("xiangxin").workspace_path(workpiece)
            gzhu = load_mode("gzhu").workspace_path(workpiece)
            self.assertNotEqual(xiangxin, gzhu)
            self.assertIn("xiangxin", xiangxin.name)
            self.assertIn("gzhu", gzhu.name)
            xiangxin_roi = json.loads(xiangxin.read_text())["safe_roi_polygon_px"]
            gzhu_roi = json.loads(gzhu.read_text())["safe_roi_polygon_px"]
            self.assertNotEqual(xiangxin_roi, gzhu_roi)
            self.assertEqual(gzhu_roi, [[0, 180], [760, 180], [760, 720], [0, 720]])


if __name__ == "__main__":
    unittest.main()
