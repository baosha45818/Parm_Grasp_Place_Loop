import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "modules/module_5_left_pick/front_buffer_regrasp.py"
SPEC = importlib.util.spec_from_file_location("front_buffer_regrasp", MODULE_PATH)
front_regrasp = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(front_regrasp)


class FrontRegraspGeometryTests(unittest.TestCase):
    def test_default_independent_vertical_targets(self):
        icp = np.eye(4)
        icp[:3, 3] = [0.4, -0.1, -0.3]
        pregrasp, grasp, lift = front_regrasp._vertical_regrasp_targets(
            icp,
            pregrasp_height_m=front_regrasp.DEFAULT_PREGRASP_HEIGHT_M,
            descent_distance_m=front_regrasp.DEFAULT_DESCENT_DISTANCE_M,
        )
        np.testing.assert_allclose(pregrasp[:3, 3], [0.4, -0.1, -0.25])
        np.testing.assert_allclose(grasp[:3, 3], [0.4, -0.1, -0.27])
        np.testing.assert_allclose(lift[:3, 3], [0.4, -0.1, -0.24])
        self.assertAlmostEqual(grasp[2, 3] - icp[2, 3], 0.03)
        self.assertAlmostEqual(lift[2, 3] - icp[2, 3], 0.06)


if __name__ == "__main__":
    unittest.main()
