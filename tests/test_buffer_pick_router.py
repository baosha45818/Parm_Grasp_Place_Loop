from __future__ import annotations

import unittest

from modules.module_5_left_pick.buffer_pick_router import (
    BACK_TO_HANDOFF,
    FRONT_TO_INTERMEDIATE,
    OTHER,
    route_module4_result,
)


class BufferPickRouterTests(unittest.TestCase):
    def test_front_routes_to_intermediate(self) -> None:
        branch, _ = route_module4_result({
            "decision": "FRONT",
            "objects": [{
                "orientation": "FRONT",
                "diagnostics": {"positive": {"T_camera_object": [[1]]}},
            }],
        })
        self.assertEqual(branch, FRONT_TO_INTERMEDIATE)

    def test_workpiece2_back_routes_to_handoff_from_module4_pose(self) -> None:
        branch, _ = route_module4_result({
            "decision": "BACK",
            "objects": [{
                "orientation": "BACK",
                "diagnostics": {
                    "pose": {"T_head_camera_workpiece": [[1]]},
                },
            }],
        })
        self.assertEqual(branch, BACK_TO_HANDOFF)

    def test_invalid_report_never_selects_a_grasp(self) -> None:
        branch, _ = route_module4_result({"decision": "UNCERTAIN", "objects": []})
        self.assertEqual(branch, OTHER)


if __name__ == "__main__":
    unittest.main()
