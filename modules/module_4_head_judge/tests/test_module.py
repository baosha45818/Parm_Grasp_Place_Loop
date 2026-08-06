"""验证 Head Judge 的公开状态映射，不连接相机或机器人。"""

import unittest

from modules.module_4_head_judge import ObjectOrientation
from modules.module_4_head_judge.module import _to_object_result


class OrientationMappingTests(unittest.TestCase):
    def test_positive_and_reverse_mapping(self) -> None:
        front = _to_object_result({
            "object_key": "a",
            "classification": "positive",
        })
        back = _to_object_result({
            "object_key": "b",
            "classification": "reverse",
        })
        self.assertIs(front.orientation, ObjectOrientation.FRONT)
        self.assertIs(back.orientation, ObjectOrientation.BACK)

    def test_ambiguous_result_exposes_recovery_contract(self) -> None:
        result = _to_object_result({
            "object_key": "a",
            "classification": "unknown",
        })
        self.assertIs(result.orientation, ObjectOrientation.OTHER)
        self.assertTrue(result.tilt_recovery_candidate)

if __name__ == "__main__":
    unittest.main()
