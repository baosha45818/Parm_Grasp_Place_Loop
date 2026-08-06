from __future__ import annotations

import unittest

import numpy as np

from tests.reversed_head_observation.auto_head_d435_mask import automatic_prompts


class AutomaticPromptTests(unittest.TestCase):
    def test_normalized_prompts_scale_to_capture_size(self):
        image = np.zeros((480, 848, 3), dtype=np.uint8)
        config = {
            "roi_xyxy_normalized": [0.0, 0.0, 1.0, 1.0],
            "positive_points_normalized": [[0.5, 0.5]],
            "negative_points_normalized": [[1.0, 0.0]],
        }
        box, positives, negatives = automatic_prompts(image, config)
        self.assertEqual(box, [0, 0, 847, 479])
        self.assertEqual(positives, [[424, 240]])
        self.assertEqual(negatives, [[847, 0]])
