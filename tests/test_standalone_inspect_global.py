"""Contract tests for the isolated global Module 2 inspect entry."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

from modules.module_2_left_inspect_apart import standalone_inspect_global as global_inspect


ROOT = Path(__file__).resolve().parents[1]


class StandaloneInspectGlobalTest(unittest.TestCase):
    def parse(self, *extra: str):
        argv = [
            "standalone_inspect_global.py",
            "--workpiece", "1",
            "--result-file", "/tmp/global-inspect-result.json",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return global_inspect.parse_args()

    def test_global_defaults_are_independent_of_workpiece(self) -> None:
        args = self.parse()
        self.assertEqual(args.prompt_mode, "auto")
        self.assertEqual(args.urdf_near_black_max_rgb, 35)
        self.assertEqual(args.urdf_self_dilation_kernel_px, 3)
        self.assertEqual(
            args.output_dir,
            ROOT / "output/module_2_left_inspect_apart",
        )
        self.assertEqual(
            args.urdf_description_root,
            Path("/home/dainanjun/Public/plucky_hjt_parm/welding_config/description"),
        )
        self.assertEqual(args.urdf_active_joint_max_rad, 0.8)
        self.assertEqual(args.mask_positive_depth_tolerance_m, 0.050)
        self.assertEqual(
            args.workpiece2_extended_mask_depth_tolerance_m, 0.120
        )

    def test_explicit_manual_mode_is_still_available(self) -> None:
        args = self.parse("--prompt-mode", "manual")
        self.assertEqual(args.prompt_mode, "manual")

    def test_dilation_kernel_must_be_positive_and_odd(self) -> None:
        args = self.parse("--urdf-self-dilation-kernel-px", "4")
        with self.assertRaisesRegex(ValueError, "正奇数"):
            global_inspect._validate_args(args)

    def test_original_entry_routes_to_global_implementation(self) -> None:
        source = (
            ROOT
            / "modules/module_2_left_inspect_apart/standalone_inspect.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from standalone_inspect_global import main", source)

    def test_adaptive_components_keep_tail_and_reject_robot_material(self) -> None:
        args = self.parse("--auto-positive-count", "3")
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Brown main body and a same-material/depth lobe behind the gripper.
        cv2.rectangle(measured, (500, 300), (620, 420), 0.220, -1)
        cv2.rectangle(bgr, (500, 300), (620, 420), (55, 65, 72), -1)
        cv2.rectangle(measured, (690, 330), (730, 370), 0.222, -1)
        cv2.rectangle(bgr, (690, 330), (730, 370), (58, 67, 74), -1)

        # A closer blue screw has matching depth but dissimilar chroma.
        cv2.rectangle(measured, (640, 250), (675, 285), 0.221, -1)
        cv2.rectangle(bgr, (640, 250), (675, 285), (90, 72, 62), -1)
        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["component_selection"]["candidate_component_count"], 3)
        self.assertEqual(result["component_selection"]["associated_component_count"], 2)
        self.assertEqual(len(result["positives"]), 2)
        self.assertTrue(any(x >= 690 for x, _ in result["positives"]))
        self.assertFalse(any(640 <= x <= 675 and 250 <= y <= 285 for x, y in result["positives"]))

    def test_adaptive_components_cap_spatially_diverse_positives(self) -> None:
        args = self.parse("--auto-positive-count", "2")
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        for x in (450, 520, 590):
            cv2.rectangle(measured, (x, 320), (x + 35, 355), 0.220, -1)
            cv2.rectangle(bgr, (x, 320), (x + 35, 355), (58, 66, 74), -1)
        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["component_selection"]["associated_component_count"], 3)
        self.assertEqual(len(result["positives"]), 2)

    def test_adaptive_components_do_not_chain_from_fragment_to_arm(self) -> None:
        args = self.parse()
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        for x, depth in ((500, 0.220), (650, 0.245), (800, 0.270)):
            cv2.rectangle(measured, (x, 320), (x + 45, 365), depth, -1)
            cv2.rectangle(bgr, (x, 320), (x + 45, 365), (58, 66, 74), -1)
        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["component_selection"]["associated_component_count"], 2)
        self.assertFalse(any(x >= 800 for x, _ in result["positives"]))

    def test_full_urdf_projection_vetoes_reflective_robot_hole(self) -> None:
        args = self.parse("--auto-positive-count", "4")
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = np.full_like(measured, np.nan)

        # Main held part and a separated same-material tail remain outside the
        # full robot projection.
        cv2.rectangle(measured, (500, 300), (620, 420), 0.220, -1)
        cv2.rectangle(bgr, (500, 300), (620, 420), (58, 66, 74), -1)
        cv2.rectangle(measured, (690, 330), (730, 370), 0.222, -1)
        cv2.rectangle(bgr, (690, 330), (730, 370), (58, 67, 74), -1)

        # A reflective robot island survives measured-depth self subtraction,
        # but sits deep inside the complete URDF silhouette.
        cv2.rectangle(measured, (390, 320), (450, 380), 0.221, -1)
        cv2.rectangle(bgr, (390, 320), (450, 380), (58, 66, 74), -1)
        rendered[290:411, 360:481] = 0.230

        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args, rendered
        )
        self.assertIsNotNone(result)
        selection = result["component_selection"]
        self.assertEqual(selection["candidate_component_count"], 3)
        self.assertEqual(selection["associated_component_count"], 2)
        self.assertEqual(len(selection["robot_veto_labels"]), 1)
        self.assertTrue(any(x >= 690 for x, _ in result["positives"]))
        self.assertFalse(any(390 <= x <= 450 for x, _ in result["positives"]))
        self.assertTrue(any(390 <= x <= 450 for x, _ in result["negatives"]))

    def test_thin_urdf_edge_overlap_does_not_delete_tail(self) -> None:
        args = self.parse("--auto-positive-count", "3")
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = np.full_like(measured, np.nan)
        cv2.rectangle(measured, (500, 300), (620, 420), 0.220, -1)
        cv2.rectangle(bgr, (500, 300), (620, 420), (58, 66, 74), -1)
        cv2.rectangle(measured, (690, 330), (730, 370), 0.222, -1)
        cv2.rectangle(bgr, (690, 330), (730, 370), (58, 67, 74), -1)

        # A thin projected finger crosses the tail.  Its pixels never reach
        # the 8 px URDF interior, so this is not a robot-component veto.
        rendered[330:371, 704:714] = 0.300
        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args, rendered
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["component_selection"]["associated_component_count"], 2)
        self.assertTrue(any(x >= 690 for x, _ in result["positives"]))

    def test_close_table_fragment_with_different_depth_becomes_negative(self) -> None:
        args = self.parse(
            "--auto-positive-count", "3",
            "--global-close-fragment-depth-diff-m", "0.026",
        )
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

        cv2.rectangle(measured, (500, 300), (620, 420), 0.220, -1)
        cv2.rectangle(bgr, (500, 300), (620, 420), (58, 66, 74), -1)
        # This same-colour table component is spatially close but lies 31 mm
        # behind the held part's flat depth distribution.
        cv2.rectangle(measured, (625, 360), (665, 400), 0.251, -1)
        cv2.rectangle(bgr, (625, 360), (665, 400), (58, 66, 74), -1)

        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["component_selection"]["associated_component_count"], 1)
        self.assertFalse(any(625 <= x <= 665 for x, _ in result["positives"]))
        self.assertTrue(any(625 <= x <= 665 for x, _ in result["negatives"]))

    def test_post_sam_depth_filter_breaks_far_table_attachment(self) -> None:
        depth = np.zeros((200, 320), dtype=np.uint16)
        mask = np.zeros_like(depth, dtype=np.uint8)
        # Held workpiece, a farther table part, and their visual bridge are a
        # single SAM component before depth filtering.
        cv2.rectangle(mask, (40, 70), (140, 130), 255, -1)
        cv2.rectangle(mask, (180, 70), (280, 130), 255, -1)
        cv2.rectangle(mask, (140, 95), (180, 105), 255, -1)
        depth[70:131, 40:141] = 220
        depth[70:131, 180:281] = 270
        depth[95:106, 140:181] = 270

        filtered, report = global_inspect._filter_mask_by_positive_depth(
            mask, depth, 0.001, [[90, 100]], 0.026, 0.05
        )
        cleaned, cleanup = global_inspect._main_mask_with_filled_holes(
            filtered, [[90, 100]]
        )
        self.assertGreater(np.count_nonzero(cleaned[:, :150]), 5000)
        self.assertEqual(np.count_nonzero(cleaned[:, 170:]), 0)
        self.assertGreater(report["removed_depth_mismatch_pixels"], 5000)
        self.assertEqual(cleanup["retained_component_count"], 1)

    def test_semantic_depth_filter_can_preserve_reflective_missing_depth(self) -> None:
        depth = np.zeros((100, 180), dtype=np.uint16)
        mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (150, 80), 255, -1)
        depth[20:81, 20:81] = 220
        filtered, report = global_inspect._filter_mask_by_positive_depth(
            mask, depth, 0.001, [[50, 50]], 0.026, 0.005,
            preserve_invalid_depth=True,
        )
        self.assertGreater(np.count_nonzero(filtered[:, 80:151]), 4000)
        self.assertGreater(report["retained_invalid_depth_pixels"], 4000)

    def test_silver_hypothesis_seeds_bright_metal(self) -> None:
        args = self.parse()
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(measured, (180, 100), (320, 220), 0.220, -1)
        cv2.rectangle(bgr, (180, 100), (320, 220), (190, 195, 200), -1)
        brown = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args, hypothesis="brown"
        )
        silver = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args, hypothesis="silver"
        )
        self.assertIsNone(brown)
        self.assertIsNotNone(silver)
        self.assertEqual(silver["hypothesis"], "silver")

    def test_bright_hypothesis_requires_stronger_coverage_than_brown(self) -> None:
        support = np.zeros((100, 100), dtype=bool)
        support[10:50, 10:60] = True
        raw = np.zeros((100, 100), dtype=np.uint8)
        # 0.54 support coverage: accepted by brown (0.53), rejected by
        # bright/colour-difference hypotheses (0.55).
        raw[10:50, 10:37] = 255
        self_pixels = np.zeros_like(support)
        brown = global_inspect._automatic_hypothesis_quality(
            {
                "hypothesis": "brown",
                "debug_masks": {
                    "prompt_support": support,
                    "plausible_contact_support": support,
                },
            },
            raw,
            self_pixels,
        )
        silver = global_inspect._automatic_hypothesis_quality(
            {
                "hypothesis": "silver",
                "debug_masks": {
                    "prompt_support": support,
                    "plausible_contact_support": support,
                },
            },
            raw,
            self_pixels,
        )
        self.assertTrue(brown["valid"])
        self.assertFalse(silver["valid"])

    def test_silver_hypothesis_rejects_large_border_touching_table(self) -> None:
        support = np.zeros((100, 100), dtype=bool)
        support[55:100, 0:90] = True
        raw = support.astype(np.uint8) * 255
        quality = global_inspect._automatic_hypothesis_quality(
            {
                "hypothesis": "silver",
                "positive_to_nearest_urdf_depth_gaps_m": [0.184],
                "debug_masks": {
                    "prompt_support": support,
                    "plausible_contact_support": support,
                },
            },
            raw,
            np.zeros_like(support),
        )
        self.assertFalse(quality["valid"])
        self.assertTrue(quality["scene_background_veto"])
        self.assertTrue(quality["oversized_border_background"])
        self.assertTrue(quality["far_behind_urdf_background"])

    def test_object_in_front_of_urdf_is_not_component_vetoed(self) -> None:
        args = self.parse()
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = np.full_like(measured, np.nan)
        cv2.rectangle(measured, (180, 100), (320, 220), 0.200, -1)
        cv2.rectangle(bgr, (180, 100), (320, 220), (58, 66, 74), -1)
        rendered[70:251, 150:351] = 0.250
        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args, rendered, hypothesis="brown"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["component_selection"]["robot_veto_labels"], [])

    def test_semantic_prompt_keeps_object_pixels_inside_self_mask(self) -> None:
        args = self.parse()
        measured = np.zeros((720, 1280), dtype=np.float32)
        self_pixels = np.zeros_like(measured, dtype=bool)
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = np.full_like(measured, np.nan)
        cv2.rectangle(measured, (400, 250), (650, 430), 0.220, -1)
        cv2.rectangle(bgr, (400, 250), (650, 430), (38, 45, 52), -1)
        # The left part is indistinguishable from matched URDF depth.  It is
        # still part of one object island extending outside the projection.
        self_pixels[250:431, 400:526] = True
        rendered[220:461, 370:526] = 0.220
        result = global_inspect._auto_rgbd_prompt(
            measured, self_pixels, bgr, args, rendered, hypothesis="brown"
        )
        self.assertIsNotNone(result)
        support = result["debug_masks"]["prompt_support"]
        self.assertGreater(np.count_nonzero(support & self_pixels), 10000)
        self.assertEqual(result["component_selection"]["robot_veto_labels"], [])

    def test_icp_mask_removes_robot_but_preserves_object_in_front(self) -> None:
        semantic = np.ones((80, 120), dtype=np.uint8) * 255
        measured = np.full((80, 120), 0.220, dtype=np.float32)
        rendered = np.full((80, 120), np.nan, dtype=np.float32)
        self_pixels = np.zeros((80, 120), dtype=bool)
        rendered[:, :40] = 0.221
        self_pixels[:, :40] = True
        rendered[:, 40:80] = 0.260
        self_pixels[:, 40:80] = True
        icp_mask, report = global_inspect._automatic_icp_mask(
            semantic, measured, rendered, self_pixels
        )
        self.assertEqual(np.count_nonzero(icp_mask[:, :40]), 0)
        self.assertEqual(np.count_nonzero(icp_mask[:, 40:80]), 80 * 40)
        self.assertEqual(np.count_nonzero(icp_mask[:, 80:]), 80 * 40)
        self.assertEqual(report["removed_robot_pixels"], 80 * 40)
        self.assertEqual(report["protected_object_in_front_pixels"], 80 * 40)

    def test_pc1_left_gripper_gate_uses_workpiece_material(self) -> None:
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[100:120, 360:420] = 255
        mask[120:140, 420:480] = 255
        mask[140:160, 480:540] = 255
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        rgb[100:120, 360:420] = (45, 55, 65)  # brown metal
        rgb[120:140, 420:480] = (64, 68, 66)  # neutral silver metal
        rgb[140:160, 480:540] = (80, 65, 50)  # blue robot housing

        cleaned, report = global_inspect._pc1_left_gripper_colour_gate(
            mask, rgb, "pc1", "1"
        )

        self.assertEqual(np.count_nonzero(cleaned[100:120, 360:420]), 20 * 60)
        self.assertEqual(np.count_nonzero(cleaned[120:140, 420:480]), 0)
        self.assertEqual(np.count_nonzero(cleaned[140:160, 480:540]), 0)
        self.assertEqual(report["removed_pixels"], 2 * 20 * 60)
        self.assertEqual(report["allowed_material"], "brown")

        silver_cleaned, silver_report = (
            global_inspect._pc1_left_gripper_colour_gate(mask, rgb, "pc1", "2")
        )
        self.assertEqual(
            np.count_nonzero(silver_cleaned[120:140, 420:480]), 20 * 60
        )
        self.assertEqual(silver_report["allowed_material"], "silver")

    def test_pc1_left_gripper_gate_does_not_change_other_views(self) -> None:
        mask = np.ones((40, 60), dtype=np.uint8) * 255
        rgb = np.zeros((40, 60, 3), dtype=np.uint8)
        cleaned, report = global_inspect._pc1_left_gripper_colour_gate(
            mask, rgb, "pc2", "1"
        )
        np.testing.assert_array_equal(cleaned, mask)
        self.assertFalse(report["applied"])

    def test_pc1_left_gripper_gate_falls_back_when_it_removes_positive(self) -> None:
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[100:260, 350:550] = 255
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        cleaned, report = (
            global_inspect._pc1_left_gripper_colour_gate_with_quality_fallback(
                mask, rgb, "pc1", "1", [[405, 236]]
            )
        )
        np.testing.assert_array_equal(cleaned, mask)
        self.assertTrue(report["quality_fallback_used"])
        self.assertFalse(report["positive_points_retained"])


if __name__ == "__main__":
    unittest.main()
