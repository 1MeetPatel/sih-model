"""
Unit Test: Geometric Road Width Estimation
"""

import unittest
import numpy as np
from src.geometry.width_estimation import RoadWidthEstimator


class TestRoadWidth(unittest.TestCase):

    def test_synthetic_road_width_6_1m(self):
        """
        Creates a synthetic road raster with known physical width of 6.0m (20 pixels at 0.3m/px)
        and verifies that the Euclidean Distance Transform estimator accurately measures ~6.0m.
        """
        H, W = 200, 200
        gsd_m = 0.30  # 0.3m per pixel
        target_width_px = 20  # 20 * 0.3 = 6.0 meters

        # Create binary mask with a horizontal road corridor of width 20px
        binary_mask = np.zeros((H, W), dtype=np.uint8)
        road_top = 90
        road_bottom = road_top + target_width_px
        binary_mask[road_top:road_bottom, 20:180] = 1

        # Create a 1-pixel centerline skeleton
        skeleton_mask = np.zeros((H, W), dtype=np.uint8)
        center_y = road_top + target_width_px // 2
        skeleton_mask[center_y, 30:170] = 255

        prob_map = np.ones((H, W), dtype=np.float32) * 0.85

        estimator = RoadWidthEstimator(gsd_m=gsd_m, min_qualifying_width_m=6.0)
        segments, qual_skel = estimator.estimate_segment_widths(
            binary_mask=binary_mask,
            skeleton_mask=skeleton_mask,
            probability_map=prob_map,
            min_length_m=10.0
        )

        self.assertEqual(len(segments), 1)
        measured_width = segments[0].mean_width_m
        expected_width = target_width_px * gsd_m  # 6.0m

        self.assertLessEqual(abs(measured_width - expected_width), 0.8)
        self.assertTrue(segments[0].qualifies_20ft)
        self.assertTrue(np.any(qual_skel > 0))

    def test_narrow_road_rejection(self):
        """
        Creates a narrow footpath (width 1.8m, ~6px at 0.3m/px) and verifies
        that it is rejected by the 6.1m filter.
        """
        H, W = 100, 100
        gsd_m = 0.30
        narrow_px = 6  # 6 * 0.3 = 1.8 meters

        binary_mask = np.zeros((H, W), dtype=np.uint8)
        binary_mask[47:47 + narrow_px, 10:90] = 1

        skeleton_mask = np.zeros((H, W), dtype=np.uint8)
        skeleton_mask[50, 15:85] = 255
        prob_map = np.ones((H, W), dtype=np.float32) * 0.90

        estimator = RoadWidthEstimator(gsd_m=gsd_m, min_qualifying_width_m=6.096)
        segments, qual_skel = estimator.estimate_segment_widths(
            binary_mask=binary_mask,
            skeleton_mask=skeleton_mask,
            probability_map=prob_map
        )

        self.assertEqual(len(segments), 1)
        self.assertLess(segments[0].mean_width_m, 3.0)
        self.assertFalse(segments[0].qualifies_20ft)
        self.assertEqual(np.sum(qual_skel), 0)


if __name__ == "__main__":
    unittest.main()
