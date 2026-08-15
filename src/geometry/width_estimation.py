"""
Geometric Road Width Estimation via Euclidean Distance Transform
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class RoadSegmentMetrics:
    segment_id: int
    coords_pixel: List[Tuple[int, int]]
    length_m: float
    mean_width_m: float
    median_width_m: float
    min_width_m: float
    max_width_m: float
    qualifies_20ft: bool
    confidence: float


class RoadWidthEstimator:
    """
    Measures physical road widths (meters) along centerlines using Euclidean Distance Transforms.
    Formula: Width(p) = 2 * Radius(p) * GSD
    """

    def __init__(self, gsd_m: float, min_qualifying_width_m: float = 6.096):
        self.gsd_m = max(gsd_m, 1e-6)
        self.min_qualifying_width_m = min_qualifying_width_m

    def compute_distance_transform(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Computes Euclidean distance from each road pixel to the nearest background edge.
        """
        return distance_transform_edt(binary_mask > 0)

    def estimate_segment_widths(
        self,
        binary_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        probability_map: np.ndarray,
        min_length_m: float = 15.0
    ) -> Tuple[List[RoadSegmentMetrics], np.ndarray]:
        """
        Extracts interconnected road segments and calculates continuous physical widths.
        Returns segment metrics and a qualified 6.1m+ skeleton mask.
        """
        dist_map = self.compute_distance_transform(binary_mask)
        skel_bool = skeleton_mask > 0

        # Connected component decomposition of centerlines
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton_mask.astype(np.uint8), connectivity=8)

        segments: List[RoadSegmentMetrics] = []
        qualifying_skel = np.zeros_like(skeleton_mask, dtype=np.uint8)

        for seg_id in range(1, num_labels):
            seg_pts = np.argwhere(labels == seg_id)  # (N, 2) [row, col]
            if len(seg_pts) < 3:
                continue

            # Length estimation in meters
            # Each step is ~1 px or ~sqrt(2) px
            length_px = float(len(seg_pts))
            length_m = length_px * self.gsd_m

            # Sample radii along the centerline
            radii_px = dist_map[seg_pts[:, 0], seg_pts[:, 1]]
            widths_m = 2.0 * radii_px * self.gsd_m

            mean_w = float(np.mean(widths_m))
            median_w = float(np.median(widths_m))
            min_w = float(np.min(widths_m))
            max_w = float(np.max(widths_m))

            # Mean probability confidence
            conf = float(np.mean(probability_map[seg_pts[:, 0], seg_pts[:, 1]]))

            # Qualifying condition: meets or exceeds configured 6.096m (20ft) requirement
            qualifies = (mean_w >= self.min_qualifying_width_m or median_w >= (self.min_qualifying_width_m * 0.9)) and (length_m >= min_length_m)

            coords = [(int(pt[1]), int(pt[0])) for pt in seg_pts]  # (x, y)

            metric = RoadSegmentMetrics(
                segment_id=seg_id,
                coords_pixel=coords,
                length_m=round(length_m, 2),
                mean_width_m=round(mean_w, 2),
                median_width_m=round(median_w, 2),
                min_width_m=round(min_w, 2),
                max_width_m=round(max_w, 2),
                qualifies_20ft=qualifies,
                confidence=round(conf, 3)
            )
            segments.append(metric)

            if qualifies:
                qualifying_skel[seg_pts[:, 0], seg_pts[:, 1]] = 255

        return segments, qualifying_skel
