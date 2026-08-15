"""
Road Filtering and Quality Assurance
"""

from typing import List
from .width_estimation import RoadSegmentMetrics


class RoadFilter:
    """
    Filters segmented roads based on minimum width (6.1m), minimum length, and confidence.
    """

    @staticmethod
    def filter_qualifying_roads(
        segments: List[RoadSegmentMetrics],
        min_width_m: float = 6.096,
        min_length_m: float = 15.0,
        min_confidence: float = 0.25
    ) -> List[RoadSegmentMetrics]:
        qualifying = []
        for s in segments:
            if s.mean_width_m >= min_width_m and s.length_m >= min_length_m and s.confidence >= min_confidence:
                qualifying.append(s)
        return qualifying
