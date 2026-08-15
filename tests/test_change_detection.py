"""
Unit Test: Temporal Change Detection Matching
"""

import unittest
import geopandas as gpd
from shapely.geometry import LineString
from src.change_detection.matching import TemporalRoadMatcher


class TestChangeDetection(unittest.TestCase):

    def test_temporal_road_matcher(self):
        """
        Creates synthetic BEFORE and AFTER networks:
        - BEFORE has Line A (stays) and Line B (removed)
        - AFTER has Line A (stays) and Line C (new)
        Verifies that Line C is detected as NEW and Line B as REMOVED.
        """
        line_a = LineString([(0, 0), (100, 0)])    # Unchanged
        line_b = LineString([(0, 50), (100, 50)])  # Removed
        line_c = LineString([(0, 100), (100, 100)]) # New

        gdf_before = gpd.GeoDataFrame([
            {"road_id": 1, "mean_width_m": 7.0, "confidence": 0.9, "geometry": line_a},
            {"road_id": 2, "mean_width_m": 6.5, "confidence": 0.85, "geometry": line_b}
        ], geometry="geometry", crs="EPSG:32643")

        gdf_after = gpd.GeoDataFrame([
            {"road_id": 1, "mean_width_m": 7.0, "confidence": 0.9, "geometry": line_a},
            {"road_id": 3, "mean_width_m": 8.0, "confidence": 0.92, "geometry": line_c}
        ], geometry="geometry", crs="EPSG:32643")

        matcher = TemporalRoadMatcher(buffer_m=5.0, min_change_length_m=10.0)
        new_roads, removed_roads, all_changes = matcher.match_networks(gdf_before, gdf_after)

        self.assertEqual(len(new_roads), 1)
        self.assertEqual(new_roads.iloc[0]["change_type"], "NEW_ROAD")
        self.assertEqual(new_roads.iloc[0]["source_id"], 3)

        self.assertEqual(len(removed_roads), 1)
        self.assertEqual(removed_roads.iloc[0]["change_type"], "REMOVED_ROAD")
        self.assertEqual(removed_roads.iloc[0]["source_id"], 2)


if __name__ == "__main__":
    unittest.main()
