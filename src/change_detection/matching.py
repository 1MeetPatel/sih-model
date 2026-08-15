"""
Temporal Geospatial Road Matching Engine
"""

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from typing import Tuple, Dict, Any, List
from ..utils.logging import get_logger

logger = get_logger("TemporalMatching")


class TemporalRoadMatcher:
    """
    Performs spatial alignment and geometric buffer comparison between
    BEFORE and AFTER road vector networks.
    """

    def __init__(self, buffer_m: float = 3.5, min_change_length_m: float = 20.0):
        self.buffer_m = buffer_m
        self.min_change_length_m = min_change_length_m

    def match_networks(
        self,
        gdf_before: gpd.GeoDataFrame,
        gdf_after: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Compares BEFORE and AFTER road networks.
        Returns:
            - new_roads (gdf): Roads in AFTER not in BEFORE
            - removed_roads (gdf): Roads in BEFORE not in AFTER
            - common_roads (gdf): Roads present in both
        """
        if gdf_before.empty and gdf_after.empty:
            empty_gdf = gpd.GeoDataFrame(columns=["change_type", "length_m", "confidence", "geometry"], geometry="geometry", crs=gdf_after.crs)
            return empty_gdf, empty_gdf, empty_gdf

        # Ensure matching CRS
        if gdf_before.crs != gdf_after.crs:
            logger.info(f"Reprojecting BEFORE network from {gdf_before.crs} to {gdf_after.crs}")
            gdf_before = gdf_before.to_crs(gdf_after.crs)

        # 1. NEW ROADS: Difference (AFTER - BEFORE_buffer)
        new_records = []
        new_geoms = []

        if not gdf_before.empty:
            try:
                before_union_buffer = gdf_before.geometry.buffer(self.buffer_m).union_all()
            except AttributeError:
                before_union_buffer = gdf_before.geometry.buffer(self.buffer_m).unary_union
        else:
            before_union_buffer = None

        for idx, row in gdf_after.iterrows():
            geom = row.geometry
            if before_union_buffer is None:
                diff = geom
            else:
                diff = geom.difference(before_union_buffer)

            if diff.is_empty:
                continue

            parts = [diff] if isinstance(diff, LineString) else (list(diff.geoms) if isinstance(diff, MultiLineString) else [])
            for p in parts:
                if p.length >= self.min_change_length_m:
                    new_geoms.append(p)
                    new_records.append({
                        "change_type": "NEW_ROAD",
                        "length_m": round(p.length, 2),
                        "width_m": row.get("mean_width_m", 6.1),
                        "confidence": round(row.get("confidence", 0.8), 3),
                        "source_id": row.get("road_id", idx)
                    })

        new_roads_gdf = gpd.GeoDataFrame(new_records, geometry=new_geoms, crs=gdf_after.crs)

        # 2. REMOVED ROADS: Difference (BEFORE - AFTER_buffer)
        rem_records = []
        rem_geoms = []

        if not gdf_after.empty:
            try:
                after_union_buffer = gdf_after.geometry.buffer(self.buffer_m).union_all()
            except AttributeError:
                after_union_buffer = gdf_after.geometry.buffer(self.buffer_m).unary_union
        else:
            after_union_buffer = None

        for idx, row in gdf_before.iterrows():
            geom = row.geometry
            if after_union_buffer is None:
                diff = geom
            else:
                diff = geom.difference(after_union_buffer)

            if diff.is_empty:
                continue

            parts = [diff] if isinstance(diff, LineString) else (list(diff.geoms) if isinstance(diff, MultiLineString) else [])
            for p in parts:
                if p.length >= self.min_change_length_m:
                    rem_geoms.append(p)
                    rem_records.append({
                        "change_type": "REMOVED_ROAD",
                        "length_m": round(p.length, 2),
                        "width_m": row.get("mean_width_m", 6.1),
                        "confidence": round(row.get("confidence", 0.8), 3),
                        "source_id": row.get("road_id", idx)
                    })

        removed_roads_gdf = gpd.GeoDataFrame(rem_records, geometry=rem_geoms, crs=gdf_after.crs)

        # 3. Combined all changes
        all_changes_list = []
        if not new_roads_gdf.empty:
            all_changes_list.append(new_roads_gdf)
        if not removed_roads_gdf.empty:
            all_changes_list.append(removed_roads_gdf)

        if all_changes_list:
            all_changes_gdf = gpd.pd.concat(all_changes_list, ignore_index=True)
            all_changes_gdf = gpd.GeoDataFrame(all_changes_gdf, geometry="geometry", crs=gdf_after.crs)
        else:
            all_changes_gdf = gpd.GeoDataFrame(columns=["change_type", "length_m", "width_m", "confidence", "geometry"], geometry="geometry", crs=gdf_after.crs)

        return new_roads_gdf, removed_roads_gdf, all_changes_gdf
