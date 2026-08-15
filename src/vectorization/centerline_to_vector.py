"""
Road Centerline Vectorization with Geospatial Coordinate Mapping
"""

from typing import List, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from rasterio.transform import Affine
from rasterio.crs import CRS
from ..geometry.width_estimation import RoadSegmentMetrics
from ..utils.logging import get_logger

logger = get_logger("Vectorization")


class CenterlineToVector:
    """
    Converts pixel centerline segments to georeferenced LineString geometries.
    """

    @staticmethod
    def segments_to_geodataframe(
        segments: List[RoadSegmentMetrics],
        transform: Affine,
        crs: Optional[CRS]
    ) -> gpd.GeoDataFrame:
        geometries = []
        records = []

        for seg in segments:
            if len(seg.coords_pixel) < 2:
                continue

            # Convert pixel coords (col, row) -> spatial coordinates (x_geo, y_geo)
            geo_coords = []
            for col, row in seg.coords_pixel:
                x_geo, y_geo = transform * (col + 0.5, row + 0.5)
                geo_coords.append((x_geo, y_geo))

            line = LineString(geo_coords)
            if not line.is_valid or line.length == 0:
                continue

            geometries.append(line)
            records.append({
                "road_id": seg.segment_id,
                "length_m": seg.length_m,
                "mean_width_m": seg.mean_width_m,
                "median_width_m": seg.median_width_m,
                "min_width_m": seg.min_width_m,
                "max_width_m": seg.max_width_m,
                "confidence": seg.confidence,
                "qualifies_20ft": seg.qualifies_20ft
            })

        if not records:
            # Create empty geodataframe
            gdf = gpd.GeoDataFrame(
                columns=["road_id", "length_m", "mean_width_m", "median_width_m", "min_width_m", "max_width_m", "confidence", "qualifies_20ft", "geometry"],
                geometry="geometry",
                crs=crs
            )
        else:
            gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)

        return gdf
