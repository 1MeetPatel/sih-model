"""
Geospatial Raster Metadata Extraction and GSD Calculation
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import pyproj


@dataclass
class RasterMetadata:
    filepath: str
    width: int
    height: int
    count: int
    dtype: str
    crs: Optional[CRS]
    crs_wkt: str
    is_projected: bool
    transform: Affine
    bounds: Tuple[float, float, float, float]
    pixel_width: float
    pixel_height: float
    gsd_m: float
    nodata: Optional[float]
    estimated_pixels_per_6_1m: float
    is_gsd_reliable: bool

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "dimensions": f"{self.width} x {self.height}",
            "bands": self.count,
            "dtype": self.dtype,
            "crs": str(self.crs) if self.crs else "None (Unreferenced)",
            "is_projected": self.is_projected,
            "gsd_m": round(self.gsd_m, 4),
            "bounds": [round(b, 4) for b in self.bounds],
            "nodata": self.nodata,
            "pixels_for_6_1m_road": round(self.estimated_pixels_per_6_1m, 2),
            "gsd_reliable": self.is_gsd_reliable
        }


def extract_raster_metadata(filepath: str) -> RasterMetadata:
    """
    Extracts comprehensive metadata, calculates accurate metric GSD,
    and checks road resolution feasibility for 6.1m target width.
    """
    with rasterio.open(filepath) as src:
        width = src.width
        height = src.height
        count = src.count
        dtype = str(src.dtypes[0])
        crs = src.crs
        transform = src.transform
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        nodata = src.nodata

        px_w = abs(transform.a)
        px_h = abs(transform.e)
        
        is_projected = False
        if crs:
            try:
                is_projected = crs.is_projected
            except Exception:
                is_projected = False

        # Calculate GSD in meters
        if is_projected or (px_w >= 0.01):
            # Already in metric units (e.g. UTM)
            gsd_m = (px_w + px_h) / 2.0
        else:
            # Geographic CRS in degrees (e.g. EPSG:4326) -> convert degrees to meters at center latitude
            center_lat = (bounds[1] + bounds[3]) / 2.0
            # 1 deg lat ~ 111,139 m, 1 deg lon ~ 111,139 * cos(lat) m
            lat_m_per_deg = 111139.0
            lon_m_per_deg = 111139.0 * math.cos(math.radians(center_lat))
            gsd_x_m = px_w * lon_m_per_deg
            gsd_y_m = px_h * lat_m_per_deg
            gsd_m = (gsd_x_m + gsd_y_m) / 2.0

        # Estimate expected pixels across a 6.096m (20ft) road
        target_road_width_m = 6.096
        pixels_across_road = target_road_width_m / max(gsd_m, 1e-6)
        
        # GSD is reliable if pixel resolution gives at least ~2-3 pixels across road
        is_gsd_reliable = (gsd_m <= 3.5)

        return RasterMetadata(
            filepath=filepath,
            width=width,
            height=height,
            count=count,
            dtype=dtype,
            crs=crs,
            crs_wkt=crs.to_wkt() if crs else "",
            is_projected=is_projected,
            transform=transform,
            bounds=bounds,
            pixel_width=px_w,
            pixel_height=px_h,
            gsd_m=gsd_m,
            nodata=nodata,
            estimated_pixels_per_6_1m=pixels_across_road,
            is_gsd_reliable=is_gsd_reliable
        )
