"""
Geospatial Raster Writer with Coordinate Preservation
"""

import os
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from typing import Optional


class GeoTIFFWriter:
    """
    Writes probability maps and binary masks as fully georeferenced GeoTIFFs.
    """

    @staticmethod
    def write_single_band(
        output_path: str,
        data: np.ndarray,
        crs: Optional[CRS],
        transform: Affine,
        dtype: str = "float32",
        nodata: Optional[float] = None,
        compress: str = "deflate"
    ):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        height, width = data.shape[:2]
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": compress
        }
        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype(dtype), 1)
