"""
Raster and Dataset Validation
"""

import os
from typing import Tuple, Optional
from ..io.metadata import RasterMetadata, extract_raster_metadata
from ..utils.logging import get_logger

logger = get_logger("Validator")


def validate_raster(filepath: str, min_gsd_m: float = 0.05, max_gsd_m: float = 5.0) -> Tuple[bool, Optional[str], Optional[RasterMetadata]]:
    """
    Validates existence, readable format, CRS, and GSD sufficiency for road analysis.
    """
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}", None

    try:
        meta = extract_raster_metadata(filepath)
    except Exception as e:
        return False, f"Failed to open or parse GeoTIFF: {str(e)}", None

    if meta.width <= 0 or meta.height <= 0:
        return False, f"Invalid dimensions: {meta.width}x{meta.height}", meta

    if meta.count < 1:
        return False, f"No raster bands found in {filepath}", meta

    if meta.gsd_m > max_gsd_m:
        logger.warning(
            f"Raster GSD is very coarse ({meta.gsd_m:.2f} m/px). "
            f"A 6.1m road will be only {meta.estimated_pixels_per_6_1m:.1f} pixels wide. "
            "Road detection accuracy may be degraded."
        )

    if not meta.crs:
        logger.warning(f"Raster {filepath} has no CRS defined. Pixel-space coordinates will be used.")

    return True, None, meta
