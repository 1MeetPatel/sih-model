"""
Windowed Multi-Band Satellite Raster Reader
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from typing import List, Optional, Tuple
from ..io.metadata import RasterMetadata, extract_raster_metadata
from ..utils.logging import get_logger

logger = get_logger("RasterReader")


class WindowedRasterReader:
    """
    Streaming reader for arbitrarily large GeoTIFFs (4K-20K+ resolution)
    using Rasterio Window slices without exhausting RAM.
    """

    def __init__(self, filepath: str, bands: Optional[List[int]] = None):
        self.filepath = filepath
        self.metadata: RasterMetadata = extract_raster_metadata(filepath)
        self.dataset = rasterio.open(filepath)
        
        # Configure bands (1-indexed for rasterio)
        if bands:
            # Ensure bands are within count
            self.bands = [b for b in bands if 1 <= b <= self.dataset.count]
            if not self.bands:
                self.bands = list(range(1, min(self.dataset.count + 1, 4)))
        else:
            self.bands = list(range(1, min(self.dataset.count + 1, 4)))

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def crs(self):
        return self.metadata.crs

    @property
    def transform(self):
        return self.metadata.transform

    def read_window(self, col_off: int, row_off: int, width: int, height: int, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Reads a window slice. Returns an (H, W, C) numpy array.
        Handles image boundary clamping and padding.
        """
        # Clamp window to raster bounds
        actual_col = max(0, min(col_off, self.width - 1))
        actual_row = max(0, min(row_off, self.height - 1))
        actual_w = max(1, min(width, self.width - actual_col))
        actual_h = max(1, min(height, self.height - actual_row))

        window = Window(actual_col, actual_row, actual_w, actual_h)
        data = self.dataset.read(self.bands, window=window)  # (C, H, W)

        # Transpose to (H, W, C)
        img = np.transpose(data, (1, 2, 0))

        # Pad if window was smaller than requested width/height
        if actual_w != width or actual_h != height:
            pad_h = height - actual_h
            pad_w = width - actual_w
            img = np.pad(img, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)), mode="reflect")

        return img

    def close(self):
        if self.dataset and not self.dataset.closed:
            self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
