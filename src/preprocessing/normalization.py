"""
Raster Data Normalization Utilities
"""

import numpy as np
import cv2
from typing import Tuple


def normalize_raster_patch(
    patch: np.ndarray,
    method: str = "percentile",
    p_min: float = 2.0,
    p_max: float = 98.0
) -> np.ndarray:
    """
    Normalizes multi-band raster data (uint8, uint16, int32, float32)
    to a standardized [0.0, 1.0] float32 image suitable for PyTorch encoders.
    """
    img = patch.astype(np.float32)

    # If single band, convert to 3-channel
    if img.ndim == 2:
        img = np.dstack([img, img, img])
    elif img.shape[2] == 1:
        img = np.dstack([img[:, :, 0], img[:, :, 0], img[:, :, 0]])

    # Percentile clipping per band to avoid sensor saturation / atmospheric scatter
    norm_channels = []
    for c in range(min(img.shape[2], 3)):
        band = img[:, :, c]
        if method == "percentile":
            low = np.percentile(band, p_min)
            high = np.percentile(band, p_max)
            if high > low:
                band_norm = np.clip((band - low) / (high - low), 0.0, 1.0)
            else:
                band_norm = np.zeros_like(band)
        else:
            # Standard Min-Max
            b_min, b_max = band.min(), band.max()
            if b_max > b_min:
                band_norm = (band - b_min) / (b_max - b_min)
            else:
                band_norm = np.zeros_like(band)
        norm_channels.append(band_norm)

    # Stack to 3 channels (RGB)
    norm_img = np.dstack(norm_channels[:3])
    return norm_img.astype(np.float32)
