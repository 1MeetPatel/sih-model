"""
Morphological Post-Processing and Component Filtering
"""

import numpy as np
import cv2


def clean_road_mask(
    binary_mask: np.ndarray,
    open_radius: int = 1,
    close_radius: int = 2,
    min_area_pixels: int = 20,
    min_aspect_ratio: float = 1.6
) -> np.ndarray:
    """
    Applies morphological closing to bridge gaps, opening to remove noise,
    and connected component geometry filtering to reject isolated building blobs.
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * close_radius + 1, 2 * close_radius + 1))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    if open_radius > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * open_radius + 1, 2 * open_radius + 1))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
    else:
        opened = closed

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    clean_mask = np.zeros_like(opened)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        diag = np.sqrt(w**2 + h**2)
        aspect = max(w, h) / (min(w, h) + 1e-6)

        # Keep elongated road ribbons
        if area >= min_area_pixels and (diag >= 12 or aspect >= min_aspect_ratio):
            clean_mask[labels == i] = 1

    return clean_mask
