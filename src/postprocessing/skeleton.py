"""
Road Skeletonization and Centerline Extraction
"""

import numpy as np
from skimage.morphology import skeletonize
import cv2


def extract_road_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Computes a 1-pixel wide topological skeleton centerline of the road mask.
    """
    skel = skeletonize(binary_mask > 0)
    return (skel * 255).astype(np.uint8)
