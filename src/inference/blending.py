"""
Overlap-Aware Weighted Probability Stitching Accumulator
"""

import numpy as np
import cv2
from typing import Tuple


class OverlapBlender:
    """
    Accumulates overlapping tile probability predictions using a smooth
    2D Gaussian weighting window to eliminate tile boundary seams.
    """

    def __init__(self, full_height: int, full_width: int, tile_size: int = 512, blending_type: str = "gaussian"):
        self.height = full_height
        self.width = full_width
        self.tile_size = tile_size
        self.blending_type = blending_type

        self.prob_accum = np.zeros((full_height, full_width), dtype=np.float32)
        self.weight_accum = np.zeros((full_height, full_width), dtype=np.float32)

        # Generate 2D Gaussian weight window
        if blending_type == "gaussian":
            gx = cv2.getGaussianKernel(tile_size, tile_size / 4.0)
            self.window_weight = np.outer(gx, gx).astype(np.float32)
            self.window_weight /= max(self.window_weight.max(), 1e-6)
        else:
            self.window_weight = np.ones((tile_size, tile_size), dtype=np.float32)

    def add_tile_prediction(
        self,
        tile_pred: np.ndarray,
        col_off: int,
        row_off: int,
        tile_width: int,
        tile_height: int
    ):
        """
        Adds a single tile probability prediction to the accumulator.
        """
        # Determine actual slice in target raster
        actual_h = min(tile_height, self.height - row_off)
        actual_w = min(tile_width, self.width - col_off)

        pred_slice = tile_pred[:actual_h, :actual_w]
        weight_slice = self.window_weight[:actual_h, :actual_w]

        self.prob_accum[row_off:row_off + actual_h, col_off:col_off + actual_w] += pred_slice * weight_slice
        self.weight_accum[row_off:row_off + actual_h, col_off:col_off + actual_w] += weight_slice

    def get_final_probability_map(self) -> np.ndarray:
        """
        Computes the normalized probability map.
        Formula: sum(pred * weight) / sum(weight)
        """
        norm_prob = self.prob_accum / np.maximum(self.weight_accum, 1e-6)
        return np.clip(norm_prob, 0.0, 1.0)
