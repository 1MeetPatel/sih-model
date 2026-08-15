"""
Road Probability Thresholding and Hysteresis
"""

import numpy as np
import cv2


def apply_probability_threshold(
    prob_map: np.ndarray,
    threshold: float = 0.35
) -> np.ndarray:
    """
    Applies configurable probability threshold to produce a binary road mask.
    """
    return (prob_map >= threshold).astype(np.uint8)


def apply_hysteresis_threshold(
    prob_map: np.ndarray,
    low_thresh: float = 0.15,
    high_thresh: float = 0.30
) -> np.ndarray:
    """
    Dual-threshold hysteresis: keeps road paths that have high-confidence seeds
    and extend through low-threshold connectors.
    """
    seed_mask = (prob_map >= high_thresh).astype(np.uint8)
    cand_mask = (prob_map >= low_thresh).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand_mask, connectivity=8)
    out_mask = np.zeros_like(cand_mask)

    for i in range(1, num_labels):
        comp = (labels == i)
        if np.any(seed_mask[comp]):
            out_mask[comp] = 1

    return out_mask
