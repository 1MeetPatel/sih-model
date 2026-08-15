"""
Unit Test: Overlap Blending
"""

import unittest
import numpy as np
from src.inference.blending import OverlapBlender


class TestBlending(unittest.TestCase):

    def test_overlap_blender_uniform_reconstruction(self):
        """
        Verifies that accumulating constant predictions results in a normalized probability map.
        """
        H, W = 600, 600
        tile_size = 256
        blender = OverlapBlender(H, W, tile_size=tile_size, blending_type="gaussian")

        for r in range(0, H - 100, 100):
            for c in range(0, W - 100, 100):
                tile_pred = np.ones((tile_size, tile_size), dtype=np.float32) * 0.8
                blender.add_tile_prediction(tile_pred, col_off=c, row_off=r, tile_width=tile_size, tile_height=tile_size)

        prob_map = blender.get_final_probability_map()
        self.assertEqual(prob_map.shape, (H, W))
        covered = prob_map[150:450, 150:450]
        self.assertTrue(np.allclose(covered, 0.8, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
