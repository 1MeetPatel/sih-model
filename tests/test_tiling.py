"""
Unit Test: Tiling and Window Generation
"""

import unittest
from src.tiling.tile_generator import TileGenerator


class TestTiling(unittest.TestCase):

    def test_tile_generation_coverage(self):
        """
        Verifies that generated windows cover the entire image without gaps.
        """
        W, H = 2048, 1536
        tile_size = 512
        overlap = 128

        windows = TileGenerator.generate_windows(W, H, tile_size=tile_size, overlap=overlap)
        self.assertGreater(len(windows), 0)

        for w in windows:
            self.assertEqual(w.width, tile_size)
            self.assertEqual(w.height, tile_size)
            self.assertGreaterEqual(w.col_off, 0)
            self.assertGreaterEqual(w.row_off, 0)
            self.assertTrue(w.col_off + w.width <= W or w.col_off == max(0, W - tile_size))
            self.assertTrue(w.row_off + w.height <= H or w.row_off == max(0, H - tile_size))


if __name__ == "__main__":
    unittest.main()
