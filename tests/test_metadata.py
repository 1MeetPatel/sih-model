"""
Unit Test: Raster Metadata and GSD Calculation
"""

import unittest
from src.io.metadata import extract_raster_metadata


class TestMetadata(unittest.TestCase):

    def test_sample_metadata(self):
        meta = extract_raster_metadata("sample_satellite_t1.tif")
        self.assertEqual(meta.width, 1024)
        self.assertEqual(meta.height, 1024)
        self.assertEqual(meta.count, 3)
        self.assertAlmostEqual(meta.gsd_m, 0.30, places=2)
        self.assertTrue(meta.is_gsd_reliable)


if __name__ == "__main__":
    unittest.main()
