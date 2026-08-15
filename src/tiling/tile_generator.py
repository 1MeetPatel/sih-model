"""
Tile Slicing and Window Metadata
"""

from dataclasses import dataclass
from typing import List, Tuple
from rasterio.transform import Affine


@dataclass
class TileWindow:
    tile_id: int
    col_off: int
    row_off: int
    width: int
    height: int
    source_width: int
    source_height: int

    def get_pixel_slice(self) -> Tuple[slice, slice]:
        return (
            slice(self.row_off, min(self.row_off + self.height, self.source_height)),
            slice(self.col_off, min(self.col_off + self.width, self.source_width))
        )


class TileGenerator:
    """
    Generates overlap-aware window coordinates covering the entire raster.
    Ensures all tile windows remain strictly within [0, width] and [0, height].
    """

    @staticmethod
    def generate_windows(
        width: int,
        height: int,
        tile_size: int = 512,
        overlap: int = 128
    ) -> List[TileWindow]:
        stride = max(1, tile_size - overlap)
        windows: List[TileWindow] = []
        tile_id = 0

        # Compute raw rows and cols with stride
        raw_rows = list(range(0, height, stride))
        raw_cols = list(range(0, width, stride))

        # Clamp offsets so that col_off + tile_size <= width and row_off + tile_size <= height
        max_row = max(0, height - tile_size)
        max_col = max(0, width - tile_size)

        rows = sorted(list(set([min(r, max_row) for r in raw_rows])))
        cols = sorted(list(set([min(c, max_col) for c in raw_cols])))

        for r in rows:
            for c in cols:
                windows.append(
                    TileWindow(
                        tile_id=tile_id,
                        col_off=c,
                        row_off=r,
                        width=min(tile_size, width),
                        height=min(tile_size, height),
                        source_width=width,
                        source_height=height
                    )
                )
                tile_id += 1

        return windows
