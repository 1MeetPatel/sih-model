"""
Streaming Windowed Batch Inference Engine
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from tqdm import tqdm

from ..io.raster_reader import WindowedRasterReader
from ..tiling.tile_generator import TileWindow
from ..preprocessing.normalization import normalize_raster_patch
from .blending import OverlapBlender
from ..utils.logging import get_logger

logger = get_logger("InferenceEngine")


def run_windowed_inference(
    reader: WindowedRasterReader,
    windows: List[TileWindow],
    model: nn.Module,
    device: torch.device,
    tile_size: int = 512,
    batch_size: int = 4,
    amp: bool = False,
    blending_type: str = "gaussian"
) -> np.ndarray:
    """
    Processes large satellite GeoTIFFs tile-by-tile in streaming batches.
    Never loads the full 4K-20K+ image into GPU or RAM simultaneously.
    """
    logger.info(f"Starting windowed inference on {len(windows)} tiles (batch_size={batch_size}, device={device})")
    blender = OverlapBlender(reader.height, reader.width, tile_size=tile_size, blending_type=blending_type)

    model.eval()

    for i in tqdm(range(0, len(windows), batch_size), desc="Inferring Road Tiles"):
        batch_windows = windows[i:i + batch_size]
        batch_tensors = []

        for win in batch_windows:
            patch = reader.read_window(win.col_off, win.row_off, win.width, win.height)
            norm_patch = normalize_raster_patch(patch)  # (H, W, 3) float32 in [0, 1]
            
            # PyTorch format: (C, H, W)
            tensor = torch.from_numpy(norm_patch.transpose(2, 0, 1)).float()
            batch_tensors.append(tensor)

        batch_stack = torch.stack(batch_tensors).to(device)

        with torch.inference_mode():
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    preds = model(batch_stack)
            else:
                preds = model(batch_stack)

            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            preds_np = preds.cpu().numpy()
            if preds_np.ndim == 4 and preds_np.shape[1] == 1:
                preds_np = preds_np[:, 0, :, :]

        for idx, win in enumerate(batch_windows):
            blender.add_tile_prediction(
                tile_pred=preds_np[idx],
                col_off=win.col_off,
                row_off=win.row_off,
                tile_width=win.width,
                tile_height=win.height
            )

    logger.info("[OK] Inference complete. Blending probability surfaces...")
    return blender.get_final_probability_map()
