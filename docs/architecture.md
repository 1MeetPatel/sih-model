# Architecture Overview

## 1. System Architecture

```
                       ┌──────────────────────┐
                       │ Satellite GeoTIFF    │
                       │ 4K–20K+ resolution   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Raster Validation    │
                       │ CRS / GSD / Bands    │
                       │ NoData / Bounds      │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Windowed Tiling      │
                       │ 512×512 default      │
                       │ 128px overlap        │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ PyTorch Segmentation │
                       │ U-Net / DeepLabV3+   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Gaussian Overlap     │
                       │ Blending Stitching   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Probability GeoTIFF  │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Binary Road Mask     │
                       │ Threshold & Morph    │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Centerline /         │
                       │ Skeleton Extraction  │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Width Estimation     │
                       │ Distance Transform   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ ≥ 6.1m Road Filter   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ GIS Vector Network   │
                       │ GPKG / GeoJSON       │
                       └──────────┬───────────┘
```

## 2. Why Windowed Streaming Tiling is Essential
Large satellite rasters (4096×4096 up to 20000×20000+ pixels) exceed GPU VRAM and system memory limits if loaded as a single dense tensor.
- **Windowed Reads (`Rasterio.Window`)**: Slices small rectangular chunks from disk directly without duplicating the raster in RAM.
- **Batch Processing**: Groups patches into small streaming mini-batches (e.g. `batch_size=4`) sent to CPU/CUDA.
- **Memory Footprint**: Fixed $O(\text{batch\_size} \times \text{tile\_size}^2)$, independent of whether the input image is 1 GB or 50 GB.

## 3. Overlap-Aware Gaussian Blending
Standard grid tiling creates visible seam artifacts at patch boundaries because convolutional edge padding lowers confidence near borders.
- **2D Gaussian Kernel**: $W(x, y) = \exp\left(-\frac{(x - \mu_x)^2 + (y - \mu_y)^2}{2\sigma^2}\right)$
- **Accumulator**:
  $$\text{Final Probability} = \frac{\sum (\text{Tile Prediction} \times W)}{\sum W}$$
- Blends overlapping tile predictions smoothly, preserving spatial continuity across entire satellite scenes.
