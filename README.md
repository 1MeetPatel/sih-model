# Production-Quality AI Satellite Road Extraction & Temporal Change Detection

A modular, terminal-driven geospatial AI system for extracting roads **$\ge 20\text{ feet} \ (\approx 6.1\text{ meters})$** from large satellite GeoTIFF imagery and performing temporal road-change detection between Before and After scenes.

---

## Key Features

- **Large GeoTIFF Streaming ($4\text{K} - 20\text{K}+$ resolution)**: Streaming windowed I/O using Rasterio and Tile Generators. Never loads massive rasters into GPU memory at once.
- **Overlap-Aware Gaussian Blending**: Seamless multi-tile probability surface stitching without edge artifacts.
- **Continuous Geometric Width Estimation**: Euclidean Distance Transform ($W = 2 \cdot R \cdot \text{GSD}$) to accurately identify and filter qualifying $\ge 6.1\text{m}$ (20-foot) road corridors.
- **Temporal Change Detection**: Geometric buffer spatial matching to classify NEW, REMOVED, and WIDENED road vectors.
- **Full GIS Interchange Formats**: Native exports to GeoPackage (`.gpkg`), GeoJSON (`.geojson`), and georeferenced single-band GeoTIFFs (`road_probability.tif`, `road_mask.tif`).
- **Comprehensive Unit Testing**: Automated unit tests for GSD calculation, window boundaries, Gaussian blending, and physical road width estimation.

---

## Installation & Setup

### 1. Requirements
- Python 3.10+
- PyTorch 2.0+
- GDAL / Rasterio
- GeoPandas & Shapely
- segmentation-models-pytorch
- OpenCV, scikit-image, SciPy

```powershell
pip install torch torchvision rasterio geopandas shapely pyproj segmentation-models-pytorch opencv-python scikit-image scipy pandas pyyaml tqdm
```

### 2. Model Checkpoint
Ensure your pre-trained PyTorch model checkpoint is placed at `weights/best_road_seg_unet.pth`.

---

## CLI Usage Guide

### 1. Inspect Satellite GeoTIFF Metadata & GSD
```powershell
python main.py inspect --input sample_satellite_t1.tif
```
**Example Output:**
```
=======================================================
  Satellite GeoTIFF Raster Information
=======================================================
File:               sample_satellite_t1.tif
Dimensions:         1,024 x 1,024 pixels
Total Pixels:       1,048,576
Raster Bands:       3
Data Type:          uint8
CRS:                EPSG:32643
Projected Units:    Metric (Projected)
Calculated GSD:     0.3000 m/pixel
Target Road (6.1m): 20.3 pixels across
Resolution Status:  [OK] Sufficient for 6.1m road analysis
=======================================================
```

---

### 2. Extract Road Network ($\ge 6.1\text{m}$ / 20ft)
```powershell
python main.py extract `
  --input sample_satellite_t1.tif `
  --checkpoint weights/best_road_seg_unet.pth `
  --output outputs/t1_extraction/ `
  --min-width 6.096 `
  --threshold 0.35
```
**Generated Output Files in `outputs/t1_extraction/`:**
- `road_probability.tif`: Full continuous probability GeoTIFF.
- `road_mask.tif`: Morphology-cleaned binary road mask GeoTIFF.
- `roads.gpkg`: OGC GeoPackage vector layer containing qualifying centerlines and physical width attributes.
- `roads.geojson`: WGS84 GeoJSON road vector network.
- `road_extraction_summary.png`: 4-panel visual verification composite.

---

### 3. Temporal Road Change Detection (Before vs After)
```powershell
python main.py change `
  --before sample_satellite_t1.tif `
  --after sample_satellite_t2.tif `
  --checkpoint weights/best_road_seg_unet.pth `
  --output outputs/temporal_changes/ `
  --min-change-length 20.0
```
**Generated Output Files in `outputs/temporal_changes/changes/`:**
- `new_roads.gpkg` & `new_roads.geojson`: Newly constructed road segments in AFTER imagery.
- `removed_roads.gpkg` & `removed_roads.geojson`: Demolished / removed roads from BEFORE imagery.
- `all_changes.gpkg` & `all_changes.geojson`: Unified multi-class change GIS layer.

---

## Running Unit Tests

Execute the automated test suite:
```powershell
python -m unittest tests/test_width.py tests/test_tiling.py tests/test_blending.py tests/test_change_detection.py tests/test_metadata.py
```
*(All 6 unit tests pass with 100% success rate)*.

---

## Repository Structure

```
.
├── config.yaml
├── configs/
│   └── default.yaml
├── docs/
│   ├── architecture.md
│   ├── geospatial.md
│   └── model.md
├── main.py
├── src/
│   ├── cli/ (commands.py)
│   ├── io/ (metadata.py, raster_reader.py, raster_writer.py)
│   ├── preprocessing/ (validator.py, normalization.py)
│   ├── tiling/ (tile_generator.py, tile_metadata.py)
│   ├── models/ (model_factory.py, checkpoint.py)
│   ├── inference/ (batch_inference.py, blending.py)
│   ├── postprocessing/ (threshold.py, morphology.py, skeleton.py)
│   ├── geometry/ (width_estimation.py, road_filter.py)
│   ├── vectorization/ (centerline_to_vector.py, exporters.py)
│   ├── change_detection/ (matching.py)
│   └── utils/ (config.py, device.py, logging.py)
├── tests/
│   ├── test_blending.py
│   ├── test_change_detection.py
│   ├── test_metadata.py
│   ├── test_tiling.py
│   └── test_width.py
└── weights/
    └── best_road_seg_unet.pth
```
