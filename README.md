# 🛰️ AI Satellite Road Extraction & Topology Analyzer

A high-precision deep learning pipeline to extract and vectorize road networks from satellite imagery across Indian cities, planned grids, dense urban areas, and mountain terrains.

---

## ⚡ Quick Start: Run in Your Terminal (Super Simple!)

You can run the road detection model on any satellite image with **just one simple command**:

### 1. Clone & Install Dependencies
Open your terminal (PowerShell / Command Prompt / Bash) and run:
```bash
git clone https://github.com/1MeetPatel/sih-model.git
cd sih-model
pip install torch torchvision opencv-python scikit-image scipy matplotlib segmentation-models-pytorch
```

### 2. Run the Model on Any Image
Place your satellite image (PNG or JPG) and run:
```bash
python restore_exact_heatmap.py "path/to/your_image.png" "City Name"
```

**Examples:**
```bash
# Example 1: Run on Gandhinagar
python restore_exact_heatmap.py "gandhinagar.png" "Gandhinagar"

# Example 2: Run on Leh, Ladakh
python restore_exact_heatmap.py "leh.png" "Leh Ladakh"

# Example 3: Run on Chandigarh
python restore_exact_heatmap.py "chandigarh.png" "Chandigarh"
```

### 3. Check the Results
Outputs are automatically saved inside the **`results/`** folder:
- **`results/<city_name>_exact_4panel.png`**: High-resolution 4-panel diagnostic sheet:
  1. Original Satellite Image
  2. Neural Road Probability Heatmap
  3. Clean Road Topology Skeleton
  4. Crisp Red Vector Road Overlay
- **`results/<city_name>_exact_overlay.png`**: High-resolution standalone red road overlay.

---

## 🌟 Key Features


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
