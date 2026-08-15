"""
Generates synthetic high-resolution GeoTIFF satellite scenes (T1 & T2)
for testing the 20-foot (~6m) road detection, filtering, and change detection pipeline.
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

def create_synthetic_scenes(output_t1="sample_satellite_t1.tif", output_t2="sample_satellite_t2.tif"):
    width, height = 1024, 1024
    gsd = 0.30  # 0.30 meters/pixel (High-Res <= 0.5m requirement satisfied)

    # 1. Base terrain / landscape texture (RGB)
    np.random.seed(42)
    # Background terrain (greenish-brown vegetation/soil)
    base_t1 = np.zeros((height, width, 3), dtype=np.uint8)
    base_t1[:, :, 0] = np.random.randint(60, 90, (height, width))   # R
    base_t1[:, :, 1] = np.random.randint(110, 150, (height, width)) # G
    base_t1[:, :, 2] = np.random.randint(50, 80, (height, width))   # B

    # Add background noise/texture
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    base_t1 = np.clip(base_t1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Road widths in pixels at 0.30m/px:
    # 20-foot road (~6.1m) = 20 pixels
    # Highway (>40ft = ~15m) = 50 pixels
    # Footpath (<10ft = ~1.8m) = 6 pixels
    w_20ft_px = 20
    w_highway_px = 50
    w_footpath_px = 6

    # --- Draw Roads on T1 ---
    # Asphalt color: [80, 80, 80]
    # 1. Highway (50 px wide) -> Diagonal across bottom right
    cv2.line(base_t1, (200, 1024), (1024, 200), (70, 70, 70), thickness=w_highway_px)

    # 2. Footpath (6 px wide) -> Winding path
    pts_footpath = np.array([[50, 100], [150, 180], [100, 300], [200, 450]], np.int32)
    cv2.polylines(base_t1, [pts_footpath], False, (140, 120, 90), thickness=w_footpath_px)

    # 3. Target 20-foot (~6m) Secondary Road -> Curving through center
    pts_20ft_1 = np.array([[100, 0], [250, 300], [450, 600], [800, 900], [1024, 950]], np.int32)
    cv2.polylines(base_t1, [pts_20ft_1], False, (85, 85, 85), thickness=w_20ft_px)

    # 4. Another 20-foot branch road
    cv2.line(base_t1, (450, 600), (0, 700), (85, 85, 85), thickness=w_20ft_px)

    # --- Create T2 (Temporal Scene for Change Detection) ---
    base_t2 = base_t1.copy()
    # Add newly constructed 20-foot bypass road
    cv2.line(base_t2, (250, 300), (800, 200), (85, 85, 85), thickness=w_20ft_px)

    # Georeferencing metadata (UTM Zone 43N, EPSG:32643)
    # Origin: 500000 E, 2500000 N
    transform = from_origin(500000.0, 2500000.0, gsd, gsd)
    crs = "EPSG:32643"

    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': None,
        'width': width,
        'height': height,
        'count': 3,
        'crs': crs,
        'transform': transform,
        'photometric': 'RGB'
    }

    # Write T1
    with rasterio.open(output_t1, 'w', **profile) as dst:
        dst.write(base_t1[:, :, 0], 1)
        dst.write(base_t1[:, :, 1], 2)
        dst.write(base_t1[:, :, 2], 3)
    print(f"[✓] Created synthetic T1 GeoTIFF: {output_t1} (GSD: {gsd}m/px, Size: {width}x{height})")

    # Write T2
    with rasterio.open(output_t2, 'w', **profile) as dst:
        dst.write(base_t2[:, :, 0], 1)
        dst.write(base_t2[:, :, 1], 2)
        dst.write(base_t2[:, :, 2], 3)
    print(f"[✓] Created synthetic T2 GeoTIFF: {output_t2} (GSD: {gsd}m/px, Size: {width}x{height})")

if __name__ == "__main__":
    create_synthetic_scenes()
