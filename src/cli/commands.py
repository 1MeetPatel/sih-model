"""
Command Line Interface Sub-Commands Implementation
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..io.metadata import extract_raster_metadata
from ..io.raster_reader import WindowedRasterReader
from ..io.raster_writer import GeoTIFFWriter
from ..preprocessing.validator import validate_raster
from ..tiling.tile_generator import TileGenerator
from ..models.model_factory import ModelFactory
from ..models.checkpoint import load_model_checkpoint
from ..inference.batch_inference import run_windowed_inference
from ..postprocessing.threshold import apply_probability_threshold
from ..postprocessing.morphology import clean_road_mask
from ..postprocessing.skeleton import extract_road_skeleton
from ..geometry.width_estimation import RoadWidthEstimator
from ..vectorization.centerline_to_vector import CenterlineToVector
from ..vectorization.exporters import GISExporter
from ..change_detection.matching import TemporalRoadMatcher
from ..utils.config import load_config
from ..utils.device import resolve_device
from ..utils.logging import get_logger

logger = get_logger("CLI")


def inspect_command(input_path: str, verbose: bool = False):
    """Executes the raster inspection command."""
    valid, err, meta = validate_raster(input_path)
    if not valid:
        logger.error(f"Inspection failed: {err}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Satellite GeoTIFF Raster Information")
    print("=" * 55)
    print(f"File:               {meta.filepath}")
    print(f"Dimensions:         {meta.width:,} x {meta.height:,} pixels")
    print(f"Total Pixels:       {meta.width * meta.height:,}")
    print(f"Raster Bands:       {meta.count}")
    print(f"Data Type:          {meta.dtype}")
    print(f"CRS:                {meta.crs if meta.crs else 'None (Unreferenced)'}")
    print(f"Projected Units:    {'Metric (Projected)' if meta.is_projected else 'Angular (Geographic Degrees)'}")
    print(f"Calculated GSD:     {meta.gsd_m:.4f} m/pixel")
    print(f"NoData Value:       {meta.nodata}")
    print(f"Target Road (6.1m): {meta.estimated_pixels_per_6_1m:.1f} pixels across")
    print(f"Resolution Status:  {'[OK] Sufficient for 6.1m road analysis' if meta.is_gsd_reliable else '[WARNING] GSD coarse'}")
    print("-" * 55)
    print("Bounding Box (West, South, East, North):")
    print(f"  {meta.bounds[0]:.6f}, {meta.bounds[1]:.6f}, {meta.bounds[2]:.6f}, {meta.bounds[3]:.6f}")
    print("=" * 55 + "\n")


def run_single_image_pipeline(
    input_path: str,
    checkpoint_path: str,
    output_dir: str,
    config: dict,
    device_str: str = "auto",
    min_width_m: float = 6.096,
    threshold: float = 0.35
):
    """Core single-image road extraction pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Opening raster: {input_path}")
    valid, err, meta = validate_raster(input_path)
    if not valid:
        raise ValueError(f"Raster validation failed: {err}")

    logger.info(f"CRS: {meta.crs} | GSD: {meta.gsd_m:.4f} m/px | Dimensions: {meta.width}x{meta.height}")
    device = resolve_device(device_str)

    # 1. Tiling setup
    tile_size = config.get("tiling", {}).get("tile_size", 512)
    overlap = config.get("tiling", {}).get("overlap", 128)
    batch_size = config.get("inference", {}).get("batch_size", 4)
    amp = config.get("inference", {}).get("amp", False)

    windows = TileGenerator.generate_windows(meta.width, meta.height, tile_size=tile_size, overlap=overlap)
    logger.info(f"Generated {len(windows)} overlap-aware tiles (tile_size={tile_size}, overlap={overlap})")

    # 2. Model & Checkpoint Loading
    model_cfg = config.get("model", {})
    model = ModelFactory.create_model(
        architecture=model_cfg.get("architecture", "unet"),
        encoder=model_cfg.get("encoder", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1)
    )
    model = load_model_checkpoint(model, checkpoint_path, device=device)

    # 3. Windowed Streaming Inference
    with WindowedRasterReader(input_path, bands=config.get("input", {}).get("bands", [1, 2, 3])) as reader:
        prob_map = run_windowed_inference(
            reader=reader,
            windows=windows,
            model=model,
            device=device,
            tile_size=tile_size,
            batch_size=batch_size,
            amp=amp
        )

    # 4. Save Probability GeoTIFF
    prob_tif_path = os.path.join(output_dir, "road_probability.tif")
    GeoTIFFWriter.write_single_band(prob_tif_path, prob_map, crs=meta.crs, transform=meta.transform, dtype="float32")
    logger.info(f"[+] Saved Road Probability GeoTIFF -> {prob_tif_path}")

    # 5. Thresholding & Morphological Cleaning
    logger.info(f"Applying road probability threshold ({threshold})...")
    raw_mask = apply_probability_threshold(prob_map, threshold=threshold)
    clean_mask = clean_road_mask(raw_mask, min_area_pixels=15)

    mask_tif_path = os.path.join(output_dir, "road_mask.tif")
    GeoTIFFWriter.write_single_band(mask_tif_path, clean_mask, crs=meta.crs, transform=meta.transform, dtype="uint8")
    logger.info(f"[+] Saved Binary Road Mask GeoTIFF -> {mask_tif_path}")

    # 6. Centerlines & Geometric Width Estimation
    logger.info("Extracting topological centerlines...")
    skel = extract_road_skeleton(clean_mask)

    logger.info(f"Estimating road widths via Euclidean Distance Transform (min_width={min_width_m:.2f}m)...")
    estimator = RoadWidthEstimator(gsd_m=meta.gsd_m, min_qualifying_width_m=min_width_m)
    segments, qual_skel = estimator.estimate_segment_widths(
        binary_mask=clean_mask,
        skeleton_mask=skel,
        probability_map=prob_map,
        min_length_m=config.get("roads", {}).get("minimum_segment_length_m", 15.0)
    )

    # 7. Vectorization & GIS Export
    logger.info("Converting qualifying road centerlines to GIS vector layers...")
    gdf = CenterlineToVector.segments_to_geodataframe(segments, transform=meta.transform, crs=meta.crs)
    qualifying_gdf = gdf[gdf["qualifies_20ft"] == True].copy()
    logger.info(f"Total extracted segments: {len(gdf)} | Qualifying >= {min_width_m:.1f}m: {len(qualifying_gdf)}")

    GISExporter.export_all(qualifying_gdf, output_dir=output_dir, base_name="roads")

    # 8. Render 4-Panel Verification Image
    vis_path = os.path.join(output_dir, "road_extraction_summary.png")
    with WindowedRasterReader(input_path) as reader:
        # Read preview RGB
        preview_rgb = reader.read_window(0, 0, min(meta.width, 2048), min(meta.height, 2048))
        if preview_rgb.dtype != np.uint8:
            p2, p98 = np.percentile(preview_rgb, (2, 98))
            preview_rgb = np.clip((preview_rgb - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    axes[0].imshow(preview_rgb)
    axes[0].set_title('1. Satellite GeoTIFF', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_map[:preview_rgb.shape[0], :preview_rgb.shape[1]], cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=9)

    axes[2].imshow(clean_mask[:preview_rgb.shape[0], :preview_rgb.shape[1]], cmap='gray')
    axes[2].set_title('3. Road Mask (Morphology Cleaned)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    overlay = preview_rgb.copy()
    qual_slice = qual_skel[:preview_rgb.shape[0], :preview_rgb.shape[1]]
    dilated = cv2.dilate(qual_slice, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))) > 0
    overlay[dilated] = [255, 0, 0]

    axes[3].imshow(overlay)
    axes[3].set_title(f'4. Qualifying Roads (>= {min_width_m:.1f}m / 20ft)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.savefig(vis_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"[+] Saved Visual Summary -> {vis_path}")

    return qualifying_gdf


def extract_command(args):
    """Executes single-image road extraction."""
    config = load_config(args.config)
    run_single_image_pipeline(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        config=config,
        device_str=args.device,
        min_width_m=args.min_width,
        threshold=args.threshold
    )


def change_command(args):
    """Executes temporal road change detection between BEFORE and AFTER GeoTIFFs."""
    config = load_config(args.config)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    before_dir = os.path.join(output_dir, "before")
    after_dir = os.path.join(output_dir, "after")
    changes_dir = os.path.join(output_dir, "changes")
    os.makedirs(changes_dir, exist_ok=True)

    logger.info("=== STEP 1/3: Extracting BEFORE Road Network ===")
    gdf_before = run_single_image_pipeline(
        input_path=args.before,
        checkpoint_path=args.checkpoint,
        output_dir=before_dir,
        config=config,
        device_str=args.device
    )

    logger.info("=== STEP 2/3: Extracting AFTER Road Network ===")
    gdf_after = run_single_image_pipeline(
        input_path=args.after,
        checkpoint_path=args.checkpoint,
        output_dir=after_dir,
        config=config,
        device_str=args.device
    )

    logger.info("=== STEP 3/3: Performing Temporal Geometric Change Matching ===")
    matcher = TemporalRoadMatcher(
        buffer_m=config.get("change_detection", {}).get("buffer_m", 3.5),
        min_change_length_m=args.min_change_length
    )
    new_roads_gdf, removed_roads_gdf, all_changes_gdf = matcher.match_networks(gdf_before, gdf_after)

    logger.info(f"Detected Changes: {len(new_roads_gdf)} NEW roads, {len(removed_roads_gdf)} REMOVED roads")

    # Export GIS change products
    if not new_roads_gdf.empty:
        GISExporter.export_all(new_roads_gdf, output_dir=changes_dir, base_name="new_roads")
    if not removed_roads_gdf.empty:
        GISExporter.export_all(removed_roads_gdf, output_dir=changes_dir, base_name="removed_roads")
    if not all_changes_gdf.empty:
        GISExporter.export_all(all_changes_gdf, output_dir=changes_dir, base_name="all_changes")

    logger.info(f"[SUCCESS] Change detection completed. All GIS outputs saved to: {changes_dir}")
