"""
Thin-Line Satellite Road Network Detector (Maximum Map Visibility)
==================================================================
- Pre-Trained Satellite Road U-Net
- Skeleton Centerline Vectorization
- Ultra-Crisp Thin Road Lines (1-2px) for 100% Satellite Map Visibility
- 4-Panel Analysis + High-Resolution Thin-Line Overlay
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pretrained_satellite_road_detector import PretrainedRoadUNet, load_pretrained_model


def run_thin_line_road_detection(
    input_path: str,
    output_4panel: str = 'results/thin_roads_4panel.png',
    output_overlay: str = 'results/thin_roads_overlay.png',
    weights_path: str = 'weights/best_road_seg_unet.pth',
    line_width: int = 2,
    line_color: str = 'red',  # 'red' or 'yellow'
    high_threshold: float = 0.22,
    low_threshold: float = 0.12,
    crop_ge_ui: bool = True
):
    print("=" * 65)
    print("  Thin-Line Satellite Road Extraction (High Map Visibility)")
    print("=" * 65)

    device = torch.device('cpu')
    model = load_pretrained_model(weights_path, device=device)

    # 1. Load Image
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    H_orig, W_orig, _ = bgr.shape
    if crop_ge_ui and (H_orig > 300 and W_orig > 300):
        bgr = bgr[40:H_orig-60, 20:W_orig-20]
    
    H, W, _ = bgr.shape
    print(f"[*] Scene resolution: {W} x {H} px")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Contrast Balancing
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh_rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # 3. Dense Multi-Scale Neural Inference
    print("[*] Running Neural Inference across image tiles...")
    tile_size = 256
    stride = 64
    
    prob_accum = np.zeros((H, W), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor()])

    gx = cv2.getGaussianKernel(tile_size, tile_size / 4)
    g_weight = np.outer(gx, gx).astype(np.float32)
    g_weight /= g_weight.max()

    for y in range(0, max(1, H - tile_size + 1), stride):
        for x in range(0, max(1, W - tile_size + 1), stride):
            patch = enh_rgb[y:y+tile_size, x:x+tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                patch = cv2.resize(patch, (tile_size, tile_size))
            
            inp = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).cpu().numpy()[0, 0]
            
            prob_accum[y:y+tile_size, x:x+tile_size] += pred[:tile_size, :tile_size] * g_weight
            weight_accum[y:y+tile_size, x:x+tile_size] += g_weight

    prob_accum /= np.maximum(weight_accum, 1e-6)
    prob_heatmap = np.clip(prob_accum, 0.0, 1.0)

    # 4. Dual-Threshold Hysteresis Path Extraction
    print("[*] Tracing complete road network paths...")
    seed_mask = (prob_heatmap >= high_threshold).astype(np.uint8)
    candidate_mask = (prob_heatmap >= low_threshold).astype(np.uint8)

    num_cand, cand_labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    connected_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, num_cand):
        comp_bool = (cand_labels == i)
        if np.any(seed_mask[comp_bool]):
            area = stats[i, cv2.CC_STAT_AREA]
            w_c = stats[i, cv2.CC_STAT_WIDTH]
            h_c = stats[i, cv2.CC_STAT_HEIGHT]
            diag = np.sqrt(w_c**2 + h_c**2)
            asp = max(w_c, h_c) / (min(w_c, h_c) + 1e-6)
            if area >= 15 and (diag >= 12 or asp >= 1.6):
                connected_mask[comp_bool] = 1

    # 5. Extract Exact 1-Pixel Skeleton Centerlines
    print("[*] Computing 1-pixel topological centerlines...")
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_cross, iterations=2)
    skel = skeletonize(closed > 0)

    # 6. Render Ultra-Crisp Thin Lines Over the Satellite Map
    print(f"[*] Rendering crisp {line_width}px thin road lines (color={line_color})...")
    road_overlay = rgb.copy()
    
    # Choose color
    if line_color == 'red':
        c_core = (255, 0, 0)
        c_glow = (180, 0, 0)
    elif line_color == 'yellow':
        c_core = (255, 220, 0)
        c_glow = (180, 150, 0)
    else:
        c_core = (255, 0, 0)
        c_glow = (180, 0, 0)

    # Convert skeleton to thin connected line segments
    skel_u8 = (skel * 255).astype(np.uint8)
    
    # Draw thin lines with anti-aliasing
    if line_width == 1:
        road_overlay[skel] = c_core
    elif line_width == 2:
        # 2px smooth line with 1px soft edge for maximum clarity
        dilated_skel = cv2.dilate(skel_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1) > 0
        road_overlay[dilated_skel] = c_core
    else:
        # Custom width
        dilated_skel = cv2.dilate(skel_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_width)), iterations=1) > 0
        road_overlay[dilated_skel] = c_core

    os.makedirs(os.path.dirname(output_overlay) if os.path.dirname(output_overlay) else '.', exist_ok=True)
    cv2.imwrite(output_overlay, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))
    print(f"[+] High-Visibility Thin Road Overlay saved -> {output_overlay}")

    # 7. Render 4-Panel Visualization
    print(f"[*] Rendering 4-Panel Visualization -> {output_4panel}")
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    axes[0].imshow(rgb)
    axes[0].set_title('1. Original Satellite Raster (RGB)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=9)

    axes[2].imshow(skel, cmap='gray')
    axes[2].set_title('3. Road Centerlines / Topology', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title(f'4. Thin Road Network ({line_color.capitalize()}, {line_width}px)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.savefig(output_4panel, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] 4-Panel Analysis saved -> {output_4panel}")
    print("=" * 65)
    print("  [SUCCESS] Thin-line road detection complete!")
    print("=" * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thin-Line Satellite Road Extraction")
    parser.add_argument('--input', required=True, help='Path to satellite image')
    parser.add_argument('--output-4panel', default='results/thin_roads_4panel.png')
    parser.add_argument('--output-overlay', default='results/thin_roads_overlay.png')
    parser.add_argument('--line-width', type=int, default=2)
    parser.add_argument('--color', default='red', choices=['red', 'yellow'])
    parser.add_argument('--high-threshold', type=float, default=0.22)
    parser.add_argument('--low-threshold', type=float, default=0.12)
    args = parser.parse_args()

    run_thin_line_road_detection(
        input_path=args.input,
        output_4panel=args.output_4panel,
        output_overlay=args.output_overlay,
        line_width=args.line_width,
        line_color=args.color,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold
    )
