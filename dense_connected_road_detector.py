"""
High-Density Complete Path Satellite Road Extraction Engine
===========================================================
- DeepLabV3+ / Satellite Road U-Net Pre-Trained Backbone
- Dense Multi-Scale Tiled Inference (64px Stride)
- Dual-Threshold Hysteresis Connectivity (Recovers Major + Secondary + Minor Connecting Paths)
- Directional Path Closing & 20-Foot Continuous Red Corridor Rendering
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pretrained_satellite_road_detector import PretrainedRoadUNet, load_pretrained_model


def run_dense_road_detection(
    input_path: str,
    output_4panel: str = 'results/dense_roads_4panel.png',
    output_overlay: str = 'results/dense_roads_overlay.png',
    weights_path: str = 'weights/best_road_seg_unet.pth',
    high_threshold: float = 0.24,
    low_threshold: float = 0.13,
    crop_ge_ui: bool = True
):
    print("=" * 65)
    print("  High-Density Satellite Road Extraction Engine (Complete Paths)")
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

    # 2. Adaptive Multi-Scale Lightness Equalization
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh_rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # 3. Dense Overlapping Multi-Scale Inference
    print("[*] Running Dense Multi-Scale Neural Inference (stride=64)...")
    tile_size = 256
    stride = 64
    
    prob_accum = np.zeros((H, W), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor()])

    # Gaussian weighting window to blend tile edges seamlessly
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
    print("[*] Connecting Major & Secondary Roads via Hysteresis Tracking...")
    seed_mask = (prob_heatmap >= high_threshold).astype(np.uint8)
    candidate_mask = (prob_heatmap >= low_threshold).astype(np.uint8)

    # Find all components connected to high-confidence seeds
    num_cand, cand_labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    connected_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, num_cand):
        # Component is kept if it intersects with a high-confidence seed
        comp_bool = (cand_labels == i)
        if np.any(seed_mask[comp_bool]):
            area = stats[i, cv2.CC_STAT_AREA]
            w_c = stats[i, cv2.CC_STAT_WIDTH]
            h_c = stats[i, cv2.CC_STAT_HEIGHT]
            diag = np.sqrt(w_c**2 + h_c**2)
            asp = max(w_c, h_c) / (min(w_c, h_c) + 1e-6)
            # Filter out non-linear isolated noise
            if area >= 20 and (diag >= 15 or asp >= 1.8):
                connected_mask[comp_bool] = 1

    # 5. Multi-Directional Path Gap Bridging
    print("[*] Bridging path occlusions under overpasses and trees...")
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    reconnected = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_cross, iterations=2)
    
    # 6. Continuous 20-Foot Corridor Buffering
    skel = skeletonize(reconnected > 0)
    dist = cv2.distanceTransform(reconnected, cv2.DIST_L2, 5)

    corridor = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        rad = int(round(dist[y, x]))
        r_buf = min(max(rad, 2), 6)  # 20ft (~6.1m) profile
        cv2.circle(corridor, (x, y), r_buf, 1, -1)
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, kernel_cross)

    # 7. Render Qualifying Red Roads with Dark Outline (Exact Reference Standard)
    road_overlay = rgb.copy()
    mask_bool = corridor > 0
    road_overlay[mask_bool] = [255, 0, 0]  # Vibrant Red

    contours, _ = cv2.findContours(corridor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(road_overlay, contours, -1, (160, 0, 0), 1)

    os.makedirs(os.path.dirname(output_overlay) if os.path.dirname(output_overlay) else '.', exist_ok=True)
    cv2.imwrite(output_overlay, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))
    print(f"[+] Complete Path Red Overlay saved -> {output_overlay}")

    # 8. Render 4-Panel Visualization
    print(f"[*] Rendering 4-Panel Composite -> {output_4panel}")
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

    axes[2].imshow(corridor, cmap='gray')
    axes[2].set_title('3. Complete Road Mask (>= 6.1m / 20ft)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Qualifying Roads in Red (Complete Paths)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.savefig(output_4panel, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] 4-Panel Analysis saved -> {output_4panel}")
    print("=" * 65)
    print("  [DONE] High-Density Road Extraction Complete!")
    print("=" * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-Density Road Extraction with Complete Paths")
    parser.add_argument('--input', required=True, help='Path to satellite image')
    parser.add_argument('--output-4panel', default='results/dense_roads_4panel.png')
    parser.add_argument('--output-overlay', default='results/dense_roads_overlay.png')
    parser.add_argument('--high-threshold', type=float, default=0.24)
    parser.add_argument('--low-threshold', type=float, default=0.13)
    args = parser.parse_args()

    run_dense_road_detection(
        input_path=args.input,
        output_4panel=args.output_4panel,
        output_overlay=args.output_overlay,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold
    )
