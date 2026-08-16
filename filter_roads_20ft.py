"""
20ft+ Road Extraction & Physical Width Filtering Engine
========================================================
Extracts and isolates roads approximately 20 feet (6.1 meters) or wider
from satellite imagery and GeoTIFFs using Euclidean Distance Transform (EDT).

Outputs:
1. Original Satellite Image
2. Road Probability Heatmap
3. Width Classification Map (Highways >=40ft vs Arterials 20-40ft vs Filtered <20ft)
4. Qualifying Roads Overlay (Only >= 20 feet / 6.1 meters)
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize
import torchvision.transforms as transforms

from pretrained_satellite_road_detector import load_pretrained_model

FEET_PER_METER = 3.28084
MIN_WIDTH_METERS = 6.096  # 20.0 feet


def run_20ft_road_extraction(
    img_path: str,
    city_name: str,
    gsd_m: float = 1.0,  # Ground sampling distance in meters per pixel
    min_width_ft: float = 20.0,
    out_dir: str = "results"
):
    os.makedirs(out_dir, exist_ok=True)
    min_width_m = min_width_ft / FEET_PER_METER
    print("=" * 70)
    print(f"  20ft+ (6.1m+) Road Extraction Engine — {city_name}")
    print(f"  GSD: {gsd_m:.2f} m/px | Target Min Width: {min_width_ft:.1f} ft ({min_width_m:.2f} m)")
    print("=" * 70)

    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    H_orig, W_orig, _ = bgr.shape

    # Auto-crop Google Earth borders if present
    top, bottom, left, right = 0, 0, 0, 0
    if H_orig > 300:
        top, bottom = 50, 45
    if W_orig > 300:
        left, right = 15, 15
    bgr_cropped = bgr[top:H_orig-bottom, left:W_orig-right] if bottom > 0 else bgr
    H, W, _ = bgr_cropped.shape
    rgb = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2RGB)
    print(f"[*] Working raster resolution: {W} x {H} px")

    # 2. Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 3. Neural Road Segmentation Inference
    device = torch.device('cpu')
    model = load_pretrained_model('weights/best_road_seg_unet.pth', device=device)

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

            act_h = min(tile_size, H - y)
            act_w = min(tile_size, W - x)
            prob_accum[y:y+act_h, x:x+act_w] += pred[:act_h, :act_w] * g_weight[:act_h, :act_w]
            weight_accum[y:y+act_h, x:x+act_w] += g_weight[:act_h, :act_w]

    prob_accum /= np.maximum(weight_accum, 1e-6)
    prob_smooth = gaussian_filter(prob_accum, sigma=0.5)
    prob_heatmap = np.clip(prob_smooth / (np.percentile(prob_smooth, 99.0) + 1e-8), 0.0, 1.0)

    # 4. Binary Segmentation & Centerline Skeleton
    raw_binary = (prob_heatmap >= 0.22).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_binary = cv2.morphologyEx(raw_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    full_skel = skeletonize(clean_binary > 0)

    # 5. Euclidean Distance Transform (EDT) for Physical Width Estimation
    print("[*] Computing Euclidean Distance Transform along road centerlines...")
    dist_map_px = distance_transform_edt(clean_binary > 0)
    
    # Full road width in pixels = 2 * distance to boundary
    # Full road width in feet = 2 * distance * gsd_m * 3.28084
    width_map_ft = (2.0 * dist_map_px * gsd_m) * FEET_PER_METER

    # 6. Segment-by-Segment Geometric Width Analysis
    skel_u8 = (full_skel * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel_u8, connectivity=8)

    qualifying_skel_20ft = np.zeros((H, W), dtype=np.uint8)
    highways_skel_40ft = np.zeros((H, W), dtype=np.uint8)
    arterials_skel_20_40ft = np.zeros((H, W), dtype=np.uint8)
    filtered_skel_sub20ft = np.zeros((H, W), dtype=np.uint8)

    total_segments = 0
    qualifying_count = 0
    filtered_count = 0
    total_qualifying_len_km = 0.0

    # Width visualization canvas
    width_vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Dark subtle background
    width_vis_rgb[:] = (20, 24, 30)

    for seg_id in range(1, num_labels):
        seg_pts = np.argwhere(labels == seg_id)  # (N, 2) [row, col]
        if len(seg_pts) < 4:
            continue

        total_segments += 1
        seg_widths_ft = width_map_ft[seg_pts[:, 0], seg_pts[:, 1]]
        
        mean_w_ft = float(np.mean(seg_widths_ft))
        median_w_ft = float(np.median(seg_widths_ft))
        max_w_ft = float(np.max(seg_widths_ft))
        seg_len_km = (len(seg_pts) * gsd_m) / 1000.0

        # Qualification check: >= 20 feet (6.1 meters)
        is_qualifying = (mean_w_ft >= min_width_ft) or (median_w_ft >= (min_width_ft * 0.85)) or (max_w_ft >= (min_width_ft * 1.25))

        if is_qualifying:
            qualifying_count += 1
            total_qualifying_len_km += seg_len_km
            qualifying_skel_20ft[seg_pts[:, 0], seg_pts[:, 1]] = 255

            if mean_w_ft >= 40.0:  # Major Highway / Expressway (>= 40ft)
                highways_skel_40ft[seg_pts[:, 0], seg_pts[:, 1]] = 255
                width_vis_rgb[seg_pts[:, 0], seg_pts[:, 1]] = (255, 60, 60)  # Bright Red
            else:  # Main City Road / Arterial (20ft - 40ft)
                arterials_skel_20_40ft[seg_pts[:, 0], seg_pts[:, 1]] = 255
                width_vis_rgb[seg_pts[:, 0], seg_pts[:, 1]] = (255, 180, 20)  # Amber Orange
        else:
            filtered_count += 1
            filtered_skel_sub20ft[seg_pts[:, 0], seg_pts[:, 1]] = 255
            width_vis_rgb[seg_pts[:, 0], seg_pts[:, 1]] = (90, 100, 115)  # Muted Grey (Filtered)

    print(f"[*] Total Road Segments Analyzed: {total_segments}")
    print(f"    [+] QUALIFIED (>= 20ft / 6.1m): {qualifying_count} segments ({total_qualifying_len_km:.2f} km total length)")
    print(f"    [-] FILTERED OUT (< 20ft):     {filtered_count} narrow alleys/footpaths")

    # 7. Render Final Qualifying Road Overlay (2px clean red vectors)
    final_overlay = rgb.copy()
    dilated_20ft = cv2.dilate(qualifying_skel_20ft, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1) > 0
    final_overlay[dilated_20ft] = [255, 0, 0]

    safe_name = city_name.lower().replace(' ', '_')
    overlay_path = os.path.join(out_dir, f"{safe_name}_20ft_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))

    # Dilate width visualization for clean visibility in panel 3
    dilated_vis = cv2.dilate(width_vis_rgb, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    # 8. Render 4-Panel Analysis Figure
    fig, axes = plt.subplots(1, 4, figsize=(26, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.92, bottom=0.04)

    axes[0].imshow(rgb)
    axes[0].set_title(f'1. Satellite Image ({city_name})', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    fig.colorbar(im2, cax=cax).ax.tick_params(labelsize=9)

    axes[2].imshow(dilated_vis)
    axes[2].set_title('3. Road Width Classification', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Legend for Panel 3
    legend_elements = [
        mpatches.Patch(color='#FF3C3C', label='Highways / Expressways (>= 40 ft)'),
        mpatches.Patch(color='#FFB414', label='Arterials / Main Roads (20 - 40 ft)'),
        mpatches.Patch(color='#5A6473', label='Filtered Alleys / Paths (< 20 ft)'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.85, facecolor='#1E2228', labelcolor='white')

    axes[3].imshow(final_overlay)
    axes[3].set_title(f'4. Qualifying Roads (>= 20ft / 6.1m)\n[{qualifying_count} segments | {total_qualifying_len_km:.1f} km total]', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    four_panel_path = os.path.join(out_dir, f"{safe_name}_20ft_4panel.png")
    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[SUCCESS]\n - {four_panel_path}\n - {overlay_path}")
    print("=" * 70)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786814128429.png"
    city = sys.argv[2] if len(sys.argv) > 2 else "Gandhinagar Sectors"
    gsd = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    run_20ft_road_extraction(img, city, gsd_m=gsd, min_width_ft=20.0, out_dir=r"C:\Users\91704\Desktop\sih\results")
