"""
Restore EXACT Heatmap + True Heatmap-Matched Topology
======================================================
1. Restores the exact multi-scale neural + curvilinear fusion heatmap that gave the brilliant results.
2. Extracts centerlines strictly from the bright glowing roads in Panel 2.
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
from skimage.filters import frangi
import torchvision.transforms as transforms

from pretrained_satellite_road_detector import load_pretrained_model

def extract_heatmap_centerlines(neural_prob_raw: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """
    Extracts centerlines STRICTLY from raw neural road probability.
    
    Key fixes:
    1. Uses RAW neural probability (no normalization) — prevents background amplification
    2. NO morphological closing — prevents Voronoi ghost polygon pattern from block-filling
    3. Only skeletonizes thin road strips as they appear in the neural output
    """
    H, W = neural_prob_raw.shape

    # Threshold directly on raw neural output
    # Roads: typically 0.40-0.95 | Background: typically 0.01-0.20
    road_mask = (neural_prob_raw >= threshold).astype(np.uint8)

    # Remove tiny isolated noise specks (< 20 px) — no closing to avoid block fill-in
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
    clean_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            clean_mask[labels == i] = 1

    # Skeletonize the thin road strips — gives clean 1px centerlines
    return skeletonize(clean_mask > 0)




def run(img_path: str, city_name: str, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Processing {city_name} with Exact Heatmap Engine: {img_path}")

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[!] Error loading image: {img_path}")
        return

    H_orig, W_orig, _ = bgr.shape
    top, bottom, left, right = 0, 0, 0, 0
    if H_orig > 300:
        top, bottom = 50, 45
    if W_orig > 300:
        left, right = 15, 15
    bgr_cropped = bgr[top:H_orig-bottom, left:W_orig-right] if bottom > 0 else bgr
    H, W, _ = bgr_cropped.shape
    rgb = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2RGB)

    # 1. EXACT Multi-Scale CLAHE Enhancement
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    clahe_fine = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    L_fine = clahe_fine.apply(L)
    clahe_broad = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
    L_broad = clahe_broad.apply(L)
    L_blend = cv2.addWeighted(L_fine, 0.65, L_broad, 0.35, 0)
    blur_l = cv2.GaussianBlur(L_blend, (0, 0), 2.0)
    L_sharp = np.clip(cv2.addWeighted(L_blend, 1.4, blur_l, -0.4, 0), 0, 255).astype(np.uint8)
    lab[:, :, 0] = L_sharp
    enh_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 2. Neural Model Inference
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
    neural_prob = np.clip(prob_accum, 0.0, 1.0)

    # 3. DISPLAY Heatmap — neural + frangi fusion for visual richness (Panel 2 only)
    ridges_b = frangi(gray, sigmas=[1.2, 2.0, 3.2, 4.8], black_ridges=False)
    ridges_d = frangi(gray, sigmas=[1.2, 2.0, 3.2, 4.8], black_ridges=True)
    ridges = np.maximum(ridges_b, ridges_d)
    ridges_norm = ridges / (np.percentile(ridges, 99.0) + 1e-8)
    ridges_norm = np.clip(ridges_norm, 0.0, 1.0)

    fused_prob = neural_prob * 0.65 + ridges_norm * 0.35
    prob_smooth = gaussian_filter(fused_prob, sigma=0.6)
    # This is only used for the beautiful Panel 2 heatmap display
    prob_heatmap = np.clip(prob_smooth / (np.percentile(prob_smooth, 99.2) + 1e-8), 0.0, 1.0)

    # 4. TWO-TIER EXTRACTION — color-coded by heatmap confidence
    # Tier 1 (Bright RED)   : neural_prob >= 0.50 — bright yellow roads in heatmap
    # Tier 2 (ORANGE)       : neural_prob >= 0.20 — dimmer orange roads in heatmap
    # Background (~0.0-0.15): excluded — no ghost roads
    print("[*] Extracting two-tier roads (bright=0.50 red, dim=0.20 orange)...")
    neural_smooth = gaussian_filter(neural_prob, sigma=0.5)

    skel_bright = extract_heatmap_centerlines(neural_smooth, threshold=0.50)  # high confidence
    skel_dim    = extract_heatmap_centerlines(neural_smooth, threshold=0.20)  # all visible roads
    # Dim skeleton = all roads; bright skeleton = main roads only
    # Dim-only = secondary roads (dimmer in heatmap)
    skel_secondary = skel_dim & ~skel_bright

    # 5. Color-coded overlay:
    #    🔴 Bright RED (2px)   = high-confidence main roads
    #    🟠 Orange (2px)       = dimmer secondary roads visible in heatmap
    road_overlay = rgb.copy()
    dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Draw dimmer/secondary roads first (orange), then main roads on top (red)
    dim_u8    = (skel_secondary.astype(np.uint8) * 255)
    bright_u8 = (skel_bright.astype(np.uint8) * 255)
    dim_dilated    = cv2.dilate(dim_u8,    dil, iterations=1) > 0
    bright_dilated = cv2.dilate(bright_u8, dil, iterations=1) > 0

    road_overlay[dim_dilated]    = [255, 0, 0]   # Red for all roads (uniform)
    road_overlay[bright_dilated] = [255, 0, 0]   # Red for all roads (uniform)

    safe_name = city_name.lower().replace(' ', '_')
    overlay_path = os.path.join(out_dir, f"{safe_name}_exact_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))

    # 6. 4-Panel Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    axes[0].imshow(rgb)
    axes[0].set_title(f'1. Satellite Image ({city_name})', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    fig.colorbar(im2, cax=cax).ax.tick_params(labelsize=9)

    axes[2].imshow(skel_dim, cmap='gray')
    axes[2].set_title('3. Road Topology (Direct from Heatmap)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Extracted Road Markings (Red — All Detected Roads)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    four_panel_path = os.path.join(out_dir, f"{safe_name}_exact_4panel.png")
    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[SUCCESS]\n - {four_panel_path}\n - {overlay_path}")


if __name__ == "__main__":
    img  = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786904201993.png"
    city = sys.argv[2] if len(sys.argv) > 2 else "Kunpur Village"
    run(img, city, r"C:\Users\91704\Desktop\sih\results")
