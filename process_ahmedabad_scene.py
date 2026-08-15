"""
Process user's uploaded Ahmedabad satellite scene (media_1786807144997.jpg)
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
from skimage.morphology import skeletonize
from skimage.filters import frangi
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms

from pretrained_satellite_road_detector import load_pretrained_model

def main():
    img_path = r"C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786807144997.jpg"
    out_dir = r"C:\Users\91704\Desktop\sih\results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Loading satellite image: {img_path}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print("[!] Error loading image")
        return

    H_orig, W_orig, _ = bgr.shape
    print(f"[*] Original image resolution: {W_orig} x {H_orig} px")

    # Crop Google Earth UI headers and footers (top 50px, bottom 50px, left/right 15px)
    bgr_cropped = bgr[50:H_orig-45, 15:W_orig-15]
    H, W, _ = bgr_cropped.shape
    print(f"[*] Working raster resolution (UI cropped): {W} x {H} px")
    rgb = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2RGB)

    # Multi-Scale Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe_fine = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    L_fine = clahe_fine.apply(lab[:, :, 0])
    clahe_broad = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
    L_broad = clahe_broad.apply(lab[:, :, 0])
    L_blend = cv2.addWeighted(L_fine, 0.65, L_broad, 0.35, 0)
    blur = cv2.GaussianBlur(L_blend, (0, 0), 2.0)
    L_sharp = np.clip(cv2.addWeighted(L_blend, 1.4, blur, -0.4, 0), 0, 255).astype(np.uint8)
    lab[:, :, 0] = L_sharp
    enh_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Neural Inference
    device = torch.device('cpu')
    model = load_pretrained_model('weights/best_road_seg_unet.pth', device=device)

    print("[*] Running dense neural inference...")
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

    # Ridge continuity
    print("[*] Computing curvilinear road continuity...")
    ridges_b = frangi(gray, sigmas=[1.2, 2.0, 3.2, 4.8], black_ridges=False)
    ridges_d = frangi(gray, sigmas=[1.2, 2.0, 3.2, 4.8], black_ridges=True)
    ridges = np.maximum(ridges_b, ridges_d)
    ridges_norm = ridges / (np.percentile(ridges, 99.0) + 1e-8)
    ridges_norm = np.clip(ridges_norm, 0.0, 1.0)

    # Hybrid fusion
    fused_prob = neural_prob * 0.65 + ridges_norm * 0.35
    prob_smooth = gaussian_filter(fused_prob, sigma=0.6)
    prob_heatmap = np.clip(prob_smooth / (np.percentile(prob_smooth, 99.2) + 1e-8), 0.0, 1.0)

    # Dual-threshold hysteresis
    print("[*] Extracting connected road centerlines...")
    seed_mask = (prob_heatmap >= 0.20).astype(np.uint8)
    candidate_mask = (prob_heatmap >= 0.09).astype(np.uint8)

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
            if area >= 12 and (diag >= 10 or asp >= 1.5):
                connected_mask[comp_bool] = 1

    kernel_cross = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_cross, iterations=2)
    skel = skeletonize(closed > 0)

    # Thin crisp 2px red road overlay
    print("[*] Rendering road overlay...")
    road_overlay = rgb.copy()
    skel_u8 = (skel * 255).astype(np.uint8)
    dilated = cv2.dilate(skel_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1) > 0
    road_overlay[dilated] = [255, 0, 0]

    overlay_path = os.path.join(out_dir, "ahmedabad_full_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))

    # 4-Panel Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    axes[0].imshow(rgb)
    axes[0].set_title('1. Satellite Image (Ahmedabad, India)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=9)

    axes[2].imshow(skel, cmap='gray')
    axes[2].set_title('3. Extracted Road Network Topology', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Thin Road Network (Red 2px Vectors)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    four_panel_path = os.path.join(out_dir, "ahmedabad_full_4panel.png")
    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[SUCCESS] Output images generated:\n - {four_panel_path}\n - {overlay_path}")

if __name__ == "__main__":
    main()
