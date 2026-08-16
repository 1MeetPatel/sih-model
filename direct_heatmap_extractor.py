"""
Direct Heatmap-Faithful Road Topology Extractor (Zero Ghost Roads)
==================================================================
Ensures Panel 3 & 4 contain lines ONLY where the Heatmap (Panel 2) is bright yellow/orange.
Zero lines in the dark purple/black background.
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
import torchvision.transforms as transforms

from pretrained_satellite_road_detector import load_pretrained_model

def run_direct_faithful_extraction(img_path: str, city_name: str, out_dir: str = "results", min_heatmap_prob: float = 0.28):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Processing {city_name} with Direct Heatmap Extraction: {img_path}")

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

    # 1. CLAHE Contrast Normalization
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 2. Neural Inference
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
    
    # Heatmap exactly in [0, 1]
    prob_heatmap = np.clip(prob_accum, 0.0, 1.0)

    # 3. DIRECT HEATMAP THRESHOLDING — ONLY THE BRIGHT GLOWING ROADS
    # Pixels in the dark purple/black region are strictly ignored!
    bright_road_mask = (prob_heatmap >= min_heatmap_prob).astype(np.uint8)

    # Filter isolated tiny noise dots (< 15 pixels)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_road_mask, connectivity=8)
    clean_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            clean_mask[labels == i] = 1

    # Close tiny 1px micro-gaps along the bright roads
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. SKELETONIZATION DIRECTLY FROM BRIGHT ROAD MASK
    skel = skeletonize(closed_mask > 0)

    # 5. Render clean 2px red vectors over satellite image
    road_overlay = rgb.copy()
    skel_u8 = (skel * 255).astype(np.uint8)
    dilated_skel = cv2.dilate(skel_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1) > 0
    road_overlay[dilated_skel] = [255, 0, 0]

    safe_name = city_name.lower().replace(' ', '_')
    overlay_path = os.path.join(out_dir, f"{safe_name}_faithful_direct_overlay.png")
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

    axes[2].imshow(skel, cmap='gray')
    axes[2].set_title('3. Road Topology (Direct from Heatmap)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Extracted Road Markings (Red 2px Vectors)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    four_panel_path = os.path.join(out_dir, f"{safe_name}_faithful_direct_4panel.png")
    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[SUCCESS]\n - {four_panel_path}\n - {overlay_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786904201993.png"
    city = sys.argv[2] if len(sys.argv) > 2 else "Kunpur Village"
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.28
    run_direct_faithful_extraction(img, city, r"C:\Users\91704\Desktop\sih\results", min_heatmap_prob=thresh)
