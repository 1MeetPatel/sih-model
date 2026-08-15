"""
Run Road Extraction Pipeline on user-uploaded Ahmedabad/River satellite image (media_1786803216552.png)
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

from src.models.model_factory import ModelFactory
from src.preprocessing.normalization import normalize_raster_patch
from src.postprocessing.threshold import apply_probability_threshold
from src.postprocessing.morphology import clean_road_mask
from src.postprocessing.skeleton import extract_road_skeleton

def main():
    img_path = r"C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786803216552.png"
    out_dir = r"C:\Users\91704\Desktop\sih\results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading image: {img_path}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print("Error reading image!")
        return

    # Check if there is a header bar (e.g. text "1. Original Satellite Raster...")
    # Let's crop if there is a black banner or text at the top
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    print(f"Original image size: {w}x{h}")

    # Load neural road detection model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelFactory.create_model(architecture="unet", encoder="resnet34", in_channels=3, classes=1)
    ckpt_path = "weights/road_detector_weights.pth"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Overlapping tile inference
    tile_size = 512
    overlap = 128
    stride = tile_size - overlap

    prob_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)
    
    gx = cv2.getGaussianKernel(tile_size, tile_size / 4.0)
    window_weight = np.outer(gx, gx).astype(np.float32)

    rows = sorted(list(set(list(range(0, h, stride)) + [max(0, h - tile_size)])))
    cols = sorted(list(set(list(range(0, w, stride)) + [max(0, w - tile_size)])))

    patches = []
    coords = []
    for r in rows:
        for c in cols:
            patch = rgb[r:min(r+tile_size, h), c:min(c+tile_size, w)]
            ph, pw, _ = patch.shape
            if ph != tile_size or pw != tile_size:
                patch = cv2.copyMakeBorder(patch, 0, tile_size - ph, 0, tile_size - pw, cv2.BORDER_REFLECT)
            norm_patch = normalize_raster_patch(patch)
            patches.append(norm_patch.transpose(2, 0, 1))
            coords.append((r, c, ph, pw))

    batch_size = 4
    for i in range(0, len(patches), batch_size):
        b_patches = patches[i:i+batch_size]
        b_coords = coords[i:i+batch_size]
        tensor = torch.from_numpy(np.stack(b_patches)).float().to(device)
        with torch.inference_mode():
            preds = model(tensor)
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            preds_np = preds.cpu().numpy()[:, 0, :, :]
        for idx, (r, c, ph, pw) in enumerate(b_coords):
            p = preds_np[idx][:ph, :pw]
            wt = window_weight[:ph, :pw]
            prob_accum[r:r+ph, c:c+pw] += p * wt
            weight_accum[r:r+ph, c:c+pw] += wt

    prob_map = prob_accum / np.maximum(weight_accum, 1e-6)
    prob_map = np.clip(prob_map, 0.0, 1.0)

    # Threshold & Clean
    binary_mask = apply_probability_threshold(prob_map, threshold=0.35)
    clean_mask_img = clean_road_mask(binary_mask, min_area_pixels=15)
    skel = extract_road_skeleton(clean_mask_img)

    # Dilate skeleton slightly for clean, crisp vector overlay
    skel_dilated = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))) > 0

    overlay = rgb.copy()
    overlay[skel_dilated] = [255, 0, 0]

    # Save standalone high-res overlay
    overlay_path = os.path.join(out_dir, "ahmedabad_test_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 4-Panel Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    axes[0].imshow(rgb)
    axes[0].set_title('1. Satellite Image (Ahmedabad)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_map, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=9)

    axes[2].imshow(clean_mask_img, cmap='gray')
    axes[2].set_title('3. Road Mask (Morphology Cleaned)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('4. Extracted Road Centerlines (Overlay)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    four_panel_path = os.path.join(out_dir, "ahmedabad_test_4panel.png")
    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Results generated:\n - {four_panel_path}\n - {overlay_path}")

if __name__ == "__main__":
    main()
