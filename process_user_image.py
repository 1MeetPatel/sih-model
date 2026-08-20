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
import shutil

from pretrained_satellite_road_detector import load_pretrained_model

def process_satellite_image(img_path, out_dir="results", artifact_dir=None, prefix="scene"):
    os.makedirs(out_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    print(f"[*] Loading input image: {img_path}")
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Could not load image from {img_path}")

    H_orig, W_orig, C = bgr.shape
    print(f"[*] Input Resolution: {W_orig} x {H_orig} px, Channels: {C}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape

    # 1. Multi-Scale Contrast Enhancement (CLAHE + unsharp mask)
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

    # 2. Neural Model Inference (Batched Sliding Window with Gaussian Blending)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    weights_path = 'weights/best_road_seg_unet.pth'
    model = load_pretrained_model(weights_path, device=device)
    model.eval()

    tile_size = 256
    stride = 64
    prob_accum = np.zeros((H, W), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor()])

    gx = cv2.getGaussianKernel(tile_size, tile_size / 4)
    g_weight = np.outer(gx, gx).astype(np.float32)
    g_weight /= g_weight.max()

    print("[*] Collecting tiles for batched neural inference...")
    y_steps = list(range(0, max(1, H - tile_size + 1), stride))
    if (H - tile_size) not in y_steps and H >= tile_size:
        y_steps.append(H - tile_size)
    x_steps = list(range(0, max(1, W - tile_size + 1), stride))
    if (W - tile_size) not in x_steps and W >= tile_size:
        x_steps.append(W - tile_size)

    patches = []
    coords = []
    for y in y_steps:
        for x in x_steps:
            patch = enh_rgb[y:y+tile_size, x:x+tile_size]
            act_h, act_w, _ = patch.shape
            if act_h < tile_size or act_w < tile_size:
                patch = cv2.copyMakeBorder(patch, 0, tile_size - act_h, 0, tile_size - act_w, cv2.BORDER_REFLECT)
            tensor_patch = transform(patch)
            patches.append(tensor_patch)
            coords.append((y, x, act_h, act_w))

    batch_size = 8
    print(f"[*] Total patches: {len(patches)}, processing in batches of {batch_size}...")
    for i in range(0, len(patches), batch_size):
        b_patches = torch.stack(patches[i:i+batch_size]).to(device)
        b_coords = coords[i:i+batch_size]
        with torch.no_grad():
            preds = model(b_patches).cpu().numpy()[:, 0, :, :]

        for idx, (y, x, act_h, act_w) in enumerate(b_coords):
            p = preds[idx][:act_h, :act_w]
            wt = g_weight[:act_h, :act_w]
            prob_accum[y:y+act_h, x:x+act_w] += p * wt
            weight_accum[y:y+act_h, x:x+act_w] += wt

    prob_accum /= np.maximum(weight_accum, 1e-6)
    neural_prob = np.clip(prob_accum, 0.0, 1.0)

    # 3. Curvilinear Ridge Continuity & Heatmap Generation for Display
    print("[*] Extracting ridge continuity features...")
    ridges_b = frangi(gray, sigmas=[1.2, 2.0, 3.2, 4.8], black_ridges=False)
    ridges_d = frangi(gray, sigmas=[1.2, 2.0, 3.2, 4.8], black_ridges=True)
    ridges = np.maximum(ridges_b, ridges_d)
    ridges_norm = ridges / (np.percentile(ridges, 99.0) + 1e-8)
    ridges_norm = np.clip(ridges_norm, 0.0, 1.0)

    fused_prob = neural_prob * 0.70 + ridges_norm * 0.30
    prob_smooth = gaussian_filter(fused_prob, sigma=0.6)
    prob_heatmap = np.clip(prob_smooth / (np.percentile(prob_smooth, 99.2) + 1e-8), 0.0, 1.0)

    # 4. Neural-Guided Topological Centerline Extraction (Eliminates Forest False Positives)
    print("[*] Extracting topological centerlines strictly from neural road probabilities...")
    neural_smooth = gaussian_filter(neural_prob, sigma=0.5)

    # Threshold on neural detection (roads > 0.28, background/forest < 0.15)
    road_mask = (neural_smooth >= 0.28).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
    clean_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            clean_mask[labels == i] = 1

    skel = skeletonize(clean_mask > 0)

    # 5. Vector Overlay (Crisp Red Lines on original image)
    road_overlay = rgb.copy()
    dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    skel_u8 = (skel.astype(np.uint8) * 255)
    dilated = cv2.dilate(skel_u8, dil, iterations=1) > 0

    # Overlay with high visibility Red
    road_overlay[dilated] = [255, 0, 0]

    # Save results
    overlay_filename = f"{prefix}_road_overlay.png"
    four_panel_filename = f"{prefix}_road_4panel.png"

    overlay_path = os.path.join(out_dir, overlay_filename)
    four_panel_path = os.path.join(out_dir, four_panel_filename)

    cv2.imwrite(overlay_path, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))

    # 4-Panel Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    axes[0].imshow(rgb)
    axes[0].set_title('1. Input Satellite Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    fig.colorbar(im2, cax=cax).ax.tick_params(labelsize=9)

    axes[2].imshow(skel, cmap='gray')
    axes[2].set_title('3. Extracted Road Network Topology', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Road Centerlines Overlay (Red)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    # If artifact_dir is specified, also copy there
    if artifact_dir:
        shutil.copy2(overlay_path, os.path.join(artifact_dir, overlay_filename))
        shutil.copy2(four_panel_path, os.path.join(artifact_dir, four_panel_filename))

    print(f"[SUCCESS] Execution completed!\n - 4-Panel: {four_panel_path}\n - Overlay: {overlay_path}")
    return four_panel_path, overlay_path

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\91704\.gemini\antigravity-ide\brain\7c827e02-2c75-495d-84bc-c8fcac11281a\.user_uploaded\media_1787164503305.png"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "scene5"
    out_dir = r"c:\Users\91704\Desktop\sih1\results"
    artifact_dir = r"C:\Users\91704\.gemini\antigravity-ide\brain\7c827e02-2c75-495d-84bc-c8fcac11281a"
    process_satellite_image(img_path, out_dir, artifact_dir, prefix=prefix)
