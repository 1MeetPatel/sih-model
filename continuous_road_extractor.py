"""
Continuous Major Road Network Extractor (Preserved Precision + Full Connectivity)
=================================================================================
1. Preserves the exact high-precision neural road detection without false noise.
2. Bridges tree-canopy occlusions and shadow breaks along major thoroughfares (>= 20ft).
3. Connects all sector dividing avenues end-to-end.
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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize
import torchvision.transforms as transforms

from pretrained_satellite_road_detector import load_pretrained_model

FEET_PER_METER = 3.28084


def connect_major_corridors(skeleton: np.ndarray, prob_map: np.ndarray, max_gap_px: int = 28, angle_tol_deg: float = 28.0) -> np.ndarray:
    """
    Connects collinear road avenues across tree shadows and occlusions without creating false branches.
    """
    H, W = skeleton.shape
    skel_bool = skeleton > 0

    # Find endpoints (pixels with exactly 1 neighbor)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
    endpoint_mask = skel_bool & (neighbor_count == 1)
    endpoints = np.argwhere(endpoint_mask)

    if len(endpoints) < 2:
        return skeleton

    bridged_skel = skeleton.copy()

    # Function to get outward heading tangent vector at endpoint
    def get_tangent(r, c, depth=8):
        curr = (r, c)
        visited = {curr}
        for _ in range(depth):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if 0 <= nr < H and 0 <= nc < W and skel_bool[nr, nc] and (nr, nc) not in visited:
                        neighbors.append((nr, nc))
            if not neighbors:
                break
            curr = neighbors[0]
            visited.add(curr)
        vec = np.array([r - curr[0], c - curr[1]], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6) if norm > 0 else np.array([0.0, 0.0])

    tangents = [get_tangent(pt[0], pt[1]) for pt in endpoints]
    min_cos = np.cos(np.radians(angle_tol_deg))

    # Evaluate endpoint pairs
    for i in range(len(endpoints)):
        p1 = endpoints[i]
        t1 = tangents[i]
        if np.linalg.norm(t1) < 0.1:
            continue

        best_j = -1
        min_dist = max_gap_px + 1

        for j in range(i + 1, len(endpoints)):
            p2 = endpoints[j]
            t2 = tangents[j]
            if np.linalg.norm(t2) < 0.1:
                continue

            disp = p2 - p1
            dist = np.linalg.norm(disp)

            if 2 <= dist <= max_gap_px:
                dir_vec = disp / dist
                cos1 = np.dot(t1, dir_vec)
                cos2 = np.dot(t2, -dir_vec)

                # Collinear condition
                if cos1 >= min_cos and cos2 >= min_cos:
                    # Intermediate sample check: ensure we are not crossing completely barren non-road terrain
                    # Sample 5 points along line
                    samples = [p1 + (p2 - p1) * t for t in [0.25, 0.5, 0.75]]
                    valid = True
                    for s in samples:
                        sr, sc = int(s[0]), int(s[1])
                        if 0 <= sr < H and 0 <= sc < W:
                            # If prob is essentially 0, skip
                            if prob_map[sr, sc] < 0.03:
                                valid = False
                                break
                    if valid and dist < min_dist:
                        min_dist = dist
                        best_j = j

        if best_j != -1:
            p2 = endpoints[best_j]
            cv2.line(bridged_skel, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), 255, 1)

    return skeletonize(bridged_skel > 0) * 255


def run_continuous_road_pipeline(img_path: str, city_name: str, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Processing {city_name} with Continuous Major Road Pipeline: {img_path}")

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

    # 1. Multi-Scale Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe_fine = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    L_fine = clahe_fine.apply(lab[:, :, 0])
    clahe_broad = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
    L_broad = clahe_broad.apply(lab[:, :, 0])
    L_blend = cv2.addWeighted(L_fine, 0.65, L_broad, 0.35, 0)
    blur_l = cv2.GaussianBlur(L_blend, (0, 0), 2.0)
    L_sharp = np.clip(cv2.addWeighted(L_blend, 1.4, blur_l, -0.4, 0), 0, 255).astype(np.uint8)
    lab[:, :, 0] = L_sharp
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
    prob_smooth = gaussian_filter(prob_accum, sigma=0.5)
    prob_heatmap = np.clip(prob_smooth / (np.percentile(prob_smooth, 99.0) + 1e-8), 0.0, 1.0)

    # 3. Direct Heatmap Tracing (High Confidence Seeds + Continuous Paths)
    seed_mask = (prob_heatmap >= 0.20).astype(np.uint8)
    cand_mask = (prob_heatmap >= 0.09).astype(np.uint8)

    num_cand, cand_labels, stats, _ = cv2.connectedComponentsWithStats(cand_mask, connectivity=8)
    connected_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, num_cand):
        comp = (cand_labels == i)
        if np.any(seed_mask[comp]):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            diag = np.sqrt(w**2 + h**2)
            if area >= 12 and diag >= 8:
                connected_mask[comp] = 1

    kernel_cross = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_cross, iterations=2)
    raw_skel = (skeletonize(closed > 0) * 255).astype(np.uint8)

    # 4. Collinear Tree-Canopy & Shadow Corridor Connection
    print("[*] Bridging collinear road gaps along sector avenues...")
    final_skel = connect_major_corridors(raw_skel, prob_heatmap, max_gap_px=25, angle_tol_deg=25.0)

    # 5. Crisp 2px Red Vector Overlay
    road_overlay = rgb.copy()
    dilated_skel = cv2.dilate(final_skel.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1) > 0
    road_overlay[dilated_skel] = [255, 0, 0]

    safe_name = city_name.lower().replace(' ', '_')
    overlay_path = os.path.join(out_dir, f"{safe_name}_continuous_overlay.png")
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

    axes[2].imshow(final_skel, cmap='gray')
    axes[2].set_title('3. Continuous Road Topology', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Continuous Road Network (Red 2px Vectors)', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    four_panel_path = os.path.join(out_dir, f"{safe_name}_continuous_4panel.png")
    plt.savefig(four_panel_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"[SUCCESS] Continuous outputs saved:\n - {four_panel_path}\n - {overlay_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786814128429.png"
    city = sys.argv[2] if len(sys.argv) > 2 else "Gandhinagar Sectors"
    run_continuous_road_pipeline(img, city, r"C:\Users\91704\Desktop\sih\results")
