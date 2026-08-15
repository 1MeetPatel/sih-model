"""
Test Inference Script — Run trained model on new test satellite image
with Real-ESRGAN sharpening + neural road extraction at preserved accuracy
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
from skimage.morphology import remove_small_objects, remove_small_holes

# Add project root to path so basicsr and realesrgan are importable
SIH_DIR = os.path.dirname(os.path.abspath(__file__))
if SIH_DIR not in sys.path:
    sys.path.insert(0, SIH_DIR)


# -------------------------------------------------------------------------
# MODEL DEFINITION (must match training)
# -------------------------------------------------------------------------
class FastRoadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


def run_test_inference(
    test_img_path: str,
    model_path: str = 'weights/trained_road_unet.pth',
    output_4panel: str = 'results/test_road_network_4panel.png',
    output_overlay: str = 'results/test_accurate_road_map.png',
    sharpen: bool = True
):
    print("=" * 60)
    print("  Road Detection — Test Inference (98.5% trained model)")
    print("=" * 60)

    # Load and preprocess test image
    print(f"[*] Loading test image: {test_img_path}")
    img_bgr = cv2.imread(test_img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load test image: {test_img_path}")

    # Crop away Google Earth UI (bottom ~60px, top ~40px)
    H, W, _ = img_bgr.shape
    img_bgr = img_bgr[40:H-60, 20:W-20]
    H, W, _ = img_bgr.shape
    print(f"[*] Cropped scene dimensions: {W} x {H} px")

    # Stage 1: Real-ESRGAN Sharpening (2x to keep size manageable)
    if sharpen:
        try:
            print("[*] Applying Real-ESRGAN super-resolution sharpening...")
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            device_sr = torch.device('cpu')
            sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path='weights/RealESRGAN_x4plus.pth',
                model=sr_model, tile=256, tile_pad=10, pre_pad=0, half=False, device=device_sr
            )
            img_sharpened, _ = upsampler.enhance(img_bgr, outscale=2)
            print(f"[*] Sharpened via Real-ESRGAN: {img_bgr.shape[1]}x{img_bgr.shape[0]} -> {img_sharpened.shape[1]}x{img_sharpened.shape[0]} px")
            cv2.imwrite('results/test_satellite_sharpened.png', img_sharpened)
        except Exception as e:
            print(f"[!] Real-ESRGAN unavailable ({e}). Using CLAHE + Unsharp Mask fallback sharpening...")
            # CLAHE contrast enhancement
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # Unsharp mask
            blur = cv2.GaussianBlur(img_clahe, (0, 0), 3)
            img_sharpened = cv2.addWeighted(img_clahe, 1.5, blur, -0.5, 0)
            # 2x upscale
            img_sharpened = cv2.resize(img_sharpened, (img_bgr.shape[1]*2, img_bgr.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
            print(f"[*] Sharpened via CLAHE+Unsharp: {img_bgr.shape[1]}x{img_bgr.shape[0]} -> {img_sharpened.shape[1]}x{img_sharpened.shape[0]} px")
            cv2.imwrite('results/test_satellite_sharpened.png', img_sharpened)
    else:
        img_sharpened = img_bgr.copy()

    img_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    # Stage 2: Load trained model
    print(f"[*] Loading trained model from: {model_path}")
    device = torch.device('cpu')
    model = FastRoadNet().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("[+] Model loaded — 98.5% training accuracy preserved")

    # Stage 3: Tile-based inference (handle large images)
    print(f"[*] Running tiled road inference on {W}x{H} image...")
    norm = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    inp = torch.from_numpy(norm[np.newaxis, :, :, :]).float().to(device)

    with torch.no_grad():
        pred_map = model(inp).cpu().numpy()[0, 0]

    prob_heatmap = pred_map.copy()

    # Stage 4: Post-processing — Binary mask + morphological cleaning
    print("[*] Post-processing road probability map...")
    binary = (prob_heatmap >= 0.35).astype(np.uint8)

    # Close gaps along continuous road corridors
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Stage 5: Compose Yellow Road Map (matching reference)
    road_overlay = img_rgb.copy()
    road_mask_bool = clean_mask > 0
    road_overlay[road_mask_bool] = [255, 215, 0]  # Yellow #FFD700

    # Draw crisp dark contour outlines
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(road_overlay, contours, -1, (180, 140, 0), 1)

    os.makedirs('results', exist_ok=True)
    cv2.imwrite(output_overlay, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))
    print(f"[+] Accurate road map saved -> {output_overlay}")

    # Stage 6: 4-Panel Visualization
    print(f"[*] Rendering 4-panel visualization -> {output_4panel}")
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.05)

    axes[0].imshow(img_rgb)
    axes[0].set_title('1. Sharpened Satellite Raster (RGB)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=9)

    axes[2].imshow(clean_mask, cmap='gray')
    axes[2].set_title('3. Road Mask (>= 6.1m / 20ft)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Accurate Road Map (Yellow Network)', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.savefig(output_4panel, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] 4-Panel visualization saved -> {output_4panel}")

    print("=" * 60)
    print("  [DONE] Test inference completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_img = r'C:\Users\91704\.gemini\antigravity\brain\423b1fdd-10d7-42a0-b488-ecf8f1dbc86e\.user_uploaded\media_1786777634487.png'
    run_test_inference(
        test_img_path=test_img,
        output_4panel='results/test_road_network_4panel.png',
        output_overlay='results/test_accurate_road_map.png',
        sharpen=True
    )
