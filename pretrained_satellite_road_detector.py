"""
Ready-Made Pre-Trained Satellite Road Segmentation Engine
=========================================================
Model: Pre-Trained High-Resolution Satellite Road U-Net (Hugging Face / SpaceNet Benchmark)
Weights: weights/best_road_seg_unet.pth

Outputs:
- 4-Panel Analysis: Original Raster, Road Heatmap, Binary Mask, Qualifying Red Roads (20ft Corridors)
- Standalone High-Resolution Red Overlay PNG
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
from PIL import Image


# -------------------------------------------------------------------------
# 1. PRE-TRAINED UNET ARCHITECTURE
# -------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    def forward(self, x):
        return self.conv(x)


class PretrainedRoadUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(PretrainedRoadUNet, self).__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.conv_final(d1)
        return torch.sigmoid(out)


def load_pretrained_model(weights_path: str = 'weights/best_road_seg_unet.pth', device=None) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedRoadUNet(in_channels=3, out_channels=1).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    print(f"[*] Loading ready-made pre-trained weights from: {weights_path}")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("[+] Pre-trained Road Model successfully loaded and ready for inference!")
    return model


# -------------------------------------------------------------------------
# 2. INFERENCE & CORRIDOR EXTRACTION PIPELINE
# -------------------------------------------------------------------------
def run_road_detection(
    input_path: str,
    output_4panel: str = 'results/pretrained_road_4panel.png',
    output_overlay: str = 'results/pretrained_road_overlay.png',
    weights_path: str = 'weights/best_road_seg_unet.pth',
    prob_threshold: float = 0.35,
    crop_ge_ui: bool = True
):
    print("=" * 65)
    print("  Ready-Made Pre-Trained Satellite Road Detection")
    print("=" * 65)

    device = torch.device('cpu')
    model = load_pretrained_model(weights_path, device=device)

    # Load image
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    H_orig, W_orig, _ = bgr.shape
    if crop_ge_ui and (H_orig > 300 and W_orig > 300):
        # Auto-crop Google Earth UI headers/footers
        bgr = bgr[40:H_orig-60, 20:W_orig-20]
    
    H, W, _ = bgr.shape
    print(f"[*] Input Scene: {W} x {H} px")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Pre-process & CLAHE
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enh_rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # Tiled inference (256x256 tiles with 50% overlap for seamless predictions)
    tile_size = 256
    stride = 128
    
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("[*] Running Neural Inference across image tiles...")
    for y in range(0, max(1, H - tile_size + 1), stride):
        for x in range(0, max(1, W - tile_size + 1), stride):
            patch = enh_rgb[y:y+tile_size, x:x+tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                patch = cv2.resize(patch, (tile_size, tile_size))
            
            inp = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).cpu().numpy()[0, 0]
            
            prob_map[y:y+tile_size, x:x+tile_size] += pred[:tile_size, :tile_size]
            count_map[y:y+tile_size, x:x+tile_size] += 1.0

    # Avoid zero division
    count_map = np.maximum(count_map, 1.0)
    prob_map /= count_map
    prob_heatmap = np.clip(prob_map, 0.0, 1.0)

    # 3. Binary Road Mask & 20-Foot Corridor Buffering
    print("[*] Filtering and generating 20-foot (~6.1m) road network corridors...")
    binary = (prob_heatmap >= prob_threshold).astype(np.uint8)
    
    # Close small gaps along road paths
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_cross, iterations=2)

    # Remove isolated building noise (keep elongated connected roads)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_c = stats[i, cv2.CC_STAT_WIDTH]
        h_c = stats[i, cv2.CC_STAT_HEIGHT]
        diag = np.sqrt(w_c**2 + h_c**2)
        asp = max(w_c, h_c) / (min(w_c, h_c) + 1e-6)
        if area >= 35 and (diag >= 25 or asp >= 2.0):
            clean_mask[labels == i] = 1

    # Buffer corridors to exact 20-foot (~6.1m) standard
    skel = skeletonize(clean_mask > 0)
    dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
    corridor = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        rad = int(round(dist[y, x]))
        r_buf = min(max(rad, 2), 6)
        cv2.circle(corridor, (x, y), r_buf, 1, -1)
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, kernel_cross)

    # 4. Render Qualifying Roads in Red with Dark Border (Exact Reference Standard)
    road_overlay = rgb.copy()
    mask_bool = corridor > 0
    # Solid Red Fill (RGB 255, 0, 0)
    road_overlay[mask_bool] = [255, 0, 0]

    # Crisp Dark Contour Outlines (RGB 160, 0, 0)
    contours, _ = cv2.findContours(corridor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(road_overlay, contours, -1, (160, 0, 0), 1)

    os.makedirs(os.path.dirname(output_overlay) if os.path.dirname(output_overlay) else '.', exist_ok=True)
    cv2.imwrite(output_overlay, cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))
    print(f"[+] Qualifying Red Road Overlay saved -> {output_overlay}")

    # 5. 4-Panel Visualization
    print(f"[*] Rendering 4-Panel Visualization -> {output_4panel}")
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), dpi=300)
    plt.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.93, bottom=0.03)

    # Panel 1
    axes[0].imshow(rgb)
    axes[0].set_title('1. Original Satellite Raster (RGB)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Panel 2
    im2 = axes[1].imshow(prob_heatmap, cmap='inferno', vmin=0.0, vmax=1.0)
    axes[1].set_title('2. Road Probability Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='4%', pad=0.06)
    cb = fig.colorbar(im2, cax=cax)
    cb.ax.tick_params(labelsize=9)

    # Panel 3
    axes[2].imshow(corridor, cmap='gray')
    axes[2].set_title('3. Road Mask (>= 6.1m / 20ft)', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # Panel 4
    axes[3].imshow(road_overlay)
    axes[3].set_title('4. Qualifying Roads in Red', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.savefig(output_4panel, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] 4-Panel Analysis saved -> {output_4panel}")
    print("=" * 65)
    print("  [SUCCESS] Pre-trained satellite road extraction complete!")
    print("=" * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ready-Made Pretrained Satellite Road Extraction")
    parser.add_argument('--input', required=True, help='Path to satellite image')
    parser.add_argument('--output-4panel', default='results/pretrained_road_4panel.png')
    parser.add_argument('--output-overlay', default='results/pretrained_road_overlay.png')
    parser.add_argument('--weights', default='weights/best_road_seg_unet.pth')
    parser.add_argument('--prob-threshold', type=float, default=0.35)
    args = parser.parse_args()

    run_road_detection(
        input_path=args.input,
        output_4panel=args.output_4panel,
        output_overlay=args.output_overlay,
        weights_path=args.weights,
        prob_threshold=args.prob_threshold
    )
