# Model Architecture and Pre-trained Checkpoints

## 1. Supported Architectures
- **U-Net** (`segmentation_models_pytorch.Unet`): Encoder-decoder with skip connections, optimal for fine-scale linear connectivity.
- **DeepLabV3+** (`segmentation_models_pytorch.DeepLabV3Plus`): Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.

## 2. Checkpoint Details
- **Location**: `weights/best_road_seg_unet.pth` (124.2 MB)
- **Source**: Pre-trained on SpaceNet & DeepGlobe satellite road benchmark datasets.
- **Input Dimensions**: Dynamic window slicing ($512 \times 512$ with $128\text{px}$ overlap default).
- **Normalization**: Percentile band normalization ($2\% - 98\%$) per channel.

## 3. Loss Functions for Training/Fine-Tuning
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{BCE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}$$
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |P \cap G| + \epsilon}{|P| + |G| + \epsilon}$$
Combines smooth cross-entropy with overlap dice loss to handle severe class imbalance between road corridors and background terrain.
