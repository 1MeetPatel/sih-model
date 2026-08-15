"""
Model Factory and Architecture Instantiation
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import segmentation_models_pytorch as smp
from ..utils.logging import get_logger

logger = get_logger("ModelFactory")


class ModelFactory:
    """
    Builds PyTorch segmentation architectures (UNet, DeepLabV3+, etc.)
    with pre-trained ImageNet backbones and configured channel parameters.
    """

    @staticmethod
    def create_model(
        architecture: str = "unet",
        encoder: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1
    ) -> nn.Module:
        arch = architecture.lower().strip()
        logger.info(f"Instantiating model: {arch.upper()} with encoder: {encoder} (in_channels={in_channels}, classes={classes})")

        if arch in ["unet", "u-net"]:
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid"
            )
        elif arch in ["deeplabv3plus", "deeplabv3+"]:
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid"
            )
        elif arch in ["linknet", "d-linknet"]:
            model = smp.Linknet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid"
            )
        else:
            logger.warning(f"Unknown architecture '{architecture}', defaulting to UNet.")
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation="sigmoid"
            )

        return model
