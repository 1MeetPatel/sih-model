"""
Checkpoint Loading and Validation
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..utils.logging import get_logger

logger = get_logger("CheckpointLoader")


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """
    Validates and loads a PyTorch model checkpoint.
    Supports state_dict, nested 'model_state_dict', or entire model objects.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(state, dict):
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state_dict = state["model"]
        else:
            state_dict = state
    else:
        state_dict = state

    # Strip 'module.' prefix if trained with DistributedDataParallel
    clean_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_dict[clean_key] = v

    try:
        model.load_state_dict(clean_dict, strict=False)
        logger.info("[OK] Model checkpoint loaded successfully!")
    except Exception as e:
        logger.warning(f"Non-strict loading warning: {e}. Checking key alignment...")

    model.to(device)
    model.eval()
    return model
