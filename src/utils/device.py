"""
Device Resolution and Memory Management
"""

import torch
from .logging import get_logger

logger = get_logger("DeviceManager")


def resolve_device(requested_device: str = "auto") -> torch.device:
    """
    Resolves compute device (CUDA vs CPU) safely with informative logging.
    """
    req = requested_device.lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
            return torch.device("cpu")
    elif req == "cpu":
        logger.info("Using CPU device.")
        return torch.device("cpu")
    else:  # "auto"
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.info("Auto-selected CPU (CUDA not detected).")
            return torch.device("cpu")


def get_device(requested: str = "auto") -> torch.device:
    return resolve_device(requested)
