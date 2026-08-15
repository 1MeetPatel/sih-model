"""
Configuration Loading and Validation Utilities
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration with fallback to default.yaml.
    """
    default_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default.yaml")
    default_path = os.path.abspath(default_path)

    base_config: Dict[str, Any] = {}
    if os.path.exists(default_path):
        with open(default_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f) or {}

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
            base_config = merge_configs(base_config, user_config)

    return base_config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
