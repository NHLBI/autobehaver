"""Keypoint preprocessing (keypre).

This stage reads DeepLabCut outputs and produces per-frame engineered features.
"""

from .config import KeypreConfig

__all__ = [
    "KeypreConfig",
]
