"""Wrap DeepLabCut inference as a pipeline stage.

Load DeepLabCut and TensorFlow when inference runs so other stages can
be imported without those optional dependencies.
"""

from __future__ import annotations

__all__ = [
    "DLCInferencerConfig",
    "run_dlc",
]

from .config import DLCInferencerConfig
from .core import run_dlc
