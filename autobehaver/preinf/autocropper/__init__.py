"""Autocropper (auto chamber cropping).

This stage identifies circular fly chambers in multi-chamber videos and crops each
chamber into its own movie file.

Notes:
  - This module is intentionally tailored to the current Autobehaver lab pipeline.
  - Heavy dependencies (OpenCV, moviepy, torch) are imported only when the stage runs.
"""

from .config import AutocropperConfig
from .core import run_autocropper

__all__ = [
    "AutocropperConfig",
    "run_autocropper",
]
