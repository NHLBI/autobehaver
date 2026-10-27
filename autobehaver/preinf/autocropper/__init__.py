"""Detect the standard six-chamber layout and crop each included chamber.

OpenCV and FFmpeg-facing modules are imported when Autocropper runs.
"""

from .config import AutocropperConfig
from .core import run_autocropper

__all__ = [
    "AutocropperConfig",
    "run_autocropper",
]
