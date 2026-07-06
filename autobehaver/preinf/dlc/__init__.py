"""DeepLabCut inference wrapper.

This stage is intentionally a thin wrapper around DeepLabCut (DLC) so that DLC
inference can live inside the Autobehaver pipeline.

Notes:
  - Heavy dependencies (deeplabcut, tensorflow) are imported lazily at runtime so
    that the rest of Autobehaver can be imported in environments where DLC is
    not installed.
"""

from __future__ import annotations

__all__ = [
    "DLCInferencerConfig",
    "run_dlc",
]

from .config import DLCInferencerConfig
from .core import run_dlc
