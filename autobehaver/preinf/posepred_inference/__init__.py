"""PosePred inference (pipeline stage).

This stage consumes normalized feature HDF5 files (typically from keyprenorm)
and runs PosePred transformer checkpoint(s) to produce per-frame inference CSVs.

It is intentionally *inference-only* and is meant to be used as part of the
pipeline/config-driven workflow.
"""

from .config import PosePredInferenceConfig

__all__ = [
    "PosePredInferenceConfig",
]
