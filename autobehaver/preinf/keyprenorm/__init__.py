"""Keyprenorm: normalization for keypre output.

This stage consumes ``*_positional_calcs.csv`` files (typically produced by keypre)
and produces normalized, compressed HDF5 feature matrices.

Two primary workflows:
  1) Calculate normalization parameters (mean/stddev per feature column).
  2) Apply those parameters to generate normalized HDF5 outputs.

"""

from __future__ import annotations

__all__ = [
    "__stage_version__",
]

__stage_version__ = "41"
