"""Keypre feature-generation settings.

If you change these values, treat the outputs as a new feature set and retrain
downstream models.
"""

from __future__ import annotations

# Settings to control which features to generate
include_position = True  # include centroid x,y and angle (sin/cos) + polar features
rolling_windows = [5, 9]
include_likelihoods = True
nan_edge_effects = True
edge_skip = 5
warn_on_internal_nans = True
