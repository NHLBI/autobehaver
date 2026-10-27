"""Define KeyPre feature-generation settings.

Changing these values changes the model input schema; downstream models must
be compatible with that feature set or retrained.
"""

from __future__ import annotations

# Settings to control which features to generate
include_position = True  # include centroid x,y and angle (sin/cos) + polar features
rolling_windows = [5, 9]
include_likelihoods = True
nan_edge_effects = True
edge_skip = 5
warn_on_internal_nans = True
