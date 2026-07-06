"""Worker helpers for standardized measurements extraction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .measurements import extract_standardized_measurements_summary


def compute_measurement_result(
    *,
    config: dict,
    masked_poses: np.ndarray,
    masked_orients: Optional[np.ndarray],
    aligned_poscalc_df: pd.DataFrame,
    individual_id: str,
    group: str,
    sex: str,
) -> Optional[dict]:
    """Compute standardized measurement output for one recording.

    Returns a compact per-recording summary row (dict) rather than frame-level
    measurement DataFrames. This prevents mmgen from accumulating large raw
    measurement tables in RAM across many recordings.

    Behavior preserved:
    - only runs if config['make_measurements'] is True
    - requires masked_orients to be present
    """

    if config.get("make_measurements") and masked_orients is not None:
        runtime = config.get("runtime_settings")
        random_seed = getattr(runtime, "random_seed", None)
        return extract_standardized_measurements_summary(
            masked_poses,
            masked_orients,
            aligned_poscalc_df,
            individual_id,
            group,
            sex,
            random_seed=random_seed,
        )
    return None
