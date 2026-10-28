"""Extract bounded measurement candidates for one MMGen recording."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .measurements import extract_measurement_candidates


def compute_measurement_result(
    *,
    config: dict,
    masked_poses: np.ndarray,
    masked_orients: Optional[np.ndarray],
    aligned_poscalc_df: pd.DataFrame,
    individual_id: str,
    group: str,
    sex: str,
) -> Optional[pd.DataFrame]:
    """Return a bounded measurement candidate table when measurements are enabled and
    orientation labels are available.
    """

    if config.get("make_measurements") and masked_orients is not None:
        runtime = config.get("runtime_settings")
        random_seed = getattr(runtime, "random_seed", None)
        return extract_measurement_candidates(
            masked_poses,
            masked_orients,
            aligned_poscalc_df,
            individual_id,
            group,
            sex,
            random_seed=random_seed,
        )
    return None
