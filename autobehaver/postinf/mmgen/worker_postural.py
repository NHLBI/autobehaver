"""Align inference labels and probabilities for recording-level MMGen postural
extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Optional

from .postural import _calculate_postural_features


def _align_values_to_poscalc_length(values, target_len: int, edge_skip: int, *, fill_value=None) -> np.ndarray:
    """Align label-like arrays to the positional-calcs timeline.

    Full-timeline inference CSVs already include one row per frame and should not
    be shifted by edge_skip. Shorter inference outputs are padded into the
    positional-calcs timeline using edge_skip. If neither assumption fits cleanly, use the common prefix and leave the
    remaining frames as fill_value.
    """

    out = np.full(int(target_len), fill_value, dtype=object)
    arr = np.asarray(values, dtype=object)

    if len(arr) == int(target_len):
        return arr

    start = max(int(edge_skip or 0), 0)
    stop = start + len(arr)
    if len(arr) < int(target_len) and stop <= int(target_len):
        out[start:stop] = arr
        return out

    n = min(int(target_len), len(arr))
    if n > 0:
        out[:n] = arr[:n]
    return out


def _align_numeric_to_poscalc_length(values, target_len: int, edge_skip: int) -> np.ndarray:
    """Align numeric per-frame arrays to the positional-calcs timeline."""

    out = np.full(int(target_len), np.nan, dtype=float)
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)

    if len(arr) == int(target_len):
        return arr

    start = max(int(edge_skip or 0), 0)
    stop = start + len(arr)
    if len(arr) < int(target_len) and stop <= int(target_len):
        out[start:stop] = arr
        return out

    n = min(int(target_len), len(arr))
    if n > 0:
        out[:n] = arr[:n]
    return out


def compute_postural_features_df(
    *,
    config: dict,
    poscalc_df: pd.DataFrame,
    df_inference: pd.DataFrame,
    masked_poses: np.ndarray,
    masked_orients: Optional[np.ndarray],
    edge_skip: int,
    group: str,
    individual_id: str,
    video_path: str,
    mm_per_px: float,
) -> Optional[pd.DataFrame]:
    """Align recording labels and orientation probabilities and extract enabled posture
    features.
    """

    if not config.get("calculate_postural_features"):
        return None

    # Create a copy of poscalc_df to add orient probs to
    poscalc_df_with_orient_probs = poscalc_df.copy()

    # 1. Copy Orientation Probability Columns
    orient_prob_cols = [
        col for col in df_inference.columns if col.startswith("Orient-") and col.endswith("-Prob")
    ]
    if orient_prob_cols:
        orient_prob_values = df_inference[orient_prob_cols].copy()
        rename_dict = {
            col: f"orient_{col.replace('Orient-', '').replace('-Prob', '')}"
            for col in orient_prob_cols
        }
        orient_prob_values.rename(columns=rename_dict, inplace=True)

        # Align per-frame orientation probabilities to the same positional-calcs
        # timeline used for labels. Full-timeline inference outputs are not
        # shifted; shorter outputs use edge_skip padding.
        for col in orient_prob_values.columns:
            poscalc_df_with_orient_probs[col] = _align_numeric_to_poscalc_length(
                orient_prob_values[col].values,
                len(poscalc_df_with_orient_probs),
                edge_skip,
            )

    # 2. Align masked_poses and masked_orients to the positional-calcs timeline
    full_length_poses = _align_values_to_poscalc_length(
        masked_poses,
        len(poscalc_df_with_orient_probs),
        edge_skip,
        fill_value=None,
    )
    if masked_orients is not None:
        full_length_orients = _align_values_to_poscalc_length(
            masked_orients,
            len(poscalc_df_with_orient_probs),
            edge_skip,
            fill_value=None,
        )
    else:
        full_length_orients = np.full(len(poscalc_df_with_orient_probs), None, dtype=object)

    # 3. Call the feature function with fully aligned data
    return _calculate_postural_features(
        poscalc_df=poscalc_df_with_orient_probs,
        masked_poses=full_length_poses,
        masked_orients=full_length_orients,
        video_path=video_path,
        group=group,
        individual=individual_id,
        likelihood_threshold=0.1,
        verbose_debug=False,
        mm_per_px=mm_per_px,
    )
