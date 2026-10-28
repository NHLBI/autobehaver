"""Extract bout-based MMGen kinematics for one recording.

Retain pixel coordinates separately for spatial analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .kinematics import compute_head_movement_frame_signals, extract_bout_features_simple
from .units import _convert_kinematic_columns_to_real_units


def compute_kinematics_outputs(
    *,
    aligned_poscalc_df: pd.DataFrame,
    pose_bouts: List[Dict[str, Any]],
    pose_orient_bouts: List[Dict[str, Any]],
    fps: float,
    mm_per_px: float,
    group: str,
    individual_id: str,
    sex: str,
    source_file: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute kinematics-related outputs for one recording.

    Returns
    -------
    bout_features_dict:
        Dict keyed by suffix (e.g. '_inst') of bout-level kinematic feature tables.
    aligned_poscalc_df_kin:
        The aligned positional_calcs dataframe with kinematic columns converted into
        real units (mm, s, etc.).
    """

    # Filter out Wall-Up and Wall-Down orientations from pose+orient bouts for kinematics only.
    # Retain orientation-independent Jump/Scramble kinematics.
    pose_orient_bouts_for_kinematics = [
        b
        for b in pose_orient_bouts
        if "Wall-Up" not in str(b.get("bout_type", b.get("label", "")))
        and "Wall-Down" not in str(b.get("bout_type", b.get("label", "")))
    ]

    # Add Jump/Scramble bouts (collapsed contexts) into Pose+Orient kinematics list.
    # We intentionally DO NOT add these to pose_orient_bouts_for_boutsummary (to avoid double-counting in summaries).
    jump_scramble_bouts = [
        b
        for b in pose_bouts
        if str(b.get("label", "")).startswith("Jump")
        or str(b.get("label", "")).startswith("Scramble")
    ]
    pose_orient_bouts_for_kinematics = pose_orient_bouts_for_kinematics + jump_scramble_bouts

    # UNIT CONVERSION
    # Convert ONLY the kinematic columns to real units for kinematic feature extraction.
    # Keep aligned_poscalc_df in pixels for occupancy / spatial plots.
    aligned_poscalc_df_kin = _convert_kinematic_columns_to_real_units(
        aligned_poscalc_df, mm_per_px=mm_per_px, fps=fps
    )

    kinematic_suffixes = ["_inst"]
    head_signals = compute_head_movement_frame_signals(
        aligned_poscalc_df_kin, kinematic_suffix="_inst")

    bout_features_dict: Dict[str, pd.DataFrame] = {}
    for suffix in kinematic_suffixes:
        all_feature_dfs: List[pd.DataFrame] = []

        # Kinematics use combined pose/orientation contexts.
        # Use Pose+Orient contexts for the kinematic outputs,
        # with Jump/Scramble included as collapsed contexts via pose_orient_bouts_for_kinematics.
        if pose_orient_bouts_for_kinematics:
            all_feature_dfs.append(
                extract_bout_features_simple(
                    aligned_poscalc_df_kin,
                    pose_orient_bouts_for_kinematics,
                    individual_id,
                    group,
                    sex,
                    source_file,
                    category="Pose+Orient",
                    kinematic_suffix=suffix,
                    head_frame_signals=head_signals,
                )
            )

        if all_feature_dfs:
            bout_features_dict[suffix] = pd.concat(all_feature_dfs, ignore_index=True)
        else:
            bout_features_dict[suffix] = pd.DataFrame()

    return bout_features_dict, aligned_poscalc_df_kin
