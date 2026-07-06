"""Worker helpers for kinematics extraction.

This module contains the per-recording kinematics logic used by the mmgen
multiprocessing worker.

The helper keeps the per-recording kinematics path isolated from the main
worker so that bout-level and time-weighted outputs can be maintained together.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .kinematics import compute_head_movement_frame_signals, extract_bout_features_simple
from .units import _convert_kinematic_columns_to_real_units


def compute_kinematics_outputs(
    *,
    aligned_poscalc_df: pd.DataFrame,
    pose_bouts: List[Dict[str, Any]],
    pose_orient_bouts: List[Dict[str, Any]],
    masked_pose_valid: Optional[np.ndarray],
    masked_pose_orient_valid: Optional[np.ndarray],
    num_frames: int,
    fps: float,
    mm_per_px: float,
    kin_frames_signal_cols: List[str],
    group: str,
    individual_id: str,
    sex: str,
    source_file: str,
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """Compute kinematics-related outputs for one recording.

    Returns
    -------
    bout_features_dict:
        Dict keyed by suffix (e.g. '_inst') of bout-level kinematic feature tables.
    kin_frames_df:
        Per-frame kinematics substrate table (TW) filtered to valid contexts, or None.
    aligned_poscalc_df_kin:
        The aligned positional_calcs dataframe with kinematic columns converted into
        real units (mm, s, etc.).
    """

    # Filter out Wall-Up and Wall-Down orientations from pose+orient bouts for kinematics only.
    # IMPORTANT: Jump/Scramble are collapsed (non-orientation-specific) but we still want their kinematics.
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

    # --- UNIT CONVERSION ---
    # Convert ONLY the kinematic columns to real units for kinematic feature extraction.
    # Keep aligned_poscalc_df in pixels for occupancy / spatial plots.
    aligned_poscalc_df_kin = _convert_kinematic_columns_to_real_units(
        aligned_poscalc_df, mm_per_px=mm_per_px, fps=fps
    )

    kinematic_suffixes = ["_inst"]

    # ------------------------------------------------------------------
    # TW substrate: per-frame kinematics table for valid contexts only
    # i.e., only frames in *_valid labels).
    # ------------------------------------------------------------------
    kin_frames_df: Optional[pd.DataFrame]
    try:
        # Base context = Pose+Orient valid (this excludes invalid bouts/blips)
        if masked_pose_orient_valid is not None:
            kin_context_valid = np.array(masked_pose_orient_valid, dtype=object)
        else:
            kin_context_valid = np.full(num_frames, np.nan, dtype=object)

        # Preserve Jump/Scramble as their own contexts (they are not orientation-specific)
        if masked_pose_valid is not None:
            js_mask = np.isin(masked_pose_valid, ["Jump", "Scramble"])
            kin_context_valid[js_mask] = masked_pose_valid[js_mask]

        # Safety: ensure Wall-Up / Wall-Down never enter the TW contexts
        ser_ctx = pd.Series(kin_context_valid, dtype="object")
        bad = ser_ctx.astype(str).str.contains("Wall-Up|Wall-Down", na=False)
        if bad.any():
            kin_context_valid[bad.to_numpy()] = np.nan

        # Head derived framewise signals (already in real units because aligned_poscalc_df_kin is)
        head_vel, head_acc, head_jerk = compute_head_movement_frame_signals(
            aligned_poscalc_df_kin, kinematic_suffix="_inst"
        )

        kin_frames_df = pd.DataFrame(
            {
                "group": group,
                "individual": individual_id,
                "sex": sex,
                "source_file": source_file,
                "frame_idx": np.arange(num_frames, dtype=int),
                "kin_context_valid": kin_context_valid,
            }
        )

        # Insert derived head columns
        kin_frames_df["head_velocity_inst"] = head_vel
        kin_frames_df["head_acceleration_inst"] = head_acc
        kin_frames_df["head_jerk_inst"] = head_jerk

        # Insert all other per-frame signals (fill NaN if missing)
        for col in kin_frames_signal_cols:
            if col in ("head_velocity_inst", "head_acceleration_inst", "head_jerk_inst"):
                continue  # already filled above
            if col in aligned_poscalc_df_kin.columns:
                kin_frames_df[col] = aligned_poscalc_df_kin[col].to_numpy()
            else:
                kin_frames_df[col] = np.nan

        # Keep only frames that have a valid context label (Option B)
        kin_frames_df = kin_frames_df[kin_frames_df["kin_context_valid"].notna()].reset_index(
            drop=True
        )

    except Exception as e:
        print(f"WARNING: Failed to build kin_frames_df for {source_file}: {e}")
        kin_frames_df = None

    # ------------------------------------------------------------------
    # Existing BB bout-level kinematics extraction continues below
    # ------------------------------------------------------------------
    bout_features_dict: Dict[str, pd.DataFrame] = {}
    for suffix in kinematic_suffixes:
        all_feature_dfs: List[pd.DataFrame] = []

        # Pose-only kinematics REMOVED:
        # We only compute kinematics for Pose+Orient contexts (apples-to-apples),
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
                )
            )

        if all_feature_dfs:
            bout_features_dict[suffix] = pd.concat(all_feature_dfs, ignore_index=True)
        else:
            bout_features_dict[suffix] = pd.DataFrame()

    return bout_features_dict, kin_frames_df, aligned_poscalc_df_kin
