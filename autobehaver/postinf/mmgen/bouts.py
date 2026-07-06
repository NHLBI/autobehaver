"""Bout definition and summary feature calculations.

This module contains foundational bout-domain utilities used by mmgen:

- Converting framewise label sequences into bouts.
- Building "valid" per-frame label arrays from bout lists.
- Per-bout-type summary feature calculation.
- Sleep feature calculation (Standing bouts >= threshold).

These functions were extracted into a dedicated module to make the
pipeline modular while preserving behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd



def extract_bouts_from_label_sequence(label_sequence, min_bout=1, exempt_labels=None):
    """
    Given a 1D array-like of labels (e.g., masked_poses), returns a list of bouts.
    Each bout is a dict: {'start': int, 'end': int, 'label': str}
    Includes bouts of length >= min_bout, except labels in `exempt_labels`
    (e.g., 'Jump') which are always kept (threshold = 1).
    """
    exempt_labels = set(exempt_labels or [])
    bouts = []
    current_label = None
    start_idx = None

    def _threshold_for(label):
        if isinstance(label, str) and label in exempt_labels:
            return 1
        # Also allow pose+orient exemptions like "Jump + Floor"
        if isinstance(label, str) and any(label.startswith(f"{ex} ") or label.startswith(f"{ex}+") or label.startswith(f"{ex} +") for ex in exempt_labels):
            return 1
        return min_bout

    for idx, label in enumerate(label_sequence):
        if pd.isna(label) or label == "":
            if current_label is not None:
                bout_length = idx - start_idx
                if bout_length >= _threshold_for(current_label):
                    bouts.append({'start': start_idx, 'end': idx - 1, 'label': current_label})
                current_label = None
                start_idx = None
            continue
        if label != current_label:
            if current_label is not None:
                bout_length = idx - start_idx
                if bout_length >= _threshold_for(current_label):
                    bouts.append({'start': start_idx, 'end': idx - 1, 'label': current_label})
            current_label = label
            start_idx = idx
    if current_label is not None:
        bout_length = len(label_sequence) - start_idx
        if bout_length >= _threshold_for(current_label):
            bouts.append({'start': start_idx, 'end': len(label_sequence) - 1, 'label': current_label})

    return bouts


def _build_valid_label_array_from_bouts(bouts, length):
    """
    Build a per-frame label array where only frames inside the provided bouts
    keep their label; all other frames become NaN.

    This ignores short blips that did not survive bout filtering.
    """
    arr = np.full(int(length), np.nan, dtype=object)
    if not bouts:
        return arr

    for b in bouts:
        try:
            s = int(b.get('start', 0))
            e = int(b.get('end', -1))
            lab = b.get('label', np.nan)
            if s >= 0 and e >= s:
                arr[s:e+1] = lab
        except Exception:
            continue

    return arr


def calculate_bout_summary_features(bouts, total_frames, fps=40):
    """
    Takes a list of bouts for a single movie and calculates a full suite of summary features.

    Args:
        bouts (list of dicts): The output from extract_bouts_from_label_sequence.
        total_frames (int): The total number of frames in the analyzed video segment.
        fps (int): The frames per second of the video, for converting to seconds.

    Returns:
        pd.DataFrame: A DataFrame where each row is a behavior and columns are the summary features.
    """
    if not bouts:
        return pd.DataFrame()

    # Add a 'length' key to each bout dictionary if not present
    for bout in bouts:
        if 'length' not in bout:
            bout['length'] = bout['end'] - bout['start'] + 1

    total_seconds = total_frames / fps
    bout_df = pd.DataFrame(bouts)

    # Group bouts by their label to calculate stats per behavior
    grouped = bout_df.groupby('label')
    
    summary_list = []
    for behavior, group_df in grouped:
        # --- Basic Stats ---
        n_bouts = len(group_df)
        total_length_frames = group_df['length'].sum()
        mean_length_frames = group_df['length'].mean()
        std_length_frames = group_df['length'].std(ddof=0) # Use population std

        # --- Refined Features ---
        bout_frequency = n_bouts / total_seconds if total_seconds > 0 else 0
        bout_duration_mean_s = mean_length_frames / fps
        
        # Coefficient of Variation (CV) for bout duration
        bout_duration_coeff_of_var = (std_length_frames / mean_length_frames) if mean_length_frames > 0 else 0
        
        # Bout Initiation Probability (BIP)
        available_frames = total_frames - total_length_frames
        initiation_prob = n_bouts / available_frames if available_frames > 0 else 0

        # Inter-Bout Interval (IBI) Calculation
        inter_bout_interval_mean_s, inter_bout_interval_std_s = np.nan, np.nan
        if n_bouts > 1:
            sorted_bouts = group_df.sort_values('start')
            # Ensure start/end are numeric before calculation
            starts = pd.to_numeric(sorted_bouts['start'], errors='coerce')
            ends = pd.to_numeric(sorted_bouts['end'], errors='coerce')

            # Create slices, explicitly reset index to ensure alignment (0 to N-2)
            starts_next = starts.iloc[1:].reset_index(drop=True)
            ends_current = ends.iloc[:-1].reset_index(drop=True)

            # Check for NaNs *after* resetting index
            # valid_ibis will now be aligned and have length N-1
            valid_ibis_mask = starts_next.notna() & ends_current.notna()

            # Convert the mask to a NumPy boolean array
            valid_ibis_np = valid_ibis_mask.to_numpy()

            if valid_ibis_np.any():
                # Convert data slices to NumPy arrays
                s_vals = starts_next.to_numpy()
                e_vals = ends_current.to_numpy()

                # Filter using the NumPy boolean mask - lengths should now match
                ibis = s_vals[valid_ibis_np] - e_vals[valid_ibis_np]

                # Ensure IBIs are non-negative, filter out invalid values
                ibis_s = ibis[ibis >= 0] / fps
                if ibis_s.size > 0:
                     inter_bout_interval_mean_s = np.mean(ibis_s)
                     # Calculate std only if there's more than one valid IBI
                     inter_bout_interval_std_s = np.std(ibis_s) if ibis_s.size > 1 else 0.0
                     # Ensure std is not NaN if size is 1
                     inter_bout_interval_std_s = inter_bout_interval_std_s if pd.notna(inter_bout_interval_std_s) else 0.0
                else: # Handle cases where all ibis were negative or filtered out
                     inter_bout_interval_mean_s, inter_bout_interval_std_s = 0.0, 0.0 # Or np.nan, np.nan depending on desired output

        summary_list.append({
            'bout_type': behavior,
            'bout_frequency': bout_frequency,
            'bout_duration_mean_s': bout_duration_mean_s,
            'bout_duration_coeff_of_var': bout_duration_coeff_of_var,
            'inter_bout_interval_mean_s': inter_bout_interval_mean_s,
            'inter_bout_interval_std_s': inter_bout_interval_std_s,
            'initiation_prob': initiation_prob,
            'n_bouts': n_bouts,
            'total_length_frames': total_length_frames
        })

    return pd.DataFrame(summary_list)


def calculate_sleep_features(bouts, total_frames, fps=40, sleep_threshold_frames=7200):
    """
    Calculates sleep-specific features for Standing bouts >= sleep_threshold_frames.
    
    Sleep is defined as Standing bouts (pose-only or pose+orient) that last at least
    sleep_threshold_frames (default: 7200 frames = 3 minutes at 40fps).
    
    Args:
        bouts: List of bout dictionaries with 'start', 'end', 'label', and optionally 'bout_type'
        total_frames: Total frames in video
        fps: Frames per second (default: 40)
        sleep_threshold_frames: Minimum frames to be considered sleep (default: 7200)
    
    Returns:
        pd.DataFrame: One row per Standing bout type (Standing, Standing+Floor, etc.)
                     with columns: bout_type, n_sleep_bouts, sleep_bout_duration_mean_s,
                     sleep_bout_duration_std_s, total_sleep_time_s, total_length_frames, sleep_time_budget
    """
    if not bouts:
        return pd.DataFrame()
    
    # Add 'length' and 'bout_type' to each bout if not present
    for bout in bouts:
        if 'length' not in bout:
            bout['length'] = bout['end'] - bout['start'] + 1
        if 'bout_type' not in bout:
            bout['bout_type'] = bout.get('label', '')
    
    total_seconds = total_frames / fps
    bout_df = pd.DataFrame(bouts)
    
    # Filter to only Standing bouts (pose-only or pose+orient)
    standing_bouts = bout_df[bout_df['bout_type'].str.contains('Standing', na=False, case=False)]
    
    if standing_bouts.empty:
        # No standing bouts means no sleep features possible
        return pd.DataFrame(columns=['bout_type', 'n_sleep_bouts', 'sleep_bout_duration_mean_s', 
                                     'sleep_bout_duration_std_s', 'total_sleep_time_s', 'total_length_frames', 'sleep_time_budget'])
    
    # Filter to sleep bouts (length >= threshold)
    sleep_bouts = standing_bouts[standing_bouts['length'] >= sleep_threshold_frames].copy()
    
    if sleep_bouts.empty:
        # Return rows with 0/NaN values for all Standing bout types that exist
        standing_bout_types = standing_bouts['bout_type'].unique()
        summary_list = []
        for bout_type in standing_bout_types:
            summary_list.append({
                'bout_type': bout_type,
                'n_sleep_bouts': 0,
                'sleep_bout_duration_mean_s': np.nan,  # Can't compute mean of 0 items
                'sleep_bout_duration_std_s': np.nan,   # Can't compute std of 0 items
                'total_sleep_time_s': 0.0,
                'total_length_frames': 0,
                'sleep_time_budget': 0.0  # Filled after aggregation
            })
        return pd.DataFrame(summary_list)
    
    # Group sleep bouts by bout_type
    grouped = sleep_bouts.groupby('bout_type')
    
    summary_list = []
    for bout_type, group_df in grouped:
        n_sleep_bouts = len(group_df)
        total_length_frames = group_df['length'].sum()
        mean_length_frames = group_df['length'].mean()
        std_length_frames = group_df['length'].std(ddof=0) if n_sleep_bouts > 1 else 0.0
        
        # Convert to seconds
        sleep_bout_duration_mean_s = mean_length_frames / fps
        sleep_bout_duration_std_s = std_length_frames / fps
        total_sleep_time_s = total_length_frames / fps
        
        # sleep_time_budget will be recalculated after aggregation using animal_total_frames
        # (to match pattern used for other time budgets)
        
        summary_list.append({
            'bout_type': bout_type,
            'n_sleep_bouts': n_sleep_bouts,
            'sleep_bout_duration_mean_s': sleep_bout_duration_mean_s,
            'sleep_bout_duration_std_s': sleep_bout_duration_std_s,
            'total_sleep_time_s': total_sleep_time_s,
            'total_length_frames': total_length_frames,
            'sleep_time_budget': 0.0  # Filled after aggregation
        })
    
    return pd.DataFrame(summary_list)




# ============================================================================
# DOWNSTREAM ANALYSIS MODULES
# ============================================================================

# 2 Time Budgets
