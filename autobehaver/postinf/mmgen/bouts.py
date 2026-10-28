"""Extract valid label bouts and calculate recording-level bout and sleep summaries for
MMGen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd



def extract_bouts_from_label_sequence(label_sequence, min_bout=1, exempt_labels=None):
    """Extract consecutive nonmissing label runs as bouts with inclusive start and end
    indices.

    Require min_bout frames, except exempt poses and their combined labels, which
    require one frame.
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

    known = np.array([pd.notna(v) and v != "" for v in label_sequence], dtype=bool)
    segments = np.cumsum(known & ~np.r_[False, known[:-1]]) if len(known) else np.array([])
    for bout in bouts:
        start, end = bout['start'], bout['end']
        bout['complete'] = bool(start > 0 and end + 1 < len(known)
                                and known[start - 1] and known[end + 1])
        bout['observation_segment'] = int(segments[start])
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
    """Summarize complete bouts, retaining all eligible frame time for budgets.

    total_frames is the number of observed labels in this category. Intervals
    require complete endpoints in the same uninterrupted observation segment.
    Duration and interval moments allow exact pooling across separate recordings.
    """
    if not bouts:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(b, length=b['end'] - b['start'] + 1) for b in bouts])
    if 'complete' not in frame:
        frame['complete'] = True
    if 'observation_segment' not in frame:
        frame['observation_segment'] = 0
    rows = []
    for label, all_bouts in frame.groupby('label'):
        complete = all_bouts[all_bouts['complete']].sort_values('start')
        durations = complete['length'].to_numpy(dtype=float) / fps
        intervals = []
        for _, segment in all_bouts.sort_values('start').groupby('observation_segment', sort=False):
            records = segment.to_dict('records')
            for left, right in zip(records, records[1:]):
                if left['complete'] and right['complete']:
                    intervals.append((right['start'] - left['end']) / fps)
        intervals = np.asarray(intervals, dtype=float)
        n = len(durations)
        mean = float(durations.mean()) if n else np.nan
        total_length = float(all_bouts['length'].sum())
        seconds = total_frames / fps
        rows.append(dict(
            bout_type=label, n_bouts=n, total_length_frames=total_length,
            observed_frames=float(total_frames), observed_seconds=float(seconds),
            duration_sum_s=float(durations.sum()), duration_sumsq_s=float(np.square(durations).sum()),
            interval_count=len(intervals), interval_sum_s=float(intervals.sum()),
            interval_sumsq_s=float(np.square(intervals).sum()),
            bout_frequency=n / seconds if seconds > 0 else np.nan,
            bout_duration_mean_s=mean,
            bout_duration_coeff_of_var=float(durations.std(ddof=0)) / mean if n and mean > 0 else np.nan,
            initiation_prob=n / (total_frames - total_length) if total_frames > total_length else np.nan,
            inter_bout_interval_mean_s=float(intervals.mean()) if len(intervals) else np.nan,
            inter_bout_interval_std_s=float(intervals.std(ddof=0)) if len(intervals) else np.nan,
        ))
    return pd.DataFrame(rows)


def calculate_sleep_features(bouts, total_frames, fps=40, sleep_threshold_frames=7200):
    """Calculate sleep-specific features for Standing bouts >= sleep_threshold_frames.

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
    bouts = [b for b in bouts if b.get('complete', True)]
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





