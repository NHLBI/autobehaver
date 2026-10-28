"""Frozen pre-repair smoothing oracle; full source SHA256: 480dabb636a066d72edcb1ec863ce0d6e835d8fd3fb3b9dc2c43ffb2c04828d5."""
import numpy as np
import pandas as pd

def smooth_bouts_vectorized(labels, probs_df, min_bout, prefix, exempt_labels=None):
    """Smooth short bouts using adjacent labels and cumulative flank probabilities.

    First replace short bouts between matching valid neighbors, then recheck
    current short bouts and choose the supported flank. Preserve exempt labels
    and blank/missing gaps. Use positional row alignment, independent of a
    caller's pandas index, and return a label array.
    """
    exempt_labels = set(exempt_labels or [])
    # All indices below are positions, not a caller's pandas index labels.
    s = pd.Series(labels, dtype="object").reset_index(drop=True)
    if len(s) != len(probs_df):
        raise ValueError("Smoothing labels and probability rows must align exactly")
    if s.empty or s.isna().all():
        return s.to_numpy()

    is_blank = s.isna() | (s == "")
    work = s.mask(is_blank, other="__BLANK__")
    df = pd.DataFrame({'label': work})

    df['bout_id'] = (df['label'].ne(df['label'].shift())).cumsum()
    df['is_blank'] = df['label'] == "__BLANK__"
    df['bout_length'] = df.groupby('bout_id')['label'].transform('size')

    df['is_exempt'] = df['label'].isin(exempt_labels)
    df['is_short'] = (~df['is_blank']) & (~df['is_exempt']) & (df['bout_length'] < min_bout)

    # Retain blank bouts: omitting them invents neighbors across missing data.
    bout_info = df.drop_duplicates('bout_id').set_index('bout_id')
    bout_info['prev_label'] = bout_info['label'].shift(1)
    bout_info['next_label'] = bout_info['label'].shift(-1)
    df = df.join(bout_info[['prev_label','next_label']], on='bout_id')

    df['smoothed_label'] = df['label']

    easy = (df['is_short'] & (df['prev_label'] == df['next_label'])
            & df['prev_label'].notna() & (df['prev_label'] != "__BLANK__"))
    df.loc[easy, 'smoothed_label'] = df.loc[easy, 'prev_label']

    df['bout_id_smoothed'] = (df['smoothed_label'].ne(df['smoothed_label'].shift())).cumsum()
    df['bout_length_smoothed'] = df.groupby('bout_id_smoothed')['smoothed_label'].transform('size')

    current_exempt = df['smoothed_label'].isin(exempt_labels)
    complex_ids = df.loc[(~df['is_blank']) & (df['bout_length_smoothed'] < min_bout)
                         & (~current_exempt), 'bout_id_smoothed'].unique()
    processed = np.zeros(len(df), dtype=bool)

    for bid in complex_ids:
        idx = df.index[df['bout_id_smoothed'] == bid]
        anchor = int(idx.min())
        current_label = df['smoothed_label'].iloc[anchor]
        if current_label == "__BLANK__" or current_label in exempt_labels:
            continue
        # Earlier candidates can change adjacency. Recheck the CURRENT bout,
        # and process each current group at most once in this second pass.
        start_idx = end_idx = anchor
        while start_idx > 0 and df['smoothed_label'].iloc[start_idx - 1] == current_label:
            start_idx -= 1
        while end_idx + 1 < len(df) and df['smoothed_label'].iloc[end_idx + 1] == current_label:
            end_idx += 1
        if (end_idx - start_idx + 1 >= min_bout
                or processed[start_idx:end_idx + 1].any()
                or df['is_exempt'].iloc[start_idx:end_idx + 1].any()):
            continue

        prev_label = df['smoothed_label'].iloc[start_idx - 1] if start_idx > 0 else None
        next_label = df['smoothed_label'].iloc[end_idx + 1] if end_idx < len(df) - 1 else None
        if prev_label in [None, "__BLANK__"]:
            prev_label = None
        if next_label in [None, "__BLANK__"]:
            next_label = None
        if prev_label is None and next_label is None:
            continue

        def prob_sum(lbl):
            if lbl in [None, "__BLANK__"]:
                return 0.0
            col = f"{prefix}-{lbl}-Prob"
            return probs_df.iloc[start_idx:end_idx+1][col].sum() if col in probs_df.columns else 0.0

        prev_sum, next_sum = prob_sum(prev_label), prob_sum(next_label)
        # A blank/missing flank is never a candidate, even when its numerical
        # evidence ties a valid neighbor at zero.
        if prev_label is None:
            winner = next_label
        elif next_label is None:
            winner = prev_label
        else:
            winner = prev_label if prev_sum >= next_sum else next_label
        df.loc[start_idx:end_idx, 'smoothed_label'] = winner
        processed[start_idx:end_idx + 1] = True

    out = df['smoothed_label'].replace({"__BLANK__": np.nan}).to_numpy()

    return out
