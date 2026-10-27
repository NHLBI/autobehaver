"""Identify fully observed grooming episodes within one recording."""

import numpy as np
import pandas as pd


def complete_grooming_episodes(labels, frame_indices, max_gap_frames):
    """Return inclusive episode positions with observed nongrooming flanks.

    Existing short nongrooming gaps may join grooming subtypes. Missing labels,
    absent frame rows, and recording boundaries never join episodes. A flank
    must exceed the allowed internal gap to establish the episode boundary.
    """
    frames = np.asarray(frame_indices, dtype=int)
    labels = np.asarray(labels, dtype=object)
    episodes = []
    start = last_groom = None
    gap = 0
    preceding = 0
    complete_start = False
    for i, label in enumerate(labels):
        if i and frames[i] != frames[i - 1] + 1:
            start = last_groom = None
            gap = preceding = 0
        if pd.isna(label) or label == '':
            start = last_groom = None
            gap = preceding = 0
            continue
        if str(label).startswith('Groom-'):
            if start is None:
                start = i
                complete_start = preceding > max_gap_frames
            last_groom = i
            gap = preceding = 0
        elif start is not None:
            gap += 1
            if gap > max_gap_frames:
                if complete_start:
                    episodes.append((start, last_groom))
                start = last_groom = None
                preceding = gap
                gap = 0
        else:
            preceding += 1
    return episodes
