"""Bridge brief internal label gaps without crossing recording boundaries."""

import numpy as np
import pandas as pd

MAX_LABEL_GAP_FRAMES = 4


def fill_short_label_gaps(labels, max_gap_frames=MAX_LABEL_GAP_FRAMES):
    """Fill missing runs of at most four frames between matching known labels.

    Return labels and a mask of inferred labels. Coordinates and probabilities
    are never synthesized. Call separately for each physical recording.
    """
    result = np.asarray(labels, dtype=object).copy()
    missing = pd.isna(result)
    missing[~missing] |= result[~missing] == ""
    filled = np.zeros(len(result), dtype=bool)
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if (0 < start and stop < len(result) and stop - start <= max_gap_frames
                and result[start - 1] == result[stop]):
            result[start:stop] = result[start - 1]
            filled[start:stop] = True
    return result, filled
