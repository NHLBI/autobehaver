"""Per-recording identity metadata cannot enter numerical posture summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from autobehaver.postinf.mmgen.postural import calculate_and_save_postural_summary


def test_postural_source_identity_excluded_and_numeric_math_preserved(tmp_path):
    source = pd.DataFrame(dict(
        individual=['A'] * 4, group=['G'] * 4, masked_orient=['Floor'] * 4,
        masked_pose=['Standing'] * 4, source_file=['r1.csv'] * 2 + ['r2.csv'] * 2,
        frame_idx=[0, 1, 0, 1], sex=['FEMALE'] * 4,
        stance_area=[1., 2., 3., 4.], extension_raw_FrontLegs_avg=[0., 2., 4., 6.]))
    actual = calculate_and_save_postural_summary(source, str(tmp_path), save_outputs=False)
    row = actual.set_index('individual').loc['A']
    assert row['Floor_stance_area_mean'] == 2.5
    assert np.isclose(row['Floor_stance_area_std'], np.std([1, 2, 3, 4], ddof=1))
    assert row['Floor_extension_raw_FrontLegs_avg_mean'] == 3
    assert not any('source_file' in c or 'frame_idx' in c or 'sex' in c for c in actual)
    legacy = calculate_and_save_postural_summary(
        source.drop(columns=['source_file', 'frame_idx', 'sex']),
        str(tmp_path), save_outputs=False)
    pd.testing.assert_frame_equal(actual, legacy)
