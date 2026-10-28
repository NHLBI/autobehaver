"""An animal's transition probability cannot depend on unrelated animals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autobehaver.postinf.mmgen import transitions
from autobehaver.postinf.mmgen.categories import POSE_LABELS, SUMMARY_COMBINED_LABELS


@pytest.mark.parametrize('category,standing,walking,groom,labels,result_key', [
    ('Pose', 'Standing', 'Walking', 'Groom-Head', POSE_LABELS, 'pose_tidy_df'),
    ('Pose+Orient', 'Standing + Floor', 'Walking + Floor', 'Groom-Head + Floor',
     tuple(SUMMARY_COMBINED_LABELS) + ('Jump', 'Scramble'), 'pose_orient_tidy_df'),
])
def test_declared_transition_alphabet_and_zero_denominator(
        tmp_path, category, standing, walking, groom, labels, result_key):
    counts = pd.DataFrame([
        dict(individual='A', group='G', category=category,
             from_behavior=standing, to_behavior=walking, count=2),
        dict(individual='B', group='G', category=category,
             from_behavior=groom, to_behavior=walking, count=3),
    ])

    def calculate(rows):
        result = transitions.calculate_and_save_transition_analysis(
            rows, str(tmp_path), save_outputs=False,
            run_transition_statistics=False)
        return result[result_key].set_index(['individual', 'from_behavior', 'to_behavior'])

    cohort = calculate(counts)
    shard = calculate(counts.iloc[:1])
    # Exact all-cell equality for A, including undefined departure rows.
    pd.testing.assert_frame_equal(cohort.loc[['A']], shard.loc[['A']])
    assert len(shard) == len(labels) * (len(labels) - 1)
    assert shard.loc[('A', standing, groom), 'probability'] == 0
    assert shard.loc[('A', standing, walking), 'probability'] == 1
    assert np.isnan(shard.loc[('A', groom, walking), 'probability'])
    # Valid departure row sums to one across the full declared alphabet.
    departures = shard.xs(('A', standing), level=('individual', 'from_behavior'))
    assert departures['probability'].sum() == 1


def test_no_departures_do_not_create_transition_observations(tmp_path):
    empty = pd.DataFrame(columns=[
        'individual', 'group', 'category', 'from_behavior', 'to_behavior', 'count'])
    result = transitions.calculate_and_save_transition_analysis(
        empty, str(tmp_path), save_outputs=False, run_transition_statistics=False)
    assert result['pose_tidy_df'].empty
    assert result['pose_orient_tidy_df'].empty
