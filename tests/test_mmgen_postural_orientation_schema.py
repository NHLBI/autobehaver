"""Keep postural orientation summaries within the declared feature contract."""

import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import sha256_file
from autobehaver.postinf.mmgen.feature_schema import feature_columns
from autobehaver.postinf.mmgen.hdf_store import HDF_KEYS, hdf_safe_append, open_store
from autobehaver.postinf.mmgen.postural import (
    calculate_and_save_postural_summary,
    calculate_postural_summary_from_store,
)


def mixed_orientation_frames():
    identities = ['g_FEMALE_rpi-1_trial1_movie0_chamber1',
                  'g_FEMALE_rpi-1_trial1_movie1_chamber1']
    orientations = ['Floor', 'Floor', 'Ceiling', 'Ceiling', 'Wall-Side', 'Wall-Side'] * 2
    return pd.DataFrame({
        'individual': np.repeat(identities, 6),
        'group': ['g'] * 12,
        'masked_orient': orientations,
        'stance_area': [1., 2., 3., 4., np.nan, np.nan, 7., 8., 9., 10., np.nan, np.nan],
        'extension_raw_FrontLegs_avg': [1., 2., 3., 4., np.nan, np.nan] * 2,
        'asymmetry_magnitude_Abdomen_LR': [0., 0., np.nan, np.nan, np.nan, np.nan] * 2,
        'asymmetry_magnitude_Wings_LR': [np.nan] * 12,
        'asymmetry_magnitude_Abdomen_TB': [np.nan, np.nan, np.nan, np.nan, 1., 2.] * 2,
    })


def test_mixed_orientation_summaries_preserve_math_zeros_and_declared_blanks(tmp_path):
    source = mixed_orientation_frames()
    memory = calculate_and_save_postural_summary(source, str(tmp_path), save_outputs=False)
    path = tmp_path / 'postural.h5'
    with open_store(str(path), 'w') as store:
        hdf_safe_append(store, HDF_KEYS['postural'], source)
    digest = sha256_file(path)
    streamed = calculate_postural_summary_from_store(
        str(path), str(tmp_path), save_outputs=False, chunk_rows=2)
    pd.testing.assert_frame_equal(memory, streamed, check_exact=False, rtol=1e-12, atol=1e-12)
    assert sha256_file(path) == digest

    assert all('Postural_' + column in feature_columns()
               for column in memory if column not in {'individual', 'group'})
    row = memory.iloc[0]
    assert row['individual'] == 'g_FEMALE_rpi-1_trial1_chamber1'
    assert row['Floor_asymmetry_magnitude_Abdomen_LR_mean'] == 0
    assert row['Floor_asymmetry_magnitude_Abdomen_LR_std'] == 0
    assert row['Wall-Side_asymmetry_magnitude_Abdomen_TB_mean'] == 1.5
    assert np.isclose(row['Wall-Side_asymmetry_magnitude_Abdomen_TB_std'],
                      np.std([1, 2, 1, 2], ddof=1))
    assert row['Floor_extension_raw_FrontLegs_avg_mean'] == 1.5
    assert row['Ceiling_extension_raw_FrontLegs_avg_mean'] == 3.5
    assert pd.isna(row['Floor_asymmetry_magnitude_Wings_LR_mean'])
    assert pd.isna(row['Floor_asymmetry_magnitude_Wings_LR_std'])
    assert 'Floor_asymmetry_magnitude_Abdomen_TB_mean' not in memory
    assert 'Wall-Side_asymmetry_magnitude_Abdomen_LR_mean' not in memory
    assert 'Wall-Side_extension_raw_FrontLegs_avg_mean' not in memory
    assert 'Wall-Side_stance_area_mean' not in memory


@pytest.mark.parametrize('value', [0., 1.])
def test_undeclared_postural_measurements_fail_instead_of_losing_values(tmp_path, value):
    source = mixed_orientation_frames()
    source['unknown_measurement'] = value
    with pytest.raises(ValueError, match='Postural summary contains measurements for undeclared features'):
        calculate_and_save_postural_summary(source, str(tmp_path), save_outputs=False)


def test_undeclared_empty_combinations_do_not_remove_declared_empty_features(tmp_path):
    source = mixed_orientation_frames()
    source['unknown_measurement'] = np.nan
    actual = calculate_and_save_postural_summary(source, str(tmp_path), save_outputs=False)
    assert not any('unknown_measurement' in column for column in actual)
    assert actual['Floor_asymmetry_magnitude_Wings_LR_mean'].isna().all()
