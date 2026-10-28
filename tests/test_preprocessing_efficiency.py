"""Tiny conservation and failure-boundary checks for bounded preprocessing."""
import os
from concurrent.futures.process import BrokenProcessPool

import h5py
import numpy as np
import pandas as pd
import pytest

from autobehaver.preinf.keyprenorm.csv_chunks import RetainedCsvChunks
from autobehaver.preinf.keyprenorm import normparams
from autobehaver.postinf.mmgen import kinematics
from autobehaver.postinf.mmgen.batching import ordered_worker_results
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize('length', [0, 3, 5, 10, 11, 31])
@pytest.mark.parametrize('edge', [0, 5])
def test_chunk_trim_preserves_full_source_timeline(tmp_path, length, edge):
    source = tmp_path / 'features.csv'
    original = pd.DataFrame({'b': np.arange(length, dtype=np.float32),
                             'a': np.arange(length, dtype=np.float32) / 3})
    original.to_csv(source, index=False)
    chunks = RetainedCsvChunks(source, ['a', 'b'], edge, chunk_rows=4)
    retained = list(chunks)
    actual = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=['a', 'b'])
    expected = pd.read_csv(source, dtype=np.float32)[['a', 'b']].iloc[edge:max(edge, length-edge)]
    np.testing.assert_array_equal(actual.to_numpy(dtype=np.float32), expected.to_numpy())
    assert chunks.source_rows == length
    assert all(len(chunk) <= 4 for chunk in retained)


def test_chunk_moments_match_centered_finite_population(tmp_path, monkeypatch):
    source = tmp_path / 'features.csv'
    values = (10_000 + np.arange(31)[:, None] * np.array([.01, .1, 1])).astype(np.float32)
    values[8:12, 1] = np.nan
    pd.DataFrame(values, columns=['a', 'b', 'c']).to_csv(source, index=False)
    monkeypatch.setattr(normparams, '_STATS_CANONICAL_COLUMNS', ['a', 'b', 'c'])
    monkeypatch.setattr(normparams, '_STATS_EDGE_SKIP', 5)
    monkeypatch.setattr(normparams, 'RetainedCsvChunks',
                        lambda path, columns, edge, **kwargs: RetainedCsvChunks(path, columns, edge, chunk_rows=4))
    mean, m2, count = normparams.process_stats_batch([str(source)])
    reference = pd.read_csv(source, dtype=np.float32).to_numpy()[5:-5].astype(np.float64)
    expected_mean = np.nanmean(reference, axis=0)
    np.testing.assert_array_equal(count, np.isfinite(reference).sum(axis=0))
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-10)
    np.testing.assert_allclose(m2, np.nansum((reference-expected_mean)**2, axis=0), rtol=1e-10)


@pytest.mark.parametrize('include_jerk', [True, False])
def test_cached_head_signals_preserve_all_summary_conventions(include_jerk):
    random = np.random.default_rng(7)
    data = {}
    for kp in kinematics.HEAD_KEYPOINTS:
        data[kp+'_likelihood'] = random.random(41)
        for family in ['partvelocity', 'partaccel'] + (['partjerk'] if include_jerk else []):
            data[f'{family}_{kp}_inst'] = random.normal(size=41)
    frame = pd.DataFrame(data)
    frame.loc[0, [c for c in frame if c.endswith('likelihood')]] = np.nan
    frame.iloc[2:4, 1] = np.nan
    signals = kinematics.compute_head_movement_frame_signals(frame)
    original = kinematics.compute_head_movement_features(frame)
    cached = kinematics.compute_head_movement_features(frame, frame_signals=signals)
    np.testing.assert_array_equal(original, cached)
    original_median = kinematics.compute_head_movement_features_median_skewness(frame, '_inst', *original)
    cached_median = kinematics.compute_head_movement_features_median_skewness(frame, '_inst', *cached, frame_signals=signals)
    np.testing.assert_array_equal(original_median, cached_median)
    original_limits = kinematics.compute_head_movement_features_minmax(frame, '_inst')
    cached_limits = kinematics.compute_head_movement_features_minmax(frame, '_inst', frame_signals=signals)
    np.testing.assert_array_equal(original_limits, cached_limits)


def _square(value):
    return value * value


def _abrupt_exit(value):
    if value == 0:
        os._exit(7)
    return value


def test_bounded_workers_keep_input_order_across_recycling():
    assert list(ordered_worker_results(_square, list(range(5)), num_workers=2,
                                      tasks_per_worker=1)) == [0, 1, 4, 9, 16]


def test_abrupt_worker_exit_is_reported():
    with pytest.raises(BrokenProcessPool):
        list(ordered_worker_results(_abrupt_exit, [0, 1], num_workers=2))


def test_postural_stream_matches_pooled_animal_statistics(tmp_path):
    from autobehaver.postinf.mmgen.postural import calculate_and_save_postural_summary, calculate_postural_summary_from_store
    from autobehaver.postinf.mmgen.naming import collapse_individual_id
    from autobehaver.postinf.mmgen.hdf_store import open_store, hdf_safe_append, HDF_KEYS
    from autobehaver.atomic_io import sha256_file
    ids = ['g1_FEMALE_rpi-1_trial1_movie0_chamber5',
           'g1_FEMALE_rpi-1_trial1_movie1_chamber5',
           'g1_MALE_rpi-2_trial1_movie0_chamber6']
    frame = pd.DataFrame({'group': ['g1'] * 12,
        'individual': np.repeat(ids, 4), 'source_file': np.repeat(['a', 'b', 'c'], 4),
        'masked_orient': ['Floor', 'Floor', 'Ceiling', 'Ceiling'] * 3,
        'stance_area': np.linspace(1, 5, 12),
        'extension_raw_FrontLegs_avg': np.linspace(1, 2, 12)})
    frame.loc[2, 'stance_area'] = np.nan
    identities = {value: collapse_individual_id(value) for value in ids}
    normalizers = pd.Series({value: 2.0 for value in identities.values()})
    expected = frame.copy()
    expected['extension_norm_FrontLegs_avg'] = expected['extension_raw_FrontLegs_avg'] / 2
    reference = calculate_and_save_postural_summary(expected, str(tmp_path), save_outputs=False)
    path = tmp_path / 'postural.h5'
    with open_store(str(path), 'w') as store:
        hdf_safe_append(store, HDF_KEYS['postural'], frame)
    digest = sha256_file(path)
    actual = calculate_postural_summary_from_store(str(path), str(tmp_path),
        normalizer_map=normalizers, individual_to_animal=identities,
        save_outputs=False, chunk_rows=3)
    assert_frame_equal(actual, reference, check_exact=False, rtol=1e-12, atol=1e-12)
    assert sha256_file(path) == digest


def test_postural_subset_preserves_optional_wallside_probability():
    from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
    from autobehaver.postinf.mmgen.postural import _calculate_postural_features
    columns = required_keypre_columns(include_postural=True, include_spatial=False,
                                     include_measurements=False)
    frame = pd.DataFrame({column: np.full(12, .9 if column.endswith('_likelihood') else 1.)
                          for column in columns})
    frame['orient_Wall-Side'] = np.array([.9] * 8 + [.4, .8, .74, .9])
    poses = np.repeat('Standing', 12)
    orientations = np.repeat(['Floor', 'Ceiling', 'Wall-Side'], 4)
    actual = _calculate_postural_features(frame, poses, orientations, 'tiny.mp4', 'g', 'animal')
    assert actual.groupby('masked_orient').size().to_dict() == {'Ceiling': 4, 'Floor': 4, 'Wall-Side': 2}


def test_fused_commit_verification_reads_scientific_table_once(tmp_path, monkeypatch):
    from tests.test_scientific_mmgen_integrity import write_store
    from autobehaver.postinf.mmgen.integrity import validate_commits
    path = tmp_path / 'commits.h5'
    payload = write_store(path)
    original = pd.HDFStore.select
    scans = []
    def observe(self, key, *args, **kwargs):
        if kwargs.get('iterator') or kwargs.get('stop', 0) > kwargs.get('start', 0):
            scans.append(key)
        return original(self, key, *args, **kwargs)
    monkeypatch.setattr(pd.HDFStore, 'select', observe)
    assert validate_commits(path, payload) == {'a.csv', 'b.csv'}
    assert scans.count('/postural') == 1


@pytest.mark.parametrize('changed', ['source', 'output'])
def test_keypre_reuse_handoff_detects_changed_content(tmp_path, changed):
    from tests.test_keypre_raw_only import _write_synthetic_dlc_csv
    from autobehaver.preinf.keypre.worker import set_run_config, process_csv_file
    from autobehaver.preinf.keypre.contract import output_path, inspect_keypre_output
    from autobehaver.completion import file_signature, verified_identity_scope
    source = tmp_path / 'G_MALE_exp_trial1_rpi-1_movie0_chamber1DLC_test.csv'
    _write_synthetic_dlc_csv(str(source), n_frames=31)
    set_run_config({'output_dir': str(tmp_path), 'clobber': False, 'movie_dim': 600, 'raw_only': True})
    assert process_csv_file(str(source))['status'] == 'success'
    output = output_path(source, tmp_path)
    with verified_identity_scope():
        inspection = inspect_keypre_output(source, output, raw_only=True, require_receipt=True)
        cached = (file_signature(source), file_signature(output), inspection)
        assert inspect_keypre_output(source, output, raw_only=True, require_receipt=True,
                                     cached_inspection=cached)[0]
        if changed == 'source':
            source.write_bytes(source.read_bytes() + b'\n')
        else:
            data = pd.read_csv(output)
            data.iloc[10, 0] += 1
            data.to_csv(output, index=False)
        assert not inspect_keypre_output(source, output, raw_only=True, require_receipt=True,
                                         cached_inspection=cached)[0]


@pytest.mark.parametrize('changed', ['source', 'output'])
def test_normalization_reuse_handoff_detects_changed_content(tmp_path, monkeypatch, changed):
    from autobehaver.preinf.keyprenorm import core, normalize
    from autobehaver.preinf.keyprenorm.identity import normalization_identity
    from autobehaver.portable_paths import normalized_path
    from autobehaver.completion import file_signature, file_digest, verified_identity_scope
    source = tmp_path / 'tiny_positional_calcs.csv'
    columns = ['a', 'b']
    pd.DataFrame(np.arange(24).reshape(12, 2), columns=columns).to_csv(source, index=False)
    means, stds = np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    for name, value in {'_NORM_CANONICAL_COLUMNS': columns, '_NORM_MEANS': means,
        '_NORM_STDS': stds, '_NORM_EDGE_SKIP': 5, '_NORM_OUTPUT_DIR': str(tmp_path),
        '_NORM_LABELS_INPUT_DIR': None, '_NORM_LABELS_OUTPUT_DIR': None}.items():
        monkeypatch.setattr(normalize, name, value)
    assert normalize.normalize_one_csv(str(source))['status'] == 'success'
    output = normalized_path(source, tmp_path)
    with verified_identity_scope():
        cache = {str(source): {'rows': 2,
            'identity': normalization_identity(source, columns, means, stds, 5),
            'source_signature': file_signature(source), 'output_signature': file_signature(output),
            'output_digest': file_digest(output)}}
        def report():
            return core._write_keyprenorm_validation(all_csv_files=[str(source)],
                canonical_columns=columns, base_output_folder=str(tmp_path), edge_skip=5,
                bad_schema=[], reused_files={str(source)}, verified_outputs=cache,
                normalization_parameters=(means, stds))
        assert report().status == 'passed'
        if changed == 'source':
            frame = pd.read_csv(source)
            frame.iloc[5, 0] += 1
            frame.to_csv(source, index=False)
        else:
            with h5py.File(output, 'r+') as hf:
                hf['data'][0, 0] += 1
        assert report().status == 'failed'
