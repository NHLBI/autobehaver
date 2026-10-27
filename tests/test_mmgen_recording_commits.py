"""Exercise variable-length completion records and existing recovery guarantees."""
import json

import numpy as np
import pandas as pd
import pytest

from autobehaver.postinf.mmgen import recording_commits as commits
from autobehaver.postinf.mmgen.hdf_store import (
    HDF_KEYS, open_store, hdf_safe_append, get_done_source_files,
    file_has_any_rows, purge_file_from_store,
)
from autobehaver.postinf.mmgen.integrity import validate_commits
from .test_scientific_mmgen_integrity import write_store


def test_growing_records_and_unicode_have_no_text_width(tmp_path):
    path = tmp_path / 'growing.h5'
    expected = []
    with open_store(str(path), 'w') as store:
        for i, length in enumerate((256, 55_648, 1_000_000)):
            row = {'source_file': f'{i}.csv', 'n_frames': 160,
                   'group': '\u03bc\u8774\u8776', 'note': 'x' * length,
                   'output_inventory': json.dumps({'/postural': i}, sort_keys=True),
                   'output_fingerprints': json.dumps({'/postural': {'sha256': 'a' * 64,
                                                                   'columns': ['\u03bc', 'value']}}, sort_keys=True)}
            expected.append(row)
            commits.append_recording_commits(store, pd.DataFrame([row]))
        node = store.get_node(commits.COMMIT_NODE)
        assert node.atom.dtype == np.dtype('uint8') and node.nrows == 3
        assert HDF_KEYS['video_lengths_tmp'] not in store
        pd.testing.assert_frame_equal(commits.read_recording_commits(store), pd.DataFrame(expected))
    assert get_done_source_files(str(path)) == {'0.csv', '1.csv', '2.csv'}
    assert file_has_any_rows(str(path), source_file='2.csv')


def _convert_to_native(path):
    with open_store(str(path), 'a') as store:
        frame = store.select(HDF_KEYS['video_lengths_tmp'])
        store.remove(HDF_KEYS['video_lengths_tmp'])
        commits.append_recording_commits(store, frame)


def test_mixed_legacy_and_native_validation_and_purge(tmp_path):
    path = tmp_path / 'mixed.h5'
    identity = write_store(path)
    with open_store(str(path), 'a') as store:
        frame = store.select(HDF_KEYS['video_lengths_tmp'])
        store.remove(HDF_KEYS['video_lengths_tmp'])
        hdf_safe_append(store, HDF_KEYS['video_lengths_tmp'], frame.iloc[:1])
        commits.append_recording_commits(store, frame.iloc[1:])
    assert validate_commits(path, identity) == {'a.csv', 'b.csv'}
    before = path.read_bytes()
    # Reading old records must not mutate their store.
    assert get_done_source_files(str(path)) == {'a.csv', 'b.csv'}
    assert path.read_bytes() == before
    purge_file_from_store(str(path), source_file='a.csv')
    assert validate_commits(path, identity) == {'b.csv'}
    purge_file_from_store(str(path), source_file='b.csv')
    assert get_done_source_files(str(path)) == set()


def test_native_purge_failure_preserves_entire_original(tmp_path, monkeypatch):
    path = tmp_path / 'purge.h5'
    identity = write_store(path)
    _convert_to_native(path)
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('injected native record rewrite failure')
    with monkeypatch.context() as scoped:
        scoped.setattr(commits, 'append_recording_commits', fail)
        with pytest.raises(OSError, match='native record rewrite'):
            purge_file_from_store(str(path), source_file='a.csv')
    assert path.read_bytes() == before
    assert validate_commits(path, identity) == {'a.csv', 'b.csv'}


def test_duplicate_and_damaged_native_records_are_rejected(tmp_path):
    path = tmp_path / 'damaged.h5'
    identity = write_store(path)
    with open_store(str(path), 'a') as store:
        frame = store.select(HDF_KEYS['video_lengths_tmp'])
        commits.append_recording_commits(store, frame.iloc[:1])
    with pytest.raises(RuntimeError, match='Duplicate'):
        validate_commits(path, identity)
    _convert_path = tmp_path / 'changed.h5'
    identity = write_store(_convert_path)
    _convert_to_native(_convert_path)
    with open_store(str(_convert_path), 'a') as store:
        frame = commits.read_recording_commits(store)
        frame.loc[0, 'n_frames'] = 999
        store._handle.remove_node(commits.COMMIT_NODE)
        commits.append_recording_commits(store, frame)
    with pytest.raises(RuntimeError, match='metadata changed'):
        validate_commits(_convert_path, identity)


def test_invalid_json_is_detected_without_interpreting_pickle(tmp_path):
    path = tmp_path / 'invalid.h5'
    with open_store(str(path), 'w') as store:
        commits.append_recording_commits(store, pd.DataFrame([{'source_file': 'a.csv'}]))
        store.get_node(commits.COMMIT_NODE).append(np.frombuffer(b'broken-json', dtype=np.uint8))
    with open_store(str(path), 'r') as store:
        with pytest.raises(json.JSONDecodeError):
            commits.read_recording_commits(store)


def test_empty_native_commit_storage_does_not_block_recovery(tmp_path):
    from .test_scientific_mmgen_integrity import identity
    path = tmp_path / 'empty-native.h5'
    with open_store(str(path), 'w') as store:
        commits.append_recording_commits(store, pd.DataFrame([{'source_file': 'a.csv'}]))
        store.get_node(commits.COMMIT_NODE).truncate(0)
    assert validate_commits(path, identity()) == set()
    with open_store(str(path), 'a') as store:
        hdf_safe_append(store, HDF_KEYS['postural'], pd.DataFrame({'source_file': ['a.csv'], 'value': [1.]}))
    assert validate_commits(path, identity(), allow_uncommitted=True) == set()
    with pytest.raises(RuntimeError, match='no recording commits'):
        validate_commits(path, identity())


@pytest.mark.parametrize('native_first', [False, True])
def test_mixed_format_shard_merge_preserves_records(tmp_path, native_first):
    from autobehaver.atomic_io import atomic_write_json, sha256_file
    from autobehaver.postinf.mmgen.cohort import bind_shard_cohort
    from autobehaver.postinf.mmgen.integrity import content_hash, IDENTITY_FILENAME
    from autobehaver.postinf.mmgen.shard_merge import merge_shard_temp_stores
    cohort = bind_shard_cohort(tmp_path / '_mmgen_shards', ['a.csv', 'b.csv'],
        inference_csv_dir=tmp_path, num_shards=2, shard_by='row_identity')
    full = None
    for i, name in enumerate('ab'):
        directory = tmp_path / '_mmgen_shards' / f'shard_{i:03d}_of_002'
        work = directory / '_mmgen_work'
        work.mkdir(parents=True)
        path = directory / 'temp_store.h5'
        full = write_store(path, (name,))
        if (i == 0) == native_first:
            _convert_to_native(path)
        identity = dict(full)
        identity['sources'] = {name + '.csv': full['sources'][name + '.csv']}
        identity['sources_sha256'] = content_hash(identity['sources'])
        atomic_write_json(identity, work / IDENTITY_FILENAME)
        atomic_write_json({'num_shards': 2, 'shard_index': i, 'shard_by': 'row_identity',
                          'cohort_sha256': cohort['cohort_sha256'],
                          'selected_files': [name + '.csv'], 'n_selected_files': 1},
                         work / 'SHARD_MANIFEST.json')
        atomic_write_json({'num_shards': 2, 'shard_index': i, 'shard_by': 'row_identity',
                          'cohort_sha256': cohort['cohort_sha256'],
                          'n_inference_files_selected': 1, 'n_committed_files': 1,
                          'calculation_sha256': identity['calculation_sha256'],
                          'sources_sha256': identity['sources_sha256'],
                          'temp_store_sha256': sha256_file(path)}, work / 'SHARD_DONE.json')
    merged = merge_shard_temp_stores(output_folder=str(tmp_path), num_shards=2)
    assert validate_commits(merged, full) == {'a.csv', 'b.csv'}
    assert get_done_source_files(merged) == {'a.csv', 'b.csv'}
