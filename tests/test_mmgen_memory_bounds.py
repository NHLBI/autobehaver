"""Prevent whole-cohort frame reads and silent finalizer worker deaths."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from autobehaver.postinf.mmgen import occupancy
from autobehaver.postinf.mmgen.recording_commits import append_recording_commits
from autobehaver.postinf.mmgen.table_metadata import read_individual_metadata
from autobehaver.postinf.mmgen.spatial_stream import compute_spatial_feature_summary_csv
from autobehaver.postinf.mmgen.feature_matrix import compute_spatial_feature_summary_df
from autobehaver.postinf.mmgen.io import read_metadata_csv
from tests import test_mmgen_finalize_parallelism as fixtures


def test_recording_inventory_avoids_all_frame_reads_and_keeps_zero_bout_animals(tmp_path, monkeypatch):
    path = tmp_path / 'data.h5'
    frames = pd.DataFrame({'individual': ['a', 'b'], 'group': ['G', 'G'], 'sex': ['F', 'M']})
    commits = frames.copy()
    commits['source_file'] = ['f1', 'f2']
    commits['output_inventory'] = [json.dumps({'/positional': 40, '/bouts_inst': 0}),
                                   json.dumps({'/positional': 0, '/bouts_inst': 0})]
    with pd.HDFStore(path, 'w') as store:
        store.put('/positional', frames, format='table')
        append_recording_commits(store, commits)
    with pd.HDFStore(path, 'r') as store:
        def forbidden(*args, **kwargs):
            raise AssertionError('Metadata must not read the frame table')
        monkeypatch.setattr(store, 'select', forbidden)
        result = read_individual_metadata(store, '/positional')
    assert result['individual'].tolist() == ['a']
    assert result['group'].tolist() == ['G']


def test_legacy_metadata_scan_is_bounded_and_exact(tmp_path, monkeypatch):
    path = tmp_path / 'legacy.h5'
    frame = pd.DataFrame({'individual': ['a', 'b', 'a'] * 19,
                          'group': ['NA'] * 57, 'sex': ['F'] * 57,
                          'large_unused_text': ['x' * 512] * 57})
    frame.to_hdf(path, key='positional', format='table')
    with pd.HDFStore(path, 'r') as store:
        original = store.select
        calls = []
        def bounded(key, *args, **kwargs):
            assert kwargs['iterator'] and kwargs['chunksize'] == 7
            calls.append(key)
            return original(key, *args, **kwargs)
        monkeypatch.setattr(store, 'select', bounded)
        result = read_individual_metadata(store, 'positional', chunk_rows=7)
    assert calls == ['/positional']
    assert_frame_equal(result, frame[['group', 'individual', 'sex']].drop_duplicates().reset_index(drop=True),
                       check_dtype=False)




@pytest.mark.parametrize('workers', [1, 2])
def test_spatial_csv_stream_matches_all_at_once_with_interleaved_recordings(tmp_path, monkeypatch, workers):
    # Each animal occurs in both recordings and in many separate CSV chunks.
    rows = []
    for movie in range(2):
        for frame in range(18):
            for animal in ['001', 'NA']:
                pose = 'Standing' if frame < 4 or frame > 14 else 'Groom-Head'
                rows.append(dict(individual=animal, source_file=f'{animal}_{movie}', frame_idx=frame,
                    r_norm=(frame + movie + 1) / 30, relative_angle=frame * 4.1,
                    speed_mm_s=frame / 3.1, masked_pose=pose, masked_pose_valid=pose,
                    masked_orient='Floor', masked_orient_valid='Floor',
                    masked_pose_orient=pose+' + Floor', masked_pose_orient_valid=pose+' + Floor',
                    anglefeature_Wings_L_R=frame * .1))
    path = tmp_path / 'positions.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    master = pd.DataFrame({'individual': ['NA', 'missing', '001']})
    expected = compute_spatial_feature_summary_df(read_metadata_csv(path), None, master,
                                                  video_fps=1, num_workers=workers)
    from autobehaver.postinf.mmgen import spatial_stream
    original = spatial_stream.read_metadata_csv
    def bounded(*args, **kwargs):
        assert kwargs['chunksize'] == 7
        return original(*args, **kwargs)
    monkeypatch.setattr(spatial_stream, 'read_metadata_csv', bounded)
    work = tmp_path / 'work'
    actual = compute_spatial_feature_summary_csv(path, None, master, video_fps=1,
                    num_workers=workers, work_dir=work, chunk_rows=7)
    assert_frame_equal(actual.reindex(columns=expected.columns), expected,
                       check_dtype=False, check_exact=True)
    assert list(work.iterdir()) == []


def abrupt_worker_exit(task):
    os._exit(23)


def test_finalizer_reports_abrupt_worker_exit_instead_of_hanging(tmp_path):
    # An outer deadline prevents a future regression from hanging the test run.
    script = tmp_path / 'worker_death.py'
    script.write_text('''
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from tests.test_mmgen_finalize_parallelism import TestMMGenFinalizeParallelism
from tests.test_mmgen_memory_bounds import abrupt_worker_exit
from autobehaver.postinf.mmgen import occupancy
if __name__ == '__main__':
    root = Path(__file__).parent
    store = str(root / 'death.h5')
    TestMMGenFinalizeParallelism()._write_store(store)
    try:
        occupancy._occupancy_transform_chunk_worker = abrupt_worker_exit
        occupancy.calculate_and_save_occupancy_analysis(store, '/positional', str(root),
            num_workers=2, save_outputs=False, transformed_output_path=str(root / 'positions.csv'))
    except BrokenProcessPool:
        print('Worker death reported')
    else:
        raise AssertionError('Missing worker failure')
''', encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), env.get('PYTHONPATH')]))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=40, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Worker death reported' in result.stdout
