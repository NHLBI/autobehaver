"""Tiny name-map publication checks with stable directory timestamps."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from autobehaver import portable_paths


def test_worker_publication_refreshes_cached_maps_and_retains_index_reuse(tmp_path, monkeypatch):
    portable_paths.record_original_name(tmp_path/'first.csv', 'biological_A.csv')
    original_signature = portable_paths._map_signature
    # Model the observed Windows behavior: directory metadata stays unchanged,
    # even though another process publishes a completed mapping.
    def stable_directory_signature(directory):
        return ('stable_directory', original_signature(directory)[-1])
    monkeypatch.setattr(portable_paths, '_map_signature', stable_directory_signature)
    original_read = Path.read_text
    reads = []
    def counted_read(path, *args, **kwargs):
        if path.suffix == '.json':
            reads.append(path)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', counted_read)
    first = portable_paths.load_name_maps(tmp_path)
    assert portable_paths.load_name_maps(tmp_path) is first and len(reads) == 1
    package = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, '-B', '-c',
                    'from pathlib import Path; import sys; '
                    'from autobehaver.portable_paths import record_original_name; '
                    'record_original_name(Path(sys.argv[1]), sys.argv[2])',
                    str(tmp_path/'late.csv'), 'biological_B.csv'],
                   env={**os.environ, 'PYTHONPATH':str(package)}, check=True, timeout=30)
    refreshed = portable_paths.load_name_maps(tmp_path)
    assert refreshed == {'first.csv':'biological_A.csv', 'late.csv':'biological_B.csv'}
    assert len(reads) == 3
    assert portable_paths.load_name_maps(tmp_path) is refreshed and len(reads) == 3
    assert not list((tmp_path/'_names').glob('*.tmp'))


def test_old_imported_maps_without_generation_remain_usable_after_copy(tmp_path):
    source = tmp_path/'legacy'
    mapping_dir = source/'_names'
    mapping_dir.mkdir(parents=True)
    (mapping_dir/'old.json').write_text(json.dumps({'short_name':'short.csv',
                                                 'original_name':'biological_original.csv'}))
    destination = tmp_path/'copied'
    shutil.copytree(source, destination)
    assert not (destination/'_names/.generation').exists()
    assert portable_paths.original_name(destination/'short.csv') == 'biological_original.csv'


def test_generated_names_publish_a_generation_and_keep_original_resolution(tmp_path):
    name = 'very_long_original_'+'a'*150+'_normalized_inference.csv'
    result = portable_paths.generated_path(tmp_path, name)
    assert result.name != name
    assert (tmp_path/'_names/.generation').is_file()
    assert portable_paths.original_name(result) == name


def test_target_lookup_does_not_wait_for_unrelated_maps(tmp_path):
    target = tmp_path/'target.csv'
    portable_paths.record_original_name(target, 'unchanged_recording.csv')
    unrelated = tmp_path/'unrelated.csv'
    portable_paths.record_original_name(unrelated, 'other_recording.csv')
    # A valid target must not depend on the shared directory cache or even on
    # an unrelated damaged map. Full-directory validation still reports damage.
    portable_paths._name_map_path(unrelated).write_text('{', encoding='utf-8')
    with patch.object(portable_paths, '_map_signature', side_effect=AssertionError('global scan')):
        assert portable_paths.original_name(target) == 'unchanged_recording.csv'


def test_bulk_discovery_accepts_publication_during_snapshot(tmp_path):
    portable_paths.record_original_name(tmp_path/'target.csv', 'unchanged_recording.csv')
    real_read = Path.read_text
    published = False
    def read_with_publication(path, *args, **kwargs):
        nonlocal published
        value = real_read(path, *args, **kwargs)
        if path.suffix == '.json' and not published:
            published = True
            portable_paths.record_original_name(tmp_path/'late.csv', 'late_recording.csv')
        return value
    with patch.object(Path, 'read_text', read_with_publication):
        first = portable_paths.load_name_maps(tmp_path, refresh=True)
    assert first['target.csv'] == 'unchanged_recording.csv'
    assert portable_paths.load_name_maps(tmp_path)['late.csv'] == 'late_recording.csv'


def test_map_is_complete_before_it_becomes_visible(tmp_path):
    real_replace = portable_paths._replace_map_file
    observed = []
    def checked_replace(temporary, destination):
        if destination.suffix == '.json':
            assert not destination.exists()
            observed.append(json.loads(temporary.read_text(encoding='utf-8')))
        real_replace(temporary, destination)
    with patch.object(portable_paths, '_replace_map_file', checked_replace):
        portable_paths.record_original_name(tmp_path/'target.csv', 'original.csv')
    assert observed == [{'short_name':'target.csv', 'original_name':'original.csv'}]
    assert not list((tmp_path/'_names').glob('*.tmp'))


def test_concurrent_identical_and_conflicting_writers(tmp_path):
    package = Path(__file__).resolve().parents[1]
    worker = '''from pathlib import Path
import sys, time
from autobehaver.portable_paths import record_original_name, generated_path
root=Path(sys.argv[1])
while not (root/'go').exists(): time.sleep(0.01)
try:
    if sys.argv[3]=='record': record_original_name(root/sys.argv[4], sys.argv[2])
    else: generated_path(root, 'long_'+'x'*160+'.csv', original=sys.argv[2])
except ValueError: sys.exit(2)
'''
    for mode in ('record', 'generated'):
        for conflict in (False, True):
            root = tmp_path/f'{mode}_{conflict}'
            root.mkdir()
            writers = []
            try:
                for index in range(4):
                    original = 'B.csv' if conflict and index >= 2 else 'A.csv'
                    spelling = 'TARGET.csv' if index % 2 else 'target.csv'
                    writers.append(subprocess.Popen([sys.executable, '-B', '-c', worker,
                        str(root), original, mode, spelling],
                        env={**os.environ, 'PYTHONPATH':str(package)},
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                (root/'go').write_text('go')
                results = [(process.communicate(timeout=30), process.returncode) for process in writers]
                assert sorted(code for _, code in results) == ([0, 0, 2, 2] if conflict else [0, 0, 0, 0]), results
                maps = list((root/'_names').glob('*.json'))
                assert len(maps) == 1
                assert json.loads(maps[0].read_text())['original_name'] in ('A.csv', 'B.csv')
                assert not list((root/'_names').glob('*.tmp'))
            finally:
                for process in writers:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()


def test_conflicting_legacy_map_is_not_overwritten(tmp_path):
    mapping_dir = tmp_path/'_names'
    mapping_dir.mkdir()
    mapping = mapping_dir/'legacy.json'
    mapping.write_text(json.dumps({'short_name':'target.csv', 'original_name':'A.csv'}))
    try:
        portable_paths.record_original_name(tmp_path/'target.csv', 'B.csv')
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting legacy identity was accepted')
    assert portable_paths.original_name(tmp_path/'target.csv') == 'A.csv'
