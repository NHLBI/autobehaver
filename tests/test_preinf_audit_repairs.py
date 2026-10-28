"""Publication/identity boundaries repaired after the October audit."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from autobehaver.completion import file_digest
from autobehaver.portable_paths import normalized_path, record_original_name


def test_dlc_collisions_fail_before_backend_or_staging(monkeypatch, tmp_path):
    from autobehaver.preinf.dlc import core
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    for folder in ('a', 'b'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'same.mp4').touch()
    monkeypatch.setattr(core, '_require_deeplabcut', lambda: pytest.fail('backend initialized before name preflight'))
    cfg = DLCInferencerConfig(input_dir=str(tmp_path), dlc_config=str(tmp_path / 'unused.yaml'), recursive=True)
    with pytest.raises(ValueError, match='name collision'):
        core.run_dlc(cfg)
    with pytest.raises(ValueError, match='name collision'):
        core.analyze_videos_with_retries(config_path='unused', videos=[str(tmp_path / name / 'same.mp4') for name in ('a', 'b')],
            videotype='.mp4', shuffle=1, save_as_csv=True, max_retries=1)


def test_dlc_discovery_excludes_own_history_and_labeled_outputs(tmp_path):
    from autobehaver.preinf.dlc.core import resolve_videos
    good = tmp_path / 'a.mp4'
    good.touch()
    (tmp_path / 'a_labeled.mp4').touch()
    for name in ('_old', 'retry_attempt_1', '.dlc-work'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'b.mp4').touch()
    assert resolve_videos(str(tmp_path), recursive=True) == [str(good)]
    assert resolve_videos(str(tmp_path), videos=['a_labeled.mp4']) == [str(tmp_path / 'a_labeled.mp4')]


def test_dlc_model_identity_includes_effective_test_pose_config(tmp_path):
    from tests.test_dlc_project_layout import _project
    from autobehaver.preinf.dlc.contract import model_identity
    config, model = _project(tmp_path)
    before = model_identity(config, 1)
    (model / 'test/pose_cfg.yaml').write_text('net_type: resnet_50\nlocation_refinement: false\n')
    after = model_identity(config, 1)
    assert before != after
    assert before['snapshot_files'] == after['snapshot_files']


def _normalized_pair(tmp_path, monkeypatch):
    from autobehaver.preinf.keyprenorm import normalize
    source = tmp_path / 'a_positional_calcs.csv'
    pd.DataFrame({'x': np.arange(12), 'y': np.arange(12) * 2}).to_csv(source, index=False)
    labels = tmp_path / 'labels'
    labels.mkdir()
    annotation = labels / 'a_labels.csv'
    pd.DataFrame({'Frame': np.arange(12), 'Behavior': ['rest', 'walk'] * 6,
                  'Orientation': ['floor'] * 12}).to_csv(annotation, index=False)
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(normalize, 'setup_worker_logging', lambda *args: None)
    normalize.init_normalize_worker(['x', 'y'], np.zeros(2), np.ones(2), 2,
        str(out), str(labels), str(out), None, 'WARNING')
    assert normalize.normalize_one_csv(str(source))['status'] == 'success'
    return normalize, source, annotation, normalized_path(source, out), out / 'a_labels.h5'


def test_withdrawn_label_csv_can_rebind_normalization_without_changing_annotations(tmp_path, monkeypatch):
    from autobehaver.datasets import _validate_label_pair
    normalize, source, annotation, features, labels = _normalized_pair(tmp_path, monkeypatch)
    with h5py.File(labels) as handle:
        datasets = {name: handle[name][:] for name in handle}
        old = json.loads(handle.attrs['content_identity'])
    annotation.unlink()
    normalize._NORM_MEANS = np.array([1, 2], dtype=np.float32)
    result = normalize.normalize_one_csv(str(source))
    assert result['status'] == 'success', result
    assert result['labels_rebound'] and result['labels_saved']
    with h5py.File(labels) as handle:
        new = json.loads(handle.attrs['content_identity'])
        assert new['label_source'] == old['label_source']
        assert new['feature_identity'] != old['feature_identity']
        for name, values in datasets.items():
            np.testing.assert_array_equal(handle[name][:], values)
        _validate_label_pair(features, handle)


def test_withdrawn_labels_changed_source_preserves_pair_and_stale_binding_is_rejected(tmp_path, monkeypatch):
    from autobehaver.datasets import _validate_label_pair
    normalize, source, annotation, features, labels = _normalized_pair(tmp_path, monkeypatch)
    old_hashes = [file_digest(path) for path in (features, labels)]
    annotation.unlink()
    data = pd.read_csv(source)
    data.iloc[4, 0] += 1
    data.to_csv(source, index=False)
    result = normalize.normalize_one_csv(str(source))
    assert result['status'] == 'failed' and 'restore the original annotation CSV' in result['error']
    assert [file_digest(path) for path in (features, labels)] == old_hashes
    normalize._NORM_LABELS_INPUT_DIR = normalize._NORM_LABELS_OUTPUT_DIR = None
    assert normalize.normalize_one_csv(str(source))['status'] == 'success'
    with h5py.File(labels) as handle:
        with pytest.raises(ValueError, match='feature'):
            _validate_label_pair(features, handle)


def test_explicit_missing_annotation_directory_fails_before_output_creation(tmp_path):
    from autobehaver.preinf.keyprenorm.config import KeypreNormConfig
    from autobehaver.preinf.keyprenorm.core import run_keyprenorm
    cfg = KeypreNormConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / 'out'), labels_dir=str(tmp_path / 'absent'))
    with pytest.raises(FileNotFoundError, match='Explicit labels directory'):
        run_keyprenorm(cfg=cfg, args=argparse.Namespace())
    assert not (tmp_path / 'out').exists()


def test_keypre_dry_run_has_no_writes_and_subset_ignores_unselected_missing_rows(tmp_path):
    from autobehaver.preinf.keypre.core import run_keypre
    from autobehaver.preinf.keypre.config import KeypreConfig
    from autobehaver.registry import upsert_registry_rows
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    selected = inputs / 'selectedDLC.csv'
    selected.write_text('fixture')
    (inputs / 'missingDLC.csv').write_text('unverified basename mirror')
    registry = tmp_path / 'registry.csv'
    upsert_registry_rows(csv_path=registry, rows=[
        {'recording_id': 'selected', 'dlc_csv_relpath': 'inputs/selectedDLC.csv'},
        {'recording_id': 'missing', 'dlc_csv_relpath': 'absent/missingDLC.csv'}])
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    cfg = KeypreConfig(input_dir=str(inputs), output_dir=str(tmp_path / 'out'), dry_run=True,
        registry_enabled=True, registry_csv=str(registry), file_filter='selected')
    run_keypre(cfg=cfg)
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert before == after and not (tmp_path / 'out').exists()
    with pytest.raises(FileNotFoundError, match='declared paths'):
        run_keypre(cfg=replace(cfg, file_filter='missing'))


def test_failed_keypre_input_is_not_registered(tmp_path, monkeypatch):
    from autobehaver.preinf.keypre import core, registry_update
    from autobehaver.preinf.keypre.config import KeypreConfig
    from autobehaver.registry import upsert_registry_rows
    source = tmp_path / 'aDLC.csv'
    source.write_text('fixture')
    registry = tmp_path / 'registry.csv'
    upsert_registry_rows(csv_path=registry, rows=[{'recording_id': 'a', 'dlc_csv_relpath': source.name}])
    def failed_workers(function, files, **kwargs):
        return iter({'status': 'failed', 'file': file, 'error': 'synthetic failure'} for file in files)
    seen = []
    monkeypatch.setattr(core, 'execute_workers', failed_workers)
    monkeypatch.setattr(registry_update, 'update_registry_from_keypre_outputs', lambda **kwargs: seen.extend(kwargs['dlc_csv_files']))
    cfg = KeypreConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / 'out'), clobber=True,
        registry_enabled=True, registry_csv=str(registry))
    with pytest.raises(RuntimeError):
        core.run_keypre(cfg=cfg)
    assert seen == []
    report = json.loads((tmp_path / 'out/KEYPRE_VALIDATION.json').read_text())
    assert report['status'] == 'failed'


def test_crop_candidate_validation_precedes_ambiguity_and_raw_results_do_not_certify(tmp_path, monkeypatch):
    from autobehaver.preinf.autocropper import pipeline
    movie = tmp_path / 'movie.mp4'
    movie.touch()
    expected = tmp_path / 'movie_chamber1.mp4'
    invalid = tmp_path / 'old_movie_chamber1.mp4'
    expected.write_bytes(b'valid')
    invalid.write_bytes(b'invalid')
    monkeypatch.setattr(pipeline, '_expected_crop_output_path', lambda *args: expected)
    monkeypatch.setattr(pipeline, '_crop_identity', lambda *args: {})
    monkeypatch.setattr(pipeline, '_checked_source_shape', lambda *args: (1, 1, 1))
    monkeypatch.setattr(pipeline, 'valid_crop', lambda source, path, *args, **kwargs: path == expected)
    assert pipeline._resolve_existing_crop(movie, tmp_path, 'chamber1')['path'] == expected
    assert expected.exists() and not invalid.exists()
    assert any(path.read_bytes() == b'invalid' for path in (tmp_path / '_old').glob('*.mp4'))
    monkeypatch.setattr(pipeline, '_resolve_existing_crop', lambda *args: {'status': 'missing'})
    result = pipeline._assess_required_crops(movie, tmp_path,
        required_chambers=['chamber1'], raw_cropped_files={'chamber1': expected.name})
    assert result['cropped_files'] == {} and result['missing_required_chambers'] == ['chamber1']


def test_worker_uses_frozen_parent_metadata(tmp_path, monkeypatch):
    from autobehaver.preinf.autocropper import pipeline
    monkeypatch.setattr(pipeline, 'set_runtime', lambda **kwargs: None)
    monkeypatch.setattr(pipeline, 'run_dir', tmp_path / 'run')
    monkeypatch.setattr(pipeline, 'crop_output_dir', tmp_path / 'crop')
    monkeypatch.setattr(pipeline, 'setup_logging', lambda *args, **kwargs: None)
    for name in ('exclusions_store', 'recording_db_path', 'recording_db_index'):
        monkeypatch.setattr(pipeline, name, None)
    monkeypatch.setattr(pipeline, '_load_exclusions_if_available', lambda: pytest.fail('worker reloaded mutable exclusions'))
    monkeypatch.setattr(pipeline, '_load_recording_db_if_needed', lambda: pytest.fail('worker reloaded mutable database'))
    snapshot = (object(), tmp_path / 'original.csv', object())
    pipeline._worker_init(None, SimpleNamespace(native_threads=1, log_level='WARNING'), snapshot)
    assert (pipeline.exclusions_store, pipeline.recording_db_path, pipeline.recording_db_index) == snapshot


def test_partial_cleanup_respects_recording_scope_active_lock_and_history(tmp_path):
    from autobehaver.file_locks import FileLock
    from autobehaver.preinf.autocropper.io_utils import crop_partial_token, delete_incomplete_files
    selected = tmp_path / f'.crop-{crop_partial_token("selected")}-one.partial'
    unrelated = tmp_path / f'.crop-{crop_partial_token("other")}-two.partial'
    selected.write_bytes(b'partial')
    unrelated.write_bytes(b'partial')
    history = tmp_path / '_old'
    history.mkdir()
    (history / selected.name).write_bytes(b'prior')
    with FileLock(selected.with_suffix('.lock'), timeout_sec=0):
        delete_incomplete_files(tmp_path, movie_stems=['selected'])
        assert selected.exists()
    delete_incomplete_files(tmp_path, movie_stems=['selected'])
    assert not selected.exists() and unrelated.exists() and (history / selected.name).exists()
