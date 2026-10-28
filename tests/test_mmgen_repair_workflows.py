"""Frozen append retry and complete-cohort matrix-first workflow evidence."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import atomic_write_json, sha256_file
from autobehaver.postinf.mmgen import append_mode as app, cohort, core, pipeline, publication as pub
from autobehaver.postinf.mmgen.hdf_store import get_done_source_files
from autobehaver.postinf.mmgen.input_contract import required_keypre_columns
from autobehaver.postinf.mmgen.integrity import read_store_identity
from autobehaver.postinf.mmgen.io import read_metadata_csv
from autobehaver.postinf.mmgen.locks import canonical_lock_held
from autobehaver.postinf.mmgen.matrix_parts import merge_matrix_part_files
from tests.test_mmgen_repair_contracts import append_case, recording_name, seal_generation, write_pair


def write_native_pair(inf, pos, name, *, speed=1., count=40):
    values = {column: np.ones(count) for column in required_keypre_columns(
        include_postural=False, include_spatial=False, include_measurements=False)}
    values['speed_inst'] = np.full(count, speed)
    pd.DataFrame(values).to_csv(pos / name.replace('_inference', ''), index=False)
    # Observed flanks make the Standing bout complete under the boundary policy.
    labels = ['Walking'] * 5 + ['Standing'] * (count - 10) + ['Walking'] * 5
    pd.DataFrame({'Frame Number': np.arange(1, count + 1), 'Smoothed Pose Label': labels,
        'Smoothed Orient Label': ['Floor'] * count}).to_csv(inf / name, index=False)


def native_config(tmp_path):
    inf, pos = tmp_path / 'i', tmp_path / 'p'
    inf.mkdir(); pos.mkdir()
    return core.MMGenConfig(inference_csv_dir=str(inf), poscalc_dir=str(pos), output_dir=str(tmp_path / 'o'),
        inf_tag='_inference', num_workers=1, input_batch_size=1, edge_skip=0, video_fps=1., mm_per_px=1.,
        keep_temp_files=True, emit_full_outputs=False, make_measurements=False, calculate_occupancy=False,
        calculate_postural_features=False, make_detailed_transition_outputs=False,
        filter_features_by_bout_count=False)


@pytest.mark.parametrize('requested', [1, 8])
def test_explicit_saved_single_job_plan_executes_without_creating_a_second_plan(append_case, monkeypatch, requested):
    inf, pos, root, cfg, first = append_case
    second = recording_name(movie=1)
    write_pair(inf, pos, second)
    cfg = replace(cfg, append_batch_id='planned', num_shards=requested, shard_by='row_identity')
    plan, selected, _, batch = app._create_append_plan(cfg)
    assert plan['processing']['active_num_shards'] == 1
    calls = []
    def process(*args, **kwargs):
        assert canonical_lock_held(batch) and not canonical_lock_held(root)
        assert kwargs['selected_files'] == selected.source_file.tolist()
        assert kwargs['num_shards'] == 1 and not kwargs['process_only']
        calls.append(kwargs)
        seal_generation(batch, cfg, [first, second], value=9.)
    monkeypatch.setattr(pipeline, 'run_post_inference_analysis', process)
    write_pair(inf, pos, recording_name(trial=9), paired=False)
    core.run_mmgen(cfg)
    assert len(calls) == 1
    assert set(pub._verified_generation(root)['identity']['sources']) == {first, second}
    with pytest.raises(RuntimeError, match='Canonical generation changed'):
        core.run_mmgen(cfg)


def test_native_single_job_plan_resumes_commits_finalizes_once_and_retries_publication(tmp_path, monkeypatch):
    cfg = native_config(tmp_path)
    inf, pos, root = Path(cfg.inference_csv_dir), Path(cfg.poscalc_dir), Path(cfg.output_dir)
    first, second = recording_name(), recording_name(movie=1)
    write_native_pair(inf, pos, first)
    core.run_mmgen(cfg)
    old_receipt = sha256_file(root / pub.OUTPUT_RECEIPT_FILENAME)
    write_native_pair(inf, pos, second, speed=3.)
    cfg = replace(cfg, append_new_data=True, append_batch_id='retry', num_shards=4, shard_by='row_identity')
    plan, selected, _, batch = app._create_append_plan(cfg)
    assert plan['processing']['active_num_shards'] == 1
    write_pair(inf, pos, recording_name(trial=9), paired=False)
    worker_results = pipeline.ordered_worker_results
    def interrupt_after_one(*args, **kwargs):
        iterator = iter(worker_results(*args, **kwargs))
        yield next(iterator)
        raise RuntimeError('injected interruption after one committed recording')
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'ordered_worker_results', interrupt_after_one)
        with pytest.raises(RuntimeError, match='one committed recording'):
            core.run_mmgen(replace(cfg, process_only=True, resume=True))
    store = batch / 'temp_store.h5'
    assert len(get_done_source_files(str(store))) == 1
    seen = []
    def pending_only(worker, args_list, **kwargs):
        seen.extend(Path(args[0]).name for args in args_list)
        return worker_results(worker, args_list, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'ordered_worker_results', pending_only)
        core.run_mmgen(replace(cfg, process_only=True, resume=True))
    assert seen == [second]
    assert get_done_source_files(str(store)) == {first, second}
    assert sha256_file(root / pub.OUTPUT_RECEIPT_FILENAME) == old_receipt

    scans = []
    validate = pipeline.validate_commits
    def scan(*args, **kwargs):
        scans.append(args[0])
        return validate(*args, **kwargs)
    replace_file = pub.os.replace
    def fail_root_receipt(source, target):
        if Path(target) == root / pub.OUTPUT_RECEIPT_FILENAME and Path(source).name.endswith('.publish'):
            raise OSError('injected canonical publication failure')
        return replace_file(source, target)
    finalize = replace(cfg, merge_shards=True)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'validate_commits', scan)
        patch.setattr(pub.os, 'replace', fail_root_receipt)
        with pytest.raises(OSError, match='canonical publication failure'):
            core.run_mmgen(finalize)
    assert len(scans) == 1
    assert sha256_file(root / pub.OUTPUT_RECEIPT_FILENAME) == old_receipt
    delta_receipt = sha256_file(batch / pub.OUTPUT_RECEIPT_FILENAME)
    def must_not_repeat(*args, **kwargs):
        raise AssertionError('A verified completed delta must be reused without reprocessing or refinalizing.')
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'run_post_inference_analysis', must_not_repeat)
        core.run_mmgen(finalize)
    assert sha256_file(batch / pub.OUTPUT_RECEIPT_FILENAME) == delta_receipt
    current = pub._verified_generation(root)
    assert set(current['identity']['sources']) == {first, second}
    result = read_metadata_csv(root / 'MASTER_FEATURE_MATRIX.csv')
    assert result.loc[0, 'KinematicBB_Pose+Orient_Standing+Floor_speed_mean_mean'] == 2.
    with pytest.raises(RuntimeError, match='Canonical generation changed'):
        core.run_mmgen(finalize)


def test_finalize_only_initial_scan_still_rejects_corrupted_scientific_table(tmp_path, monkeypatch):
    cfg = native_config(tmp_path)
    name = recording_name()
    write_native_pair(Path(cfg.inference_csv_dir), Path(cfg.poscalc_dir), name)
    core.run_mmgen(replace(cfg, process_only=True))
    store_path = Path(cfg.output_dir) / 'temp_store.h5'
    with pd.HDFStore(store_path, 'a') as store:
        frame = store['bout_summary_tmp']
        frame['n_bouts'] = 1000.
        store.put('bout_summary_tmp', frame, format='table', data_columns=['individual', 'source_file'])
    called = []
    validate = pipeline.validate_commits
    def scan(*args, **kwargs):
        called.append(args[0])
        return validate(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'validate_commits', scan)
    with pytest.raises(RuntimeError, match='(?i)(fingerprint|content|commit)'):
        core.run_mmgen(replace(cfg, finalize_only=True))
    assert len(called) == 1
    assert not (Path(cfg.output_dir) / 'MASTER_FEATURE_MATRIX.csv').exists()


def test_cohort_change_is_rejected_before_hashing_or_processing_sources(tmp_path, monkeypatch):
    cfg = native_config(tmp_path)
    inf, pos = Path(cfg.inference_csv_dir), Path(cfg.poscalc_dir)
    first, second = recording_name(), recording_name(trial=2)
    for name in (first, second):
        write_pair(inf, pos, name)
    root = Path(cfg.output_dir)
    saved = cohort.bind_shard_cohort(root / '_mmgen_shards', [first], inference_csv_dir=inf,
                                    num_shards=2, shard_by='row_identity')
    def forbidden(*args, **kwargs):
        raise AssertionError('Changed full cohort must fail before source-content hashing.')
    monkeypatch.setattr(pipeline, 'source_identities', forbidden)
    with pytest.raises(RuntimeError, match='Selected shard cohort changed'):
        pipeline.run_post_inference_analysis(str(inf), str(pos), str(root), 5, 0, '_inference',
            runtime_settings=cfg.runtime_settings, num_shards=2, shard_index=1, shard_by='row_identity',
            process_only=True, selected_files=[first, second])
    assert cohort.read_shard_cohort(root / '_mmgen_shards', 2) == saved
    assert not list(root.glob('_mmgen_shards/*/temp_store.h5'))


def test_native_matrix_first_cohort_includes_empty_shard_and_matches_single_job(tmp_path):
    cfg = native_config(tmp_path)
    inf, pos = Path(cfg.inference_csv_dir), Path(cfg.poscalc_dir)
    names = [recording_name(trial=trial, movie=movie) for trial in (1, 2) for movie in (0, 1)]
    for i, name in enumerate(names):
        write_native_pair(inf, pos, name, speed=float(i + 1))
    core.run_mmgen(cfg)
    reference = read_metadata_csv(Path(cfg.output_dir) / 'MASTER_FEATURE_MATRIX.csv')
    root = tmp_path / 'parts'
    for index in range(3):
        core.run_mmgen(replace(cfg, output_dir=str(root), num_shards=3, shard_index=index,
            shard_by='row_identity', matrix_first_shards=True))
    core.run_mmgen(replace(cfg, output_dir=str(root), num_shards=3, merge_matrix_parts=True))
    result = read_metadata_csv(root / 'MASTER_FEATURE_MATRIX.csv')
    pd.testing.assert_frame_equal(reference, result, check_dtype=False, atol=1e-12, rtol=1e-12)
    inventory = cohort.read_shard_cohort(root / '_mmgen_shards', 3)
    assert {row['source_file'] for row in inventory['files']} == set(names)
    markers = list(root.glob('_mmgen_shards/*/_mmgen_work/SHARD*DONE.json'))
    assert len(markers) == 6
    for marker in markers:
        assert json.loads(marker.read_text())['cohort_sha256'] == inventory['cohort_sha256']
    empty = json.loads((root / '_mmgen_shards/shard_002_of_003/_mmgen_work/SHARD_MATRIX_DONE.json').read_text())
    assert empty['n_selected_files'] == empty['n_matrix_rows'] == 0
    matrix_hash = sha256_file(root / 'MASTER_FEATURE_MATRIX.csv')
    bad_marker = root / '_mmgen_shards/shard_001_of_003/_mmgen_work/SHARD_MATRIX_DONE.json'
    bad = json.loads(bad_marker.read_text())
    bad['cohort_sha256'] = 'a different complete selection'
    atomic_write_json(bad, bad_marker)
    with pytest.raises(RuntimeError, match='cohort proof'):
        merge_matrix_part_files(output_dir=str(root), num_shards=3)
    assert sha256_file(root / 'MASTER_FEATURE_MATRIX.csv') == matrix_hash
    cohort.cohort_path(root / '_mmgen_shards', 3).unlink()
    with pytest.raises(RuntimeError, match='complete frozen cohort inventory'):
        merge_matrix_part_files(output_dir=str(root), num_shards=3)
    assert sha256_file(root / 'MASTER_FEATURE_MATRIX.csv') == matrix_hash
