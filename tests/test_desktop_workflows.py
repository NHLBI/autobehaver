import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from autobehaver.job_batches import build_plan
from autobehaver import desktop_jobs as jobs
from autobehaver.desktop_recovery import recovery_config

ROOT = Path(__file__).resolve().parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def crops(root, count):
    directory = root / '2_cropped_movies'; directory.mkdir(parents=True)
    names = [f'CONTROL_MALE_trial1_rpi-1_movie0_chamber{i}' for i in range(count)]
    for name in names: (directory / (name + '.mp4')).touch()
    return names


def committed(root, names):
    output = root / '8_master_matrix'; output.mkdir()
    ledger = output / 'MMGEN_SOURCE_LEDGER.csv'
    ledger.write_text('source_file,row_identity\n' + ''.join(
        f'{name}_positional_calcs_normalized_inference.csv,{name}\n' for name in names))
    (output / 'MASTER_FEATURE_MATRIX.csv').write_text('individual,group,sex\nold,CONTROL,MALE\n')
    (output / '_mmgen_work').mkdir()
    (output / '_mmgen_work/MMGEN_CALCULATION_IDENTITY.json').write_text('{}')
    (output / 'MMGEN_OUTPUT_RECEIPT.json').write_text(json.dumps({'artifacts': {
        ledger.name: hashlib.sha256(ledger.read_bytes()).hexdigest()}}))


def test_shared_batching_balances_inputs_and_caps_cpu_gpu(tmp_path):
    names = crops(tmp_path, 1030)
    plan, lists = build_plan(speed='fast', start_at='dlc', local_project=tmp_path,
        batching=dict(full_movies=2, chambers=20, cpu_jobs=7, gpu_jobs=50))
    for stage, cap in [('dlc', 50), ('features', 7), ('posepred', 50), ('mmgen', 7)]:
        stage_jobs = [j for j in plan['jobs'] if j['stage'] == stage]
        assert len(stage_jobs) == cap
        if stage != 'mmgen':
            selected = [lists[j['file_list']] for j in stage_jobs]
            assert sorted(x for batch in selected for x in batch) == sorted(names)
            assert max(map(len, selected)) - min(map(len, selected)) <= 1
            assert min(map(len, selected)) >= 20
    small = tmp_path / 'small'; crops(small, 3)
    small_plan, _ = build_plan(speed='fast', start_at='dlc', local_project=small,
        batching=dict(full_movies=2, chambers=20, cpu_jobs=7, gpu_jobs=50))
    assert sum(j['stage'] == 'dlc' for j in small_plan['jobs']) == 1


def test_new_only_selects_uncommitted_and_uses_native_append(tmp_path):
    project = tmp_path / 'project'; names = crops(project, 12); committed(project, names[:6])
    native = helper('test_job_batching').prepare
    target = native.prepare_jobs(output_dir=tmp_path / 'bundle', remote_job_dir='/data/jobs/example',
        project='/data/project', package='/data/package', python='/data/python', resources='/data/resources',
        local_project=project, start_at='dlc', new_only=True,
        batching=dict(full_movies=2, chambers=2, cpu_jobs=2, gpu_jobs=3))
    plan = json.loads((target / 'plan.json').read_text())
    assert plan['incremental']['previous_recordings'] == 6
    assert plan['incremental']['new_recordings'] == 6
    assert (target / 'batches/all_chambers.txt').read_text().splitlines() == sorted(names[6:])
    assert sum(j['stage'] == 'mmgen' for j in plan['jobs']) == 1
    assert not any(j['stage'] == 'mmgen_finish' for j in plan['jobs'])
    assert (target / 'append_batch_id.txt').read_text().strip() == 'example'
    assert '--append_new_data' in (target / 'templates/mmgen.sbatch').read_text()
    assert (target / 'continuation_supported.txt').is_file()
    # Desktop preparation did not change the committed matrix/ledger.
    assert (project / '8_master_matrix/MASTER_FEATURE_MATRIX.csv').read_text().endswith('old,CONTROL,MALE\n')
    (project / '8_master_matrix/MMGEN_SOURCE_LEDGER.csv').write_text('changed')
    with pytest.raises(ValueError, match='does not match'):
        build_plan(speed='fast', start_at='dlc', local_project=project, new_only=True)


def test_new_only_requires_baseline_and_refuses_empty_delta(tmp_path):
    names = crops(tmp_path, 2)
    with pytest.raises(ValueError, match='normal run first'):
        build_plan(speed='fast', start_at='dlc', local_project=tmp_path, new_only=True)
    committed(tmp_path, names)
    with pytest.raises(ValueError, match='No new recordings'):
        build_plan(speed='fast', start_at='dlc', local_project=tmp_path, new_only=True)
    with pytest.raises(ValueError, match='requires Fast'):
        build_plan(speed='slow', start_at='dlc', local_project=tmp_path, new_only=True)


def test_new_full_movies_skip_committed_movies_and_reject_mixed_chambers(tmp_path):
    h = helper('test_job_batching')
    from autobehaver.job_batches import inventory_recordings
    project = h.recording_fixture(tmp_path/'project')
    movies, _, by_movie, _ = inventory_recordings(project, 'crop')
    committed(project, by_movie[movies[0]] + by_movie[movies[1]])
    plan, lists = build_plan(speed='fast', start_at='crop', local_project=project, new_only=True)
    selected = [name for job in plan['jobs'] if job['stage'] == 'crop' for name in lists[job['file_list']]]
    assert selected == movies[2:]
    assert plan['chamber_movies'] == 24
    mixed = h.recording_fixture(tmp_path/'mixed')
    committed(mixed, by_movie[movies[0]][:1])
    with pytest.raises(ValueError, match='both committed and new chambers'):
        build_plan(speed='fast', start_at='crop', local_project=mixed, new_only=True)


def test_new_dlc_inputs_need_no_original_movies(tmp_path):
    h = helper('test_dlc_restart')
    names = [f'CONTROL_MALE_trial1_rpi-1_movie0_chamber{i}' for i in (1, 2)]
    for name in names:
        h.coordinate_csv(tmp_path/'3_dlc'/(name+'DLCmodel.csv'))
    committed(tmp_path, names[:1])
    plan, lists = build_plan(speed='fast', start_at='features', local_project=tmp_path, new_only=True)
    assert lists['all_chambers.txt'] == names[1:]
    assert {j['stage'] for j in plan['jobs']} == {'features','posepred','mmgen','collect'}


def test_featureimp_recovery_uses_frozen_matrix_and_preserves_selection(tmp_path):
    h = helper('test_desktop_jobs')
    config = h.setup_config(tmp_path)
    parent = jobs.build_job(config, ROOT, tmp_path / 'local')
    snapshot = dict(source='Live scheduler status', updated='now', rows=[
        dict(key='featureimp', job_id='101', state='FAILED')])
    recovery = recovery_config(parent, snapshot, {'featureimp': [8, 64]})
    assert recovery['matrix'] == str(Path(parent['prepared']) / 'featureimp_matrix.csv')
    assert recovery['group_a'] == config['group_a'] and recovery['group_b'] == config['group_b']
    assert recovery['fi_cpus'] == 8 and recovery['fi_memory'] == 64
    assert recovery['run_name'] != config['run_name']
    # User's original matrix is no longer needed for the retry.
    Path(config['matrix']).unlink()
    child = jobs.build_job(recovery, ROOT, tmp_path / 'local')
    assert jobs.digest(Path(parent['prepared']) / 'featureimp_matrix.csv') == jobs.digest(Path(child['prepared']) / 'featureimp_matrix.csv')
    script = (Path(child['prepared']) / 'submit.sh').read_text()
    assert 'sacct' in script and 'recovery_claims.tsv' in script and 'flock -n 8' in script
    assert 'scancel' not in script
    for state in ('RUNNING', 'COMPLETED', 'Unknown'):
        snapshot['rows'][0]['state'] = state
        with pytest.raises(ValueError, match='only failed'):
            recovery_config(parent, snapshot)


def test_gui_shared_controls_migrate_and_resume_is_separate(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    app = QApplication.instance() or QApplication([])
    window = make_window(tmp_path / 'state')
    window.apply_values({'crop_batch': 4, 'dlc_batch': 30, 'dlc_ram': 10, 'feature_workers': 2})
    assert window.values()['full_movies'] == 4 and window.values()['chambers'] == 30
    assert 'dlc_batch' not in window.fields and window.values()['dlc_ram'] == 10
    assert not window.resume_button.isEnabled()
    assert 'new_only' in window.fields
    window.close()


def test_partial_resume_requires_new_bundle_support(tmp_path):
    record = dict(kind='pipeline', published=True, mounted=str(tmp_path), prepared=str(tmp_path), job_counts={'features': 2})
    (tmp_path / 'jobs.tsv').write_text('features_000\t101\n')
    assert not jobs.can_resume_submission(record)
    (tmp_path / 'continuation_supported.txt').write_text('1\n')
    assert jobs.can_resume_submission(record)
    assert jobs.submission_button_state(record) == ('3  Partially submitted', False)
    (tmp_path / 'jobs.tsv').write_text('features_000\t101\nfeatures_001\t102\n')
    assert not jobs.can_resume_submission(record)
