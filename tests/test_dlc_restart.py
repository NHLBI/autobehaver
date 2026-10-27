"""Exercise transferred inputs and native KeyPre-through-MMGen preparation."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from autobehaver.job_batches import build_plan, inventory_dlc_outputs
from autobehaver.batch_inputs import select_batch_inputs
from autobehaver.portable_paths import record_original_name
from autobehaver.preinf.autocropper.selection import write_selection

ROOT = Path(__file__).resolve().parents[1]


def coordinate_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows([
            ['scorer', 'model', 'model', 'model'],
            ['bodyparts', 'Head', 'Head', 'Head'],
            ['coords', 'x', 'y', 'likelihood'], [0, 10, 20, .99]])
    return path


@pytest.mark.parametrize('speed', ['fast', 'slow'])
def test_transferred_only_dlc_routes_every_recording_without_upstream_jobs(tmp_path, speed):
    source = tmp_path / 'project' / '3_dlc'
    names = [f'CONTROL_MALE_exp_trial1_rpi-{rpi}_movie{movie}_chamber1'
             for rpi in (1, 2) for movie in (0, 1)]
    inputs = [coordinate_csv(source / (name + 'DLCmodel.csv')) for name in names]
    plan, lists = build_plan(speed=speed, start_at='features', local_project=source.parent,
                             moviechecker_mode='recording-only')
    assert not (source.parent / '1_full_movies').exists()
    assert not (source.parent / '2_cropped_movies').exists()
    assert plan['chamber_movies'] == 4
    assert lists['all_chambers.txt'] == names
    stages = {job['stage'] for job in plan['jobs']}
    assert stages == {'features', 'posepred', 'mmgen', 'collect'}
    all_jobs = {job['key']: job for job in plan['jobs']}
    covered = []
    for job in plan['jobs']:
        if job['stage'] == 'features':
            assert job['depends_on'] == []
            selection = tmp_path / job['file_list']
            selection.write_text('\n'.join(lists[job['file_list']]))
            covered.extend(select_batch_inputs(inputs, selection))
        if job['stage'] == 'posepred':
            assert all(all_jobs[key]['stage'] == 'features' for key in job['depends_on'])
        if job['stage'] == 'mmgen':
            assert all(all_jobs[key]['stage'] == 'posepred' for key in job['depends_on'])
    assert sorted(covered) == sorted(inputs)


def test_legacy_subfolder_mapped_names_and_exclusions(tmp_path):
    source = tmp_path / '3_dlc' / 'DLC'
    path = coordinate_csv(source / 'short.csv')
    name = 'CONTROL_MALE_exp_trial1_rpi-1_movie0_chamber1'
    record_original_name(path, name + 'DLCmodel.csv')
    rejected = 'exp_trial1_rpi-1_movie0_chamber2'
    coordinate_csv(source / (rejected + 'DLCmodel.csv'))
    write_selection(source, {rejected: {'included': False}})
    assert inventory_dlc_outputs(tmp_path) == [name]


def test_prediction_csv_is_rejected_before_any_job_is_prepared(tmp_path):
    source = tmp_path / '3_dlc'; source.mkdir()
    (source / 'fly_positional_calcs_normalized_inference.csv').write_text(
        'Frame Number,Pose-Walking-Prob\n1,.9\n2,.8\n')
    with pytest.raises(ValueError, match='PosePred inference CSVs cannot be used'):
        build_plan(speed='fast', start_at='features', local_project=tmp_path)


def test_multiple_models_cannot_duplicate_a_recording(tmp_path):
    source = tmp_path / '3_dlc'
    for model in ('one', 'two'):
        coordinate_csv(source / ('CONTROL_MALE_exp_trial1_rpi-1_movie0_chamber1DLC' + model + '.csv'))
    with pytest.raises(ValueError, match='same recording'):
        inventory_dlc_outputs(tmp_path)


def test_incomplete_coordinate_triplet_is_rejected(tmp_path):
    path = coordinate_csv(tmp_path / '3_dlc' / 'flyDLCmodel.csv')
    path.write_text(path.read_text().replace('likelihood', 'invalid'))
    with pytest.raises(ValueError, match='coordinate columns'):
        inventory_dlc_outputs(tmp_path)


def test_gui_and_native_builder_support_dlc_start_and_record_parallelism(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    from autobehaver import desktop_jobs
    app = QApplication.instance() or QApplication([])
    spec = importlib.util.spec_from_file_location('desktop_dlc_fixture', ROOT / 'tests/test_desktop_jobs.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    config = fixture.setup_config(tmp_path)
    coordinate_csv(Path(config['dataset']) / '3_dlc' / 'CONTROL_MALE_exp_trial1_rpi-1_movie0_chamber1DLCmodel.csv')
    window = make_window(tmp_path / 'state')
    try:
        window.apply_values({'start_at': 'features'})
        assert window.fields['start_at'].currentData() == 'features'
        assert window.fields['feature_workers'].value() == 0
        assert not window.fields['crop_cpu'].isEnabled()
        assert not window.fields['dlc_cpu'].isEnabled()
        assert not window.fields['moviechecker_mode'].isEnabled()
        assert window.fields['features_cpu'].isEnabled()
        config.update(kind='pipeline', start_at='features', speed='fast', models=[],
                      moviechecker_mode='reviewed', feature_workers=2,
                      allocations={'features': [4, 32]})
        record = desktop_jobs.build_job(config, ROOT, tmp_path / 'work')
        assert set(record['job_counts']) == {'features', 'posepred', 'mmgen', 'collect'}
        assert record['config']['feature_workers'] == 2
        prepared = Path(record['prepared'])
        assert '--num_workers 10' not in (prepared / 'templates/features.sbatch').read_text()
        assert (prepared / 'templates/features.sbatch').read_text().count('--num_workers 2') == 2
        assert '#SBATCH --mem=32g' in (prepared / 'templates/features.sbatch').read_text()
        assert json.loads((prepared / 'plan.json').read_text())['start_at'] == 'features'
        assert not Path(record['mounted']).exists()
        window.set_record(record)
        assert 'worker ceiling: 2' in window.summary.toPlainText()
        app.processEvents()
    finally:
        window.close()


def test_automatic_workers_use_gui_allocation_and_migrate_old_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    from autobehaver import desktop_jobs
    app = QApplication.instance() or QApplication([])
    spec = importlib.util.spec_from_file_location('desktop_memory_fixture', ROOT/'tests/test_desktop_jobs.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    config = fixture.setup_config(tmp_path)
    coordinate_csv(Path(config['dataset'])/'3_dlc'/'CONTROL_MALE_exp_trial1_rpi-1_movie0_chamber1DLCmodel.csv')
    window = make_window(tmp_path/'state')
    try:
        assert window.fields['features_cpu'].value() == 8
        assert window.fields['features_ram'].value() == 32
        window.apply_values({'feature_workers': 2, 'features_cpu': 4, 'features_ram': 64})
        assert window.fields['feature_workers'].value() == 0
        assert window.fields['features_cpu'].value() == 8
        assert window.fields['features_ram'].value() == 64
        window.apply_values({'feature_execution_version': 1, 'feature_workers': 2, 'features_cpu': 4})
        assert window.fields['feature_workers'].value() == 2
        assert window.fields['features_cpu'].value() == 4
        assert window.values()['feature_execution_version'] == 1
        config.update(kind='pipeline', start_at='features', speed='fast', models=[],
                      moviechecker_mode='reviewed', feature_workers=0, allocations={'features':[8,12]})
        record = desktop_jobs.build_job(config, ROOT, tmp_path/'work')
        script = (Path(record['prepared'])/'templates/features.sbatch').read_text()
        assert script.count('--num_workers 0 --memory_gb 12') == 2
        assert '#SBATCH --cpus-per-task=8' in script and '#SBATCH --mem=12g' in script
        window.set_record(record)
        assert 'automatic CPU/RAM budgeting' in window.summary.toPlainText()
        assert not Path(record['mounted']).exists()
        app.processEvents()
    finally:
        window.close()
