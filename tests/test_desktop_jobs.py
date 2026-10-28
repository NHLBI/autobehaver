"""Exercise frozen preparation/publication and selection without connecting to HPC."""
import csv
import importlib.util
import json
from pathlib import Path
import shlex
import time

import pytest

from autobehaver import desktop_jobs as jobs

ROOT = Path(__file__).resolve().parents[1]


def setup_config(tmp_path):
    data = tmp_path / 'mounted data'
    project = data / 'experiment'
    project.mkdir(parents=True)
    bundles = data / '_job_bundles'
    bundles.mkdir()
    matrix = project / 'matrix.csv'
    with matrix.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['individual', 'group', 'sex', 'Kinematic_speed'])
        writer.writerows([['a', 'control', 'MALE', 1], ['b', 'mutant', 'MALE', 2],
                          ['c', 'rescue', 'FEMALE', 3], ['d', 'control', 'FEMALE', 4]])
    return dict(kind='featureimp', run_name='example', dataset=str(project), data_local=str(data),
                data_remote='/data/lab/experiments', jobs_local=str(bundles),
                jobs_remote='/data/lab/experiments/_job_bundles', python='/data/lab/env/bin/python',
                resources='/data/lab/resources', matrix=str(matrix), group_a='control', group_b='mutant',
                control_group='control', selected_identities=[['a', 'control', 'MALE'], ['b', 'mutant', 'MALE'],
                                                           ['c', 'rescue', 'FEMALE']], sex='all', pca=False, fsea=False)


def test_featureimp_preserves_selection_freezes_and_publishes(tmp_path):
    config = setup_config(tmp_path)
    original = Path(config['matrix']).read_bytes()
    record = jobs.build_job(config, ROOT, tmp_path / 'local')
    assert not Path(record['mounted']).exists()  # Build cannot publish or submit.
    plan = json.loads((Path(record['prepared']) / 'plan.json').read_text())
    assert plan['group_counts_after_sex_filter'] == {'control': 1, 'mutant': 1, 'rescue': 1}
    assert [r['individual'] for r in jobs.matrix_inventory(Path(record['prepared']) / 'featureimp_matrix.csv')] == ['a', 'b', 'c']
    assert Path(config['matrix']).read_bytes() == original
    # Generated arguments must be accepted by the maintained science CLI parser.
    from autobehaver.postinf.featureimp.config import build_arg_parser
    # Resource-monitor arguments are attached by cli.py, not the core parser.
    command = plan['command']; cli_args = command[3:command.index('--resource-monitor')]
    parsed = build_arg_parser().parse_args(cli_args)
    assert parsed.training_group_a == 'control' and parsed.training_group_b == 'mutant'
    assert not parsed.run_pca_analysis and not parsed.run_fsea_analysis
    published = jobs.publish(record)
    assert published['published']
    assert not (Path(record['mounted']) / 'jobs.tsv').exists()
    assert jobs.verify_submission(published) == 'bash ' + shlex.quote(record['remote'] + '/submit.sh')
    (Path(record['mounted']) / 'jobs.tsv').write_text('featureimp\t123\n')
    with pytest.raises(ValueError, match='accepted jobs'):
        jobs.verify_submission(published)
    with pytest.raises(FileExistsError):
        jobs.publish(record)


def test_changed_prepared_code_cannot_be_sent(tmp_path):
    record = jobs.build_job(setup_config(tmp_path), ROOT, tmp_path / 'local')
    (Path(record['prepared']) / 'submit.sh').write_text('changed')
    with pytest.raises(ValueError, match='changed after Build'):
        jobs.publish(record)
    assert not Path(record['mounted']).exists()


def test_invalid_training_selection_and_ssh_inputs(tmp_path):
    config = setup_config(tmp_path)
    config['sex'] = 'female'
    with pytest.raises(ValueError, match='training groups'):
        jobs.build_job(config, ROOT, tmp_path / 'local')
    with pytest.raises(ValueError): jobs.ssh_arguments('-oProxyCommand=bad', 'user')
    with pytest.raises(ValueError): jobs.ssh_arguments('hpc.example.org', 'user;bad')
    with pytest.raises(ValueError): jobs.linux_path('/data/../etc')


def test_pipeline_uses_native_schedule_and_allocations(tmp_path):
    config = setup_config(tmp_path)
    fixture_spec = importlib.util.spec_from_file_location('desktop_fixture', ROOT / 'tests/test_job_batching.py')
    fixture = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture)
    fixture.recording_fixture(Path(config['dataset']))
    config.update(kind='pipeline', start_at='crop', speed='fast', moviechecker_mode='reviewed',
                  models=[], batch_sizes=dict(crop=4, dlc=4, features=20, posepred=20, mmgen=40),
                  allocations={'mmgen': [12, 32], 'mmgen_finish': [8, 80], 'posepred': [10, 64]},
                  walltime_hours={'mmgen': 6, 'mmgen_finish': 48})
    record = jobs.build_job(config, ROOT, tmp_path / 'local')
    plan = json.loads((Path(record['prepared']) / 'plan.json').read_text())
    assert plan['usable_original_movies'] == 6 and plan['chamber_movies'] == 36
    assert plan['batch_sizes'] == config['batch_sizes']
    assert '#SBATCH --mem=32g' in (Path(record['prepared']) / 'templates/mmgen.sbatch').read_text()
    assert '#SBATCH --time=6:00:00' in (Path(record['prepared']) / 'templates/mmgen.sbatch').read_text()
    finish = (Path(record['prepared']) / 'templates/mmgen_finish.sbatch').read_text()
    assert '#SBATCH --mem=80g' in finish and '#SBATCH --cpus-per-task=8' in finish
    assert '#SBATCH --time=48:00:00' in finish
    posepred = (Path(record['prepared']) / 'templates/posepred.sbatch').read_text()
    assert '#SBATCH --mem=64g' in posepred and '#SBATCH --cpus-per-task=10' in posepred
    assert '--memory_gb 64' in posepred and '--batch 0' in posepred and '--num_workers -1' in posepred
    assert not Path(record['mounted']).exists()


def test_desktop_window_loads_without_remote_actions(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    app = QApplication.instance() or QApplication([])
    window = make_window(tmp_path / 'state')
    assert window.tabs.count() == 6
    assert window.build_button.isEnabled() and not window.send_button.isEnabled()
    assert not window.submit_button.isEnabled()
    window.fields['username'].setText('example')
    window.save_settings()
    window.close()
    restored = make_window(tmp_path / 'state')
    assert restored.fields['username'].text() == 'example'
    restored.tabs.setCurrentIndex(3); app.processEvents()
    assert not restored.build_button.isEnabled()
    restored.close()


def test_desktop_buttons_build_send_and_submit_with_fake_ssh(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    app = QApplication.instance() or QApplication([])
    window = make_window(tmp_path / 'state')
    config = setup_config(tmp_path)
    window.apply_values(dict(config, fi_sharded=False, fi_override=True, fi_dataset=config['dataset'], fi_name='button_test',
                             username='tester', host='hpc.example.org'))
    failures = []
    monkeypatch.setattr(window, 'error', failures.append)
    def wait():
        deadline = time.monotonic() + 60
        while window.busy and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.01)
        assert not window.busy
        assert not failures
    window.tabs.setCurrentIndex(1)
    window.load_matrix(); wait()
    assert window.table.rowCount() == 4
    window.apply_values(config)
    window.build(); wait()
    assert window.record['kind'] == 'featureimp'
    assert window.send_button.isEnabled() and not window.submit_button.isEnabled()
    window.send(); wait()
    assert not window.submit_button.isEnabled()  # Must connect before submitting.
    commands = []
    class FakeSSH:
        connected = True
        host = 'hpc.example.org'
        username = 'tester'
        def execute(self, command, output):
            (Path(window.record['mounted']) / 'jobs.tsv').write_text('featureimp\t123\n')
            output('mesg: ttyname failed: Inappropriate ioctl for device\n')
            commands.append(command); output('Accepted example job\n'); return 0
        def close(self): pass
    window.session = FakeSSH()
    window.update_buttons()
    assert window.submit_button.isEnabled()
    window.submit_button.click(); wait()
    assert commands == ['bash ' + shlex.quote(window.record['remote'] + '/submit.sh')]
    assert window.submit_button.text() == '3  Submitted'
    assert not window.submit_button.isEnabled()
    app.processEvents()
    assert 'mesg:' not in window.output.toPlainText()
    assert 'Accepted example job' in window.output.toPlainText()
    assert 'mesg:' in (Path(window.record['root']) / 'submission_output.log').read_text()
    window.save_settings()
    window.close()
    restored = make_window(tmp_path / 'state')
    restored.session = FakeSSH()
    restored.update_buttons()
    assert restored.submit_button.text() == '3  Submitted'
    assert not restored.submit_button.isEnabled()
    restored.close()


def test_submission_button_uses_recorded_acceptance(tmp_path):
    record = dict(published=True, mounted=str(tmp_path), job_counts={'features': 2})
    assert jobs.submission_button_state(record) == ('3  Submit job', True)
    path = tmp_path / 'jobs.tsv'
    path.write_text('features_000\t123\nfeatures_000\t124\n')
    assert jobs.submission_button_state(record) == ('3  Partially submitted', False)
    path.write_text('features_000\t123\nfeatures_001\t124\n')
    assert jobs.submission_button_state(record) == ('3  Submitted', False)
    path.write_text('')
    assert jobs.submission_button_state(record) == ('3  Check submission', False)
    path.write_bytes(b'\xff')
    assert jobs.submission_button_state(record) == ('3  Check submission', False)
    assert jobs.submission_button_state(None) == ('3  Submit job', False)


def test_small_screen_keeps_primary_actions_visible(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPoint
    from autobehaver.desktop import make_window
    app = QApplication.instance() or QApplication([])
    window = make_window(tmp_path / 'state')
    window.resize(640, 480); window.show(); app.processEvents()
    assert window.width() <= 640 and window.height() <= 480
    for tab in range(window.tabs.count()):
        window.tabs.setCurrentIndex(tab); app.processEvents()
        for button in (window.build_button, window.send_button, window.submit_button, window.connect_button):
            corner = button.mapTo(window, QPoint(button.width(), button.height()))
            assert button.isVisible() and window.rect().contains(corner)
    window.close()


def test_login_dialog_answers_worker_without_saving_password(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication, QDialog, QLineEdit
    from PySide6.QtCore import QTimer
    from autobehaver.desktop import make_window
    app = QApplication.instance() or QApplication([])
    window = make_window(tmp_path / 'state')
    errors = []; monkeypatch.setattr(window, 'error', errors.append)
    class FakeSession:
        connected = False
        def connect(self, host, username, prompt):
            answer = prompt('authentication', {'title': 'Test login', 'instructions': 'Local test',
                                                'prompts': [('Password', False)]})
            assert answer == ['local-test-password']
            self.connected = True
            return 'Connected'
        def close(self): self.connected = False
    window.session = FakeSession()
    timer = QTimer()
    def answer_dialog():
        for dialog in app.topLevelWidgets():
            if isinstance(dialog, QDialog) and dialog.windowTitle() == 'Test login':
                field = dialog.findChild(QLineEdit)
                assert field.echoMode() == QLineEdit.Password
                field.setText('local-test-password'); dialog.accept(); timer.stop()
    timer.timeout.connect(answer_dialog); timer.start(20)
    window.connect()
    deadline = time.monotonic() + 10
    while window.busy and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.01)
    timer.stop()
    assert not window.busy and window.session.connected and not errors
    assert 'local-test-password' not in (tmp_path / 'state/settings.json').read_text()
    window.close()
