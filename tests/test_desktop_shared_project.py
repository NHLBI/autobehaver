"""Check automatic project inputs and stale asynchronous matrix results."""
import json
from pathlib import Path
import threading
import time

import pytest


@pytest.fixture
def gui(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    app = QApplication.instance() or QApplication([])
    window = make_window(tmp_path / 'state')
    errors = []; monkeypatch.setattr(window, 'error', errors.append)
    yield app, window, errors
    deadline = time.monotonic() + 5
    while window.busy and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.01)
    window.close()


def matrix(project, group='control'):
    path = project / '8_master_matrix/MASTER_FEATURE_MATRIX.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'individual,group,sex,feature\na,{group},MALE,1\nb,mutant,FEMALE,2\n')
    return path


def until(app, condition):
    end = time.monotonic() + 10
    while not condition() and time.monotonic() < end:
        app.processEvents(); time.sleep(.01)
    assert condition()


def test_shared_project_automatically_loads_and_keeps_comparison_explicit(gui, tmp_path):
    app, window, errors = gui
    project = tmp_path / 'project'; path = matrix(project)
    window.fields['dataset'].setText(str(project))
    window.tabs.setCurrentIndex(1)
    until(app, lambda: window.loaded_matrix == str(path) and not window.busy)
    assert window.fields['fi_dataset'].text() == str(project)
    assert window.fields['matrix'].text() == str(path)
    assert window.fi_output_note.text() == str(project / '9_feature_importance')
    assert window.table.rowCount() == 2
    assert window.fields['group_a'].currentData() is None
    assert window.fields['group_b'].currentData() is None
    assert not errors


def test_missing_matrix_waits_then_is_found_without_reentering_path(gui, tmp_path):
    app, window, errors = gui
    project = tmp_path / 'waiting'
    window.fields['dataset'].setText(str(project)); window.tabs.setCurrentIndex(1)
    window.auto_load_featureimp()
    assert 'Waiting for MMGen' in window.fi_matrix_status.text()
    assert window.loaded_matrix is None and not errors
    path = matrix(project); window.auto_load_featureimp()
    until(app, lambda: window.loaded_matrix == str(path) and not window.busy)


def test_switch_project_clears_selection_and_discards_old_inflight_read(gui, tmp_path, monkeypatch):
    from autobehaver import desktop_jobs
    app, window, errors = gui
    first = tmp_path / 'first'; first_path = matrix(first, 'first_group')
    second = tmp_path / 'second'; second_path = matrix(second, 'second_group')
    entered, release = threading.Event(), threading.Event()
    native = desktop_jobs.matrix_inventory
    def delayed(path):
        if Path(path) == first_path:
            entered.set(); assert release.wait(5)
        return native(path)
    monkeypatch.setattr(desktop_jobs, 'matrix_inventory', delayed)
    window.fields['dataset'].setText(str(first)); window.tabs.setCurrentIndex(1)
    window.auto_load_featureimp()
    until(app, entered.is_set)
    window.fields['dataset'].setText(str(second))
    assert window.loaded_matrix is None and window.table.rowCount() == 0
    release.set()
    until(app, lambda: window.loaded_matrix == str(second_path) and not window.busy)
    assert {row['group'] for row in window.matrix_rows} == {'second_group', 'mutant'}
    assert window.fields['group_a'].currentData() is None
    assert not errors


def test_custom_override_survives_shared_project_change_and_can_be_disabled(gui, tmp_path):
    app, window, errors = gui
    main = tmp_path / 'main'; main_path = matrix(main)
    other = tmp_path / 'custom'; other_path = matrix(other, 'custom_group')
    window.apply_values(dict(dataset=str(main), fi_override=True,
                             fi_dataset=str(other), matrix=str(other_path)))
    window.tabs.setCurrentIndex(1); window.auto_load_featureimp()
    until(app, lambda: window.loaded_matrix == str(other_path) and not window.busy)
    window.fields['dataset'].setText(str(tmp_path / 'different'))
    assert window.loaded_matrix == str(other_path)
    window.apply_values(dict(dataset=str(main), fi_override=False))
    until(app, lambda: window.loaded_matrix == str(main_path) and not window.busy)
    assert not errors


def test_changed_matrix_blocks_build_before_job_preparation(gui, tmp_path, monkeypatch):
    from autobehaver import desktop_jobs
    app, window, errors = gui
    project = tmp_path / 'project'; path = matrix(project)
    window.fields['dataset'].setText(str(project)); window.tabs.setCurrentIndex(1)
    window.auto_load_featureimp()
    until(app, lambda: window.loaded_matrix is not None and not window.busy)
    window.apply_values(dict(group_a='control', group_b='mutant', control_group='control'))
    def unexpected(*args):
        pytest.fail('Changed matrix must not build a job')
    monkeypatch.setattr(desktop_jobs, 'build_job', unexpected)
    with path.open('a') as stream:
        stream.write('c,control,MALE,3\n')
    window.build()
    assert errors and 'matrix changed' in errors[-1]
    assert window.record is None and window.loaded_matrix is None


def test_refresh_preserves_deselection_and_settings_restore_same_project(gui, tmp_path):
    from PySide6.QtCore import Qt
    from autobehaver.desktop import make_window
    app, window, errors = gui
    project = tmp_path / 'project'; path = matrix(project)
    window.fields['dataset'].setText(str(project)); window.tabs.setCurrentIndex(1)
    window.auto_load_featureimp()
    until(app, lambda: window.loaded_matrix == str(path) and not window.busy)
    window.table.item(0, 0).setCheckState(Qt.Unchecked)
    window.apply_values(dict(group_a='control', group_b='mutant', control_group='control'))
    window.load_matrix()
    until(app, lambda: not window.busy)
    assert window.table.item(0, 0).checkState() == Qt.Unchecked
    window.save_settings()
    restored = make_window(tmp_path / 'state')
    try:
        restored.tabs.setCurrentIndex(1); restored.auto_load_featureimp()
        until(app, lambda: restored.loaded_matrix == str(path) and not restored.busy)
        assert restored.fields['group_a'].currentData() == 'control'
        assert restored.fields['group_b'].currentData() == 'mutant'
    finally:
        restored.close()
