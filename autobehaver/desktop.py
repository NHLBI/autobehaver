"""Small desktop front end for the maintained HPC job preparation functions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from . import desktop_jobs as jobs
from . import desktop_monitor as monitor
from .desktop_recovery import build_recovery
from .desktop_ssh import DesktopSSH, SubmissionProgress
from .desktop_training import add_training_tab


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=None)
    parser.add_argument('--open', dest='open_draft', type=Path, help='Open a validated draft for review; does not build or submit.')
    parser.add_argument('--tab', choices=['pipeline', 'featureimp', 'training', 'monitor'], help='Tab to display when opening the launcher.')
    args = parser.parse_args(argv)
    initial_draft = None
    if args.open_draft:
        from .desktop_agent import load_draft
        try: initial_draft = (load_draft(args.open_draft), args.open_draft.resolve())
        except (ValueError, OSError, KeyError) as error: parser.error(str(error))
    state_dir = args.state_dir or (args.open_draft.resolve().with_suffix('.gui-state') if args.open_draft else Path.home() / '.autobehaver' / 'desktop')
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise SystemExit('Install the desktop dependency: python -m pip install -e ".[desktop]"')
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    window = make_window(state_dir, initial_draft=initial_draft)
    if args.tab:
        window.tabs.setCurrentIndex({'pipeline': 0, 'featureimp': 1, 'training': 2,
                                     'monitor': window.tabs.indexOf(window.monitor_page)}[args.tab])
    window.show()
    app.processEvents()
    window.tabs.currentWidget().verticalScrollBar().setValue(0)
    return app.exec()


def make_window(state_dir, *, initial_draft=None):
    from PySide6 import QtCore, QtGui, QtWidgets as W

    class Signals(QtCore.QObject):
        done = QtCore.Signal(object)
        failed = QtCore.Signal(str)
        progress = QtCore.Signal(str)

    class ConnectionSignals(QtCore.QObject):
        prompt = QtCore.Signal(object)
        output = QtCore.Signal(str)

    class Task(QtCore.QRunnable):
        def __init__(self, function):
            super().__init__()
            self.function, self.signals = function, Signals()

        def run(self):
            try:
                from .preparation_io import preparation_progress
                with preparation_progress(self.signals.progress.emit):
                    result = self.function()
                self.signals.done.emit(result)
            except Exception as error:
                self.signals.failed.emit(f'{type(error).__name__}: {error}')

    class Window(W.QMainWindow):
        def __init__(self):
            super().__init__()
            self.state_dir = Path(state_dir)
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.source_root = Path(__file__).resolve().parents[1]
            self.record, self.busy = None, False
            self.draft_path, self.draft_kind = None, "pipeline"
            self.record_snapshot = None
            self.session = DesktopSSH(self.state_dir / 'known_hosts')
            self.connection_signals = ConnectionSignals(self)
            self.connection_signals.prompt.connect(self.authentication_dialog)
            self.connection_signals.output.connect(self.append_output)
            self.fields, self.matrix_rows, self.loaded_matrix = {}, [], None
            self.advanced_sections = []
            self.saved_groups = {}
            self.saved_group_context = None
            self.pending_restored_selection = None
            self.saved_batch_sizes = {}
            self.fi_context = None
            self.matrix_generation = 0
            self.loaded_matrix_signature = None
            self.failed_matrix_signature = None
            self.pool = QtCore.QThreadPool(self)
            self.setWindowTitle('Autobehaver — HPC job launcher')
            screen = W.QApplication.primaryScreen().availableGeometry()
            self.resize(min(940, screen.width()-40), min(740, screen.height()-80))
            self.setMinimumSize(540, 400)
            self.setStyleSheet('''
                QMainWindow {background:#f4f6fa;} QWidget {font-family:'Segoe UI'; font-size:10pt; color:#263348;}
                QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPlainTextEdit,QTableWidget {background:white; padding:5px; border:1px solid #cdd5e1; border-radius:4px;}
                QPushButton {padding:7px 12px; background:#e8edf5; border:1px solid #c6d0df; border-radius:5px;}
                QPushButton:hover {background:#dce6f6;} QPushButton:disabled {color:#929bab; background:#eff1f5;}
                QPushButton#primary {background:#285daa; color:white;} QPushButton#primary:disabled {background:#a8b8d0;}
                QTabWidget::pane {border:1px solid #d3dae5; background:#f9fafc;}
                QTabBar::tab {padding:8px 12px; background:#e7ecf3;} QTabBar::tab:selected {background:white; color:#174b91;}
                QGroupBox {font-weight:600; border:1px solid #d3dae5; border-radius:5px; margin-top:14px; padding:12px;}
                QGroupBox::title {subcontrol-origin:margin; left:12px; padding:0 4px;}
            ''')
            central = W.QWidget()
            self.setCentralWidget(central)
            layout = W.QVBoxLayout(central)
            layout.setContentsMargins(12, 10, 12, 10)
            title = W.QLabel('Autobehaver')
            title.setStyleSheet('font-size:23pt; font-weight:600;')
            layout.addWidget(title)
            top = W.QHBoxLayout()
            self.new_button = self.button(top, 'New run', self.new_run)
            self.open_button = self.button(top, 'Open saved run...', self.open_run)
            self.save_button = self.button(top, 'Save run', self.save_run)
            top.addStretch()
            self.connection_label = W.QLabel('Disconnected')
            top.addWidget(self.connection_label)
            self.connect_button = self.button(top, 'Connect to HPC', self.connect)
            layout.addLayout(top)
            self.run_label = W.QLabel('New run - not built')
            self.run_label.setWordWrap(True)
            layout.addWidget(self.run_label)
            project_form = W.QFormLayout()
            project_form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
            self.form_entry(project_form, 'Project folder', 'dataset', browse='directory')
            layout.addLayout(project_form)
            actions = W.QHBoxLayout()
            self.build_button = self.button(actions, '1  Build job', self.build)
            self.build_button.setObjectName('primary')
            self.send_button = self.button(actions, '2  Send to HPC', self.send)
            self.submit_button = self.button(actions, '3  Submit job', self.submit)
            for button in (self.build_button, self.send_button, self.submit_button):
                button.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Fixed)
            layout.addLayout(actions)
            self.tabs = W.QTabWidget()
            layout.addWidget(self.tabs, 3)
            pipeline, p = self.page('Pipeline')
            self.form_entry(p, 'Job label', 'pipeline_name', 'pipeline')
            self.form_combo(p, 'Start from', 'start_at', [('Full movies → MMGen', 'crop'), ('Existing crops → MMGen', 'dlc'), ('Existing DLC → MMGen (start at KeyPre)', 'features')])
            self.form_combo(p, 'Processing', 'speed', [('Fast — parallel batches', 'fast'), ('Slow — entire dataset per stage', 'slow')])
            self.form_combo(p, 'Recording review', 'moviechecker_mode', [('Use saved MovieChecker review', 'reviewed'), ('Use recording occupancy only', 'recording-only')])
            review_button = W.QPushButton('Open MovieChecker for this dataset')
            review_button.clicked.connect(self.moviechecker)
            p.addRow('', review_button)
            note = W.QLabel('Select the project folder containing the numbered stage folders. Existing DLC starts from coordinate CSVs in 3_dlc.\nRecording review applies to full-movie starts. Saved chamber exclusions remain respected. FeatureImp is a separate run.')
            note.setWordWrap(True)
            p.addRow(note)
            new_only = W.QCheckBox('Process new recordings and update the master matrix')
            self.fields['new_only'] = new_only
            new_only.setToolTip('Uses the existing MMGen source ledger. Completed recordings are excluded from upstream jobs. Prior full reports are retained; affected-animal reports go into an append batch folder.')
            p.addRow(new_only)
            batch_box = W.QGroupBox('Fast batching')
            batch_form = W.QFormLayout(batch_box)
            for name, label, value in [('full_movies', 'Minimum full movies per job', 2),
                                       ('chambers', 'Minimum chamber recordings per job', 20),
                                       ('cpu_jobs', 'Maximum CPU jobs per stage', 50),
                                       ('gpu_jobs', 'Maximum GPU jobs per stage', 50)]:
                spin = W.QSpinBox(); spin.setRange(1, 100000); spin.setValue(value)
                self.fields[name] = spin; batch_form.addRow(label, spin)
            hint = W.QLabel('Inputs are balanced within these limits. A small dataset uses one job.\nCaps apply per stage, not across projects or your whole account. MMGen shards use at least 80 recordings per job.')
            hint.setWordWrap(True); batch_form.addRow(hint); p.addRow(batch_box)
            saved_batches = W.QCheckBox('Use saved per-stage batch sizes')
            self.fields['use_saved_batch_sizes'] = saved_batches
            self.saved_batch_note = W.QLabel(); self.saved_batch_note.setWordWrap(True)
            p.addRow(saved_batches); p.addRow(self.saved_batch_note)
            saved_batches.hide(); self.saved_batch_note.hide()
            advanced = W.QGroupBox('Job settings')
            grid = W.QGridLayout(advanced)
            for col, label in enumerate(['Stage', 'CPUs', 'RAM (GiB)', 'Hours']):
                grid.addWidget(W.QLabel(label), 0, col)
            self.allocations = {}
            for row, (stage, label, cpu, ram) in enumerate([
                ('crop', 'Autocropper', 8, 8), ('dlc', 'DeepLabCut', 6, 16),
                ('features', 'KeyPre + normalization', 10, 16), ('posepred', 'PosePred', 12, 24),
                ('mmgen', 'MMGen shards', 12, 32), ('mmgen_finish', 'MMGen finalization', 12, 80)], 1):
                grid.addWidget(W.QLabel(label), row, 0)
                widgets = []
                for col, (name, value, limit) in enumerate([('cpu', cpu, 256), ('ram', ram, 4096)], 1):
                    spin = W.QSpinBox(); spin.setRange(1, limit); spin.setValue(value)
                    grid.addWidget(spin, row, col); widgets.append(spin)
                    self.fields[f'{stage}_{name}'] = spin
                if stage in ('mmgen', 'mmgen_finish'):
                    hours = W.QSpinBox(); hours.setRange(1, 336)
                    hours.setValue(48 if stage == 'mmgen_finish' else 18)
                    self.fields[f'{stage}_hours'] = hours
                    grid.addWidget(hours, row, 3); widgets.append(hours)
                self.allocations[stage] = widgets
            parallel = W.QSpinBox(); parallel.setRange(0, 255); parallel.setSpecialValueText('Automatic'); parallel.setValue(8)
            self.fields['feature_workers'] = parallel
            grid.addWidget(W.QLabel('KeyPre / normalization worker ceiling'), 7, 0, 1, 2)
            grid.addWidget(parallel, 7, 2)
            parallel.setToolTip("Automatic uses each stage's CPU/RAM budget, with bounded chunks and 25% RAM headroom. A number is an optional ceiling; actual workers may be fewer.")
            p.addRow(advanced)
            from .desktop_advanced import add_controls, apply_shared_defaults
            add_controls(self, p, "pipeline", W)
            def update_start_controls():
                fast = self.fields['speed'].currentData() == 'fast'
                batch_box.setEnabled(fast and not saved_batches.isChecked())
                new_only.setEnabled(fast)
                from_movies = self.fields['start_at'].currentData() == 'crop'
                self.fields['full_movies'].setEnabled(fast and from_movies)
                self.fields['moviechecker_mode'].setEnabled(from_movies)
                review_button.setEnabled(from_movies)
                order = ['crop', 'dlc', 'features', 'posepred', 'mmgen']
                start = order.index(self.fields['start_at'].currentData())
                for stage, widgets in self.allocations.items():
                    for widget in widgets:
                        widget.setEnabled(order.index('mmgen' if stage == 'mmgen_finish' else stage) >= start)
            self.fields['start_at'].currentIndexChanged.connect(update_start_controls)
            self.fields['speed'].currentIndexChanged.connect(update_start_controls)
            self.update_start_controls = update_start_controls
            saved_batches.toggled.connect(update_start_controls)
            update_start_controls()
            fi, f = self.page('FeatureImp')
            self.fi_project_note = W.QLabel(); self.fi_project_note.setWordWrap(True)
            self.fi_matrix_note = W.QLabel(); self.fi_matrix_note.setWordWrap(True)
            self.fi_output_note = W.QLabel(); self.fi_output_note.setWordWrap(True)
            f.addRow('Project', self.fi_project_note)
            f.addRow('Master matrix', self.fi_matrix_note)
            f.addRow('Results folder', self.fi_output_note)
            override = W.QCheckBox('Use a different project or matrix')
            self.fields['fi_override'] = override
            f.addRow(override)
            self.fi_override_panel = W.QWidget()
            custom = W.QFormLayout(self.fi_override_panel); custom.setContentsMargins(0, 0, 0, 0)
            self.form_entry(custom, 'Results project', 'fi_dataset', browse='directory')
            self.form_entry(custom, 'Master matrix', 'matrix', browse='file')
            f.addRow(self.fi_override_panel)
            self.form_entry(f, 'Analysis name', 'fi_name', 'featureimp')
            load = W.QPushButton('Refresh matrix and groups')
            load.clicked.connect(self.load_matrix)
            f.addRow('', load)
            self.fi_matrix_status = W.QLabel('Choose a project folder above.'); self.fi_matrix_status.setWordWrap(True)
            f.addRow(self.fi_matrix_status)
            for label, name in [('Training group A', 'group_a'), ('Training group B', 'group_b'), ('Normalization reference', 'control_group')]:
                self.form_combo(f, label, name, [])
            self.form_combo(f, 'Sex', 'sex', [('All', 'all'), ('Male', 'male'), ('Female', 'female')])
            options = W.QHBoxLayout()
            for name, label in [('pca', 'Include PCA'), ('fsea', 'Include FSEA')]:
                check = W.QCheckBox(label); check.setChecked(True); self.fields[name] = check; options.addWidget(check)
            f.addRow('Additional analyses', options)
            resources = W.QHBoxLayout()
            for name, label, value, maximum in [('seed', 'Seed', 7, 2147483647), ('fi_cpus', 'CPUs', 12, 256), ('fi_memory', 'RAM (GiB)', 32, 4096)]:
                caption = W.QLabel(label); caption.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                resources.addWidget(caption); spin = W.QSpinBox(); spin.setRange(0 if name == 'seed' else 2, maximum); spin.setValue(value)
                self.fields[name] = spin; resources.addWidget(spin)
            f.addRow('Training settings', resources)
            check = W.QCheckBox('Plan locally and run training shards'); check.setChecked(True)
            self.fields['fi_sharded'] = check; f.addRow('Execution', check)
            for title, controls in [
                ('Shards', [('fi_shards', 'Training jobs', 8, 256)]),
                ('Finalization / PCA', [('fi_finish_cpus', 'CPUs', 8, 256), ('fi_finish_memory', 'RAM (GiB)', 32, 4096)]),
                ('FSEA', [('fi_fsea_cpus', 'CPUs', 8, 256), ('fi_fsea_memory', 'RAM (GiB)', 64, 4096)])]:
                row = W.QHBoxLayout()
                for name, label, default, maximum in controls:
                    caption = W.QLabel(label); caption.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                    row.addWidget(caption); spin = W.QSpinBox()
                    spin.setRange(1 if name == 'fi_shards' else 2, maximum); spin.setValue(default)
                    self.fields[name] = spin; row.addWidget(spin)
                    check.toggled.connect(spin.setEnabled)
                f.addRow(title, row)
            note = W.QLabel('Build freezes the matrix and plans locally. Send copies the plan; Submit schedules training, finalization and selected analyses. Worker counts adapt to available RAM. Analysis settings are available below under Advanced.')
            note.setWordWrap(True); f.addRow(note)
            f.addRow(W.QLabel('Included groups beyond A and B are scored as intermediate/test groups.\nUses nine predefined feature families. Uncheck individuals to exclude them.'))
            add_controls(self, f, 'featureimp', W)
            from .desktop_selection import add_selection_controls
            add_selection_controls(self, f, W)
            choices = W.QHBoxLayout()
            self.button(choices, 'Include all', lambda: self.select_rows(True))
            self.button(choices, 'Exclude all', lambda: self.select_rows(False))
            choices.addStretch(); f.addRow(choices)
            self.table = W.QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(['Include', 'Individual', 'Group', 'Sex', 'Trial'])
            self.table.itemChanged.connect(self.update_selection_filter)
            self.table.horizontalHeader().setSectionResizeMode(1, W.QHeaderView.Stretch)
            self.table.setMinimumHeight(200)
            f.addRow(self.table)
            settings, s = self.page('Connection')
            for label, name, default, browse in [
                ('SSH hostname', 'host', '', None), ('SSH username', 'username', '', None),
                ('Data root on this computer', 'data_local', '', 'directory'), ('Same data root on HPC', 'data_remote', '', None),
                ('Job-bundle root on this computer', 'jobs_local', '', 'directory'), ('Same job-bundle root on HPC', 'jobs_remote', '', None),
                ('Python executable on HPC', 'python', '', None), ('Resources folder on HPC', 'resources', '', None),
                ('PosePred model 16 (optional)', 'model16', '', None), ('PosePred model 32 (optional)', 'model32', '', None),
                ('PosePred model 64 (optional)', 'model64', '', None)]:
                self.form_entry(s, label, name, default, browse)
            connection_note = W.QLabel('Connect signs in here using password/MFA dialogs. Submit uses that same connection. Passwords are not saved.')
            connection_note.setWordWrap(True)
            s.addRow(connection_note)
            review, review_form = self.page('Review / output')
            self.review_page = review
            summary = W.QGroupBox('Prepared run — review before sending')
            sl = W.QVBoxLayout(summary)
            self.summary = W.QPlainTextEdit(); self.summary.setReadOnly(True)
            self.summary.setPlaceholderText('Choose settings above and click Build job. No files are sent and no jobs are submitted during Build.')
            self.summary.setMinimumHeight(100)
            self.summary.setMaximumHeight(170)
            sl.addWidget(self.summary)
            review_form.addRow(summary)
            self.output = W.QPlainTextEdit(); self.output.setReadOnly(True)
            self.output.setMaximumBlockCount(2000)
            self.output.setPlaceholderText('Submission progress appears here after you click Submit.')
            self.output.setMinimumHeight(100)
            review_form.addRow('Submission progress', self.output)
            log_actions = W.QHBoxLayout()
            self.refresh_button = self.button(log_actions, 'Read accepted job IDs', self.refresh)
            self.button(log_actions, 'Open logs', self.open_logs)
            review_form.addRow(log_actions)
            self.resume_button = W.QPushButton('Resume interrupted submission')
            self.resume_button.clicked.connect(lambda: self.submit(resume=True))
            self.resume_button.setToolTip('Recheck scheduler states and resume this frozen schedule. Active/completed jobs are retained; failed jobs may be retried. No jobs are cancelled.')
            review_form.addRow(self.resume_button)
            monitoring, monitor_form = self.page('Monitor')
            self.monitor_page = monitoring
            self.monitor_note = W.QLabel('Open a saved run or build a job to see its stages.')
            self.monitor_note.setWordWrap(True)
            monitor_form.addRow(self.monitor_note)
            monitor_actions = W.QHBoxLayout()
            self.monitor_refresh = self.button(monitor_actions, 'Refresh status', self.refresh_status)
            self.recovery_button = self.button(monitor_actions, 'Build recovery', self.recover)
            self.auto_refresh = W.QCheckBox('Refresh every 60 seconds')
            monitor_actions.addWidget(self.auto_refresh)
            monitor_form.addRow(monitor_actions)
            self.cards_body = W.QWidget()
            self.cards_grid = W.QGridLayout(self.cards_body)
            monitor_form.addRow(self.cards_body)
            legend = W.QLabel('Gray: waiting   Orange: running   Green: finished   Red: failed / blocked\nUnknown means no current scheduler state is available. Counts are jobs, not movies.\nFinished reports scheduler success; review stage outputs for scientific validation.')
            legend.setWordWrap(True)
            monitor_form.addRow(legend)
            self.monitor_data = None
            self.monitor_timer = QtCore.QTimer(self)
            self.monitor_timer.timeout.connect(self.poll_status)
            self.monitor_timer.start(60000)
            self.status = W.QLabel('Ready. Build creates a new local bundle.')
            self.status.setWordWrap(True)
            self.status.setMaximumHeight(65)
            layout.addWidget(self.status)
            training = add_training_tab(self, W)
            self.tabs.removeTab(self.tabs.indexOf(training))
            self.tabs.insertTab(2, training, 'PosePred training')
            apply_shared_defaults(self)
            self.default_values = self.values()
            if initial_draft is None:
                self.read_settings()
            else:
                self.restore_draft(*initial_draft)
            self.fi_refresh_timer = QtCore.QTimer(self); self.fi_refresh_timer.setSingleShot(True)
            self.fi_refresh_timer.timeout.connect(self.auto_load_featureimp)
            self.fi_poll_timer = QtCore.QTimer(self)
            self.fi_poll_timer.timeout.connect(self.auto_load_featureimp); self.fi_poll_timer.start(15000)
            for name in ('dataset', 'fi_dataset', 'matrix'):
                self.fields[name].textChanged.connect(self.featureimp_inputs_changed)
            override.toggled.connect(self.featureimp_inputs_changed)
            self.tabs.currentChanged.connect(self.featureimp_inputs_changed)
            self.sync_featureimp()
            self.update_buttons()
            self.connection_timer = QtCore.QTimer(self)
            self.connection_timer.timeout.connect(self.update_buttons)
            self.connection_timer.start(3000)
            from .desktop_selection import install_wheel_navigation
            install_wheel_navigation(self)
            for key in ('group_a', 'group_b'):
                self.fields[key].currentIndexChanged.connect(self.update_selection_filter)

        def page(self, title):
            scroll = W.QScrollArea(); scroll.setWidgetResizable(True)
            body = W.QWidget(); form = W.QFormLayout(body)
            form.setContentsMargins(16, 16, 16, 16)
            form.setFieldGrowthPolicy(W.QFormLayout.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(W.QFormLayout.WrapLongRows)
            scroll.setWidget(body); self.tabs.addTab(scroll, title)
            return scroll, form

        def button(self, layout, label, handler):
            button = W.QPushButton(label); button.clicked.connect(handler); layout.addWidget(button); return button

        def form_entry(self, form, label, name, default='', browse=None):
            entry = W.QLineEdit(default); self.fields[name] = entry
            if browse:
                row = W.QHBoxLayout(); row.addWidget(entry)
                self.button(row, 'Browse…', lambda: self.browse(entry, browse))
                form.addRow(label, row)
            else:
                form.addRow(label, entry)

        def form_combo(self, form, label, name, choices):
            combo = W.QComboBox(); self.fields[name] = combo
            for text, value in choices:
                combo.addItem(text, value)
            form.addRow(label, combo)

        def browse(self, entry, kind):
            value = W.QFileDialog.getExistingDirectory(self, 'Select folder', entry.text()) if kind == 'directory' else W.QFileDialog.getOpenFileName(self, 'Select file', entry.text(), 'All files (*)' if kind == 'anyfile' else 'CSV files (*.csv)')[0]
            if value:
                entry.setText(value)

        def values(self):
            result = {}
            for key, widget in self.fields.items():
                if isinstance(widget, W.QLineEdit): result[key] = widget.text().strip()
                elif isinstance(widget, W.QComboBox):
                    result[key] = widget.currentData()
                    if key in ('group_a','group_b','control_group') and result[key] is None and self.loaded_matrix is None and self.saved_group_context == self.fi_context:
                        result[key] = self.saved_groups.get(key)
                elif isinstance(widget, (W.QSpinBox, W.QDoubleSpinBox)): result[key] = widget.value()
                elif isinstance(widget, W.QCheckBox): result[key] = widget.isChecked()
            result['fi_execution_version'] = 1
            result['feature_execution_version'] = 1
            result['mmgen_allocation_version'] = 1
            result['saved_batch_sizes'] = dict(self.saved_batch_sizes)
            return result

        def apply_values(self, values, *, migrate=True):
            values = dict(values)
            if migrate and values.get('fi_execution_version', 0) < 1 and values.get('fi_memory') == 192:
                values['fi_memory'] = 32
            if migrate and values.get('feature_execution_version', 0) < 1:
                if values.get('feature_workers') == 2:
                    values['feature_workers'] = 0
                if values.get('features_cpu') == 4:
                    values['features_cpu'] = 8
            if migrate and values.get('mmgen_allocation_version', 0) < 1:
                # The old shared allocation was sized for finalization.
                values.setdefault('mmgen_finish_cpu', values.get('mmgen_cpu', 12))
                values.setdefault('mmgen_finish_ram', values.get('mmgen_ram', 80))
                values['mmgen_cpu'], values['mmgen_ram'] = 12, 32
            # Older per-stage settings migrate to the two shared controls.
            if 'full_movies' not in values and 'crop_batch' in values:
                values['full_movies'] = values['crop_batch']
            if 'chambers' not in values and 'dlc_batch' in values:
                values['chambers'] = values['dlc_batch']
            blockers = [QtCore.QSignalBlocker(widget) for widget in self.fields.values()]
            if 'saved_batch_sizes' in values:
                self.saved_batch_sizes = dict(values['saved_batch_sizes'])
                self.fields['use_saved_batch_sizes'].setVisible(bool(self.saved_batch_sizes))
                self.saved_batch_note.setVisible(bool(self.saved_batch_sizes))
                self.saved_batch_note.setText('Saved inputs per job: ' + ', '.join(f'{k}: {v}' for k,v in self.saved_batch_sizes.items()) + '. Uncheck to use the shared batching controls instead.')
            for key, value in values.items():
                widget = self.fields.get(key)
                if widget is None: continue
                if value is None:
                    if isinstance(widget, W.QComboBox): widget.setCurrentIndex(0 if widget.count() else -1)
                    continue
                if isinstance(widget, W.QLineEdit): widget.setText(str(value))
                elif isinstance(widget, W.QComboBox):
                    index = widget.findData(value)
                    if index >= 0: widget.setCurrentIndex(index)
                elif isinstance(widget, W.QDoubleSpinBox): widget.setValue(float(value))
                elif isinstance(widget, W.QSpinBox): widget.setValue(int(value))
                elif isinstance(widget, W.QCheckBox): widget.setChecked(bool(value))
            blockers.clear()
            for refresh in self.advanced_sections: refresh(expand=True)
            self.update_start_controls()
            if hasattr(self, 'fi_refresh_timer'):
                self.featureimp_inputs_changed()

        def current_kind(self):
            return {0: 'pipeline', 1: 'featureimp', 2: 'training'}.get(
                self.tabs.currentIndex(), self.record['kind'] if self.record else self.draft_kind)

        def clear_run(self):
            self.record = None
            self.record_snapshot = None
            self.draft_path = None
            self.monitor_data = None
            self.auto_refresh.setChecked(False)
            self.summary.clear()
            self.output.clear()
            self.show_monitor(dict(rows=[], source='No built run selected', updated=''))
            self.run_label.setText('New run - not built')
            self.update_buttons()

        def new_run(self):
            if self.busy: return
            # Preserve a draft before starting a fresh setup, without editing a built record.
            self.save_run()
            current = self.values()
            keep = ('host', 'username', 'data_local', 'data_remote', 'jobs_local',
                    'jobs_remote', 'python', 'resources', 'model16', 'model32', 'model64')
            values = dict(self.default_values)
            values.update({key: current[key] for key in keep})
            self.clear_run()
            self.clear_matrix_selection()
            self.saved_groups = {}
            self.saved_group_context = None
            self.pending_restored_selection = None
            self.apply_values(values, migrate=False)
            self.sync_featureimp()
            self.draft_kind = 'pipeline'
            self.tabs.setCurrentIndex(0)
            self.save_settings()
            self.status.setText('New run. Choose a project and job label, then Save run or Build job.')

        def save_run(self):
            if self.busy: return
            if self.record and self.values() == self.record_snapshot:
                self.save_settings()
                self.status.setText('This built run is already saved. New run starts a separate setup.')
                return
            from .desktop_drafts import write_draft
            kind, config = self.current_kind(), self.values()
            if self.loaded_matrix == config.get('matrix') and self.matrix_rows:
                config['selected_identities'] = [[row[k] for k in ('individual', 'group', 'sex')]
                    for index, row in enumerate(self.matrix_rows)
                    if self.table.item(index, 0).checkState() == QtCore.Qt.Checked]
            elif self.pending_restored_selection and self.pending_restored_selection['matrix'] == config.get('matrix'):
                config['selected_identities'] = [list(row) for row in sorted(self.pending_restored_selection['identities'])]
            path = write_draft(self.state_dir, kind, config, self.draft_path)
            self.clear_run()
            self.draft_path, self.draft_kind = path, kind
            self.run_label.setText(f'Saved draft: {path.stem} - not built')
            self.save_settings()
            self.status.setText(f'Saved separate run setup: {path}. No jobs changed.')

        def restore_draft(self, draft, path):
            if 'settings_schema' in draft:
                from .desktop_settings import SCHEMA_VERSION, validate_config
                if draft['settings_schema'] != SCHEMA_VERSION: raise ValueError('Unsupported draft settings schema.')
                draft = dict(draft, config=validate_config(draft['config'], draft['kind']))
            self.clear_run()
            self.clear_matrix_selection()
            values = dict(self.default_values)
            values.update(draft['config'])
            self.apply_values(values, migrate=False)
            self.sync_featureimp()
            self.saved_groups = {k: values.get(k) for k in ('group_a', 'group_b', 'control_group')}
            self.saved_group_context = self.fi_context
            self.pending_restored_selection = ({'matrix': values.get('matrix', ''),
                'identities': set(map(tuple, values['selected_identities']))}
                if 'selected_identities' in values else None)
            self.draft_path, self.draft_kind = Path(path), draft['kind']
            self.tabs.setCurrentIndex({'pipeline': 0, 'featureimp': 1, 'training': 2}[draft['kind']])
            self.run_label.setText(f'Saved draft: {Path(path).stem} - not built')
            self.update_buttons()

        def save_settings(self):
            jobs.save_json(self.state_dir / 'settings.json', self.values())
            path = self.draft_path or (Path(self.record['root']) / 'desktop_run.json' if self.record else '')
            jobs.save_json(self.state_dir / 'last_run.json', {'path': str(path)})

        def read_settings(self):
            try:
                path = self.state_dir / 'settings.json'
                if path.exists():
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    self.saved_groups = {key: saved.get(key) for key in ('group_a', 'group_b', 'control_group')}
                    self.saved_group_context = (saved.get('fi_dataset', ''), saved.get('matrix', ''))
                    self.apply_values(saved)
                last = self.state_dir / 'last_run.json'
                if last.exists():
                    run_path = Path(json.loads(last.read_text())['path'])
                    if run_path.is_file():
                        record = json.loads(run_path.read_text(encoding='utf-8'))
                        if record.get('record_type') == 'draft': self.restore_draft(record, run_path)
                        else: self.restore_run(record)
            except Exception as error:
                self.status.setText(f'Could not restore settings: {error}')

        def record_matches_form(self):
            if not self.record: return False
            kind = self.current_kind()
            if kind != self.record['kind']: return False
            from .desktop_drafts import identity
            from .desktop_restore import restored_values
            return identity(kind, self.values()) == identity(kind, restored_values(self.record, self.default_values))

        def update_buttons(self):
            for button in (self.new_button, self.open_button, self.save_button): button.setEnabled(not self.busy)
            self.build_button.setEnabled(not self.busy and self.tabs.currentIndex() in (0, 1, 2))
            self.send_button.setEnabled(not self.busy and self.record is not None and not self.record.get('published') and self.record_matches_form())
            submit_label, can_submit = jobs.submission_button_state(self.record)
            self.submit_button.setText(submit_label)
            self.submit_button.setEnabled(not self.busy and self.session.connected and can_submit and self.record_matches_form())
            self.refresh_button.setEnabled(not self.busy and self.record is not None)
            self.monitor_refresh.setEnabled(not self.busy and self.record is not None)
            sharded_fi = bool(self.record and (self.record.get('featureimp_plan') or {}).get('local_plan'))
            self.recovery_button.setEnabled(not self.busy and self.session.connected and self.record is not None and self.record['kind'] != 'training' and not sharded_fi)
            self.recovery_button.setToolTip('Sharded FeatureImp needs a stage-specific retry using its original plan; automatic failed-stage recovery is not yet available.' if sharded_fi else ('For training, build a new run with the saved checkpoint to resume.' if self.record and self.record['kind'] == 'training' else 'Build replacements for failed or cancelled jobs.'))
            self.resume_button.setEnabled(not self.busy and self.session.connected and jobs.can_resume_submission(self.record))
            self.connect_button.setEnabled(not self.busy)
            self.connect_button.setText('Disconnect' if self.session.connected else 'Connect to HPC')
            self.connection_label.setText('Connected' if self.session.connected else 'Disconnected')

        def work(self, message, function, done, failed=None):
            if self.busy: return
            self.busy = True; self.update_buttons(); self.status.setText(message)
            self.task = Task(function)
            self.task.signals.progress.connect(self.status.setText)
            def success(result):
                self.busy = False
                try: done(result)
                except Exception as error: self.error(str(error))
                self.update_buttons()
            def failure(error):
                self.busy = False
                (failed or self.error)(error)
                self.update_buttons()
            self.task.signals.done.connect(success); self.task.signals.failed.connect(failure)
            self.pool.start(self.task)

        def error(self, text):
            self.status.setText(text)
            W.QMessageBox.warning(self, 'Autobehaver', text)

        def build(self):
            self.sync_featureimp()
            config = self.values()
            kind = {0: 'pipeline', 1: 'featureimp', 2: 'training'}.get(self.tabs.currentIndex())
            if kind is None: return
            from .desktop_settings import validate_config
            try: config = validate_config(config, kind)
            except ValueError as error:
                self.error(str(error)); return
            config['kind'] = kind
            config['run_name'] = config[{'featureimp': 'fi_name', 'pipeline': 'pipeline_name', 'training': 'training_name'}[kind]]
            if kind == 'training':
                config['dataset'] = config['training_directory']
                config['training_options'] = {k.removeprefix('train_opt_'): v for k, v in config.items() if k.startswith('train_opt_')}
            if kind == 'featureimp':
                if self.loaded_matrix != config['matrix']:
                    self.error('Load the selected matrix before building FeatureImp.'); return
                if self.matrix_signature(config['matrix']) != self.loaded_matrix_signature:
                    self.clear_matrix_selection()
                    self.fi_refresh_timer.start(0)
                    self.error('The master matrix changed. Refresh its groups and review the selection before building.'); return
                config['dataset'] = config['fi_dataset']
                config['selected_identities'] = [[row[k] for k in ('individual', 'group', 'sex')]
                    for index, row in enumerate(self.matrix_rows)
                    if self.table.item(index, 0).checkState() == QtCore.Qt.Checked]
            config['models'] = [config[name] for name in ('model16', 'model32', 'model64')]
            if not any(config['models']): config['models'] = []
            config['batching'] = {key: config[key] for key in ('full_movies', 'chambers', 'cpu_jobs', 'gpu_jobs')} if kind == 'pipeline' and config['speed'] == 'fast' else None
            if kind == 'pipeline' and config['use_saved_batch_sizes']:
                config['batching'] = None
                config['batch_sizes'] = dict(self.saved_batch_sizes)
            config['allocations'] = {stage: [widgets[0].value(), widgets[1].value()] for stage, widgets in self.allocations.items()}
            config['walltime_hours'] = {stage: self.fields[f'{stage}_hours'].value() for stage in ('mmgen', 'mmgen_finish')}
            self.save_settings()
            self.work('Building locally: inventory, plan and frozen source…',
                      lambda: jobs.build_job(config, self.source_root, self.state_dir / 'runs'), self.built)

        def built(self, record):
            # Recovery creates a new run identity. Adopt it before evaluating
            # the Send/Submit guards, just as when opening a saved run.
            if record['config'].get('recovery'):
                self.restore_run(record)
            else:
                self.set_record(record)
            self.save_settings()
            self.tabs.setCurrentWidget(self.review_page)
            self.review_page.verticalScrollBar().setValue(0)
            self.status.setText('Built locally. Review the summary, then Send to HPC. Nothing submitted.')

        def set_record(self, record):
            changed = self.record is None or self.record['root'] != record['root']
            self.record = record
            self.draft_path = None
            self.record_snapshot = self.values()
            self.run_label.setText(f'Built run: {record["name"]}')
            if changed:
                rows = []
                for stage, count in record.get('monitor_job_counts', record['job_counts']).items():
                    rows.extend(dict(stage='mmgen' if stage == 'mmgen_finish' else stage,
                                     category='unknown', key='', job_id='', state='Not refreshed', exit='', reason='')
                                for _ in range(count))
                data = dict(rows=rows, source='Not refreshed', updated='')
                cache = Path(record['root']) / 'desktop_status.json'
                try:
                    if cache.exists():
                        data = json.loads(cache.read_text(encoding='utf-8'))
                        data['source'] = 'Saved snapshot (refresh for current status)'
                except (OSError, ValueError):
                    pass
                self.show_monitor(data)
            config = record['config']
            lines = [record['name'], f"Workflow: {record['kind']}   |   Dataset: {record['project']}",
                     'Jobs: ' + ', '.join(f'{stage}: {count}' for stage, count in record['job_counts'].items()),
                     f"Total: {sum(record['job_counts'].values())} jobs"]
            if record['kind'] == 'pipeline':
                if record.get('movies') is not None:
                    lines.append(f"Coverage: {record['chambers']} chamber recordings" if config['start_at'] == 'features' else f"Coverage: {record['movies']} original movies / {record['chambers']} chamber recordings")
                if config.get('recovery'):
                    lines.append('Recovery: only the jobs listed above will be submitted. Earlier successful stages will not rerun.')
                else:
                    lines.append(f"Start: {config['start_at']}   Speed: {config['speed']}   Review: {config['moviechecker_mode']}")
                lines.append('Allocations (CPUs / GiB): ' + ', '.join(f'{k} {v[0]}/{v[1]}' for k, v in config['allocations'].items() if not config.get('recovery') or k in record['job_counts']))
                from .desktop_restore import restored_values
                saved_limits = restored_values(record, self.default_values)
                lines.append('MMGen time limits: shards ' + str(saved_limits['mmgen_hours']) + ' h; finalization ' + str(saved_limits['mmgen_finish_hours']) + ' h')
                batch = config.get('pp_batch', 0)
                lines.append('PosePred: ' + ('automatic batch sizing' if batch == 0 else f'batch ceiling {batch}') + '; CPU/RAM/VRAM budgeting enabled')
                if 'feature_workers' in config:
                    lines.append("KeyPre / normalization: automatic CPU/RAM budgeting" if config['feature_workers'] == 0
                                 else f"KeyPre / normalization worker ceiling: {config['feature_workers']} (RAM limited)")
            elif record['kind'] == 'training':
                training = record['training']
                lines.append(f"Mode: {training['mode']}; metadata rows: {training['metadata_rows']}")
                lines.append(f"Outputs: {training['output']}")
                lines.append(f"Resources: {config.get('train_cpus', 16)} CPUs / {config.get('train_memory', 32)} GiB / {config.get('train_gres', 'gpu:a100:1')}; time {config.get('train_time', '24:00:00')}")
                import shlex
                lines.append('Command: ' + shlex.join(training['command']))
            else:
                lines.append(f"A: {config['group_a']}   B: {config['group_b']}   Reference: {config['control_group']}   Sex: {config['sex']}")
                lines.append(f"Selected individuals: {len(config['selected_identities'])}; PCA: {config['pca']}; FSEA: {config['fsea']}")
                fi = record.get('featureimp_plan') or {}
                if fi.get('local_plan'):
                    lines.append(f"Local plan: {fi['training_rows']} training animals, {fi['selected_features']} features, {fi['planned_tasks']} subset tasks across {fi['active_shards']} training jobs")
                    lines.append(f"CPUs / GiB: training {config.get('fi_cpus',12)}/{config.get('fi_memory',32)}; finalization/PCA {config.get('fi_finish_cpus',8)}/{config.get('fi_finish_memory',32)}; FSEA {config.get('fi_fsea_cpus',8)}/{config.get('fi_fsea_memory',64)}")
                    lines.append('Results: ' + fi['result_dir'])
            if config.get('batching'):
                b = config['batching']
                lines.append(f"Batch minima: {b['full_movies']} full movies / {b['chambers']} chambers; stage caps: {b['cpu_jobs']} CPU / {b['gpu_jobs']} GPU jobs")
            if record.get('incremental'):
                inc = record['incremental']
                lines.append(f"New recordings: {inc['new_recordings']}; previously committed: {inc['previous_recordings']}. One MMGen append job updates the combined matrix.")
                lines.append(inc['full_reports'])
            if config.get('recovery'):
                recovery = config['recovery']
                if record['kind'] == 'pipeline':
                    lines.append(f"Recovery: {len(recovery['replace_keys'])} processing jobs replaced; {len(recovery['retained'])} retained, plus new accounting.")
                    lines.append('Existing running and completed jobs are preserved. Batch membership follows the original plan.')
                else:
                    lines.append('FeatureImp retry: original frozen matrix and comparisons; a new results folder preserves the failed attempt.')
            lines.extend([f"Bundle: {record['remote']}", 'Publication: ' + ('Verified' if record.get('published') else 'Local only'),
                          'Send/Submit use this frozen plan. Changing fields above requires a new Build.'])
            from .desktop_settings import changed_advanced
            changes = changed_advanced(config, record['kind'])
            lines.append('Advanced overrides: ' + (', '.join(f'{k}={v}' for k,v in changes.items()) or 'none'))
            self.summary.setPlainText('\n'.join(lines))
            self.update_buttons()

        def send(self):
            if self.busy or not self.record_matches_form():
                self.status.setText('Build this setup before sending it. The previous run is unchanged.'); return
            record = self.record
            self.work('Sending bundle and verifying every file…', lambda: jobs.publish(record), self.sent)

        def sent(self, record):
            self.set_record(record); self.save_settings()
            self.status.setText('Sent and verified. Click Submit.' if self.session.connected else 'Sent and verified. Connect to HPC, then Submit.')

        def prompt_from_worker(self, kind, data):
            request = {'kind': kind, 'data': data, 'event': threading.Event(), 'response': None}
            self.connection_signals.prompt.emit(request)
            if not request['event'].wait(240):
                raise TimeoutError('Login dialog timed out. Connect again.')
            return request.pop('response')

        def authentication_dialog(self, request):
            data = request['data']
            try:
                if request['kind'] == 'host_key':
                    box = W.QMessageBox(self)
                    box.setWindowTitle('Verify HPC identity')
                    box.setTextFormat(QtCore.Qt.PlainText)
                    box.setText(f"First connection to {data['host']}.\n\n{data['type']}\n{data['fingerprint']}\n\nTrust and remember this host key?")
                    box.setStandardButtons(W.QMessageBox.Yes | W.QMessageBox.No)
                    box.setDefaultButton(W.QMessageBox.No)
                    request['response'] = box.exec() == W.QMessageBox.Yes
                else:
                    dialog = W.QDialog(self); dialog.setWindowTitle(data['title'])
                    form = W.QFormLayout(dialog)
                    label = W.QLabel(data['instructions']); label.setWordWrap(True); label.setTextFormat(QtCore.Qt.PlainText)
                    form.addRow(label)
                    inputs = []
                    for text, echo in data['prompts']:
                        field = W.QLineEdit(); field.setEchoMode(W.QLineEdit.Normal if echo else W.QLineEdit.Password)
                        prompt_label = W.QLabel(text); prompt_label.setWordWrap(True); prompt_label.setTextFormat(QtCore.Qt.PlainText)
                        form.addRow(prompt_label, field); inputs.append(field)
                    buttons = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
                    buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
                    form.addRow(buttons)
                    if inputs: inputs[0].setFocus()
                    if dialog.exec() == W.QDialog.Accepted:
                        request['response'] = [field.text() for field in inputs]
                    for field in inputs: field.clear()
                    dialog.deleteLater()
            finally:
                request['event'].set()

        def append_output(self, text):
            cursor = self.output.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End); cursor.insertText(text)
            self.output.setTextCursor(cursor); self.output.ensureCursorVisible()

        def connect(self):
            if self.session.connected:
                self.session.close(); self.update_buttons(); self.status.setText('Disconnected.'); return
            config = self.values()
            self.save_settings()
            self.work('Connecting to HPC…',
                lambda: self.session.connect(config['host'], config['username'], self.prompt_from_worker),
                lambda message: self.status.setText(message))

        def submit(self, *, resume=False):
            if not self.record_matches_form():
                self.status.setText('Open the intended built run or build this setup before submitting.'); return
            if not self.session.connected:
                self.error('Connect to HPC first.'); return
            current = self.values()
            if (current['host'], current['username']) != (self.session.host, self.session.username):
                self.error('Connection settings changed. Disconnect and connect again before submitting.'); return
            self.tabs.setCurrentWidget(self.review_page); self.output.clear()
            QtCore.QTimer.singleShot(0, lambda: self.review_page.ensureWidgetVisible(self.output))
            record = self.record
            def run():
                command = jobs.verify_submission(record, resume=resume)
                with (Path(record['root']) / 'submission_output.log').open('a', encoding='utf-8') as log:
                    progress = SubmissionProgress(self.connection_signals.output.emit, log)
                    try:
                        return self.session.execute(command, progress)
                    finally:
                        progress.finish()
            self.work('Submitting on the existing connection…', run, self.submitted)

        def submitted(self, code):
            self.status.setText('Submission finished. Read accepted job IDs to review the recorded jobs.')

        def refresh(self):
            record = self.record
            def read():
                path = Path(record['mounted']) / 'jobs.tsv'
                return path.read_text(encoding='utf-8') if path.exists() else ''
            def display(text):
                self.set_record(record)
                self.summary.appendPlainText('\nAccepted job IDs (submission records, not completion):\n' + (text or 'None recorded.'))
                count = len(text.strip().splitlines()) if text.strip() else 0
                self.status.setText(f'{count} job IDs recorded of {sum(record["job_counts"].values())} planned. This does not report processing completion.')
            self.work('Reading the bundle’s accepted job IDs…', read, display)

        def recover(self):
            if not self.record or not self.session.connected or self.busy:
                return
            parent = self.record
            allocations = {stage: [widgets[0].value(), widgets[1].value()] for stage, widgets in self.allocations.items()}
            if parent['kind'] == 'featureimp':
                allocations = {'featureimp': [self.fields['fi_cpus'].value(), self.fields['fi_memory'].value()]}
            walltime = {stage: self.fields[f'{stage}_hours'].value() for stage in ('mmgen', 'mmgen_finish')}
            def prepare():
                current = monitor.snapshot(parent, self.session)
                return build_recovery(parent, current, self.source_root, self.state_dir / 'runs', allocations, walltime_hours=walltime)
            self.work('Checking current job states and building replacements locally...', prepare, self.built)

        def poll_status(self):
            if self.auto_refresh.isChecked() and self.record and self.session.connected and not self.busy:
                self.refresh_status(automatic=True)

        def refresh_status(self, automatic=False):
            if not self.record or self.busy:
                return
            record = self.record
            self.monitor_note.setText('Refreshing status... Previous colors remain until the query finishes.')
            def read():
                try:
                    return monitor.snapshot(record, self.session)
                except Exception as error:
                    return {'error': str(error)}
            def done(data):
                if 'error' in data:
                    self.monitor_note.setText('Refresh failed; displayed status is stale. ' + data['error'])
                    self.status.setText('Could not refresh monitoring. No jobs were changed.')
                    return
                jobs.save_json(Path(record['root']) / 'desktop_status.json', data)
                self.show_monitor(data)
                self.status.setText('Job status refreshed. No jobs were changed.')
            self.work('Reading job status...', read, done)

        def show_monitor(self, data):
            self.monitor_data = data
            self.monitor_note.setText(data['source'] + (' | Updated ' + data['updated'] if data['updated'] else ''))
            while self.cards_grid.count():
                item = self.cards_grid.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            for index, (stage, counts) in enumerate(monitor.stage_counts(data['rows']).items()):
                total = sum(counts.values())
                card = W.QGroupBox(monitor.LABELS.get(stage, stage))
                box = W.QVBoxLayout(card)
                button = W.QPushButton(f"{counts['finished']} / {total} finished")
                button.clicked.connect(lambda checked=False, stage=stage: self.stage_details(stage))
                box.addWidget(button)
                bar = W.QHBoxLayout(); bar.setSpacing(1)
                for key, color in monitor.COLORS.items():
                    count = counts[key]
                    if count:
                        segment = W.QLabel(); segment.setFixedHeight(14)
                        segment.setStyleSheet(f'background:{color}; border-radius:2px;')
                        segment.setToolTip(f'{count} {key}')
                        bar.addWidget(segment, count)
                box.addLayout(bar)
                text = W.QLabel(f"{counts['waiting']} waiting | {counts['running']} running\n{counts['failed']} failed | {counts['unknown']} unknown")
                text.setWordWrap(True); box.addWidget(text)
                self.cards_grid.addWidget(card, index // 2, index % 2)

        def stage_details(self, stage):
            dialog = W.QDialog(self)
            dialog.setWindowTitle(monitor.LABELS.get(stage, stage) + ' jobs')
            dialog.resize(700, 420)
            layout = W.QVBoxLayout(dialog)
            rows = [row for row in self.monitor_data['rows'] if row['stage'] == stage]
            table = W.QTableWidget(len(rows), 5)
            table.setHorizontalHeaderLabels(['Job', 'ID', 'State', 'Exit', 'Reason'])
            table.setSelectionBehavior(W.QAbstractItemView.SelectRows)
            table.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
            for i, row in enumerate(rows):
                for j, key in enumerate(('key', 'job_id', 'state', 'exit', 'reason')):
                    table.setItem(i, j, W.QTableWidgetItem(row.get(key, '')))
            table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeToContents)
            layout.addWidget(table)
            actions = W.QHBoxLayout()
            def open_selected_log(suffix):
                selected = table.currentRow()
                if selected < 0:
                    return
                row = rows[selected]
                root = row.get('bundle') or (self.record['mounted'] if self.record.get('published') else self.record['prepared'])
                path = Path(root) / 'logs' / f"{row['key']}_{row['job_id']}.{suffix}"
                if not path.is_file():
                    W.QMessageBox.information(dialog, 'Log not available', 'This log is not available yet.')
                    return
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
            self.button(actions, 'Output log', lambda: open_selected_log('out'))
            self.button(actions, 'Error log', lambda: open_selected_log('err'))
            self.button(actions, 'Log folder', self.open_logs)
            self.button(actions, 'Close', dialog.accept)
            layout.addLayout(actions)
            dialog.exec()

        def open_run(self):
            if self.busy:
                self.status.setText('Wait for the current operation to finish before opening another run.'); return
            path, _ = W.QFileDialog.getOpenFileName(self, 'Open saved run or draft', str(self.state_dir), 'Saved runs (*.json)')
            if path:
                try:
                    record = json.loads(Path(path).read_text(encoding='utf-8'))
                    if record.get('format') != 1: raise ValueError('Unsupported run record.')
                    if record.get('record_type') == 'draft':
                        self.restore_draft(record, path)
                    else:
                        self.restore_run(record)
                        self.tabs.setCurrentWidget(self.monitor_page)
                    self.save_settings()
                except Exception as error: self.error(str(error))

        def restore_run(self, record):
            from .desktop_restore import restored_values
            values = restored_values(record, self.default_values)
            self.clear_matrix_selection()
            self.apply_values(values, migrate=False)
            self.sync_featureimp()
            self.saved_groups = {key: values.get(key) for key in ('group_a','group_b','control_group')}
            self.saved_group_context = self.fi_context
            self.pending_restored_selection = ({'matrix': self.fields['matrix'].text().strip(),
                'identities': set(map(tuple, record['config']['selected_identities']))}
                if record['kind'] == 'featureimp' and 'selected_identities' in record['config'] else None)
            for combo in self.selection_filters.values(): combo.setCurrentIndex(0)
            self.set_record(record)

        def open_logs(self):
            if self.record:
                root = self.record['mounted'] if self.record.get('published') else self.record['prepared']
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(root) / 'logs')))

        @staticmethod
        def matrix_signature(matrix):
            if not matrix:
                return None
            try:
                info = Path(matrix).stat()
                return (info.st_size, info.st_mtime_ns) if Path(matrix).is_file() else None
            except OSError:
                return None

        def clear_matrix_selection(self):
            self.matrix_generation += 1
            self.loaded_matrix, self.loaded_matrix_signature = None, None
            self.matrix_rows = []; self.table.setRowCount(0)
            for key in ('group_a', 'group_b', 'control_group'):
                self.fields[key].clear()

        def sync_featureimp(self):
            override = self.fields['fi_override'].isChecked()
            self.fi_override_panel.setVisible(override)
            if not override:
                project = self.fields['dataset'].text().strip()
                matrix = str(Path(project) / '8_master_matrix' / 'MASTER_FEATURE_MATRIX.csv') if project else ''
                for key, value in (('fi_dataset', project), ('matrix', matrix)):
                    blocker = QtCore.QSignalBlocker(self.fields[key])
                    self.fields[key].setText(value)
                    del blocker
            project = self.fields['fi_dataset'].text().strip()
            matrix = self.fields['matrix'].text().strip()
            context = (project, matrix)
            if context != self.fi_context:
                self.clear_matrix_selection()
                self.failed_matrix_signature = None
                if context != self.saved_group_context:
                    self.saved_groups = {}
                self.fi_context = context
                self.fi_matrix_status.setText('Checking for the project master matrix…' if matrix else 'Choose a project folder above.')
            self.fi_project_note.setText(project or 'Use the project folder above')
            self.fi_matrix_note.setText(matrix or 'Automatically selected from the project')
            self.fi_output_note.setText(str(Path(project) / '9_feature_importance') if project else 'Automatically selected from the project')

        def featureimp_inputs_changed(self, *_):
            self.sync_featureimp()
            self.fi_refresh_timer.start(350)

        def auto_load_featureimp(self):
            if self.tabs.currentIndex() != 1 or self.busy:
                return
            self.sync_featureimp()
            matrix = self.fields['matrix'].text().strip()
            signature = self.matrix_signature(matrix)
            if signature is None:
                if self.loaded_matrix is not None:
                    self.clear_matrix_selection()
                self.fi_matrix_status.setText('Waiting for MMGen to create the master matrix.' if matrix else 'Choose a project folder above.')
                return
            if self.loaded_matrix == matrix and self.loaded_matrix_signature == signature:
                return
            if self.failed_matrix_signature == (self.fi_context, signature):
                return
            self.load_matrix(automatic=True)

        def load_matrix(self, _checked=False, *, automatic=False):
            if self.busy:
                return
            self.sync_featureimp()
            matrix = self.fields['matrix'].text().strip()
            signature = self.matrix_signature(matrix)
            if signature is None:
                self.fi_matrix_status.setText('Waiting for MMGen to create the master matrix.' if matrix else 'Choose a project folder above.')
                return
            context, generation = self.fi_context, self.matrix_generation
            old = self.values()
            included = {tuple(row[key] for key in ('individual', 'group', 'sex'))
                        for index, row in enumerate(self.matrix_rows)
                        if self.table.item(index, 0).checkState() == QtCore.Qt.Checked}
            same_matrix = self.loaded_matrix == matrix and self.loaded_matrix_signature == signature
            restored = self.pending_restored_selection
            restoring = restored is not None and restored['matrix'] == matrix
            if restoring:
                included = restored['identities']; same_matrix = True
            def read():
                rows = jobs.matrix_inventory(matrix)
                if self.matrix_signature(matrix) != signature:
                    raise ValueError('The master matrix changed while reading. Wait for MMGen to finish, then refresh.')
                return rows
            def loaded(rows):
                if self.fi_context != context or self.matrix_generation != generation:
                    self.fi_refresh_timer.start(0); return
                self.loaded_matrix = matrix; self.loaded_matrix_signature = signature
                self.failed_matrix_signature = None; self.matrix_rows = rows
                blocker = QtCore.QSignalBlocker(self.table)
                self.table.setRowCount(len(rows))
                for index, row in enumerate(rows):
                    include = W.QTableWidgetItem(); include.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                    selected = not same_matrix or tuple(row[key] for key in ('individual', 'group', 'sex')) in included
                    include.setCheckState(QtCore.Qt.Checked if selected else QtCore.Qt.Unchecked); self.table.setItem(index, 0, include)
                    for col, key in enumerate(('individual', 'group', 'sex', 'trial'), 1):
                        item = W.QTableWidgetItem(row.get(key, '') or ('Unknown' if key == 'trial' else '')); item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                        self.table.setItem(index, col, item)
                del blocker
                for key, combo in self.selection_filters.items():
                    previous = combo.currentData(); guard = QtCore.QSignalBlocker(combo)
                    combo.clear(); combo.addItem('All', None)
                    if key == 'group': combo.addItem('Training groups A + B', '__training_groups__')
                    for value in sorted({row.get(key, '') for row in rows}):
                        combo.addItem(value or 'Unknown', value)
                    combo.setCurrentIndex(max(0, combo.findData(previous)))
                    del guard
                groups = sorted({row['group'] for row in rows})
                for key in ('group_a', 'group_b', 'control_group'):
                    combo = self.fields[key]; combo.clear()
                    combo.addItem('Select a group…', None)
                    for group in groups: combo.addItem(group, group)
                    index = combo.findData(old.get(key) or self.saved_groups.get(key))
                    combo.setCurrentIndex(index if index >= 0 else 0)
                message = f'Loaded {len(rows)} individuals in {len(groups)} groups. Choose A, B and the normalization reference.'
                self.status.setText(message); self.fi_matrix_status.setText(message)
                if restoring and self.pending_restored_selection is restored:
                    missing = included - {tuple(row[k] for k in ('individual','group','sex')) for row in rows}
                    self.pending_restored_selection = None
                    if missing:
                        self.status.setText(f'Restored selection: {len(missing)} saved individuals are missing from this matrix. Review before building.')
                self.update_selection_filter()
            def failed(message):
                if self.fi_context != context or self.matrix_generation != generation:
                    self.fi_refresh_timer.start(0); return
                self.clear_matrix_selection()
                self.failed_matrix_signature = (context, signature)
                self.fi_matrix_status.setText(message)
                if not automatic:
                    self.error(message)
            self.work('Reading matrix identities and groups…', read, loaded, failed=failed)

        def select_rows(self, checked):
            blocker = QtCore.QSignalBlocker(self.table)
            for row in range(self.table.rowCount()):
                self.table.item(row, 0).setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            del blocker
            self.update_selection_filter()

        def selection_matches(self, row):
            from .desktop_selection import matches
            filters = {key: combo.currentData() for key, combo in self.selection_filters.items()}
            if filters['group'] == '__training_groups__':
                filters['group'] = None
                filters['training_groups'] = {self.fields[key].currentData() for key in ('group_a', 'group_b')}
            return matches(row, **filters)

        def update_selection_filter(self, *_):
            if not hasattr(self, 'table'): return
            matched = included = 0
            for index, row in enumerate(self.matrix_rows):
                match = self.selection_matches(row); matched += match
                self.table.setRowHidden(index, not match)
                item = self.table.item(index, 0)
                included += bool(item and item.checkState() == QtCore.Qt.Checked)
            self.selection_count.setText(f'Matching: {matched} / {len(self.matrix_rows)} animals. Included: {included} / {len(self.matrix_rows)} (before analysis Sex filter).')

        def apply_selection_filter(self, mode):
            if self.selection_filters['group'].currentData() == '__training_groups__' and not all(self.fields[k].currentData() for k in ('group_a', 'group_b')):
                self.error('Choose training groups A and B first.'); return
            blocker = QtCore.QSignalBlocker(self.table)
            for index, row in enumerate(self.matrix_rows):
                match = self.selection_matches(row)
                if mode == 'only' or match:
                    checked = match if mode == 'only' else mode == 'include'
                    self.table.item(index, 0).setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            del blocker
            self.update_selection_filter()

        def moviechecker(self):
            dataset = self.fields['dataset'].text().strip()
            if not Path(dataset, '1_full_movies').is_dir():
                self.error('Choose a dataset containing 1_full_movies.'); return
            subprocess.Popen([sys.executable, '-m', 'autobehaver.preinf.moviechecker', '--main_dir', dataset], cwd=self.source_root)

        def closeEvent(self, event):
            if self.busy:
                self.status.setText('Wait for the current local operation to finish before closing.'); event.ignore(); return
            self.save_run(); self.save_settings(); self.session.close(); event.accept()

    window = Window()
    window.tabs.currentChanged.connect(lambda _: window.update_buttons())
    return window


if __name__ == '__main__':
    sys.exit(main())
