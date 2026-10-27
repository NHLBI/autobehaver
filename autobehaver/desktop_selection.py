"""Identity-based bulk selection; filters do not alter scientific matrix columns."""
import re


def trial_for_row(row):
    explicit = str(row.get('trial') or '').strip()
    if explicit:
        return re.sub(r'^trial', '', explicit, flags=re.I) or explicit
    match = re.search(r'(?:^|_)trial([^_]+)(?:_|$)', str(row.get('individual', '')), re.I)
    return match.group(1) if match else ''


def matches(row, *, group=None, sex=None, trial=None, training_groups=None):
    if training_groups is not None and row['group'] not in training_groups:
        return False
    return ((group is None or row['group'] == group)
            and (sex is None or row['sex'].casefold() == sex.casefold())
            and (trial is None or trial_for_row(row) == trial))


def install_wheel_navigation(window):
    """Wheel over value selectors scrolls their tab instead of changing values."""
    from PySide6 import QtCore, QtWidgets as W

    class Navigation(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() != QtCore.QEvent.Wheel or not isinstance(obj, W.QWidget):
                return False
            widget = obj
            selector = None
            while widget is not None and widget is not window:
                if isinstance(widget, (W.QAbstractSpinBox, W.QComboBox)):
                    selector = widget
                    break
                widget = widget.parentWidget()
            if selector is None or selector.window() is not window:
                return False
            parent = selector.parentWidget()
            while parent is not None and not isinstance(parent, W.QScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                bar = parent.verticalScrollBar()
                delta = event.pixelDelta().y() or event.angleDelta().y() / 120 * bar.singleStep() * 3
                bar.setValue(bar.value() - round(delta))
            event.accept()
            return True

    handler = Navigation(window)
    W.QApplication.instance().installEventFilter(handler)
    window.wheel_navigation = handler


def add_selection_controls(window, form, W):
    """Install explicit include/exclude actions above the individual table."""
    window.selection_filters = {}
    for name, label in [('group', 'Filter group'), ('sex', 'Filter sex'), ('trial', 'Filter trial')]:
        combo = W.QComboBox(); combo.addItem('All', None)
        if name == 'group': combo.addItem('Training groups A + B', '__training_groups__')
        window.selection_filters[name] = combo
        form.addRow(label, combo)
        combo.currentIndexChanged.connect(window.update_selection_filter)
    actions = W.QHBoxLayout()
    for text, mode in [('Include matches', 'include'), ('Exclude matches', 'exclude'), ('Include only matches', 'only')]:
        button = W.QPushButton(text)
        button.clicked.connect(lambda checked=False, mode=mode: window.apply_selection_filter(mode))
        actions.addWidget(button)
    form.addRow(actions)
    window.selection_count = W.QLabel('No individuals loaded.'); window.selection_count.setWordWrap(True)
    form.addRow(window.selection_count)
    note = W.QLabel('Filters combine and show matching animals below. Apply a button to change inclusion. Include only matches excludes everyone else; the analysis Sex setting still applies.')
    note.setWordWrap(True); form.addRow(note)
