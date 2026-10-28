"""Collapsible settings sections with visible overrides and local reset actions."""
from .desktop_settings import specifications, advanced_specs


def section(window, parent_form, title, keys, W, content=None):
    from PySide6 import QtCore
    outer = W.QWidget()
    layout = W.QVBoxLayout(outer); layout.setContentsMargins(0, 0, 0, 0)
    layout.setSizeConstraint(W.QLayout.SetMinimumSize)
    toggle = W.QToolButton(); toggle.setCheckable(True)
    toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    layout.addWidget(toggle)
    body = W.QWidget(); body_layout = W.QVBoxLayout(body)
    body_layout.setSizeConstraint(W.QLayout.SetMinimumSize)
    if content is not None:
        content.layout().setSizeConstraint(W.QLayout.SetMinimumSize)
        body_layout.addWidget(content)
    reset = W.QPushButton('Reset this section to defaults'); body_layout.addWidget(reset)
    layout.addWidget(body); body.hide()
    specs = specifications()
    def refresh(expand=False):
        values = window.values()
        count = sum(values.get(k) != specs[k]['default'] for k in keys)
        toggle.setText(f'Advanced: {title} ({count} changed)')
        if expand and count: toggle.setChecked(True)
        toggle.setArrowType(QtCore.Qt.DownArrow if toggle.isChecked() else QtCore.Qt.RightArrow)
        body.setVisible(toggle.isChecked())
        if content is not None:
            content.ensurePolished()
            content.layout().activate()
            content.updateGeometry()
        body_layout.invalidate(); body_layout.activate()
        outer.setMinimumHeight(toggle.sizeHint().height() +
            (layout.spacing() + body.sizeHint().height() if toggle.isChecked() else 0))
        outer.updateGeometry()
        parent_form.invalidate()
    def restore():
        window.apply_values({k: specs[k]['default'] for k in keys}, migrate=False)
        refresh()
    reset.clicked.connect(restore)
    toggle.toggled.connect(lambda _: refresh())
    for key in keys:
        widget = window.fields[key]
        signal = (widget.textChanged if isinstance(widget,W.QLineEdit) else
                  widget.currentIndexChanged if isinstance(widget,W.QComboBox) else
                  widget.toggled if isinstance(widget,W.QCheckBox) else widget.valueChanged)
        signal.connect(lambda *_: refresh())
    window.advanced_sections.append(refresh)
    parent_form.addRow(outer)
    refresh()
    return outer


def add_controls(window, parent_form, workflow, W):
    from PySide6 import QtCore
    class DecimalBox(W.QDoubleSpinBox):
        def textFromValue(self, value):
            return format(value, '.10f').rstrip('0').rstrip('.').replace('.', self.locale().decimalPoint())
    specs = advanced_specs()
    groups = list(dict.fromkeys(s['group'] for s in specs.values() if s['workflow']==workflow))
    for group in groups:
        body = W.QWidget(); form = W.QFormLayout(body)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        keys = [k for k,s in specs.items() if s['group']==group]
        for key in keys:
            spec = specs[key]
            if spec['type']=='boolean':
                widget = W.QCheckBox(); widget.setChecked(spec['default'])
            else:
                widget = DecimalBox() if spec['type']=='number' else W.QSpinBox()
                if spec['type']=='number':
                    widget.setDecimals(10)
                    widget.setSingleStep(.000001 if 'VARIANCE' in key else .001 if 'noise' in key or 'learning_rate' in key else .01)
                widget.setRange(spec['minimum'],spec['maximum']); widget.setValue(spec['default'])
            widget.setToolTip(spec['help'] + '\nDefault: ' + str(spec['default']))
            window.fields[key] = widget
            form.addRow(spec['label'], widget)
        section(window,parent_form,group,keys,W,body)


def apply_shared_defaults(window):
    """The contract, rather than widget construction, owns defaults and limits."""
    from PySide6 import QtWidgets as W
    specs = specifications()
    missing = window.fields.keys() - specs.keys()
    if missing: raise ValueError('Missing launcher specifications: '+', '.join(sorted(missing)))
    for key,widget in window.fields.items():
        spec = specs[key]
        if isinstance(widget, W.QAbstractSpinBox): widget.setRange(spec['minimum'],spec['maximum'])
        if isinstance(widget,W.QComboBox) and 'choices' in spec:
            labels = {widget.itemData(i):widget.itemText(i) for i in range(widget.count())}
            widget.blockSignals(True)
            widget.clear()
            for value in spec['choices']: widget.addItem(labels.get(value,str(value)),value)
            widget.blockSignals(False)
        if not widget.toolTip(): widget.setToolTip(spec['help'])
    window.apply_values({k:s['default'] for k,s in specs.items()}, migrate=False)
