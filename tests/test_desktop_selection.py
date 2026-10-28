import csv
from pathlib import Path
from tests.test_desktop_shared_project import gui, until


def test_trial_metadata_and_combined_filters():
    from autobehaver.desktop_selection import trial_for_row, matches
    assert trial_for_row({'individual':'WT_MALE_rpi-1_trial12_chamber3'}) == '12'
    assert trial_for_row({'individual':'unrecognized','trial':'trial3'}) == '3'
    assert trial_for_row({'individual':'unrecognized'}) == ''
    row={'individual':'WT_MALE_trial1_chamber2','group':'WT','sex':'MALE'}
    assert matches(row,training_groups={'WT','mutant'},sex='male',trial='1')
    assert not matches(row,group='mutant')
    assert not matches(row,trial='2')


def test_bulk_selection_preserves_manual_choices_and_refresh(gui,tmp_path):
    from PySide6.QtCore import Qt
    app,w,errors=gui
    project=tmp_path/'project';path=project/'8_master_matrix/MASTER_FEATURE_MATRIX.csv'
    path.parent.mkdir(parents=True)
    rows=[['WT_MALE_trial1_chamber1','WT','MALE',1],['WT_FEMALE_trial2_chamber1','WT','FEMALE',2],
          ['MUT_MALE_trial2_chamber1','MUT','MALE',3],['other_MALE_trial1_chamber1','other','MALE',4]]
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['individual','group','sex','feature']);writer.writerows(rows)
    w.fields['dataset'].setText(str(project));w.tabs.setCurrentIndex(1)
    until(app,lambda:w.loaded_matrix==str(path) and not w.busy)
    def choose(combo,value):combo.setCurrentIndex(combo.findData(value))
    choose(w.fields['group_a'],'WT');choose(w.fields['group_b'],'MUT')
    choose(w.selection_filters['group'],'__training_groups__')
    w.apply_selection_filter('only')
    included=lambda:[i for i in range(4) if w.table.item(i,0).checkState()==Qt.Checked]
    assert included()==[0,1,2] and w.table.isRowHidden(3)
    choose(w.selection_filters['sex'],'MALE');choose(w.selection_filters['trial'],'2')
    w.apply_selection_filter('exclude');assert included()==[0,1]
    w.apply_selection_filter('include');assert included()==[0,1,2]
    w.table.item(0,0).setCheckState(Qt.Unchecked);assert included()==[1,2]
    w.load_matrix();until(app,lambda:not w.busy)
    assert included()==[1,2] and w.selection_filters['trial'].currentData()=='2'
    assert w.table.item(2,4).text()=='2'
    w.apply_selection_filter('only');assert included()==[2]
    assert not errors


def test_wheel_scrolls_tab_without_changing_even_focused_controls(gui):
    from PySide6 import QtCore,QtGui,QtWidgets as W
    app,w,errors=gui
    w.resize(650,600);w.show();w.tabs.setCurrentIndex(1);app.processEvents()
    scroll=w.tabs.currentWidget();bar=scroll.verticalScrollBar()
    assert bar.maximum()>0
    controls=[w.fields['fi_memory'],w.fields['sex'],w.selection_filters['group'],w.fields['fi_memory'].lineEdit()]
    for control in controls:
        control.setFocus();bar.setValue(0)
        before=(w.fields['fi_memory'].value(),w.fields['sex'].currentIndex(),w.selection_filters['group'].currentIndex())
        point=control.rect().center()
        event=QtGui.QWheelEvent(QtCore.QPointF(point),QtCore.QPointF(control.mapToGlobal(point)),
            QtCore.QPoint(),QtCore.QPoint(0,-120),QtCore.Qt.NoButton,QtCore.Qt.NoModifier,QtCore.Qt.ScrollUpdate,False)
        W.QApplication.sendEvent(control,event)
        assert before==(w.fields['fi_memory'].value(),w.fields['sex'].currentIndex(),w.selection_filters['group'].currentIndex())
        assert bar.value()>0
    assert not errors
