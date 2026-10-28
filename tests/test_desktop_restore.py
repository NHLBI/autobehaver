import json
from pathlib import Path
from tests.test_desktop_shared_project import gui, until, matrix


def record(tmp_path,config):
    return {'format':1,'kind':config['kind'],'name':'saved','config':config,
        'root':str(tmp_path),'prepared':str(tmp_path),'mounted':str(tmp_path/'mounted'),
        'remote':'/data/jobs/saved','project':'/data/project','published':False,'job_counts':{},'build_hashes':{},'training':{'mode':'train','metadata_rows':2,'output':'output','command':['train']}}


def test_built_recovery_can_be_sent_without_reopening(gui, tmp_path, monkeypatch):
    app, w, errors = gui
    cfg = dict(kind='pipeline', dataset=str(tmp_path/'project'),
        pipeline_name='original', run_name='recovery_original', start_at='crop',
        speed='fast', moviechecker_mode='recording-only',
        allocations={'mmgen':[12,80], 'mmgen_finish':[12,80]},
        recovery={'replace_keys':['mmgen_000','mmgen_finish'], 'retained':{'posepred_000':'123'}})
    r = record(tmp_path, cfg)
    r['job_counts'] = {'mmgen':1, 'mmgen_finish':1, 'collect':1}
    w.fields['pipeline_name'].setText('original')
    w.tabs.setCurrentWidget(w.monitor_page)
    w.built(r)
    assert w.tabs.currentWidget() is w.review_page
    assert w.fields['pipeline_name'].text() == 'recovery_original'
    assert w.record_matches_form() and w.send_button.isEnabled()
    assert 'Start: crop' not in w.summary.toPlainText()
    assert 'Earlier successful stages will not rerun' in w.summary.toPlainText()
    calls = []
    monkeypatch.setattr(w, 'work', lambda *args: calls.append(args))
    w.send_button.click()
    assert len(calls) == 1 and 'Sending bundle' in calls[0][0]
    assert not errors


def test_open_saved_pipeline_restores_old_values_without_migration(gui,tmp_path,monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    app,w,errors=gui
    cfg=dict(kind='pipeline',dataset=str(tmp_path/'spd2'),run_name='spd2_final',
        start_at='features',speed='fast',moviechecker_mode='recording-only',features_cpu=4,features_ram=64,feature_workers=2,
        mmgen_cpu=12,mmgen_ram=120,allocations={'features':[4,64],'mmgen':[12,120]},
        batch_sizes={'crop':4,'dlc':20,'features':16,'posepred':20,'mmgen':40},
        group_a=None,group_b=None,control_group=None,fi_memory=192)
    r=record(tmp_path,cfg);path=tmp_path/'desktop_run.json';path.write_text(json.dumps(r))
    (tmp_path/'templates').mkdir();(tmp_path/'templates/mmgen.sbatch').write_text('#SBATCH --time=18:00:00\n')
    w.fields['pipeline_name'].setText('uba5_final');w.fields['features_cpu'].setValue(10)
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a: (str(path),''))
    original=path.read_bytes();w.open_run()
    v=w.values()
    assert v['pipeline_name']=='spd2_final' and v['dataset']==cfg['dataset']
    assert (v['features_cpu'],v['feature_workers'],v['fi_memory'])==(4,2,192)
    assert v['mmgen_ram']==v['mmgen_finish_ram']==120
    assert v['mmgen_hours']==v['mmgen_finish_hours']==18
    assert v['use_saved_batch_sizes'] and v['saved_batch_sizes']==cfg['batch_sizes']
    assert not v['fi_sharded']
    assert w.tabs.currentWidget() is w.monitor_page
    assert path.read_bytes()==original and not errors


def test_featureimp_saved_groups_and_exact_inclusions_restore(gui,tmp_path):
    from PySide6.QtCore import Qt
    app,w,errors=gui
    project=tmp_path/'project';path=matrix(project)
    cfg=dict(kind='featureimp',dataset=str(project),fi_dataset=str(project),matrix=str(path),
        fi_override=False,run_name='saved_comparison',group_a='control',group_b='mutant',control_group='control',
        selected_identities=[['b','mutant','FEMALE']],sex='all',fi_sharded=True,fi_shards=3,
        fi_finish_memory=48,seed=21,pca=False,fsea=False)
    w.restore_run(record(tmp_path,cfg));w.tabs.setCurrentIndex(1)
    until(app,lambda:w.loaded_matrix==str(path) and not w.busy)
    assert w.fields['group_a'].currentData()=='control'
    assert w.fields['group_b'].currentData()=='mutant'
    assert w.fields['fi_name'].text()=='saved_comparison'
    assert w.fields['fi_finish_memory'].value()==48
    assert w.table.item(0,0).checkState()==Qt.Unchecked
    assert w.table.item(1,0).checkState()==Qt.Checked
    w.load_matrix();until(app,lambda:not w.busy)
    assert w.table.item(0,0).checkState()==Qt.Unchecked
    assert not errors


def test_training_nested_options_and_absent_fields_reset(gui,tmp_path):
    app,w,errors=gui
    w.fields['fi_shards'].setValue(99)
    cfg=dict(kind='training',dataset=str(tmp_path/'train'),run_name='my_training',
        train_cpus=6,train_memory=24,training_options={'n_hid_units':16})
    w.restore_run(record(tmp_path,cfg))
    assert w.fields['training_name'].text()=='my_training'
    assert w.fields['training_directory'].text()==cfg['dataset']
    assert w.fields['fi_shards'].value()==8
    assert w.fields['train_cpus'].value()==6 and not errors
