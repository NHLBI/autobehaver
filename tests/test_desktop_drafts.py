import json
from tests.test_desktop_shared_project import gui
from tests.test_desktop_restore import record


def test_new_project_preserves_built_record(gui, tmp_path):
    app,w,errors=gui
    r=record(tmp_path,dict(kind='pipeline',dataset=str(tmp_path/'spd2'),run_name='spd2_final',start_at='features',speed='fast',moviechecker_mode='recording-only',allocations={}))
    p=tmp_path/'desktop_run.json';p.write_text(json.dumps(r));original=p.read_bytes()
    w.restore_run(r);w.tabs.setCurrentIndex(0)
    assert w.record_matches_form()
    w.fields['dataset'].setText(str(tmp_path/'trpa1'))
    w.fields['pipeline_name'].setText('trpa1_final');w.update_buttons()
    assert not w.send_button.isEnabled()
    w.save_run()
    assert p.read_bytes()==original and w.record is None
    assert json.loads(w.draft_path.read_text())['name']=='trpa1_final'
    assert not w.submit_button.isEnabled() and not errors


def test_new_run_and_draft_roundtrip(gui,tmp_path):
    app,w,errors=gui
    w.fields['host'].setText('test-host')
    w.fields['dataset'].setText(str(tmp_path/'trpa1'))
    w.fields['pipeline_name'].setText('trpa1_final');w.save_run();p=w.draft_path
    w.fields['features_ram'].setValue(28);w.save_run();assert w.draft_path==p
    w.fields['pipeline_name'].setText('second');w.save_run();assert w.draft_path!=p
    w.new_run()
    assert w.record is None and w.draft_path is None
    assert w.fields['dataset'].text()=='' and w.fields['host'].text()=='test-host'
    w.restore_draft(json.loads(p.read_text()),p);w.save_settings()
    from autobehaver.desktop import make_window
    other=make_window(w.state_dir)
    assert other.draft_path==p and other.fields['features_ram'].value()==28
    assert other.fields['pipeline_name'].text()=='trpa1_final'
    assert other.record is None and not other.submit_button.isEnabled()
    other.close();assert not errors
