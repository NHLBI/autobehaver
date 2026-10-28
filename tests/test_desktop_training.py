import json
from pathlib import Path
import shlex
import time
import os
import shutil
import subprocess

import pytest

from autobehaver import desktop_jobs as jobs
from autobehaver.desktop_training import option_arguments, prepare_training
from autobehaver.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path):
    data = tmp_path/'data'; dataset = data/'training'; dataset.mkdir(parents=True)
    (data/'bundles').mkdir()
    for name in ('a.h5', 'b.h5', 'a.labels.h5', 'b.labels.h5'):
        (dataset/name).write_bytes(b'input fixture; heavy HDF5 checks belong to native training')
    metadata = dataset/'metadata.csv'
    metadata.write_text('filename,labels,set\na.h5,a.labels.h5,train\nb.h5,b.labels.h5,val\n')
    return dict(kind='training',run_name='train_example',dataset=str(dataset),
                training_directory=str(dataset),training_metadata=str(metadata),
                data_local=str(data),data_remote='/data/lab',jobs_local=str(data/'bundles'),
                jobs_remote='/data/lab/bundles',python='/data/env/bin/python',training_options={},
                train_cpus=16,train_memory=32,training_mode='supervised')


def test_training_build_publish_monitor_and_duplicate_guard(tmp_path):
    c=config(tmp_path)
    r=jobs.build_job(c,ROOT,tmp_path/'local')
    plan=json.loads((Path(r['prepared'])/'plan.json').read_text())
    args=build_parser().parse_args(plan['command'][3:])
    assert args.directory=='/data/lab/training'
    assert args.output_directory==r['remote']+'/results'
    assert args.metadata_path==r['remote']+'/training_metadata.csv'
    assert args.batch==1024 and args.nhead==32 and args.num_workers==14
    assert not Path(c['dataset'],'Checkpoints').exists()
    assert r['job_counts']=={'training':1}
    published=jobs.publish(r)
    assert jobs.verify_submission(published)=='bash '+shlex.quote(r['remote']+'/submit.sh')
    (Path(r['mounted'])/'jobs.tsv').write_text('training\t123\n')
    assert jobs.submission_button_state(published)==('3  Submitted',False)
    with pytest.raises(ValueError,match='accepted jobs'): jobs.verify_submission(published)
    from autobehaver.desktop_monitor import snapshot
    class Session:
        connected=True
        def query(self,command): return '123|RUNNING|0:0\n__QUEUE__\n123|RUNNING|None\n'
    assert snapshot(published,Session())['rows'][0]['stage']=='training'
    assert snapshot(published,Session())['rows'][0]['category']=='running'


@pytest.mark.parametrize('mode,flag',[('pretrain','--pretrain_mode'),('lr_finder','--lr_finder_mode')])
def test_modes_and_external_output(tmp_path,mode,flag):
    c=config(tmp_path); c.update(training_mode=mode,training_output=str(tmp_path/'data'/'models'))
    if mode=='pretrain': Path(c['training_metadata']).write_text('filename\na.h5\nb.h5\n')
    p=prepare_training(c,tmp_path/'job','/data/jobs/unique','/data/lab/training')
    assert flag in p['command'] and p['output']=='/data/lab/models/unique'


def test_checkpoint_and_column_list_frozen(tmp_path):
    c=config(tmp_path)
    checkpoint=Path(c['dataset'])/'prior.pth'; checkpoint.write_bytes(b'checkpoint')
    columns=Path(c['dataset'])/'columns.txt'; columns.write_text('feature1\n')
    c.update(training_checkpoint=str(checkpoint),training_columns=str(columns),
             training_options={'restart_optimizer':True,'nhead':'16','random_seed':'7'})
    p=prepare_training(c,tmp_path/'job','/data/jobs/unique','/data/lab/training')
    assert p['checkpoint'][0]['sha256']==jobs.digest(checkpoint)
    assert (tmp_path/'job/feature_columns.txt').read_text()=='feature1\n'
    args=build_parser().parse_args(p['command'][3:])
    assert args.restart_optimizer and args.nhead==16 and args.random_seed==7


@pytest.mark.parametrize('change,match',[
    ({'train_cpus':1},'workers'),
    ({'train_partition':'gpu\n#SBATCH --mem=100g'},'partition'),
    ({'train_gres':'gpu:a100:2'},'one GPU'),
    ({'train_time':'00:00:00'},'Wall time'),
    ({'training_options':{'nhead':3}},'divisible'),
    ({'training_options':{'base_lr':'nan'}},'finite'),
    ({'training_options':{'restart_optimizer':True}},'checkpoint'),
    ({'training_options':{'swa_mode':True,'num_epochs':5}},'SWA'),
])
def test_invalid_training_config(tmp_path,change,match):
    c=config(tmp_path);c.update(change)
    with pytest.raises(ValueError,match=match): prepare_training(c,tmp_path/'job','/data/jobs/unique','/data/lab/training')


@pytest.mark.parametrize('body',[
    'filename,labels,set\n../outside.h5,a.labels.h5,train\n',
    'filename,labels,set\na.h5,missing.h5,train\n',
    'filename,labels,set\na.h5,a.labels.h5,train\na.h5,a.labels.h5,val\n',
    'filename,labels,set\na.h5,a.labels.h5,train\n',
])
def test_bad_metadata_rejected(tmp_path,body):
    c=config(tmp_path);Path(c['training_metadata']).write_text(body)
    with pytest.raises(ValueError): prepare_training(c,tmp_path/'job','/data/jobs/unique','/data/lab/training')


def test_option_values_preserve_negative_flags_and_native_defaults():
    argv,args=option_arguments({'pose_pred':False,'defined_split':False,'noval':True,
                               'ablation_exclude_columns':'a,b','random_seed':''})
    assert '--no-pose-pred' in argv and '--no-defined-split' in argv
    assert not args.pose_pred and not args.defined_split and args.noval
    assert args.random_seed == 7 and args.ablation_exclude_columns==['a,b']


def test_native_cli_uses_frozen_metadata_and_separate_outputs(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from autobehaver import cli, lr_finder, pretraining
    from autobehaver import io as training_io
    dataset=tmp_path/'inputs';dataset.mkdir()
    metadata=tmp_path/'frozen.csv';metadata.write_text('filename\na.h5\n')
    captured={}
    def finder(**kwargs): captured.update(kwargs)
    monkeypatch.setattr(lr_finder,'find_lr',finder)
    common=['--directory',str(dataset),'--metadata_path',str(metadata),'--num_workers','0',
            '--ablation_preset','none','--random_seed','7']
    output=tmp_path/'lr_result'
    cli.main(common+['--lr_finder_mode','--output_directory',str(output)])
    assert captured['metadata']['filename'].tolist()==['a.h5']
    assert captured['args'].output_directory==str(output)
    assert (output/'Records').is_dir() and not (dataset/'Records').exists()
    monkeypatch.setattr(training_io,'get_headers_from_first_hdf5',lambda _:['feature1'])
    calls=[]
    monkeypatch.setattr(pretraining,'run_pretraining',lambda *a,**k:calls.append((a,k)))
    output=tmp_path/'pretrain_result'
    cli.main(common+['--pretrain_mode','--output_directory',str(output)])
    assert calls[0][0][8]==str(output/'Checkpoints')
    assert Path(calls[0][0][15]).parent==output/'Records'
    assert not (dataset/'Checkpoints').exists()


def test_gui_training_buttons_and_persistence(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from autobehaver.desktop import make_window
    app=QApplication.instance() or QApplication([])
    w=make_window(tmp_path/'state');c=config(tmp_path)
    c.update(training_name='train_example',train_opt_num_epochs='12',train_opt_cosine_decay_epochs='11',train_opt_nhead='16',host='hpc.example.org')
    w.apply_values(c);w.tabs.setCurrentWidget(w.training_page);w.update_buttons()
    errors=[];monkeypatch.setattr(w,'error',errors.append)
    def wait():
        deadline=time.monotonic()+35
        while w.busy and time.monotonic()<deadline: app.processEvents();time.sleep(.01)
        assert not w.busy and not errors, errors
    w.build_button.click();wait()
    assert w.record['kind']=='training' and w.tabs.currentWidget() is w.review_page
    args=build_parser().parse_args(w.record['training']['command'][3:])
    assert args.num_epochs==12 and args.nhead==16
    w.send_button.click();wait()
    class Session:
        connected=True
        host='hpc.example.org'
        username=''
        def execute(self,command,output):
            assert command==jobs.verify_submission(w.record)
            (Path(w.record['mounted'])/'jobs.tsv').write_text('training\t456\n')
            output('Training accepted\n');return 0
        def close(self): self.connected=False
    w.session=Session();w.update_buttons();w.submit_button.click();wait()
    assert not w.submit_button.isEnabled() and 'Submitted' in w.submit_button.text()
    assert not w.recovery_button.isEnabled()
    w.save_settings();w.close()
    restored=make_window(tmp_path/'state')
    assert restored.fields['train_opt_nhead'].text()=='16'
    assert restored.record['kind']=='training'
    restored.close()


def test_generated_submission_script_local_scheduler(tmp_path):
    bash=shutil.which('bash') if os.name!='nt' else 'C:/Program Files/Git/bin/bash.exe'
    if not bash or not Path(bash).exists(): pytest.skip('Bash unavailable')
    def posix(path):
        text=Path(path).resolve().as_posix()
        return '/'+text[0].lower()+text[2:] if os.name=='nt' else text
    c=config(tmp_path);target=tmp_path/'job';remote=posix(target)
    prepare_training(c,target,remote,'/data/lab/training')
    for name in ('submit.sh','training.sbatch'):
        subprocess.run([bash,'-n',posix(target/name)],capture_output=True,check=True)
    fake=tmp_path/'bin';fake.mkdir()
    # This test exercises submission/receipt guards, not OS advisory locking.
    for name,text in [('flock','#!/bin/bash\nexit 0\n'),
                      ('sbatch','#!/bin/bash\necho called >> "$TRACE"\nprintf "123;cluster\\n"\n')]:
        path=fake/name;path.write_text(text);path.chmod(0o755)
    env=dict(os.environ,PATH=posix(fake)+os.pathsep+os.environ['PATH'],TRACE=posix(tmp_path/'trace'))
    # Windows converts PATH separators on entry to Git Bash.
    if os.name=='nt': env['PATH']=str(fake)+os.pathsep+os.environ['PATH']
    def submit(): return subprocess.run([bash,posix(target/'submit.sh')],env=env,capture_output=True,text=True)
    assert submit().returncode!=0 and not (tmp_path/'trace').exists()
    (target/'PUBLICATION_RECEIPT.json').write_text('{"verified":true}')
    original=(target/'training_metadata.csv').read_bytes()
    (target/'training_metadata.csv').write_bytes(original+b'changed')
    assert submit().returncode!=0 and not (tmp_path/'trace').exists()
    (target/'training_metadata.csv').write_bytes(original)
    result=submit();assert result.returncode==0,result.stderr
    assert (target/'jobs.tsv').read_text()=='training\t123\n'
    assert submit().returncode!=0
    assert (tmp_path/'trace').read_text().splitlines()==['called']
