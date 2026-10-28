import json
from pathlib import Path
import pytest
from autobehaver.desktop_recovery import recovery_config


def fixture(tmp_path):
    (tmp_path/'batches').mkdir()
    jobs = [dict(key='crop_0',stage='crop',depends_on=[],file_list='crop_0.txt',shard_index=0,num_shards=1),
            dict(key='crop_1',stage='crop',depends_on=[],file_list='crop_1.txt',shard_index=0,num_shards=1),
            dict(key='dlc_0',stage='dlc',depends_on=['crop_0'],file_list='dlc_0.txt',shard_index=0,num_shards=1),
            dict(key='dlc_1',stage='dlc',depends_on=['crop_1'],file_list='dlc_1.txt',shard_index=0,num_shards=1),
            dict(key='mmgen',stage='mmgen',depends_on=['dlc_0','dlc_1'],file_list='all.txt',shard_index=0,num_shards=1)]
    (tmp_path/'plan.json').write_text(json.dumps({'jobs':jobs}))
    for job in jobs: (tmp_path/'batches'/job['file_list']).write_text('one\n')
    parent=dict(kind='pipeline',prepared=str(tmp_path),root=str(tmp_path),mounted=str(tmp_path),remote='/data/bundle',name='original',config={})
    states=['FAILED','COMPLETED','CANCELLED','RUNNING','CANCELLED']
    snap=dict(source='Live scheduler status',updated='now',rows=[dict(key=j['key'],state=s,job_id=str(i+101)) for i,(j,s) in enumerate(zip(jobs,states))])
    return parent,snap


def test_recovery_retains_running_jobs_and_original_batch_lists(tmp_path):
    parent,snap=fixture(tmp_path)
    config=recovery_config(parent,snap)
    jobs,lists=config['_prepared_plan']
    assert {j['key'] for j in jobs['jobs'] if not j.get('retained')} == {'crop_0','dlc_0','mmgen','collect'}
    assert config['recovery']['retained']['dlc_1']['job_id']=='104'
    assert lists['crop_0.txt']==['one']
    assert next(j for j in jobs['jobs'] if j['key']=='mmgen')['depends_on']==['dlc_0','dlc_1']


def test_no_cancellation_or_unknown_state_guessing(tmp_path):
    parent,snap=fixture(tmp_path)
    snap['rows'][2]['state']='PENDING'
    with pytest.raises(ValueError,match='Wait for scheduler cancellation'):
        recovery_config(parent,snap)
    snap['rows'][2]['state']='CANCELLED'; snap['rows'][3]['state']='Unknown'
    with pytest.raises(ValueError,match='Unknown scheduler state'):
        recovery_config(parent,snap)


def test_stale_attempt_and_offline_snapshot_blocked(tmp_path):
    parent,snap=fixture(tmp_path)
    (tmp_path/'recovery_claims.tsv').write_text('crop_0\t101\t201\t/data/new\n')
    with pytest.raises(ValueError,match='replacement already exists'):
        recovery_config(parent,snap)
    snap['source']='Saved snapshot'
    with pytest.raises(ValueError,match='Refresh live'):
        recovery_config(parent,snap)


def test_native_recovery_rewires_only_failed_jobs_and_checks_races(tmp_path):
    import importlib.util, os, subprocess
    spec=importlib.util.spec_from_file_location('recovery_helpers',Path(__file__).with_name('test_hpc_submission.py'))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    target=helpers.prepared_fixture(tmp_path)
    parent=tmp_path/'original';parent.mkdir()
    (target/'recovery_parent.txt').write_text(helpers.bash_path(parent)+'\n',newline='\n')
    (target/'recovery_expected.tsv').write_text('crop_bad\t101\treplace\ncrop_good\t102\tretain\ndlc_good\t104\tretain\ndlc_bad\t103\treplace\nmmgen\t105\treplace\n',newline='\n')
    (target/'schedule.tsv').write_text('crop_bad\tcrop\t-\t-\t0\t1\ndlc_bad\tdlc\tcrop_bad\t-\t0\t1\nmmgen\tmmgen\tdlc_bad,dlc_good\t-\t0\t1\ncollect\tcollect\tcrop_bad,crop_good,dlc_bad,dlc_good,mmgen\t-\t0\t1\n',newline='\n')
    fake=tmp_path/'bin';fake.mkdir()
    scripts={
        'flock':'#!/bin/bash\nexit 0\n',  # Test scheduling only; Linux flock itself is not emulated.
        'sleep':'#!/bin/bash\nexit 0\n',
        'sacct':'#!/bin/bash\ncat "$ACCOUNTING"\n',
        'sbatch': '\n'.join(['#!/bin/bash','n=0; [[ ! -f "$STATE" ]] || read -r n < "$STATE"',
            'n=$((n+1)); echo "$n" > "$STATE"',
            'printf "%s\\t" "$@" >> "$CALLS"; printf "\\n" >> "$CALLS"', 'echo "$((200+n))"',''])}
    for name,body in scripts.items():
        path=fake/name;path.write_text(body,newline='\n');path.chmod(0o755)
    accounting=tmp_path/'accounting';accounting.write_text('101|FAILED|\n102|COMPLETED|\n103|CANCELLED|\n104|RUNNING|\n105|CANCELLED|\n',newline='\n')
    env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'],ACCOUNTING=helpers.bash_path(accounting),STATE=helpers.bash_path(tmp_path/'state'),CALLS=helpers.bash_path(tmp_path/'calls'))
    command=[helpers.bash_executable(),'--noprofile','--norc',str(target/'submit.sh')]
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    calls=(tmp_path/'calls').read_text().splitlines()
    assert len(calls)==4
    assert '--dependency=afterok:201' in calls[1]
    assert '--dependency=afterok:202:104' in calls[2]
    assert not any('scancel' in call for call in calls)
    assert len((parent/'recovery_claims.tsv').read_text().splitlines())==3
    # Another recovery bundle cannot reclaim the same failed job IDs.
    (target/'jobs.tsv').unlink()
    second=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    assert second.returncode!=0 and 'already exists' in second.stderr
    assert len((tmp_path/'calls').read_text().splitlines())==4
    (parent/'recovery_claims.tsv').unlink()
    accounting.write_text('101|FAILED|\n102|COMPLETED|\n103|CANCELLED|\n104|FAILED|\n105|CANCELLED|\n',newline='\n')
    changed=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
    assert changed.returncode!=0 and 'rebuild recovery' in changed.stderr
    assert len((tmp_path/'calls').read_text().splitlines())==4
