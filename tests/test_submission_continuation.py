"""Exercise real shell control flow with fake scheduler tools (no cluster calls)."""
import importlib.util
import os
from pathlib import Path
import subprocess


def helpers():
    path = Path(__file__).with_name('test_hpc_submission.py')
    spec = importlib.util.spec_from_file_location('continuation_helpers', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def scheduler(tmp_path, h, accounting):
    fake = tmp_path / 'bin'; fake.mkdir()
    scripts = {
        'flock': '#!/bin/bash\nexit 0\n',  # Scheduling tests only; Linux lock behavior is tested separately.
        'sleep': '#!/bin/bash\nexit 0\n',
        'sacct': '#!/bin/bash\ncat "$ACCOUNTING"\n',
        'sbatch': '''#!/bin/bash
n=0; [[ ! -f "$STATE" ]] || read -r n < "$STATE"
n=$((n+1)); echo "$n" > "$STATE"
printf '%s\\t' "$@" >> "$CALLS"; printf '\\n' >> "$CALLS"
[[ "$n" != "${FAIL_AT:-0}" ]] || { echo 'simulated rejection' >&2; exit 1; }
echo "$((200+n))"
'''}
    for name, text in scripts.items():
        path = fake / name; path.write_text(text, newline='\n'); path.chmod(0o755)
    return dict(os.environ, PATH=str(fake)+os.pathsep+os.environ['PATH'],
        ACCOUNTING=h.bash_path(accounting), STATE=h.bash_path(tmp_path/'state'), CALLS=h.bash_path(tmp_path/'calls'))


def test_partial_recovery_continues_own_claims_without_duplicate_jobs(tmp_path):
    h = helpers(); target = h.prepared_fixture(tmp_path)
    parent = tmp_path / 'parent'; parent.mkdir()
    (target/'recovery_parent.txt').write_text(h.bash_path(parent)+'\n', newline='\n')
    (target/'recovery_expected.tsv').write_text('crop_bad\t101\treplace\ndlc_good\t102\tretain\nmmgen\t103\treplace\n', newline='\n')
    (target/'schedule.tsv').write_text('crop_bad\tcrop\t-\t-\t0\t1\nmmgen\tmmgen\tcrop_bad,dlc_good\t-\t0\t1\ncollect\tcollect\tcrop_bad,dlc_good,mmgen\t-\t0\t1\n', newline='\n')
    accounting = tmp_path/'accounting'
    accounting.write_text('101|FAILED|\n102|RUNNING|\n103|CANCELLED|\n201|RUNNING|\n203|PENDING|\n204|PENDING|\n', newline='\n')
    env = scheduler(tmp_path, h, accounting); env['FAIL_AT'] = '2'
    command = [h.bash_executable(), '--noprofile', '--norc', str(target/'submit.sh')]
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert first.returncode != 0 and 'Submission stopped' in first.stderr
    assert (target/'jobs.tsv').read_text() == 'crop_bad\t201\n'
    second = subprocess.run(command+['--resume'], env=env, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr
    accepted = (target/'jobs.tsv').read_text().splitlines()
    assert accepted == ['crop_bad\t201', 'mmgen\t203', 'collect\t204']
    calls = (tmp_path/'calls').read_text().splitlines()
    assert len(calls) == 4 and '--dependency=afterok:201:102' in calls[2]
    assert not any('scancel' in call for call in calls)
    third = subprocess.run(command+['--resume'], env=env, capture_output=True, text=True, timeout=30)
    assert third.returncode == 0, third.stderr
    assert len((tmp_path/'calls').read_text().splitlines()) == 4
    # A foreign recovery claim must still block even when this bundle has IDs.
    with (parent/'recovery_claims.tsv').open('a', newline='\n') as stream:
        stream.write('crop_bad\t101\t999\t/data/foreign\n')
    raced = subprocess.run(command+['--resume'], env=env, capture_output=True, text=True, timeout=30)
    assert raced.returncode != 0 and 'already exists' in raced.stderr
    assert len((tmp_path/'calls').read_text().splitlines()) == 4


def test_featureimp_recovery_rechecks_failure_and_claims(tmp_path):
    from autobehaver import desktop_jobs as jobs
    from autobehaver.desktop_recovery import recovery_config
    h = helpers()
    path = Path(__file__).with_name('test_desktop_jobs.py')
    spec = importlib.util.spec_from_file_location('fi_recovery_fixture', path)
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    config = fixture.setup_config(tmp_path)
    parent_dir = tmp_path/'original'
    jobs._featureimp(config, parent_dir, h.bash_path(parent_dir), '/data/project')
    parent = dict(kind='featureimp', prepared=str(parent_dir), mounted=str(parent_dir),
        remote=h.bash_path(parent_dir), root=str(parent_dir), config=config)
    snap = dict(source='Live scheduler status', updated='now', rows=[dict(key='featureimp',job_id='101',state='FAILED')])
    recovery = recovery_config(parent, snap)
    target = tmp_path/'retry'
    jobs._featureimp(recovery, target, h.bash_path(target), '/data/project')
    (target/'PUBLICATION_RECEIPT.json').write_text('{}')
    accounting = tmp_path/'accounting'; accounting.write_text('101|RUNNING|\n', newline='\n')
    env = scheduler(tmp_path, h, accounting)
    command = [h.bash_executable(), '--noprofile', '--norc', str(target/'submit.sh')]
    active = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert active.returncode != 0 and 'No jobs submitted' in active.stderr
    assert not (tmp_path/'calls').exists()
    accounting.write_text('101|FAILED|\n', newline='\n')
    failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert failed.returncode == 0, failed.stderr
    assert (target/'jobs.tsv').read_text() == 'featureimp\t201\n'
    assert 'featureimp\t101\t201\t' in (parent_dir/'recovery_claims.tsv').read_text()
    # Simulate a second independent bundle trying to replace the same failed job.
    (target/'jobs.tsv').unlink()
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0 and 'replacement already exists' in repeated.stderr
    assert len((tmp_path/'calls').read_text().splitlines()) == 1


def test_incremental_generation_guard_allows_only_accounting_after_commit(tmp_path):
    h = helpers(); target = h.prepared_fixture(tmp_path)
    receipt = tmp_path/'receipt.json'; receipt.write_text('{}')
    (target/'incremental_receipt.sha256').write_text('0'*64+'  '+h.bash_path(receipt)+'\n', newline='\n')
    accounting = tmp_path/'accounting'; accounting.write_text('')
    env = scheduler(tmp_path, h, accounting)
    command = [h.bash_executable(), '--noprofile', '--norc', str(target/'submit.sh')]
    rejected = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and 'generation changed' in rejected.stderr
    assert not (tmp_path/'calls').exists()
    entries = [(line.split('\t')[0], str(i+101)) for i,line in enumerate((target/'schedule.tsv').read_text().splitlines()) if line.split('\t')[1] != 'collect']
    (target/'jobs.tsv').write_text(''.join(f'{key}\t{job}\n' for key,job in entries), newline='\n')
    accounting.write_text(''.join(f'{job}|COMPLETED|\n' for key,job in entries), newline='\n')
    resumed = subprocess.run(command+['--resume'], env=env, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    calls = (tmp_path/'calls').read_text().splitlines()
    assert len(calls) == 1 and 'collect.sbatch' in calls[0]
