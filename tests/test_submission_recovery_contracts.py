"""Fake-scheduler checks for Linux submission and repeatable report collection."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

_spec = importlib.util.spec_from_file_location('submission_helpers', Path(__file__).with_name('test_hpc_submission.py'))
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def test_overlapping_submissions_schedule_one_chain(tmp_path):
    bash = helpers.bash_executable(require_flock=True)
    target = helpers.prepared_fixture(tmp_path)
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    scheduler = bin_dir/'sbatch'
    scheduler.write_text('''#!/bin/bash
set -euo pipefail
[[ ! -e /proc/$$/fd/9 ]] || { echo "lock descriptor leaked" >&2; exit 8; }
touch "$ENTERED"
sleep 0.15
count=0; [[ ! -f "$STATE" ]] || read -r count < "$STATE"
count=$((count+1)); echo "$count" > "$STATE"
echo "$((100000+count))"
''', newline='\n')
    scheduler.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
               ENTERED=str(tmp_path/'entered'), STATE=str(tmp_path/'state'))
    args = [bash,'--noprofile','--norc',str(target/'submit.sh')]
    first = subprocess.Popen(args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic()+10
        while not (tmp_path/'entered').exists() and first.poll() is None and time.monotonic()<deadline:
            time.sleep(0.01)
        assert (tmp_path/'entered').exists()
        second = subprocess.run(args,env=env,text=True,capture_output=True,timeout=10)
        output,error = first.communicate(timeout=10)
        assert first.returncode == 0, error
        assert second.returncode != 0 and 'Another submission command' in second.stderr
        assert len((target/'jobs.tsv').read_text().splitlines()) == 6
        assert (tmp_path/'state').read_text().strip() == '6'
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate()


def test_collector_retains_two_successful_invocations(tmp_path):
    bash = helpers.bash_executable()
    bundle = tmp_path/'bundle'
    (bundle/'resources').mkdir(parents=True)
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    sacct = bin_dir/'sacct'
    sacct.write_text('#!/bin/bash\nprintf "JobID|State\\n1|COMPLETED\\n"\n',newline='\n')
    sacct.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'], SLURM_JOB_ID='123')
    args = [bash,'--noprofile','--norc',str(helpers.TEMPLATES/'collect.sbatch'),
            helpers.bash_path(helpers.PACKAGE), helpers.bash_path(sys.executable),
            '/unused',helpers.bash_path(bundle),'1']
    for _ in range(2):
        result = subprocess.run(args,env=env,text=True,capture_output=True,timeout=30)
        assert result.returncode == 0, result.stderr
    assert len(list(bundle.glob('scheduler_report_*.psv'))) == 2
    assert len(list(bundle.glob('resource_summaries_*.csv'))) == 2
