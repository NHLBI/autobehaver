import json
from pathlib import Path
import pytest
from autobehaver.desktop_monitor import accepted_jobs, parse_scheduler, scheduler_command, snapshot, stage_counts


def test_states_steps_queue_override_and_unknown():
    rows = parse_scheduler('mesg: warning\n1|COMPLETED|0:0\n1.batch|FAILED|1:0\n'
                           '2|RUNNING|0:0\n3|COMPLETED|1:0\n4|OUT_OF_MEMORY|0:9\n'
                           '__QUEUE__\n2|PENDING|DependencyNeverSatisfied\n5|RUNNING|node01\n')
    assert rows['1']['category'] == 'finished'
    assert rows['2']['category'] == 'failed'
    assert rows['3']['category'] == 'failed'
    assert rows['4']['category'] == 'failed'
    assert rows['5']['category'] == 'running'
    assert '1.batch' not in rows


def test_snapshot_requires_merge_and_preserves_missing_state(tmp_path):
    plan = {'jobs': [{'key': 'mmgen_000', 'stage': 'mmgen'},
                     {'key': 'mmgen_finish', 'stage': 'mmgen_finish'},
                     {'key': 'collect', 'stage': 'collect'}]}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    (tmp_path / 'jobs.tsv').write_text('mmgen_000\t1\nmmgen_finish\t2\n')
    record = dict(prepared=str(tmp_path), mounted=str(tmp_path), published=True, config={})
    class Session:
        connected = True
        def query(self, command):
            assert command.count('sacct ') == 1 and command.count('squeue ') == 1
            return '1|COMPLETED|0:0\n__QUEUE__\n'
    data = snapshot(record, Session())
    counts = stage_counts(data['rows'])
    assert counts['mmgen'] == {'finished': 1, 'unknown': 1}
    assert counts['collect'] == {'waiting': 1}
    offline = snapshot(record)
    assert stage_counts(offline['rows'])['mmgen'] == {'unknown': 2}


def test_invalid_ids_never_reach_shell():
    with pytest.raises(ValueError):
        scheduler_command(['1; touch bad'])
    assert accepted_jobs('crop\t1\ncrop\t2\n') == {'crop': '2'}
