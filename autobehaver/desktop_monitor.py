"""Read-only Slurm job status for a frozen desktop run."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

LABELS = {'crop': 'Autocropper', 'dlc': 'DLC', 'features': 'KeyPre + normalization',
          'posepred': 'PosePred', 'mmgen': 'MMGen', 'featureimp': 'FeatureImp', 'collect': 'Accounting',
          'training': 'PosePred training', 'featureimp_train': 'FeatureImp training',
          'featureimp_merge_shards': 'FeatureImp finalization', 'featureimp_pca': 'PCA', 'featureimp_fsea': 'FSEA'}
COLORS = {'waiting': '#c7ced8', 'running': '#efa937', 'finished': '#51a775',
          'failed': '#d75757', 'unknown': '#8c98ad'}


def category(state, exit_code='', reason=''):
    state = state.split()[0].rstrip('+') if state else ''
    if reason == 'DependencyNeverSatisfied':
        return 'failed'
    if state == 'COMPLETED':
        return 'finished' if exit_code in ('', '0:0') else 'failed'
    if state in ('RUNNING', 'COMPLETING', 'CONFIGURING', 'STAGE_OUT'):
        return 'running'
    if state in ('PENDING', 'REQUEUED', 'REQUEUE_HOLD', 'SUSPENDED', 'RESIZING'):
        return 'waiting'
    if state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'):
        return 'failed'
    return 'unknown'


def accepted_jobs(text):
    result = {}
    for line in text.splitlines():
        fields = line.split('\t')
        if len(fields) == 2 and re.fullmatch(r'\d+', fields[1]):
            result[fields[0]] = fields[1]
    return result


def scheduler_command(ids):
    if not ids or any(not re.fullmatch(r'\d+', value) for value in ids):
        raise ValueError('Expected numeric scheduler job IDs.')
    joined = ','.join(sorted(set(ids), key=int))
    # Two batched queries, never one scheduler request per job.
    return (f"sacct -X -n -P -j {joined} --format=JobIDRaw,State%40,ExitCode; "
            "a=$?; printf '\\n__QUEUE__\\n'; "
            "squeue --me -h -o '%i|%T|%r'; b=$?; "
            '[ "$a" -eq 0 ] && [ "$b" -eq 0 ]')


def parse_scheduler(text):
    result, queue = {}, False
    for line in text.splitlines():
        if line.strip() == '__QUEUE__':
            queue = True
            continue
        fields = [value.strip() for value in line.split('|')]
        if len(fields) < 3 or not re.fullmatch(r'\d+', fields[0]):
            continue  # Ignore shell warnings and step rows, never count them as jobs.
        job, state, detail = fields[:3]
        result[job] = {'state': state, 'exit': '' if queue else detail,
                       'reason': detail if queue else '',
                       'category': category(state, '' if queue else detail, detail if queue else '')}
    return result


def snapshot(record, session=None):
    root = Path(record['mounted'] if record.get('published') else record['prepared'])
    plan = json.loads((Path(record['prepared']) / 'plan.json').read_text(encoding='utf-8'))
    path = root / 'jobs.tsv'
    inherited = record['config'].get('recovery', {}).get('retained', {})
    accepted = {key: value['job_id'] for key, value in inherited.items()}
    accepted.update(accepted_jobs(path.read_text(encoding='utf-8') if path.exists() else ''))
    states, source = {}, 'Submission records only; connect and refresh for live status.'
    if accepted and session is not None and session.connected:
        states = parse_scheduler(session.query(scheduler_command(list(accepted.values()))))
        source = 'Live scheduler status'
    rows = []
    for job in plan['jobs']:
        key, stage = job['key'], job['stage']
        job_id = accepted.get(key, '')
        state = states.get(job_id, {'state': 'Unknown' if job_id else 'Not submitted',
                                   'exit': '', 'reason': '',
                                   'category': 'unknown' if job_id else 'waiting'})
        rows.append(dict(state, key=key, stage='mmgen' if stage == 'mmgen_finish' else stage,
                         job_id=job_id, bundle=inherited.get(key, {}).get('bundle', str(root)),
                         remote_bundle=inherited.get(key, {}).get('remote_bundle', record.get('remote', ''))))
    return {'rows': rows, 'source': source,
            'updated': datetime.now(timezone.utc).isoformat(timespec='seconds')}


def stage_counts(rows):
    result = {}
    for row in rows:
        result.setdefault(row['stage'], Counter())[row['category']] += 1
    return result
