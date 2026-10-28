"""Prepare a new native schedule while retaining surviving jobs and their IDs."""
import copy
import json
import uuid
from pathlib import Path
from . import desktop_jobs

TERMINAL_FAILURE = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                    'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
KEEP = {'COMPLETED', 'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
        'REQUEUED', 'RESIZING', 'STAGE_OUT'}


def recovery_config(parent, snapshot, allocations=None):
    if snapshot.get('source') != 'Live scheduler status':
        raise ValueError('Refresh live scheduler status before building recovery.')
    plan = json.loads((Path(parent['prepared']) / 'plan.json').read_text(encoding='utf-8'))
    if parent['kind'] == 'featureimp':
        if plan.get('local_plan'):
            raise ValueError('Automatic failed-stage recovery is not available for sharded FeatureImp. Retain this analysis and its completed shards; a stage-specific retry must use its original frozen plan. Resume interrupted submission only adds jobs that were never accepted.')
        rows = [row for row in snapshot['rows'] if row['key'] == 'featureimp']
        if len(rows) != 1 or not rows[0].get('job_id') or not rows[0]['job_id'].isdigit():
            raise ValueError('FeatureImp recovery needs a recorded job ID and live terminal failure.')
        row = rows[0]
        if (row.get('state') or 'Unknown').split()[0].rstrip('+') not in TERMINAL_FAILURE:
            raise ValueError('FeatureImp recovery replaces only failed/cancelled jobs; active and completed jobs are preserved.')
        matrix = Path(parent['prepared']) / 'featureimp_matrix.csv'
        if desktop_jobs.digest(matrix) != plan['matrix_sha256']:
            raise ValueError('The original frozen FeatureImp matrix changed; no recovery built.')
        claims = Path(parent['mounted']) / 'recovery_claims.tsv'
        if claims.exists() and any(line.split('\t')[:2] == ['featureimp', row['job_id']] for line in claims.read_text().splitlines()):
            raise ValueError('A replacement already exists. Open the newer recovery run.')
        config = copy.deepcopy(parent['config'])
        config['matrix'] = str(matrix)
        config['selected_identities'] = plan['included_identities']
        config['run_name'] = 'retry_' + uuid.uuid4().hex[:8] + '_' + parent['config']['run_name'][:50]
        if allocations:
            config['fi_cpus'], config['fi_memory'] = allocations['featureimp']
        config['recovery'] = {
            'parent_remote': parent['remote'], 'parent_mounted': parent['mounted'],
            'parent_record': str(Path(parent['root']) / 'desktop_run.json'),
            'snapshot_time': snapshot['updated'], 'expected': [{'key': 'featureimp', 'job_id': row['job_id'], 'role': 'replace'}],
            'retained': {}, 'replace_keys': ['featureimp'], 'matrix_sha256': plan['matrix_sha256']}
        return config
    if parent['kind'] != 'pipeline':
        raise ValueError('Unknown recovery workflow.')
    states = {row['key']: row for row in snapshot['rows']}
    expected, replaced, retained = [], set(), {}
    jobs = []
    for original in plan['jobs']:
        if original['stage'] == 'collect':
            continue  # A fresh accounting job records this attempt; old collector is untouched.
        row = states.get(original['key'])
        if not row or not row.get('job_id'):
            raise ValueError('Missing accepted job/state. Complete or recover the partial submission first.')
        state = row['state'].split()[0].rstrip('+')
        key = original['key']
        if state in TERMINAL_FAILURE:
            role = 'replace'; replaced.add(key)
        elif state in KEEP:
            role = 'retain'; retained[key] = {'job_id': row['job_id'],
                'bundle': row.get('bundle', parent['mounted']),
                'remote_bundle': row.get('remote_bundle', parent['remote'])}
        else:
            raise ValueError(f'Unknown scheduler state for {key}; no recovery built.')
        if role == 'retain' and any(p in replaced for p in original['depends_on']):
            raise ValueError(f'{key} is still {state} but its parent failed. Wait for scheduler cancellation, then refresh; no jobs were cancelled.')
        expected.append({'key': key, 'job_id': row['job_id'], 'role': role})
        jobs.append(dict(original, retained=(role == 'retain')))
    if not replaced:
        raise ValueError('No failed or cancelled processing jobs need replacement.')
    # Refuse stale recovery plans already superseded by another accepted recovery.
    claims = Path(parent['mounted']) / 'recovery_claims.tsv'
    if claims.exists():
        for line in claims.read_text().splitlines():
            values = line.split('\t')
            if len(values) >= 3 and any(r['key'] == values[0] and r['job_id'] == values[1] and r['role'] == 'replace' for r in expected):
                raise ValueError('A replacement already exists. Open the newer recovery run before retrying again.')
    jobs.append(dict(key='collect', stage='collect', depends_on=[j['key'] for j in jobs],
                     file_list='-', shard_index=0, num_shards=1))
    plan['jobs'] = jobs
    lists = {path.name: path.read_text(encoding='utf-8').splitlines()
             for path in (Path(parent['prepared']) / 'batches').glob('*.txt')}
    config = copy.deepcopy(parent['config'])
    config.pop('recovery', None)
    config['run_name'] = 'recovery_' + parent['name'][:55]
    if allocations:
        config['allocations'] = allocations
    recovery = {'parent_remote': parent['remote'], 'parent_mounted': parent['mounted'],
                'parent_record': str(Path(parent['root']) / 'desktop_run.json'),
                'snapshot_time': snapshot['updated'], 'expected': expected,
                'retained': retained, 'replace_keys': sorted(replaced)}
    config['recovery'] = recovery
    config['_prepared_plan'] = (plan, lists)
    return config


def build_recovery(parent, snapshot, source_root, work_root, allocations=None, *, walltime_hours=None):
    config = recovery_config(parent, snapshot, allocations)
    if walltime_hours is not None:
        config['walltime_hours'] = walltime_hours
    if parent['kind'] == 'featureimp':
        return desktop_jobs.build_job(config, source_root, work_root)
    folder = Path(parent['source']) / 'autobehaver/preinf/autocropper'
    paths = [folder/name for name in ('detection.py', 'cropping.py', 'contract.py')]
    if (folder/'grid_fit.py').exists():
        paths.append(folder/'grid_fit.py')
    sources = parent['config'].get('recovery', {}).get('crop_reuse_sources', [])
    source = {'root': parent['remote'] + '/package/autobehaver/preinf/autocropper',
              'files': {path.name: desktop_jobs.digest(path) for path in paths}}
    config['recovery']['crop_reuse_sources'] = sources + [source]
    return desktop_jobs.build_job(config, source_root, work_root)
