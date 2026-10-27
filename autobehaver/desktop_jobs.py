"""Desktop job preparation; no GUI dependency and no implicit remote execution."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
from collections import Counter
from datetime import datetime
import uuid

from .job_publication import publish_jobs
from .source_inventory import source_files
from .preparation_io import io_results, report

FEATURES = ['BoutSummary_', 'TimeBudget_', 'StdMeasure_', 'Kinematic_',
            'KinematicBB_', 'Postural_', 'Transition_', 'Spatial_', 'SleepFeat_']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def linux_path(value):
    value = str(value)
    if not value.startswith('/') or any(ord(c) < 32 for c in value) or '..' in PurePosixPath(value).parts:
        raise ValueError(f'Expected an absolute HPC path: {value!r}')
    return value.rstrip('/')


def remote_for(local, local_root, remote_root):
    relative = Path(local).resolve().relative_to(Path(local_root).resolve())
    return linux_path(remote_root) + '/' + relative.as_posix()


def matrix_inventory(path):
    """Read identities only; the scientific pipeline remains responsible for analysis."""
    from .desktop_selection import trial_for_row
    result = []
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'individual', 'group', 'sex'}.issubset(reader.fieldnames or []):
            raise ValueError('Matrix must contain individual, group, and sex columns.')
        for row in reader:
            result.append({**{key: row[key] for key in ('individual', 'group', 'sex')}, 'trial': trial_for_row(row)})
    keys = [(r['individual'], r['group'], r['sex']) for r in result]
    if not result or len(keys) != len(set(keys)):
        raise ValueError('Matrix is empty or contains duplicate individual/group/sex identities.')
    return result


def _featureimp(config, target, remote, project):
    matrix = Path(config['matrix'])
    before = digest(matrix)
    inventory = matrix_inventory(matrix)
    chosen = config.get('selected_identities')
    selected = set(map(tuple, chosen)) if chosen is not None else {
        (r['individual'], r['group'], r['sex']) for r in inventory}
    known = {(r['individual'], r['group'], r['sex']) for r in inventory}
    if not selected or not selected <= known:
        raise ValueError('Select at least one valid matrix individual; reload if the matrix changed.')
    sex = config.get('sex', 'all')
    if sex not in ('all', 'male', 'female'):
        raise ValueError('Invalid sex filter.')
    # FeatureImp applies the selected sex itself. Check the available training groups now.
    eligible = [r for r in inventory if (r['individual'], r['group'], r['sex']) in selected
                and (sex == 'all' or r['sex'].lower() == sex)]
    a, b = config['group_a'], config['group_b']
    counts = Counter(r['group'] for r in eligible)
    if a == b or not counts[a] or not counts[b]:
        raise ValueError('Choose two different training groups with included individuals after sex filtering.')
    control = config.get('control_group', a)
    if not counts[control]:
        raise ValueError('The normalization reference group has no included individuals.')
    cpus, memory = int(config.get('fi_cpus', 12)), int(config.get('fi_memory', 192))
    if not config.get('fi_sharded') and config.get('fi_execution_version', 0) >= 1:
        # A single job must accommodate every selected stage.
        memory = max(memory, int(config.get('fi_finish_memory', 32)),
                     int(config.get('fi_fsea_memory', 64)) if config.get('fsea') else 0)
    if cpus < 2 or memory < 1:
        raise ValueError('FeatureImp requires at least two CPUs and positive RAM.')
    target.mkdir(parents=True)
    (target / 'logs').mkdir()
    (target / 'resources').mkdir()
    frozen_matrix = target / 'featureimp_matrix.csv'
    with matrix.open(newline='', encoding='utf-8-sig') as src, frozen_matrix.open('w', newline='', encoding='utf-8') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if (row['individual'], row['group'], row['sex']) in selected:
                writer.writerow(row)
    if digest(matrix) != before:
        raise ValueError('The source matrix changed during preparation. Build again after MMGen finishes.')
    matrix_hash = digest(frozen_matrix)
    if config.get('fi_sharded'):
        if config.get('recovery'):
            raise ValueError('Use the saved sharded analysis for stage recovery; do not rebuild its frozen plan.')
        linux_path(config['python']); linux_path(config['resources'])
        from .desktop_featureimp import prepare_sharded
        return prepare_sharded(config, target, remote, project, {
            'kind': 'featureimp', 'project': project,
            'matrix_source': str(matrix), 'matrix_source_sha256': before, 'matrix_sha256': matrix_hash,
            'included_identities': sorted(selected), 'sex': sex, 'group_counts_after_sex_filter': dict(counts),
            'training_group_a': a, 'training_group_b': b, 'control_group': control,
            'feature_categories': FEATURES})
    recovery_preflight, recovery_claim = '', '' 
    if config.get('recovery'):
        recovery = config['recovery']
        parent = linux_path(recovery['parent_remote'])
        previous = recovery['expected'][0]['job_id']
        if not re.fullmatch(r'[0-9]+', previous):
            raise ValueError('Invalid FeatureImp recovery job ID.')
        if matrix_hash != recovery['matrix_sha256']:
            raise ValueError('FeatureImp recovery must preserve the original frozen matrix exactly.')
        recovery_preflight = f'''
parent={shlex.quote(parent)}
previous={shlex.quote(previous)}
exec 8>>"$parent/.recovery.lock"
flock -n 8 || {{ echo 'Another recovery submission is active.' >&2; exit 1; }}
if [[ -f "$parent/recovery_claims.tsv" ]] && awk -F '\\t' -v j="$previous" '$1=="featureimp" && $2==j {{found=1}} END {{exit !found}}' "$parent/recovery_claims.tsv"; then
    echo 'A FeatureImp replacement already exists. Open its recovery run.' >&2; exit 1
fi
report=$(sacct -X -j "$previous" --noheader --parsable2 --format=JobIDRaw,State 8>&- 9>&-)
state=$(printf '%s\\n' "$report" | awk -F '|' -v j="$previous" '$1==j {{print $2}}')
state="${{state%% *}}"; state="${{state%+}}"
case "$state" in FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
    *) echo "FeatureImp is now $state; refresh before recovery. No jobs submitted." >&2; exit 1;; esac
'''
        recovery_claim = '''printf 'featureimp\\t%s\\t%s\\t%s\\n' "$previous" "$job" "$bundle" >> "$parent/recovery_claims.tsv"'''
    command = [linux_path(config['python']), '-m', 'autobehaver.postinf.featureimp.cli',
               '--main_dir', project, '--matrix_path', remote + '/featureimp_matrix.csv',
               '--analysis_run_name', config['run_name'], '--stage', 'all',
               '--control_group_tag', control, '--training_group_a', a, '--training_group_b', b,
               '--include_feature_categories', *FEATURES, '--sex_filter', sex,
               '--random_seed', str(int(config.get('seed', 7))), '--num_workers', str(cpus - 2 or 1),
               '--run_pca_analysis' if config.get('pca') else '--no-run_pca_analysis',
               '--run_fsea_analysis' if config.get('fsea') else '--no-run_fsea_analysis',
               '--resource-monitor', '--no-resource-monitor-gpu', '--resource-monitor-output',
               remote + '/resources/featureimp', '--resource-monitor-interval', '2']
    from .desktop_settings import featureimp_arguments
    command.extend(featureimp_arguments(config))
    script = f'''#!/bin/bash
#SBATCH --job-name=autobehaver_fi
#SBATCH --partition=norm
#SBATCH --constraint=e7543
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}g
#SBATCH --time=24:00:00
set -euo pipefail
bundle={shlex.quote(remote)}
test -f "$bundle/PUBLICATION_RECEIPT.json"
printf '%s  %s\\n' {shlex.quote(matrix_hash)} "$bundle/featureimp_matrix.csv" | sha256sum -c -
export PYTHONPATH="$bundle/package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export AUTOBEHAVER_FI_MEMORY_GB={memory}
/usr/bin/time -v -o "$bundle/logs/featureimp_${{SLURM_JOB_ID}}_time.txt" {shlex.join(command)}
'''
    (target / 'featureimp.sbatch').write_text(script, encoding='utf-8', newline='\n')
    submit = f'''#!/bin/bash
set -euo pipefail
bundle={shlex.quote(remote)}
exec 9>>"$bundle/.submission.lock"
flock -n 9 || {{ echo 'Another submission is in progress.' >&2; exit 1; }}
[[ ! -e "$bundle/jobs.tsv" ]] || {{ echo 'This FeatureImp bundle has already been submitted.' >&2; exit 1; }}
test -f "$bundle/PUBLICATION_RECEIPT.json"
test ! -f "$bundle/PUBLICATION_PENDING.txt"
{recovery_preflight}
job=$(sbatch --parsable --output="$bundle/logs/featureimp_%j.out" --error="$bundle/logs/featureimp_%j.err" "$bundle/featureimp.sbatch" 8>&- 9>&-)
job=${{job%%;*}}
[[ "$job" =~ ^[0-9]+$ ]] || {{ echo "Unexpected sbatch result: $job" >&2; exit 1; }}
printf 'featureimp\\t%s\\n' "$job" > "$bundle/jobs.tsv"
{recovery_claim}
echo "FeatureImp job accepted: $job"
'''
    (target / 'submit.sh').write_text(submit, encoding='utf-8', newline='\n')
    plan = {'kind': 'featureimp', 'project': project, 'jobs': [{'stage': 'featureimp', 'key': 'featureimp'}],
            'matrix_source': str(matrix), 'matrix_source_sha256': before, 'matrix_sha256': matrix_hash,
            'included_identities': sorted(selected), 'sex': sex, 'group_counts_after_sex_filter': dict(counts),
            'training_group_a': a, 'training_group_b': b, 'control_group': control, 'command': command}
    save_json(target / 'plan.json', plan)
    return plan


def build_job(config, source_root, work_root):
    """Build locally and freeze source; sending and submission are separate actions."""
    config = dict(config)
    prepared_plan = config.pop('_prepared_plan', None)
    if config.get('kind') not in ('pipeline', 'featureimp', 'training'):
        raise ValueError('Choose Pipeline, FeatureImp or PosePred training.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', config['run_name']):
        raise ValueError('Run name: use letters, numbers, underscores or hyphens (up to 80 characters).')
    project = remote_for(config['dataset'], config['data_local'], config['data_remote'])
    if not Path(config['dataset']).is_dir():
        raise ValueError('The selected dataset folder is unavailable.')
    bundle_name = config['run_name'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    remote = linux_path(config['jobs_remote']) + '/' + bundle_name
    if PurePosixPath(remote).is_relative_to(PurePosixPath(project)):
        raise ValueError('Job bundles must be outside the dataset.')
    mounted = Path(config['jobs_local']) / bundle_name
    if mounted.resolve().is_relative_to(Path(config['dataset']).resolve()):
        raise ValueError('Mounted job bundles must be outside the dataset.')
    root = Path(work_root) / bundle_name
    if root.resolve().is_relative_to(Path(config['data_local']).resolve()):
        raise ValueError('Local preparation must be outside mounted data storage.')
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / 'source'
    frozen.mkdir()
    source_root = Path(source_root)
    files = list(source_files(source_root))
    needed = sum((source_root / p).stat().st_size for p in files)
    if config['kind'] == 'featureimp':
        needed += Path(config['matrix']).stat().st_size * 2
    if shutil.disk_usage(root).free < needed + 64 * 1024 * 1024:
        raise OSError('Insufficient local storage to prepare this bundle.')
    def freeze_file(relative):
        out = frozen / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, out)
        if digest(out) != digest(source_root / relative):
            raise OSError('Source changed while freezing the bundle; rebuild.')
    list(io_results(freeze_file, files, label='Freezing package files', workers=4))
    report('Planning dataset jobs...')
    target = root / 'job'
    if config['kind'] == 'pipeline':
        spec = importlib.util.spec_from_file_location('autobehaver_native_prepare', source_root / 'scripts/hpc/prepare.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models = config.get('models') or None
        if models and (len(models) != 3 or not all(models)):
            raise ValueError('Supply all three model paths, or leave all three blank to use the resource defaults.')
        batching = config.get('batching') if config.get('speed', 'fast') == 'fast' else None
        batch_sizes = config.get('batch_sizes') if config.get('speed', 'fast') == 'fast' and not batching else None
        module.prepare_jobs(output_dir=target, remote_job_dir=remote, project=project,
            python=linux_path(config['python']), resources=linux_path(config['resources']),
            package=remote + '/package', local_project=config['dataset'],
            start_at=config.get('start_at', 'crop'), speed=config.get('speed', 'fast'),
            moviechecker_mode=config.get('moviechecker_mode', 'reviewed'),
            inference_model=models, batch_sizes=batch_sizes, prepared_plan=prepared_plan,
            batching=batching, new_only=config.get('new_only', False))
        for stage, values in config.get('allocations', {}).items():
            if stage not in ('crop', 'dlc', 'features', 'posepred', 'mmgen', 'mmgen_finish'):
                raise ValueError('Unknown resource stage.')
            cpu, ram = map(int, values)
            if cpu < 1 or ram < 1:
                raise ValueError('CPU and RAM must be positive.')
            path = target / 'templates' / (stage + '.sbatch')
            original = path.read_text()
            original_cpu = int(re.search(r'(?m)^#SBATCH --cpus-per-task=(\d+)', original).group(1))
            content = re.sub(r'(?m)^#SBATCH --cpus-per-task=.*$', f'#SBATCH --cpus-per-task={cpu}', original)
            content = re.sub(r'(?m)^#SBATCH --mem=.*$', f'#SBATCH --mem={ram}g', content)
            if cpu < original_cpu:
                content = re.sub(r'--num_workers (\d+)', lambda m: '--num_workers ' + str(min(int(m[1]), max(1, cpu - 1))), content)
                if stage == 'crop':
                    workers = int(re.search(r'--num_workers (\d+)', content)[1])
                    content = re.sub(r'--max_workers (\d+)', lambda m: '--max_workers ' + str(min(int(m[1]), max(1, cpu // workers))), content)
            hours = config.get('walltime_hours', {}).get(stage)
            if hours is not None:
                if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
                    raise ValueError('Walltime hours must be a positive integer.')
                content = re.sub(r'(?m)^#SBATCH --time=.*$', f'#SBATCH --time={hours}:00:00', content)
            path.write_text(content, encoding='utf-8', newline='\n')
        pose_template = target / 'templates/posepred.sbatch'
        content = pose_template.read_text(encoding='utf-8')
        ram = int(re.search(r'(?m)^#SBATCH --mem=(\d+)g', content)[1])
        content = re.sub(r'--memory_gb [\d.]+', '--memory_gb ' + str(ram), content)
        from .desktop_settings import update_posepred_template
        content = update_posepred_template(content, config)
        pose_template.write_text(content, encoding='utf-8', newline='\n')
        # Concurrent full recordings, distinct from inputs assigned to a batch.
        feature_template = target / 'templates/features.sbatch'
        content = feature_template.read_text()
        cpu = int(re.search(r'(?m)^#SBATCH --cpus-per-task=(\d+)', content)[1])
        workers = config.get('feature_workers', 8)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError('KeyPre worker ceiling must be a nonnegative integer (0 = automatic).')
        workers = min(workers, cpu)
        config['feature_workers'] = workers
        ram = int(re.search(r'(?m)^#SBATCH --mem=(\d+)g', content)[1])
        content = re.sub(r'--memory_gb [\d.]+', '--memory_gb ' + str(ram), content)
        content = re.sub(r'--num_workers \d+', '--num_workers ' + str(workers), content)
        feature_template.write_text(content, encoding='utf-8', newline='\n')
        plan = json.loads((target / 'plan.json').read_text())
        if config.get('recovery'):
            recovery = config['recovery']
            (target / 'recovery_parent.txt').write_text(recovery['parent_remote'] + '\n', encoding='utf-8', newline='\n')
            (target / 'recovery_expected.tsv').write_text(''.join(
                f"{row['key']}\t{row['job_id']}\t{row['role']}\n" for row in recovery['expected']), encoding='utf-8', newline='\n')
            save_json(target / 'recovery.json', recovery)
            save_json(target / 'recovery_crop_sources.json', {'sources': recovery.get('crop_reuse_sources', [])})
    elif config['kind'] == 'training':
        from .desktop_training import prepare_training
        plan = prepare_training(config, target, remote, project)
    else:
        plan = _featureimp(config, target, remote, project)
    record = {'format': 1, 'kind': config['kind'], 'name': bundle_name, 'config': config,
              'root': str(root), 'prepared': str(target), 'source': str(frozen),
              'mounted': str(mounted), 'remote': remote, 'project': project,
              'job_counts': dict(Counter(job['stage'] for job in plan['jobs'] if not job.get('retained'))),
              'monitor_job_counts': dict(Counter(job['stage'] for job in plan['jobs'])),
              'movies': plan.get('usable_original_movies'), 'chambers': plan.get('chamber_movies'),
              'incremental': plan.get('incremental'),
              'featureimp_plan': {k: plan.get(k) for k in ('local_plan', 'result_dir', 'active_shards', 'selected_features', 'training_rows', 'planned_tasks')} if config['kind'] == 'featureimp' else None,
              'training': {k: plan[k] for k in ('mode', 'output', 'metadata_rows', 'command')} if config['kind'] == 'training' else None,
              'published': False,
              'build_hashes': dict(io_results(lambda p: (str(p.relative_to(root)), digest(p)),
                  [p for p in root.rglob('*') if p.is_file()], label='Verifying prepared bundle', workers=4))}
    save_json(root / 'desktop_run.json', record)
    return record


def publish(record):
    expected_files = set(record['build_hashes'])
    actual_files = {str(p.relative_to(record['root'])) for folder in ('source', 'job')
                    for p in (Path(record['root']) / folder).rglob('*') if p.is_file()}
    if actual_files != expected_files:
        raise ValueError('Prepared file list changed after Build. Build a new bundle.')
    def verify_frozen(item):
        relative, expected = item
        if digest(Path(record['root']) / relative) != expected:
            raise ValueError('Prepared files changed after Build. Build a new bundle before sending.')
    list(io_results(verify_frozen, record['build_hashes'].items(), label='Verifying saved build', workers=4))
    receipt = publish_jobs(record['prepared'], record['mounted'], source_root=record['source'])
    record = dict(record, published=True, publication_files=receipt['file_count'])
    save_json(Path(record['root']) / 'desktop_run.json', record)
    return record


def submission_button_state(record):
    """Reflect accepted job records without querying or changing the scheduler."""
    if not record or not record.get('published'):
        return '3  Submit job', False
    try:
        text = (Path(record['mounted']) / 'jobs.tsv').read_text(encoding='utf-8')
    except FileNotFoundError:
        return '3  Submit job', True
    except (OSError, UnicodeError):
        return '3  Check submission', False
    accepted = {fields[0] for line in text.splitlines()
                if len(fields := line.split('\t')) >= 2 and fields[1].isdigit()}
    planned = sum(record.get('job_counts', {}).values())
    if planned and len(accepted) >= planned:
        return '3  Submitted', False
    return ('3  Partially submitted' if accepted else '3  Check submission'), False


def can_resume_submission(record):
    return bool(record and record.get('kind') in ('pipeline', 'featureimp')
                and submission_button_state(record)[0] == '3  Partially submitted'
                and (Path(record['prepared']) / 'continuation_supported.txt').is_file())


def verify_submission(record, *, resume=False):
    mounted = Path(record['mounted'])
    if (mounted / 'PUBLICATION_PENDING.txt').exists():
        raise ValueError('Publication is incomplete. Review the Send error.')
    receipt = json.loads((mounted / 'PUBLICATION_RECEIPT.json').read_text(encoding='utf-8'))
    if not receipt.get('verified'):
        raise ValueError('No verified publication receipt.')
    # Check executable scripts against the local Build record, rather than
    # trusting a receipt and launcher that could both have been replaced.
    for relative, expected in record['build_hashes'].items():
        rel = Path(relative)
        if rel.parts[0] == 'job' and rel.suffix in ('.sh', '.sbatch'):
            published = mounted.joinpath(*rel.parts[1:])
            if digest(published) != expected:
                raise ValueError('A published job script changed. Build a new bundle.')
    if resume and not can_resume_submission(record):
        raise ValueError('Resume requires a partially submitted bundle built with continuation support. Existing frozen bundles are not modified.')
    if not resume and (mounted / 'jobs.tsv').exists():
        raise ValueError('This bundle already has accepted jobs. Open its job IDs; do not submit it again.')
    return 'bash ' + shlex.quote(record['remote'] + '/submit.sh') + (' --resume' if resume else '')


def ssh_arguments(host, username, command=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host):
        raise ValueError('Invalid SSH hostname.')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', username):
        raise ValueError('Enter a valid SSH username.')
    args = ['ssh', '-t', '-o', 'ConnectTimeout=20', f'{username}@{host}']
    if command:
        args.append(command)
    return args
