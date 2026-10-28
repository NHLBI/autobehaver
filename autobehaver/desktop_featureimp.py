"""Native desktop preparation of locally planned, sharded FeatureImp jobs."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys

from .preparation_io import report


def prepare_sharded(config, target, remote, project, metadata):
    from .postinf.featureimp.config import DEFAULT_SETTINGS
    target = Path(target)
    settings = dict(DEFAULT_SETTINGS)
    from .desktop_settings import featureimp_options
    settings.update(featureimp_options(config))
    shards = int(config.get('fi_shards', 8))
    if not 1 <= shards <= 256:
        raise ValueError('FeatureImp shards must be between 1 and 256.')
    settings.update(analysis_run_name=config['run_name'], num_shards=shards,
        control_group_tag=metadata['control_group'], training_group_a=config['group_a'],
        training_group_b=config['group_b'], sex_filter=config.get('sex', 'all'),
        include_feature_categories=metadata['feature_categories'], GLOBAL_RANDOM_SEED=int(config.get('seed', 7)),
        run_pca_analysis=False, run_fsea_analysis=False)
    (target/'local_plan_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    source = target.parent/'source'
    # Use the same interpreter as the GUI; only numpy/pandas/sklearn are needed.
    executable = str(Path(sys.executable).with_name('python.exe')) if sys.platform == 'win32' else sys.executable
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('AUTOBEHAVER_FI_MEMORY_GB', None)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    command = [executable, '-u', '-m', 'autobehaver.featureimp_local_plan', str(target.resolve())]
    with (target/'local_planning.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', env=dict(env, PYTHONIOENCODING='utf-8'),
            creationflags=flags)
        for line in process.stdout:
            log.write(line); log.flush()
            if line.strip(): report('FeatureImp planning: ' + line.strip()[-240:])
        if process.wait():
            raise RuntimeError(f'Local FeatureImp planning failed. See {target / "local_planning.log"}')
    seed = target/'analysis_seed'
    manifest = json.loads((seed/'_featureimp_plan/PLAN_MANIFEST.json').read_text(encoding='utf-8'))
    count = int(manifest['active_num_shards'])
    result_dir = project + '/9_feature_importance/FEATURE_ANALYSIS_' + Path(remote).name
    hashes = []
    for path in sorted(seed.rglob('*')):
        if path.is_file():
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(seed).as_posix())
    (target/'analysis_seed.sha256').write_text('\n'.join(hashes)+'\n', encoding='utf-8', newline='\n')
    jobs = []
    def stage(key, cli_stage, cpu, ram, dependencies=(), extra=()):
        cpu, ram = int(cpu), int(ram)
        if cpu < 2 or ram < 1: raise ValueError('FeatureImp CPUs must be >=2 and RAM positive.')
        args = [config['python'], '-u', '-m', 'autobehaver.postinf.featureimp.cli',
                '--stage', cli_stage, '--analysis_folder_path', result_dir,
                '--num_workers', str(max(1, cpu-2)), '--no-run_pca_analysis', '--no-run_fsea_analysis',
                '--postanalysis_strict', *map(str, extra),
                '--resource-monitor', '--no-resource-monitor-gpu',
                '--resource-monitor-output', remote+'/resources/'+key]
        script = f'''#!/bin/bash
#SBATCH --job-name=autobehaver_{key}
#SBATCH --partition=norm
#SBATCH --constraint=e7543
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={ram}g
#SBATCH --time=24:00:00
set -euo pipefail
export PYTHONPATH={shlex.quote(remote+'/package')} PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export AUTOBEHAVER_FI_MEMORY_GB={ram}
test -f {shlex.quote(remote+'/PUBLICATION_RECEIPT.json')}
/usr/bin/time -v -o {shlex.quote(remote+'/logs/'+key)}_"${{SLURM_JOB_ID}}"_time.txt {shlex.join(args)}
'''
        (target/(key+'.sbatch')).write_text(script, encoding='utf-8', newline='\n')
        jobs.append({'key': key, 'stage': 'featureimp_train' if cli_stage=='process_shard' else 'featureimp_'+cli_stage,
                     'dependencies': list(dependencies), 'cpus': cpu, 'ram_gib': ram})
    for index in range(count):
        stage(f'fi_train_{index:03d}', 'process_shard', config.get('fi_cpus',12), config.get('fi_memory',32),
              extra=['--shard_index',index])
    stage('fi_finalize', 'merge_shards', config.get('fi_finish_cpus',8), config.get('fi_finish_memory',32),
          [j['key'] for j in jobs])
    if config.get('pca'):
        stage('fi_pca', 'pca', config.get('fi_finish_cpus',8), config.get('fi_finish_memory',32), ['fi_finalize'])
    if config.get('fsea'):
        stage('fi_fsea', 'fsea', config.get('fi_fsea_cpus',8), config.get('fi_fsea_memory',64),
              ['fi_pca' if config.get('pca') else 'fi_finalize'])
    collector = source/'scripts/hpc/collect.sbatch'
    shutil.copyfile(collector, target/'collect.sbatch')
    jobs.append({'key':'collect','stage':'collect','dependencies':[j['key'] for j in jobs]})
    (target/'fi_schedule.tsv').write_text(''.join(j['key']+'\t'+','.join(j['dependencies'])+'\n' for j in jobs),encoding='utf-8', newline='\n')
    # Native submission owns a durable ledger. --resume only fills missing jobs;
    # it never blindly resubmits an accepted job or overwrites a results folder.
    script = '''#!/bin/bash
set -euo pipefail
bundle=__BUNDLE__
results=__RESULTS__
exec 9>>"$bundle/.submission.lock"
flock -n 9 || { echo 'Another submission is active.' >&2; exit 1; }
resume=false
if [[ "${1:-}" == --resume ]]; then resume=true; elif [[ $# != 0 ]]; then exit 2; fi
test -f "$bundle/PUBLICATION_RECEIPT.json"
test ! -f "$bundle/PUBLICATION_PENDING.txt"
if [[ -e "$bundle/jobs.tsv" && "$resume" != true ]]; then
  echo 'Already submitted or partially submitted; use Resume interrupted submission.' >&2; exit 1
fi
if [[ ! -f "$results/.desktop_plan_ready" ]]; then
  if [[ -e "$results" ]]; then echo 'Results directory exists without its completed plan marker; inspect before retrying.' >&2; exit 1; fi
  mkdir -p "$(dirname "$results")"
  mkdir "$results"
  cp -R "$bundle/analysis_seed/." "$results/"
  (cd "$results" && sha256sum -c "$bundle/analysis_seed.sha256" >/dev/null)
  sha256sum "$bundle/analysis_seed.sha256" | cut -d' ' -f1 > "$results/.desktop_plan_ready"
else
  expected=$(sha256sum "$bundle/analysis_seed.sha256" | cut -d' ' -f1)
  [[ "$(cat "$results/.desktop_plan_ready")" == "$expected" ]] || { echo 'Results belong to a different plan.' >&2; exit 1; }
fi
declare -A accepted states
if [[ -f "$bundle/jobs.tsv" ]]; then
  while IFS=$'\t' read -r key id; do
    [[ "$id" =~ ^[0-9]+$ ]] || exit 1
    accepted[$key]=$id
  done < "$bundle/jobs.tsv"
  ids=$(IFS=,; echo "${accepted[*]}")
  report=$(sacct -X -n -P -j "$ids" --format=JobIDRaw,State 9>&-) || exit 1
  while IFS='|' read -r id state rest; do
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    state="${state%% *}"; states[$id]="${state%+}"
  done <<< "$report"
  for id in "${accepted[@]}"; do
    case "${states[$id]:-}" in COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED) ;;
      *) echo "Accepted job $id is ${states[$id]:-unknown}; inspect it before continuing." >&2; exit 1;; esac
  done
fi
submitted=0
while IFS=$'\t' read -r key parents; do
  [[ -z "${accepted[$key]:-}" ]] || continue
  options=(--parsable --kill-on-invalid-dep=yes --output="$bundle/logs/${key}_%j.out" --error="$bundle/logs/${key}_%j.err")
  ids=()
  if [[ -n "$parents" ]]; then
    IFS=, read -r -a parent_keys <<< "$parents"
    for parent in "${parent_keys[@]}"; do
      id="${accepted[$parent]:-}"; [[ "$id" =~ ^[0-9]+$ ]] || exit 1
      [[ "${states[$id]:-}" != COMPLETED ]] || continue
      ids+=("$id")
    done
  fi
  if (( ${#ids[@]} )); then
    dep=$(IFS=:; echo "${ids[*]}")
    kind=afterok; [[ "$key" != collect ]] || kind=afterany
    options+=(--dependency="$kind:$dep")
  fi
  arguments=()
  if [[ "$key" == collect ]]; then
    all=$(IFS=,; echo "${accepted[*]}")
    arguments=("$bundle/package" __PYTHON__ __RESOURCES__ "$bundle" "$all")
  fi
  if (( submitted > 0 )); then sleep 1; fi
  id=$(sbatch "${options[@]}" "$bundle/$key.sbatch" "${arguments[@]}" 9>&-) || exit 1
  id=${id%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Unrecognized scheduler response; inspect accepted jobs before retrying.' >&2; exit 1; }
  printf '%s\t%s\n' "$key" "$id" >> "$bundle/jobs.tsv"
  accepted[$key]=$id
  submitted=$((submitted+1))
  echo "$key accepted: $id"
done < "$bundle/fi_schedule.tsv"
'''
    for token,value in [('__BUNDLE__',remote),('__RESULTS__',result_dir),('__PYTHON__',config['python']),('__RESOURCES__',config['resources'])]:
        script=script.replace(token,shlex.quote(value))
    (target/'submit.sh').write_text(script,encoding='utf-8',newline='\n')
    (target/'continuation_supported.txt').write_text('featureimp missing-job continuation v1\n')
    metadata.update(kind='featureimp', jobs=jobs, local_plan=True, result_dir=result_dir,
                    active_shards=count, selected_features=manifest['selected_feature_count'],
                    training_rows=manifest['training_row_count'], planned_tasks=manifest['planned_task_count'])
    (target/'plan.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata
