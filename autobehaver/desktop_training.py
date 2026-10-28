"""Training submission controls and bundles using the maintained PosePred CLI."""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path, PurePosixPath
import re
import shlex

from .cli import build_parser


MANAGED = {'help', 'directory', 'output_directory', 'metadata_path', 'metadata_tag',
           'run_name', 'pretrained_model_path', 'ablation_include_columns_file',
           'pretrain_mode', 'lr_finder_mode'}


def option_groups():
    """Read defaults/help/choices directly from the native lightweight parser."""
    groups = []
    for group in build_parser()._action_groups:
        if group.title in ('Inference Mode', 'Resource monitoring (optional)'):
            continue
        actions = [a for a in group._group_actions if a.dest not in MANAGED]
        if actions:
            groups.append((group.title, actions))
    return groups


def option_arguments(values):
    result = []
    known = {a.dest: a for _, actions in option_groups() for a in actions}
    if set(values) - known.keys():
        raise ValueError('Unknown training option: ' + ', '.join(sorted(set(values) - known.keys())))
    for name, action in known.items():
        value = values.get(name, action.default)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a checkbox value.')
            if value == action.const:
                result.append(action.option_strings[0])
            continue
        if value is None or str(value).strip() == '':
            continue
        items = value if isinstance(value, list) else [value]
        if not isinstance(action, argparse._AppendAction) and len(items) != 1:
            raise ValueError(f'{name} requires one value.')
        for item in items:
            result.extend([action.option_strings[0], str(item)])
    parser = build_parser()
    parser.error = lambda message: (_ for _ in ()).throw(ValueError(message))
    args = parser.parse_args(result)
    for key, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{key} must be finite.')
    for name in ('batch', 'num_epochs', 'checkpoint_interval', 'sequence_length', 'd_model',
                 'nhead', 'dim_feedforward', 'num_encoder_layers', 'num_decoder_layers',
                 'num_cls', 'pretrain_files_per_group', 'log_interval', 'projection_dim'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if args.num_workers < 0 or args.d_model % args.nhead:
        raise ValueError('Workers must be nonnegative and model dimension must be divisible by attention heads.')
    if not 0 <= args.dropout_rate < 1 or not 0 <= args.mask_p <= 1:
        raise ValueError('Dropout must be in [0, 1) and masking probability in [0, 1].')
    if args.base_lr <= 0 or args.noise_level < 0:
        raise ValueError('Learning rate must be positive and noise nonnegative.')
    if args.swa_mode and not 0 < args.swa_epochs < args.num_epochs:
        raise ValueError('SWA epochs must be positive and smaller than total epochs.')
    if args.cosine_decay_epochs not in (None, 0) and (args.cosine_decay_epochs < 1 or args.scheduler_type != 'cosine'):
        raise ValueError('Epoch-based cosine decay requires a positive epoch count and the cosine scheduler.')
    return result, args


def prepare_training(config, target, remote, project):
    from .desktop_jobs import digest, linux_path, remote_for, save_json

    options, args = option_arguments(config.get('training_options', {}))
    mode = config.get('training_mode', 'supervised')
    if mode not in ('supervised', 'pretrain', 'lr_finder'):
        raise ValueError('Choose supervised training, pretraining or learning-rate sweep.')
    if mode != 'supervised' and args.swa_mode:
        raise ValueError('SWA applies only to supervised training.')
    if mode != 'pretrain' and not (args.pose_pred or args.orient_pred or args.recon):
        raise ValueError('Enable at least one prediction or reconstruction head.')
    cpus, memory = int(config.get('train_cpus', 16)), int(config.get('train_memory', 32))
    if cpus < 1 or memory < 1 or args.num_workers > cpus:
        raise ValueError('Use positive CPU/RAM requests and no more loader workers than allocated CPUs.')
    partition = config.get('train_partition', 'gpu')
    gres = config.get('train_gres', 'gpu:a100:1')
    walltime = config.get('train_time', '24:00:00')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', partition):
        raise ValueError('Invalid partition name.')
    if not re.fullmatch(r'gpu(?::[A-Za-z0-9_-]+)?:1', gres):
        raise ValueError('Training uses one GPU: specify gpu:1 or gpu:TYPE:1.')
    if not re.fullmatch(r'(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d', walltime) or not any(c in '123456789' for c in walltime):
        raise ValueError('Wall time must be HH:MM:SS or D-HH:MM:SS and positive.')
    directory = Path(config['dataset']).resolve()
    metadata = Path(config['training_metadata']).resolve()
    original = metadata.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode('utf-8-sig')))
    fields = reader.fieldnames or []
    rows = list(reader)
    required = {'filename'} if mode == 'pretrain' else {'filename', 'labels'}
    if not rows or not required <= set(fields):
        raise ValueError('Metadata requires nonempty rows and columns: ' + ', '.join(sorted(required)))
    names = set()
    for row in rows:
        for field in required:
            name = (row.get(field) or '').strip().replace('\\', '/')
            rel = PurePosixPath(name)
            if not name or rel.is_absolute() or '..' in rel.parts or ':' in name:
                raise ValueError(f'Metadata {field} must be a relative path inside the training folder: {name}')
            path = directory.joinpath(*rel.parts).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError(f'Metadata input missing or outside training folder: {name}')
            row[field] = name
        if row['filename'] in names:
            raise ValueError('Metadata contains duplicate feature filenames.')
        names.add(row['filename'])
    if mode != 'pretrain' and args.defined_split and not args.noval:
        splits = {(r.get('set') or '').strip().lower() for r in rows}
        if splits != {'train', 'val'}:
            raise ValueError('Defined split requires both train and val values in the metadata set column.')
    output_root = config.get('training_output', '').strip()
    output = (remote_for(output_root, config['data_local'], config['data_remote']).rstrip('/')
              + '/' + PurePosixPath(remote).name) if output_root else remote + '/results'
    command = [linux_path(config['python']), '-m', 'autobehaver.training.posepred_training',
               '--directory', project, '--output_directory', output,
               '--metadata_path', remote + '/training_metadata.csv', '--run_name', config['run_name'], *options]
    if mode != 'supervised':
        command.append('--pretrain_mode' if mode == 'pretrain' else '--lr_finder_mode')
    external = []
    for key, option in [('training_checkpoint', '--pretrained_model_path'),
                        ('training_columns', '--ablation_include_columns_file')]:
        value = config.get(key, '').strip()
        if value:
            path = Path(value)
            if not path.is_file():
                raise ValueError(f'{key}: selected file is unavailable.')
            # Small column lists are frozen; checkpoints are referenced and verified.
            if key == 'training_columns':
                command.extend([option, remote + '/feature_columns.txt'])
            else:
                mapped = remote_for(path, config['data_local'], config['data_remote'])
                command.extend([option, mapped])
                external.append(dict(path=mapped, sha256=digest(path)))
    if args.restart_optimizer and not config.get('training_checkpoint'):
        raise ValueError('Weights-only restart requires a checkpoint.')
    command.extend(['--resource-monitor', '--resource-monitor-output', remote + '/resources/posepred_training',
                    '--resource-monitor-interval', '2'])
    target.mkdir(parents=True)
    for folder in ('logs', 'resources'):
        (target / folder).mkdir()
    with (target / 'training_metadata.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    if metadata.read_bytes() != original:
        raise ValueError('Metadata changed during preparation. Build again.')
    inputs = [dict(path=remote + '/training_metadata.csv', sha256=digest(target / 'training_metadata.csv'))]
    if config.get('training_columns'):
        (target / 'feature_columns.txt').write_bytes(Path(config['training_columns']).read_bytes())
        inputs.append(dict(path=remote + '/feature_columns.txt', sha256=digest(target / 'feature_columns.txt')))
    checks = '\n'.join('printf \'%s  %s\\n\' ' + shlex.quote(item['sha256']) + ' ' + shlex.quote(item['path'])
                       + ' | sha256sum -c -' for item in inputs + external)
    script = f'''#!/bin/bash
#SBATCH --job-name=autobehaver_training
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}g
#SBATCH --time={walltime}
set -euo pipefail
bundle={shlex.quote(remote)}
test -f "$bundle/PUBLICATION_RECEIPT.json"
test ! -f "$bundle/PUBLICATION_PENDING.txt"
{checks}
output={shlex.quote(output)}
mkdir -p -- "$(dirname "$output")"
mkdir -- "$output" || {{ echo 'Training output already exists; build a new run to preserve it.' >&2; exit 1; }}
cd -- "$output"
export PYTHONPATH="$bundle/package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/time -v -o "$bundle/logs/training_${{SLURM_JOB_ID}}_time.txt" {shlex.join(command)}
'''
    (target / 'training.sbatch').write_text(script, encoding='utf-8', newline='\n')
    submit = f'''#!/bin/bash
set -euo pipefail
bundle={shlex.quote(remote)}
exec 9>>"$bundle/.submission.lock"
flock -n 9 || {{ echo 'Another submission is in progress.' >&2; exit 1; }}
[[ ! -e "$bundle/jobs.tsv" ]] || {{ echo 'This training bundle has already been submitted.' >&2; exit 1; }}
test -f "$bundle/PUBLICATION_RECEIPT.json"
test ! -f "$bundle/PUBLICATION_PENDING.txt"
{checks}
[[ ! -e {shlex.quote(output)} ]] || {{ echo 'Training output already exists.' >&2; exit 1; }}
job=$(sbatch --parsable --output="$bundle/logs/training_%j.out" --error="$bundle/logs/training_%j.err" "$bundle/training.sbatch" 9>&-)
job=${{job%%;*}}
[[ "$job" =~ ^[0-9]+$ ]] || {{ echo "Unexpected sbatch result: $job" >&2; exit 1; }}
printf 'training\\t%s\\n' "$job" > "$bundle/jobs.tsv"
echo "PosePred training job accepted: $job"
'''
    (target / 'submit.sh').write_text(submit, encoding='utf-8', newline='\n')
    plan = dict(kind='training', project=project, jobs=[dict(stage='training', key='training')],
                mode=mode, command=command, output=output, metadata_source=str(metadata),
                metadata_sha256=inputs[0]['sha256'], metadata_rows=len(rows),
                input_policy='Metadata and column list frozen; features/labels read in place and validated by native training.',
                checkpoint=external)
    save_json(target / 'plan.json', plan)
    return plan


def add_training_tab(window, W):
    page, form = window.page('PosePred training')
    window.training_page = page
    note = W.QLabel('Submit one training configuration per run. Input features and labels stay in place; each run gets its own checkpoints and records.')
    note.setWordWrap(True); form.addRow(note)
    for label, key, default, browse in [
        ('Job label', 'training_name', 'posepred_training', None),
        ('Training data folder', 'training_directory', '', 'directory'),
        ('Metadata CSV', 'training_metadata', '', 'file'),
        ('Output parent (optional)', 'training_output', '', 'directory'),
        ('Checkpoint (optional)', 'training_checkpoint', '', 'anyfile'),
        ('Ordered feature list (optional)', 'training_columns', '', 'anyfile')]:
        window.form_entry(form, label, key, default, browse)
    hint = W.QLabel('Blank output parent uses this job bundle/results. An output parent gets a unique run subfolder.\nMetadata paths must be relative to the training folder. Full resume restores checkpoint settings and checks unchanged data; weights-only restart is below.')
    hint.setWordWrap(True); form.addRow(hint)
    window.form_combo(form, 'Mode', 'training_mode', [('Supervised training', 'supervised'),
                      ('Self-supervised pretraining', 'pretrain'), ('Learning-rate sweep', 'lr_finder')])
    for label, key, value in [('CPUs', 'train_cpus', 16), ('RAM (GiB)', 'train_memory', 32)]:
        spin = W.QSpinBox(); spin.setRange(1, 4096); spin.setValue(value)
        window.fields[key] = spin; form.addRow(label, spin)
    for label, key, value in [('Partition', 'train_partition', 'gpu'),
                              ('GPU request', 'train_gres', 'gpu:a100:1'),
                              ('Time limit', 'train_time', '24:00:00')]:
        window.form_entry(form, label, key, value)
    for title, actions in option_groups():
        box = W.QGroupBox(title if title != 'options' else 'Reproducibility and loading')
        inner = W.QFormLayout(box)
        for action in actions:
            key = 'train_opt_' + action.dest
            label = {'defined_split': 'Use metadata train / validation split',
                     'noval': 'Train without validation', 'pose_pred': 'Predict behavior',
                     'orient_pred': 'Predict orientation', 'restart_optimizer': 'Weights-only restart',
                     'nhead': 'Attention heads', 'num_workers': 'Data-loader workers',
                     'num_epochs': 'Training epochs', 'batch': 'Batch size'}.get(
                         action.dest, action.dest.replace('_', ' ').capitalize())
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                widget = W.QCheckBox(label); widget.setChecked(bool(action.default))
                window.fields[key] = widget; inner.addRow(widget)
            elif action.choices:
                window.form_combo(inner, label, key, [(str(c), c) for c in action.choices])
                widget = window.fields[key]; widget.setCurrentIndex(widget.findData(action.default))
            else:
                value = '' if action.default is None else str(action.default)
                window.form_entry(inner, label, key, value)
                widget = window.fields[key]
                if action.default is None: widget.setPlaceholderText('Optional — native default')
            widget.setToolTip(action.help or '')
            widget.setMinimumHeight(30)
        from .desktop_advanced import section
        section(window, form, title if title != 'options' else 'Reproducibility and loading',
                ['train_opt_' + a.dest for a in actions], W, box)
    return page
