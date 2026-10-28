"""Shared, lightweight settings contract for launcher controls and agent drafts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import re

from .desktop_base_settings import BASE_SETTINGS

SCHEMA_VERSION = 1
WORKFLOWS = ('pipeline', 'featureimp', 'training')


def advanced_specs():
    from .preinf.posepred_inference.config import PosePredInferenceConfig
    from .postinf.featureimp.config import DEFAULT_SETTINGS, build_arg_parser
    result = {}
    pose = PosePredInferenceConfig(input_dir='')
    for name, label, minimum, maximum, help_text, group in [
        ('batch', 'Batch size (0 = automatic)', 0, 1048576, 'Automatic profiles throughput within the available RAM and VRAM budget.', 'PosePred compute'),
        ('num_workers', 'Loader workers (-1 = automatic)', -1, 255, 'Worker ceiling; automatic adapts to available CPUs and RAM.', 'PosePred compute'),
        ('noise_workers', 'Augmentation workers (0 = automatic)', 0, 255, 'CPU noise preparation worker ceiling.', 'PosePred compute'),
        ('vram_fraction', 'VRAM budget fraction', .1, .9, 'VRAM budget fraction for automatic batch profiling; not a utilization target or memory guarantee.', 'PosePred compute'),
        ('inf_average', 'Augmented passes per model', 1, 10000, 'Changes predictions: number of inference passes averaged for each model.', 'PosePred predictions'),
        ('use_tta', 'Use test-time augmentation', None, None, 'Changes predictions: add reproducible noise during inference.', 'PosePred predictions'),
        ('tta_noise_level', 'Augmentation noise', 0., 10., 'Changes predictions: Gaussian noise standard deviation.', 'PosePred predictions'),
        ('tta_seed', 'Augmentation seed', 0, 2147483647, 'Changes predictions: deterministic noise seed.', 'PosePred predictions'),
        ('min_pose_bout', 'Minimum behavior bout (frames)', 0, 100000, 'Changes predictions: smoothing threshold; 0 or 1 disables smoothing.', 'PosePred predictions'),
        ('min_orient_bout', 'Minimum orientation bout (frames)', 0, 100000, 'Changes predictions: smoothing threshold; 0 or 1 disables smoothing.', 'PosePred predictions'),
    ]:
        value = getattr(pose, name)
        result['pp_'+name] = dict(default=value, type='boolean' if isinstance(value, bool) else 'number' if isinstance(value, float) else 'integer',
            minimum=minimum, maximum=maximum, label=label, help=help_text, group=group, workflow='pipeline', native=name)
    actions = {a.dest: a for a in build_arg_parser()._actions}
    labels = dict(n_splits='Cross-validation folds', H_hits_per_feature='Hits per feature',
        n_features_per_model='Features per specialist', n_estimators='Trees per model',
        learning_rate='Learning rate', max_depth='Maximum tree depth', model_subsample='Training row fraction per tree',
        min_feature_completeness_per_group='Minimum completeness per group', MIN_VARIANCE_THRESHOLD='Minimum feature standard deviation',
        n_permutations='Label-shuffle repetitions', subset_weight_tau_quantile='Weighting quantile',
        subset_weight_gamma='Weighting exponent', pca_fixed_k='PCA components', fsea_permutations='FSEA permutations',
        fsea_min_size='Minimum features per FSEA set', fsea_max_size='Maximum features per FSEA set', fsea_fdr_threshold='FSEA FDR threshold')
    for name, minimum, maximum in [
        ('n_splits', 2, 100), ('H_hits_per_feature', 1, 100000), ('n_features_per_model', 1, 100000),
        ('n_estimators', 1, 100000), ('learning_rate', .000001, 1.), ('max_depth', 1, 100),
        ('model_subsample', .000001, 1.), ('min_feature_completeness_per_group', 0., 1.),
        ('MIN_VARIANCE_THRESHOLD', 0., 100.), ('n_permutations', 1, 100000),
        ('subset_weight_tau_quantile', .000001, .999999), ('subset_weight_gamma', .000001, 100.),
        ('pca_fixed_k', 1, 10000), ('fsea_permutations', 1, 10000000),
        ('fsea_min_size', 1, 100000), ('fsea_max_size', 1, 100000), ('fsea_fdr_threshold', .000001, 1.),
    ]:
        value = DEFAULT_SETTINGS[name]
        result['fi_opt_'+name] = dict(default=value, type='number' if isinstance(value, float) else 'integer',
            minimum=minimum, maximum=maximum, label=labels[name], help=actions[name].help,
            group='FeatureImp analysis', workflow='featureimp', native=name, flag=actions[name].option_strings[0])
    return result


def specifications():
    from .desktop_training import option_groups
    specs = deepcopy(BASE_SETTINGS)
    specs.update(advanced_specs())
    for title, actions in option_groups():
        for action in actions:
            boolean = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
            # Text fields intentionally retain their native numeric spelling.
            default = bool(action.default) if boolean else action.default if action.choices else '' if action.default is None else str(action.default)
            spec = dict(default=default, type='boolean' if boolean else 'string', help=action.help or '',
                        workflow='training', group='Training: '+title)
            if action.choices: spec['choices'] = list(action.choices)
            specs['train_opt_'+action.dest] = spec
    return specs


def defaults():
    return {key: spec['default'] for key, spec in specifications().items()}


def validate_value(key, value, spec):
    kind = spec['type']
    valid = (type(value) is bool if kind == 'boolean' else
             type(value) is int if kind == 'integer' else
             type(value) in (int, float) and math.isfinite(value) if kind == 'number' else
             isinstance(value, str))
    if not valid: raise ValueError(f'{key}: expected {kind}.')
    if kind == 'number' and value != round(value, 10): raise ValueError(f'{key}: at most 10 decimal places are supported by the GUI.')
    if kind == 'string' and any(c in value for c in ('\x00', '\r', '\n')):
        raise ValueError(f'{key}: use a single-line value.')
    if 'choices' in spec and value not in spec['choices']: raise ValueError(f'{key}: choose one of {spec["choices"]}.')
    for bound, op in [('minimum', lambda a,b: a < b), ('maximum', lambda a,b: a > b)]:
        if spec.get(bound) is not None and op(value, spec[bound]):
            raise ValueError(f'{key}: {bound} is {spec[bound]}.')
    return value


def validate_config(config, workflow, *, check_inputs=False, strict=True):
    """Validate a GUI draft, without importing Qt, running analysis, or writing files."""
    if workflow not in WORKFLOWS: raise ValueError('Unknown workflow.')
    if not isinstance(config, dict): raise ValueError('Config must be an object.')
    specs = specifications()
    extras = {'selected_identities', 'saved_batch_sizes', 'fi_execution_version', 'feature_execution_version', 'mmgen_allocation_version'}
    unknown = config.keys() - specs.keys() - extras
    if strict and unknown: raise ValueError('Unknown settings: '+', '.join(sorted(unknown)))
    result = defaults()
    for key, value in config.items():
        if key in specs:
            # Empty dynamic group combos are represented as null by Qt.
            if key in ('group_a','group_b','control_group') and value is None: value = ''
            result[key] = validate_value(key, value, specs[key])
        elif key in extras: result[key] = deepcopy(value)
    for key in ('fi_execution_version','feature_execution_version','mmgen_allocation_version'):
        if key in result and (type(result[key]) is not int or result[key] != 1): raise ValueError(f'Unsupported {key}.')
    batches = result.get('saved_batch_sizes', {})
    if not isinstance(batches, dict) or any(k not in ('crop','dlc','features','posepred','mmgen') or type(v) is not int or v < 1 for k,v in batches.items()):
        raise ValueError('Saved batch sizes must be positive integers for known stages.')
    if result['use_saved_batch_sizes'] and not batches: raise ValueError('Saved batch sizes are enabled but empty.')
    selected = result.get('selected_identities')
    if selected is not None:
        if not isinstance(selected, list) or not selected or any(not isinstance(row, list) or len(row)!=3 or any(not isinstance(v,str) or not v for v in row) for row in selected):
            raise ValueError('selected_identities must be a nonempty list of [individual, group, sex] rows.')
        if len(set(map(tuple, selected))) != len(selected): raise ValueError('Duplicate selected identities.')
    name = result[{'pipeline':'pipeline_name','featureimp':'fi_name','training':'training_name'}[workflow]]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', name): raise ValueError('Run name must contain 1-80 letters, digits, dots, underscores or hyphens, starting with a letter or digit.')
    from .desktop_jobs import linux_path
    for key in ('data_remote','jobs_remote','python','resources','model16','model32','model64'):
        if result[key]: linux_path(result[key])
    if workflow == 'pipeline':
        if result['new_only'] and result['speed'] != 'fast': raise ValueError('New recordings mode requires Fast batching.')
        models = [result[k] for k in ('model16','model32','model64')]
        if any(models) and not all(models): raise ValueError('Supply all three model paths or leave all blank.')
        if result['feature_workers'] > result['features_cpu']: raise ValueError('Feature workers exceed the allocated CPUs.')
        for key in ('pp_num_workers','pp_noise_workers'):
            if result[key] > result['posepred_cpu']: raise ValueError(f'{key} exceeds the allocated CPUs.')
    if not result['fi_override']:
        result['fi_dataset'] = result['dataset']
        result['matrix'] = str(Path(result['dataset'])/'8_master_matrix/MASTER_FEATURE_MATRIX.csv') if result['dataset'] else ''
    if workflow == 'featureimp':
        if not all(result[k] for k in ('group_a','group_b','control_group')): raise ValueError('Choose training groups A and B and the normalization reference.')
        if result['group_a'] == result['group_b']: raise ValueError('Training groups A and B must differ.')
        if result['fi_opt_fsea_min_size'] > result['fi_opt_fsea_max_size']: raise ValueError('FSEA minimum set size exceeds maximum.')
    if workflow == 'training':
        from .desktop_training import option_arguments
        _, args = option_arguments({k.removeprefix('train_opt_'): v for k,v in result.items() if k.startswith('train_opt_')})
        if args.num_workers > result['train_cpus']: raise ValueError('Training loader workers exceed the allocated CPUs.')
        if result['training_mode'] != 'supervised' and args.swa_mode: raise ValueError('SWA requires supervised training.')
    project_key = {'pipeline':'dataset','featureimp':'fi_dataset','training':'training_directory'}[workflow]
    if not result[project_key]: raise ValueError('Choose a project/data folder.')
    if check_inputs:
        project = Path(result[project_key])
        if not project.is_dir(): raise ValueError(f'Folder is unavailable: {project}')
        if workflow == 'pipeline':
            stage = {'crop':'1_full_movies','dlc':'2_cropped_movies','features':'3_dlc'}[result['start_at']]
            if not (project/stage).is_dir(): raise ValueError(f'Missing starting folder: {project/stage}')
        if workflow == 'featureimp':
            from .desktop_jobs import matrix_inventory
            rows = matrix_inventory(result['matrix'])
            identities = {(r['individual'],r['group'],r['sex']) for r in rows}
            if selected is not None and not set(map(tuple,selected)) <= identities: raise ValueError('Selected identities are not all in the matrix.')
            usable = [r for r in rows if (selected is None or (r['individual'],r['group'],r['sex']) in set(map(tuple,selected))) and (result['sex']=='all' or r['sex'].lower()==result['sex'])]
            groups = {r['group'] for r in usable}
            if not {result[k] for k in ('group_a','group_b','control_group')} <= groups: raise ValueError('Selection/sex filter leaves a required group empty.')
        if workflow == 'training' and not Path(result['training_metadata']).is_file(): raise ValueError('Training metadata file is unavailable.')
    return result


def changed_advanced(config, workflow):
    specs = specifications()
    return {key: value for key,value in config.items() if key in specs
            and specs[key].get('workflow') == workflow and specs[key].get('group')
            and value != specs[key]['default']}


def featureimp_options(config):
    return {spec['native']: validate_value(key, config.get(key, spec['default']), spec)
            for key,spec in advanced_specs().items() if spec['workflow']=='featureimp'}


def featureimp_arguments(config):
    specs = advanced_specs()
    return [word for name,value in featureimp_options(config).items()
            for word in (specs['fi_opt_'+name]['flag'], str(value))]


def update_posepred_template(content, config):
    """Set numeric/boolean options only; never accept arbitrary shell fragments."""
    for key,spec in advanced_specs().items():
        if spec['workflow'] != 'pipeline': continue
        value = validate_value(key, config.get(key, spec['default']), spec)
        name = spec['native']
        if spec['type']=='boolean':
            content, count = re.subn(r'--(?:no[-_])?use_tta\b', '--use_tta' if value else '--no_use_tta', content)
        else:
            content, count = re.subn(r'--'+name+r'\s+[-+\d.eE]+', '--'+name+' '+str(value), content)
        if not count:
            marker = '--resource-monitor '
            if marker not in content: raise ValueError('PosePred template lacks its command marker.')
            content = content.replace(marker, '--'+name+' '+str(value)+' '+marker, 1)
    return content
