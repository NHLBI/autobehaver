"""Restore saved job settings without applying new-run default migrations."""
from pathlib import Path
import re


def restored_values(record, defaults):
    config = record['config']
    values = dict(defaults)
    values.update(config)
    name = {'pipeline':'pipeline_name', 'featureimp':'fi_name', 'training':'training_name'}[record['kind']]
    values[name] = config.get('run_name', config.get(name, ''))
    if record['kind'] == 'training':
        values['training_directory'] = config['dataset']
    for stage, allocation in config.get('allocations', {}).items():
        values[stage+'_cpu'], values[stage+'_ram'] = allocation[:2]
    if 'mmgen_allocation_version' not in config and 'mmgen_finish' not in config.get('allocations', {}):
        values['mmgen_finish_cpu'] = values['mmgen_cpu']
        values['mmgen_finish_ram'] = values['mmgen_ram']
    for stage in ('mmgen','mmgen_finish'):
        hours = config.get('walltime_hours', {}).get(stage, config.get(stage+'_hours'))
        if hours is None:
            template = Path(record['prepared'])/'templates'/(stage+'.sbatch')
            if not template.exists() and stage == 'mmgen_finish':
                template = template.with_name('mmgen.sbatch')
            if template.is_file():
                match = re.search(r'^#SBATCH --time=(\d+):00:00\s*$', template.read_text(encoding='utf-8'), re.M)
                if match: hours = int(match[1])
        if hours is not None: values[stage+'_hours'] = hours
    for key, value in (config.get('batching') or {}).items(): values[key] = value
    legacy = config.get('batch_sizes') if not config.get('batching') else None
    values['saved_batch_sizes'] = dict(legacy or config.get('saved_batch_sizes') or {})
    values['use_saved_batch_sizes'] = bool(legacy or config.get('use_saved_batch_sizes'))
    if 'full_movies' not in config: values['full_movies'] = config.get('crop_batch', (legacy or {}).get('crop', values['full_movies']))
    if 'chambers' not in config: values['chambers'] = config.get('dlc_batch', (legacy or {}).get('dlc', values['chambers']))
    if 'fi_sharded' not in config: values['fi_sharded'] = False
    for key,value in config.get('training_options', {}).items(): values['train_opt_'+key] = value
    for key,value in zip(('model16','model32','model64'),config.get('models') or []):
        values[key] = value
    return values
