"""Named local setup drafts, kept separate from immutable built job records."""
from pathlib import Path
from datetime import datetime
import json
import re
from uuid import uuid4
from .desktop_jobs import save_json


def identity(kind, config):
    name_key = {'pipeline': 'pipeline_name', 'featureimp': 'fi_name', 'training': 'training_name'}[kind]
    project_key = {'pipeline': 'dataset', 'featureimp': 'fi_dataset', 'training': 'training_directory'}[kind]
    return kind, str(config.get(project_key, '')).replace('\\', '/').rstrip('/'), config.get(name_key, '')


def write_draft(state_dir, kind, config, existing=None):
    path = Path(existing) if existing else None
    if path and path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous.get('record_type') != 'draft' or identity(previous['kind'], previous['config']) != identity(kind, config):
            path = None
    if path is None:
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', identity(kind, config)[2]).strip('._') or kind
        path = Path(state_dir) / 'saved_runs' / (name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, dict(format=1, record_type='draft', kind=kind,
                        name=identity(kind, config)[2], config=config))
    return path
