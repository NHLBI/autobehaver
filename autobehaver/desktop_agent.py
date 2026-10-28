"""Inspect settings and prepare a local launcher draft for human review.

This entrypoint never builds, publishes, submits, or connects to HPC.
"""
import argparse
import json
from pathlib import Path

from .desktop_settings import SCHEMA_VERSION, WORKFLOWS, defaults, specifications, validate_config, changed_advanced


def read_json(path):
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result: raise ValueError(f'Duplicate JSON key: {k}')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Invalid number: '+value)))


def load_draft(path, *, check_inputs=False):
    draft = read_json(path)
    if not isinstance(draft,dict): raise ValueError('Draft must be a JSON object.')
    if draft.get('format') != 1 or draft.get('record_type') != 'draft': raise ValueError('Choose a saved draft, not a built job record.')
    if draft.get('settings_schema', SCHEMA_VERSION) != SCHEMA_VERSION: raise ValueError('Unsupported draft settings schema.')
    draft['config'] = validate_config(draft['config'], draft['kind'], check_inputs=check_inputs)
    return draft


def write_draft(config, workflow, output, *, check_inputs=False):
    config = validate_config(config,workflow,check_inputs=check_inputs)
    draft = dict(format=1,record_type='draft',settings_schema=SCHEMA_VERSION,kind=workflow,
                 name=config[{'pipeline':'pipeline_name','featureimp':'fi_name','training':'training_name'}[workflow]],config=config)
    path = Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects saved drafts and built run records alike.
    with path.open('x',encoding='utf-8') as stream: json.dump(draft,stream,indent=2); stream.write('\n')
    return path.resolve()


def inspect_project(path):
    """Read directory names and matrix metadata, not recordings or feature values."""
    from .desktop_jobs import matrix_inventory
    project = Path(path)
    if not project.is_dir(): raise ValueError('Project directory is unavailable.')
    result = dict(project=str(project.resolve()), folders=sorted(p.name for p in project.iterdir() if p.is_dir()))
    matrix = project/'8_master_matrix/MASTER_FEATURE_MATRIX.csv'
    if matrix.is_file():
        rows = matrix_inventory(matrix)
        result.update(matrix=str(matrix), individuals=len(rows), groups=sorted({r['group'] for r in rows}),
                      sexes=sorted({r['sex'] for r in rows}), identities=[[r[k] for k in ('individual','group','sex')] for r in rows])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('settings',help='Print types, defaults, limits and help as JSON.')
    sub.add_parser('defaults',help='Print a blank GUI configuration as JSON.')
    inspect=sub.add_parser('inspect',help='Read project folders and matrix groups/identities.')
    inspect.add_argument('project',type=Path)
    for name in ('validate','draft'):
        p=sub.add_parser(name)
        p.add_argument('--workflow',choices=WORKFLOWS,required=True)
        p.add_argument('--config',type=Path,required=True,help='JSON object of GUI settings.')
        p.add_argument('--check-inputs',action='store_true',help='Also check local starting folders and matrix selection.')
        if name=='draft': p.add_argument('--output',type=Path,required=True,help='New JSON file; existing files are never overwritten.')
    args=parser.parse_args(argv)
    try:
        if args.command=='settings': result=dict(settings_schema=SCHEMA_VERSION,settings=specifications())
        elif args.command=='defaults': result=defaults()
        elif args.command=='inspect': result=inspect_project(args.project)
        else:
            config=validate_config(read_json(args.config),args.workflow,check_inputs=args.check_inputs)
            result=dict(valid=True,workflow=args.workflow,advanced_changes=changed_advanced(config,args.workflow))
            if args.command=='draft': result['draft']=str(write_draft(config,args.workflow,args.output))
        print(json.dumps(result,indent=2))
    except (ValueError,OSError,KeyError) as exc: parser.exit(2,str(exc)+'\n')


if __name__=='__main__': main()
