"""Identify DeepLabCut model settings and selected snapshot content."""
from pathlib import Path
import yaml
from autobehaver.completion import request_identity, file_digest


def model_identity(config_path, shuffle):
    """Identify one selected snapshot and its inference-relevant model settings."""
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    root = Path(config.get('project_path') or Path(config_path).parent)
    if not root.exists():
        root = Path(config_path).parent
    folders = [p for p in (root/'dlc-models'/f'iteration-{config.get("iteration",0)}').glob('*/train') if f'shuffle{shuffle}' in p.parent.name]
    fractions = config.get('TrainingFraction', [])
    if fractions:
        folders = [p for p in folders if f'trainset{int(fractions[0]*100)}shuffle{shuffle}' in p.parent.name]
    if len(folders) != 1:
        raise ValueError(f'Expected one DLC model folder for shuffle {shuffle}; found {len(folders)}')
    folder=folders[0]
    snapshots=sorted(folder.glob('snapshot-*.index'), key=lambda p:int(p.stem.split('-')[-1]))
    if not snapshots:
        raise FileNotFoundError(f'No DLC snapshot in {folder}')
    selected=config.get('snapshotindex',-1)
    if selected == 'all':
        raise ValueError('Inference requires one DLC snapshot; select snapshotindex instead of all')
    snapshot=snapshots[int(selected)]
    assets=sorted(folder.glob(snapshot.stem+'.*'))
    semantic={k:v for k,v in config.items() if k not in ('project_path','video_sets','video_sets_original','date','scorer','TrainingFraction')}
    train_cfg=yaml.safe_load((folder/'pose_cfg.yaml').read_text())
    semantic['pose_cfg']={k:v for k,v in train_cfg.items() if k not in ('project_path','dataset','metadataset','init_weights','log_dir')}
    test_path = folder.parent / 'test' / 'pose_cfg.yaml'
    test_cfg = yaml.safe_load(test_path.read_text(encoding='utf-8'))
    semantic['test_pose_cfg'] = {k: v for k, v in test_cfg.items()
                                if k not in ('project_path', 'dataset', 'metadataset', 'init_weights', 'log_dir')}
    semantic['shuffle']=int(shuffle)
    return {'settings':semantic, 'snapshot_files':[file_digest(p) for p in assets]}


def dlc_request(video, model):
    """Build a request identity for a video, model, and inference implementation."""
    folder=Path(__file__).parent
    return request_identity('dlc-v1',[video],model,
        [folder/name for name in ('core.py','contract.py','backend.py','_backend_worker.py')])
