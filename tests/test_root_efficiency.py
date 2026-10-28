"""Check cached inputs, worker cleanup and stable numerical results."""
import hashlib
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from autobehaver import completion
from autobehaver.datasets import KeypointsDataset, build_dataloader
from autobehaver.portable_paths import original_name, record_original_name
from autobehaver.scratch import scratch_workspace


def _inputs(directory, lengths=(9, 11, 7)):
    directory.mkdir(exist_ok=True, parents=True)
    rows = []
    for i, length in enumerate(lengths):
        filename, labels = f'f{i}.h5', f'l{i}.h5'
        with h5py.File(directory / filename, 'w') as hf:
            hf['data'] = (np.arange(length * 4).reshape(length, 4) / 13).astype(np.float64)
            hf.attrs['columns'] = np.array(['x', 'y', 'speed', 'angle'], dtype='S')
            hf['source_frame_indices'] = np.arange(5, 5 + length)
            hf.attrs['original_frame_count'] = length + 10
        pose = np.resize(np.array(['B', 'A', 'C'], dtype='S3'), length)
        orient = np.resize(np.array(['Left', 'Right'], dtype='S5'), length)
        pose[3] = b'nan'
        orient[4] = b''
        with h5py.File(directory / labels, 'w') as hf:
            hf['Behavior'], hf['Orientation'] = pose, orient
            hf['source_frame_indices'] = np.arange(5, 5 + length)
        rows.append({'filename': filename, 'labels': labels, 'set': 'val' if i == 2 else 'train'})
    return pd.DataFrame(rows)


def test_memmap_spawn_loader_matches_serial_and_cleans(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOBEHAVER_SCRATCH_DIR', str(tmp_path))
    metadata = _inputs(tmp_path / 'in', (9,))
    args = (str(tmp_path / 'in'), metadata, 5)
    serial = KeypointsDataset(*args, 0, True, True, False, False)
    disk = KeypointsDataset(*args, 1, True, True, False, False)
    store = disk.data_in_memory
    directory = store.directory
    loader = torch.utils.data.DataLoader(disk, batch_size=2, num_workers=1,
                                        multiprocessing_context='spawn')
    actual = list(loader)
    expected = list(build_dataloader(serial, 2, False, 0, pin_memory=False))
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(isinstance(value, str) for value in pickle.loads(pickle.dumps(store)).paths.values())
    del loader, actual
    store.close()
    assert not directory.exists()


def test_checkpoint_cache_reuses_deserialization_and_invalidates(tmp_path, monkeypatch):
    from autobehaver.inference import load_inference_checkpoint
    path = tmp_path / 'model.pth'
    torch.save({'global_vars': {'value': 1}, 'optimizer_state_dict': {'large': torch.ones(10)}}, path)
    original = torch.load
    calls = []
    def count(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', count)
    with completion.verified_identity_scope():
        a = load_inference_checkpoint(path)
        assert 'optimizer_state_dict' not in a
        assert load_inference_checkpoint(path) is a and len(calls) == 1
        torch.save({'global_vars': {'value': 2}}, path)
        assert load_inference_checkpoint(path)['global_vars']['value'] == 2
        assert len(calls) == 2
    assert completion.current_identity_scope() is None


def test_digest_cache_change_replacement_and_corruption(tmp_path, monkeypatch):
    path = tmp_path / 'data.bin'
    path.write_bytes(b'first')
    real_open = open
    calls = []
    def count(pathname, *args, **kwargs):
        calls.append(str(pathname))
        return real_open(pathname, *args, **kwargs)
    monkeypatch.setattr(completion, 'open', count, raising=False)
    with completion.verified_identity_scope():
        first = completion.file_digest(path)
        assert completion.file_digest(path) == first and len(calls) == 1
        identity = {'purpose': 'tiny'}
        completion.write_receipt(path, identity)
        assert completion.receipt_matches(path, identity)
        path.write_bytes(b'other')
        assert not completion.receipt_matches(path, identity)
        assert completion.file_digest(path) == hashlib.sha256(b'other').hexdigest()
        replacement = tmp_path / 'replace'
        replacement.write_bytes(b'third')
        os.replace(replacement, path)
        assert completion.file_digest(path) == hashlib.sha256(b'third').hexdigest()
    completion.file_digest(path)
    assert len(calls) == 4


def test_directory_scoped_mapping_and_refresh(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    record_original_name(a / 'same.csv', 'biological_A.csv')
    record_original_name(b / 'same.csv', 'biological_B.csv')
    with completion.verified_identity_scope():
        assert original_name(a / 'same.csv') == 'biological_A.csv'
        assert original_name(b / 'same.csv') == 'biological_B.csv'
        record_original_name(a / 'second.csv', 'biological_C.csv')
        assert original_name(a / 'second.csv') == 'biological_C.csv'


def test_owned_scratch_cleanup_and_error_retention(tmp_path):
    with scratch_workspace('tiny', scratch_dir=tmp_path) as successful:
        (successful / 'data').write_text('intermediate')
    assert not successful.exists()
    with pytest.raises(ValueError):
        with scratch_workspace('tiny', scratch_dir=tmp_path) as failed:
            raise ValueError('controlled failure')
    assert failed.is_dir()
    with pytest.raises(RuntimeError, match='ownership'):
        with scratch_workspace('tiny', scratch_dir=tmp_path) as changed:
            (changed / '.autobehaver-scratch.json').unlink()
    assert changed.is_dir()


@pytest.mark.parametrize('fail_setup', [False, True])
def test_pretraining_owner_cleanup_without_chunks(tmp_path, monkeypatch, fail_setup):
    from autobehaver import pretraining
    tmp_path = tmp_path.parent / ('pa' if fail_setup else 'pb')
    tmp_path.mkdir()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (tmp_path / 'checkpoints').mkdir()
    monkeypatch.setenv('AUTOBEHAVER_SCRATCH_DIR', str(scratch))
    metadata = _inputs(tmp_path / 'in', (8,))
    if fail_setup:
        def fail(*args, **kwargs):
            raise RuntimeError('controlled setup failure')
        monkeypatch.setattr(pretraining, 'model_setup', fail)
    args = dict(directory=str(tmp_path / 'in'), metadata=metadata, sequence_length=5, batch=2,
                num_workers=1, num_epochs=0, checkpoint_interval=10, log_interval=10,
                checkpoint_dir=str(tmp_path / 'checkpoints'), run_name='tiny', projection_dim=8,
                temperature=0.1, pretrain_files_per_group=1, mask_p=0.1, noise_level=0.1,
                records_path=str(tmp_path / 'records.csv'), d_model=8, nhead=2,
                num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, dropout_rate=0.1,
                frame_representation='middle', num_cls=1, prediction_head='fc',
                positional_encoding_type='learned', base_lr=0.001, warmup_steps=1,
                patience=2, plateau_factor=0.5, pretrain_loss='info_nce', barlow_lambda=0.005,
                scheduler_type='cosine', weight_decay=0.0, t_0=10, t_mult=1, eta_min=0.0,
                grad_clip_norm=1.0, random_seed=123, metadata_tag='tiny')
    if fail_setup:
        with pytest.raises(RuntimeError, match='controlled setup'):
            pretraining.run_pretraining(**args)
        assert len(list(scratch.glob('ab_pretraining_*'))) == 1
    else:
        pretraining.run_pretraining(**args)
        assert not list(scratch.glob('ab_pretraining_*'))
        assert (tmp_path / 'checkpoints/tiny_final_pretrain.pth').is_file()


def test_persistent_training_workers_close_on_error(tmp_path, monkeypatch):
    from autobehaver.window_index import close_feature_stores
    tmp_path = tmp_path.parent / 'wc'
    tmp_path.mkdir()
    monkeypatch.setenv('AUTOBEHAVER_SCRATCH_DIR', str(tmp_path))
    metadata = _inputs(tmp_path / 'in', (8,))
    dataset = KeypointsDataset(str(tmp_path / 'in'), metadata, 5, 1, True, True, False, False)
    store = dataset.data_in_memory
    capture = {}
    @close_feature_stores
    def fail(train_dataset, val_dataset=None):
        loader = build_dataloader(train_dataset, 2, False, 1, pin_memory=False)
        capture['loader'] = loader
        next(iter(loader))
        capture['workers'] = list(loader._iterator._workers)
        raise RuntimeError('controlled training failure')
    with pytest.raises(RuntimeError, match='controlled training'):
        fail(dataset)
    assert capture['loader']._iterator._shutdown
    assert all(not worker.is_alive() for worker in capture['workers'])
    assert store.directory.is_dir() and store._context is None


def test_directory_index_retries_when_maps_arrive_during_scan(tmp_path, monkeypatch):
    from autobehaver import portable_paths
    record_original_name(tmp_path / 'first.csv', 'biological_A.csv')
    original = portable_paths._map_signature
    calls = []
    def signature(directory):
        calls.append(directory)
        if len(calls) == 2:
            record_original_name(tmp_path / 'late.csv', 'biological_B.csv')
        return original(directory)
    monkeypatch.setattr(portable_paths, '_map_signature', signature)
    with completion.verified_identity_scope():
        values = portable_paths.load_name_maps(tmp_path)
        assert values['late.csv'] == 'biological_B.csv'


def test_complete_inference_stage_loads_each_checkpoint_once(tmp_path, monkeypatch):
    from autobehaver.model import PosePredTransformer
    from autobehaver.preinf.posepred_inference.config import PosePredInferenceConfig
    from autobehaver.preinf.posepred_inference.core import run_posepred_inference
    tmp_path = tmp_path.parent / 'st'
    tmp_path.mkdir()
    _inputs(tmp_path / 'in', (8,))
    (tmp_path / 'in/l0.h5').unlink()
    gv = dict(num_features=4, sequence_length=5, d_model=8, nhead=2,
              num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16,
              dropout_rate=0.1, frame_representation='middle', num_cls=1,
              prediction_head='fc', pose_pred=True, orient_pred=True, recon=False,
              positional_encoding_type='learned')
    paths = []
    for i in range(3):
        model = PosePredTransformer(**gv, num_pose_classes=2, num_orient_classes=2)
        path = tmp_path / f'm{i}.pth'
        torch.save({'checkpoint_kind': 'supervised', 'checkpoint_schema_version': 1,
                    'epoch': 0, 'global_step': 0,
                    'global_vars': {**gv, 'feature_columns_included': ['x', 'y', 'speed', 'angle']}, 'model_state_dict': model.state_dict(),
                    'pose_dict': {'A': 0, 'B': 1}, 'orient_dict': {'Left': 0, 'Right': 1}}, path)
        paths.append(str(path))
    original = torch.load
    calls = []
    def count(*args, **kwargs):
        calls.append(str(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', count)
    cfg = PosePredInferenceConfig(input_dir=str(tmp_path / 'in'), output_dir=str(tmp_path / 'out'),
                                 inference_model=paths, num_workers=0, batch=2,
                                 min_pose_bout=1, min_orient_bout=1, sequence_length=5,
                                 write_run_log=False)
    run_posepred_inference(cfg)
    assert sorted(calls) == sorted(paths)
    assert (tmp_path / 'out/f0_inference.csv.provenance.json').is_file()
