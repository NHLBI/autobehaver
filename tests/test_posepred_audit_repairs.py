"""Tiny CPU integration checks for checkpoint, training and metadata repairs."""
import argparse
import csv
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from autobehaver import _training_core as training
from autobehaver.checkpoint import load_checkpoint_variables
from autobehaver.feature_selection import resolve_feature_plan


def _files(root):
    root.mkdir(exist_ok=True)
    columns = ['x', 'y', 'speed', 'angle']
    for i, name in enumerate(('a', 'b')):
        path = root / f'{name}.h5'
        if path.exists():
            continue
        with h5py.File(path, 'w') as handle:
            handle['data'] = np.random.default_rng(i).normal(size=(9, 4)).astype(np.float32)
            handle.attrs['columns'] = columns
        with h5py.File(root / f'{name}_labels.h5', 'w') as handle:
            handle['Behavior'] = np.asarray(['rest', 'walk', 'rest'] * 3, dtype='S')
            handle['Orientation'] = np.asarray(['floor', 'wall', 'floor'] * 3, dtype='S')
            handle['Frame'] = np.arange(9)
    metadata = pd.DataFrame({'filename': ['a.h5', 'b.h5'], 'labels': ['a_labels.h5', 'b_labels.h5'],
                             'set': ['train', 'val']})
    return metadata, resolve_feature_plan(columns, preset='none')


def _training_args(tmp_path, *, checkpoint=None, **overrides):
    torch.set_num_threads(1)
    metadata, plan = _files(tmp_path / 'data')
    output = tmp_path / 'c'
    output.mkdir(exist_ok=True)
    args = dict(num_workers=0, sequence_length=5, batch=2, pretrained_model_path=checkpoint,
        num_features=4, d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=16, dropout_rate=0., frame_representation='middle', num_cls=1,
        prediction_head='fc', records_path=str(tmp_path / 'records.csv'), wandb_log=False,
        num_epochs=3, checkpoint_interval=99, checkpoint_dir=str(output), run_name='tiny',
        pose_conf_matrix_path=str(tmp_path / 'pose.csv'), orient_conf_matrix_path=str(tmp_path / 'orient.csv'),
        combined_conf_matrix_path=str(tmp_path / 'combined.csv'), pose_pred=True, orient_pred=False,
        recon=False, positional_encoding_type='learned', pose_weight=1., orient_weight=1., recon_weight=1.,
        patience=20, plateau_factor=.5, freeze_encoder=False, weight_encoder=1., restart_optimizer=False,
        noval=False, base_lr=.01, weight_decay=0., scheduler_type='cosine', warmup_steps=1, t_0=30,
        t_mult=1, eta_min=0., mask_p=0., noise_level=0., max_class_weight=7., grad_clip_norm=1.,
        random_seed=4, metadata_tag='synthetic', feature_plan=plan)
    args.update(overrides)
    prepared = training.prepare_data(str(tmp_path / 'data'), metadata, args['sequence_length'],
        num_workers=0, random_seed=4, pretrained_model_path=checkpoint, pose_pred=args['pose_pred'],
        orient_pred=args['orient_pred'], recon=args['recon'], defined_split=True, noval=args['noval'],
        max_class_weight=args['max_class_weight'], feature_plan=plan, restart_optimizer=args['restart_optimizer'])
    for key, value in zip(('train_dataset', 'val_dataset', 'pose_dict', 'orient_dict', 'columns_to_noise',
                           'train_class_weights', 'val_class_weights', 'split_filenames'), prepared):
        args[key] = value
    return args


def _restore_for_worker(checkpoint):
    # Deliberately incompatible command defaults must be replaced BEFORE loaders/schedulers.
    args = argparse.Namespace(batch=99, scheduler_type='linear', num_epochs=99,
        warmup_epochs=None, cosine_decay_epochs=None, weight_encoder=1., freeze_encoder=False)
    load_checkpoint_variables(args, str(checkpoint), use_restart_optimizer=False, current_num_features=4)
    accepted = inspect.signature(training.main_worker).parameters
    return {key: value for key, value in vars(args).items() if key in accepted}


def test_epoch_cosine_full_resume_restores_before_construction_and_reaches_noninterval_endpoint(tmp_path, monkeypatch):
    from autobehaver.schedulers import EpochCosineScheduler
    traces = []
    setup = training.model_setup
    def capture(*args, **kwargs):
        model, optimizer, scheduler, scaler = setup(*args, **kwargs)
        assert isinstance(scheduler, EpochCosineScheduler)
        trace = []
        traces.append(trace)
        optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: trace.append([g['lr'] for g in optimizer.param_groups]))
        return model, optimizer, scheduler, scaler
    monkeypatch.setattr(training, 'model_setup', capture)
    initial = _training_args(tmp_path, num_epochs=5, checkpoint_interval=2,
        warmup_epochs=1, cosine_decay_epochs=4, weight_encoder=.2)
    training.main_worker(**initial)
    endpoint = tmp_path / 'c/tiny_model_epoch_5.pth'
    saved = torch.load(endpoint, weights_only=False)
    assert saved['epoch'] == 4 and saved['global_step'] == 15
    assert saved['global_vars']['max_class_weight'] == 7
    resume = tmp_path / 'c/tiny_model_epoch_2.pth'
    restored = _restore_for_worker(resume)
    assert (restored['batch'], restored['num_epochs'], restored['cosine_decay_epochs']) == (2, 5, 4)
    training.main_worker(**_training_args(tmp_path, checkpoint=str(resume), **restored))
    np.testing.assert_allclose(traces[1], traces[0][6:], rtol=0, atol=1e-15)
    assert torch.load(endpoint, weights_only=False)['global_step'] == 15
    assert pd.read_csv(tmp_path / 'records.csv')['Epoch'].tolist() == [1, 2, 3, 4, 5]


def test_plateau_warmup_precedes_updates_and_reconstruction_decoder_is_optimized(tmp_path, monkeypatch):
    states = []
    setup = training.model_setup
    def capture(*args, **kwargs):
        model, optimizer, scheduler, scaler = setup(*args, **kwargs)
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
        state = {'model': model, 'before': model.output_linear.weight.detach().clone(), 'lrs': []}
        optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: state['lrs'].append([g['lr'] for g in optimizer.param_groups]))
        states.append(state)
        return model, optimizer, scheduler, scaler
    monkeypatch.setattr(training, 'model_setup', capture)
    training.main_worker(**_training_args(tmp_path, num_epochs=1, scheduler_type='plateau',
        weight_encoder=.2, recon=True, mask_p=.5, noise_level=.02))
    state = states[0]
    np.testing.assert_allclose(state['lrs'], [[.002 / 3, .01 / 3], [.004 / 3, .02 / 3], [.002, .01]])
    assert not torch.equal(state['before'], state['model'].output_linear.weight.detach())


def test_frozen_encoder_policy_is_reconstructed_on_supervised_resume(tmp_path, monkeypatch):
    frozen = []
    setup = training.model_setup
    def capture(*args, **kwargs):
        result = setup(*args, **kwargs)
        frozen.append(all(not p.requires_grad for p in result[0].transformer_encoder.parameters()))
        return result
    monkeypatch.setattr(training, 'model_setup', capture)
    training.main_worker(**_training_args(tmp_path, num_epochs=2, checkpoint_interval=1, freeze_encoder=True))
    resume = tmp_path / 'c/tiny_model_epoch_1.pth'
    training.main_worker(**_training_args(tmp_path, checkpoint=str(resume), **_restore_for_worker(resume)))
    assert frozen == [True, True]


@pytest.mark.parametrize('pose', [True, False])
def test_single_head_swa_and_no_epoch_resume_finalize_with_configured_augmentation(tmp_path, monkeypatch, pose):
    validation = training.validate_one_epoch
    seen = []
    def capture(*args, **kwargs):
        seen.append((args[-2], args[-1]))
        return validation(*args, **kwargs)
    monkeypatch.setattr(training, 'validate_one_epoch', capture)
    training.main_worker(**_training_args(tmp_path, pose_pred=pose, orient_pred=not pose,
        recon=True, mask_p=.25, noise_level=.15, swa_mode=True, swa_epochs=1, swa_lr=.002))
    terminal = tmp_path / 'c/tiny_model_epoch_3.pth'
    exported = tmp_path / 'c/tiny_swa_model.pth'
    saved = torch.load(exported, weights_only=False)
    assert saved['epoch'] == 2 and saved['swa_n_averaged'] == 1
    assert saved['global_vars']['feature_schema_hash']
    assert 'optimizer_state_dict' not in saved
    assert all(not name.startswith('module.') for name in saved['model_state_dict'])
    exported.unlink()  # Simulate loss/failure of only the final derived export.
    seen.clear()
    monkeypatch.setattr(training, 'train_one_epoch', lambda *args, **kwargs: pytest.fail('completed SWA run repeated training'))
    training.main_worker(**_training_args(tmp_path, checkpoint=str(terminal), **_restore_for_worker(terminal)))
    assert seen == [(.25, .15)]
    recovered = torch.load(exported, weights_only=False)
    assert recovered['global_step'] == saved['global_step']
    for name, value in saved['model_state_dict'].items():
        assert torch.equal(value, recovered['model_state_dict'][name])
    rows = pd.read_csv(tmp_path / 'records.csv')
    assert rows['Epoch'].tolist() == ['1', '2', '3', 'SWA_FINAL']


def test_early_stop_records_and_checkpoints_the_stopping_epoch(tmp_path, monkeypatch):
    validate = training.validate_one_epoch
    def fixed_score(*args, **kwargs):
        values = list(validate(*args, **kwargs))
        values[10] = .5
        return tuple(values)
    monkeypatch.setattr(training, 'validate_one_epoch', fixed_score)
    training.main_worker(**_training_args(tmp_path, num_epochs=6, patience=1))
    stopped = torch.load(tmp_path / 'c/tiny_model_epoch_2.pth', weights_only=False)
    assert stopped['epoch'] == 1 and stopped['training_runtime_state']['num_completed_epochs'] == 2
    assert pd.read_csv(tmp_path / 'records.csv')['Epoch'].tolist() == [1, 2]


def test_noval_records_do_not_report_training_scores_as_validation(tmp_path):
    training.main_worker(**_training_args(tmp_path, num_epochs=1, noval=True))
    rows = list(csv.DictReader((tmp_path / 'records.csv').open(newline='')))
    assert rows[0]['Validation Available'] == '0'
    assert all(value == '' for key, value in rows[0].items() if key.startswith('Val '))
    assert float(rows[0]['Train Total Loss Avg']) > 0


def test_atomic_checkpoint_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    from autobehaver import checkpoint
    path = tmp_path / 'best.pth'
    path.write_bytes(b'previous complete checkpoint')
    def fail_save(value, file):
        file.write(b'partial')
        raise OSError('synthetic serialization failure')
    monkeypatch.setattr(checkpoint.torch, 'save', fail_save)
    with pytest.raises(OSError, match='synthetic'):
        checkpoint.atomic_save_checkpoint({'model': 1}, path)
    assert path.read_bytes() == b'previous complete checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def test_failed_new_best_save_does_not_delete_previous_best(tmp_path, monkeypatch):
    saved = training.save_checkpoint
    open_records = []
    setup_records = training.setup_records_file
    def track_records(*args, **kwargs):
        result = setup_records(*args, **kwargs)
        open_records.append(result[0])
        return result
    def fail_second_best(*args, **kwargs):
        if kwargs.get('filename_override') == 'tiny_best_model_epoch_2.pth':
            raise OSError('synthetic new-best failure')
        return saved(*args, **kwargs)
    scores = iter([.5, .6])
    monkeypatch.setattr(training, 'setup_records_file', track_records)
    monkeypatch.setattr(training, 'save_checkpoint', fail_second_best)
    monkeypatch.setattr(training, '_select_primary_validation_metric', lambda **kwargs: (next(scores), 'pose_macro_f1'))
    try:
        with pytest.raises(OSError, match='new-best failure'):
            training.main_worker(**_training_args(tmp_path))
    finally:
        for stream in open_records:
            if stream is not None:
                stream.close()
    previous = tmp_path / 'c/tiny_best_model_epoch_1.pth'
    assert torch.load(previous, weights_only=False)['epoch'] == 0


def test_explicit_missing_checkpoint_is_not_treated_as_fresh_training(tmp_path):
    from autobehaver.checkpoint import load_checkpoint_model
    missing = str(tmp_path / 'missing.pth')
    with pytest.raises(FileNotFoundError, match='Requested checkpoint'):
        load_checkpoint_variables(argparse.Namespace(), missing, False)
    with pytest.raises(FileNotFoundError, match='Requested checkpoint'):
        load_checkpoint_model(None, None, None, missing, None)
    with pytest.raises(FileNotFoundError, match='Requested checkpoint'):
        training.prepare_data(str(tmp_path), pd.DataFrame(), 5, pretrained_model_path=missing)


def test_lr_finder_rejects_equal_width_different_feature_meaning_before_preparation(tmp_path, monkeypatch):
    from autobehaver import lr_finder
    from tests.test_posepred_inference_audit import _make_supervised_checkpoint
    metadata, _ = _files(tmp_path / 'data')
    checkpoint = tmp_path / 'weights.pth'
    _make_supervised_checkpoint(checkpoint, columns=['x', 'y', 'different_speed', 'angle'], pose_dict={'rest': 0, 'walk': 1})
    monkeypatch.setattr(lr_finder, 'prepare_data', lambda **kwargs: pytest.fail('schema mismatch reached data/model setup'))
    args = argparse.Namespace(ablation_preset='none', pretrained_model_path=str(checkpoint), sequence_length=5)
    with pytest.raises(ValueError, match='feature_schema_hash mismatch'):
        lr_finder.find_lr(args=args, directory=str(tmp_path / 'data'), metadata=metadata)


def test_empty_record_file_recovers_and_changed_heads_migrate_common_metrics(tmp_path):
    from autobehaver.records import setup_records_file
    path = tmp_path / 'records.csv'
    path.touch()
    file, writer, headers = setup_records_file(str(path), 0, True, False, False)
    assert file is not None
    row = {name: '' for name in headers}
    row.update({'Epoch': '1', 'Train Total Loss Avg': '1.5', 'Val Pose Macro F1': '.4'})
    writer.writerow([row[name] for name in headers])
    file.close()
    file, writer, headers = setup_records_file(str(path), 1, False, True, True)
    file.close()
    rows = list(csv.DictReader(path.open(newline='')))
    assert rows[0]['Train Total Loss Avg'] == '1.5'
    assert rows[0]['Val Orient Macro F1'] == '' and rows[0]['Val Recon Loss Avg'] == ''
    assert 'Val Pose Macro F1' not in headers
    assert len(list((tmp_path / 'previous_records').glob('*'))) == 2


def test_corrected_losses_have_population_correlation_and_differentiable_empty_mask():
    from autobehaver.losses import barlow_twins_loss, masked_mse_loss
    views = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], requires_grad=True)
    loss = barlow_twins_loss(views, views)
    assert loss.item() < 1e-8
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    zero = masked_mse_loss(prediction, torch.zeros_like(prediction), torch.ones_like(prediction, dtype=torch.bool))
    zero.backward()
    assert zero.item() == 0 and torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_relative_split_distinguishes_duplicate_basenames(tmp_path):
    from types import SimpleNamespace
    from autobehaver.window_index import WindowInstructions
    owner = SimpleNamespace(directory=str(tmp_path))
    windows = WindowInstructions(owner, [{'path': str(tmp_path / 'a/same.h5'), 'count': 2},
                                          {'path': str(tmp_path / 'b/same.h5'), 'count': 3}])
    assert windows.indices_for_files(['a/same.h5']).tolist() == [0, 1]
    assert windows.indices_for_files(['b/same.h5']).tolist() == [2, 3, 4]


def test_nested_augmented_movie_cannot_cross_train_validation_boundary(tmp_path):
    import shutil
    root = tmp_path / 'data'
    _, plan = _files(root)
    (root / 'sub').mkdir()
    for prefix in ('original_', 'rotated90_'):
        shutil.copyfile(root / 'a.h5', root / 'sub' / f'{prefix}same.h5')
        shutil.copyfile(root / 'a_labels.h5', root / 'sub' / f'{prefix}labels.h5')
    metadata = pd.DataFrame({'filename': ['sub/original_same.h5', 'sub/rotated90_same.h5', 'b.h5'],
        'labels': ['sub/original_labels.h5', 'sub/rotated90_labels.h5', 'b_labels.h5'], 'set': ['train', 'val', 'train']})
    with pytest.raises(ValueError, match='Augmented versions'):
        training.prepare_data(str(root), metadata, 5, pose_pred=True, orient_pred=False,
            defined_split=True, num_workers=0, feature_plan=plan)
    prepared = training.prepare_data(str(root), metadata, 5, pose_pred=True, orient_pred=False,
        defined_split=False, num_workers=0, feature_plan=plan)
    for names in (prepared[-1]['train_filenames'], prepared[-1]['val_filenames']):
        assert ('sub/original_same.h5' in names) == ('sub/rotated90_same.h5' in names)


def test_pretraining_input_order_content_guard_and_completed_cursor(tmp_path, monkeypatch):
    from autobehaver import pretraining
    torch.set_num_threads(1)
    metadata, plan = _files(tmp_path / 'data')
    checkpoints = tmp_path / 'c'
    checkpoints.mkdir()
    kwargs = dict(directory=str(tmp_path / 'data'), metadata=metadata, sequence_length=5, batch=4,
        num_workers=0, num_epochs=1, checkpoint_interval=99, log_interval=99,
        checkpoint_dir=str(checkpoints), run_name='tiny', projection_dim=4, temperature=.1,
        pretrain_files_per_group=2, mask_p=.1, noise_level=.05, records_path=str(tmp_path / 'records.csv'),
        d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16,
        dropout_rate=0., frame_representation='middle', num_cls=1, prediction_head='fc',
        positional_encoding_type='learned', base_lr=.001, warmup_steps=1, patience=5,
        plateau_factor=.5, pretrain_loss='barlow_twins', barlow_lambda=.005, scheduler_type='cosine',
        weight_decay=0., t_0=10, t_mult=1, eta_min=0., grad_clip_norm=1., random_seed=4,
        metadata_tag='synthetic', feature_plan=plan)
    pretraining.run_pretraining(**kwargs)
    endpoint = checkpoints / 'tiny_final_pretrain.pth'
    saved = torch.load(endpoint, weights_only=False)
    assert saved['pretraining_runtime_state']['resume_epoch'] == 1 and saved['global_step'] > 0
    monkeypatch.setattr(pretraining.PretrainingDataset, '__getitem__', lambda *args: pytest.fail('completed pretraining repeated a batch'))
    pretraining.run_pretraining(**kwargs, checkpoint_path=str(endpoint))
    resumed = torch.load(endpoint, weights_only=False)
    assert resumed['global_step'] == saved['global_step']
    for name, value in saved['model_state_dict'].items():
        assert torch.equal(value, resumed['model_state_dict'][name])
    with pytest.raises(ValueError, match='order|Training inputs changed'):
        pretraining.run_pretraining(**{**kwargs, 'metadata': metadata.iloc[::-1]}, checkpoint_path=str(endpoint))
    with h5py.File(tmp_path / 'data/a.h5', 'r+') as handle:
        handle['data'][0, 0] += 1
    with pytest.raises(ValueError, match='Training inputs changed'):
        pretraining.run_pretraining(**kwargs, checkpoint_path=str(endpoint))


def test_semantic_ablation_rejects_anonymous_headers_and_reports_unmatched_exclusions(caplog):
    with pytest.raises(ValueError, match='anonymous|semantic|headers'):
        resolve_feature_plan(['col_0', 'col_1'], preset='no_pos')
    with caplog.at_level('WARNING'):
        plan = resolve_feature_plan(['speed', 'angle'], preset='none', exclude_columns=['typo'])
    assert plan.included_columns == ['speed', 'angle']
    assert plan.spec['unmatched_exclude_columns'] == ['typo'] and 'matched no columns' in caplog.text


def test_labels_are_canonicalized_consistently_without_changing_source_files(tmp_path):
    metadata, plan = _files(tmp_path / 'data')
    source = tmp_path / 'data/a_labels.h5'
    with h5py.File(source, 'r+') as handle:
        del handle['Behavior']
        handle['Behavior'] = np.asarray([' rest ', ' walk ', 'nan', 'rest', ' walk ', 'rest', 'rest', 'walk', 'rest'], dtype='S')
    before = source.read_bytes()
    prepared = training.prepare_data(str(tmp_path / 'data'), metadata, 5, pose_pred=True,
        orient_pred=False, defined_split=True, num_workers=0, feature_plan=plan)
    assert set(prepared[2]) == {'rest', 'walk'}
    assert len(prepared[0]) == 4  # One center label is missing after canonicalization.
    assert source.read_bytes() == before
