"""Prepare supervised PosePred data and run training, validation and checkpointing."""

from __future__ import annotations

from autobehaver.progress import progress

import os
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Subset
from .window_index import WindowInstructions, close_feature_stores
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .constants import MEMORY_CLEANUP_EVERY_EPOCHS, STANDARD_WEIGHT_INIT_POLICY
from .resource_monitor import record_work
from .model import PosePredTransformer
from .schedulers import InverseSquareRootDecayScheduler, CosineAnnealingScheduler, LinearAnnealingScheduler, EpochCosineScheduler
from .datasets import KeypointsDataset, build_dataloader, validate_supervised_training_metadata
from .augmentation import augment_features, select_columns_for_augmentation
from .losses import calculate_class_weights, masked_mse_loss
from .checkpoint import load_checkpoint_model, save_checkpoint, atomic_save_checkpoint, is_pretrain_checkpoint_dict, is_pretrain_checkpoint_path, set_checkpoint_globals, build_checkpoint_global_vars, validate_supervised_class_schema
from .records import setup_records_file, setup_confusion_matrix_files, log_epoch_results, log_confusion_matrix, log_wandb_metrics
from .utils import cleanup_memory, print_gpu_memory_summary
from .training_identity import build_training_data_identity, validate_training_data_identity

if TYPE_CHECKING:  # pragma: no cover
    from .feature_selection import FeaturePlan

__all__ = ['prepare_data', 'model_setup', 'train_one_epoch', 'validate_one_epoch', 'main_worker']


DEFAULT_SWA_ANNEAL_EPOCHS: int = 10


def _default_swa_anneal_epochs(swa_epochs: int) -> int:
    """Return the bounded annealing window for an SWA tail."""
    return max(1, min(int(swa_epochs), int(DEFAULT_SWA_ANNEAL_EPOCHS)))


def _extract_swa_runtime_state(
    *,
    swa_mode: bool,
    swa_start: Optional[int],
    swa_epochs: int,
    swa_lr: float,
    swa_anneal_epochs: int,
    swa_model: Optional[AveragedModel],
    swa_scheduler: Optional[SWALR],
) -> dict:
    """Serialize SWA runtime state needed for exact resume of an in-progress SWA tail."""
    state = {
        'swa_mode': bool(swa_mode),
        'swa_start': int(swa_start) if swa_start is not None else None,
        'swa_epochs': int(swa_epochs),
        'swa_lr': float(swa_lr),
        'swa_anneal_epochs': int(swa_anneal_epochs),
        'swa_started': bool(swa_model is not None and swa_scheduler is not None),
    }
    if swa_model is not None and swa_scheduler is not None:
        n_averaged = getattr(swa_model, 'n_averaged', None)
        if torch.is_tensor(n_averaged):
            n_averaged = int(n_averaged.item())
        elif n_averaged is None:
            n_averaged = 0
        state.update({
            'swa_n_averaged': int(n_averaged),
            'swa_model_state_dict': swa_model.state_dict(),
            'swa_scheduler_state_dict': swa_scheduler.state_dict(),
        })
    return state


def _extract_training_runtime_state(
    *,
    best_val_loss: float,
    best_pose_macro_f1: float,
    best_orient_macro_f1: float,
    best_model_path: Optional[str],
    epochs_since_improvement: int,
    total_epoch_time: float,
    num_completed_epochs: int,
    total_time_forward_pass: float,
    total_time_optimizer_step: float,
    total_inputs: int,
) -> dict:
    """Serialize early stopping, best-checkpoint bookkeeping and timing aggregates."""
    return {
        'best_val_loss': float(best_val_loss),
        'best_pose_macro_f1': float(best_pose_macro_f1),
        'best_orient_macro_f1': float(best_orient_macro_f1),
        'best_model_path': str(best_model_path) if best_model_path else None,
        'epochs_since_improvement': int(epochs_since_improvement),
        'total_epoch_time': float(total_epoch_time),
        'num_completed_epochs': int(num_completed_epochs),
        'total_time_forward_pass': float(total_time_forward_pass),
        'total_time_optimizer_step': float(total_time_optimizer_step),
        'total_inputs': int(total_inputs),
    }


def _apply_resume_training_runtime_state(state: Optional[dict]) -> dict:
    """Normalize persisted supervised runtime state loaded from a checkpoint."""
    if not isinstance(state, dict):
        return {
            'best_val_loss': float('inf'),
            'best_pose_macro_f1': float('-inf'),
            'best_orient_macro_f1': float('-inf'),
            'best_model_path': None,
            'epochs_since_improvement': 0,
            'total_epoch_time': 0.0,
            'num_completed_epochs': 0,
            'total_time_forward_pass': 0.0,
            'total_time_optimizer_step': 0.0,
            'total_inputs': 0,
        }
    return {
        'best_val_loss': float(state.get('best_val_loss', float('inf'))),
        'best_pose_macro_f1': float(state.get('best_pose_macro_f1', float('-inf'))),
        'best_orient_macro_f1': float(state.get('best_orient_macro_f1', float('-inf'))),
        'best_model_path': state.get('best_model_path', None),
        'epochs_since_improvement': int(state.get('epochs_since_improvement', 0)),
        'total_epoch_time': float(state.get('total_epoch_time', 0.0)),
        'num_completed_epochs': int(state.get('num_completed_epochs', 0)),
        'total_time_forward_pass': float(state.get('total_time_forward_pass', 0.0)),
        'total_time_optimizer_step': float(state.get('total_time_optimizer_step', 0.0)),
        'total_inputs': int(state.get('total_inputs', 0)),
    }


def _select_primary_validation_metric(
    *,
    noval: bool,
    pose_pred: bool,
    orient_pred: bool,
    train_total_loss_avg: float,
    val_total_loss_avg: float,
    pose_macro_f1: float,
    orient_macro_f1: float,
) -> tuple[float, str]:
    """Return the scalar used for early stopping / best-checkpoint selection.

    Preference order:
      - no validation: negative training loss
      - pose supervision present: pose macro-F1
      - orient-only supervision: orient macro-F1
      - otherwise: negative validation loss
    """
    if noval:
        return -float(train_total_loss_avg), 'train_total_loss'
    if pose_pred:
        return float(pose_macro_f1), 'pose_macro_f1'
    if orient_pred:
        return float(orient_macro_f1), 'orient_macro_f1'
    return -float(val_total_loss_avg), 'val_total_loss'


def _base_movie_identity(directory, filename):
    """Group augmented siblings by their normalized parent path and base filename."""
    from pathlib import Path
    path = (Path(directory) / str(filename).replace('\\', '/')).resolve()
    for prefix in ('original_', 'rotated90_', 'rotated180_', 'rotated270_',
                   'mirrored_', 'mirrored90_', 'mirrored180_', 'mirrored270_'):
        if path.name.startswith(prefix):
            path = path.with_name(path.name[len(prefix):])
            break
    return os.path.normcase(str(path))


def prepare_data(
    directory: str,
    metadata: pd.DataFrame,
    sequence_length: int,
    num_workers: int = 0,
    random_seed: Optional[int] = None,
    pretrained_model_path: Optional[str] = None,
    pose_pred: bool = True,
    orient_pred: bool = True,
    recon: bool = False,
    max_class_weight: float = 200,
    defined_split: bool = False,
    noval: bool = False,
    feature_plan: Optional["FeaturePlan"] = None,
    ablation_strict: bool = True,
    restart_optimizer: bool = False,
    preserve_checkpoint_split: bool = True,
) -> tuple[
                     Subset, Subset, Optional[dict[str, int]], Optional[dict[str, int]], 
                     list[int], torch.Tensor, torch.Tensor, dict[str, object]]:
    """Prepare validated supervised windows, class mappings and file splits.

    Use metadata-defined splits or group augmented versions by base movie for
    a random split. True resume requires unchanged input contents and retains
    saved splits. Supervised weights-only restarts also retain the saved split;
    pretraining handoff and training-only LR sweeps use current splits. Explicit
    noval training combines all files. Class weights use training-window frequencies only. Return train/val
    subsets, class dictionaries, augmentation indices, global-index weight
    tensors and split/input-identity metadata.
    """
    start_time = time.time()
    if pretrained_model_path and not os.path.isfile(pretrained_model_path):
        raise FileNotFoundError(f'Requested checkpoint does not exist: {pretrained_model_path}')

    inference_mode = False
    pretrain_mode = False

    # Validate the declared inventory before constructing any training windows.
    valid_metadata = validate_supervised_training_metadata(
        directory,
        metadata,
        sequence_length=sequence_length,
        pose_pred=pose_pred,
        orient_pred=orient_pred,
        defined_split=defined_split,
        noval=noval,
        scan_values=not ablation_strict,  # Strict loading checks values once; non-strict skips must not hide invalid data.
    )
    print(f"Validated all {len(valid_metadata)} files declared by training metadata.")
    training_data_identity = build_training_data_identity(
        directory, valid_metadata, sequence_length=sequence_length,
        pose_pred=pose_pred, orient_pred=orient_pred,
    )

    full_dataset = KeypointsDataset(
        directory,
        valid_metadata,
        sequence_length,
        num_workers=num_workers,
        pose_pred=pose_pred,
        orient_pred=orient_pred,
        recon=recon,
        inference_mode=inference_mode,
        feature_plan=feature_plan,
        ablation_strict=ablation_strict,
    )

    train_filenames = set()
    val_filenames = set()
    is_handoff = False
    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        is_pretrain_checkpoint = is_pretrain_checkpoint_dict(checkpoint)
        is_handoff = is_pretrain_checkpoint and not pretrain_mode
        if not is_handoff and not restart_optimizer:
            validate_training_data_identity(checkpoint.get('training_data_identity'), training_data_identity)
        if not is_handoff:
            full_dataset.pose_dict, full_dataset.orient_dict = validate_supervised_class_schema(
                checkpoint, full_dataset.pose_dict, full_dataset.orient_dict,
                pose_pred=pose_pred, orient_pred=orient_pred)
            full_dataset._add_indices_to_instructions()

        if ('train_filenames' in checkpoint and 'val_filenames' in checkpoint and not is_handoff
                and (preserve_checkpoint_split or not restart_optimizer)):
            train_filenames = set(checkpoint['train_filenames'])
            val_filenames = set(checkpoint['val_filenames'])
            overlap = sorted(train_filenames & val_filenames)
            if overlap:
                raise ValueError(
                    "Checkpoint train/validation file lists overlap; evaluation would leak "
                    f"training data. Examples: {overlap[:20]}"
                )
            print(f"Loaded train/val split from checkpoint: {len(train_filenames)} train, {len(val_filenames)} val files.")

    if noval:
        print("noval=True: Moving all validation files to the training set.")
        train_filenames.update(val_filenames)
        val_filenames.clear()

    all_valid_filenames = set(valid_metadata['filename'])

    files_to_split = all_valid_filenames - train_filenames - val_filenames
    print(f"Found {len(files_to_split)} files to assign to train/val splits.")


    if files_to_split:
        if noval:
            train_additions = files_to_split
            val_additions = set()
            print(f"noval=True: Assigning all {len(files_to_split)} new files to the training set.")
        else:
            files_to_split_metadata = valid_metadata[valid_metadata['filename'].isin(files_to_split)]
            if defined_split:
                train_additions = set(files_to_split_metadata[files_to_split_metadata['set'] == 'train']['filename'])
                val_additions = set(files_to_split_metadata[files_to_split_metadata['set'] == 'val']['filename'])
            else:
                # Use by-movie grouping logic for random split
                files_to_split_metadata = files_to_split_metadata.copy()
                files_to_split_metadata['base_movie'] = files_to_split_metadata['filename'].apply(
                    lambda filename: _base_movie_identity(directory, filename))
                unique_base_movies = files_to_split_metadata['base_movie'].unique().tolist()
                if len(unique_base_movies) < 2:
                    raise ValueError(
                        "At least two independent base movies are required to create a train/validation split. "
                        "Use --noval only when validation is intentionally disabled."
                    )
                
                from sklearn.model_selection import train_test_split
                split_seed = random_seed if random_seed is not None else 42
                train_base, val_base = train_test_split(unique_base_movies, test_size=0.2, random_state=split_seed)
                
                train_additions = set(files_to_split_metadata[files_to_split_metadata['base_movie'].isin(train_base)]['filename'])
                val_additions = set(files_to_split_metadata[files_to_split_metadata['base_movie'].isin(val_base)]['filename'])
        
        train_filenames.update(train_additions)
        val_filenames.update(val_additions)

    split_overlap = sorted(train_filenames & val_filenames)
    if split_overlap:
        raise ValueError(
            "Train/validation file lists overlap; evaluation would leak training data. "
            f"Examples: {split_overlap[:20]}"
        )
    stale_split_files = sorted((train_filenames | val_filenames) - all_valid_filenames)
    if stale_split_files:
        raise ValueError(
            "The saved train/validation split references files absent from current metadata. "
            f"Examples: {stale_split_files[:20]}"
        )
    unassigned_files = sorted(all_valid_filenames - train_filenames - val_filenames)
    if unassigned_files:
        raise ValueError(
            "Some training files were not assigned to train or validation. "
            f"Examples: {unassigned_files[:20]}"
        )
    if not train_filenames:
        raise ValueError("The training split is empty.")
    if not noval and not val_filenames:
        raise ValueError("The validation split is empty; use --noval only when intentional.")

    train_base_movies = {_base_movie_identity(directory, name) for name in train_filenames}
    val_base_movies = {_base_movie_identity(directory, name) for name in val_filenames}
    leaking_base_movies = sorted(train_base_movies & val_base_movies)
    if leaking_base_movies:
        raise ValueError(
            "Augmented versions of the same base movie occur in both train and validation. "
            f"Examples: {leaking_base_movies[:20]}"
        )

    train_indices = []
    val_indices = []
    if isinstance(full_dataset.dataset_instructions, WindowInstructions):
        train_indices = full_dataset.dataset_instructions.indices_for_files(train_filenames)
        val_indices = full_dataset.dataset_instructions.indices_for_files(val_filenames)
    else:
        train_paths = {os.path.normcase(os.path.abspath(os.path.join(directory, name))) for name in train_filenames}
        val_paths = {os.path.normcase(os.path.abspath(os.path.join(directory, name))) for name in val_filenames}
        for idx, instr in enumerate(full_dataset.dataset_instructions):
            input_path = os.path.normcase(os.path.abspath(instr['input_file']))
            if input_path in train_paths:
                train_indices.append(idx)
            elif input_path in val_paths:
                val_indices.append(idx)

    print(f"Final split: {len(train_indices)} train, {len(val_indices)} val instructions.")
    if len(train_indices) == 0:
        raise ValueError("No valid training sequences remain after label validation.")
    if not noval and len(val_indices) == 0:
        raise ValueError("No valid validation sequences remain after label validation.")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Number of samples in train_dataset: {len(train_dataset)}")
    print(f"Number of samples in val_dataset: {len(val_dataset)}\n")

    if feature_plan is not None:
        headers_for_aug = list(feature_plan.included_columns)
    else:
        from .io import get_headers_from_first_hdf5 as _get_headers_from_first_hdf5
        headers_for_aug = _get_headers_from_first_hdf5(directory)
    columns_to_noise = select_columns_for_augmentation(headers_for_aug)
    print(f"Augmentation columns: noise={len(columns_to_noise)} of {len(headers_for_aug)} features")

    train_class_weights, val_class_weights = calculate_class_weights(
        full_dataset, train_indices, val_indices, pose_pred, orient_pred, max_class_weight
    )

    if noval:
        if train_class_weights is not None:
            if isinstance(train_class_weights, dict):
                val_class_weights = {}
                if 'pose' in train_class_weights and train_class_weights['pose'] is not None:
                    val_class_weights['pose'] = torch.ones_like(train_class_weights['pose'])
                if 'orient' in train_class_weights and train_class_weights['orient'] is not None:
                    val_class_weights['orient'] = torch.ones_like(train_class_weights['orient'])
            else:
                val_class_weights = torch.ones_like(train_class_weights)
        else:
            val_class_weights = torch.tensor([])


    split_filenames = {
        'train_filenames': list(train_filenames),
        'val_filenames': list(val_filenames),
        'training_data_identity': training_data_identity,
    }

    return (
        train_dataset,
        val_dataset,
        train_dataset.dataset.pose_dict if hasattr(train_dataset.dataset, 'pose_dict') else None,
        train_dataset.dataset.orient_dict if hasattr(train_dataset.dataset, 'orient_dict') else None,
        columns_to_noise,
        train_class_weights,
        val_class_weights,
        split_filenames
    )


def model_setup(
    sequence_length: int,
    num_features: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout_rate: float,
    frame_representation: str,
    num_cls: int,
    prediction_head: str,
    device: torch.device,
    base_lr: float,
    pose_pred: bool,
    orient_pred: bool,
    pose_dict: Optional[dict[str, int]],
    orient_dict: Optional[dict[str, int]],
    recon: bool,
    positional_encoding_type: str,
    patience: int,
    plateau_factor: float,
    pretrain_mode: bool = False,
    projection_dim: int = 128,
    batches_per_epoch: int = 1,
    create_scheduler: bool = True,
    scheduler_type: str = 'cosine',
    weight_decay: float = 1e-3,
    warmup_steps: int = 0,
    t_0: int = 1000,
    t_mult: int = 1,
    eta_min: float = 1e-6,
    # Pretraining handoff controls.
    is_handoff: bool = False,
    freeze_encoder: bool = False,
    weight_encoder: float = 1.0,
    warmup_epochs: Optional[int] = None,
    cosine_decay_epochs: Optional[int] = None,
) -> tuple[
    PosePredTransformer,
    torch.optim.AdamW,
    Union[InverseSquareRootDecayScheduler, CosineAnnealingScheduler, ReduceLROnPlateau],
    GradScaler
]:
    """Build the model, optimizer, scheduler and scaler without restoring state.

    Checkpoint loading is separate. During pretraining-to-supervised handoff,
    ``freeze_encoder`` trains prediction heads only; otherwise ``weight_encoder``
    can multiply the backbone learning rate.
    """
    num_pose_classes = len(pose_dict) if pose_pred and pose_dict else 0
    num_orient_classes = len(orient_dict) if orient_pred and orient_dict else 0

    config = {
        "sequence_length": sequence_length,
        "num_features": num_features,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_feedforward": dim_feedforward,
        "dropout_rate": dropout_rate,
        "frame_representation": frame_representation,
        "num_cls": num_cls,
        "prediction_head": prediction_head,
        "pose_pred": pose_pred,
        "orient_pred": orient_pred,
        "recon": recon,
        "num_pose_classes": num_pose_classes,
        "num_orient_classes": num_orient_classes,
        "positional_encoding_type": positional_encoding_type,
        "pretrain_mode": pretrain_mode,
        "projection_dim": projection_dim,
    }

    print(f"Initializing model with sequence_length={sequence_length}, num_features={num_features}")

    model = PosePredTransformer(**config)

    model.to(device)

    print("\nShapes after model initialization:")
    print(f"Initial Linear Input Shape: {model.initial_linear.weight.shape}")
    if hasattr(model, "positional_encoding"):
        try:
            print(f"Positional Encoding Shape: {model.positional_encoding.shape}")
        except Exception:
            pass
    print(f"Number of Encoder Layers: {num_encoder_layers}")
    print(f"Encoder Layer Shape: {model.transformer_encoder.layers[0].self_attn.in_proj_weight.shape}")

    if pose_pred:
        if model.prediction_head == "fc" and hasattr(model, "pose_fc"):
            print(f"Pose FC Layer Weight Shape: {model.pose_fc.weight.shape}")
        elif model.prediction_head == "mlp" and hasattr(model, "pose_mlp"):
            print(f"Pose MLP First Layer Weight Shape: {model.pose_mlp[0].weight.shape}")
            print(f"Pose MLP Second Layer Weight Shape: {model.pose_mlp[4].weight.shape}")
            print(f"Pose MLP Third Layer Weight Shape: {model.pose_mlp[8].weight.shape}")

    if orient_pred:
        if model.prediction_head == "fc" and hasattr(model, "orient_fc"):
            print(f"Orientation FC Layer Weight Shape: {model.orient_fc.weight.shape}")
        elif model.prediction_head == "mlp" and hasattr(model, "orient_mlp"):
            print(f"Orientation MLP First Layer Weight Shape: {model.orient_mlp[0].weight.shape}")
            print(f"Orientation MLP Second Layer Weight Shape: {model.orient_mlp[4].weight.shape}")
            print(f"Orientation MLP Third Layer Weight Shape: {model.orient_mlp[8].weight.shape}")

    if recon:
        print(f"Number of Decoder Layers: {num_decoder_layers}")
        print(f"Decoder Layer Shape: {model.transformer_decoder.layers[0].self_attn.in_proj_weight.shape}")
        print(f"Output Linear Weight Shape: {model.output_linear.weight.shape}")

    print("\nTotal number of parameters:", sum(p.numel() for p in model.parameters()))

    # Use fused AdamW on CUDA when available.
    is_cuda = device.type == "cuda"

    if not pretrain_mode and freeze_encoder:
        print("Freezing encoder backbone parameters according to the supervised run configuration.")
        for param in model.initial_linear.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        if hasattr(model, "positional_encoding") and isinstance(model.positional_encoding, nn.Parameter):
            model.positional_encoding.requires_grad = False
        if hasattr(model, "frame_cls"):
            model.frame_cls.requires_grad = False
        if hasattr(model, "frame_cls_positional_encoding"):
            model.frame_cls_positional_encoding.requires_grad = False
        if hasattr(model, "concat_projection"):
            for param in model.concat_projection.parameters():
                param.requires_grad = False
        if hasattr(model, "cls_projection"):
            for param in model.cls_projection.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
            fused=is_cuda,
        )

    elif not pretrain_mode and (not freeze_encoder) and (weight_encoder != 1.0):
        print(f"Using differential LR with encoder weight: {weight_encoder}")

        backbone_params = list(model.initial_linear.parameters()) + list(model.transformer_encoder.parameters())
        if hasattr(model, "positional_encoding") and isinstance(model.positional_encoding, nn.Parameter):
            backbone_params.append(model.positional_encoding)
        if hasattr(model, "frame_cls"):
            backbone_params.append(model.frame_cls)
        if hasattr(model, "frame_cls_positional_encoding"):
            backbone_params.append(model.frame_cls_positional_encoding)
        if hasattr(model, "concat_projection"):
            backbone_params.extend(model.concat_projection.parameters())
        if hasattr(model, "cls_projection"):
            backbone_params.extend(model.cls_projection.parameters())

        # Safety: remove duplicates while preserving order
        def _unique_params(params):
            seen = set()
            uniq = []
            for p in params:
                if id(p) not in seen:
                    uniq.append(p)
                    seen.add(id(p))
            return uniq

        backbone_params = _unique_params(backbone_params)
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]

        param_groups = [
            {"params": backbone_params, "lr": base_lr * weight_encoder, "weight_decay": weight_decay},
            {"params": head_params, "lr": base_lr, "weight_decay": weight_decay},
        ]
        optimizer = torch.optim.AdamW(param_groups, fused=is_cuda)

    else:
        if is_handoff:
            print("[CKPT] Handoff detected. Training all layers with the same LR.")
        else:
            print("Not a handoff. Training all layers with the same LR.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, fused=is_cuda)

    if create_scheduler:
        if cosine_decay_epochs is not None:
            if scheduler_type != 'cosine':
                raise ValueError('Epoch cosine mode requires scheduler_type=cosine.')
            scheduler = EpochCosineScheduler(optimizer, batches_per_epoch,
                1 if warmup_epochs is None else warmup_epochs, cosine_decay_epochs, eta_min)
        elif scheduler_type == "cosine":
            print("Using CosineAnnealingScheduler with a linear warmup.")
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
            t_0_batches = int(t_0)  # t_0 is steps per cycle
            main_scheduler = CosineAnnealingScheduler(optimizer, t_0_batches, t_mult, eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
        elif scheduler_type == "linear":
            print("Using LinearAnnealingScheduler with a linear warmup.")
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
            t_0_batches = int(t_0)  # t_0 is steps per cycle
            main_scheduler = LinearAnnealingScheduler(optimizer, t_0_batches, t_mult, eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
        elif scheduler_type == "sqrt":
            print("Using InverseSquareRootDecayScheduler with a linear warmup.")
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
            main_scheduler = InverseSquareRootDecayScheduler(optimizer)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
        elif scheduler_type == "plateau":
            # ReduceLROnPlateau is stepped once per epoch (after validation/train summary).
            # We intentionally set its patience shorter than early stopping so LR reduction can
            # kick in before training halts. `patience` is the early-stopping patience (in epochs).
            plateau_patience = max(1, int(patience // 2))
            print(f"Using ReduceLROnPlateau (no warmup scheduler; warmup handled manually in epoch 0). "
                  f"factor={plateau_factor}, plateau_patience={plateau_patience}")
            scheduler = ReduceLROnPlateau(optimizer, factor=plateau_factor, patience=plateau_patience)
        else:
            print(f"Warning: Invalid scheduler type '{scheduler_type}'. Defaulting to InverseSquareRootDecayScheduler with warmup.")
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
            main_scheduler = InverseSquareRootDecayScheduler(optimizer)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    else:
        scheduler = None

    scaler = GradScaler(enabled=(device.type == "cuda"))

    return model, optimizer, scheduler, scaler


def train_one_epoch(model: PosePredTransformer, train_dataloader: DataLoader, optimizer: torch.optim.AdamW, 
                   scheduler: Union[InverseSquareRootDecayScheduler, CosineAnnealingScheduler, ReduceLROnPlateau],
                   device: torch.device, scaler: GradScaler, columns_to_noise: list[int], pose_pred: bool, orient_pred: bool, recon: bool, 
                   recon_weight: float, pose_weight: float, orient_weight: float, 
                   train_class_weights: torch.Tensor, global_step: int, 
                   is_plateau_warmup: bool = False, base_lr: float = 0.001, grad_clip_norm: float = 1.0,
                   step_scheduler_per_batch: bool = True, mask_p: float = 0.05, noise_level: float = 0.05,
                   scheduler_type: str = 'cosine') -> tuple[float, float, float, float, float, float, float, float, float, int, float, float, int]:
    """Train one epoch with Gaussian noise followed by elementwise masking.

    Use training-derived global-window weights for classification cross-entropy
    with label smoothing 0.1. Reconstruction targets remain clean and its MSE
    uses masked elements only. Unscale gradients before norm clipping. Return
    loss/accuracy aggregates, sample count, timing totals in seconds and the
    updated global step; step batch-based schedulers after optimization.
    """
    model.train()  

    train_recon_loss_total = 0.0
    train_weighted_recon_loss_total = 0.0
    train_pose_loss_total = 0.0
    train_weighted_pose_loss_total = 0.0
    train_orient_loss_total = 0.0
    train_weighted_orient_loss_total = 0.0
    train_correct_pose_predictions = 0
    train_total_pose_predictions = 0
    train_correct_orient_predictions = 0
    train_total_orient_predictions = 0
    total_inputs = 0

    time_forward_pass = 0.0
    time_optimizer_step = 0.0

    optimizer.zero_grad()  

    warmup_targets = [group['lr'] for group in optimizer.param_groups]
    for batch_idx, batch in enumerate(progress(train_dataloader, desc='Training batches')):
        if step_scheduler_per_batch and is_plateau_warmup:
            fraction = (batch_idx + 1) / len(train_dataloader)
            for group, target_lr in zip(optimizer.param_groups, warmup_targets):
                group['lr'] = target_lr * fraction
        sequences = batch[0].to(device, non_blocking=True)  
        poses = batch[1].to(device, non_blocking=True) if pose_pred else None  
        orients = batch[2].to(device, non_blocking=True) if orient_pred else None  
        indices = batch[3]  

        batch_class_weights = train_class_weights[indices].to(device, non_blocking=True)

        # Apply noise before masking; reconstruction compares masked elements with clean targets.
        if recon:
            sequences_aug, aug_keep_mask = augment_features(
                sequences,
                mask_p,
                noise_level,
                columns_to_noise,
                return_mask=True,
            )
        else:
            sequences_aug = augment_features(
                sequences,
                mask_p,
                noise_level,
                columns_to_noise,
            )
            aug_keep_mask = None


        start_time = time.time()
        with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            pose_pred_out, orient_pred_out, recon_output, context_vector = model(sequences_aug)

            if recon:
                if aug_keep_mask is None:
                    raise RuntimeError("Internal error: recon=True but aug_keep_mask is None")
                recon_loss = masked_mse_loss(recon_output, sequences, aug_keep_mask)
            else:
                recon_loss = torch.zeros((), device=device, dtype=torch.float32)

            if pose_pred:
                pose_pred_out = pose_pred_out.view(-1, pose_pred_out.size(-1))
                pose_labels = poses.view(-1).long()  
                pose_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)(pose_pred_out, pose_labels)
                weighted_pose_loss = (pose_loss * batch_class_weights).mean()  
            else:
                weighted_pose_loss = 0.0

            if orient_pred:
                orient_pred_out = orient_pred_out.view(-1, orient_pred_out.size(-1))
                orient_labels = orients.view(-1).long()  
                orient_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)(orient_pred_out, orient_labels)
                weighted_orient_loss = (orient_loss * batch_class_weights).mean()  
            else:
                weighted_orient_loss = 0.0

        time_forward_pass += time.time() - start_time  

        weighted_recon_loss = recon_weight * recon_loss
        weighted_pose_loss = pose_weight * weighted_pose_loss
        weighted_orient_loss = orient_weight * weighted_orient_loss
        total_loss = weighted_recon_loss + weighted_pose_loss + weighted_orient_loss

        start_time = time.time()
        optimizer.zero_grad()  
        scaler.scale(total_loss).backward()  

        # Unscale before clipping so the norm threshold is independent of AMP scale.
        if grad_clip_norm is not None and grad_clip_norm > 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)  
        scaler.update()  
        time_optimizer_step += time.time() - start_time  

        # Batch scheduler stepping and plateau warmup.
        if step_scheduler_per_batch:
            if not is_plateau_warmup:
            # Step batch-based schedulers after optimizer updates.
                if scheduler_type in ['sqrt', 'cosine', 'linear']:
                    scheduler.step()

        global_step += 1
        record_work("training_examples", completed=int(sequences.size(0)))
        
        if recon:
            train_recon_loss_total += recon_loss.item()
            train_weighted_recon_loss_total += weighted_recon_loss.item()

        if pose_pred:
            train_pose_loss_total += pose_loss.mean().item()
            train_weighted_pose_loss_total += weighted_pose_loss.item()
            _, predicted_poses = torch.max(pose_pred_out, 1)
            train_correct_pose_predictions += (predicted_poses == pose_labels).sum().item()
            train_total_pose_predictions += poses.size(0)

        if orient_pred:
            train_orient_loss_total += orient_loss.mean().item()
            train_weighted_orient_loss_total += weighted_orient_loss.item()
            _, predicted_orients = torch.max(orient_pred_out, 1)
            train_correct_orient_predictions += (predicted_orients == orient_labels).sum().item()
            train_total_orient_predictions += orients.size(0)

    train_recon_loss_avg = train_recon_loss_total / len(train_dataloader)
    train_weighted_recon_loss_avg = train_weighted_recon_loss_total / len(train_dataloader)
    train_pose_loss_avg = train_pose_loss_total / len(train_dataloader) if pose_pred else 0.0
    train_weighted_pose_loss_avg = train_weighted_pose_loss_total / len(train_dataloader) if pose_pred else 0.0
    train_orient_loss_avg = train_orient_loss_total / len(train_dataloader) if orient_pred else 0.0
    train_weighted_orient_loss_avg = train_weighted_orient_loss_total / len(train_dataloader) if orient_pred else 0.0    

    train_total_loss_avg = (train_weighted_recon_loss_avg + train_weighted_pose_loss_avg + train_weighted_orient_loss_avg)

    train_pose_accuracy = train_correct_pose_predictions / train_total_pose_predictions if pose_pred else 0.0
    train_orient_accuracy = train_correct_orient_predictions / train_total_orient_predictions if orient_pred else 0.0

    return (train_pose_accuracy, train_orient_accuracy, train_recon_loss_avg, train_weighted_recon_loss_avg, 
            train_pose_loss_avg, train_weighted_pose_loss_avg, train_orient_loss_avg, train_weighted_orient_loss_avg,
            train_total_loss_avg, total_inputs, time_forward_pass, time_optimizer_step, global_step)


def validate_one_epoch(
    model: PosePredTransformer,
    val_dataloader: DataLoader,
    device: torch.device,
    columns_to_noise: list[int],
    pose_pred: bool,
    orient_pred: bool,
    recon: bool,
    val_class_weights: torch.Tensor,
    pose_dict: Optional[dict] = None,
    orient_dict: Optional[dict] = None,
    pose_weight: float = 1.0,
    orient_weight: float = 1.0,
    recon_weight: float = 1.0,
    mask_p: float = 0.05,
    noise_level: float = 0.05,
) -> tuple[float, float, float, float, float, float, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], Optional[np.ndarray], Optional[float]]:
    """Evaluate clean classification windows without gradients.

    Use deterministic corruption only for masked reconstruction against clean
    targets, restoring RNG states afterward. Return weighted/unweighted losses,
    accuracies, ordered confusion matrices and macro-F1 scores. Class weights
    retain training-derived frequencies.
    """
    model.eval()  

    val_recon_loss_total = 0.0
    val_weighted_recon_loss_total = 0.0
    val_pose_loss_total = 0.0
    val_weighted_pose_loss_total = 0.0
    val_orient_loss_total = 0.0
    val_weighted_orient_loss_total = 0.0
    val_correct_pose_predictions = 0
    val_total_pose_predictions = 0
    val_correct_orient_predictions = 0
    val_total_orient_predictions = 0

    all_pose_preds = []
    all_pose_labels = []
    all_orient_preds = []
    all_orient_labels = []

    # Fix reconstruction-validation corruption and restore RNG states afterward.
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    if recon:
        torch.manual_seed(0)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(0)

    try:
        with torch.no_grad():  
            for batch in progress(val_dataloader, desc='Validation batches'):
                sequences = batch[0].to(device)

                if sequences.dim() != 3:
                    raise ValueError(f"Expected 3 dimensions, but got {sequences.dim()} dimensions")

                poses = batch[1].to(device) if pose_pred else None
                orients = batch[2].to(device) if orient_pred else None
                indices = batch[3]  

                batch_class_weights = val_class_weights[indices].to(device)

                # Evaluate classification on clean sequences.
                if pose_pred or orient_pred:
                    with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        pose_pred_out, orient_pred_out, _recon_unused, _ctx_unused = model(sequences)

                        if pose_pred:
                            pose_pred_out = pose_pred_out.view(-1, pose_pred_out.size(-1))
                            pose_labels = poses.view(-1).long()  
                            pose_loss = nn.CrossEntropyLoss(reduction='none')(pose_pred_out, pose_labels)
                            weighted_pose_loss = (pose_loss * batch_class_weights).mean()
                            val_pose_loss_total += pose_loss.mean().item()
                            val_weighted_pose_loss_total += (pose_weight * weighted_pose_loss).item()
                            _, predicted_poses = torch.max(pose_pred_out, 1)

                            val_correct_pose_predictions += (predicted_poses == pose_labels).sum().item()
                            val_total_pose_predictions += poses.size(0)

                            all_pose_preds.extend(predicted_poses.cpu().numpy())
                            all_pose_labels.extend(pose_labels.cpu().numpy())

                        if orient_pred:
                            orient_pred_out = orient_pred_out.view(-1, orient_pred_out.size(-1))
                            orient_labels = orients.view(-1).long()  
                            orient_loss = nn.CrossEntropyLoss(reduction='none')(orient_pred_out, orient_labels)
                            weighted_orient_loss = (orient_loss * batch_class_weights).mean()
                            val_orient_loss_total += orient_loss.mean().item()
                            val_weighted_orient_loss_total += (orient_weight * weighted_orient_loss).item()
                            _, predicted_orients = torch.max(orient_pred_out, 1)

                            val_correct_orient_predictions += (predicted_orients == orient_labels).sum().item()
                            val_total_orient_predictions += orients.size(0)

                            all_orient_preds.extend(predicted_orients.cpu().numpy())
                            all_orient_labels.extend(orient_labels.cpu().numpy())

                # Evaluate masked reconstruction on corrupted inputs with clean targets.
                if recon:
                    sequences_aug, aug_keep_mask = augment_features(
                        sequences,
                        mask_p,
                        noise_level,
                        columns_to_noise,
                        return_mask=True,
                    )
                    with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        _pose_unused, _orient_unused, recon_output, _ctx_unused = model(sequences_aug)
                        recon_loss = masked_mse_loss(recon_output, sequences, aug_keep_mask)

                    val_recon_loss_total += recon_loss.item()
                    val_weighted_recon_loss_total += (recon_weight * recon_loss).item()
    finally:
        # Restore RNG states so validation augmentation does not affect training randomness.
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    pose_conf_matrix = confusion_matrix(all_pose_labels, all_pose_preds, labels=list(range(len(pose_dict)))) if pose_pred else None
    orient_conf_matrix = confusion_matrix(all_orient_labels, all_orient_preds, labels=list(range(len(orient_dict)))) if orient_pred else None

    pose_macro_f1 = f1_score(all_pose_labels, all_pose_preds, average='macro', zero_division=0.0) if pose_pred else None
    orient_macro_f1 = f1_score(all_orient_labels, all_orient_preds, average='macro', zero_division=0.0) if orient_pred else None

    combined_conf_matrix = None
    combined_macro_f1 = None
    if pose_pred and orient_pred:
        reverse_pose_dict = {v: k for k, v in pose_dict.items()}
        reverse_orient_dict = {v: k for k, v in orient_dict.items()}

        pose_true_names = [reverse_pose_dict[i] for i in all_pose_labels]
        pose_pred_names = [reverse_pose_dict[i] for i in all_pose_preds]
        orient_true_names = [reverse_orient_dict[i] for i in all_orient_labels]
        orient_pred_names = [reverse_orient_dict[i] for i in all_orient_preds]

        combined_true_labels = [f"{p}-{o}" for p, o in zip(pose_true_names, orient_true_names)]
        combined_pred_labels = [f"{p}-{o}" for p, o in zip(pose_pred_names, orient_pred_names)]

        # Define all possible combined class names to ensure matrix dimensions are correct
        all_combined_names = [f"{p}-{o}" for p in sorted(pose_dict.keys()) for o in sorted(orient_dict.keys())]

        combined_conf_matrix = confusion_matrix(combined_true_labels, combined_pred_labels, labels=all_combined_names)
        combined_macro_f1 = f1_score(combined_true_labels, combined_pred_labels, average='macro', zero_division=0.0)

    val_recon_loss_avg = val_recon_loss_total / len(val_dataloader) if recon else 0.0
    val_weighted_recon_loss_avg = val_weighted_recon_loss_total / len(val_dataloader) if recon else 0.0
    val_pose_loss_avg = val_pose_loss_total / len(val_dataloader) if pose_pred else 0.0
    val_weighted_pose_loss_avg = val_weighted_pose_loss_total / len(val_dataloader) if pose_pred else 0.0
    val_orient_loss_avg = val_orient_loss_total / len(val_dataloader) if orient_pred else 0.0
    val_weighted_orient_loss_avg = val_weighted_orient_loss_total / len(val_dataloader) if orient_pred else 0.0
    val_total_loss_avg = val_weighted_recon_loss_avg + val_weighted_pose_loss_avg + val_weighted_orient_loss_avg
    val_pose_accuracy = val_correct_pose_predictions / val_total_pose_predictions if pose_pred else 0.0
    val_orient_accuracy = val_correct_orient_predictions / val_total_orient_predictions if orient_pred else 0.0

    return (val_pose_accuracy, val_orient_accuracy, val_recon_loss_avg,
            val_pose_loss_avg, val_weighted_pose_loss_avg,
            val_orient_loss_avg, val_weighted_orient_loss_avg,
            val_total_loss_avg, pose_conf_matrix, orient_conf_matrix,
            pose_macro_f1, orient_macro_f1,
            combined_conf_matrix, combined_macro_f1)


@close_feature_stores
def main_worker(train_dataset: Subset, val_dataset: Subset, pose_dict: Optional[dict[str, int]], 
                orient_dict: Optional[dict[str, int]], num_workers: int, sequence_length: int, batch: int, 
                pretrained_model_path: str, split_filenames: dict[str, object], num_features: int, 
                d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, 
                dim_feedforward: int, dropout_rate: float, frame_representation: str, num_cls: int, 
                prediction_head: str, records_path: str, wandb_log: bool, num_epochs: int, 
                checkpoint_interval: int, checkpoint_dir: str, run_name: str, columns_to_noise: list[int], pose_conf_matrix_path: Optional[str], 
                orient_conf_matrix_path: Optional[str], combined_conf_matrix_path: Optional[str], pose_pred: bool, orient_pred: bool, recon: bool, 
                positional_encoding_type: str, pose_weight: float, orient_weight: float, 
                recon_weight: float, train_class_weights: torch.Tensor, val_class_weights: torch.Tensor, 
                patience: int, plateau_factor: float,
                freeze_encoder: bool, weight_encoder: float, restart_optimizer: bool, noval: bool,
                base_lr: float, weight_decay: float, scheduler_type: str, warmup_steps: int, t_0: int,
                t_mult: int, eta_min: float, mask_p: float, noise_level: float, max_class_weight: float,
                grad_clip_norm: float, random_seed: Optional[int], metadata_tag: str,
                swa_mode: bool = False, swa_epochs: int = 0, swa_lr: float = 1e-5,
                feature_plan: Optional["FeaturePlan"] = None,
                warmup_epochs: Optional[int] = None,
                cosine_decay_epochs: Optional[int] = None) -> None:
    """Run supervised optimization, evaluation and checkpoint/log publication.

    Distinguish full resume from weights-only restart and pretraining handoff.
    Select early stopping by pose macro-F1, orientation macro-F1 or negative
    loss according to enabled heads and validation availability. Optional SWA
    is a final training tail with a separate averaged-weight checkpoint;
    optimizer state belongs to ordinary resumable checkpoints.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Current device: {device}\n")

    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        seed=random_seed
    )

    if not noval:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        val_dataloader = None

    # Resolve the SWA schedule before saving checkpoint metadata.
    if swa_mode:
        swa_epochs = int(swa_epochs)
        if swa_epochs <= 0:
            raise ValueError("swa_mode=True requires swa_epochs > 0.")
        effective_num_epochs = int(num_epochs)
        if swa_epochs >= effective_num_epochs:
            raise ValueError("swa_mode=True requires swa_epochs to be smaller than num_epochs.")
        swa_start = effective_num_epochs - swa_epochs
        swa_anneal_epochs = _default_swa_anneal_epochs(swa_epochs)
        print(
            f"[SWA] Enabled: total_epochs={effective_num_epochs}, normal_epochs={swa_start}, "
            f"swa_epochs={swa_epochs}, swa_lr={swa_lr}, swa_anneal_epochs={swa_anneal_epochs}."
        )
        print("[SWA] Early stopping is DISABLED in swa_mode.")
    else:
        effective_num_epochs = int(num_epochs)
        swa_start = None
        swa_epochs = 0
        swa_anneal_epochs = 0

    schedule_globals = {}
    # Resolve epoch schedule only after --noval and the actual loader are known.
    if warmup_epochs is not None and cosine_decay_epochs is None:
        raise ValueError('--warmup_epochs requires --cosine_decay_epochs.')
    if cosine_decay_epochs is not None:
        warmup_epochs = 1 if warmup_epochs is None else int(warmup_epochs)
        if scheduler_type != 'cosine' or swa_mode:
            raise ValueError('Epoch cosine mode requires cosine scheduling without SWA.')
        if warmup_epochs < 0 or int(cosine_decay_epochs) < 1:
            raise ValueError('Invalid epoch cosine durations.')
        if warmup_epochs + int(cosine_decay_epochs) != effective_num_epochs:
            raise ValueError('Warmup epochs plus cosine decay epochs must equal num_epochs.')
        if pretrained_model_path and not restart_optimizer:
            saved = torch.load(pretrained_model_path, map_location='cpu')
            if not is_pretrain_checkpoint_dict(saved):
                saved_schedule = saved.get('global_vars', {})
                expected = {'schedule_mode': 'epoch_cosine_no_restart',
                            'warmup_epochs': warmup_epochs, 'cosine_decay_epochs': int(cosine_decay_epochs),
                            'batches_per_epoch': len(train_dataloader)}
                if any(saved_schedule.get(key) != value for key, value in expected.items()):
                    raise ValueError('Epoch cosine resume requires the saved durations and batches per epoch. '
                                     'Use --restart_optimizer for an intentional schedule change.')
        warmup_steps = warmup_epochs * len(train_dataloader)
        t_0 = int(cosine_decay_epochs) * len(train_dataloader)
        t_mult = 1
        schedule_globals.update(schedule_mode='epoch_cosine_no_restart',
            warmup_epochs=warmup_epochs, cosine_decay_epochs=int(cosine_decay_epochs),
            batches_per_epoch=len(train_dataloader))
        print(f'[SCHEDULE] Epoch cosine: {warmup_steps} warmup steps, {t_0} decay steps; no restarts.')

    # Checkpoint metadata
    extra_globals = {}
    if feature_plan is not None:
        extra_globals = {
            'feature_schema_hash': feature_plan.schema_hash,
            'feature_columns_included': list(feature_plan.included_columns),
            'feature_columns_excluded': list(feature_plan.excluded_columns),
            'feature_columns_canonical': list(feature_plan.canonical_columns),
            'ablation_spec': dict(feature_plan.spec),
        }

    extra_globals.update(schedule_globals)
    set_checkpoint_globals(
        reset=True,
        run_name=run_name,
        metadata_tag=metadata_tag,
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        frame_representation=frame_representation,
        num_cls=num_cls,
        prediction_head=prediction_head,
        positional_encoding_type=positional_encoding_type,
        pose_pred=pose_pred,
        orient_pred=orient_pred,
        recon=recon,
        batch=batch,
        scheduler_type=scheduler_type,
        base_lr=base_lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        t_0=t_0,
        t_mult=t_mult,
        eta_min=eta_min,
        mask_p=mask_p,
        noise_level=noise_level,
        max_class_weight=max_class_weight,
        pose_weight=pose_weight,
        orient_weight=orient_weight,
        recon_weight=recon_weight,
        grad_clip_norm=grad_clip_norm,
        random_seed=random_seed,
        columns_to_noise=columns_to_noise,
        patience=patience,
        plateau_factor=plateau_factor,
        freeze_encoder=freeze_encoder,
        weight_encoder=weight_encoder,
        weight_initialization=STANDARD_WEIGHT_INIT_POLICY,
        num_epochs=effective_num_epochs,
        checkpoint_interval=checkpoint_interval,
        swa_mode=bool(swa_mode),
        swa_start=int(swa_start) if swa_start is not None else None,
        swa_epochs=int(swa_epochs),
        swa_lr=float(swa_lr),
        swa_anneal_epochs=int(swa_anneal_epochs),
        **extra_globals,
    )
    # Model initialization
    is_handoff = False
    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        is_handoff = is_pretrain_checkpoint_path(pretrained_model_path)

    model, optimizer, scheduler, scaler = model_setup(
        sequence_length, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
        dropout_rate, frame_representation, num_cls, prediction_head,
        device, base_lr, pose_pred, orient_pred, 
        pose_dict, orient_dict, recon, positional_encoding_type,
        patience, plateau_factor, pretrain_mode=False, projection_dim=128, batches_per_epoch=len(train_dataloader),
        scheduler_type=scheduler_type,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        t_0=t_0,
        t_mult=t_mult,
        eta_min=eta_min,
        is_handoff=is_handoff,
        freeze_encoder=freeze_encoder,
        weight_encoder=weight_encoder,
        warmup_epochs=warmup_epochs,
        cosine_decay_epochs=cosine_decay_epochs
    )
    print_gpu_memory_summary()


    pose_conf_matrix = None
    orient_conf_matrix = None
    # Restore optimizer/scheduler/scaler state only for true resume.
    load_optimizer_state = (not restart_optimizer) and (not is_handoff)
    if pretrained_model_path and os.path.isfile(pretrained_model_path) and not load_optimizer_state:
        print("[CKPT] Loading weights only (no optimizer/scheduler/scaler state).")

    resume_swa_runtime = None
    resume_training_runtime = None
    if pretrained_model_path and os.path.isfile(pretrained_model_path) and load_optimizer_state:
        try:
            resume_checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if isinstance(resume_checkpoint, dict):
                resume_swa_runtime = resume_checkpoint.get('swa_runtime_state', None)
                resume_training_runtime = resume_checkpoint.get('training_runtime_state', None)
        except Exception as e:
            print(f"[CKPT] Warning: could not inspect checkpoint for resume state: {e}")

    start_epoch, global_step = load_checkpoint_model(
        model, optimizer, scheduler, pretrained_model_path, scaler, load_optimizer_state=load_optimizer_state, pretrain_mode=False,
        pose_dict=train_dataloader.dataset.dataset.pose_dict, orient_dict=train_dataloader.dataset.dataset.orient_dict
    )

    file, writer, headers = setup_records_file(
        records_path, 
        start_epoch,  
        pose_pred=pose_pred, 
        orient_pred=orient_pred,
        recon=recon
    )

    if file is None or writer is None or headers is None:
        raise RuntimeError(f"Failed to initialize records file at {records_path}. See earlier errors above.")

    if not noval:
        setup_confusion_matrix_files(
            val_dataloader=val_dataloader,
            pose_conf_matrix_path=pose_conf_matrix_path,
            orient_conf_matrix_path=orient_conf_matrix_path,
            combined_conf_matrix_path=combined_conf_matrix_path,
            pose_pred=pose_pred, 
            orient_pred=orient_pred
        )          

    # Resumable timing and early-stopping state
    restored_training_runtime = _apply_resume_training_runtime_state(resume_training_runtime)
    total_epoch_time = float(restored_training_runtime['total_epoch_time'])
    num_completed_epochs = int(restored_training_runtime['num_completed_epochs'])
    total_time_forward_pass = float(restored_training_runtime['total_time_forward_pass'])
    total_time_optimizer_step = float(restored_training_runtime['total_time_optimizer_step'])
    total_inputs = int(restored_training_runtime['total_inputs'])
    epochs_since_improvement = int(restored_training_runtime['epochs_since_improvement'])
    best_val_loss = float(restored_training_runtime['best_val_loss'])
    best_pose_macro_f1 = float(restored_training_runtime['best_pose_macro_f1'])
    best_orient_macro_f1 = float(restored_training_runtime['best_orient_macro_f1'])
    best_model_path = restored_training_runtime['best_model_path']
    if load_optimizer_state and start_epoch > 0 and not isinstance(resume_training_runtime, dict):
        print(
            "[CKPT] Warning: no persisted training_runtime_state was found. "
            "Best-metric / early-stopping bookkeeping will restart from scratch for this resumed run."
        )

    # SWA runtime state
    # Train the base model during the SWA tail while maintaining an averaged shadow.
    swa_model: Optional[AveragedModel] = None
    swa_scheduler: Optional[SWALR] = None
    resumed_swa_state = False

    if swa_mode and isinstance(resume_swa_runtime, dict) and bool(resume_swa_runtime.get('swa_started', False)):
        resume_swa_start = resume_swa_runtime.get('swa_start', swa_start)
        if resume_swa_start is not None:
            swa_start = int(resume_swa_start)
        resume_swa_lr = float(resume_swa_runtime.get('swa_lr', swa_lr))
        resume_swa_anneal_epochs = int(resume_swa_runtime.get('swa_anneal_epochs', swa_anneal_epochs or _default_swa_anneal_epochs(swa_epochs)))
        if (resume_swa_lr != float(swa_lr)) or (resume_swa_anneal_epochs != int(swa_anneal_epochs)):
            print(
                f"[SWA] Resume overrides active SWA scheduler config from checkpoint: "
                f"swa_lr {swa_lr} -> {resume_swa_lr}, swa_anneal_epochs {swa_anneal_epochs} -> {resume_swa_anneal_epochs}."
            )
        swa_lr = resume_swa_lr
        swa_anneal_epochs = resume_swa_anneal_epochs
        print(f"[SWA] Restoring in-progress SWA state from checkpoint (swa_start={swa_start}, n_averaged={resume_swa_runtime.get('swa_n_averaged', 'unknown')}).")
        swa_model = AveragedModel(model)
        swa_model.to(device)
        swa_model.load_state_dict(resume_swa_runtime['swa_model_state_dict'])
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs)
        swa_sched_state = resume_swa_runtime.get('swa_scheduler_state_dict', None)
        if swa_sched_state is not None:
            swa_scheduler.load_state_dict(swa_sched_state)
        resumed_swa_state = True

    if swa_mode and (start_epoch > (swa_start if swa_start is not None else effective_num_epochs)) and (not resumed_swa_state):
        print(
            "[SWA] Warning: resuming inside the SWA tail but no persisted SWA runtime state was found. "
            "A fresh SWA average will be started from the resumed base-model weights, which may not exactly match an uninterrupted run."
        )

    if start_epoch >= effective_num_epochs:
        print(f"No epochs to run: start_epoch={start_epoch} >= effective_num_epochs={effective_num_epochs}.")
        if not (swa_mode and resumed_swa_state):
            file.close()
            return
        print('[SWA] Completing final validation and publication from the saved average.')

    # Training loop
    for epoch in range(start_epoch, effective_num_epochs):
        epoch_start_time = time.time()
        
        current_lr = [pg['lr'] for pg in optimizer.param_groups]

        in_swa_phase = bool(swa_mode and (swa_start is not None) and (epoch >= swa_start))

        if in_swa_phase and swa_model is None:
            print(f"[SWA] Starting SWA phase at epoch {epoch + 1} (0-index={epoch}, swa_start={swa_start}).")
            swa_model = AveragedModel(model)
            swa_model.to(device)
            swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs)

        active_scheduler = swa_scheduler if in_swa_phase else scheduler

        # The 'plateau' scheduler needs a manual warmup, handled by this flag.
        # Other batch schedulers handle their own warmup.
        is_plateau_warmup_epoch = (not in_swa_phase) and (epoch == 0 and start_epoch == 0 and scheduler_type == 'plateau')

        (train_pose_accuracy, train_orient_accuracy, train_recon_loss_avg, train_weighted_recon_loss_avg, 
        train_pose_loss_avg, train_weighted_pose_loss_avg, train_orient_loss_avg, train_weighted_orient_loss_avg,  
        train_total_loss_avg, epoch_inputs, time_forward_pass, time_optimizer_step, global_step) = train_one_epoch(
            model, train_dataloader, optimizer, active_scheduler, device, scaler, 
            columns_to_noise, pose_pred, orient_pred, recon,
            recon_weight, pose_weight, orient_weight, train_class_weights, global_step,
            is_plateau_warmup=is_plateau_warmup_epoch, base_lr=base_lr, grad_clip_norm=grad_clip_norm,
            step_scheduler_per_batch=(not in_swa_phase), mask_p=mask_p, noise_level=noise_level,
            scheduler_type=scheduler_type
        )
        total_inputs += epoch_inputs

        if not noval:
            (val_pose_accuracy, val_orient_accuracy, val_recon_loss_avg, val_pose_loss_avg, val_weighted_pose_loss_avg,
            val_orient_loss_avg, val_weighted_orient_loss_avg, val_total_loss_avg, pose_conf_matrix, orient_conf_matrix, pose_macro_f1, orient_macro_f1,
            combined_conf_matrix, combined_macro_f1) = validate_one_epoch(
                model,
                val_dataloader,
                device,
                columns_to_noise,
                pose_pred,
                orient_pred,
                recon,
                val_class_weights,
                pose_dict,
                orient_dict,
                pose_weight,
                orient_weight,
                recon_weight,
                mask_p,
                noise_level,
            )
        else:
            print("noval=True: Validation metrics are unavailable; training loss controls checkpoint selection.")
            val_pose_accuracy = val_orient_accuracy = float('nan')
            val_recon_loss_avg = val_pose_loss_avg = val_weighted_pose_loss_avg = float('nan')
            val_orient_loss_avg = val_weighted_orient_loss_avg = val_total_loss_avg = float('nan')
            pose_conf_matrix = None
            orient_conf_matrix = None
            pose_macro_f1 = None
            orient_macro_f1 = None
            combined_conf_matrix = None
            combined_macro_f1 = None


        if in_swa_phase:
            # SWA phase: update the running average model and step SWALR *per epoch*.
            if swa_model is None or swa_scheduler is None:
                raise RuntimeError("Internal error: in_swa_phase=True but swa_model/swa_scheduler is not initialized")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if scheduler_type == 'plateau' and not is_plateau_warmup_epoch:
                # The plateau scheduler is always stepped per-epoch, but only after its warmup is complete.
                if noval:
                    # Without validation, step the plateau scheduler using training loss.
                    metric_for_scheduler = train_total_loss_avg
                    scheduler.step(train_total_loss_avg)
                else:
                    if pose_pred:
                        scheduler.step(-pose_macro_f1)
                    else:
                        scheduler.step(val_total_loss_avg)

        if not noval:
            if pose_pred:
                log_confusion_matrix(pose_conf_matrix, val_dataloader.dataset.dataset.pose_dict.keys(), epoch+1, pose_conf_matrix_path)
            if orient_pred:
                log_confusion_matrix(orient_conf_matrix, val_dataloader.dataset.dataset.orient_dict.keys(), epoch+1, orient_conf_matrix_path)

            if pose_pred and orient_pred:
                pose_class_names = sorted(val_dataloader.dataset.dataset.pose_dict.keys())
                orient_class_names = sorted(val_dataloader.dataset.dataset.orient_dict.keys())
                combined_class_names = [f"{p}-{o}" for p in pose_class_names for o in orient_class_names]
                log_confusion_matrix(combined_conf_matrix, combined_class_names, epoch+1, combined_conf_matrix_path)
                
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_epoch_time += epoch_time
        num_completed_epochs += 1
        record_work("supervised_epochs", completed=1)
        avg_time_per_epoch = total_epoch_time / num_completed_epochs


        total_time_forward_pass += time_forward_pass
        total_time_optimizer_step += time_optimizer_step

        avg_time_forward_pass = total_time_forward_pass / num_completed_epochs
        avg_time_optimizer_step = total_time_optimizer_step / num_completed_epochs

        # Early stopping / best checkpoint selection.
        early_stopping_metric, metric_name = _select_primary_validation_metric(
            noval=bool(noval),
            pose_pred=bool(pose_pred),
            orient_pred=bool(orient_pred),
            train_total_loss_avg=float(train_total_loss_avg),
            val_total_loss_avg=float(val_total_loss_avg),
            pose_macro_f1=pose_macro_f1,
            orient_macro_f1=orient_macro_f1,
        )
        if metric_name == 'pose_macro_f1':
            best_metric = best_pose_macro_f1
        elif metric_name == 'orient_macro_f1':
            best_metric = best_orient_macro_f1
        else:
            best_metric = -best_val_loss

        if early_stopping_metric > best_metric:
            if metric_name == 'pose_macro_f1':
                best_pose_macro_f1 = float(early_stopping_metric)
            elif metric_name == 'orient_macro_f1':
                best_orient_macro_f1 = float(early_stopping_metric)
            else:
                best_val_loss = -float(early_stopping_metric)
            epochs_since_improvement = 0  
            previous_best_path = best_model_path
            best_model_filename = f"{run_name}_best_model_epoch_{epoch+1}.pth"
            best_model_path = os.path.join(checkpoint_dir, best_model_filename)
            save_checkpoint(
                epoch, global_step, model, optimizer, scheduler, pose_dict, orient_dict, scaler,
                checkpoint_dir, run_name,
                pose_pred, orient_pred, recon, split_filenames,
                filename_override=best_model_filename,
                swa_runtime_state=_extract_swa_runtime_state(
                    swa_mode=swa_mode,
                    swa_start=swa_start,
                    swa_epochs=swa_epochs,
                    swa_lr=swa_lr,
                    swa_anneal_epochs=swa_anneal_epochs,
                    swa_model=swa_model,
                    swa_scheduler=swa_scheduler,
                ),
                training_runtime_state=_extract_training_runtime_state(
                    best_val_loss=best_val_loss,
                    best_pose_macro_f1=best_pose_macro_f1,
                    best_orient_macro_f1=best_orient_macro_f1,
                    best_model_path=best_model_path,
                    epochs_since_improvement=epochs_since_improvement,
                    total_epoch_time=total_epoch_time,
                    num_completed_epochs=num_completed_epochs,
                    total_time_forward_pass=total_time_forward_pass,
                    total_time_optimizer_step=total_time_optimizer_step,
                    total_inputs=total_inputs,
                ),
            )
            if previous_best_path and previous_best_path != best_model_path and os.path.exists(previous_best_path):
                try:
                    os.remove(previous_best_path)
                except OSError as exc:
                    print(f'Warning: previous best checkpoint retained at {previous_best_path}: {exc}')
        else:
            epochs_since_improvement += 1

        stop_requested = (not swa_mode) and epochs_since_improvement >= patience

        if in_swa_phase:
            lr_phase = 'swa'
        elif is_plateau_warmup_epoch or (scheduler_type != 'plateau' and warmup_steps > 0 and global_step < warmup_steps):
            lr_phase = 'warmup'
        else:
            lr_phase = 'main'
        swa_n_averaged = 0
        if swa_model is not None:
            n_averaged = getattr(swa_model, 'n_averaged', 0)
            if torch.is_tensor(n_averaged):
                swa_n_averaged = int(n_averaged.item())
            else:
                swa_n_averaged = int(n_averaged)

        metrics = {
            'model_variant': 'base',
            'lr_phase': lr_phase,
            'swa_active': bool(in_swa_phase),
            'swa_n_averaged': swa_n_averaged,
            'train_recon_loss_avg': train_recon_loss_avg,
            'train_weighted_recon_loss_avg': train_weighted_recon_loss_avg,
            'val_recon_loss_avg': val_recon_loss_avg,
            'train_pose_loss_avg': train_pose_loss_avg,
            'train_weighted_pose_loss_avg': train_weighted_pose_loss_avg,
            'val_pose_loss_avg': val_pose_loss_avg,
            'val_weighted_pose_loss_avg': val_weighted_pose_loss_avg,
            'train_pose_accuracy': train_pose_accuracy,
            'val_pose_accuracy': val_pose_accuracy,
            'train_orient_loss_avg': train_orient_loss_avg,
            'train_weighted_orient_loss_avg': train_weighted_orient_loss_avg,
            'val_orient_loss_avg': val_orient_loss_avg,
            'val_weighted_orient_loss_avg': val_weighted_orient_loss_avg,
            'train_orient_accuracy': train_orient_accuracy,
            'val_orient_accuracy': val_orient_accuracy,
            'train_total_loss_avg': train_total_loss_avg,
            'val_total_loss_avg': val_total_loss_avg,
            'val_pose_macro_f1': pose_macro_f1,
            'val_orient_macro_f1': orient_macro_f1,
            'epoch_time': epoch_time,
            'avg_time_per_epoch': avg_time_per_epoch,
            'current_lr': current_lr,
            'time_forward_pass': time_forward_pass,
            'time_optimizer_step': time_optimizer_step,
            'avg_time_forward_pass': avg_time_forward_pass,
            'avg_time_optimizer_step': avg_time_optimizer_step,
            'epochs_since_improvement': epochs_since_improvement
        }

        metrics['validation_available'] = not noval
        log_epoch_results(
            file, writer, epoch, metrics, headers,
            pose_pred, orient_pred, recon
        )

        if wandb_log:
            log_wandb_metrics(epoch, metrics, pose_pred, orient_pred, recon)

        print(f'\nEpoch {epoch + 1}/{effective_num_epochs}, Avg Time per Epoch: {avg_time_per_epoch:.4f} sec, Learning Rate: {current_lr}')
        if pose_pred:
            print(f'Pose: Train Loss: {train_pose_loss_avg:.4f}, Val Loss: {val_pose_loss_avg:.4f}, Train Acc: {train_pose_accuracy:.4f}, Val Acc: {val_pose_accuracy:.4f}')
        if orient_pred:
            print(f'Orient: Train Loss: {train_orient_loss_avg:.4f}, Val Loss: {val_orient_loss_avg:.4f}, Train Acc: {train_orient_accuracy:.4f}, Val Acc: {val_orient_accuracy:.4f}')
        if recon:
            print(f'Recon: Train Loss: {train_recon_loss_avg:.4f}, Val Loss: {val_recon_loss_avg:.4f}')


        if ((checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0)
                or stop_requested or epoch + 1 == effective_num_epochs):
            save_checkpoint(
                epoch, global_step, model, optimizer, scheduler, pose_dict, orient_dict, scaler, checkpoint_dir, run_name,
                pose_pred, orient_pred, recon, split_filenames,
                swa_runtime_state=_extract_swa_runtime_state(
                    swa_mode=swa_mode,
                    swa_start=swa_start,
                    swa_epochs=swa_epochs,
                    swa_lr=swa_lr,
                    swa_anneal_epochs=swa_anneal_epochs,
                    swa_model=swa_model,
                    swa_scheduler=swa_scheduler,
                ),
                training_runtime_state=_extract_training_runtime_state(
                    best_val_loss=best_val_loss,
                    best_pose_macro_f1=best_pose_macro_f1,
                    best_orient_macro_f1=best_orient_macro_f1,
                    best_model_path=best_model_path,
                    epochs_since_improvement=epochs_since_improvement,
                    total_epoch_time=total_epoch_time,
                    num_completed_epochs=num_completed_epochs,
                    total_time_forward_pass=total_time_forward_pass,
                    total_time_optimizer_step=total_time_optimizer_step,
                    total_inputs=total_inputs,
                ),
            )

        if stop_requested:
            print(f"No improvement in {patience} epochs. Stopping after recording/checkpointing epoch {epoch + 1}.")
            break

        if MEMORY_CLEANUP_EVERY_EPOCHS and ((epoch + 1) % int(MEMORY_CLEANUP_EVERY_EPOCHS) == 0):
            print(f"[Memory cleanup] epoch {epoch + 1} (every {MEMORY_CLEANUP_EVERY_EPOCHS} epochs)")
            cleanup_memory()



    # Final averaged-weight artifact.
    if swa_mode:
        if swa_model is None:
            print("[SWA] swa_mode enabled but SWA phase never initialized; skipping SWA checkpoint.")
        else:
            print("\n[SWA] Finalizing SWA model (update_bn + optional validation + checkpoint save)...")

            # Update BatchNorm statistics (no-op if the model has no BatchNorm layers)
            try:
                torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)
            except Exception as e:
                print(f"[SWA] Warning: update_bn failed (continuing anyway): {e}")

            if not noval:
                (swa_val_pose_accuracy, swa_val_orient_accuracy, swa_val_recon_loss_avg, swa_val_pose_loss_avg, swa_val_weighted_pose_loss_avg,
                swa_val_orient_loss_avg, swa_val_weighted_orient_loss_avg, swa_val_total_loss_avg, swa_pose_conf_matrix, swa_orient_conf_matrix, swa_pose_macro_f1, swa_orient_macro_f1,
                swa_combined_conf_matrix, swa_combined_macro_f1) = validate_one_epoch(
                    swa_model,
                    val_dataloader,
                    device,
                    columns_to_noise,
                    pose_pred,
                    orient_pred,
                    recon,
                    val_class_weights,
                    pose_dict,
                    orient_dict,
                    pose_weight,
                    orient_weight,
                    recon_weight,
                    mask_p,
                    noise_level,
                )

                # Log SWA confusion matrices using a sentinel epoch label
                swa_epoch_label = 9999
                if pose_pred and pose_conf_matrix_path:
                    log_confusion_matrix(swa_pose_conf_matrix, val_dataloader.dataset.dataset.pose_dict.keys(), swa_epoch_label, pose_conf_matrix_path)
                if orient_pred and orient_conf_matrix_path:
                    log_confusion_matrix(swa_orient_conf_matrix, val_dataloader.dataset.dataset.orient_dict.keys(), swa_epoch_label, orient_conf_matrix_path)
                if pose_pred and orient_pred and combined_conf_matrix_path:
                    pose_class_names = sorted(val_dataloader.dataset.dataset.pose_dict.keys())
                    orient_class_names = sorted(val_dataloader.dataset.dataset.orient_dict.keys())
                    combined_class_names = [f"{p}-{o}" for p in pose_class_names for o in orient_class_names]
                    log_confusion_matrix(swa_combined_conf_matrix, combined_class_names, swa_epoch_label, combined_conf_matrix_path)

                swa_n_averaged = getattr(swa_model, 'n_averaged', 0)
                if torch.is_tensor(swa_n_averaged):
                    swa_n_averaged = int(swa_n_averaged.item())
                else:
                    swa_n_averaged = int(swa_n_averaged)
                swa_row = {
                    'Epoch': 'SWA_FINAL',
                    'Model Variant': 'swa_final',
                    'LR Phase': 'swa',
                    'SWA Active': 1,
                    'SWA Snapshots Averaged': swa_n_averaged,
                    'Validation Available': 1,
                    'Val Total Loss Avg': swa_val_total_loss_avg,
                    'Val Pose Accuracy': swa_val_pose_accuracy,
                    'Val Orient Accuracy': swa_val_orient_accuracy,
                    'Val Recon Loss Avg': swa_val_recon_loss_avg,
                    'Val Pose Loss Avg': swa_val_pose_loss_avg,
                    'Val Weighted Pose Loss Avg': swa_val_weighted_pose_loss_avg,
                    'Val Orient Loss Avg': swa_val_orient_loss_avg,
                    'Val Weighted Orient Loss Avg': swa_val_weighted_orient_loss_avg,
                    'Val Pose Macro F1': swa_pose_macro_f1,
                    'Val Orient Macro F1': swa_orient_macro_f1,
                    'Learning Rate': [pg['lr'] for pg in optimizer.param_groups],
                }
                try:
                    writer.writerow([swa_row.get(h, '') for h in headers])
                    file.flush()
                except Exception as e:
                    print(f"[SWA] Warning: failed to append SWA row to records CSV (continuing): {e}")

                print(f"[SWA] Validation summary: total_loss={swa_val_total_loss_avg:.4f} | pose_F1={swa_pose_macro_f1} | orient_F1={swa_orient_macro_f1}")
            else:
                print("[SWA] noval=True: Skipping SWA validation.")

            # Final SWA weights omit optimizer/scheduler state.
            swa_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_swa_model.pth")
            global_vars = build_checkpoint_global_vars(
                pose_pred=pose_pred,
                orient_pred=orient_pred,
                recon=recon,
                pretrain_mode=False,
                extra_overrides={
                    'swa_mode': True,
                    'swa_start': int(swa_start) if swa_start is not None else None,
                    'swa_epochs': int(swa_epochs),
                    'swa_lr': float(swa_lr),
                    'swa_anneal_epochs': int(swa_anneal_epochs),
                },
            )

            checkpoint_data = {
                'checkpoint_schema_version': 1,
                'checkpoint_kind': 'swa',
                'checkpoint_created_at': datetime.now().isoformat(),
                'run_name': run_name,
                'epoch': effective_num_epochs - 1,
                'global_step': global_step,
                'model_state_dict': swa_model.module.state_dict(),
                'global_vars': global_vars,
                'swa_n_averaged': int(getattr(swa_model, 'n_averaged', torch.tensor(0)).item() if torch.is_tensor(getattr(swa_model, 'n_averaged', torch.tensor(0))) else getattr(swa_model, 'n_averaged', 0)),
            }

            if pose_pred:
                checkpoint_data['pose_dict'] = pose_dict
            if orient_pred:
                checkpoint_data['orient_dict'] = orient_dict

            if split_filenames:
                checkpoint_data.update({key: split_filenames[key] for key in
                    ('train_filenames', 'val_filenames', 'training_data_identity')
                    if key in split_filenames})

            atomic_save_checkpoint(checkpoint_data, swa_checkpoint_path)
            print(f"[SWA] Checkpoint saved to {swa_checkpoint_path}")



    file.close()
    print("Training loop completed")
