"""Supervised training pipeline for pose/orientation prediction.

Contains data preparation, training/validation loops, AMP + gradient clipping, checkpointing,
scheduler stepping, and early stopping logic."""

from __future__ import annotations

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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .constants import MEMORY_CLEANUP_EVERY_EPOCHS, STANDARD_WEIGHT_INIT_POLICY
from .model import PosePredTransformer
from .schedulers import InverseSquareRootDecayScheduler, CosineAnnealingScheduler, LinearAnnealingScheduler
from .datasets import KeypointsDataset, build_dataloader
from .augmentation import augment_features, select_columns_for_augmentation
from .losses import calculate_class_weights, masked_mse_loss
from .checkpoint import load_checkpoint_model, save_checkpoint, is_pretrain_checkpoint_dict, is_pretrain_checkpoint_path, set_checkpoint_globals, build_checkpoint_global_vars
from .records import setup_records_file, setup_confusion_matrix_files, log_epoch_results, log_confusion_matrix, log_wandb_metrics
from .utils import cleanup_memory, print_gpu_memory_summary

if TYPE_CHECKING:  # pragma: no cover
    from .feature_selection import FeaturePlan

__all__ = ['prepare_data', 'model_setup', 'train_one_epoch', 'validate_one_epoch', 'main_worker']


DEFAULT_SWA_ANNEAL_EPOCHS: int = 10


def _default_swa_anneal_epochs(swa_epochs: int) -> int:
    """Return a sensible SWA annealing window for the given tail length."""
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
    """Serialize non-model runtime state needed for accurate supervised resume.

    This keeps early stopping, best-checkpoint bookkeeping, and running timing aggregates
    consistent across interrupted/resumed runs.
    """
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
) -> tuple[
                     Subset, Subset, Optional[dict[str, int]], Optional[dict[str, int]], 
                     list[int], torch.Tensor, torch.Tensor, dict[str, list[str]]]:
    """
    Prepares comprehensive training and validation datasets with intelligent data splitting and preprocessing.
    
    This function orchestrates the complete data preparation pipeline for transformer training, handling
    file validation, dataset creation, train/validation splitting, checkpoint resume logic, and class
    weight calculation. It supports both predefined splits and intelligent movie-based grouping to
    prevent data leakage between augmented versions of the same video.
    
    Data Preparation Pipeline:
    1. **File Validation**: Validates existence of data and label files from metadata
    2. **Dataset Creation**: Initializes KeypointsDataset with all valid sequences
    3. **Split Management**: Handles checkpoint resuming and orphan file assignment
    4. **Intelligent Splitting**: Groups augmented movie versions to prevent leakage
    5. **Index Mapping**: Maps filenames to dataset instruction indices
    6. **Preprocessing Setup**: Determines augmentation columns and calculates class weights
    7. **Checkpointing Preparation**: Packages split information for robust resuming
    
    Splitting Logic:
    - **Defined Split**: Uses 'set' column from metadata ('train'/'val' values)
    - **Random Split**: Groups by base movie name to keep augmentations together
    - **Checkpoint Resume**: Preserves existing splits and handles new files
    - **Handoff Detection**: Recognizes pretraining-to-finetuning transitions
    
    Parameters:
    - directory: Path to directory containing HDF5 data and label files.
    - metadata: DataFrame with 'filename', 'labels', and optionally 'set' columns.
    - sequence_length: Number of frames in each training sequence.
    - pretrained_model_path: Optional path to checkpoint for split resume and handoff detection.
    - pose_pred: Whether to enable pose classification prediction head.
    - orient_pred: Whether to enable orientation classification prediction head.
    - recon: Whether to enable sequence reconstruction objective.
    - max_class_weight: Maximum class weight to prevent over-weighting rare classes.
    - defined_split: Whether to use predefined train/val split from metadata 'set' column.
    
    Returns:
    - train_dataset: Subset containing training sequences with augmentation support.
    - val_dataset: Subset containing validation sequences for evaluation.
    - pose_dict: Dictionary mapping pose class names to integer indices (None if pose_pred=False).
    - orient_dict: Dictionary mapping orientation class names to integer indices (None if orient_pred=False).
    - columns_to_noise: List of column indices for noise augmentation.
    - train_class_weights: Tensor of sample weights for training class balance.
    - val_class_weights: Tensor of sample weights for validation consistency.
    - split_filenames: Dictionary with 'train_filenames' and 'val_filenames' lists for checkpointing.
    
    Notes:
    - Handles missing files gracefully with warnings
    - Prevents data leakage by grouping augmented movie versions
    - Supports seamless checkpoint resuming and pretraining handoffs
    - Calculates class weights to handle imbalanced datasets effectively
    """
    start_time = time.time()

    # This helper prepares *training* and *validation* datasets. It is not used for inference.
    inference_mode = False
    # Pretraining uses a different codepath; here we assume supervised training/fine-tuning.
    pretrain_mode = False

    # Step 1: Gather usable files from metadata
    valid_rows = []
    missing_files = []
    missing_labels = []
    for idx, row in metadata.iterrows():
        data_path = os.path.join(directory, row['filename'])
        label_path = os.path.join(directory, row['labels']) if 'labels' in row and not pd.isna(row['labels']) else None
        data_exists = os.path.isfile(data_path)
        label_exists = os.path.isfile(label_path) if label_path else True
        if data_exists and label_exists:
            valid_rows.append(row)
        else:
            if not data_exists:
                missing_files.append(row['filename'])
            if label_path and not label_exists:
                missing_labels.append(row['labels'])
    if missing_files:
        print(f"Warning: The following data files listed in metadata were not found and will be skipped: {missing_files}")
    if missing_labels:
        print(f"Warning: The following label files listed in metadata were not found and will be skipped: {missing_labels}")
    valid_metadata = pd.DataFrame(valid_rows)
    print(f"Found {len(valid_metadata)} valid files out of {len(metadata)} in metadata.")

    # Step 2: Initialize the full dataset with all valid files
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

    # Checkpoint-based splitting logic
    train_filenames = set()
    val_filenames = set()
    is_handoff = False
    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        is_pretrain_checkpoint = is_pretrain_checkpoint_dict(checkpoint)
        is_handoff = is_pretrain_checkpoint and not pretrain_mode

        # If it's NOT a handoff, load the file lists from the checkpoint
        if 'train_filenames' in checkpoint and 'val_filenames' in checkpoint and not is_handoff:
            train_filenames = set(checkpoint['train_filenames'])
            val_filenames = set(checkpoint['val_filenames'])
            print(f"Loaded train/val split from checkpoint: {len(train_filenames)} train, {len(val_filenames)} val files.")

    if noval:
        print("noval=True: Moving all validation files to the training set.")
        train_filenames.update(val_filenames)
        val_filenames.clear()

    # Identify all files available in the current dataset
    all_valid_filenames = set(valid_metadata['filename'])

    # Determine which files need to be assigned to a split.
    # If it's a handoff or a new run, this is all files.
    # If resuming, this is just the orphan files.
    files_to_split = all_valid_filenames - train_filenames - val_filenames
    print(f"Found {len(files_to_split)} files to assign to train/val splits.")


    # Assign files_to_split
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
                AUG_PREFIXES = [
                    "original_", "rotated90_", "rotated180_", "rotated270_",
                    "mirrored_", "mirrored90_", "mirrored180_", "mirrored270_"
                ]
                def get_base_movie_name(filename):
                    for prefix in AUG_PREFIXES:
                        if filename.startswith(prefix):
                            return filename[len(prefix):]
                    return filename
                
                files_to_split_metadata = files_to_split_metadata.copy()
                files_to_split_metadata['base_movie'] = files_to_split_metadata['filename'].apply(get_base_movie_name)
                unique_base_movies = files_to_split_metadata['base_movie'].unique()
                
                from sklearn.model_selection import train_test_split
                split_seed = random_seed if random_seed is not None else 42
                train_base, val_base = train_test_split(unique_base_movies, test_size=0.2, random_state=split_seed)
                
                train_additions = set(files_to_split_metadata[files_to_split_metadata['base_movie'].isin(train_base)]['filename'])
                val_additions = set(files_to_split_metadata[files_to_split_metadata['base_movie'].isin(val_base)]['filename'])
        
        train_filenames.update(train_additions)
        val_filenames.update(val_additions)

    # Map filenames to instruction indices
    train_indices = []
    val_indices = []
    for idx, instr in enumerate(full_dataset.dataset_instructions):
        file_base = os.path.basename(instr['input_file'])
        if file_base in train_filenames:
            train_indices.append(idx)
        elif file_base in val_filenames:
            val_indices.append(idx)

    print(f"Final split: {len(train_indices)} train, {len(val_indices)} val instructions.")

    # ... after train_indices and val_indices are created ...
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Number of samples in train_dataset: {len(train_dataset)}")
    print(f"Number of samples in val_dataset: {len(val_dataset)}\n")

    # Determine which columns to apply noise augmentation to
    if feature_plan is not None:
        headers_for_aug = list(feature_plan.included_columns)
    else:
        from .io import get_headers_from_first_hdf5 as _get_headers_from_first_hdf5
        headers_for_aug = _get_headers_from_first_hdf5(directory)
    columns_to_noise = select_columns_for_augmentation(headers_for_aug)
    print(f"Augmentation columns: noise={len(columns_to_noise)} of {len(headers_for_aug)} features")

    # Calculate and create class weights tensor for the training dataset
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
            # Fallback if no weights were calculated
            val_class_weights = torch.tensor([])


    # For checkpointing: save train/val filename lists
    split_filenames = {
        'train_filenames': list(train_filenames),
        'val_filenames': list(val_filenames)
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
    # --- Fine-tuning / handoff controls (explicit, no implicit checkpoint reading) ---
    is_handoff: bool = False,
    freeze_encoder: bool = False,
    weight_encoder: float = 1.0,
) -> tuple[
    PosePredTransformer,
    torch.optim.AdamW,
    Union[InverseSquareRootDecayScheduler, CosineAnnealingScheduler, ReduceLROnPlateau],
    GradScaler
]:
    """
    Initializes and configures all components required for transformer model training or inference.

    This function is intentionally *checkpoint-agnostic*: it builds the model/optimizer/scheduler/scaler
    **without** loading any checkpoint weights or optimizer/scheduler state. All checkpoint loading
    is handled by the dedicated `load_checkpoint_model()` function to avoid duplicated state restores
    and subtle resume bugs.

    Parameters (key additions):
    - is_handoff: True if this run is a *pretrain → supervised* handoff (i.e., checkpoint is pretraining).
                 Used only to apply optional fine-tuning optimizer policies (freeze/differential LR).
    - freeze_encoder: If True AND is_handoff, freezes the backbone and trains only prediction heads.
    - weight_encoder: If != 1.0 AND is_handoff (and not freeze_encoder), uses differential LR for backbone.

    Returns:
    - model, optimizer, scheduler, scaler
    """
    # Calculate the number of pose and orientation classes
    num_pose_classes = len(pose_dict) if pose_pred and pose_dict else 0
    num_orient_classes = len(orient_dict) if orient_pred and orient_dict else 0

    # Configuration dictionary for model initialization
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

    # Initialize the PosePredTransformer model
    model = PosePredTransformer(**config)

    # Move the model to the correct device
    model.to(device)

    # Print model layer shapes.
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

    # Initialize the optimizer with weight decay
    # Use the 'fused' implementation if available (on CUDA) for a potential speedup
    is_cuda = device.type == "cuda"

    # --- Optional fine-tuning policies for handoff runs ---
    if is_handoff and freeze_encoder:
        print("[CKPT] Handoff detected + freeze_encoder=True. Freezing encoder backbone parameters.")
        # Freeze backbone parameters
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

    elif is_handoff and (not freeze_encoder) and (weight_encoder != 1.0):
        print(f"[CKPT] Handoff detected. Using differential LR with encoder weight: {weight_encoder}")

        # Group 1: Backbone parameters with a lower learning rate
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

        # Group 2: Prediction head parameters with the full learning rate
        head_params: list[nn.Parameter] = []
        if pose_pred:
            if hasattr(model, "pose_fc"):
                head_params.extend(model.pose_fc.parameters())
            elif hasattr(model, "pose_mlp"):
                head_params.extend(model.pose_mlp.parameters())
        if orient_pred:
            if hasattr(model, "orient_fc"):
                head_params.extend(model.orient_fc.parameters())
            elif hasattr(model, "orient_mlp"):
                head_params.extend(model.orient_mlp.parameters())

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
        head_params = _unique_params(head_params)

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

    # Initialize the learning rate scheduler
    if create_scheduler:
        if scheduler_type == "cosine":
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

    # Initialize the gradient scaler for mixed precision training
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
    """
    Executes one complete training epoch with data augmentation, multi-objective loss computation, and comprehensive metrics tracking.
    
    This function performs a full training pass through the dataset, handling data augmentation,
    forward propagation, multi-objective loss calculation (reconstruction, pose, orientation),
    gradient propagation with scaling, and detailed performance monitoring. It supports
    flexible training configurations with optional prediction heads and weighted loss combinations.
    
    Training Process:
    1. **Model Preparation**: Sets model to training mode for dropout and batch normalization
    2. **Batch Processing**: Iterates through all training batches with data loading
    3. **Data Augmentation**: Applies masking and noise transformations
    4. **Multi-Objective Forward Pass**: Computes predictions for enabled heads (pose/orient/recon)
    5. **Loss Computation**: Calculates individual and weighted losses with class balancing
    6. **Gradient Processing**: Performs scaled gradient pass with gradient clipping
    7. **Weight Updates**: Updates model parameters with optimizer step
    8. **Metrics Accumulation**: Tracks accuracy, loss, and timing information
    
    Loss Components:
    - **Reconstruction Loss**: Mean squared error between input and decoded sequences
    - **Pose Classification Loss**: Cross-entropy with label smoothing and class weighting
    - **Orientation Classification Loss**: Cross-entropy with label smoothing and class weighting
    - **Total Loss**: Weighted combination of all enabled loss components
    
    Performance Optimizations:
    - **Mixed Precision**: Automatic FP16/FP32 casting for memory efficiency
    - **Gradient Scaling**: Prevents underflow in mixed precision training
    - **Gradient Clipping**: Stabilizes training with max norm constraint
    - **Non-blocking Transfer**: Asynchronous GPU memory transfers
    
    Parameters:
    - model: PosePredTransformer model in training mode for parameter updates.
    - train_dataloader: DataLoader providing batched training sequences and labels.
    - optimizer: AdamW optimizer for model parameter updates with weight decay.
    - scheduler: Learning rate scheduler with current state and configuration.
    - device: PyTorch device for computation (CPU or CUDA GPU).
    - scaler: Gradient scaler for automatic mixed precision training stability.
    - columns_to_noise: List of feature column indices for noise augmentation.
    - pose_pred: Whether pose classification head is enabled for training.
    - orient_pred: Whether orientation classification head is enabled for training.
    - recon: Whether reconstruction objective is enabled for training.
    - recon_weight: Loss weighting factor for reconstruction objective.
    - pose_weight: Loss weighting factor for pose classification objective.
    - orient_weight: Loss weighting factor for orientation classification objective.
    - train_class_weights: Per-sample class weights for handling dataset imbalance.
    - global_step: The total number of batches processed so far.
    - is_plateau_warmup: If True, a special per-batch linear warmup is performed for the plateau scheduler.
    - base_lr: The target learning rate to reach at the end of the warmup.
    
    Returns:
    - train_pose_accuracy: Classification accuracy for pose predictions (0.0 if disabled).
    - train_orient_accuracy: Classification accuracy for orientation predictions (0.0 if disabled).
    - train_recon_loss_avg: Average reconstruction loss across all batches (0.0 if disabled).
    - train_weighted_recon_loss_avg: Average weighted reconstruction loss (0.0 if disabled).
    - train_pose_loss_avg: Average pose classification loss (0.0 if disabled).
    - train_weighted_pose_loss_avg: Average weighted pose classification loss (0.0 if disabled).
    - train_orient_loss_avg: Average orientation classification loss (0.0 if disabled).
    - train_weighted_orient_loss_avg: Average weighted orientation classification loss (0.0 if disabled).
    - train_total_loss_avg: Average combined weighted loss across all enabled objectives.
    - total_inputs: Total number of input sequences processed during the epoch.
    - time_forward_pass: Total time spent on forward propagation (seconds).
    - time_optimizer_step: Total time spent on gradient computation and parameter updates (seconds).
    - global_step: The updated total number of batches processed.
    
    Notes:
    - Uses label smoothing (0.1) for classification losses to improve generalization
    - Applies gradient clipping (max_norm=grad_clip_norm) to prevent exploding gradients
    - Tracks detailed timing information for performance analysis and bottleneck identification
    - Supports flexible multi-head training with independent loss weighting
    - **Non-blocking Transfer**: Asynchronous GPU memory transfers
    
    Parameters:
    - model: PosePredTransformer model in training mode for parameter updates.
    - train_dataloader: DataLoader providing batched training sequences and labels.
    - optimizer: AdamW optimizer for model parameter updates with weight decay.
    - scheduler: Learning rate scheduler with current state and configuration.
    - device: PyTorch device for computation (CPU or CUDA GPU).
    - scaler: Gradient scaler for automatic mixed precision training stability.
    - columns_to_noise: List of feature column indices for noise augmentation.
    - pose_pred: Whether pose classification head is enabled for training.
    - orient_pred: Whether orientation classification head is enabled for training.
    - recon: Whether reconstruction objective is enabled for training.
    - recon_weight: Loss weighting factor for reconstruction objective.
    - pose_weight: Loss weighting factor for pose classification objective.
    - orient_weight: Loss weighting factor for orientation classification objective.
    - train_class_weights: Per-sample class weights for handling dataset imbalance.
    - global_step: The total number of batches processed so far.
    - is_plateau_warmup: If True, a special per-batch linear warmup is performed for the plateau scheduler.
    - base_lr: The target learning rate to reach at the end of the warmup.
    
    Returns:
    - train_pose_accuracy: Classification accuracy for pose predictions (0.0 if disabled).
    - train_orient_accuracy: Classification accuracy for orientation predictions (0.0 if disabled).
    - train_recon_loss_avg: Average reconstruction loss across all batches (0.0 if disabled).
    - train_weighted_recon_loss_avg: Average weighted reconstruction loss (0.0 if disabled).
    - train_pose_loss_avg: Average pose classification loss (0.0 if disabled).
    - train_weighted_pose_loss_avg: Average weighted pose classification loss (0.0 if disabled).
    - train_orient_loss_avg: Average orientation classification loss (0.0 if disabled).
    - train_weighted_orient_loss_avg: Average weighted orientation classification loss (0.0 if disabled).
    - train_total_loss_avg: Average combined weighted loss across all enabled objectives.
    - total_inputs: Total number of input sequences processed during the epoch.
    - time_forward_pass: Total time spent on forward propagation (seconds).
    - time_optimizer_step: Total time spent on gradient computation and parameter updates (seconds).
    - global_step: The updated total number of batches processed.
    
    Notes:
    - Uses label smoothing (0.1) for classification losses to improve generalization
    - Applies gradient clipping (max_norm=grad_clip_norm) to prevent exploding gradients
    - Tracks detailed timing information for performance analysis and bottleneck identification
    - Supports flexible multi-head training with independent loss weighting
    """
    model.train()  # Set the model to training mode

    # Initialize accumulators for losses and metrics
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

    # Initialize timers for performance measurement
    time_forward_pass = 0.0
    time_optimizer_step = 0.0

    optimizer.zero_grad()  # Clear gradients before starting the training loop

    # Iterate over batches in the DataLoader
    for batch_idx, batch in enumerate(train_dataloader):
        sequences = batch[0].to(device, non_blocking=True)  # Move input sequences to the device
        poses = batch[1].to(device, non_blocking=True) if pose_pred else None  # Move pose labels to the device if needed
        orients = batch[2].to(device, non_blocking=True) if orient_pred else None  # Move orientation labels to the device if needed
        indices = batch[3]  # Indices of the samples in the batch

        # Retrieve class weights for the current batch
        batch_class_weights = train_class_weights[indices].to(device, non_blocking=True)

        # Apply data augmentation (Gaussian noise + optional masking).
        # - Always use augmented inputs for the forward pass.
        # - If recon is enabled, return the keep_mask so recon loss is computed ONLY on masked elements.
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


        # Forward pass
        start_time = time.time()
        with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            pose_pred_out, orient_pred_out, recon_output, context_vector = model(sequences_aug)

            # --- Reconstruction objective (denoising + masked-only loss) ---
            # Target is ALWAYS the original clean sequence.
            # Loss is computed ONLY on masked elements.
            if recon:
                if aug_keep_mask is None:
                    raise RuntimeError("Internal error: recon=True but aug_keep_mask is None")
                recon_loss = masked_mse_loss(recon_output, sequences, aug_keep_mask)
            else:
                recon_loss = torch.zeros((), device=device, dtype=torch.float32)

            # Calculate pose prediction loss if enabled
            if pose_pred:
                pose_pred_out = pose_pred_out.view(-1, pose_pred_out.size(-1))
                pose_labels = poses.view(-1).long()  # Use poses directly as class indices
                pose_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)(pose_pred_out, pose_labels)
                weighted_pose_loss = (pose_loss * batch_class_weights).mean()  # Apply class weights
            else:
                weighted_pose_loss = 0.0

            # Calculate orientation prediction loss if enabled
            if orient_pred:
                orient_pred_out = orient_pred_out.view(-1, orient_pred_out.size(-1))
                orient_labels = orients.view(-1).long()  # Use orients directly as class indices
                orient_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)(orient_pred_out, orient_labels)
                weighted_orient_loss = (orient_loss * batch_class_weights).mean()  # Apply class weights
            else:
                weighted_orient_loss = 0.0

        time_forward_pass += time.time() - start_time  # Update forward pass time

        # Calculate total weighted loss
        weighted_recon_loss = recon_weight * recon_loss
        weighted_pose_loss = pose_weight * weighted_pose_loss
        weighted_orient_loss = orient_weight * weighted_orient_loss
        total_loss = weighted_recon_loss + weighted_pose_loss + weighted_orient_loss

        # Gradient pass and optimizer step
        start_time = time.time()
        optimizer.zero_grad()  # Clear gradients before the gradient pass
        scaler.scale(total_loss).backward()  # Gradient pass with scaled loss

        # IMPORTANT (AMP): Gradients are scaled when using GradScaler. We must unscale them
        # before applying gradient clipping, otherwise the effective clip threshold becomes
        # scale-dependent and training can become unstable or overly damped.
        if grad_clip_norm is not None and grad_clip_norm > 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)  # Update weights
        scaler.update()  # Update the scale for next iteration
        time_optimizer_step += time.time() - start_time  # Update optimizer step time

        # --- Scheduler Step / Warmup Logic ---
        if step_scheduler_per_batch:
            if is_plateau_warmup:
                # Manual linear warmup for the first epoch of the plateau scheduler
                start_lr = 1e-8
                num_warmup_steps = len(train_dataloader)
                lr = start_lr + (base_lr - start_lr) * (batch_idx + 1) / num_warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # The other schedulers are wrapped in SequentialLR and are stepped per-batch
                if scheduler_type in ['sqrt', 'cosine', 'linear']:
                    scheduler.step()

        global_step += 1
        
        # Accumulate metrics
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

    # Calculate average losses and accuracy
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
    """
    Performs comprehensive model validation with detailed performance metrics and statistical analysis.
    
    This function executes a complete validation pass through the dataset without gradient computation,
    evaluating model performance across all enabled prediction heads. It computes accuracy metrics,
    loss values, confusion matrices, and F1 scores to provide thorough assessment of model quality
    and generalization capability for early stopping and hyperparameter selection.
    
    Validation Process:
    1. **Model Preparation**: Sets model to evaluation mode (disables dropout/batch norm updates)
    2. **Gradient Disabling**: Uses torch.no_grad() context for memory efficiency
    3. **Batch Processing**: Iterates through validation data without augmentation
    4. **Multi-Objective Evaluation**: Computes predictions for all enabled heads
    5. **Loss Computation**: Calculates weighted and unweighted losses with class balancing
    6. **Metrics Accumulation**: Tracks predictions and labels for statistical analysis
    7. **Statistical Analysis**: Computes confusion matrices and macro F1 scores
    8. **Performance Summary**: Aggregates all metrics for comprehensive evaluation
    
    Evaluation Metrics:
    - **Accuracy**: Simple classification accuracy for pose and orientation
    - **Loss Values**: Both raw and class-weighted losses for all objectives
    - **Confusion Matrices**: Full confusion matrices for detailed error analysis
    - **Macro F1 Scores**: Class-balanced F1 scores robust to class imbalance
    - **Reconstruction Error**: Mean squared error for sequence reconstruction
    
    Statistical Robustness:
    - **Class Weighting**: Consistent with training for fair comparison
    - **Zero Division Handling**: Graceful handling of edge cases in F1 computation
    - **Memory Efficiency**: No gradient storage for faster validation
    - **Mixed Precision**: Automatic casting for consistent evaluation
    
    Parameters:
    - model: PosePredTransformer model in evaluation mode for inference.
    - val_dataloader: DataLoader providing batched validation sequences and labels.
    - device: PyTorch device for computation (CPU or CUDA GPU).
    - pose_pred: Whether pose classification head is enabled for evaluation.
    - orient_pred: Whether orientation classification head is enabled for evaluation.
    - recon: Whether reconstruction objective is enabled for evaluation.
    - val_class_weights: Per-sample class weights for consistent loss computation.
    
    Returns:
    - val_pose_accuracy: Classification accuracy for pose predictions (0.0 if disabled).
    - val_orient_accuracy: Classification accuracy for orientation predictions (0.0 if disabled).
    - val_recon_loss_avg: Average reconstruction loss across validation set (0.0 if disabled).
    - val_pose_loss_avg: Average unweighted pose classification loss (0.0 if disabled).
    - val_weighted_pose_loss_avg: Average class-weighted pose loss (0.0 if disabled).
    - val_orient_loss_avg: Average unweighted orientation classification loss (0.0 if disabled).
    - val_weighted_orient_loss_avg: Average class-weighted orientation loss (0.0 if disabled).
    - val_total_loss_avg: Average combined loss across all enabled objectives.
    - pose_conf_matrix: Confusion matrix for pose predictions (None if disabled).
    - orient_conf_matrix: Confusion matrix for orientation predictions (None if disabled).
    - pose_macro_f1: Macro-averaged F1 score for pose classification (None if disabled).
    - orient_macro_f1: Macro-averaged F1 score for orientation classification (None if disabled).
    
    Notes:
    - Pose/orient metrics are computed on clean sequences (no augmentation) for consistent evaluation.
    - If recon is enabled, reconstruction loss is evaluated on corrupted inputs (noise+mask)
      with clean targets, and the corruption is made deterministic during validation.
    - Confusion matrices provide detailed per-class error analysis
    - Macro F1 scores give equal weight to all classes regardless of frequency
    - All metrics are computed efficiently in a single validation pass
    - Supports partial evaluation when only some prediction heads are enabled
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize accumulators for losses and metrics
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

    # Initialize lists to store predictions and true labels for confusion matrix calculation
    all_pose_preds = []
    all_pose_labels = []
    all_orient_preds = []
    all_orient_labels = []

    # NOTE: For denoising/masked reconstruction validation we need augmentation.
    # To keep val metrics stable across epochs, we make recon-augmentation deterministic
    # and then restore the RNG state when we return.
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    if recon:
        torch.manual_seed(0)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(0)

    try:
        with torch.no_grad():  # Disable gradient calculation for validation
            for batch in val_dataloader:
                # Unpack the batch based on the flags
                sequences = batch[0].to(device)

                # Ensure the input is 3D
                if sequences.dim() != 3:
                    raise ValueError(f"Expected 3 dimensions, but got {sequences.dim()} dimensions")

                poses = batch[1].to(device) if pose_pred else None
                orients = batch[2].to(device) if orient_pred else None
                indices = batch[3]  # Assuming indices are passed as part of the batch

                # Retrieve class weights for the current batch
                batch_class_weights = val_class_weights[indices].to(device)

                # --- Classification evaluation on CLEAN sequences (no augmentation) ---
                if pose_pred or orient_pred:
                    with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        pose_pred_out, orient_pred_out, _recon_unused, _ctx_unused = model(sequences)

                        # Calculate pose prediction loss if enabled
                        if pose_pred:
                            pose_pred_out = pose_pred_out.view(-1, pose_pred_out.size(-1))
                            pose_labels = poses.view(-1).long()  # Use poses directly as class indices
                            pose_loss = nn.CrossEntropyLoss(reduction='none')(pose_pred_out, pose_labels)
                            weighted_pose_loss = (pose_loss * batch_class_weights).mean()
                            val_pose_loss_total += pose_loss.mean().item()
                            val_weighted_pose_loss_total += (pose_weight * weighted_pose_loss).item()
                            _, predicted_poses = torch.max(pose_pred_out, 1)

                            # Update correct predictions count
                            val_correct_pose_predictions += (predicted_poses == pose_labels).sum().item()
                            val_total_pose_predictions += poses.size(0)

                            # Store predictions and true labels for poses
                            all_pose_preds.extend(predicted_poses.cpu().numpy())
                            all_pose_labels.extend(pose_labels.cpu().numpy())

                        # Calculate orientation prediction loss if enabled
                        if orient_pred:
                            orient_pred_out = orient_pred_out.view(-1, orient_pred_out.size(-1))
                            orient_labels = orients.view(-1).long()  # Use orients directly as class indices
                            orient_loss = nn.CrossEntropyLoss(reduction='none')(orient_pred_out, orient_labels)
                            weighted_orient_loss = (orient_loss * batch_class_weights).mean()
                            val_orient_loss_total += orient_loss.mean().item()
                            val_weighted_orient_loss_total += (orient_weight * weighted_orient_loss).item()
                            _, predicted_orients = torch.max(orient_pred_out, 1)

                            # Update correct predictions count
                            val_correct_orient_predictions += (predicted_orients == orient_labels).sum().item()
                            val_total_orient_predictions += orients.size(0)

                            # Store predictions and true labels for orientations
                            all_orient_preds.extend(predicted_orients.cpu().numpy())
                            all_orient_labels.extend(orient_labels.cpu().numpy())

                # --- Reconstruction evaluation on CORRUPTED inputs, clean targets ---
                if recon:
                    # For denoising-style recon runs:
                    # - target is ALWAYS the clean original sequence
                    # - loss is computed ONLY on masked elements
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

    # Compute confusion matrices
    pose_conf_matrix = confusion_matrix(all_pose_labels, all_pose_preds) if pose_pred else None
    orient_conf_matrix = confusion_matrix(all_orient_labels, all_orient_preds) if orient_pred else None

    # Compute macro F1 scores
    pose_macro_f1 = f1_score(all_pose_labels, all_pose_preds, average='macro', zero_division=0.0) if pose_pred else None
    orient_macro_f1 = f1_score(all_orient_labels, all_orient_preds, average='macro', zero_division=0.0) if orient_pred else None

    # Compute combined confusion matrix and F1
    combined_conf_matrix = None
    combined_macro_f1 = None
    if pose_pred and orient_pred:
        # Create reverse dictionaries to map index -> name
        reverse_pose_dict = {v: k for k, v in pose_dict.items()}
        reverse_orient_dict = {v: k for k, v in orient_dict.items()}

        # Convert index lists to name lists
        pose_true_names = [reverse_pose_dict[i] for i in all_pose_labels]
        pose_pred_names = [reverse_pose_dict[i] for i in all_pose_preds]
        orient_true_names = [reverse_orient_dict[i] for i in all_orient_labels]
        orient_pred_names = [reverse_orient_dict[i] for i in all_orient_preds]

        # Create combined labels (e.g., 'walk-forward')
        combined_true_labels = [f"{p}-{o}" for p, o in zip(pose_true_names, orient_true_names)]
        combined_pred_labels = [f"{p}-{o}" for p, o in zip(pose_pred_names, orient_pred_names)]

        # Define all possible combined class names to ensure matrix dimensions are correct
        all_combined_names = [f"{p}-{o}" for p in sorted(pose_dict.keys()) for o in sorted(orient_dict.keys())]

        # Calculate combined matrix and F1 score
        combined_conf_matrix = confusion_matrix(combined_true_labels, combined_pred_labels, labels=all_combined_names)
        combined_macro_f1 = f1_score(combined_true_labels, combined_pred_labels, average='macro', zero_division=0.0)

    # Calculate average losses and accuracy
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


def main_worker(train_dataset: Subset, val_dataset: Subset, pose_dict: Optional[dict[str, int]], 
                orient_dict: Optional[dict[str, int]], num_workers: int, sequence_length: int, batch: int, 
                pretrained_model_path: str, split_filenames: dict[str, list[str]], num_features: int, 
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
                feature_plan: Optional["FeaturePlan"] = None) -> None:
    """
    Main worker function for training and validation.
    
    Execution Flow:
    
    1. **Setup Phase**
       - Configure GPU device and memory optimizations
       - Create DataLoaders with performance settings (pin_memory, persistent_workers)
       - Initialize training infrastructure and timing variables
    
    2. **Model Initialization**
       - Build PosePredTransformer with specified architecture parameters
       - Setup optimizer (AdamW), scheduler (warmup/cosine/plateau), and mixed precision scaler
       - Load from checkpoint if resuming training, otherwise start fresh
    
    3. **Logging Infrastructure Setup**
       - Create CSV records file for epoch-wise metrics tracking
       - Initialize confusion matrix logging files for classification analysis
       - Setup WandB experiment tracking if enabled
    
    4. **Main Training Loop** (for each epoch until convergence or max_epochs)
       - **Training Phase**:
         • Run train_one_epoch() with data augmentation (masking and noise)
         • Collect metrics: losses (pose, orient, recon), accuracies, F1 scores, timing
         • Perform optimizer step with gradient scaling for mixed precision
       
       - **Validation Phase**:
         • Run validate_one_epoch() for clean classification evaluation and
           deterministic reconstruction evaluation when recon is enabled
         • Generate confusion matrices for pose/orientation classification
         • Calculate macro F1 scores and validation losses
       
       - **Performance Monitoring**:
         • Log all metrics to CSV file and WandB (if enabled)
         • Check for best model performance using the active primary metric
           (pose macro-F1, orient macro-F1, or loss depending on run mode)
         • Save best model checkpoint when improvement detected
         • Apply learning rate scheduling based on validation metrics
       
       - **Progress Reporting**:
         • Print comprehensive epoch summary with all metrics and timing
         • Show early stopping countdown if applicable
    
    5. **Checkpoint Management**
       - Save periodic training state every checkpoint_interval epochs
       - Preserve best model separately for final deployment
       - Handle graceful resumption from any saved checkpoint
    
    6. **Early Stopping & Cleanup**
       - Monitor the active primary metric for plateau detection
       - Stop training after patience epochs without improvement
       - Close all file handles and cleanup GPU memory
       - Print final training summary with best achieved performance
    
    Key Features:
    - **Multi-task Learning**: Supports pose classification, orientation prediction, and reconstruction
    - **Robust Checkpointing**: Resume training from any epoch with exact state restoration
    - **Smart Early Stopping**: Uses appropriate metric based on enabled prediction heads
    - **Comprehensive Logging**: CSV metrics, confusion matrices, WandB integration
    - **Memory Optimization**: Mixed precision training with automatic loss scaling
    - **Data Augmentation**: Element-wise masking and Gaussian noise injection
    
    Parameters:
        train_dataset, val_dataset: PyTorch Subsets for training and validation
        pose_dict, orient_dict: Class name to index mappings (None if not predicting)
        num_workers, batch: DataLoader configuration for performance
        pretrained_model_path: Checkpoint file for resuming or starting fresh
        Model architecture params: d_model, nhead, num_encoder_layers, etc.
        Training config: num_epochs, checkpoint_interval, learning rates, weights
        Logging paths: records_path, confusion matrix files, run_name
        Data augmentation: columns_to_noise, mask/noise parameters
    
    Outputs:
        - CSV training records with epoch metrics
        - Confusion matrix files for classification analysis  
        - Model checkpoints (periodic + best model)
        - WandB experiment logs (if enabled)
    """
    # Set the device to GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the current device
    print(f"Current device: {device}\n")

    # Create DataLoaders for training and validation
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

    # ----------------------------------------------------------------------------------
    # Compute SWA schedule up front so checkpoints capture the exact run contract.
    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    # Populate checkpoint module globals so saved checkpoints include full hyperparams
    # ----------------------------------------------------------------------------------
    extra_globals = {}
    if feature_plan is not None:
        extra_globals = {
            'feature_schema_hash': feature_plan.schema_hash,
            'feature_columns_included': list(feature_plan.included_columns),
            'feature_columns_excluded': list(feature_plan.excluded_columns),
            'feature_columns_canonical': list(feature_plan.canonical_columns),
            'ablation_spec': dict(feature_plan.spec),
        }

    set_checkpoint_globals(
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
    # MODEL INITIALIZATION
    # Determine if this is a pretrain → supervised handoff (affects optimizer policy)
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
        weight_encoder=weight_encoder
    )
    print_gpu_memory_summary()

    # JIT Compile the model - this might be worth trying out, although in our case it causes the training to run slightly slower.
    #print("Compiling the model for optimized performance...")
    #model = torch.compile(model)

    # Initialize weights
    pose_conf_matrix = None
    orient_conf_matrix = None
    # Load the pretrained weights if the file exists
    # NOTE: Optimizer/scheduler/scaler state should only be loaded for a true resume.
    # If you are doing any kind of new training stage (restart_optimizer, freeze_encoder, differential LR, or pretrain→supervised handoff),
    # we intentionally skip loading optimizer state to avoid param-group mismatches and silent training bugs.
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
        model, optimizer, scheduler, pretrained_model_path, scaler, load_optimizer_state=load_optimizer_state, pretrain_mode=False
    )

    # Setup the records file and capture headers
    file, writer, headers = setup_records_file(
        records_path, 
        start_epoch,  
        pose_pred=pose_pred, 
        orient_pred=orient_pred,
        recon=recon
    )

    if file is None or writer is None or headers is None:
        raise RuntimeError(f"Failed to initialize records file at {records_path}. See earlier errors above.")

    # Setup confusion matrix files using the val_dataloader
    if not noval:
        setup_confusion_matrix_files(
            val_dataloader=val_dataloader,
            pose_conf_matrix_path=pose_conf_matrix_path,
            orient_conf_matrix_path=orient_conf_matrix_path,
            combined_conf_matrix_path=combined_conf_matrix_path,
            pose_pred=pose_pred, 
            orient_pred=orient_pred
        )          

    # GLOBAL TIME VARIABLES / RESUMABLE TRAINING STATE
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

    # --- SWA runtime state (optional) ---
    # SWA is implemented as a final training tail. During that tail we keep training the base
    # model normally while maintaining an AveragedModel shadow copy and an SWALR scheduler.
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
        print(f"No epochs to run: start_epoch={start_epoch} >= effective_num_epochs={effective_num_epochs}. Exiting.")
        file.close()
        return

    # MAIN TRAINING LOOP
    for epoch in range(start_epoch, effective_num_epochs):
        epoch_start_time = time.time()
        
        current_lr = [pg['lr'] for pg in optimizer.param_groups]

        # Determine whether we're in the SWA phase for this epoch.
        in_swa_phase = bool(swa_mode and (swa_start is not None) and (epoch >= swa_start))

        # Lazily initialize SWA objects the moment we enter the SWA phase.
        if in_swa_phase and swa_model is None:
            print(f"[SWA] Starting SWA phase at epoch {epoch + 1} (0-index={epoch}, swa_start={swa_start}).")
            swa_model = AveragedModel(model)
            swa_model.to(device)
            swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs)

        # Pick the active scheduler for this epoch.
        active_scheduler = swa_scheduler if in_swa_phase else scheduler

        # Training
        # The 'plateau' scheduler needs a manual warmup, handled by this flag.
        # The other schedulers have warmup built into their SequentialLR wrapper.
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

        # Validation
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
            # Substitute training metrics for validation metrics when noval=True
            print("noval=True: Substituting training metrics for validation metrics.")
            val_pose_accuracy = train_pose_accuracy
            val_orient_accuracy = train_orient_accuracy
            val_recon_loss_avg = train_recon_loss_avg
            val_pose_loss_avg = train_pose_loss_avg
            val_weighted_pose_loss_avg = train_weighted_pose_loss_avg
            val_orient_loss_avg = train_orient_loss_avg
            val_weighted_orient_loss_avg = train_weighted_orient_loss_avg
            val_total_loss_avg = train_total_loss_avg
            # Set metrics without training equivalents to default values
            pose_conf_matrix = None
            orient_conf_matrix = None
            pose_macro_f1 = 0.0  # Default to 0 as it's used for comparison
            orient_macro_f1 = 0.0
            combined_conf_matrix = None
            combined_macro_f1 = 0.0


        # Step the scheduler(s) after validation / epoch end
        if in_swa_phase:
            # SWA phase: update the running average model and step SWALR *per epoch*.
            if swa_model is None or swa_scheduler is None:
                raise RuntimeError("Internal error: in_swa_phase=True but swa_model/swa_scheduler is not initialized")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # Normal (non-SWA) training phase: keep existing scheduler behavior.
            if scheduler_type == 'plateau' and not is_plateau_warmup_epoch:
                # The plateau scheduler is always stepped per-epoch, but only after its warmup is complete.
                if noval:
                    # When noval is true, we need a metric to step the scheduler.
                    # If pose_pred is on, we'll use training loss as a proxy.
                    # A more robust approach could be to calculate F1 on the training set,
                    # but that would add overhead. Using loss is a reasonable default.
                    metric_for_scheduler = train_total_loss_avg
                    scheduler.step(train_total_loss_avg)
                else:
                    if pose_pred:
                        scheduler.step(-pose_macro_f1)
                    else:
                        scheduler.step(val_total_loss_avg)
        # The per-batch schedulers are stepped in train_one_epoch, so no 'else' block is needed here.

        # Log confusion matrices
        if not noval:
            if pose_pred:
                log_confusion_matrix(pose_conf_matrix, val_dataloader.dataset.dataset.pose_dict.keys(), epoch+1, pose_conf_matrix_path)
            if orient_pred:
                log_confusion_matrix(orient_conf_matrix, val_dataloader.dataset.dataset.orient_dict.keys(), epoch+1, orient_conf_matrix_path)

            # Log combined confusion matrix
            if pose_pred and orient_pred:
                pose_class_names = sorted(val_dataloader.dataset.dataset.pose_dict.keys())
                orient_class_names = sorted(val_dataloader.dataset.dataset.orient_dict.keys())
                combined_class_names = [f"{p}-{o}" for p in pose_class_names for o in orient_class_names]
                log_confusion_matrix(combined_conf_matrix, combined_class_names, epoch+1, combined_conf_matrix_path)
                
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_epoch_time += epoch_time
        num_completed_epochs += 1
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
            pose_macro_f1=float(pose_macro_f1),
            orient_macro_f1=float(orient_macro_f1),
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
            epochs_since_improvement = 0  # Reset epochs since improvement
            # Save best model checkpoint (only keep one)
            if best_model_path is not None and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except Exception as e:
                    print(f"Warning: Could not delete previous best model checkpoint: {best_model_path}. Error: {e}")
            # Save new best model
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
        else:
            epochs_since_improvement += 1  # Increment epochs since improvement

        if (not swa_mode) and epochs_since_improvement >= patience:
            print(f"No improvement in {patience} epochs. Stopping training.")
            break

        # Prepare a dictionary of metrics for logging
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

        # Log epoch results
        log_epoch_results(
            file, writer, epoch, metrics, headers,
            pose_pred, orient_pred, recon
        )

        # Also log to wandb
        if wandb_log:
            log_wandb_metrics(epoch, metrics, pose_pred, orient_pred, recon)

        print(f'\nEpoch {epoch + 1}/{effective_num_epochs}, Avg Time per Epoch: {avg_time_per_epoch:.4f} sec, Learning Rate: {current_lr}')
        if pose_pred:
            print(f'Pose: Train Loss: {train_pose_loss_avg:.4f}, Val Loss: {val_pose_loss_avg:.4f}, Train Acc: {train_pose_accuracy:.4f}, Val Acc: {val_pose_accuracy:.4f}')
        if orient_pred:
            print(f'Orient: Train Loss: {train_orient_loss_avg:.4f}, Val Loss: {val_orient_loss_avg:.4f}, Train Acc: {train_orient_accuracy:.4f}, Val Acc: {val_orient_accuracy:.4f}')
        if recon:
            print(f'Recon: Train Loss: {train_recon_loss_avg:.4f}, Val Loss: {val_recon_loss_avg:.4f}')


        if (epoch + 1) % checkpoint_interval == 0:
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

        # Periodic memory cleanup.
        if MEMORY_CLEANUP_EVERY_EPOCHS and ((epoch + 1) % int(MEMORY_CLEANUP_EVERY_EPOCHS) == 0):
            print(f"[Memory cleanup] epoch {epoch + 1} (every {MEMORY_CLEANUP_EVERY_EPOCHS} epochs)")
            cleanup_memory()



    # --- Finalize + save SWA checkpoint (optional) ---
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

            # Optional SWA validation
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

                # Append a final SWA summary row to the training record
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

                print(f"[SWA] Validation summary: total_loss={swa_val_total_loss_avg:.4f} | pose_F1={swa_pose_macro_f1:.4f} | orient_F1={swa_orient_macro_f1:.4f}")
            else:
                print("[SWA] noval=True: Skipping SWA validation.")

            # Save SWA checkpoint (weights + metadata; no optimizer/scheduler state because this is a final averaged artifact)
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

            torch.save(checkpoint_data, swa_checkpoint_path)
            print(f"[SWA] Checkpoint saved to {swa_checkpoint_path}")



    file.close()
    print("Training loop completed")


