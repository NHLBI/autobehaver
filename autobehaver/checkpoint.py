"""Checkpoint utilities for training/inference.

Contains helpers to save and load model checkpoints robustly across different run modes
(supervised training, pretraining, SWA fine-tuning, and inference). Checkpoints store both model
weights and enough metadata ("global_vars") to ensure the loaded model matches the training
configuration."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler

from .constants import STANDARD_WEIGHT_INIT_POLICY
from .model import PosePredTransformer

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def set_checkpoint_globals(**kwargs):
    """
    Set module-level variables that will be serialized into checkpoint['global_vars'].

    The training and pretraining code calls this helper before saving checkpoints so
    that :func:`save_checkpoint` can capture a complete and consistent metadata
    snapshot in ``checkpoint['global_vars']``.
    """
    globals().update(kwargs)


CHECKPOINT_GLOBAL_KEYS = [
    'sequence_length', 'num_features', 'd_model', 'nhead', 'num_encoder_layers',
    'num_decoder_layers', 'dim_feedforward', 'dropout_rate', 'frame_representation',
    'num_cls', 'prediction_head', 'positional_encoding_type',
    # Feature schema / ablation (optional but recommended for reproducibility)
    'feature_schema_hash', 'feature_columns_included', 'feature_columns_excluded', 'feature_columns_canonical',
    'ablation_spec',
    # Augmentation columns (important when features are ablated)
    'columns_to_noise',
    'batch', 'base_lr', 'weight_decay', 'warmup_steps', 'mask_p', 'noise_level',
    'pose_pred', 'orient_pred', 'recon', 'pose_weight', 'orient_weight', 'recon_weight',
    'scheduler_type', 't_0', 't_mult', 'eta_min',
    'projection_dim', 'temperature', 'pretrain_files_per_group', 'log_interval', 'pretrain_loss', 'barlow_lambda',
    'pretrain_mode', 'patience', 'plateau_factor',
    'grad_clip_norm', 'random_seed', 'num_epochs', 'checkpoint_interval',
    # SWA config (for exact resume / auditability)
    'swa_mode', 'swa_start', 'swa_epochs', 'swa_lr', 'swa_anneal_epochs',
    # Stage-specific knobs (saved for transparency; not always reloaded)
    'freeze_encoder', 'weight_encoder', 'weight_initialization',
]


def build_checkpoint_global_vars(
    *,
    pose_pred: bool,
    orient_pred: bool,
    recon: bool,
    pretrain_mode: bool = False,
    extra_overrides: Optional[dict] = None,
) -> dict:
    """Collect the metadata snapshot stored in ``checkpoint['global_vars']``.

    Using a single helper keeps normal checkpoints and special-case SWA checkpoints
    in sync, which matters for inference compatibility (feature schema, ablation
    metadata, augmentation columns, etc.).
    """
    global_vars = {k: globals()[k] for k in CHECKPOINT_GLOBAL_KEYS if k in globals()}
    global_vars['pose_pred'] = bool(pose_pred)
    global_vars['orient_pred'] = bool(orient_pred)
    global_vars['recon'] = bool(recon)
    global_vars['pretrain_mode'] = bool(pretrain_mode)
    global_vars['weight_initialization'] = STANDARD_WEIGHT_INIT_POLICY
    if extra_overrides:
        global_vars.update(dict(extra_overrides))
    return global_vars
def load_checkpoint_variables(
    args: argparse.Namespace,
    checkpoint_path: str,
    use_restart_optimizer: bool,
    current_num_features: Optional[int] = None,
    current_feature_schema_hash: Optional[str] = None,
    current_feature_columns_included: Optional[list[str]] = None,
) -> None:
    """
    Load model/training configuration variables from a checkpoint and apply them to an argparse Namespace.

    Behavior:
    - Always restores **architecture-critical** settings (sequence_length, d_model, etc.) so the model
      structure matches the checkpoint.
    - If `use_restart_optimizer=True`, we treat this as a *new stage* and do **not** overwrite training
      hyperparameters (LR/scheduler/etc.).
    - If the checkpoint is a **pretraining** checkpoint and the current run is supervised (args.pretrain_mode=False),
      we treat it as a *handoff* and only restore architecture settings (never overwrite supervised flags).
    - If checkpoint global_vars includes `num_features`, this function can validate against `current_num_features`
      and will raise if they mismatch.

    Note:
        This function is intentionally conservative: if you want to override architecture by hand, do it
        by training from scratch or by exporting a compatible checkpoint.
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        # No checkpoint to load.
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    global_vars = checkpoint.get("global_vars", {})
    if not isinstance(global_vars, dict):
        global_vars = {}

    is_pretrain_ckpt = bool(global_vars.get("pretrain_mode", False))
    is_handoff = bool(is_pretrain_ckpt and (not bool(getattr(args, "pretrain_mode", False))))

    # --- Architecture: always restore ---
    current_sequence_length = int(getattr(args, "sequence_length", global_vars.get("sequence_length", 0) or 0) or 0)
    checkpoint_sequence_length = int(global_vars.get("sequence_length", current_sequence_length))
    if checkpoint_sequence_length % 2 == 0:
        print(f"[SEQ_LEN] Loaded even sequence_length={checkpoint_sequence_length} from checkpoint; bumping to {checkpoint_sequence_length + 1}.")
        checkpoint_sequence_length += 1
    if current_sequence_length and checkpoint_sequence_length != current_sequence_length:
        print(f"[CKPT] Overriding sequence_length: {current_sequence_length} -> {checkpoint_sequence_length}")
    args.sequence_length = checkpoint_sequence_length

    # num_features guardrail (do not silently override mismatched feature layouts)
    ckpt_num_features = global_vars.get("num_features", None)
    if ckpt_num_features is not None:
        ckpt_num_features = int(ckpt_num_features)
        if current_num_features is not None:
            if int(current_num_features) != ckpt_num_features:
                raise ValueError(
                    f"[CKPT] num_features mismatch: data reports {int(current_num_features)}, checkpoint expects {ckpt_num_features}. "
                    "This usually means you are pointing to a different HDF5 feature layout than the checkpoint was trained on."
                )
        # Keep args in sync as well (useful for downstream prints / metadata)
        setattr(args, "num_features", ckpt_num_features)

    # Feature schema guardrails (optional but strongly recommended)
    # Checkpoints store an ordered list of included columns + a schema hash.
    ckpt_schema_hash = global_vars.get("feature_schema_hash", None)
    ckpt_included = global_vars.get("feature_columns_included", None)
    ckpt_excluded = global_vars.get("feature_columns_excluded", None)
    ckpt_canonical = global_vars.get("feature_columns_canonical", None)
    ckpt_ablation_spec = global_vars.get("ablation_spec", None)

    if ckpt_schema_hash is not None:
        if current_feature_schema_hash is not None and str(current_feature_schema_hash) != str(ckpt_schema_hash):
            raise ValueError(
                f"[CKPT] feature_schema_hash mismatch: current run computed {current_feature_schema_hash} "
                f"but checkpoint expects {ckpt_schema_hash}. "
                "This usually means you changed ablation/feature-selection settings or pointed to a different HDF5 schema."
            )
        setattr(args, "feature_schema_hash", ckpt_schema_hash)

    if ckpt_included is not None:
        # Basic validation
        if not isinstance(ckpt_included, (list, tuple)) or len(ckpt_included) == 0:
            raise ValueError("[CKPT] feature_columns_included exists but is not a non-empty list/tuple.")
        ckpt_included_list = [str(x) for x in ckpt_included]
        if current_feature_columns_included is not None:
            cur_list = [str(x) for x in current_feature_columns_included]
            if ckpt_included_list != cur_list:
                raise ValueError(
                    "[CKPT] feature_columns_included mismatch between current run and checkpoint. "
                    "Refuse to proceed to avoid silent feature-semantic corruption."
                )
        setattr(args, "feature_columns_included", ckpt_included_list)

    # These are audit-only; safe to load if present.
    if ckpt_excluded is not None:
        setattr(args, "feature_columns_excluded", [str(x) for x in ckpt_excluded])
    if ckpt_canonical is not None:
        setattr(args, "feature_columns_canonical", [str(x) for x in ckpt_canonical])
    if ckpt_ablation_spec is not None:
        setattr(args, "ablation_spec", ckpt_ablation_spec)

    # Core architecture hyperparams
    for key in [
        "d_model",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout_rate",
        "frame_representation",
        "num_cls",
        "prediction_head",
        "positional_encoding_type",
        "projection_dim",
    ]:
        if key in global_vars:
            setattr(args, key, global_vars[key])

    # If this is a pretrain→supervised handoff, do NOT overwrite supervised flags/hyperparams.
    if is_handoff:
        print("[CKPT] Detected pretraining checkpoint → supervised handoff. Restoring architecture only (skipping training hyperparams).")
        return

    # If this is a new stage (restart optimizer), do NOT overwrite training hyperparameters.
    if use_restart_optimizer:
        print("[CKPT] restart_optimizer=True: Restoring architecture only (keeping current training hyperparams).")
        return

    # --- Full resume: restore training hyperparameters ---
    for key in [
        "pose_pred",
        "orient_pred",
        "recon",
        "base_lr",
        "weight_decay",
        "warmup_steps",
        "mask_p",
        "noise_level",
        "scheduler_type",
        "t_0",
        "t_mult",
        "eta_min",
        "pose_weight",
        "orient_weight",
        "recon_weight",
        "max_class_weight",
        "grad_clip_norm",
        "patience",
        "plateau_factor",
        "random_seed",
        "num_epochs",
        "checkpoint_interval",
        "swa_mode",
        "swa_epochs",
        "swa_lr",
        "swa_anneal_epochs",
        "freeze_encoder",
        "weight_encoder",
        "temperature",
        "pretrain_files_per_group",
        "log_interval",
        "pretrain_loss",
        "barlow_lambda",
    ]:
        if key in global_vars:
            setattr(args, key, global_vars[key])

    print("[CKPT] Loaded full configuration (architecture + training hyperparams) for resume.")

__all__ = ['load_checkpoint_variables', 'build_checkpoint_global_vars', 'is_pretrain_checkpoint_dict', 'is_pretrain_checkpoint_path', '_align_state_dict_module_prefix', '_move_optimizer_state_to_device', 'load_checkpoint_model', 'save_checkpoint']

def is_pretrain_checkpoint_dict(checkpoint: dict) -> bool:
    """Return True if the checkpoint came from a pretraining run."""
    gv = checkpoint.get('global_vars', {})
    return bool(isinstance(gv, dict) and gv.get('pretrain_mode', False))


def is_pretrain_checkpoint_path(checkpoint_path: Optional[str]) -> bool:
    """Safe file-based wrapper for `is_pretrain_checkpoint_dict`."""
    if not checkpoint_path:
        return False
    if not os.path.isfile(checkpoint_path):
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"[CKPT] Warning: Could not read checkpoint '{checkpoint_path}' to detect type: {e}")
        return False
    return is_pretrain_checkpoint_dict(ckpt)


def _align_state_dict_module_prefix(state_dict: dict, model: nn.Module) -> dict:
    """
    Align DataParallel ('module.') prefixes between a checkpoint state_dict and the current model.

    This prevents hard-to-debug load failures when a checkpoint was saved under DataParallel but
    is being loaded into a non-wrapped model (or vice versa).
    """
    if not isinstance(state_dict, dict):
        return state_dict

    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return state_dict

    state_keys = list(state_dict.keys())
    if not state_keys:
        return state_dict

    state_has_module = any(k.startswith('module.') for k in state_keys)
    model_has_module = any(k.startswith('module.') for k in model_keys)

    if state_has_module and not model_has_module:
        return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    if (not state_has_module) and model_has_module:
        return {f'module.{k}': v for k, v in state_dict.items()}

    return state_dict


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors onto `device` (important when loading on CPU and training on GPU)."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint_model(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    checkpoint_path: Optional[str],
    scaler: Optional[GradScaler],
    load_optimizer_state: bool = True,
    strict: bool = True,
    *,
    pretrain_mode: bool = False,
) -> tuple[int, int]:
    """
    Loads checkpoint weights (and optionally optimizer/scheduler/scaler state) into an existing model.

    Modes:
    - **handoff**: checkpoint is from pretraining and current run is supervised → load backbone weights only.
    - **resume**: load model + optimizer/scheduler/scaler and return (epoch+1, global_step).
    - **weights_only**: load model only, reset (epoch, global_step) to (0, 0).

    Returns:
        (start_epoch, global_step)
    """
    if not checkpoint_path:
        return 0, 0

    if not os.path.exists(checkpoint_path):
        print(f"[CKPT] Checkpoint file not found: {checkpoint_path}")
        return 0, 0

    # Always load on CPU first (lighter on GPU memory). We'll move optimizer state to GPU if needed.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    ckpt_kind = checkpoint.get('checkpoint_kind', None)
    if ckpt_kind is None:
        raise KeyError(f"Checkpoint '{checkpoint_path}' is missing 'checkpoint_kind'.")
    is_pretrain_checkpoint = is_pretrain_checkpoint_dict(checkpoint)

    # If the caller requested a full resume, but the checkpoint does not contain the needed state,
    # fall back to a weights-only load with a clear warning.
    if load_optimizer_state:
        missing = []
        if optimizer is None or 'optimizer_state_dict' not in checkpoint:
            missing.append('optimizer_state_dict')
        if scheduler is None or 'scheduler_state_dict' not in checkpoint:
            missing.append('scheduler_state_dict')
        # scaler is optional; we only warn if it's requested and missing
        if scaler is not None and 'scaler_state_dict' not in checkpoint:
            missing.append('scaler_state_dict')
        if missing:
            print(f"[CKPT] Requested resume but checkpoint lacks {missing}. Forcing weights-only load.")
            load_optimizer_state = False

    is_handoff = bool(is_pretrain_checkpoint) and (not bool(pretrain_mode))

    mode_str = 'handoff' if is_handoff else ('resume' if load_optimizer_state else 'weights_only')
    print(f"[CKPT] Loading checkpoint kind={ckpt_kind} mode={mode_str} from: {checkpoint_path}")

    # --- Model weights ---
    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint '{checkpoint_path}' is missing 'model_state_dict'.")

    state_dict = _align_state_dict_module_prefix(checkpoint['model_state_dict'], model)

    if is_handoff:
        # Pretrain → supervised handoff: load overlapping keys, exclude projection head
        model_state = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if (k in model_state) and (not k.startswith('projection_head.'))
        }
        load_result = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"[CKPT] Handoff load complete. Missing keys: {len(load_result.missing_keys)} | Unexpected keys: {len(load_result.unexpected_keys)}")
        return 0, 0

    # Standard supervised checkpoint load
    if strict:
        model.load_state_dict(state_dict)
    else:
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"[CKPT] Non-strict load. Missing keys: {len(load_result.missing_keys)} | Unexpected keys: {len(load_result.unexpected_keys)}")

    # --- Optimizer / scheduler / scaler ---
    if load_optimizer_state:
        # True resume. These *must* match; if they don't, it's safer to crash loudly.
        model_device = next(model.parameters()).device

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            _move_optimizer_state_to_device(optimizer, model_device)

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"[CKPT] Warning: Failed to load scheduler state_dict (continuing): {e}")

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                print(f"[CKPT] Warning: Failed to load GradScaler state_dict (continuing): {e}")

        start_epoch = checkpoint.get('epoch', -1) + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"[CKPT] Resume state loaded. start_epoch={start_epoch}, global_step={global_step}")
        return start_epoch, global_step

    return 0, 0


def save_checkpoint(epoch: int, global_step: int, model: PosePredTransformer, optimizer: torch.optim.AdamW, 
                   scheduler: Union[InverseSquareRootDecayScheduler, CosineAnnealingScheduler, ReduceLROnPlateau], 
                   pose_dict: Optional[dict[str, int]], orient_dict: Optional[dict[str, int]], 
                   scaler: GradScaler, checkpoint_dir: str, run_name: str, 
                   pose_pred: bool, orient_pred: bool, recon: bool, 
                   split_filenames: Optional[dict[str, list[str]]], pretrain_mode: bool = False,
                   filename_override: Optional[str] = None, swa_runtime_state: Optional[dict] = None,
                   training_runtime_state: Optional[dict] = None,
                   pretraining_runtime_state: Optional[dict] = None) -> None:
    """
    Saves a comprehensive checkpoint containing all training state and configuration for resumption.
    
    This function creates a complete snapshot of the current training state, including model weights,
    optimizer state, scheduler configuration, and training metadata. The checkpoint enables exact
    training resumption from any point, handles pretraining-to-finetuning handoffs, and preserves
    data split information for consistent evaluation across training sessions.
    
    Checkpoint Contents:
    - **Model State**: Complete model weights and architecture parameters
    - **Training State**: Optimizer and scheduler states for exact resumption
    - **Configuration**: All hyperparameters and checkpoint metadata used in training
    - **Data Splits**: Train/validation file lists for consistent data partitioning
    - **Class Mappings**: Label-to-index dictionaries for classification heads
    - **Mixed Precision**: Gradient scaler state for stable FP16/FP32 training
    - **Run Metadata**: Epoch number, prediction flags, and training mode information
    
    The checkpoint file is saved with a descriptive filename including the run name and epoch number,
    making it easy to identify and resume from specific training points. Checkpoint
    metadata is captured to ensure the model can be rebuilt with identical architecture
    and hyperparameters.
    
    Parameters:
    - epoch: Current epoch number (0-indexed) for checkpoint identification.
    - global_step: The total number of batches processed across all epochs.
    - model: PosePredTransformer model with trained weights to save.
    - optimizer: AdamW optimizer with accumulated gradient statistics and momentum.
    - scheduler: Learning rate scheduler with current state and configuration.
    - pose_dict: Dictionary mapping pose class names to integer indices (None if pose_pred=False).
    - orient_dict: Dictionary mapping orientation class names to integer indices (None if orient_pred=False).
    - scaler: Gradient scaler for mixed precision training with loss scaling state.
    - checkpoint_dir: Directory path where the checkpoint file will be saved.
    - run_name: Identifier string for this training run used in checkpoint filename.
    - pose_pred: Whether pose classification head was active during training.
    - orient_pred: Whether orientation classification head was active during training.
    - recon: Whether reconstruction objective was active during training.
    - split_filenames: Dictionary containing 'train_filenames' and 'val_filenames' lists for data consistency.
    - pretrain_mode: Whether this checkpoint is from a pretraining run (affects handoff detection).
    
    Returns:
    - None: Checkpoint is written to disk with confirmation message.
    
    Output File:
    - Creates '{run_name}_model_epoch_{epoch+1}.pth' in the specified checkpoint directory
    - File contains all necessary information for complete training state restoration
    
    Notes:
    - Automatically handles missing dictionaries for disabled prediction heads
    - Preserves exact checkpoint metadata needed for architecture consistency
    - Supports both pretraining and supervised learning checkpoint formats
    - Essential for long-running training jobs and experiment reproducibility
    """
    # Construct the path for the checkpoint file
    if filename_override:
        checkpoint_path = os.path.join(checkpoint_dir, filename_override)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'{run_name}_model_epoch_{epoch + 1}.pth')
    global_vars = build_checkpoint_global_vars(
        pose_pred=pose_pred,
        orient_pred=orient_pred,
        recon=recon,
        pretrain_mode=pretrain_mode,
    )



    # --- Checkpoint metadata ---
    checkpoint_schema_version = 1
    checkpoint_kind = 'pretrain' if pretrain_mode else 'supervised'
    checkpoint_created_at = datetime.now().isoformat()

    # Prepare the checkpoint data dictionary
    checkpoint_data = {
        'checkpoint_schema_version': checkpoint_schema_version,
        'checkpoint_kind': checkpoint_kind,
        'checkpoint_created_at': checkpoint_created_at,
        'run_name': run_name,
        'epoch': epoch,  # Save the current epoch number
        'global_step': global_step,
        'model_state_dict': model.state_dict(),  # Save the model state
        'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer state
        'scheduler_state_dict': scheduler.state_dict(),  # Save the scheduler state
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,  # Save the scaler state if available
        'global_vars': global_vars,  # Store only relevant checkpoint metadata
    }
    if swa_runtime_state is not None:
        checkpoint_data['swa_runtime_state'] = dict(swa_runtime_state)
    if training_runtime_state is not None:
        checkpoint_data['training_runtime_state'] = dict(training_runtime_state)
    if pretraining_runtime_state is not None:
        checkpoint_data['pretraining_runtime_state'] = dict(pretraining_runtime_state)

    # Only add split filenames if they exist (i.e., not in pre-training mode)
    if split_filenames:
        checkpoint_data['train_filenames'] = split_filenames['train_filenames']
        checkpoint_data['val_filenames'] = split_filenames['val_filenames']

    if pose_pred:
        checkpoint_data.update({'pose_dict': pose_dict})
    if orient_pred:
        checkpoint_data.update({'orient_dict': orient_dict})

    torch.save(checkpoint_data, checkpoint_path)
    print(f"\nCheckpoint ({checkpoint_kind}) saved to {checkpoint_path}")


