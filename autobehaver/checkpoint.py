"""Save and restore PosePred weights, training state and model/input metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler

from .constants import STANDARD_WEIGHT_INIT_POLICY
from .model import PosePredTransformer


def set_checkpoint_globals(*, reset=False, **kwargs):
    """Set metadata to include in subsequent checkpoint global_vars snapshots."""
    if reset:
        for name in CHECKPOINT_GLOBAL_KEYS:
            globals().pop(name, None)
    globals().update(kwargs)


CHECKPOINT_GLOBAL_KEYS = [
    'sequence_length', 'num_features', 'd_model', 'nhead', 'num_encoder_layers',
    'num_decoder_layers', 'dim_feedforward', 'dropout_rate', 'frame_representation',
    'num_cls', 'prediction_head', 'positional_encoding_type',
    # Ordered feature schema and selection metadata.
    'feature_schema_hash', 'feature_columns_included', 'feature_columns_excluded', 'feature_columns_canonical',
    'ablation_spec',
    # Augmentation indices follow the selected model columns.
    'columns_to_noise',
    'batch', 'base_lr', 'weight_decay', 'warmup_steps', 'mask_p', 'noise_level', 'max_class_weight',
    'pose_pred', 'orient_pred', 'recon', 'pose_weight', 'orient_weight', 'recon_weight',
    'scheduler_type', 't_0', 't_mult', 'eta_min',
    'schedule_mode', 'warmup_epochs', 'cosine_decay_epochs', 'batches_per_epoch',
    'projection_dim', 'temperature', 'pretrain_files_per_group', 'log_interval', 'pretrain_loss', 'barlow_lambda',
    'pretrain_mode', 'patience', 'plateau_factor',
    'grad_clip_norm', 'random_seed', 'num_epochs', 'checkpoint_interval',
    # Retain the SWA schedule for resume.
    'swa_mode', 'swa_start', 'swa_epochs', 'swa_lr', 'swa_anneal_epochs',
    # Save stage settings even when they are not restored on a handoff.
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
    """Collect shared model/input metadata for ordinary and SWA checkpoints."""
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
    """Restore checkpoint configuration into a CLI namespace.

    Always restore architecture and verify supplied feature width/schema.
    A weights-only restart or pretraining-to-supervised handoff keeps current
    training settings; full resume also restores saved training hyperparameters.
    """
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Requested checkpoint does not exist: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    global_vars = checkpoint.get("global_vars", {})
    if not isinstance(global_vars, dict):
        global_vars = {}

    is_pretrain_ckpt = bool(global_vars.get("pretrain_mode", False))
    is_handoff = bool(is_pretrain_ckpt and (not bool(getattr(args, "pretrain_mode", False))))

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
        setattr(args, "num_features", ckpt_num_features)

    # Equal feature width alone does not establish equal feature meanings.
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

    # Preserve optional selection metadata for provenance.
    if ckpt_excluded is not None:
        setattr(args, "feature_columns_excluded", [str(x) for x in ckpt_excluded])
    if ckpt_canonical is not None:
        setattr(args, "feature_columns_canonical", [str(x) for x in ckpt_canonical])
    if ckpt_ablation_spec is not None:
        setattr(args, "ablation_spec", ckpt_ablation_spec)

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
        "batch", "schedule_mode", "warmup_epochs", "cosine_decay_epochs", "batches_per_epoch",
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
    """Return whether a readable checkpoint file identifies pretraining."""
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
    """Align DataParallel ``module.`` prefixes with the destination model."""
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
    """Move optimizer-state tensors to the destination model device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def validate_supervised_class_schema(checkpoint, pose_dict, orient_dict, *, pose_pred, orient_pred):
    """Validate head meaning and return dictionaries in checkpoint index order."""
    ordered = []
    for name, current, enabled in (("pose", pose_dict, pose_pred), ("orient", orient_dict, orient_pred)):
        saved = checkpoint.get(f"{name}_dict")
        saved_enabled = bool(saved)
        if bool(enabled) != saved_enabled:
            raise ValueError(f"Supervised checkpoint {name} head does not match the active head/schema")
        if not enabled:
            ordered.append({})
            continue
        if not isinstance(saved, dict) or not saved:
            raise ValueError(f"Supervised checkpoint lacks required {name} class dictionary")
        if (any(not isinstance(i, int) or isinstance(i, bool) for i in saved.values())
                or sorted(saved.values()) != list(range(len(saved)))):
            raise ValueError(f"Checkpoint {name} indices must be unique and contiguous from zero")
        if set(saved) != set(current or {}):
            raise ValueError(f"Supervised checkpoint {name} labels changed: saved={sorted(saved)}, current={sorted(current or {})}")
        ordered.append(dict(sorted(saved.items(), key=lambda item: item[1])))
    return tuple(ordered)


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
    pose_dict=None,
    orient_dict=None,
    checkpoint_data=None,
) -> tuple[int, int]:
    """Load weights and optionally restore optimizer, scheduler and scaler state.

    A pretraining-to-supervised handoff loads compatible backbone weights,
    excluding the projection head. Weights-only loading returns (0, 0); full
    resume returns (saved_epoch + 1, saved_global_step). Missing requested
    training state falls back to weights-only with a warning. Supplied
    supervised class mappings must retain checkpoint index meanings.
    """
    if not checkpoint_path:
        return 0, 0

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Requested checkpoint does not exist: {checkpoint_path}')

    # Always load on CPU first (lighter on GPU memory). We'll move optimizer state to GPU if needed.
    checkpoint = checkpoint_data if checkpoint_data is not None else torch.load(checkpoint_path, map_location='cpu')

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
        # Requested scaler state participates in the resume availability check.
        if scaler is not None and 'scaler_state_dict' not in checkpoint:
            missing.append('scaler_state_dict')
        if missing:
            print(f"[CKPT] Requested resume but checkpoint lacks {missing}. Forcing weights-only load.")
            load_optimizer_state = False

    is_handoff = bool(is_pretrain_checkpoint) and (not bool(pretrain_mode))
    if load_optimizer_state and checkpoint.get('global_vars', {}).get('schedule_mode') == 'epoch_cosine_no_restart':
        from .schedulers import EpochCosineScheduler
        if not isinstance(scheduler, EpochCosineScheduler):
            raise ValueError('Epoch cosine checkpoint requires the same epoch-cosine scheduler; '
                             'restore checkpoint configuration before requesting a full resume.')
    if not is_handoff and not pretrain_mode and load_optimizer_state and pose_dict is None and orient_dict is None:
        raise ValueError("True supervised resume requires current pose/orient class dictionaries to verify head meanings")
    if not is_handoff and not pretrain_mode and (pose_dict is not None or orient_dict is not None):
        saved_pose, saved_orient = validate_supervised_class_schema(checkpoint, pose_dict, orient_dict,
                                        pose_pred=bool(pose_dict), orient_pred=bool(orient_dict))
        if dict(pose_dict or {}) != saved_pose or dict(orient_dict or {}) != saved_orient:
            raise ValueError("Current class indices differ from the checkpoint; remap dataset instructions to saved indices before loading")

    mode_str = 'handoff' if is_handoff else ('resume' if load_optimizer_state else 'weights_only')
    print(f"[CKPT] Loading checkpoint kind={ckpt_kind} mode={mode_str} from: {checkpoint_path}")

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

    if strict:
        model.load_state_dict(state_dict)
    else:
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"[CKPT] Non-strict load. Missing keys: {len(load_result.missing_keys)} | Unexpected keys: {len(load_result.unexpected_keys)}")

    if load_optimizer_state:
        # Restore optimizer state on the model device; report scheduler/scaler failures below.
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
                   split_filenames: Optional[dict[str, object]], pretrain_mode: bool = False,
                   filename_override: Optional[str] = None, swa_runtime_state: Optional[dict] = None,
                   training_runtime_state: Optional[dict] = None,
                   pretraining_runtime_state: Optional[dict] = None) -> None:
    """Save model/training state, class mappings and run metadata.

    Store input identities and file splits when supplied, plus optional
    supervised, pretraining or SWA runtime state. ``epoch`` is zero-based; the
    default filename uses epoch + 1. ``filename_override`` selects an exact name.
    """
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



    checkpoint_schema_version = 1
    checkpoint_kind = 'pretrain' if pretrain_mode else 'supervised'
    checkpoint_created_at = datetime.now().isoformat()

    checkpoint_data = {
        'checkpoint_schema_version': checkpoint_schema_version,
        'checkpoint_kind': checkpoint_kind,
        'checkpoint_created_at': checkpoint_created_at,
        'run_name': run_name,
        'epoch': epoch,  
        'global_step': global_step,
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'scheduler_state_dict': scheduler.state_dict(),  
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,  
        'global_vars': global_vars,  
    }
    if swa_runtime_state is not None:
        checkpoint_data['swa_runtime_state'] = dict(swa_runtime_state)
    if training_runtime_state is not None:
        checkpoint_data['training_runtime_state'] = dict(training_runtime_state)
    if pretraining_runtime_state is not None:
        checkpoint_data['pretraining_runtime_state'] = dict(pretraining_runtime_state)

    if split_filenames:
        checkpoint_data['train_filenames'] = split_filenames['train_filenames']
        checkpoint_data['val_filenames'] = split_filenames['val_filenames']
        if 'training_data_identity' in split_filenames:
            checkpoint_data['training_data_identity'] = dict(split_filenames['training_data_identity'])

    if pose_pred:
        checkpoint_data.update({'pose_dict': pose_dict})
    if orient_pred:
        checkpoint_data.update({'orient_dict': orient_dict})

    atomic_save_checkpoint(checkpoint_data, checkpoint_path)
    print(f"\nCheckpoint ({checkpoint_kind}) saved to {checkpoint_path}")


def atomic_save_checkpoint(data, path):
    """Publish complete checkpoint bytes before replacing any prior checkpoint."""
    path = Path(path)
    temporary = path.with_name('.checkpoint-' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


