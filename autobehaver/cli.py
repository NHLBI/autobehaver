"""Command-line interface implementation for pose prediction training and inference.

The organized training entrypoint lives at
``python -m autobehaver.training.posepred_training``.

Defines the argument parser and main entrypoint that routes to the appropriate mode:
supervised training, pretraining, LR finder, or inference.

Note on SWA:
    SWA is implemented as an **optional phase within supervised training** (enabled via
    `--swa_mode`), rather than as a standalone post-hoc stage.
"""

from __future__ import annotations

import os
import random
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict

import yaml

from .feature_selection import PRESETS, DEFAULT_PRESET, FeaturePlan
from .pipeline_layout import config_main_dir_default


def _load_training_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load posepred training defaults from a pipeline YAML config.

    Reads the top-level ``posepred_training:`` section. Relative path values
    are resolved relative to the config file.
    """

    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a YAML mapping (dict) at the top level")

    base_dir = cfg_path.parent
    section = data.get("posepred_training", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'posepred_training' section in config to be a mapping (dict)")

    resolved: Dict[str, Any] = {}
    global_main_dir = config_main_dir_default(data, base_dir)
    if global_main_dir and "directory" not in section:
        resolved["directory"] = str((Path(global_main_dir) / "5_normalized_features").resolve())

    path_keys = {"directory", "pretrained_model_path", "ablation_include_columns_file"}
    for k, v in section.items():
        if k in path_keys and isinstance(v, str) and v.strip() != "":
            p = Path(v.strip())
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            resolved[k] = str(p)
        else:
            resolved[k] = v
    return resolved


def _write_feature_plan_artifacts(records_dir: str, run_name: str, feature_plan: FeaturePlan) -> str:
    """Write a single, human-readable feature-column report under Records/."""
    out_dir = os.path.join(records_dir, "feature_columns")
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, f"{run_name}_feature_columns.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("INCLUDED COLUMNS\n")
        f.write("================\n")
        if feature_plan.included_columns:
            for col in feature_plan.included_columns:
                f.write(f"{col}\n")
        else:
            f.write("(none)\n")

        f.write("\n\n\n")
        f.write("EXCLUDED COLUMNS\n")
        f.write("================\n")
        if feature_plan.excluded_columns:
            for col in feature_plan.excluded_columns:
                f.write(f"{col}\n")
        else:
            f.write("(none)\n")

    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Transformer-based pose prediction training and inference.')
    parser.add_argument('--config', type=str, default=None, help="Path to global pipeline YAML config (reads the 'posepred_training:' section).")
    parser.add_argument('--random_seed', type=int, default=None, help='Set random seed for reproducibility. If not provided, the seed will be random.')

    # --- Data Loader Parameters ---
    parser.add_argument('--num_workers', type=int, default=14, help='Number of workers for the DataLoader.')

    # --- Pretraining Mode Parameters ---
    # REQUIRED ARGUMENTS: --pretrain_mode
    # RECOMMENDED ARGUMENTS: --run_name --base_lr --num_epochs --pretrain_loss --projection_dim --pretrain_files_per_group --log_interval
    pretrain_group = parser.add_argument_group('Pretraining Mode')
    pretrain_group.add_argument('--pretrain_mode', action='store_true', help='Enable pretraining mode (default: False).')
    pretrain_group.add_argument('--projection_dim', type=int, default=128, help='Dimension of the output from the projection head in pretraining mode.')
    pretrain_group.add_argument('--temperature', type=float, default=0.1, help='Temperature for InfoNCE pretraining (ignored for Barlow Twins).')
    pretrain_group.add_argument('--pretrain_files_per_group', type=int, default=500, help='Number of files to group together for pretraining batches.')
    pretrain_group.add_argument('--log_interval', type=int, default=200, help='Log metrics every X steps in pretrain_mode.')
    pretrain_group.add_argument('--pretrain_loss', type=str, default='barlow_twins', choices=['info_nce', 'barlow_twins'], help='Loss function for pretraining mode.')
    pretrain_group.add_argument('--barlow_lambda', type=float, default=5e-3, help='Lambda parameter for Barlow Twins pretraining (ignored for InfoNCE).')

    # --- LR Finder Mode Parameters ---
    # REQUIRED ARGUMENTS: --lr_finder_mode
    # RECOMMENDED ARGUMENTS: --pretrained_model_path --lrfinder_lr
    lr_finder_group = parser.add_argument_group('LR Finder Mode')
    lr_finder_group.add_argument('--lr_finder_mode', action='store_true', help='Enable LR finder mode (default: False).')
    lr_finder_group.add_argument('--lrfinder_lr', type=float, default=1e-8, help='Initial learning rate for the LR finder.')

    # --- Inference Mode Parameters ---
    # REQUIRED ARGUMENTS: --inference_mode --inference_model
    # RECOMMENDED ARGUMENTS: --inf_average --use_tta
    inference_group = parser.add_argument_group('Inference Mode')
    inference_group.add_argument('--inference_mode', action='store_true', help='Enable inference mode (default: False).')
    inference_group.add_argument('--inference_model', type=str, nargs='+', help='Path(s) to model(s) for inference. Provide multiple paths for ensembling.')
    inference_group.add_argument('--inf_average', type=int, default=10, help='Number of times to repeat inference for averaging.')
    inference_group.add_argument('--use_tta', action='store_true', help='Enable Test-Time Augmentation for inference (default: False).')
    inference_group.add_argument('--tta_noise_level', type=float, default=0.01, help='Standard deviation of noise for TTA.')
    inference_group.add_argument('--min_pose_bout', type=float, default=5, help='Minimum consecutive frames for Pose smoothing.')
    inference_group.add_argument('--min_orient_bout', type=float, default=5, help='Minimum consecutive frames for Orient smoothing.')


    # --- SWA Training Option ---
    # SWA runs as the *final* phase within a supervised training run:
    #   - run normal training for the first (num_epochs - swa_epochs) epochs
    #   - then run SWA for the final `swa_epochs` epochs and save an additional SWA checkpoint
    swa_group = parser.add_argument_group('SWA (optional)')
    swa_group.add_argument('--swa_mode', action='store_true', help='Enable Stochastic Weight Averaging (SWA) during the final supervised-training epochs (default: False).')
    swa_group.add_argument('--swa_epochs', type=int, default=75, help='Number of final epochs to dedicate to the SWA phase. Must be > 0 and < num_epochs when --swa_mode is set.')
    swa_group.add_argument('--swa_lr', type=float, default=0.00001, help='Target learning rate for the SWA phase (used by torch.optim.swa_utils.SWALR).')

    # --- FINE-TUNING STEP 1: LINEAR PROBING ---
    # REQUIRED ARGUMENTS: --pretrained_model_path --freeze_encoder
    # RECOMMENDED ARGUMENTS: --run_name --base_lr --patience

    # --- FINE-TUNING STEP 2: FULL JOINT TRAINING ---
    # REQUIRED ARGUMENTS: --pretrained_model_path --restart_optimizer
    # RECOMMENDED ARGUMENTS: --run_name --base_lr --patience --weight_encoder (optional)

    # --- Model Parameters ---
    model_group = parser.add_argument_group('Model Parameters')
    # Input Parameters
    model_group.add_argument('--sequence_length', type=int, default=41, help='Length of the input sequence.')
    # Model Architecture
    model_group.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    model_group.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of the feedforward network.')
    model_group.add_argument('--nhead', type=int, default=32, help='Number of attention heads.')
    model_group.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers.')
    model_group.add_argument('--num_decoder_layers', type=int, default=1, help='Number of decoder layers.')
    model_group.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
    # Context Vector
    model_group.add_argument('--frame_representation', type=str, default='cls', choices=['middle', 'cls', 'cls+middle'], help='Method for generating the context vector.')
    model_group.add_argument('--num_cls', type=int, default=1, help='Number of CLS tokens.')
    # Prediction Head
    model_group.add_argument('--prediction_head', type=str, default='fc', choices=['fc', 'mlp'], help='Type of prediction head.')
    # Positional Encoding
    model_group.add_argument('--positional_encoding_type', type=str, default='sinusoidal', choices=['learned', 'sinusoidal'], help='Type of positional encoding.')

    # --- Training Hyperparameters ---
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument('--batch', type=int, default=1024, help='Batch size.')
    training_group.add_argument('--num_epochs', type=int, default=6061, help='Number of training epochs.')
    training_group.add_argument('--checkpoint_interval', type=int, default=601, help='Checkpoint save interval. In supervised training this is epochs; in pretraining it is global steps.')
    training_group.add_argument('--patience', type=int, default=601, help='Patience for early stopping.')
    # Scheduler
    training_group.add_argument('--scheduler_type', type=str, default='cosine', choices=['sqrt', 'cosine', 'linear', 'plateau'], help='Type of learning rate scheduler.')
    training_group.add_argument('--base_lr', type=float, default=0.00005, help='Base learning rate.')
    training_group.add_argument('--weight_decay', type=float, default=0.001, help='AdamW weight decay (L2 regularization).')
    training_group.add_argument('--t_0', type=int, default=1476600, help='Number of batches for the first cycle of cosine/linear annealing.')
    training_group.add_argument('--t_mult', type=int, default=1, help='Multiplier for subsequent cycles in cosine/linear annealing.')
    training_group.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine/linear annealing.')
    training_group.add_argument('--plateau_factor', type=float, default=0.5, help='Factor to reduce LR on plateau.')
    training_group.add_argument('--warmup_steps', type=int, default=2461, help='Number of warmup steps for the scheduler.')
    # Data Augmentation
    training_group.add_argument('--mask_p', type=float, default=0.05, help='Probability of masking a frame.')
    training_group.add_argument('--noise_level', type=float, default=0.05, help='Standard deviation of noise for augmentation.')
    # Prediction Heads & Weights
    training_group.add_argument('--no-pose-pred', dest='pose_pred', action='store_false', help='Disable pose prediction head.')
    training_group.add_argument('--no-orient-pred', dest='orient_pred', action='store_false', help='Disable orientation prediction head.')
    training_group.add_argument('--recon', action='store_true', help='Enable reconstruction head.')
    training_group.add_argument('--pose_weight', type=float, default=1.0, help='Weight for the pose loss.')
    training_group.add_argument('--orient_weight', type=float, default=0.1, help='Weight for the orientation loss.')
    training_group.add_argument('--recon_weight', type=float, default=1.0, help='Weight for the reconstruction loss.')
    training_group.add_argument('--max_class_weight', type=float, default=200, help='Maximum weight for a class.')
    training_group.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping (applied after AMP unscale). Set <=0 to disable.')
    # Data Split
    training_group.add_argument('--no-defined-split', dest='defined_split', action='store_false', help='Use random split instead of the one defined in metadata.')
    # Optimizer State
    training_group.add_argument('--restart_optimizer', action='store_true', help='Load checkpoint weights but restart optimizer/scheduler/scaler (weights-only fine-tune stage).')
    pretrain_group.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder for fine-tuning (default: unfrozen).')
    pretrain_group.add_argument('--weight_encoder', type=float, default=1, help='Weight for the encoder loss during fine-tuning.')

    # --- Logging and Directories ---
    dir_group = parser.add_argument_group('Logging and Directories')
    dir_group.add_argument('--wandb_log', action='store_true', help='Enable logging with Weights & Biases.')
    dir_group.add_argument('--run_name', type=str, default="autobehaver_run", help='Name of the training run.')
    dir_group.add_argument('--directory', type=str, help='Main directory for data, checkpoints, and records.')
    dir_group.add_argument('--pretrained_model_path', type=str, help='Path to a pretrained model checkpoint.')
    dir_group.add_argument('--metadata_tag', type=str, default='fullcatvalsplitmetadata', help='Tag to identify the metadata CSV file.')

    training_group.add_argument('--noval', action='store_true', help='Do not use validation - train on the full dataset.')

    # --- Feature selection / ablation ---
    ablate_group = parser.add_argument_group('Feature Selection / Ablation')
    ablate_group.add_argument(
        '--ablation_preset',
        type=str,
        default=DEFAULT_PRESET,
        choices=sorted(PRESETS.keys()),
        help=(
            "Feature ablation preset to apply before training (default: no_pos). "
            "Use 'none' to disable ablation."
        ),
    )
    ablate_group.add_argument(
        '--ablation_exclude_columns',
        type=str,
        action='append',
        default=None,
        help=(
            "Explicit column names to exclude (comma-separated). "
            "May be provided multiple times; values are concatenated."
        ),
    )
    ablate_group.add_argument(
        '--ablation_exclude_keywords',
        type=str,
        action='append',
        default=None,
        help=(
            "Keyword substrings; any column containing one will be excluded (comma-separated). "
            "May be provided multiple times; values are concatenated."
        ),
    )
    ablate_group.add_argument(
        '--ablation_include_columns_file',
        type=str,
        default=None,
        help=(
            "Path to a newline-delimited included-columns file. If provided, "
            "this exact ordered list defines the model input schema."
        ),
    )
    ablate_group.add_argument(
        '--ablation_case_sensitive',
        action='store_true',
        help='Use case-sensitive matching for exclude columns/keywords (default: False).',
    )
    ablate_group.add_argument(
        '--ablation_non_strict',
        action='store_true',
        help=(
            "Non-strict mode: skip files missing required columns instead of erroring. "
            "(Default is strict.)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Autobehaver pose prediction CLI entrypoint."""
    # Build and parse args. ``--config`` provides parser defaults; explicit CLI
    # arguments still override the YAML.
    argv_list = list(argv) if argv is not None else None
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv_list)

    parser = build_parser()
    if pre_args.config:
        defaults = _load_training_config_defaults(pre_args.config)
        valid_dests = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_dests})

    args = parser.parse_args(argv_list)

    # Heavy training/runtime imports intentionally happen after argparse so
    # ``--help`` remains fast and does not initialize Torch/sklearn stacks.
    import numpy as np
    import pandas as pd
    import torch

    # --- Torch perf defaults ---
    torch.backends.cudnn.benchmark = True

    # --- Seed handling ---
    if args.random_seed is not None:
        print(f"Using random seed: {args.random_seed}")
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("No random seed provided. Running with random seed.")

    # --- Force odd sequence length (core invariant) ---
    if args.sequence_length % 2 == 0:
        provided_sequence_length = args.sequence_length
        args.sequence_length = int(args.sequence_length) + 1
        print(f"[SEQ_LEN] Even sequence_length={provided_sequence_length} provided; bumping to {args.sequence_length} (must be odd).")

    # --- Modes ---
    pretrain_mode = args.pretrain_mode
    lr_finder_mode = args.lr_finder_mode
    inference_mode = args.inference_mode
    swa_mode = args.swa_mode

    # --- Paths / metadata ---
    run_name = args.run_name
    directory = args.directory
    pretrained_model_path = args.pretrained_model_path
    defined_split = args.defined_split
    noval = args.noval

    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # --- Inference mode ---
    # Inference does NOT require a metadata CSV in the target directory. We simply
    # scan for HDF5 files and write outputs under <directory>/inferences/.
    if inference_mode:
        if not args.inference_model:
            parser.error("Inference Mode requires --inference_model to be specified.")
        from .inference import run_inference

        run_inference(
            directory=directory,
            inference_model_paths=args.inference_model,
            num_workers=args.num_workers,
            batch=args.batch,
            inf_average=args.inf_average,
            sequence_length=args.sequence_length,
            min_pose_bout=args.min_pose_bout,
            min_orient_bout=args.min_orient_bout,
            use_tta=args.use_tta,
            tta_noise_level=args.tta_noise_level,
        )
        return


    metadata_tag = args.metadata_tag
    if metadata_tag:
        metadata_files = [f for f in os.listdir(directory) if f.endswith(f"{metadata_tag}.csv")]
        if len(metadata_files) == 0:
            raise FileNotFoundError(f"No metadata file ending with '{metadata_tag}.csv' found in the directory.")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files ending with '{metadata_tag}.csv' found in the directory.")
        metadata_path = os.path.join(directory, metadata_files[0])
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = None

    # Set up directories inside input folder
    checkpoint_dir = os.path.join(directory, "Checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    records_dir = os.path.join(directory, "Records")
    os.makedirs(records_dir, exist_ok=True)

    records_path = os.path.join(records_dir, f"{run_name}_training_record.csv")
    pose_conf_matrix_path = os.path.join(records_dir, f"{run_name}_pose_confusion_matrices.csv") if args.pose_pred else None
    orient_conf_matrix_path = os.path.join(records_dir, f"{run_name}_orient_confusion_matrices.csv") if args.orient_pred else None
    combined_conf_matrix_path = os.path.join(records_dir, f"{run_name}_combined_confusion_matrices.csv") if (args.pose_pred and args.orient_pred) else None

    # Optional wandb init
    wandb_log = args.wandb_log
    if wandb_log:
        import wandb
        wandb.init(
            project="Autobehaver",
            name=run_name,
            config=vars(args),
            mode="offline",
        )

    # === WARNING FILTERING ===
    warnings.filterwarnings('ignore', message='enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True')
    warnings.filterwarnings('ignore', category=FutureWarning, message=r'You are using `torch.load` with `weights_only=False.*')

    # --- Dispatch by mode ---
    if pretrain_mode:
        print(f'Starting pretraining for run "{run_name}".\n')

        from .checkpoint import load_checkpoint_variables
        from .feature_selection import resolve_feature_plan
        from .io import get_headers_from_first_hdf5
        from .pretraining import run_pretraining

        # Resolve feature schema for pretraining as well (keeps pretrain→supervised handoffs consistent)
        raw_headers = get_headers_from_first_hdf5(directory)
        feature_plan = resolve_feature_plan(
            raw_headers,
            preset=args.ablation_preset,
            exclude_columns=args.ablation_exclude_columns,
            exclude_keywords=args.ablation_exclude_keywords,
            include_columns_file=args.ablation_include_columns_file,
            case_sensitive=bool(args.ablation_case_sensitive),
        )
        ablation_strict = not bool(args.ablation_non_strict)

        try:
            feature_artifacts_dir = _write_feature_plan_artifacts(records_dir, run_name, feature_plan)
            print(f"[FEATURES] Wrote feature column report to: {feature_artifacts_dir}")
        except Exception as e:
            print(f"[FEATURES] Warning: could not write feature column report: {e}")

        load_checkpoint_variables(
            args=args,
            checkpoint_path=pretrained_model_path,
            use_restart_optimizer=bool(args.restart_optimizer),
            current_num_features=len(feature_plan.included_columns),
            current_feature_schema_hash=feature_plan.schema_hash,
            current_feature_columns_included=feature_plan.included_columns,
        )

        run_pretraining(
            directory, metadata, args.sequence_length, args.batch, args.num_workers, args.num_epochs,
            args.checkpoint_interval, args.log_interval, checkpoint_dir, run_name,
            args.projection_dim, args.temperature, args.pretrain_files_per_group,
            args.mask_p, args.noise_level, records_path,
            args.d_model, args.nhead, args.num_encoder_layers, args.num_decoder_layers,
            args.dim_feedforward, args.dropout_rate,
            args.frame_representation, args.num_cls, args.prediction_head, args.positional_encoding_type,
            args.base_lr, args.warmup_steps,
            args.patience, args.plateau_factor,
            args.pretrain_loss, args.barlow_lambda,
            scheduler_type=args.scheduler_type,
            weight_decay=args.weight_decay,
            t_0=args.t_0,
            t_mult=args.t_mult,
            eta_min=args.eta_min,
            grad_clip_norm=args.grad_clip_norm,
            random_seed=args.random_seed,
            metadata_tag=args.metadata_tag,
            feature_plan=feature_plan,
            ablation_strict=ablation_strict,
            checkpoint_path=pretrained_model_path,
            restart_optimizer=bool(args.restart_optimizer),
        )
        return

    if lr_finder_mode:
        # LR finder reuses the supervised training path but runs a short
        # learning-rate sweep, saving outputs to Records/.
        from .lr_finder import find_lr

        find_lr(args=args, directory=directory, metadata=metadata)
        return

    # --- Default: supervised training ---
    print(f'Starting posepred training for run "{run_name}".\n')

    from .checkpoint import load_checkpoint_variables
    from .feature_selection import resolve_feature_plan
    from .io import get_headers_from_first_hdf5
    from .training import main_worker, prepare_data

    # --- Resolve feature schema (ablation) ---
    raw_headers = get_headers_from_first_hdf5(directory)
    feature_plan = resolve_feature_plan(
        raw_headers,
        preset=args.ablation_preset,
        exclude_columns=args.ablation_exclude_columns,
        exclude_keywords=args.ablation_exclude_keywords,
        include_columns_file=args.ablation_include_columns_file,
        case_sensitive=bool(args.ablation_case_sensitive),
    )
    num_features = len(feature_plan.included_columns)
    print(
        f"[FEATURES] Raw={len(raw_headers)} Included={len(feature_plan.included_columns)} "
        f"Excluded={len(feature_plan.excluded_columns)} Preset={feature_plan.spec.get('preset')}"
    )
    print(f"[FEATURES] Schema hash: {feature_plan.schema_hash}")

    try:
        feature_artifacts_dir = _write_feature_plan_artifacts(records_dir, run_name, feature_plan)
        print(f"[FEATURES] Wrote feature column report to: {feature_artifacts_dir}")
    except Exception as e:
        print(f"[FEATURES] Warning: could not write feature column report: {e}")

    # If resuming from checkpoint, optionally override args from ckpt.
    load_checkpoint_variables(
        args=args,
        checkpoint_path=pretrained_model_path,
        use_restart_optimizer=bool(args.restart_optimizer),
        current_num_features=num_features,
        current_feature_schema_hash=feature_plan.schema_hash,
        current_feature_columns_included=feature_plan.included_columns,
    )

    # --- Final SWA arg validation (after any full-resume checkpoint overrides) ---
    swa_mode = bool(getattr(args, 'swa_mode', False))
    if swa_mode:
        if args.swa_epochs is None or int(args.swa_epochs) <= 0:
            parser.error("--swa_epochs must be > 0 when --swa_mode is set")
        if int(args.swa_epochs) >= int(args.num_epochs):
            parser.error("--swa_epochs must be smaller than --num_epochs when --swa_mode is set")

    ablation_strict = not bool(args.ablation_non_strict)

    train_dataset, val_dataset, pose_dict, orient_dict, columns_to_noise, train_class_weights, val_class_weights, split_filenames = prepare_data(
        directory=directory, metadata=metadata, sequence_length=args.sequence_length,
        num_workers=args.num_workers, random_seed=args.random_seed,
        pretrained_model_path=pretrained_model_path,
        pose_pred=args.pose_pred, orient_pred=args.orient_pred, recon=args.recon,
        max_class_weight=args.max_class_weight, defined_split=defined_split, noval=noval,
        feature_plan=feature_plan,
        ablation_strict=ablation_strict,
    )

    print("pose_dict:", pose_dict)
    print("orient_dict:", orient_dict)

    main_worker(
        train_dataset, val_dataset, pose_dict, orient_dict, args.num_workers, args.sequence_length,
        args.batch, pretrained_model_path, split_filenames, num_features, args.d_model, args.nhead,
        args.num_encoder_layers, args.num_decoder_layers, args.dim_feedforward, args.dropout_rate,
        args.frame_representation, args.num_cls, args.prediction_head,
        records_path, wandb_log, args.num_epochs, args.checkpoint_interval, checkpoint_dir, run_name,
        columns_to_noise, pose_conf_matrix_path, orient_conf_matrix_path, combined_conf_matrix_path,
        args.pose_pred, args.orient_pred, args.recon, args.positional_encoding_type,
        args.pose_weight, args.orient_weight, args.recon_weight,
        train_class_weights, val_class_weights,
        args.patience, args.plateau_factor, args.freeze_encoder, args.weight_encoder,
        bool(args.restart_optimizer), noval,
        args.base_lr, args.weight_decay, args.scheduler_type, args.warmup_steps, args.t_0,
        args.t_mult, args.eta_min, args.mask_p, args.noise_level, args.max_class_weight,
        args.grad_clip_norm, args.random_seed, args.metadata_tag,
        swa_mode=bool(swa_mode),
        swa_epochs=args.swa_epochs,
        swa_lr=args.swa_lr,
        feature_plan=feature_plan,
    )


if __name__ == '__main__':
    main()
