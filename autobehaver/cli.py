"""Parse and dispatch PosePred supervised training, pretraining, inference and LR sweeps.

The preferred entrypoint is ``python -m autobehaver.training.posepred_training``.
Optional stochastic weight averaging is a final supervised-training phase.
"""

from __future__ import annotations

import os
import random
import argparse
import warnings
from pathlib import Path

from .feature_selection import PRESETS, DEFAULT_PRESET, FeaturePlan
from .argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings


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
    parser.add_argument('--random_seed', type=int, default=7, help='Set the reproducibility seed (default: 7).')

    # Data loaders
    parser.add_argument('--num_workers', type=int, default=14, help='Number of workers for the DataLoader.')

    # Pretraining
    pretrain_group = parser.add_argument_group('Pretraining Mode')
    pretrain_group.add_argument('--pretrain_mode', action='store_true', help='Train the encoder and projection head with a self-supervised loss.')
    pretrain_group.add_argument('--projection_dim', type=int, default=128, help='Dimension of the output from the projection head in pretraining mode.')
    pretrain_group.add_argument('--temperature', type=float, default=0.1, help='Temperature for InfoNCE pretraining (ignored for Barlow Twins).')
    pretrain_group.add_argument('--pretrain_files_per_group', type=int, default=500, help='Maximum files loaded together in a pretraining chunk.')
    pretrain_group.add_argument('--log_interval', type=int, default=200, help='Record pretraining metrics after this many optimizer steps.')
    pretrain_group.add_argument('--pretrain_loss', type=str, default='barlow_twins', choices=['info_nce', 'barlow_twins'], help='Loss function for pretraining mode.')
    pretrain_group.add_argument('--barlow_lambda', type=float, default=5e-3, help='Lambda parameter for Barlow Twins pretraining (ignored for InfoNCE).')

    # Learning-rate sweep
    lr_finder_group = parser.add_argument_group('LR Finder Mode')
    lr_finder_group.add_argument('--lr_finder_mode', action='store_true', help='Run a short learning-rate sweep and save its loss records.')
    lr_finder_group.add_argument('--lrfinder_lr', type=float, default=1e-8, help='Initial learning rate for the LR finder.')

    # Inference
    inference_group = parser.add_argument_group('Inference Mode')
    inference_group.add_argument('--inference_mode', action='store_true', help='Predict labels from saved supervised model checkpoints.')
    inference_group.add_argument('--inference_model', type=str, nargs='+', help='Path(s) to model(s) for inference. Provide multiple paths for ensembling.')
    inference_group.add_argument('--inf_average', type=int, default=10, help='Passes to average per model; without augmentation, reuse one forward pass.')
    add_bool_pair(inference_group, 'use_tta', default=True,
                  help='Use noise-augmented passes for inference (default: enabled).')
    inference_group.add_argument('--tta_noise_level', type=float, default=0.01, help='Gaussian-noise standard deviation for test-time augmentation.')
    inference_group.add_argument('--tta_seed', type=int, default=42, help='Reproducible per-recording TTA seed (default: 42).')
    inference_group.add_argument('--min_pose_bout', type=float, default=5, help='Pose smoothing bout threshold in frames; <=1 disables smoothing.')
    inference_group.add_argument('--min_orient_bout', type=float, default=5, help='Orientation smoothing bout threshold in frames; <=1 disables smoothing.')


    # Stochastic weight averaging
    # Average weights during the final supervised-training epochs.
    swa_group = parser.add_argument_group('SWA (optional)')
    swa_group.add_argument('--swa_mode', action='store_true', help='Enable Stochastic Weight Averaging (SWA) during the final supervised-training epochs (default: False).')
    swa_group.add_argument('--swa_epochs', type=int, default=75, help='Number of final epochs to dedicate to the SWA phase. Must be > 0 and < num_epochs when --swa_mode is set.')
    swa_group.add_argument('--swa_lr', type=float, default=0.00001, help='Target learning rate for the SWA phase (used by torch.optim.swa_utils.SWALR).')



    # Model architecture
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--sequence_length', type=int, default=41, help='Input window length in frames; even values advance to the next odd length.')
    model_group.add_argument('--d_model', type=int, default=1024, help='Dimension of the model.')
    model_group.add_argument('--dim_feedforward', type=int, default=4096, help='Dimension of the feedforward network.')
    model_group.add_argument('--nhead', type=int, default=32, help='Number of attention heads.')
    model_group.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers.')
    model_group.add_argument('--num_decoder_layers', type=int, default=1, help='Transformer decoder layers used when reconstruction is enabled.')
    model_group.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
    model_group.add_argument('--frame_representation', type=str, default='middle', choices=['middle', 'cls', 'cls+middle'], help='Method for generating the context vector.')
    model_group.add_argument('--num_cls', type=int, default=1, help='Number of CLS tokens.')
    model_group.add_argument('--prediction_head', type=str, default='mlp', choices=['fc', 'mlp'], help='Type of prediction head.')
    model_group.add_argument('--positional_encoding_type', type=str, default='sinusoidal', choices=['learned', 'sinusoidal'], help='Type of positional encoding.')

    # Training
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument('--batch', type=int, default=1024, help='Batch size.')
    training_group.add_argument('--num_epochs', type=int, default=301, help='Number of training epochs.')
    training_group.add_argument('--checkpoint_interval', type=int, default=301, help='Checkpoint save interval. In supervised training this is epochs; in pretraining it is global steps.')
    training_group.add_argument('--patience', type=int, default=601, help='Supervised-training epochs without improvement before early stopping.')
    # Scheduler
    training_group.add_argument('--scheduler_type', type=str, default='cosine', choices=['sqrt', 'cosine', 'linear', 'plateau'], help='Type of learning rate scheduler.')
    training_group.add_argument('--base_lr', type=float, default=0.0001, help='Base learning rate.')
    training_group.add_argument('--weight_decay', type=float, default=0.001, help='AdamW decoupled weight decay.')
    training_group.add_argument('--t_0', type=int, default=1476600, help='Number of batches for the first cycle of cosine/linear annealing.')
    training_group.add_argument('--t_mult', type=int, default=1, help='Multiplier for subsequent cycles in cosine/linear annealing.')
    training_group.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine/linear annealing.')
    training_group.add_argument('--plateau_factor', type=float, default=0.5, help='Factor to reduce LR on plateau.')
    training_group.add_argument('--warmup_steps', type=int, default=2461, help='Number of warmup steps for the scheduler.')
    training_group.add_argument('--warmup_epochs', type=int, default=None, help='With --cosine_decay_epochs, loader-sized warmup (default: one epoch).')
    training_group.add_argument('--cosine_decay_epochs', type=int, default=300, help='Cosine decay epochs after warmup (default: 300); total must equal num_epochs. Set 0 for step-based scheduling. SWA, pretraining and LR sweeps use their own step-based schedules.')
    # Data Augmentation
    training_group.add_argument('--mask_p', type=float, default=0.05, help='Probability of masking each feature element after adding noise.')
    training_group.add_argument('--noise_level', type=float, default=0.05, help='Standard deviation of noise for augmentation.')
    # Prediction Heads & Weights
    training_group.add_argument('--no-pose-pred', dest='pose_pred', action='store_false', help='Disable pose prediction head.')
    training_group.add_argument('--no-orient-pred', dest='orient_pred', action='store_false', help='Disable orientation prediction head.')
    training_group.add_argument('--recon', action='store_true', help='Enable reconstruction head.')
    training_group.add_argument('--pose_weight', type=float, default=1.0, help='Weight for the pose loss.')
    training_group.add_argument('--orient_weight', type=float, default=0.1, help='Weight for the orientation loss.')
    training_group.add_argument('--recon_weight', type=float, default=1.0, help='Weight for the reconstruction loss.')
    training_group.add_argument('--max_class_weight', type=float, default=200, help='Cap inverse-frequency weights estimated from joint training-window labels.')
    training_group.add_argument('--grad_clip_norm', type=float, default=1.0, help='Max norm for gradient clipping (applied after AMP unscale). Set <=0 to disable.')
    # Data Split
    training_group.add_argument('--no-defined-split', dest='defined_split', action='store_false', help='Use a movie-grouped random split instead of metadata-defined train/val sets.')
    # Optimizer State
    training_group.add_argument('--restart_optimizer', action='store_true', help='Load checkpoint weights but restart optimizer/scheduler/scaler (weights-only fine-tune stage).')
    pretrain_group.add_argument('--freeze_encoder', action='store_true', help='Freeze the backbone during a pretraining-to-supervised handoff.')
    pretrain_group.add_argument('--weight_encoder', type=float, default=1, help='Backbone learning-rate multiplier during a pretraining-to-supervised handoff.')

    # Output records and input paths
    dir_group = parser.add_argument_group('Logging and Directories')
    dir_group.add_argument('--wandb_log', action='store_true', help='Enable optional offline Weights & Biases training records.')
    dir_group.add_argument('--run_name', type=str, default="autobehaver_run", help='Name of the training run.')
    dir_group.add_argument('--directory', type=str, help='Main directory for data, checkpoints, and records.')
    dir_group.add_argument('--output_directory', type=str, help='Optional separate folder for checkpoints and records; defaults to --directory.')
    dir_group.add_argument('--metadata_path', type=str, help='Optional explicit metadata CSV; feature and label paths remain relative to --directory.')
    dir_group.add_argument('--pretrained_model_path', type=str, help='Path to a pretrained model checkpoint.')
    dir_group.add_argument('--metadata_tag', type=str, default='fullcatvalsplitmetadata', help='Find the metadata CSV whose filename ends with <metadata_tag>.csv.')

    training_group.add_argument('--noval', action='store_true', help='Do not use validation - train on the full dataset.')

    # Ordered feature selection
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
            "Allow training loaders to skip files missing selected columns; "
            "input validation still applies. Inference remains strict."
        ),
    )
    from .inference_options import add_inference_runtime_arguments
    return add_monitor_arguments(add_inference_runtime_arguments(parser))


def resolve_schedule_mode(args):
    """Keep epoch scheduling specific to ordinary supervised training."""
    if args.cosine_decay_epochs == 0 or args.swa_mode or args.pretrain_mode or args.lr_finder_mode:
        args.cosine_decay_epochs = None
        args.warmup_epochs = None
    return args


@monitored_cli("posepred_training")
def main(argv: list[str] | None = None) -> None:
    """Parse the requested mode, validate its arguments and dispatch processing."""
    parser = build_parser()
    args = parse_cli_args(parser, argv)
    resolve_schedule_mode(args)
    record_settings(args)

    # Import training dependencies after parsing so --help remains lightweight.
    import numpy as np
    import pandas as pd
    import torch

    torch.backends.cudnn.benchmark = True

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

    # Center-frame supervision requires an odd sequence length.
    if args.sequence_length % 2 == 0:
        provided_sequence_length = args.sequence_length
        args.sequence_length = int(args.sequence_length) + 1
        print(f"[SEQ_LEN] Even sequence_length={provided_sequence_length} provided; bumping to {args.sequence_length} (must be odd).")

    record_settings(args)
    record_settings({"mode": "inference" if args.inference_mode else
                     "pretraining" if args.pretrain_mode else
                     "lr_finder" if args.lr_finder_mode else "supervised_training"})
    pretrain_mode = args.pretrain_mode
    lr_finder_mode = args.lr_finder_mode
    inference_mode = args.inference_mode
    swa_mode = args.swa_mode

    run_name = args.run_name
    directory = args.directory
    pretrained_model_path = args.pretrained_model_path
    if pretrained_model_path and not os.path.isfile(pretrained_model_path):
        raise FileNotFoundError(f'Requested checkpoint does not exist: {pretrained_model_path}')
    defined_split = args.defined_split
    noval = args.noval

    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # Inference discovers HDF5 inputs without requiring training metadata.
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
            tta_seed=args.tta_seed,
            memory_gb=args.memory_gb, noise_workers=args.noise_workers, vram_fraction=args.vram_fraction,
        )
        return


    metadata_tag = args.metadata_tag
    if args.metadata_path:
        metadata = pd.read_csv(args.metadata_path)
    elif metadata_tag:
        metadata_files = [f for f in os.listdir(directory) if f.endswith(f"{metadata_tag}.csv")]
        if len(metadata_files) == 0:
            raise FileNotFoundError(f"No metadata file ending with '{metadata_tag}.csv' found in the directory.")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files ending with '{metadata_tag}.csv' found in the directory.")
        metadata_path = os.path.join(directory, metadata_files[0])
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = None

    output_directory = args.output_directory or directory
    checkpoint_dir = os.path.join(output_directory, "Checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    records_dir = os.path.join(output_directory, "Records")
    os.makedirs(records_dir, exist_ok=True)

    records_path = os.path.join(records_dir, f"{run_name}_training_record.csv")
    pose_conf_matrix_path = os.path.join(records_dir, f"{run_name}_pose_confusion_matrices.csv") if args.pose_pred else None
    orient_conf_matrix_path = os.path.join(records_dir, f"{run_name}_orient_confusion_matrices.csv") if args.orient_pred else None
    combined_conf_matrix_path = os.path.join(records_dir, f"{run_name}_combined_confusion_matrices.csv") if (args.pose_pred and args.orient_pred) else None

    wandb_log = args.wandb_log
    if wandb_log:
        import wandb
        wandb.init(
            project="Autobehaver",
            name=run_name,
            config=vars(args),
            mode="offline",
        )

    warnings.filterwarnings('ignore', message='enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True')
    warnings.filterwarnings('ignore', category=FutureWarning, message=r'You are using `torch.load` with `weights_only=False.*')

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

    print(f'Starting posepred training for run "{run_name}".\n')

    from .checkpoint import load_checkpoint_variables
    from .feature_selection import resolve_feature_plan
    from .io import get_headers_from_first_hdf5
    from .training import main_worker, prepare_data

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

    load_checkpoint_variables(
        args=args,
        checkpoint_path=pretrained_model_path,
        use_restart_optimizer=bool(args.restart_optimizer),
        current_num_features=num_features,
        current_feature_schema_hash=feature_plan.schema_hash,
        current_feature_columns_included=feature_plan.included_columns,
    )

    # --- Final SWA arg validation (after any full-resume checkpoint overrides) ---
    record_settings(args)
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
        restart_optimizer=bool(args.restart_optimizer),
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
        warmup_epochs=args.warmup_epochs,
        cosine_decay_epochs=args.cosine_decay_epochs,
    )


if __name__ == '__main__':
    main()
