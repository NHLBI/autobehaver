"""Learning-rate range test (LR Finder).

This module provides a lightweight LR range test that does *not* depend on
external libraries (e.g. torch-lr-finder). It runs a short training loop while
increasing the learning rate exponentially and records the loss.

Design goals:
  - Uses the *same* model + loss composition as normal training (pose/orient
    heads and optional reconstruction), including augmentation (noise+mask).
  - Works on headless HPC nodes (saves CSV, optionally saves a PNG if
    matplotlib is available).
  - Avoids adding new runtime dependencies.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F

from .augmentation import augment_features
from .checkpoint import load_checkpoint_model, is_pretrain_checkpoint_path
from .datasets import build_dataloader
from .io import get_headers_from_first_hdf5
from .losses import masked_mse_loss
from ._training_core import model_setup, prepare_data
from .feature_selection import resolve_feature_plan


@dataclass(frozen=True)
class LRFinderResult:
    """Outputs produced by an LR range test."""

    csv_path: str
    plot_path: Optional[str]
    best_lr: Optional[float]


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_save_plot(df: pd.DataFrame, out_path: str) -> Optional[str]:
    """Save a loss-vs-lr plot if matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df["lr"].values, df["loss"].values)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("loss")
        ax.set_title("LR range test")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    except Exception:
        return None


def find_lr(*, args, directory: str, metadata: pd.DataFrame) -> LRFinderResult:
    """Run an LR range test.

    Parameters
    ----------
    args
        Parsed CLI args Namespace from :mod:`autobehaver.cli`.
    directory
        Dataset directory containing normalized HDF5 files.
    metadata
        Metadata dataframe (same one used for training).

    Returns
    -------
    LRFinderResult
        Paths to saved outputs.
    """

    if metadata is None or len(metadata) == 0:
        raise ValueError(
            "LR finder requires a metadata CSV (same as supervised training). "
            "No usable metadata rows were provided."
        )

    # Resolve feature schema (match supervised training defaults)
    raw_headers = get_headers_from_first_hdf5(directory)
    feature_plan = resolve_feature_plan(
        raw_headers,
        preset=getattr(args, 'ablation_preset', None),
        exclude_columns=getattr(args, 'ablation_exclude_columns', None),
        exclude_keywords=getattr(args, 'ablation_exclude_keywords', None),
        include_columns_file=getattr(args, 'ablation_include_columns_file', None),
        case_sensitive=bool(getattr(args, 'ablation_case_sensitive', False)),
    )
    num_features = len(feature_plan.included_columns)
    print(f"[LRFinder] Features: raw={len(raw_headers)} included={num_features} hash={feature_plan.schema_hash}")

    (
        train_dataset,
        _val_dataset,
        pose_dict,
        orient_dict,
        columns_to_noise,
        train_class_weights,
        _val_class_weights,
        _split_filenames,
    ) = prepare_data(
        directory=directory,
        metadata=metadata,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        pretrained_model_path=getattr(args, "pretrained_model_path", None),
        pose_pred=args.pose_pred,
        orient_pred=args.orient_pred,
        recon=args.recon,
        max_class_weight=args.max_class_weight,
        defined_split=bool(getattr(args, "defined_split", False)),
        noval=bool(getattr(args, "noval", False)),
        feature_plan=feature_plan,
        ablation_strict=(not bool(getattr(args, 'ablation_non_strict', False))),
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.random_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LRFinder] Using device: {device}")

    start_lr = float(args.lrfinder_lr)
    if start_lr <= 0:
        raise ValueError(f"lrfinder_lr must be > 0, got {start_lr}")

    base_lr_guess = float(getattr(args, "base_lr", 1e-3))
    end_lr = min(max(base_lr_guess * 100.0, base_lr_guess), 1.0)
    print(f"[LRFinder] LR sweep: start_lr={start_lr:g} -> end_lr={end_lr:g}")

    # Build model using the same explicit optimizer/scheduler hyperparameters as the
    # supervised training path, without relying on any CLI-mutated module globals.

    pretrained_model_path = getattr(args, "pretrained_model_path", None)
    is_handoff = False
    if pretrained_model_path and os.path.isfile(pretrained_model_path) and not getattr(args, "pretrain_mode", False):
        try:
            is_handoff = is_pretrain_checkpoint_path(pretrained_model_path)
        except Exception:
            is_handoff = False

    model, optimizer, _scheduler, _scaler_unused = model_setup(
        sequence_length=args.sequence_length,
        num_features=num_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout_rate=args.dropout_rate,
        frame_representation=args.frame_representation,
        num_cls=args.num_cls,
        prediction_head=args.prediction_head,
        device=device,
        base_lr=start_lr,
        pose_pred=args.pose_pred,
        orient_pred=args.orient_pred,
        pose_dict=pose_dict,
        orient_dict=orient_dict,
        recon=args.recon,
        positional_encoding_type=args.positional_encoding_type,
        patience=args.patience,
        plateau_factor=args.plateau_factor,
        pretrain_mode=getattr(args, "pretrain_mode", False),
        projection_dim=args.projection_dim,
        batches_per_epoch=max(1, len(train_loader)),
        create_scheduler=False,
        scheduler_type=getattr(args, "scheduler_type", "cosine"),
        weight_decay=getattr(args, "weight_decay", 1e-3),
        warmup_steps=getattr(args, "warmup_steps", 0),
        t_0=getattr(args, "t_0", 1000),
        t_mult=getattr(args, "t_mult", 1),
        eta_min=getattr(args, "eta_min", 1e-6),
        is_handoff=is_handoff,
        freeze_encoder=getattr(args, "freeze_encoder", False),
        weight_encoder=getattr(args, "weight_encoder", 1),
    )
    model.to(device)

    pretrained_path = getattr(args, "pretrained_model_path", None)
    if pretrained_path and os.path.isfile(pretrained_path):
        try:
            print(f"[LRFinder] Loading pretrained weights from: {pretrained_path}")
            # Weights only: optimizer/scheduler state is irrelevant for LR sweeping.
            load_checkpoint_model(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                checkpoint_path=pretrained_path,
                scaler=None,
                load_optimizer_state=False,
                strict=False,
            )
        except Exception as e:
            print(f"[LRFinder] WARNING: Could not load pretrained weights: {e}")

    initial_group_lrs = [pg.get("lr", start_lr) for pg in optimizer.param_groups]
    group_lr_ratios = [lr / start_lr for lr in initial_group_lrs]

    train_class_weights = train_class_weights.to(device)

    amp_enabled = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    grad_clip_norm = float(getattr(args, "grad_clip_norm", 1.0) or 0.0)

    num_iter = int(getattr(args, "lrfinder_iters", 200) or 200)
    num_iter = max(10, num_iter)
    gamma = (end_lr / start_lr) ** (1.0 / max(1, (num_iter - 1)))

    best_loss = float("inf")
    best_lr: Optional[float] = None
    lrs: list[float] = []
    losses: list[float] = []

    model.train()
    data_iter = iter(train_loader)

    for step in range(num_iter):
        try:
            sequences, pose_targets, orient_targets, _idx = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            sequences, pose_targets, orient_targets, _idx = next(data_iter)

        lr = start_lr * (gamma**step)
        for pg, ratio in zip(optimizer.param_groups, group_lr_ratios):
            pg["lr"] = lr * ratio

        sequences = sequences.to(device, non_blocking=True)
        pose_targets = pose_targets.to(device, non_blocking=True)
        orient_targets = orient_targets.to(device, non_blocking=True)

        # Augmentation interface note:
        # `augment_features()` does not take a `recon=` flag; instead it can optionally
        # return the elementwise keep-mask (True=kept, False=masked) via `return_mask=True`.
        #
        # LR-finder should mimic the *actual* training objective:
        #   - For supervised-only runs, we don't need the mask.
        #   - For recon/denoising runs, we need the keep-mask so we can compute masked-MSE
        #     only on masked elements.
        if args.recon:
            sequences_aug, keep_mask = augment_features(
                sequences,
                mask_p=args.mask_p,
                noise_level=args.noise_level,
                columns_to_noise=columns_to_noise,
                return_mask=True,
            )
        else:
            sequences_aug = augment_features(
                sequences,
                mask_p=args.mask_p,
                noise_level=args.noise_level,
                columns_to_noise=columns_to_noise,
                return_mask=False,
            )
            keep_mask = None

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=amp_enabled,
        ):
            pose_logits, orient_logits, recon_out, _ctx = model(sequences_aug)
            total_loss = torch.tensor(0.0, device=device)

            combined_targets = pose_targets * len(orient_dict) + orient_targets
            sample_weights = train_class_weights[combined_targets]

            if args.pose_pred:
                pose_loss = F.cross_entropy(pose_logits, pose_targets, reduction="none")
                pose_loss = (pose_loss * sample_weights).mean()
                total_loss = total_loss + float(args.pose_weight) * pose_loss

            if args.orient_pred:
                orient_loss = F.cross_entropy(orient_logits, orient_targets, reduction="none")
                orient_loss = (orient_loss * sample_weights).mean()
                total_loss = total_loss + float(args.orient_weight) * orient_loss

            if args.recon:
                if keep_mask is None:
                    raise RuntimeError("LRFinder internal error: keep_mask is None but recon=True")
                recon_loss = masked_mse_loss(recon_out, sequences, keep_mask)
                total_loss = total_loss + float(args.recon_weight) * recon_loss

        if amp_enabled:
            scaler.scale(total_loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_val = float(total_loss.detach().cpu().item())
        lrs.append(float(lr))
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_lr = float(lr)

        if not math.isfinite(loss_val) or (best_loss < float("inf") and loss_val > best_loss * 4.0):
            print(
                f"[LRFinder] Stopping early at step {step+1}/{num_iter} "
                f"(lr={lr:g}, loss={loss_val:g})."
            )
            break

        if (step + 1) % 25 == 0:
            print(f"[LRFinder] step {step+1:>4}/{num_iter} | lr={lr:g} | loss={loss_val:g}")

    records_dir = os.path.join(directory, "Records")
    _safe_mkdir(records_dir)

    df = pd.DataFrame({"lr": lrs, "loss": losses})
    csv_path = os.path.join(records_dir, f"{args.run_name}_lr_finder.csv")
    df.to_csv(csv_path, index=False)

    plot_path = _maybe_save_plot(df, os.path.join(records_dir, f"{args.run_name}_lr_finder.png"))

    if best_lr is not None:
        suggested = best_lr / 10.0
        print(f"[LRFinder] Best (min-loss) LR observed: {best_lr:g}")
        print(f"[LRFinder] Heuristic suggested base_lr: ~{suggested:g} (best_lr/10)")

    return LRFinderResult(csv_path=csv_path, plot_path=plot_path, best_lr=best_lr)
