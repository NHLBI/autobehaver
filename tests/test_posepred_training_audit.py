from __future__ import annotations

import argparse
from pathlib import Path

import torch

from autobehaver._training_core import _apply_resume_training_runtime_state, _select_primary_validation_metric
from autobehaver.checkpoint import load_checkpoint_variables, save_checkpoint, set_checkpoint_globals
from autobehaver.model import PosePredTransformer
from autobehaver.records import setup_records_file


def _tiny_model() -> PosePredTransformer:
    return PosePredTransformer(
        num_features=4,
        sequence_length=5,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout_rate=0.1,
        frame_representation="middle",
        num_cls=1,
        prediction_head="fc",
        pose_pred=True,
        orient_pred=False,
        recon=False,
        num_pose_classes=2,
        num_orient_classes=0,
        positional_encoding_type="learned",
        pretrain_mode=False,
        projection_dim=16,
    )


def test_full_resume_restores_supervised_training_hyperparams(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "resume_supervised.pth"
    checkpoint = {
        "global_vars": {
            "sequence_length": 5,
            "num_features": 4,
            "d_model": 8,
            "nhead": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 16,
            "dropout_rate": 0.1,
            "frame_representation": "middle",
            "num_cls": 1,
            "prediction_head": "fc",
            "positional_encoding_type": "learned",
            "pose_pred": True,
            "orient_pred": True,
            "recon": False,
            "base_lr": 2e-4,
            "weight_decay": 5e-4,
            "warmup_steps": 123,
            "mask_p": 0.1,
            "noise_level": 0.2,
            "scheduler_type": "plateau",
            "t_0": 456,
            "t_mult": 2,
            "eta_min": 1e-7,
            "pose_weight": 3.0,
            "orient_weight": 0.5,
            "recon_weight": 0.25,
            "max_class_weight": 77.0,
            "grad_clip_norm": 2.5,
            "patience": 42,
            "plateau_factor": 0.33,
            "random_seed": 999,
            "num_epochs": 88,
            "checkpoint_interval": 11,
            "freeze_encoder": False,
            "weight_encoder": 1.0,
            "swa_mode": True,
            "swa_epochs": 9,
            "swa_lr": 1e-5,
            "swa_anneal_epochs": 4,
        },
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }
    torch.save(checkpoint, ckpt_path)

    args = argparse.Namespace(pretrain_mode=False, sequence_length=5)
    load_checkpoint_variables(args, str(ckpt_path), use_restart_optimizer=False, current_num_features=4)

    assert args.base_lr == 2e-4
    assert args.pose_weight == 3.0
    assert args.orient_weight == 0.5
    assert args.recon_weight == 0.25
    assert args.max_class_weight == 77.0
    assert args.grad_clip_norm == 2.5
    assert args.patience == 42
    assert args.plateau_factor == 0.33
    assert args.random_seed == 999
    assert args.swa_mode is True
    assert args.swa_epochs == 9


def test_save_checkpoint_persists_training_runtime_state(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    set_checkpoint_globals(
        sequence_length=5,
        num_features=4,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout_rate=0.1,
        frame_representation="middle",
        num_cls=1,
        prediction_head="fc",
        positional_encoding_type="learned",
        batch=2,
        base_lr=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        mask_p=0.0,
        noise_level=0.0,
        pose_weight=1.0,
        orient_weight=1.0,
        recon_weight=1.0,
        max_class_weight=10.0,
        scheduler_type="cosine",
        t_0=10,
        t_mult=1,
        eta_min=0.0,
        projection_dim=16,
        patience=5,
        plateau_factor=0.5,
        grad_clip_norm=1.0,
        random_seed=123,
        freeze_encoder=False,
        weight_encoder=1.0,
        weight_initialization="transformer_standard_v1",
        num_epochs=12,
        checkpoint_interval=3,
    )

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    runtime_state = {
        "best_val_loss": 0.42,
        "best_pose_macro_f1": 0.77,
        "best_orient_macro_f1": 0.66,
        "best_model_path": str(ckpt_dir / "demo_best.pth"),
        "epochs_since_improvement": 2,
        "total_epoch_time": 12.5,
        "num_completed_epochs": 7,
        "total_time_forward_pass": 6.1,
        "total_time_optimizer_step": 2.4,
        "total_inputs": 999,
    }
    save_checkpoint(
        epoch=3,
        global_step=44,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pose_dict={"a": 0, "b": 1},
        orient_dict=None,
        scaler=scaler,
        checkpoint_dir=str(ckpt_dir),
        run_name="demo",
        pose_pred=True,
        orient_pred=False,
        recon=False,
        split_filenames={"train_filenames": ["a.h5"], "val_filenames": ["b.h5"],
                         "training_data_identity": {"schema":"autobehaver.training_inputs.v1", "files":[]}},
        training_runtime_state=runtime_state,
    )

    ckpt = torch.load(ckpt_dir / "demo_model_epoch_4.pth", map_location="cpu")
    assert ckpt["training_runtime_state"]["best_pose_macro_f1"] == 0.77
    assert ckpt["training_runtime_state"]["epochs_since_improvement"] == 2
    assert ckpt["training_data_identity"]["schema"] == "autobehaver.training_inputs.v1"


def test_orientation_only_primary_metric_uses_orient_macro_f1() -> None:
    metric, name = _select_primary_validation_metric(
        noval=False,
        pose_pred=False,
        orient_pred=True,
        train_total_loss_avg=1.0,
        val_total_loss_avg=0.9,
        pose_macro_f1=0.1,
        orient_macro_f1=0.7,
    )
    assert name == "orient_macro_f1"
    assert metric == 0.7


def test_resume_runtime_state_defaults_are_safe() -> None:
    restored = _apply_resume_training_runtime_state(None)
    assert restored["best_val_loss"] == float("inf")
    assert restored["best_pose_macro_f1"] == float("-inf")
    assert restored["best_orient_macro_f1"] == float("-inf")
    assert restored["epochs_since_improvement"] == 0


def test_records_headers_include_weighted_train_recon_loss(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    file, writer, headers = setup_records_file(
        str(path), start_epoch=0, pose_pred=False, orient_pred=False,
        recon=True, pretrain_mode=False,
    )
    try:
        assert "Weighted Train Recon Loss Avg" in headers
    finally:
        file.close()
