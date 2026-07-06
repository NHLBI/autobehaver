from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from autobehaver.checkpoint import load_checkpoint_variables, save_checkpoint, set_checkpoint_globals
from autobehaver.cli import build_parser
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


def test_cli_help_no_longer_exposes_swa_start() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "--swa_start" not in help_text
    assert "--swa_mode" in help_text
    assert "--swa_epochs" in help_text


def test_training_records_include_swa_columns(tmp_path: Path) -> None:
    records_path = tmp_path / "train.csv"
    file, writer, headers = setup_records_file(
        str(records_path),
        start_epoch=0,
        pose_pred=True,
        orient_pred=True,
        recon=False,
        pretrain_mode=False,
    )
    try:
        assert "Model Variant" in headers
        assert "LR Phase" in headers
        assert "SWA Active" in headers
        assert "SWA Snapshots Averaged" in headers
    finally:
        file.close()


def test_save_checkpoint_persists_swa_runtime_state(tmp_path: Path) -> None:
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
        swa_mode=True,
        swa_start=8,
        swa_epochs=4,
        swa_lr=5e-5,
        swa_anneal_epochs=4,
    )

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    save_checkpoint(
        epoch=8,
        global_step=123,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pose_dict={"rest": 0, "walk": 1},
        orient_dict=None,
        scaler=scaler,
        checkpoint_dir=str(ckpt_dir),
        run_name="demo",
        pose_pred=True,
        orient_pred=False,
        recon=False,
        split_filenames={"train_filenames": ["a.h5"], "val_filenames": ["b.h5"]},
        swa_runtime_state={
            "swa_mode": True,
            "swa_start": 8,
            "swa_epochs": 4,
            "swa_lr": 5e-5,
            "swa_anneal_epochs": 4,
            "swa_started": True,
            "swa_n_averaged": 2,
            "swa_model_state_dict": {"dummy": torch.tensor([1.0])},
            "swa_scheduler_state_dict": {"foo": "bar"},
        },
    )

    ckpt_path = ckpt_dir / "demo_model_epoch_9.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "swa_runtime_state" in ckpt
    assert ckpt["swa_runtime_state"]["swa_started"] is True
    assert ckpt["swa_runtime_state"]["swa_n_averaged"] == 2


def test_load_checkpoint_variables_restores_swa_resume_config(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "resume.pth"
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
            "orient_pred": False,
            "recon": False,
            "base_lr": 1e-3,
            "weight_decay": 0.0,
            "warmup_steps": 10,
            "mask_p": 0.0,
            "noise_level": 0.0,
            "scheduler_type": "cosine",
            "t_0": 100,
            "t_mult": 1,
            "eta_min": 1e-6,
            "num_epochs": 30,
            "checkpoint_interval": 5,
            "swa_mode": True,
            "swa_epochs": 6,
            "swa_lr": 5e-5,
            "swa_anneal_epochs": 6,
        },
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }
    torch.save(checkpoint, ckpt_path)

    args = argparse.Namespace(
        pretrain_mode=False,
        sequence_length=5,
        num_epochs=999,
        checkpoint_interval=999,
        swa_mode=False,
        swa_epochs=0,
        swa_lr=1e-4,
    )
    load_checkpoint_variables(args, str(ckpt_path), use_restart_optimizer=False, current_num_features=4)

    assert args.num_epochs == 30
    assert args.checkpoint_interval == 5
    assert args.swa_mode is True
    assert args.swa_epochs == 6
    assert args.swa_lr == 5e-5
    assert getattr(args, "swa_anneal_epochs") == 6
