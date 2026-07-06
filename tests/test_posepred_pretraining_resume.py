from __future__ import annotations

import argparse
from pathlib import Path

import torch

from autobehaver.checkpoint import load_checkpoint_model, load_checkpoint_variables, save_checkpoint, set_checkpoint_globals
from autobehaver.model import PosePredTransformer
from autobehaver.pretraining import _apply_resume_pretraining_runtime_state
from autobehaver.records import setup_records_file


def _tiny_pretrain_model() -> PosePredTransformer:
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
        pose_pred=False,
        orient_pred=False,
        recon=False,
        num_pose_classes=0,
        num_orient_classes=0,
        positional_encoding_type="learned",
        pretrain_mode=True,
        projection_dim=8,
    )


def test_load_checkpoint_variables_restores_pretraining_specific_args(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "resume_pretrain.pth"
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
            "projection_dim": 32,
            "base_lr": 2e-4,
            "weight_decay": 1e-3,
            "scheduler_type": "cosine",
            "warmup_steps": 12,
            "t_0": 456,
            "t_mult": 2,
            "eta_min": 1e-7,
            "mask_p": 0.1,
            "noise_level": 0.2,
            "grad_clip_norm": 2.5,
            "random_seed": 999,
            "num_epochs": 88,
            "checkpoint_interval": 11,
            "pretrain_mode": True,
            "temperature": 0.22,
            "pretrain_files_per_group": 123,
            "log_interval": 77,
            "pretrain_loss": "info_nce",
            "barlow_lambda": 0.004,
        },
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }
    torch.save(checkpoint, ckpt_path)

    args = argparse.Namespace(pretrain_mode=True, sequence_length=5)
    load_checkpoint_variables(args, str(ckpt_path), use_restart_optimizer=False, current_num_features=4)

    assert args.projection_dim == 32
    assert args.temperature == 0.22
    assert args.pretrain_files_per_group == 123
    assert args.log_interval == 77
    assert args.pretrain_loss == "info_nce"
    assert args.barlow_lambda == 0.004


def test_save_checkpoint_persists_pretraining_runtime_state(tmp_path: Path) -> None:
    model = _tiny_pretrain_model()
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
        projection_dim=8,
        temperature=0.1,
        pretrain_files_per_group=50,
        log_interval=25,
        pretrain_loss="barlow_twins",
        barlow_lambda=5e-3,
        batch=2,
        base_lr=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        mask_p=0.0,
        noise_level=0.0,
        scheduler_type="cosine",
        t_0=10,
        t_mult=1,
        eta_min=0.0,
        patience=5,
        plateau_factor=0.5,
        grad_clip_norm=1.0,
        random_seed=123,
        weight_initialization="transformer_standard_v1",
        num_epochs=12,
        checkpoint_interval=3,
        pretrain_mode=True,
    )

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    runtime_state = {
        "resume_epoch": 2,
        "next_chunk_index": 4,
        "next_batch_index_in_chunk": 7,
        "epoch_total_loss": 3.25,
        "epoch_num_batches": 12,
        "interval_loss": 0.42,
        "interval_steps": 5,
        "best_interval_loss": 0.11,
        "interval_elapsed_s": 8.5,
    }
    save_checkpoint(
        epoch=2,
        global_step=44,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pose_dict=None,
        orient_dict=None,
        scaler=scaler,
        checkpoint_dir=str(ckpt_dir),
        run_name="demo_pretrain",
        pose_pred=False,
        orient_pred=False,
        recon=False,
        split_filenames=None,
        pretrain_mode=True,
        pretraining_runtime_state=runtime_state,
    )

    ckpt = torch.load(ckpt_dir / "demo_pretrain_model_epoch_3.pth", map_location="cpu")
    assert ckpt["pretraining_runtime_state"]["next_chunk_index"] == 4
    assert ckpt["pretraining_runtime_state"]["best_interval_loss"] == 0.11


def test_setup_records_file_trims_pretraining_by_global_step(tmp_path: Path) -> None:
    path = tmp_path / "pretrain_records.csv"
    path.write_text(
        "Global Step,Epoch,Avg Loss (Interval),Learning Rate,Time Elapsed (s)\n"
        "10,1,0.5,0.001,1.0\n"
        "20,1,0.4,0.001,1.1\n"
        "30,2,0.3,0.0005,1.2\n"
    )
    file, writer, headers = setup_records_file(
        str(path),
        start_epoch=1,
        pose_pred=False,
        orient_pred=False,
        recon=False,
        pretrain_mode=True,
        start_global_step=20,
    )
    file.close()
    rows = path.read_text().strip().splitlines()
    assert rows == [
        "Global Step,Epoch,Avg Loss (Interval),Learning Rate,Time Elapsed (s)",
        "10,1,0.5,0.001,1.0",
        "20,1,0.4,0.001,1.1",
    ]


def test_load_checkpoint_model_respects_pretrain_mode_for_true_resume(tmp_path: Path) -> None:
    model = _tiny_pretrain_model()
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
        projection_dim=8,
        temperature=0.1,
        pretrain_files_per_group=50,
        log_interval=25,
        pretrain_loss="barlow_twins",
        barlow_lambda=5e-3,
        batch=2,
        base_lr=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        mask_p=0.0,
        noise_level=0.0,
        scheduler_type="cosine",
        t_0=10,
        t_mult=1,
        eta_min=0.0,
        patience=5,
        plateau_factor=0.5,
        grad_clip_norm=1.0,
        random_seed=123,
        weight_initialization="transformer_standard_v1",
        num_epochs=12,
        checkpoint_interval=3,
        pretrain_mode=True,
    )

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    save_checkpoint(
        epoch=2,
        global_step=44,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pose_dict=None,
        orient_dict=None,
        scaler=scaler,
        checkpoint_dir=str(ckpt_dir),
        run_name="demo_pretrain",
        pose_pred=False,
        orient_pred=False,
        recon=False,
        split_filenames=None,
        pretrain_mode=True,
    )

    fresh_model = _tiny_pretrain_model()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.LinearLR(fresh_optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
    fresh_scaler = torch.amp.GradScaler("cpu", enabled=False)
    start_epoch, global_step = load_checkpoint_model(
        fresh_model,
        fresh_optimizer,
        fresh_scheduler,
        str(ckpt_dir / "demo_pretrain_model_epoch_3.pth"),
        fresh_scaler,
        load_optimizer_state=True,
        pretrain_mode=True,
    )
    assert start_epoch == 3
    assert global_step == 44


def test_apply_resume_pretraining_runtime_state_defaults_are_safe() -> None:
    restored = _apply_resume_pretraining_runtime_state(None)
    assert restored["next_chunk_index"] == 0
    assert restored["next_batch_index_in_chunk"] == 0
    assert restored["best_interval_loss"] == float("inf")
