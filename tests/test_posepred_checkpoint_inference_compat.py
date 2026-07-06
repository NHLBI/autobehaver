from __future__ import annotations

import numpy as np
import h5py
import pytest
import torch

from autobehaver.checkpoint import build_checkpoint_global_vars, set_checkpoint_globals
from autobehaver.feature_selection import compute_schema_hash
from autobehaver.inference import (
    check_ensemble_compatibility_and_get_permutations,
    load_model_for_inference,
    prepare_data_inference,
)
from autobehaver.model import PosePredTransformer


def _write_h5(path, columns: list[str], rows: int = 9) -> None:
    data = np.arange(rows * len(columns), dtype=np.float32).reshape(rows, len(columns))
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("data", data=data)
        h5f.attrs["columns"] = np.array(columns, dtype="S")


def _set_training_globals(*, canonical: list[str], included: list[str], columns_to_noise: list[int]) -> None:
    set_checkpoint_globals(
        sequence_length=5,
        num_features=len(included),
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
        feature_schema_hash=compute_schema_hash(included),
        feature_columns_included=list(included),
        feature_columns_excluded=[c for c in canonical if c not in set(included)],
        feature_columns_canonical=list(canonical),
        ablation_spec={"preset": "none"},
        columns_to_noise=list(columns_to_noise),
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
        projection_dim=128,
        patience=5,
        plateau_factor=0.5,
        grad_clip_norm=1.0,
        random_seed=123,
        freeze_encoder=False,
        weight_encoder=1.0,
        weight_initialization="kaiming",
    )


def _make_supervised_checkpoint(path, *, global_vars: dict, pose_dict: dict[str, int], orient_dict: dict[str, int]) -> None:
    model = PosePredTransformer(
        num_features=int(global_vars["num_features"]),
        sequence_length=int(global_vars["sequence_length"]),
        d_model=int(global_vars["d_model"]),
        nhead=int(global_vars["nhead"]),
        num_encoder_layers=int(global_vars["num_encoder_layers"]),
        num_decoder_layers=int(global_vars["num_decoder_layers"]),
        dim_feedforward=int(global_vars["dim_feedforward"]),
        dropout_rate=float(global_vars["dropout_rate"]),
        frame_representation=str(global_vars["frame_representation"]),
        num_cls=int(global_vars["num_cls"]),
        prediction_head=str(global_vars["prediction_head"]),
        pose_pred=bool(global_vars.get("pose_pred", False)),
        orient_pred=bool(global_vars.get("orient_pred", False)),
        recon=bool(global_vars.get("recon", False)),
        num_pose_classes=len(pose_dict),
        num_orient_classes=len(orient_dict),
        positional_encoding_type=str(global_vars["positional_encoding_type"]),
        pretrain_mode=False,
        projection_dim=int(global_vars.get("projection_dim", 128)),
    )
    checkpoint = {
        "checkpoint_schema_version": 1,
        "checkpoint_kind": "supervised",
        "run_name": "synthetic",
        "epoch": 0,
        "global_step": 0,
        "model_state_dict": model.state_dict(),
        "global_vars": global_vars,
        "pose_dict": dict(pose_dict),
        "orient_dict": dict(orient_dict),
    }
    torch.save(checkpoint, path)


def test_swa_checkpoint_is_inference_compatible_with_feature_schema(tmp_path):
    canonical = ["f0", "f1", "f2", "f3", "f4"]
    included = ["f1", "f3", "f4"]
    _write_h5(tmp_path / "sample.h5", canonical)

    _set_training_globals(canonical=canonical, included=included, columns_to_noise=[0, 1, 2])
    pose_dict = {"rest": 0, "walk": 1}
    orient_dict = {"left": 0, "right": 1}

    normal_globals = build_checkpoint_global_vars(pose_pred=True, orient_pred=True, recon=False)
    swa_globals = build_checkpoint_global_vars(
        pose_pred=True,
        orient_pred=True,
        recon=False,
        extra_overrides={"swa_mode": True, "swa_start": 1, "swa_epochs": 2, "swa_lr": 1e-5},
    )

    normal_path = tmp_path / "normal.pth"
    swa_path = tmp_path / "swa.pth"
    _make_supervised_checkpoint(normal_path, global_vars=normal_globals, pose_dict=pose_dict, orient_dict=orient_dict)
    _make_supervised_checkpoint(swa_path, global_vars=swa_globals, pose_dict=pose_dict, orient_dict=orient_dict)

    files, pose_loaded, orient_loaded, cols_noise, feature_plan = prepare_data_inference(
        str(tmp_path), str(swa_path), num_workers=0, inference_mode=True
    )
    assert len(files) == 1
    assert pose_loaded == pose_dict
    assert orient_loaded == orient_dict
    assert feature_plan is not None
    assert feature_plan.included_columns == included
    assert cols_noise == [0, 1, 2]

    _, pose_perms, orient_perms = check_ensemble_compatibility_and_get_permutations(
        [str(normal_path), str(swa_path)],
        reference_pose_dict=pose_dict,
        reference_orient_dict=orient_dict,
    )
    assert pose_perms == [None, None]
    assert orient_perms == [None, None]

    model, checkpoint = load_model_for_inference(str(swa_path), pose_loaded, orient_loaded)
    assert checkpoint["global_vars"]["feature_schema_hash"] == compute_schema_hash(included)
    with torch.no_grad():
        pose_logits, orient_logits, recon_out, cls = model(torch.randn(2, 5, len(included)))
    assert recon_out is None
    assert pose_logits is not None and tuple(pose_logits.shape) == (2, len(pose_dict))
    assert orient_logits is not None and tuple(orient_logits.shape) == (2, len(orient_dict))
    assert tuple(cls.shape) == (2, int(checkpoint["global_vars"]["d_model"]))


def test_prepare_data_inference_rejects_pretraining_checkpoint(tmp_path):
    _write_h5(tmp_path / "sample.h5", ["f0", "f1", "f2"])
    ckpt = {
        "checkpoint_kind": "pretrain",
        "pretrain_mode": True,
        "global_vars": {"pretrain_mode": True, "pose_pred": False, "orient_pred": False, "num_features": 3},
    }
    ckpt_path = tmp_path / "pretrain.pth"
    torch.save(ckpt, ckpt_path)

    with pytest.raises(ValueError, match="pretraining checkpoint"):
        prepare_data_inference(str(tmp_path), str(ckpt_path), num_workers=0, inference_mode=True)


def test_prepare_data_inference_rejects_num_features_mismatch_without_schema_metadata(tmp_path):
    _write_h5(tmp_path / "sample.h5", ["f0", "f1", "f2", "f3", "f4"])
    ckpt = {
        "global_vars": {
            "pose_pred": True,
            "orient_pred": False,
            "num_features": 3,
        },
        "pose_dict": {"rest": 0, "walk": 1},
    }
    ckpt_path = tmp_path / "mismatch.pth"
    torch.save(ckpt, ckpt_path)

    with pytest.raises(ValueError, match="Cannot safely infer the feature mapping"):
        prepare_data_inference(str(tmp_path), str(ckpt_path), num_workers=0, inference_mode=True)


def test_prepare_data_inference_repairs_invalid_columns_to_noise(tmp_path):
    canonical = ["f0", "f1", "f2", "f3", "f4"]
    included = ["f1", "f3", "f4"]
    _write_h5(tmp_path / "sample.h5", canonical)

    _set_training_globals(canonical=canonical, included=included, columns_to_noise=[0, 1, 2, 3, 4])
    ckpt = {
        "global_vars": build_checkpoint_global_vars(pose_pred=True, orient_pred=False, recon=False),
        "pose_dict": {"rest": 0, "walk": 1},
    }
    ckpt_path = tmp_path / "bad_noise_cols.pth"
    torch.save(ckpt, ckpt_path)

    _, _, _, cols_noise, feature_plan = prepare_data_inference(str(tmp_path), str(ckpt_path), num_workers=0, inference_mode=True)
    assert feature_plan is not None
    assert feature_plan.included_columns == included
    assert cols_noise == [0, 1, 2]
