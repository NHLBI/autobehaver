from __future__ import annotations

import os
import json

import numpy as np
import h5py
import pytest
import torch

from autobehaver.inference import prepare_data_inference, run_inference
from autobehaver.atomic_io import sha256_file
from autobehaver.model import PosePredTransformer
from autobehaver.feature_selection import compute_schema_hash


def _write_h5(path, columns: list[str], rows: int = 9) -> None:
    data = np.arange(rows * len(columns), dtype=np.float32).reshape(rows, len(columns))
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("data", data=data)
        h5f.attrs["columns"] = np.array(columns, dtype="S")


def _make_supervised_checkpoint(path, *, columns: list[str], pose_dict: dict[str, int], orient_dict: dict[str, int] | None = None) -> None:
    orient_dict = orient_dict or {}
    global_vars = {
        "pose_pred": True,
        "orient_pred": bool(orient_dict),
        "recon": False,
        "pretrain_mode": False,
        "num_features": len(columns),
        "feature_columns_included": list(columns),
        "feature_schema_hash": compute_schema_hash(columns),
        "sequence_length": 5,
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
        "weight_initialization": "kaiming",
    }
    model = PosePredTransformer(
        num_features=len(columns),
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
        orient_pred=bool(orient_dict),
        recon=False,
        num_pose_classes=len(pose_dict),
        num_orient_classes=len(orient_dict),
        positional_encoding_type="learned",
        pretrain_mode=False,
        projection_dim=128,
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


def test_prepare_data_inference_accepts_candidate_files_outside_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    data_dir = tmp_path / "external"
    data_dir.mkdir()
    h5_path = data_dir / "sample.h5"
    _write_h5(h5_path, ["f0", "f1", "f2"], rows=7)

    ckpt_path = tmp_path / "model.pth"
    _make_supervised_checkpoint(ckpt_path, columns=["f0", "f1", "f2"], pose_dict={"rest": 0, "walk": 1})

    files, pose_dict, orient_dict, columns_to_noise, feature_plan = prepare_data_inference(
        str(empty_dir),
        str(ckpt_path),
        num_workers=0,
        inference_mode=True,
        candidate_files=[str(h5_path)],
    )

    assert files == [str(h5_path)]
    assert pose_dict == {"rest": 0, "walk": 1}
    assert orient_dict == {}
    assert columns_to_noise == [0, 1, 2]
    assert feature_plan.included_columns == ["f0", "f1", "f2"]


def test_run_inference_supports_explicit_files_outside_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    data_dir = tmp_path / "external"
    data_dir.mkdir()
    h5_path = data_dir / "sample.h5"
    _write_h5(h5_path, ["f0", "f1", "f2"], rows=7)

    ckpt_path = tmp_path / "ensemble_member.pth"
    _make_supervised_checkpoint(ckpt_path, columns=["f0", "f1", "f2"], pose_dict={"rest": 0, "walk": 1})

    out_dir = tmp_path / "out"
    run_inference(
        directory=str(empty_dir),
        inference_model_paths=[str(ckpt_path)],
        num_workers=0,
        batch=4,
        inf_average=1,
        sequence_length=5,
        min_pose_bout=1,
        min_orient_bout=1,
        use_tta=False,
        output_dir=str(out_dir),
        use_inferences_subdir=False,
        explicit_files=[str(h5_path)],
    )

    assert (out_dir / "sample_inference.csv").exists()
    manifest_path = out_dir / "sample_inference.csv.provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["inputs"]["source_hdf5"]["sha256"] == sha256_file(h5_path)
    assert manifest["inputs"]["checkpoints"][0]["sha256"] == sha256_file(ckpt_path)
    assert {item["path"] for item in manifest["outputs"]} >= {
        "sample_inference.csv",
        os.path.join("inference_details", "sample_logits.csv"),
    }
    assert (out_dir / "inference_details" / "sample_logits.csv").exists()
    cv_path = out_dir / "inference_details" / "cv" / "m01" / "sample_cv.h5"
    assert cv_path.exists()
    with h5py.File(cv_path, "r") as h5f:
        assert int(h5f["frame_number"].shape[0]) == 7
        # 7 rows with sequence_length=5 => 3 valid center frames
        assert int(np.asarray(h5f["valid"][:]).sum()) == 3


def test_run_inference_compacts_overlong_windows_context_artifact_path(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    data_dir = tmp_path / "external"
    data_dir.mkdir()
    long_stem = "CONTROL-SPD2-" + ("recording-segment-" * 5)
    h5_path = data_dir / f"{long_stem}.h5"
    _write_h5(h5_path, ["f0", "f1", "f2"], rows=7)

    ckpt_path = tmp_path / "ensemble_member.pth"
    _make_supervised_checkpoint(ckpt_path, columns=["f0", "f1", "f2"], pose_dict={"rest": 0, "walk": 1})

    out_dir = tmp_path / "out"
    run_inference(
        directory=str(empty_dir),
        inference_model_paths=[str(ckpt_path)],
        num_workers=0,
        batch=4,
        inf_average=1,
        sequence_length=5,
        min_pose_bout=1,
        min_orient_bout=1,
        use_tta=False,
        output_dir=str(out_dir),
        use_inferences_subdir=False,
        explicit_files=[str(h5_path)],
    )

    from autobehaver.portable_paths import prediction_path, original_name
    prediction = prediction_path(h5_path, out_dir)
    assert prediction.exists()
    assert original_name(prediction) == f"{long_stem}_inference.csv"
    context_paths = list((out_dir / "inference_details" / "cv").glob("**/*.h5"))
    assert len(context_paths) == 1
    with h5py.File(context_paths[0], "r") as h5f:
        assert h5f.attrs["source_filename"] == h5_path.name
        if os.name == "nt":
            assert int(h5f.attrs["artifact_name_compacted"]) == 1


def test_run_inference_rejects_duplicate_output_stems(tmp_path):
    _write_h5(tmp_path / "dup.h5", ["f0", "f1", "f2"], rows=7)
    _write_h5(tmp_path / "dup.hdf5", ["f0", "f1", "f2"], rows=7)

    ckpt_path = tmp_path / "model.pth"
    _make_supervised_checkpoint(ckpt_path, columns=["f0", "f1", "f2"], pose_dict={"rest": 0, "walk": 1})

    with pytest.raises(ValueError, match="collide on output filenames"):
        run_inference(
            directory=str(tmp_path),
            inference_model_paths=[str(ckpt_path)],
            num_workers=0,
            batch=4,
            inf_average=1,
            sequence_length=5,
            min_pose_bout=1,
            min_orient_bout=1,
            use_tta=False,
            output_dir=str(tmp_path / "out"),
            use_inferences_subdir=False,
        )
