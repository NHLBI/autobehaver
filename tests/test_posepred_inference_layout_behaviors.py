from __future__ import annotations

from pathlib import Path


def test_run_posepred_inference_discovers_h5_files_in_input_dir(monkeypatch, tmp_path: Path) -> None:
    import h5py
    import numpy as np

    from autobehaver.preinf.posepred_inference.config import PosePredInferenceConfig
    from autobehaver.preinf.posepred_inference.core import run_posepred_inference

    tmp_path = tmp_path.parent / 'layout'
    tmp_path.mkdir()

    input_dir = tmp_path / "5_normalized_features"
    input_dir.mkdir(parents=True)
    h5_path = input_dir / "sample_normalized.h5"
    with h5py.File(h5_path, "w") as hf:
        dset = hf.create_dataset("data", data=np.zeros((4, 2), dtype=np.float32))
        dset.attrs["columns"] = [b"f1", b"f2"]

    ckpt = tmp_path / "model.pth"
    import torch
    torch.save({'global_vars':{'pose_pred':True,'orient_pred':False},'pose_dict':{'A':0}},ckpt)

    captured: dict[str, object] = {}

    def _fake_run_inference(*args, **kwargs):
        captured["explicit_files"] = list(kwargs.get("explicit_files") or [])

    monkeypatch.setattr("autobehaver.preinf.posepred_inference.core.run_inference", _fake_run_inference)
    monkeypatch.setattr(
        "autobehaver.preinf.posepred_inference.core._completed_inference_matches",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "autobehaver.preinf.posepred_inference.core._validate_inference_csv",
        lambda *args, **kwargs: (True, "ok", {"rows": 4, "columns": 2}),
    )

    cfg = PosePredInferenceConfig(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "6_posepred_inference"),
        inference_model=[str(ckpt)],
        write_run_log=False,
    )

    run_posepred_inference(cfg)

    assert captured["explicit_files"] == [str(h5_path.resolve())]


def test_run_inference_does_not_archive_when_no_inputs_are_selected(monkeypatch, tmp_path: Path) -> None:
    from autobehaver.inference import run_inference

    input_dir = tmp_path / "5_normalized_features"
    input_dir.mkdir(parents=True)
    output_dir = tmp_path / "6_posepred_inference"
    output_dir.mkdir(parents=True)

    (output_dir / "sample_inference.csv").write_text("frame,pred\n0,A\n")
    details_dir = output_dir / "inference_details"
    details_dir.mkdir(parents=True)
    (details_dir / "sample_logits.csv").write_text("frame,logit\n0,0.1\n")

    monkeypatch.setattr("autobehaver.inference._discover_inference_input_files", lambda *a, **k: [])

    run_inference(
        directory=str(input_dir),
        inference_model_paths=[],
        num_workers=0,
        batch=1,
        inf_average=1,
        sequence_length=41,
        min_pose_bout=1,
        min_orient_bout=1,
        output_dir=str(output_dir),
        clobber=True,
        archive_existing=True,
        use_inferences_subdir=False,
        explicit_files=[],
    )

    assert (output_dir / "sample_inference.csv").exists()
    assert (details_dir / "sample_logits.csv").exists()
    assert not (output_dir / "past_inferences").exists()
