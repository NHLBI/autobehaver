"""Check weights-only checkpoint preparation for the learning-rate finder."""
from types import SimpleNamespace

import h5py
import pytest
import torch

from autobehaver._training_core import prepare_data
from autobehaver import lr_finder
from tests.test_handoff_registry_training import _training_files


def test_lr_finder_uses_current_inputs_and_split_for_weights_only_checkpoint(tmp_path, monkeypatch):
    metadata = _training_files(tmp_path)
    initial = prepare_data(str(tmp_path), metadata, 5, num_workers=0,
                           pose_pred=True, orient_pred=False, defined_split=True)
    prepared = [initial]
    try:
        checkpoint = tmp_path / "resume.pth"
        torch.save({"checkpoint_kind": "supervised", "pose_dict": initial[2], **initial[-1]}, checkpoint)
        with h5py.File(tmp_path / "a.h5", "r+") as handle:
            handle["data"][0, 0] += 1
        metadata["set"] = ["val", "train"]
        with pytest.raises(ValueError, match="Training inputs changed"):
            prepare_data(str(tmp_path), metadata, 5, num_workers=0,
                         pose_pred=True, orient_pred=False, defined_split=True,
                         pretrained_model_path=str(checkpoint))

        class PreparationComplete(Exception):
            """Stop after native preparation, before model construction or sweeping."""

        def prepare_and_stop(**kwargs):
            result = prepare_data(**kwargs)
            prepared.append(result)
            raise PreparationComplete

        monkeypatch.setattr(lr_finder, "prepare_data", prepare_and_stop)
        args = SimpleNamespace(sequence_length=5, num_workers=0, random_seed=0,
                               pretrained_model_path=str(checkpoint), restart_optimizer=False,
                               pose_pred=True, orient_pred=False, recon=False,
                               max_class_weight=200, defined_split=True, noval=False,
                               ablation_preset="none")
        with pytest.raises(PreparationComplete):
            lr_finder.find_lr(args=args, directory=str(tmp_path), metadata=metadata)
        current = prepared[-1]
        assert len(prepared) == 2
        assert current[-1]["train_filenames"] == ["b.h5"]
        assert current[-1]["val_filenames"] == ["a.h5"]
        assert current[-1]["training_data_identity"] != initial[-1]["training_data_identity"]
        assert current[2] == initial[2]
    finally:
        for result in prepared:
            store = result[0].dataset.data_in_memory
            if hasattr(store, "close"):
                store.close()
