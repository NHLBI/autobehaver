"""Content identities for supervised training inputs and checkpoint continuity."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .completion import file_digest, verified_run


@verified_run
def build_training_data_identity(directory, metadata, *, sequence_length, pose_pred, orient_pred):
    """Fingerprint validated inputs once per preparation, independently of root location."""
    root = Path(directory)
    files = []
    for _, row in metadata.iterrows():
        filename = str(row["filename"]).replace("\\", "/")
        feature_path = root / str(row["filename"])
        label_value = row.get("labels")
        labels_required = bool(pose_pred or orient_pred)
        label_name = None
        label_digest = None
        if labels_required and not pd.isna(label_value) and str(label_value).strip():
            label_name = str(label_value).replace("\\", "/")
            label_digest = file_digest(root / str(label_value))
        files.append({"filename": filename, "feature_sha256": file_digest(feature_path),
                      "labels": label_name, "label_sha256": label_digest})
    return {"schema": "autobehaver.training_inputs.v1", "sequence_length": int(sequence_length),
            "pose_pred": bool(pose_pred), "orient_pred": bool(orient_pred),
            "files": sorted(files, key=lambda item: item["filename"])}


def validate_training_data_identity(saved, current):
    """Protect true resumes while retaining readable historical checkpoints."""
    if saved is None:
        logging.warning(
            "Historical checkpoint has no input content fingerprints. Its existing split/schema "
            "checks remain active; the next saved checkpoint will record current input contents."
        )
        return
    if not isinstance(saved, dict) or saved.get("schema") != "autobehaver.training_inputs.v1":
        raise ValueError("Checkpoint contains an unsupported training-input identity record")
    saved_files = saved.get("files")
    if not isinstance(saved_files, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("filename"), str)
        for item in saved_files
    ):
        raise ValueError("Checkpoint has an invalid training-input file record")
    # Label-file location is provenance, not mathematical identity. A moved,
    # byte-identical label file remains the same input to the same recording.
    keys = ("filename", "feature_sha256", "label_sha256")
    old_files = {item["filename"]: {key: item.get(key) for key in keys} for item in saved_files}
    new_files = {item["filename"]: {key: item.get(key) for key in keys} for item in current["files"]}
    if len(old_files) != len(saved_files):
        raise ValueError("Checkpoint has duplicate training-input identities")
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    settings_changed = any(saved.get(key) != current.get(key)
                           for key in ("sequence_length", "pose_pred", "orient_pred"))
    if not changed and not settings_changed:
        return
    detail = f"Changed input files: {changed[:10]}" if changed else "Input preparation settings changed"
    if settings_changed and changed:
        detail += "; input preparation settings also changed"
    raise ValueError(
        "Training inputs changed since this checkpoint was saved. A full training resume "
        "requires the same input contents. " + detail + ". For intentional training on changed "
        "data, use the existing --restart_optimizer weights-only training option and a new run name."
    )
