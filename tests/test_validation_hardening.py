from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import sha256_file
from autobehaver.datasets import (
    validate_feature_hdf5_file,
    validate_pretraining_metadata,
    validate_supervised_training_metadata,
)
from autobehaver.postinf.featureimp.matrix import validate_feature_matrix
from autobehaver.postinf.featureimp.plan import validate_hhit_fold_plan
from autobehaver.postinf.featureimp.integrity import seal_receipt
from autobehaver.postinf.featureimp.shard_runner import _validate_completed_shard
from autobehaver.postinf.infmovie.pipeline import _inspect_rendered_movie
from autobehaver.postinf.mmgen.feature_matrix import validate_master_matrix_dataframe
from autobehaver.postinf.mmgen.worker_io import _align_poscalc_to_inference
from autobehaver.preinf.keypre.core import inspect_keypre_output
from autobehaver.preinf.keyprenorm.inspect_outputs import inspect_normalized_h5
from autobehaver.preinf.posepred_inference.core import _validate_inference_csv


def test_keypre_output_must_match_dlc_frame_count(tmp_path) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("scorer", "bodyparts", "coords"),
            ("scorer", "head", "x"),
            ("scorer", "head", "y"),
            ("scorer", "head", "likelihood"),
        ]
    )
    source = tmp_path / "source.csv"
    pd.DataFrame(np.ones((4, 4)), columns=columns).to_csv(source)
    output = tmp_path / "output.csv"
    from autobehaver.preinf.keypre.spec import parts
    columns = [f'{part}_{coord}_raw' for part in parts for coord in ('x', 'y', 'likelihood')]
    pd.DataFrame(np.ones((3, len(columns))), columns=columns).to_csv(output, index=False)

    ok, reason, _rows, _columns = inspect_keypre_output(str(source), str(output), raw_only=True)
    assert not ok
    assert "row count" in reason


def test_keyprenorm_h5_checks_exact_schema_rows_and_finite_values(tmp_path) -> None:
    output = tmp_path / "normalized.h5"
    with h5py.File(output, "w") as handle:
        handle.create_dataset("data", data=np.array([[1.0, np.inf]], dtype=np.float32))
        handle.attrs["columns"] = np.asarray(["f1", "f2"], dtype="S")

    ok, _rows, reason = inspect_normalized_h5(
        str(output),
        expected_n_cols=2,
        expected_rows=1,
        expected_columns=["f1", "f2"],
    )
    assert not ok
    assert reason == "nonfinite_values"


def test_posepred_csv_checks_frame_and_probability_contract(tmp_path) -> None:
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data", data=np.zeros((3, 2), dtype=np.float32))
    output = tmp_path / "inference.csv"
    pd.DataFrame(
        {
            "Frame Number": [1, 2, 3],
            "Pose-rest-Prob": [np.nan, 0.7, np.nan],
            "Pose-walk-Prob": [np.nan, 0.3, np.nan],
            "Predicted Pose Label": [np.nan, "rest", np.nan],
        }
    ).to_csv(output, index=False)
    assert _validate_inference_csv(str(source), output)[0]

    frame = pd.read_csv(output)
    frame.loc[1, "Pose-rest-Prob"] = 1.2
    frame.to_csv(output, index=False)
    ok, reason, _details = _validate_inference_csv(str(source), output)
    assert not ok
    assert "outside [0, 1]" in reason


def test_mmgen_alignment_refuses_heuristic_truncation_by_default() -> None:
    inference = pd.DataFrame({"label": ["a", "b", "c", "d"]})
    positional = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="No rows were truncated"):
        _align_poscalc_to_inference(
            inference,
            positional,
            edge_skip=0,
            source_name="sample.csv",
        )

    aligned_inference, aligned_positional = _align_poscalc_to_inference(
        inference,
        positional,
        edge_skip=0,
        source_name="sample.csv",
        allow_unsafe_fallback=True,
    )
    assert len(aligned_inference) == len(aligned_positional) == 3


def test_mmgen_frame_number_must_be_unique_integer_and_in_bounds() -> None:
    inference = pd.DataFrame({"Frame Number": [1, 1], "label": ["a", "b"]})
    positional = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Invalid Frame Number"):
        _align_poscalc_to_inference(
            inference,
            positional,
            edge_skip=0,
            source_name="sample.csv",
        )


def test_master_matrix_rejects_invalid_identity_and_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "individual": ["fly1", "fly2"],
            "group": ["control", "test"],
            "sex": ["F", "M"],
            "TimeBudget_Pose_Walk": [0.4, 1.2],
            "Spatial_distance": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="outside"):
        validate_master_matrix_dataframe(frame)

    frame.loc[1, "TimeBudget_Pose_Walk"] = 0.6
    frame.loc[1, "individual"] = "fly1"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_master_matrix_dataframe(frame)


def test_featureimp_matrix_rejects_duplicate_individuals(tmp_path) -> None:
    path = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "individual": ["same", "same"],
            "group": ["a", "b"],
            "sex": ["F", "M"],
            "BoutSummary_feature": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    ok, reason, _frame = validate_feature_matrix(str(path))
    assert not ok
    assert "Duplicate individual" in str(reason)


def test_fold_plan_requires_exact_nonoverlapping_coverage() -> None:
    valid = [
        {
            "h_hit": 1,
            "folds": [
                {"fold_idx": 0, "train_indices": [2, 3], "test_indices": [0, 1]},
                {"fold_idx": 1, "train_indices": [0, 1], "test_indices": [2, 3]},
            ],
        }
    ]
    validate_hhit_fold_plan(valid, n_training_rows=4, n_splits=2, expected_h_hits=1)

    invalid = json.loads(json.dumps(valid))
    invalid[0]["folds"][1]["test_indices"] = [1, 2, 3]
    with pytest.raises(ValueError, match="overlap|cover"):
        validate_hhit_fold_plan(invalid, n_training_rows=4, n_splits=2, expected_h_hits=1)


def test_featureimp_done_marker_detects_changed_artifact(tmp_path) -> None:
    plan = tmp_path / "_featureimp_plan" / "PLAN_MANIFEST.json"
    plan.parent.mkdir()
    artifact = tmp_path / "artifact.bin"
    done = tmp_path / "SHARD_DONE.json"
    plan.write_text("{}", encoding="utf-8")
    artifact.write_bytes(b"original")
    done.write_text(
        json.dumps(seal_receipt(
            {
                "status": "completed",
                "analysis_folder": str(tmp_path),
                "plan_manifest_sha256": sha256_file(plan),
                "artifacts": {"real_partials": "artifact.bin"},
            }, str(done)
        )),
        encoding="utf-8",
    )
    assert _validate_completed_shard(
        str(done),
        analysis_folder_path=str(tmp_path),
        plan_manifest_path=str(plan),
    )[0]

    artifact.write_bytes(b"changed")
    ok, reason = _validate_completed_shard(
        str(done),
        analysis_folder_path=str(tmp_path),
        plan_manifest_path=str(plan),
    )
    assert not ok
    assert "changed" in reason

    artifact.write_bytes(b"original")
    plan.write_text('{"changed_plan": true}', encoding="utf-8")
    ok, reason = _validate_completed_shard(
        str(done), analysis_folder_path=str(tmp_path), plan_manifest_path=str(plan)
    )
    assert not ok
    assert "changed" in reason


def test_infmovie_missing_output_is_not_valid() -> None:
    ok, reason, _details = _inspect_rendered_movie(Path("missing.mp4"))
    assert not ok
    assert "missing" in reason


def _write_training_feature(path: Path, values: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)
        handle.attrs["columns"] = np.asarray(["f1", "f2"], dtype="S")


def _write_training_labels(path: Path, behaviors: list[str]) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Behavior", data=np.asarray(behaviors, dtype=object), dtype=string_dtype)


def test_training_feature_validation_rejects_nonfinite_values(tmp_path) -> None:
    feature_path = tmp_path / "features.h5"
    _write_training_feature(feature_path, np.asarray([[1.0, 2.0], [3.0, np.nan]]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_feature_hdf5_file(feature_path, sequence_length=2)


def test_pretraining_metadata_requires_exact_unique_inventory(tmp_path) -> None:
    feature_path = tmp_path / "features.h5"
    _write_training_feature(feature_path, np.ones((3, 2), dtype=np.float32))
    metadata = pd.DataFrame({"filename": [feature_path.name, feature_path.name]})
    with pytest.raises(ValueError, match="duplicate filenames"):
        validate_pretraining_metadata(str(tmp_path), metadata, sequence_length=3)


def test_supervised_training_metadata_checks_label_length(tmp_path) -> None:
    feature_path = tmp_path / "features.h5"
    labels_path = tmp_path / "labels.h5"
    _write_training_feature(feature_path, np.ones((3, 2), dtype=np.float32))
    _write_training_labels(labels_path, ["walk", "rest"])
    metadata = pd.DataFrame(
        {"filename": [feature_path.name], "labels": [labels_path.name]}
    )
    with pytest.raises(ValueError, match="does not match feature rows"):
        validate_supervised_training_metadata(
            str(tmp_path),
            metadata,
            sequence_length=3,
            pose_pred=True,
            orient_pred=False,
            defined_split=False,
            noval=True,
        )


def test_supervised_training_metadata_rejects_path_escape(tmp_path) -> None:
    metadata = pd.DataFrame(
        {"filename": ["../outside.h5"], "labels": ["labels.h5"]}
    )
    with pytest.raises(ValueError, match="escapes the dataset directory"):
        validate_supervised_training_metadata(
            str(tmp_path),
            metadata,
            sequence_length=3,
            pose_pred=True,
            orient_pred=False,
            defined_split=False,
            noval=True,
        )
