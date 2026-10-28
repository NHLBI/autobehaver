from __future__ import annotations

import json

import pandas as pd
import pytest

from autobehaver.atomic_io import atomic_write_dataframe_csv, atomic_write_json, sha256_file
from autobehaver.inference import _build_inference_provenance, _completed_inference_matches
from autobehaver.registry import RegistryValidationError, read_registry, upsert_registry_rows


def test_atomic_writers_leave_no_temporary_files(tmp_path):
    csv_path = tmp_path / "table.csv"
    json_path = tmp_path / "metadata.json"

    atomic_write_dataframe_csv(pd.DataFrame({"value": [1, 2]}), csv_path, index=False)
    atomic_write_json({"status": "complete"}, json_path)

    assert pd.read_csv(csv_path)["value"].tolist() == [1, 2]
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_failed_atomic_csv_write_preserves_previous_file(monkeypatch, tmp_path):
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("value\nold\n", encoding="utf-8")

    def _fail_write(self, destination, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _fail_write)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_dataframe_csv(pd.DataFrame({"value": ["new"]}), csv_path, index=False)

    assert csv_path.read_text(encoding="utf-8") == "value\nold\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_registry_write_has_matching_completion_provenance(tmp_path):
    registry = tmp_path / "registry.csv"
    upsert_registry_rows(csv_path=registry, rows=[{"recording_id": "recording-1"}])

    provenance_path = registry.with_suffix(".csv.provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema"] == "autobehaver.registry.provenance.v1"
    assert provenance["status"] == "complete"
    assert provenance["registry"]["row_count"] == 1
    assert provenance["registry"]["sha256"] == sha256_file(registry)


def test_missing_registry_is_valid_but_corrupt_existing_registry_is_not(tmp_path):
    registry = tmp_path / "registry.csv"
    assert read_registry(registry).empty

    registry.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(RegistryValidationError, match="could not be read"):
        read_registry(registry)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"recording_id": "same"}, {"recording_id": "same"}], "duplicate recording_id"),
        ([{"recording_id": "a", "active": "possibly"}], "unknown active"),
        ([{"recording_id": "a", "cropped_movie_relpath": "../escape.mp4"}], "parent-traversing"),
        ([{"recording_id": "a", "chamber_number": 7}], "chamber_number"),
    ],
)
def test_registry_rejects_ambiguous_or_unsafe_rows(tmp_path, rows, message):
    registry = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    with pytest.raises(RegistryValidationError, match=message):
        read_registry(registry)


def test_inference_completion_manifest_detects_changed_outputs(tmp_path):
    source = tmp_path / "source.h5"
    checkpoint = tmp_path / "model.pth"
    output = tmp_path / "sample_inference.csv"
    import h5py
    with h5py.File(source, "w") as handle:
        handle['data'] = [[1.0]]
    checkpoint.write_bytes(b"checkpoint")
    output.write_text("frame,label\n1,rest\n", encoding="utf-8")

    expected = _build_inference_provenance(str(source), [str(checkpoint)], {"inf_average": 1})
    manifest = {
        **expected,
        "status": "complete",
        "outputs": [{"path": output.name, "sha256": sha256_file(output)}],
    }
    manifest_path = tmp_path / "sample_inference.csv.provenance.json"
    atomic_write_json(manifest, manifest_path)

    assert _completed_inference_matches(str(manifest_path), expected)
    output.write_text("frame,label\n1,walk\n", encoding="utf-8")
    assert not _completed_inference_matches(str(manifest_path), expected)
