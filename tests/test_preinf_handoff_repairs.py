"""Tiny native boundaries for selected inputs, current labels and safe reuse."""
import argparse
import csv
import json
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest


def _video(path, frames=6, size=32):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 40, (size, size))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((size, size, 3), i * 20, dtype=np.uint8))
    writer.release()


def test_optional_movie_list_is_distinct_from_default_folder(monkeypatch, tmp_path):
    from autobehaver.preinf.autocropper import pipeline
    movie = tmp_path / "exp_trial1_rpi-1_movie0_x.mp4"
    movie.touch()
    monkeypatch.setattr(pipeline, "input_dir", tmp_path)
    monkeypatch.setattr(pipeline, "args", SimpleNamespace(movie_list=None, file_filter=None))
    assert pipeline._select_movie_files([movie]) == [movie]
    empty = tmp_path / "empty.txt"
    empty.write_text("# no selected recordings\n")
    pipeline.args.movie_list = str(empty)
    assert pipeline._select_movie_files([movie]) == []
    pipeline.args.movie_list = str(tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError, match="Supplied movie_list"):
        pipeline._select_movie_files([movie])
    empty.write_text("absent.mp4\n")
    pipeline.args.movie_list = str(empty)
    with pytest.raises(FileNotFoundError, match="Movie listed"):
        pipeline._select_movie_files([movie])


@pytest.mark.parametrize("problem", ["missing_column", "invalid_flag", "conflicting_duplicate", "duplicate_header"])
def test_malformed_exclusions_identify_the_problem(tmp_path, problem):
    from autobehaver.preinf.moviechecker.exclusions import ExclusionsStore, EXCLUSIONS_HEADER
    path = tmp_path / "exclusions.csv"
    fields = list(EXCLUSIONS_HEADER)
    row = dict(zip(fields, ["exp_trial1_rpi-1_movie", "movie.mp4", *([0] * 6), 1]))
    rows = [row]
    if problem == "missing_column":
        fields.remove("exclude_ch6")
        row.pop("exclude_ch6")
    elif problem == "invalid_flag":
        row["exclude_ch2"] = "maybe"
    elif problem == "conflicting_duplicate":
        rows.append({**row, "exclude_ch4": 1})
    else:
        fields.append("exclude_ch2")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="exclusions.csv"):
        ExclusionsStore(str(path)).load()


def test_unchanged_duplicate_exclusions_are_unambiguous(tmp_path):
    from autobehaver.preinf.moviechecker.exclusions import ExclusionsStore, EXCLUSIONS_HEADER
    path = tmp_path / "exclusions.csv"
    row = dict(zip(EXCLUSIONS_HEADER, ["pair", "movie0.mp4", *([0] * 6), 1]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXCLUSIONS_HEADER)
        writer.writeheader()
        writer.writerows([row, {**row, "reviewed_movie": "movie1.mp4", "updated_at": 2}])
    store = ExclusionsStore(str(path))
    store.load()
    assert store.get("pair").exclude_ch == [0] * 6


def test_old_excluded_crop_stays_excluded_after_stage_transfer_and_subset_update(tmp_path):
    from autobehaver.portable_paths import record_original_name
    from autobehaver.preinf.autocropper.selection import write_selection, read_selection, filter_included_paths, propagate_selection
    crops = tmp_path / "crops"
    crops.mkdir()
    key = "exp_trial1_rpi-1_movie0_stamp_chamber1"
    other_key = "exp_trial2_rpi-1_movie0_stamp_chamber1"
    old = crops / "short.mp4"
    old.touch()
    record_original_name(old, "A_FEMALE_" + key + ".mp4")
    write_selection(crops, {key: {"included": False}, other_key: {"included": True}})
    assert filter_included_paths([old]) == []
    dlc = tmp_path / "dlc"
    dlc.mkdir()
    old_dlc = dlc / "shortDLCtag.csv"
    old_dlc.touch()
    record_original_name(old_dlc, "A_FEMALE_" + key + "DLCtag.csv")
    propagate_selection([crops], dlc)
    assert filter_included_paths([old_dlc]) == []
    write_selection(crops, {key: {"included": True}})
    assert read_selection(crops)[other_key]["included"] is True
    propagate_selection([crops], dlc)
    assert filter_included_paths([old_dlc]) == [old_dlc]
    assert old.exists()
    write_selection(crops, {key: {"included": True, "label_prefix": "B_FEMALE_"}})
    propagate_selection([crops], dlc)
    assert filter_included_paths([old_dlc]) == []
    current = dlc / "current.csv"
    current.touch()
    record_original_name(current, "B_FEMALE_" + key + "DLCtag.csv")
    assert filter_included_paths([old_dlc, current]) == [current]


def test_registry_deactivates_old_excluded_chamber_and_current_label_wins(tmp_path):
    from autobehaver.registry import upsert_registry_rows, read_registry
    from autobehaver.preinf.autocropper.registry_update import update_registry_from_autocropper_logs
    movie = "exp_trial1_rpi-1_movie0_stamp.mp4"
    crops = tmp_path / "crops"
    crops.mkdir()
    crop = crops / ("A_MALE_" + Path(movie).stem + "_chamber1.mp4")
    crop.touch()
    logs = tmp_path / "run" / "movie_logs"
    logs.mkdir(parents=True)
    payload = {"movie_name": movie, "cropped_files": {"chamber1": crop.name},
               "chamber_labels": {"chamber1": "B_FEMALE_"}, "excluded_chambers": ["chamber2"]}
    (logs / "movie_log.json").write_text(json.dumps(payload))
    registry = tmp_path / "registry.csv"
    excluded_id = Path(movie).stem + "_chamber2"
    upsert_registry_rows(csv_path=registry, rows=[{"recording_id": excluded_id, "active": True}])
    update_registry_from_autocropper_logs(registry_csv=registry, project_root=tmp_path,
        input_movies_dir=tmp_path, cropped_movies_dir=crops, run_dir=logs.parent,
        recording_database_csv=None, movie_names=[movie])
    table = read_registry(registry).set_index("recording_id")
    assert str(table.loc[excluded_id, "active"]).lower() in {"false", "0"}
    assert table.loc[Path(movie).stem + "_chamber1", "genotype"] == "B"
    assert table.loc[Path(movie).stem + "_chamber1", "sex"] == "FEMALE"


def test_label_correction_republishes_identical_crop_without_recropping(monkeypatch, tmp_path):
    from autobehaver.preinf.autocropper import pipeline
    from autobehaver.preinf.autocropper.contract import crop_request
    from autobehaver.preinf.autocropper.recording_db import load_recording_database_index
    from autobehaver.completion import write_receipt, file_digest, receipt_matches
    source = tmp_path / "exp_trial1_rpi-1_movie0_stamp.mp4"
    _video(source)
    old = tmp_path / ("A_MALE_" + source.stem + "_chamber1.mp4")
    _video(old)
    args = SimpleNamespace(skip_labeling=False, chamber_columns=1, chamber_rows=1)
    monkeypatch.setattr(pipeline, "args", args)
    index_path = tmp_path / "recording_database.csv"
    pd.DataFrame([{"experiment": "exp", "subexperiment": "", "trial_number": 1,
                   "camera_mappings": "rpi-1_chamber1:B_FEMALE_"}]).to_csv(index_path, index=False)
    monkeypatch.setattr(pipeline, "recording_db_index", load_recording_database_index(index_path))
    pipeline._movie_requests.clear()
    pipeline._movie_shapes.clear()
    identity = crop_request(source, "chamber1", args)
    write_receipt(old, identity)
    digest = file_digest(old)
    result = pipeline._resolve_existing_crop(source, tmp_path, "chamber1")
    assert result["status"] == "expected"
    assert result["path"].name.startswith("B_FEMALE_")
    assert file_digest(result["path"]) == digest
    assert receipt_matches(result["path"], identity)
    assert not old.exists()
    assert len(list((tmp_path / "_old").glob("*.mp4"))) == 1


def test_original_name_filters_use_the_full_path(monkeypatch, tmp_path):
    from autobehaver.portable_paths import record_original_name
    from autobehaver.preinf.autocropper import pipeline
    from autobehaver.preinf.dlc.core import resolve_videos
    video = tmp_path / "compact.mp4"
    video.touch()
    record_original_name(video, "group_trial14_rpi-1_movie0_stamp.mp4")
    monkeypatch.setattr(pipeline, "input_dir", tmp_path)
    monkeypatch.setattr(pipeline, "args", SimpleNamespace(movie_list=None, file_filter="trial14"))
    assert pipeline._select_movie_files([video]) == [video]
    assert resolve_videos(str(tmp_path), video_glob="*trial14*") == [str(video)]


def test_dlc_failed_regeneration_and_skip_analysis_cannot_accept_old_model(monkeypatch, tmp_path):
    from autobehaver.preinf.dlc import core
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    movies = tmp_path / "movies"
    movies.mkdir()
    _video(movies / "a.mp4")
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text("bodyparts: [a]\nproject_path: " + tmp_path.as_posix() + "\n")
    model = {"value": "old"}
    should_fail = {"value": False}
    calls = []
    def analyze(config_path, videos, **kwargs):
        calls.extend(videos)
        if should_fail["value"]:
            raise RuntimeError("intentional tiny backend failure")
        directory = Path(kwargs["destfolder"])
        columns = pd.MultiIndex.from_tuples([("scorer", "a", item) for item in ("x", "y", "likelihood")])
        frame = pd.DataFrame(np.full((6, 3), .8), columns=columns)
        for source in videos:
            stem = Path(source).stem + "DLCfixture"
            frame.to_csv(directory / (stem + ".csv"))
            frame.to_hdf(directory / (stem + ".h5"), key="df_with_missing", format="table")
            (directory / (stem + "_meta.pickle")).write_bytes(b"fixture")
    monkeypatch.setattr(core, "_require_deeplabcut", lambda: SimpleNamespace(analyze_videos=analyze))
    monkeypatch.setattr(core, "setup_gpu", lambda: None)
    monkeypatch.setattr(core, "model_identity", lambda *a: {"settings": {"bodyparts": ["a"]}, "model": model["value"]})
    cfg = DLCInferencerConfig(input_dir=str(movies), output_dir=str(output), dlc_config=str(config), max_retries=1,
                              scratch_dir=str(tmp_path / "scratch"))
    assert core.run_dlc(cfg) == 0
    old_csv = next(output.glob("*.csv")).read_bytes()
    model["value"] = "new"
    should_fail["value"] = True
    assert core.run_dlc(cfg) == 1
    assert next(output.glob("*.csv")).read_bytes() == old_csv
    validation = json.loads((output / "DLC_VALIDATION.json").read_text())
    assert validation["status"] == "failed"
    attempted = len(calls)
    cfg = replace(cfg, skip_analysis=True)
    assert core.run_dlc(cfg) == 1
    assert len(calls) == attempted
    model["value"] = "old"
    assert core.run_dlc(cfg) == 0


@pytest.mark.parametrize("change", ["finite_value", "shifted_frames", "changed_frame_count", "missing_receipt"])
def test_normalized_reuse_binds_published_values_and_exact_timeline(monkeypatch, tmp_path, change):
    from autobehaver.preinf.keyprenorm import normalize
    from autobehaver.preinf.keyprenorm.inspect_outputs import inspect_normalized_h5
    from autobehaver.preinf.keyprenorm.identity import normalization_identity
    from autobehaver.portable_paths import normalized_path
    source = tmp_path / "a_positional_calcs.csv"
    pd.DataFrame({"x": np.arange(12), "y": np.arange(12) * 2}).to_csv(source, index=False)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(normalize, "setup_worker_logging", lambda *a: None)
    means, stds = np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    normalize.init_normalize_worker(["x", "y"], means, stds, 2, str(out), None, None, None, "INFO")
    assert normalize.normalize_one_csv(str(source))["status"] == "success"
    path = normalized_path(source, out)
    identity = normalization_identity(source, ["x", "y"], means, stds, 2)
    kwargs = dict(expected_n_cols=2, expected_rows=8, expected_columns=["x", "y"],
                  expected_identity=identity, expected_source_rows=12, expected_edge_skip=2)
    assert inspect_normalized_h5(str(path), **kwargs)[0]
    with h5py.File(path, "r") as hf:
        assert hf.attrs["source_positional_sha256"] == json.loads(identity)["source"]
    if change == "missing_receipt":
        Path(str(path) + ".done.json").unlink()
    else:
        with h5py.File(path, "r+") as hf:
            if change == "finite_value":
                hf["data"][0, 0] = 500
            elif change == "shifted_frames":
                hf["source_frame_indices"][:] += 1
            else:
                hf.attrs["original_frame_count"] = 13
    ok, rows, reason = inspect_normalized_h5(str(path), **kwargs)
    assert not ok
    assert rows == 8
    assert "mismatch" in reason or "missing" in reason


def test_run_manifests_keep_every_input_not_only_examples(tmp_path):
    from autobehaver.preinf.keypre.manifest import _build_run_manifest
    from autobehaver.preinf.keyprenorm.manifest import build_run_manifest
    inputs = [str(tmp_path / f"movie{i}.csv") for i in range(12)]
    args = argparse.Namespace()
    _, keypre = _build_run_manifest(args=args, input_folder=str(tmp_path), csv_source_folder=str(tmp_path),
        out_dir=str(tmp_path), csv_files=inputs, start_time_iso=datetime.now().isoformat())
    _, norm = build_run_manifest(out_dir=str(tmp_path), args=args, input_folder=str(tmp_path), csv_glob="*.csv",
        norm_params_path=str(tmp_path / "norm.csv"), start_time=datetime.now(), all_csv_files=inputs, stage_version="test")
    assert keypre["discovery"]["csv_files"] == inputs
    assert norm["discovery"]["csv_files"] == inputs
    assert len(keypre["discovery"]["example_csv_files"]) == 10
