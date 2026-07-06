from __future__ import annotations

import argparse
import json
from pathlib import Path

from autobehaver.preinf.autocropper import pipeline as ac_pipeline
from autobehaver.preinf.autocropper import registry_update as ac_registry_update
from autobehaver.preinf.autocropper.io_utils import summarize_results


def test_process_single_movie_marks_incomplete_when_required_crops_are_missing(
    monkeypatch, tmp_path: Path
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "cropped"
    output_dir.mkdir()

    movie = input_dir / "exp_subexp_trial1_rpi-1_movie0_20260101.mp4"
    movie.write_bytes(b"not a real movie")

    ac_pipeline.args = argparse.Namespace(
        chamber_columns=3,
        chamber_rows=2,
        skip_labeling=True,
        down_factor=2,
        approx_radius=10,
        max_workers=1,
    )
    ac_pipeline.input_dir = input_dir
    ac_pipeline.run_dir = run_dir
    ac_pipeline.crop_output_dir = output_dir
    ac_pipeline.recording_db_path = None
    ac_pipeline.chamber_validator = None

    monkeypatch.setattr(ac_pipeline, "ensure_user_permissions", lambda *a, **k: None)
    monkeypatch.setattr(ac_pipeline, "_excluded_set_for_movie", lambda _movie: set())
    monkeypatch.setattr(
        ac_pipeline,
        "_parse_full_movie_identity",
        lambda _movie: ("exp", "subexp", 1, 1),
    )
    monkeypatch.setattr(
        ac_pipeline,
        "all_cropped_movies_exist",
        lambda _movie, _output_dir, excluded_chambers=None: False,
    )
    monkeypatch.setattr(
        ac_pipeline,
        "detect_and_match_chambers",
        lambda _movie, _args: ({"chamber1": (10, 20)}, object(), [], 100),
    )
    monkeypatch.setattr(
        ac_pipeline,
        "run_chamber_validation",
        lambda _movie, allowed_coords, _validator, _crop_dim: (allowed_coords, {}),
    )
    monkeypatch.setattr(ac_pipeline, "draw_circles", lambda *a, **k: None)
    monkeypatch.setattr(ac_pipeline, "save_chamber_coordinates_image", lambda *a, **k: None)

    def fake_resolve_existing_crop(
        _movie: Path,
        _output_dir: Path,
        chamber_id: str,
        *,
        allow_rename_unlabeled: bool = True,
    ) -> dict[str, object]:
        return {
            "status": "missing",
            "expected_name": f"{movie.stem}_{chamber_id}.mp4",
            "expected_path": output_dir / f"{movie.stem}_{chamber_id}.mp4",
            "path": None,
            "matches": [],
        }

    monkeypatch.setattr(ac_pipeline, "_resolve_existing_crop", fake_resolve_existing_crop)
    monkeypatch.setattr(
        ac_pipeline,
        "crop_chambers_in_parallel",
        lambda *_a, **_k: {"chamber1": "failed: synthetic crop failure"},
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        ac_pipeline,
        "log_movie_results",
        lambda _movie_path, _run_dir, movie_data: captured.update(movie_data),
    )

    ac_pipeline.process_single_movie(movie)

    assert captured["status"] == "incomplete"
    assert captured["missing_required_chambers"] == ["chamber1"]
    assert captured["crop_failures"] == {"chamber1": "failed: synthetic crop failure"}
    assert captured["cropped_files"] == {}
    assert "error" not in captured


def test_summarize_results_filters_to_current_run_and_tolerates_missing_fields(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_file = run_dir / "autocropper_created_processing_log.txt"
    log_file.write_text("autocropper header\n")

    movie_logs_dir = run_dir / "movie_logs"
    movie_logs_dir.mkdir()

    (movie_logs_dir / "selected_log.json").write_text(
        json.dumps(
            {
                "movie_name": "selected.mp4",
                "status": "skipped",
                "skip_reason": "already processed",
                "cropped_files": {"chamber1": "selected_chamber1.mp4"},
            }
        )
    )
    (movie_logs_dir / "stale_log.json").write_text(
        json.dumps(
            {
                "movie_name": "stale.mp4",
                "status": "completed",
                "chamber_coordinates": {"chamber1": {"x": 10, "y": 20}},
                "cropped_files": {"chamber1": "stale_chamber1.mp4"},
            }
        )
    )

    summarize_results(run_dir, movie_names=["selected.mp4"])

    summary_text = log_file.read_text()
    assert "Movie: selected.mp4" in summary_text
    assert "Movie: stale.mp4" not in summary_text
    assert "Chamber Coordinates: not available" in summary_text
    assert "Summary Counts:" in summary_text


def test_registry_update_filters_to_current_run_movie_names(
    monkeypatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    input_movies_dir = project_root / "1_full_movies"
    input_movies_dir.mkdir()
    cropped_movies_dir = project_root / "2_cropped_movies"
    cropped_movies_dir.mkdir()
    run_dir = project_root / "2_cropped_movies"
    movie_logs_dir = run_dir / "movie_logs"
    movie_logs_dir.mkdir(parents=True, exist_ok=True)

    selected_movie = "selected_movie.mp4"
    stale_movie = "stale_movie.mp4"

    (input_movies_dir / selected_movie).write_bytes(b"x")
    (input_movies_dir / stale_movie).write_bytes(b"x")
    (cropped_movies_dir / "selected_movie_chamber1.mp4").write_bytes(b"x")
    (cropped_movies_dir / "stale_movie_chamber1.mp4").write_bytes(b"x")

    (movie_logs_dir / "selected_movie_log.json").write_text(
        json.dumps(
            {
                "movie_name": selected_movie,
                "cropped_files": {"chamber1": "selected_movie_chamber1.mp4"},
            }
        )
    )
    (movie_logs_dir / "stale_movie_log.json").write_text(
        json.dumps(
            {
                "movie_name": stale_movie,
                "cropped_files": {"chamber1": "stale_movie_chamber1.mp4"},
            }
        )
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(ac_registry_update, "extract_label_prefix", lambda **_k: "")
    monkeypatch.setattr(
        ac_registry_update,
        "parse_genotype_sex_from_prefix",
        lambda _prefix: (None, None),
    )
    monkeypatch.setattr(
        ac_registry_update,
        "upsert_registry_rows",
        lambda csv_path, rows, lock_timeout_sec, lock_stale_sec: captured.update(
            {
                "csv_path": csv_path,
                "rows": rows,
                "lock_timeout_sec": lock_timeout_sec,
                "lock_stale_sec": lock_stale_sec,
            }
        ),
    )

    ac_registry_update.update_registry_from_autocropper_logs(
        registry_csv=project_root / "registry.csv",
        project_root=project_root,
        input_movies_dir=input_movies_dir,
        cropped_movies_dir=cropped_movies_dir,
        run_dir=run_dir,
        recording_database_csv=None,
        movie_names=[selected_movie],
    )

    rows = captured["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["recording_id"] == "selected_movie_chamber1"
    assert rows[0]["cropped_movie_relpath"].endswith("selected_movie_chamber1.mp4")
