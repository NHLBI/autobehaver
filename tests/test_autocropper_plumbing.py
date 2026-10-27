from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from autobehaver.preinf.autocropper.cli import (
    _config_from_args,
    build_arg_parser,
)
from autobehaver.preinf.autocropper.core import _sync_runtime_args
from autobehaver.preinf.autocropper.cropping import crop_single_chamber
from autobehaver.preinf.autocropper import pipeline as ac_pipeline


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def test_autocropper_cli_args_map_to_config_and_runtime_args() -> None:
    args = _parse_args(
        [
            "--input_dir",
            "/tmp/full_movies",
            "--file_filter",
            "movieA",
            "--no_require_exclusions",
            "--crop_crf",
            "23",
        ]
    )
    cfg = _config_from_args(args)

    assert cfg.file_filter == "movieA"
    assert cfg.require_exclusions is False
    assert cfg.crop_crf == 23

    _sync_runtime_args(cfg=cfg, args=args)

    assert args.file_filter == "movieA"
    assert args.require_exclusions is False
    assert args.crop_crf == 23
    assert args.crop_dim_2k == cfg.crop_dim_2k
    assert not hasattr(args, "crop_dim_4k")


def test_autocropper_cli_defaults_match_runtime_defaults() -> None:
    from autobehaver.preinf.autocropper.config import AutocropperConfig

    defaults = AutocropperConfig(input_dir="1_full_movies")
    args = _parse_args(["--input_dir", defaults.input_dir])

    assert _config_from_args(args) == defaults


def test_standard_geometry_preserves_existing_chamber_assignments() -> None:
    from autobehaver.preinf.autocropper.detection import calculate_theoretical_centers

    expected = {
        "chamber1": (373, 449), "chamber2": (996, 443), "chamber3": (1622, 440),
        "chamber4": (378, 1073), "chamber5": (1001, 1070), "chamber6": (1626, 1064),
    }
    for width in (2016, 2028):
        assert calculate_theoretical_centers(width, 1520, 3, 2) == expected


def test_nonstandard_geometry_cannot_be_enabled_by_calibration() -> None:
    import pytest
    from autobehaver.preinf.autocropper.detection import calculate_theoretical_centers

    centers = calculate_theoretical_centers(2016, 1520, 3, 2)
    for width, height in ((4032, 3040), (4056, 3040), (3840, 2160), (3000, 2000)):
        calibration = {"image_width": width, "image_height": height, "centers": centers,
                       "max_center_distance_px": 10}
        with pytest.raises(ValueError, match="standard recording sizes"):
            calculate_theoretical_centers(width, height, 3, 2, calibration)


def test_measured_centers_remain_supported_at_standard_movie_size() -> None:
    import pytest
    from autobehaver.preinf.autocropper.detection import calculate_theoretical_centers

    centers = calculate_theoretical_centers(2016, 1520, 3, 2)
    measured = {key: (x + 3, y - 2) for key, (x, y) in centers.items()}
    calibration = {"image_width": 2016, "image_height": 1520, "centers": measured,
                   "max_center_distance_px": 10}
    assert calculate_theoretical_centers(2016, 1520, 3, 2, calibration) == measured
    with pytest.raises(ValueError, match="resolution does not match"):
        calculate_theoretical_centers(2028, 1520, 3, 2, calibration)


def test_autocropper_select_movie_files_respects_canonical_file_filter(monkeypatch) -> None:
    ac_pipeline.args = argparse.Namespace(file_filter="movie_b", movie_list=None)
    ac_pipeline.input_dir = Path("/tmp/full_movies")

    movies = [
        Path("/tmp/full_movies/movie_a.mp4"),
        Path("/tmp/full_movies/movie_b_trial1.mp4"),
        Path("/tmp/full_movies/movie_c.mp4"),
    ]

    selected = ac_pipeline._select_movie_files(movies)
    assert selected == [Path("/tmp/full_movies/movie_b_trial1.mp4")]


def test_crop_single_chamber_uses_crf_and_timeout(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_get_video_dimensions(video_path: str | Path) -> tuple[int, int]:
        return (1200, 900)

    def fake_get_ffmpeg_executable() -> str:
        return "/usr/bin/ffmpeg"

    def fake_run(cmd, *, total_frames, desc, timeout):
        calls["cmd"] = list(cmd)
        calls["timeout"] = timeout
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake mp4 bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "autobehaver.preinf.autocropper.cropping._get_video_dimensions",
        fake_get_video_dimensions,
    )
    monkeypatch.setattr(
        "autobehaver.preinf.autocropper.cropping._get_ffmpeg_executable",
        fake_get_ffmpeg_executable,
    )
    monkeypatch.setattr(
        "autobehaver.preinf.autocropper.cropping.run_ffmpeg",
        fake_run,
    )

    monkeypatch.setattr("autobehaver.preinf.autocropper.cropping.valid_crop", lambda *a, **k: True)
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"not a real video")
    output_video = tmp_path / "cropped.mp4"

    result = crop_single_chamber(
        str(input_video),
        output_video,
        x=500,
        y=400,
        crop_dim=600,
        timeout=17,
        crf=23,
    )

    assert result == str(output_video)
    assert output_video.exists()
    assert calls["timeout"] == 17
    assert "-crf" in calls["cmd"]
    crf_idx = calls["cmd"].index("-crf")
    assert calls["cmd"][crf_idx + 1] == "23"


def test_autocropper_trial_number_selection_is_exact() -> None:
    ac_pipeline.args = argparse.Namespace(
        file_filter=None,
        movie_list=None,
        trial_number=1,
    )
    ac_pipeline.input_dir = Path("/tmp/full_movies")

    movies = [
        Path("/tmp/full_movies/exp_sub_trial1_rpi-1_movie0_20260101.mp4"),
        Path("/tmp/full_movies/exp_sub_trial10_rpi-1_movie0_20260101.mp4"),
        Path("/tmp/full_movies/exp_sub_trial14_rpi-1_movie0_20260101.mp4"),
    ]

    selected = ac_pipeline._select_movie_files(movies)
    assert selected == [movies[0]]


def test_autocropper_trial_substring_filter_warns_when_ambiguous(caplog) -> None:
    ac_pipeline.args = argparse.Namespace(
        file_filter="trial1",
        movie_list=None,
        trial_number=None,
    )
    ac_pipeline.input_dir = Path("/tmp/full_movies")

    movies = [
        Path("/tmp/full_movies/exp_sub_trial1_rpi-1_movie0_20260101.mp4"),
        Path("/tmp/full_movies/exp_sub_trial10_rpi-1_movie0_20260101.mp4"),
        Path("/tmp/full_movies/exp_sub_trial14_rpi-1_movie0_20260101.mp4"),
    ]

    with caplog.at_level("WARNING"):
        selected = ac_pipeline._select_movie_files(movies)

    assert selected == movies
    assert "substring match" in caplog.text
    assert "--trial_number 1" in caplog.text


def test_crop_single_chamber_limits_ffmpeg_threads(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_get_video_dimensions(video_path: str | Path) -> tuple[int, int]:
        raise AssertionError("source_size should avoid reopening the video")

    def fake_get_ffmpeg_executable() -> str:
        return "/usr/bin/ffmpeg"

    def fake_run(cmd, *, total_frames, desc, timeout):
        calls["cmd"] = list(cmd)
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"fake mp4 bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "autobehaver.preinf.autocropper.cropping._get_video_dimensions",
        fake_get_video_dimensions,
    )
    monkeypatch.setattr(
        "autobehaver.preinf.autocropper.cropping._get_ffmpeg_executable",
        fake_get_ffmpeg_executable,
    )
    monkeypatch.setattr(
        "autobehaver.preinf.autocropper.cropping.run_ffmpeg",
        fake_run,
    )

    output_video = tmp_path / "cropped.mp4"
    monkeypatch.setattr("autobehaver.preinf.autocropper.cropping.valid_crop", lambda *a, **k: True)
    result = crop_single_chamber(
        "input.mp4",
        output_video,
        x=500,
        y=400,
        crop_dim=600,
        timeout=17,
        crf=23,
        ffmpeg_threads=1,
        source_size=(1200, 900),
    )

    assert result == str(output_video)
    cmd = calls["cmd"]
    assert "-nostdin" in cmd
    assert "-threads" in cmd
    assert "-filter_threads" in cmd
    assert "-x264-params" in cmd
    assert "threads=1" in cmd
    assert output_video.exists()


def test_delete_incomplete_files_is_scoped_and_age_gated(tmp_path: Path) -> None:
    from autobehaver.preinf.autocropper.io_utils import delete_incomplete_files
    import os
    import time

    target_partial = tmp_path / "label_exp_sub_trial1_rpi-1_movie0_20260101_chamber1.mp4.pid1.partial"
    other_partial = tmp_path / "label_exp_sub_trial2_rpi-1_movie0_20260101_chamber1.mp4.pid1.partial"
    recent_partial = tmp_path / "label_exp_sub_trial1_rpi-2_movie0_20260101_chamber1.mp4.pid1.partial"

    for path in [target_partial, other_partial, recent_partial]:
        path.write_bytes(b"partial")

    old_time = time.time() - 7200
    os.utime(target_partial, (old_time, old_time))
    os.utime(other_partial, (old_time, old_time))

    delete_incomplete_files(
        tmp_path,
        movie_stems=["exp_sub_trial1_rpi-1_movie0_20260101"],
        min_partial_age_seconds=3600,
    )

    assert not target_partial.exists()
    assert other_partial.exists()
    assert recent_partial.exists()
