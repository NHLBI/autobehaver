from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from autobehaver.preinf.autocropper.cli import (
    _config_from_args,
    _load_config_defaults,
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
            "--no_validate_chambers",
        ]
    )
    cfg = _config_from_args(args)

    assert cfg.file_filter == "movieA"
    assert cfg.require_exclusions is False
    assert cfg.crop_crf == 23
    assert cfg.skip_validation is True

    _sync_runtime_args(cfg=cfg, args=args)

    assert args.file_filter == "movieA"
    assert args.require_exclusions is False
    assert args.crop_crf == 23
    assert args.skip_validation is True
    assert args.validate_chambers is False
    assert args.crop_dim_4k == cfg.crop_dim_2k * 2


def test_autocropper_config_defaults_support_canonical_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "autocropper": {
                    "input_dir": "1_full_movies",
                    "file_filter": "movie_token",
                    "require_exclusions": False,
                    "crop_crf": 21,
                }
            }
        )
    )

    defaults = _load_config_defaults(str(cfg_path))

    assert defaults["input_dir"] == str((tmp_path / "1_full_movies").resolve())
    assert defaults["file_filter"] == "movie_token"
    assert defaults["require_exclusions"] is False
    assert defaults["crop_crf"] == 21


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

    def fake_run(cmd, check, stdout, stderr, text, timeout):
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
        "autobehaver.preinf.autocropper.cropping.subprocess.run",
        fake_run,
    )

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

    def fake_run(cmd, check, stdout, stderr, text, timeout):
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
        "autobehaver.preinf.autocropper.cropping.subprocess.run",
        fake_run,
    )

    output_video = tmp_path / "cropped.mp4"
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
