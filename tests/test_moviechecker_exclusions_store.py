from __future__ import annotations

import os
from pathlib import Path

from autobehaver.preinf.moviechecker.core import MovieCheckerApp
from autobehaver.preinf.moviechecker.exclusions import ExclusionsStore


def test_moviechecker_scan_ignores_exclusion_csv_files(tmp_path: Path) -> None:
    (tmp_path / "trial_rpi-1_movie0_20250101120000AM.mp4").write_bytes(b"not a real movie")
    (tmp_path / "trial_rpi-1_movie1_20250101120000AM.mp4").write_bytes(b"not a real movie")
    (tmp_path / "exclusions.csv").write_text("pair_key,reviewed_movie\n")
    (tmp_path / "exclusions_abcdef.csv").write_text("pair_key,reviewed_movie\n")

    app = MovieCheckerApp(
        input_dir=str(tmp_path),
        exclusions_csv_path=str(tmp_path / "missing_exclusions.csv"),
    )
    app.scan_movies()

    assert app.review_movies == [str(tmp_path / "trial_rpi-1_movie0_20250101120000AM.mp4")]
    assert all(Path(p).suffix == ".mp4" for p in app.review_movies)


def test_exclusions_write_atomic_retries_replace(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "exclusions.csv"
    store = ExclusionsStore(str(csv_path))
    store.upsert("pair-a", "movie0.mp4", [1, 0, 0, 0, 0, 0])

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("simulated transient lock")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    store.write_atomic(attempts=3, delay_s=0)

    assert calls["n"] == 3
    assert csv_path.exists()
    text = csv_path.read_text()
    assert "pair-a" in text
    assert "movie0.mp4" in text
    assert not list(tmp_path.glob("exclusions_*.csv"))


def test_exclusions_write_atomic_falls_back_to_direct_write(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "exclusions.csv"
    store = ExclusionsStore(str(csv_path))
    store.upsert("pair-b", "movie1.mp4", [0, 1, 0, 0, 0, 0])

    def blocked_replace(src: str, dst: str) -> None:
        raise PermissionError("simulated share rejects replace")

    monkeypatch.setattr(os, "replace", blocked_replace)
    store.write_atomic(attempts=2, delay_s=0)

    assert csv_path.exists()
    text = csv_path.read_text()
    assert "pair-b" in text
    assert "movie1.mp4" in text
    assert not list(tmp_path.glob("exclusions_*.csv"))
