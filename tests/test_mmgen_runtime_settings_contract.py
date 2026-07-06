from __future__ import annotations

import inspect
from pathlib import Path

from autobehaver.postinf.mmgen import core as mm_core
from autobehaver.postinf.mmgen import pipeline as mm_pipeline


class _FakeDirLock:
    def __init__(self, *args, **kwargs):
        self.acquired = False

    def acquire(self, metadata=None):
        self.acquired = True

    def release(self):
        self.acquired = False


def _make_cfg(tmp_path: Path, **overrides):
    inf = tmp_path / "inf"
    pos = tmp_path / "pos"
    out = tmp_path / "out"
    inf.mkdir()
    pos.mkdir()
    out.mkdir()
    return mm_core.MMGenConfig(
        inference_csv_dir=str(inf),
        poscalc_dir=str(pos),
        output_dir=str(out),
        **overrides,
    )


def test_run_post_inference_analysis_uses_runtime_settings_only() -> None:
    params = inspect.signature(mm_pipeline.run_post_inference_analysis).parameters
    assert "runtime_settings" in params
    assert "settings" not in params
    assert "emit_master_matrix" not in params
    assert "emit_full_outputs" not in params


def test_run_mmgen_forwards_runtime_settings_without_stale_output_kwargs(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_post(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mm_core.pipeline, "run_post_inference_analysis", _fake_run_post)

    cfg = _make_cfg(
        tmp_path,
        emit_master_matrix=True,
        emit_full_outputs=False,
        process_only=True,
        random_seed=7,
    )
    mm_core.run_mmgen(cfg)

    kwargs = captured["kwargs"]
    runtime_settings = kwargs["runtime_settings"]
    assert runtime_settings.emit_master_matrix is True
    assert runtime_settings.emit_full_outputs is False
    assert kwargs["process_only"] is True
    assert "emit_master_matrix" not in kwargs
    assert "emit_full_outputs" not in kwargs


def test_run_mmgen_merge_shards_forwards_runtime_settings_without_stale_output_kwargs(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_post(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    def _fake_merge(*, output_folder: str, num_shards: int, strict_missing: bool) -> None:
        captured["merge"] = {
            "output_folder": output_folder,
            "num_shards": num_shards,
            "strict_missing": strict_missing,
        }

    monkeypatch.setattr(mm_core.pipeline, "run_post_inference_analysis", _fake_run_post)
    monkeypatch.setattr(mm_core, "merge_shard_temp_stores", _fake_merge)
    monkeypatch.setattr(mm_core, "DirLock", _FakeDirLock)

    cfg = _make_cfg(
        tmp_path,
        emit_master_matrix=False,
        emit_full_outputs=True,
        merge_shards=True,
        num_shards=10,
    )
    mm_core.run_mmgen(cfg)

    assert captured["merge"] == {
        "output_folder": str(tmp_path / "out"),
        "num_shards": 10,
        "strict_missing": True,
    }
    kwargs = captured["kwargs"]
    runtime_settings = kwargs["runtime_settings"]
    assert runtime_settings.emit_master_matrix is False
    assert runtime_settings.emit_full_outputs is True
    assert kwargs["finalize_only"] is True
    assert kwargs["num_shards"] == 1
    assert "emit_master_matrix" not in kwargs
    assert "emit_full_outputs" not in kwargs
