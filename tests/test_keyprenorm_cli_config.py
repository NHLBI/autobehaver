from __future__ import annotations

from pathlib import Path

def test_keyprenorm_cli_options_reach_runtime(monkeypatch, tmp_path: Path) -> None:
    """Explicit options reach normalization unchanged without loading YAML."""

    # Capture settings without executing normalization.
    input_dir = tmp_path / "positional_calcs"
    input_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def _fake_run_keyprenorm(*, cfg, args) -> None:  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        captured["args"] = args

    from autobehaver.preinf.keyprenorm import cli as keyprenorm_cli

    monkeypatch.setattr(keyprenorm_cli, "run_keyprenorm", _fake_run_keyprenorm)

    keyprenorm_cli.main([
        "--input-dir", str(input_dir), "--output-dir", str(tmp_path / "out"),
        "--pattern", "*_positional_calcs.csv", "--file-filter", "trial",
        "--clobber", "--calculate-norm", "--edge-skip", "7", "--num-workers", "3",
    ])

    assert "cfg" in captured
    kcfg = captured["cfg"]

    # Path mapping
    assert str(kcfg.input_dir) == str(input_dir)
    assert str(kcfg.output_dir) == str(tmp_path / "out")

    # Basic passthrough
    assert kcfg.pattern == "*_positional_calcs.csv"
    assert kcfg.file_filter == "trial"
    assert kcfg.calculate_norm is True
    assert kcfg.edge_skip == 7
    assert kcfg.num_workers == 3

    assert kcfg.clobber is True
