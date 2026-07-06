from __future__ import annotations

from pathlib import Path

import yaml


def test_keyprenorm_cli_supports_pipeline_config(monkeypatch, tmp_path: Path) -> None:
    """Smoke test: keyprenorm CLI accepts --config and maps keyprenorm: defaults.

    This ensures the README and pipeline template config are accurate.
    """

    # Create a minimal input folder (does not need real CSVs because we monkeypatch run_keyprenorm).
    input_dir = tmp_path / "positional_calcs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create a YAML config with keyprenorm section.
    cfg_path = tmp_path / "config.yaml"
    cfg = {
        "registry": {
            "enabled": False,
        },
        "keyprenorm": {
            "input_dir": str(input_dir),
            "output_dir": str(tmp_path / "out"),
            "pattern": "*_positional_calcs.csv",
            "file_filter": "trial",
            "clobber": True,
            "calculate_norm": True,
            "edge_skip": 7,
            "max_rows": 0,
            "num_workers": 3,
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Capture the cfg passed to run_keyprenorm
    captured = {}

    def _fake_run_keyprenorm(*, cfg, args) -> None:  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        captured["args"] = args

    from autobehaver.preinf.keyprenorm import cli as keyprenorm_cli

    monkeypatch.setattr(keyprenorm_cli, "run_keyprenorm", _fake_run_keyprenorm)

    keyprenorm_cli.main(["--config", str(cfg_path)])

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
    assert kcfg.max_rows == 0
    assert kcfg.num_workers == 3

    assert kcfg.clobber is True
