"""Small boundary checks for full-length normalization and explicit exclusions."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


def test_normalization_retains_unequal_inputs_and_tail_beyond_old_cap(tmp_path):
    from autobehaver.preinf.keyprenorm.config import KeypreNormConfig
    from autobehaver.preinf.keyprenorm.core import run_keyprenorm
    source = tmp_path / "in"
    output = tmp_path / "out"
    labels = tmp_path / "labels"
    for path in (source, output, labels):
        path.mkdir()
    arrays = []
    for name, n in [("long", 64762), ("short", 48)]:
        values = np.zeros(n, dtype=np.float32)
        values[-7:] = 10
        arrays.append(values[5:-5])
        pd.DataFrame({"x": values}).to_csv(source / f"{name}_positional_calcs.csv", index=False)
        pd.DataFrame({"Frame": np.arange(n), "Behavior": ["Walking"] * n}).to_csv(labels / f"{name}_labels.csv", index=False)
    cfg = KeypreNormConfig(input_dir=str(source), output_dir=str(output), labels_dir=str(labels),
                          calculate_norm=True, num_workers=2, batch_size=1, log_level="ERROR")
    run_keyprenorm(cfg=cfg, args=argparse.Namespace())
    params = pd.read_csv(output / "norm_params.csv")
    values = np.concatenate(arrays).astype(np.float64)
    assert params.Mean.iloc[0] == pytest.approx(values.mean(), rel=1e-12)
    assert params.StdDev.iloc[0] == pytest.approx(values.std(), rel=1e-12)
    for name, n in [("long", 64762), ("short", 48)]:
        with h5py.File(output / f"{name}_positional_calcs_normalized.h5", "r") as hf:
            assert hf["data"].shape == (n - 10, 1)
            assert np.isfinite(hf["data"][:]).all()
            assert hf["data"][-1, 0] == pytest.approx((10 - params.Mean.iloc[0]) / params.StdDev.iloc[0], rel=1e-6)
        with h5py.File(output / "labels" / f"{name}_labels.h5", "r") as hf:
            np.testing.assert_array_equal(hf["Frame"][:], np.arange(5, n-5))
    # The same variable-length outputs remain reusable with the same frozen scale.
    reuse_cfg = KeypreNormConfig(input_dir=str(source), output_dir=str(output), labels_dir=str(labels),
                                norm_params=str(output / "norm_params.csv"), num_workers=1, log_level="ERROR")
    run_keyprenorm(cfg=reuse_cfg, args=argparse.Namespace())
    manifests = [json.loads(path.read_text()) for path in output.glob("keyprenorm_run_manifest_*.json")]
    assert max(m["results"].get("existing_outputs_skipped", 0) for m in manifests) == 2


def test_crop_completion_ignores_historical_classifier_and_obeys_explicit_exclusions(monkeypatch, tmp_path):
    from autobehaver.preinf.autocropper import pipeline
    monkeypatch.setattr(pipeline, "args", argparse.Namespace(chamber_rows=2, chamber_columns=3))
    monkeypatch.setattr(pipeline, "run_dir", tmp_path)
    movie = Path("exp_sub_trial1_rpi-1_movie0.mp4")
    legacy = tmp_path / "validation_results"
    legacy.mkdir()
    (legacy / f"{movie.stem}_validation.json").write_text(json.dumps({"chamber1": {"is_occupied": False}}))
    asked = []
    def resolve(_movie, _output, chamber):
        asked.append(chamber)
        return {"status": "missing", "expected_name": chamber}
    monkeypatch.setattr(pipeline, "_resolve_existing_crop", resolve)
    assert not pipeline.all_cropped_movies_exist(movie, tmp_path)
    assert asked == ["chamber1"]
    asked.clear()
    assert pipeline.all_cropped_movies_exist(movie, tmp_path, excluded_chambers={f"chamber{i}" for i in range(1, 7)})
    assert asked == []


def test_removed_length_and_classifier_options_are_not_public():
    from autobehaver.preinf.autocropper.cli import build_arg_parser as crop_parser
    from autobehaver.preinf.keyprenorm.cli import build_arg_parser as norm_parser
    for parser, option in [(crop_parser(), "--validate_chambers"), (crop_parser(), "--skip_validation"),
                           (norm_parser(), "--max_rows"), (norm_parser(), "--require_full_length")]:
        with pytest.raises(SystemExit):
            parser.parse_args(["--input_dir", "example", option])
