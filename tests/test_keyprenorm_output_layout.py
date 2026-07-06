from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd


def test_keyprenorm_explicit_output_dir_is_flat(tmp_path: Path) -> None:
    from autobehaver.preinf.keyprenorm.config import KeypreNormConfig
    from autobehaver.preinf.keyprenorm.core import run_keyprenorm

    input_dir = tmp_path / "4_keypoint_features"
    output_dir = tmp_path / "5_normalized_features"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    csv_path = input_dir / "sample_positional_calcs.csv"
    pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_csv(csv_path, index=False)

    cfg = KeypreNormConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        calculate_norm=True,
        clobber=True,
        num_workers=1,
        edge_skip=0,
        max_rows=0,
        batch_size=4,
        log_level="ERROR",
    )

    run_keyprenorm(cfg=cfg, args=argparse.Namespace())

    norm_h5 = output_dir / "sample_positional_calcs_normalized.h5"
    norm_params = output_dir / "norm_params.csv"

    assert norm_h5.exists()
    assert norm_params.exists()
    assert not (output_dir / "normalization_h5").exists()
    assert not (output_dir / "norm_params").exists()

    with h5py.File(norm_h5, "r") as hf:
        assert "data" in hf
        assert hf["data"].shape == (4, 2)
