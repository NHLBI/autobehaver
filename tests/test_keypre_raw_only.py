import os
import tempfile

import numpy as np
import pandas as pd


def _write_synthetic_dlc_csv(path: str, *, n_frames: int = 12) -> pd.DataFrame:
    """Write a small synthetic DLC-style CSV with a 3-level header."""

    from autobehaver.preinf.keypre.spec import parts

    scorer = "scorer"
    coords = ["x", "y", "likelihood"]
    all_parts = list(parts) + ["Front", "Back"]

    columns = pd.MultiIndex.from_tuples(
        [(scorer, part, coord) for part in all_parts for coord in coords]
    )

    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_frames, len(columns))).astype(np.float32)

    df = pd.DataFrame(data, index=np.arange(n_frames), columns=columns)
    df.to_csv(path)
    return df


def test_keypre_raw_only_exports_expected_columns_and_values():
    from autobehaver.preinf.keypre.spec import parts
    from autobehaver.preinf.keypre.worker import process_csv_file, set_run_config

    with tempfile.TemporaryDirectory() as tmp:
        dlc_dir = os.path.join(tmp, "DLC")
        os.makedirs(dlc_dir, exist_ok=True)

        csv_path = os.path.join(dlc_dir, "example.csv")
        src_df = _write_synthetic_dlc_csv(csv_path, n_frames=12)

        out_dir = os.path.join(tmp, "raw_positional_calcs")
        os.makedirs(out_dir, exist_ok=True)

        set_run_config(
            {
                "output_dir": out_dir,
                "clobber": True,
                "movie_dim": 600,
                "raw_only": True,
            }
        )

        result = process_csv_file(csv_path)
        assert result["status"] == "success"
        out_path = result["out"]
        assert out_path is not None and os.path.exists(out_path)

        out_df = pd.read_csv(out_path)

        # Ensure Front/Back are excluded.
        assert "Front_x" not in out_df.columns
        assert "Back_x" not in out_df.columns
        assert "Front_x_raw" not in out_df.columns
        assert "Back_x_raw" not in out_df.columns

        # Ensure standard parts are included as x/y/likelihood (with `_raw` suffix).
        for part in parts:
            assert f"{part}_x_raw" in out_df.columns
            assert f"{part}_y_raw" in out_df.columns
            assert f"{part}_likelihood_raw" in out_df.columns

        # Raw-only mode should not apply edge NaNs.
        assert not np.isnan(out_df.loc[0, "Ocelli_x_raw"])
        assert not np.isnan(out_df.loc[len(out_df) - 1, "Ocelli_x_raw"])

        # Values should match the original DLC (after flattening), for at least one keypoint.
        middle = 6
        # Flatten the source like keypre does: bodypart_coord
        flat_cols = [f"{c[1]}_{c[2]}" for c in src_df.columns]
        flat_df = src_df.copy()
        flat_df.columns = flat_cols

        assert np.isclose(out_df.loc[middle, "Ocelli_x_raw"], flat_df.loc[middle, "Ocelli_x"], atol=1e-4)
