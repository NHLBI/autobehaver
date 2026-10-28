"""Check schema selection, normalization values and HDF5 metadata on small arrays."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import h5py


class TestKeypreNormSynthetic(unittest.TestCase):
    def test_normparams_and_apply(self):
        from autobehaver.preinf.keyprenorm import __stage_version__
        from autobehaver.preinf.keyprenorm.schema import read_csv_header_columns, filter_files_by_schema
        from autobehaver.preinf.keyprenorm.normparams import calculate_normalization_params
        from autobehaver.preinf.keyprenorm.normalize import init_normalize_worker, normalize_one_csv

        # Small synthetic dataset
        cols = ["f1", "f2", "const"]

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = tmpdir
            out_dir = os.path.join(inp, "normalization_h5")
            os.makedirs(out_dir, exist_ok=True)
            norm_dir = os.path.join(inp, "norm_params")
            os.makedirs(norm_dir, exist_ok=True)
            norm_path = os.path.join(norm_dir, "norm_params.csv")

            # File 1: canonical column order
            df1 = pd.DataFrame(
                {
                    "f1": [0.0, 1.0, 2.0, 3.0],
                    "f2": [10.0, 11.0, 12.0, 13.0],
                    "const": [5.0, 5.0, 5.0, 5.0],
                }
            )
            f1 = os.path.join(inp, "a_positional_calcs.csv")
            df1.to_csv(f1, index=False)

            # File 2: same columns but different order
            df2 = df1[["const", "f2", "f1"]].copy()
            f2 = os.path.join(inp, "b_positional_calcs.csv")
            df2.to_csv(f2, index=False)

            # File 3: schema mismatch (missing f2)
            df3 = df1.drop(columns=["f2"])
            f3 = os.path.join(inp, "bad_positional_calcs.csv")
            df3.to_csv(f3, index=False)

            all_csv = [f1, f2, f3]
            canonical_cols, err = read_csv_header_columns(f1)
            self.assertIsNone(err)
            self.assertEqual(canonical_cols, cols)

            good, bad = filter_files_by_schema(all_csv, canonical_cols, max_examples=10)
            self.assertEqual(set(good), {f1, f2})
            self.assertEqual(len(bad), 1)
            self.assertTrue(bad[0]["file"].endswith("bad_positional_calcs.csv"))

            # Both schema-compatible files contribute to the normalization scale.
            # Combine df1 + df1 (same values) so mean/std are easy.
            expected_mean = {"f1": 1.5, "f2": 11.5, "const": 5.0}
            expected_std = {
                "f1": np.sqrt(1.25),
                "f2": np.sqrt(1.25),
                # constant columns => std=0
                "const": 0.0,
            }

            # Direct worker initialization changes this process's logging.
            # Own a consumer and restore logging even when an assertion fails.
            from autobehaver.preinf.keyprenorm.logging_utils import setup_logging_queue

            q, listener = setup_logging_queue(log_level="ERROR", log_file=None)
            self.addCleanup(listener.stop)
            norm_df = calculate_normalization_params(
                csv_files=good,
                canonical_columns=canonical_cols,
                norm_params_path=norm_path,
                norm_params_dir=norm_dir,
                num_workers=1,
                edge_skip=0,
                batch_size=10,
                log_queue=q,
                log_level="ERROR",
            )
            self.assertTrue(os.path.exists(norm_path))

            # Validate means/stds
            for _, row in norm_df.iterrows():
                h = row["Header"]
                self.assertAlmostEqual(float(row["Mean"]), float(expected_mean[h]), places=6)
                self.assertAlmostEqual(float(row["StdDev"]), float(expected_std[h]), places=6)

            means = norm_df["Mean"].to_numpy(dtype=np.float32, copy=False)
            stds = norm_df["StdDev"].to_numpy(dtype=np.float32, copy=False)

            init_normalize_worker(
                canonical_columns=canonical_cols,
                means=means,
                stds=stds,
                edge_skip=0,
                output_dir=out_dir,
                labels_input_dir=None,
                labels_output_dir=None,
                log_queue=q,
                log_level="ERROR",
            )

            r1 = normalize_one_csv(f1)
            r2 = normalize_one_csv(f2)
            self.assertEqual(r1["status"], "success")
            self.assertEqual(r2["status"], "success")

            # Validate output H5 format + values
            for csv_path in (f1, f2):
                base = os.path.basename(csv_path)
                out_name = base.replace(".csv", "_normalized.h5")
                out_path = os.path.join(out_dir, out_name)
                self.assertTrue(os.path.exists(out_path))

                with h5py.File(out_path, "r") as hf:
                    self.assertIn("data", hf)
                    data = hf["data"][:]
                    self.assertEqual(data.shape, (4, 3))

                    cols_attr = hf.attrs.get("columns")
                    self.assertIsNotNone(cols_attr)
                    decoded = [
                        c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                        for c in cols_attr
                    ]
                    self.assertEqual(decoded, canonical_cols)

                    ver = hf.attrs.get("keyprenorm_version")
                    self.assertIsNotNone(ver)
                    if isinstance(ver, (bytes, bytearray)):
                        ver = ver.decode("utf-8")
                    self.assertEqual(str(ver), str(__stage_version__))

                # f1/f2 should be z-scored (mean=1.5/11.5, std=sqrt(1.25))
                z_f1 = (df1["f1"].to_numpy() - 1.5) / np.sqrt(1.25)
                z_f2 = (df1["f2"].to_numpy() - 11.5) / np.sqrt(1.25)
                np.testing.assert_allclose(data[:, 0], z_f1.astype(np.float32), rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(data[:, 1], z_f2.astype(np.float32), rtol=1e-5, atol=1e-6)

                # const has std=0, so it is forced to 0.0 after normalization
                self.assertTrue(np.allclose(data[:, 2], 0.0))


if __name__ == "__main__":
    unittest.main()
