import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autobehaver.postinf.mmgen.ledger import (
    build_row_identity_summary_df,
    build_source_ledger_df,
    write_batch_manifest_json,
)
from autobehaver.postinf.mmgen.mapping import create_animal_mapping
from autobehaver.postinf.mmgen.naming import collapse_individual_id, parse_group_and_individual


class TestMMGenLedger(unittest.TestCase):
    def _make_source_files(self, base: Path):
        inf = base / "inf"
        pos = base / "pos"
        out = base / "out"
        inf.mkdir()
        pos.mkdir()
        out.mkdir()

        names = [
            "CONTROL-PLPdeltaR-plus_FEMALE_plp_deltaR_experiment_trial1_rpi-1_movie0_20250501124915PM_chamber5_positional_calcs_normalized_inference.csv",
            "CONTROL-PLPdeltaR-plus_FEMALE_plp_deltaR_experiment_trial1_rpi-1_movie1_20250501134915PM_chamber5_positional_calcs_normalized_inference.csv",
        ]
        for name in names:
            (inf / name).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\n")
            pos_name = name.replace("_normalized_inference", "")
            (pos / pos_name).write_text("centroid_x,centroid_y\n1,2\n")

        return inf, pos, out, names

    def test_build_source_ledger_and_row_identity_summary(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf, pos, out, names = self._make_source_files(base)

            animal_mapping, file_to_animal = create_animal_mapping(str(inf), str(pos), str(out))
            group, individual0, sex = parse_group_and_individual(names[0])
            _, individual1, _ = parse_group_and_individual(names[1])
            collapsed = collapse_individual_id(individual0)

            lengths_df = pd.DataFrame([
                {
                    "source_file": names[0],
                    "group": group,
                    "individual": individual0,
                    "sex": sex,
                    "n_frames": 100,
                },
                {
                    "source_file": names[1],
                    "group": group,
                    "individual": individual1,
                    "sex": sex,
                    "n_frames": 120,
                },
            ])

            ledger_df = build_source_ledger_df(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                lengths_df=lengths_df,
                animal_mapping=animal_mapping,
                file_to_animal=file_to_animal,
                inf_tag="_normalized_inference",
            )
            self.assertEqual(len(ledger_df), 2)
            self.assertEqual(set(ledger_df["row_identity"].tolist()), {collapsed})
            self.assertTrue(bool(ledger_df["source_exists"].all()))
            self.assertTrue(bool(ledger_df["poscalc_exists"].all()))
            self.assertEqual(int(ledger_df["n_frames"].sum()), 220)

            master_matrix_df = pd.DataFrame([
                {"individual": collapsed, "group": group, "sex": sex, "Feature_A": 1.23}
            ])
            row_df = build_row_identity_summary_df(
                source_ledger_df=ledger_df,
                master_matrix_df=master_matrix_df,
            )
            self.assertEqual(len(row_df), 1)
            self.assertEqual(row_df.loc[0, "row_identity"], collapsed)
            self.assertEqual(int(row_df.loc[0, "n_source_files"]), 2)
            self.assertEqual(int(row_df.loc[0, "total_n_frames"]), 220)
            self.assertTrue(bool(row_df.loc[0, "in_master_matrix"]))

    def test_manifest_includes_schema_and_counts(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf, pos, out, names = self._make_source_files(base)

            animal_mapping, file_to_animal = create_animal_mapping(str(inf), str(pos), str(out))
            group, individual0, sex = parse_group_and_individual(names[0])
            _, individual1, _ = parse_group_and_individual(names[1])
            collapsed = collapse_individual_id(individual0)

            lengths_df = pd.DataFrame([
                {"source_file": names[0], "group": group, "individual": individual0, "sex": sex, "n_frames": 100},
                {"source_file": names[1], "group": group, "individual": individual1, "sex": sex, "n_frames": 120},
            ])
            ledger_df = build_source_ledger_df(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                lengths_df=lengths_df,
                animal_mapping=animal_mapping,
                file_to_animal=file_to_animal,
                inf_tag="_normalized_inference",
            )
            row_df = build_row_identity_summary_df(
                source_ledger_df=ledger_df,
                master_matrix_df=pd.DataFrame([
                    {"individual": collapsed, "group": group, "sex": sex, "Feature_A": 1.23}
                ]),
            )
            manifest_path = write_batch_manifest_json(
                output_dir=str(out),
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                inf_tag="_normalized_inference",
                min_bout=5,
                edge_skip=5,
                resume=False,
                finalize_only=False,
                process_only=False,
                file_filter=None,
                num_shards=1,
                shard_index=0,
                shard_by="hash",
                emit_master_matrix=True,
                emit_full_outputs=True,
                lengths_df=lengths_df,
                source_ledger_df=ledger_df,
                row_identity_df=row_df,
                master_matrix_df=pd.DataFrame([
                    {"individual": collapsed, "group": group, "sex": sex, "Feature_A": 1.23}
                ]),
                animal_mapping=animal_mapping,
            )
            payload = json.loads(Path(manifest_path).read_text())
            self.assertEqual(payload["row_identity"]["column"], "individual")
            self.assertEqual(payload["counts"]["committed_source_files"], 2)
            self.assertEqual(payload["counts"]["row_identity_count"], 1)
            self.assertTrue(bool(payload["master_matrix"]["schema_sha256"]))

    def test_duplicate_master_matrix_row_identity_raises(self):
        ledger_df = pd.DataFrame([
            {"row_identity": "fly1", "source_file": "a.csv", "n_frames": 10, "group": "g", "sex": "F", "rpi": "rpi-1", "chamber": "chamber1", "trial": 1},
            {"row_identity": "fly1", "source_file": "b.csv", "n_frames": 20, "group": "g", "sex": "F", "rpi": "rpi-1", "chamber": "chamber1", "trial": 1},
        ])
        master_matrix_df = pd.DataFrame([
            {"individual": "fly1", "group": "g", "sex": "F", "Feature_A": 1.0},
            {"individual": "fly1", "group": "g", "sex": "F", "Feature_A": 2.0},
        ])
        with self.assertRaises(ValueError):
            build_row_identity_summary_df(
                source_ledger_df=ledger_df,
                master_matrix_df=master_matrix_df,
            )


if __name__ == "__main__":
    unittest.main()
