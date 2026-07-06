import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autobehaver.postinf.mmgen.append_mode import run_mmgen_append_new_data
from autobehaver.postinf.mmgen.core import MMGenConfig
from autobehaver.postinf.mmgen.ledger import SOURCE_LEDGER_FILENAME
from autobehaver.postinf.mmgen.naming import collapse_individual_id, parse_group_and_individual
from autobehaver.postinf.mmgen.sharding import select_files_for_shard


class TestMMGenAppendMode(unittest.TestCase):
    def _write_source_pair(self, inf_dir: Path, pos_dir: Path, name: str, *, source_text: str = None):
        if source_text is None:
            source_text = "Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\n"
        (inf_dir / name).write_text(source_text)
        pos_name = name.replace("_normalized_inference", "")
        (pos_dir / pos_name).write_text("centroid_x,centroid_y\n1,2\n")
        return pos_name

    def _ledger_row(self, inf_dir: Path, pos_dir: Path, name: str):
        group, individual_id, sex = parse_group_and_individual(name)
        row_identity = collapse_individual_id(individual_id)
        source_path = inf_dir / name
        pos_name = name.replace("_normalized_inference", "")
        pos_path = pos_dir / pos_name
        s_stat = source_path.stat()
        p_stat = pos_path.stat()
        return {
            "source_file": name,
            "source_path": str(source_path),
            "source_exists": True,
            "source_file_size_bytes": int(s_stat.st_size),
            "source_file_mtime_utc": datetime.fromtimestamp(s_stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "poscalc_file": pos_name,
            "poscalc_path": str(pos_path),
            "poscalc_exists": True,
            "poscalc_file_size_bytes": int(p_stat.st_size),
            "poscalc_file_mtime_utc": datetime.fromtimestamp(p_stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "n_frames": 100,
            "group": group,
            "sex": sex,
            "individual_id": individual_id,
            "animal_id": row_identity,
            "row_identity": row_identity,
            "rpi": "rpi-1",
            "chamber": "chamber5",
            "trial": 1,
        }

    def test_append_new_data_updates_canonical_matrix_and_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            a0 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie0_20250101120000PM_chamber5_positional_calcs_normalized_inference.csv"
            a1 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie1_20250101130000PM_chamber5_positional_calcs_normalized_inference.csv"
            b0 = "GROUPB_MALE_demo_trial1_rpi-2_movie0_20250101120000PM_chamber6_positional_calcs_normalized_inference.csv"
            c0 = "GROUPC_FEMALE_demo_trial2_rpi-3_movie0_20250102120000PM_chamber7_positional_calcs_normalized_inference.csv"

            self._write_source_pair(inf, pos, a0)
            self._write_source_pair(inf, pos, a1)
            self._write_source_pair(inf, pos, b0)
            self._write_source_pair(inf, pos, c0)

            a_row = collapse_individual_id(parse_group_and_individual(a0)[1])
            b_row = collapse_individual_id(parse_group_and_individual(b0)[1])
            c_row = collapse_individual_id(parse_group_and_individual(c0)[1])

            existing_matrix = pd.DataFrame([
                {"individual": a_row, "group": "GROUPA", "sex": "FEMALE", "Feature_A": 1.0},
                {"individual": b_row, "group": "GROUPB", "sex": "MALE", "Feature_A": 2.0},
            ])
            existing_matrix.to_csv(out / "MASTER_FEATURE_MATRIX.csv", index=False)

            existing_ledger = pd.DataFrame([
                self._ledger_row(inf, pos, a0),
                self._ledger_row(inf, pos, b0),
            ])
            existing_ledger.to_csv(out / SOURCE_LEDGER_FILENAME, index=False)

            captured = {}

            def _fake_delta_run(inf_csv_folder_path, poscalc_folder_path, output_folder_path, min_bout, edge_skip, inf_tag, **kwargs):
                captured["selected_files"] = list(kwargs.get("selected_files") or [])
                out_dir = Path(output_folder_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([
                    {"individual": a_row, "group": "GROUPA", "sex": "FEMALE", "Feature_A": 10.0},
                    {"individual": c_row, "group": "GROUPC", "sex": "FEMALE", "Feature_A": 3.0},
                ]).to_csv(out_dir / "MASTER_FEATURE_MATRIX.csv", index=False)
                pd.DataFrame([
                    self._ledger_row(inf, pos, a0),
                    self._ledger_row(inf, pos, a1),
                    self._ledger_row(inf, pos, c0),
                ]).to_csv(out_dir / SOURCE_LEDGER_FILENAME, index=False)

            cfg = MMGenConfig(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                output_dir=str(out),
                append_new_data=True,
                emit_master_matrix=True,
                emit_full_outputs=False,
            )

            with patch("autobehaver.postinf.mmgen.append_mode.pipeline.run_post_inference_analysis", side_effect=_fake_delta_run):
                run_mmgen_append_new_data(cfg)

            self.assertEqual(set(captured["selected_files"]), {a0, a1, c0})

            merged = pd.read_csv(out / "MASTER_FEATURE_MATRIX.csv")
            self.assertEqual(set(merged["individual"].tolist()), {a_row, b_row, c_row})
            self.assertEqual(float(merged.loc[merged["individual"] == a_row, "Feature_A"].iloc[0]), 10.0)
            self.assertEqual(float(merged.loc[merged["individual"] == b_row, "Feature_A"].iloc[0]), 2.0)
            self.assertEqual(float(merged.loc[merged["individual"] == c_row, "Feature_A"].iloc[0]), 3.0)

            merged_ledger = pd.read_csv(out / SOURCE_LEDGER_FILENAME)
            self.assertEqual(set(merged_ledger["source_file"].tolist()), {a0, a1, b0, c0})

            row_summary = pd.read_csv(out / "MMGEN_ROW_IDENTITY_SUMMARY.csv")
            self.assertEqual(set(row_summary["row_identity"].tolist()), {a_row, b_row, c_row})

            manifest = json.loads((out / "MMGEN_BATCH_MANIFEST.json").read_text())
            self.assertTrue(bool(manifest["processing"]["append_new_data"]))
            latest_report = json.loads((out / "MMGEN_APPEND_LATEST.json").read_text())
            self.assertEqual(latest_report["counts"]["replaced_rows"], 1)
            self.assertEqual(latest_report["counts"]["appended_rows"], 1)

    def test_append_new_data_fails_on_changed_previously_processed_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            a0 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie0_20250101120000PM_chamber5_positional_calcs_normalized_inference.csv"
            self._write_source_pair(inf, pos, a0)
            a_row = collapse_individual_id(parse_group_and_individual(a0)[1])

            pd.DataFrame([
                {"individual": a_row, "group": "GROUPA", "sex": "FEMALE", "Feature_A": 1.0},
            ]).to_csv(out / "MASTER_FEATURE_MATRIX.csv", index=False)
            pd.DataFrame([self._ledger_row(inf, pos, a0)]).to_csv(out / SOURCE_LEDGER_FILENAME, index=False)

            # Mutate the already-processed inference file so append mode sees it as changed.
            (inf / a0).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\nStanding,Floor\n")

            cfg = MMGenConfig(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                output_dir=str(out),
                append_new_data=True,
                emit_master_matrix=True,
                emit_full_outputs=False,
            )

            with self.assertRaises(RuntimeError):
                run_mmgen_append_new_data(cfg)

    def test_append_new_data_catches_changed_old_file_within_affected_row_even_if_filter_targets_only_new_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            a0 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie0_20250101120000PM_chamber5_positional_calcs_normalized_inference.csv"
            a1 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie1_20250101130000PM_chamber5_positional_calcs_normalized_inference.csv"
            self._write_source_pair(inf, pos, a0)
            self._write_source_pair(inf, pos, a1)
            a_row = collapse_individual_id(parse_group_and_individual(a0)[1])

            pd.DataFrame([
                {"individual": a_row, "group": "GROUPA", "sex": "FEMALE", "Feature_A": 1.0},
            ]).to_csv(out / "MASTER_FEATURE_MATRIX.csv", index=False)
            pd.DataFrame([self._ledger_row(inf, pos, a0)]).to_csv(out / SOURCE_LEDGER_FILENAME, index=False)

            # Mutate the already-processed file, but use a file_filter that only directly selects the *new* file.
            (inf / a0).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\nStanding,Floor\n")

            cfg = MMGenConfig(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                output_dir=str(out),
                append_new_data=True,
                emit_master_matrix=True,
                emit_full_outputs=False,
                file_filter="movie1",
            )

            with self.assertRaises(RuntimeError):
                run_mmgen_append_new_data(cfg)

    def test_sharded_append_plan_process_and_merge(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            a0 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie0_20250101120000PM_chamber5_positional_calcs_normalized_inference.csv"
            a1 = "GROUPA_FEMALE_demo_trial1_rpi-1_movie1_20250101130000PM_chamber5_positional_calcs_normalized_inference.csv"
            b0 = "GROUPB_MALE_demo_trial1_rpi-2_movie0_20250101120000PM_chamber6_positional_calcs_normalized_inference.csv"
            c0 = "GROUPC_FEMALE_demo_trial2_rpi-3_movie0_20250102120000PM_chamber7_positional_calcs_normalized_inference.csv"

            self._write_source_pair(inf, pos, a0)
            self._write_source_pair(inf, pos, a1)
            self._write_source_pair(inf, pos, b0)
            self._write_source_pair(inf, pos, c0)

            a_row = collapse_individual_id(parse_group_and_individual(a0)[1])
            b_row = collapse_individual_id(parse_group_and_individual(b0)[1])
            c_row = collapse_individual_id(parse_group_and_individual(c0)[1])

            pd.DataFrame([
                {"individual": a_row, "group": "GROUPA", "sex": "FEMALE", "Feature_A": 1.0},
                {"individual": b_row, "group": "GROUPB", "sex": "MALE", "Feature_A": 2.0},
            ]).to_csv(out / "MASTER_FEATURE_MATRIX.csv", index=False)
            pd.DataFrame([
                self._ledger_row(inf, pos, a0),
                self._ledger_row(inf, pos, b0),
            ]).to_csv(out / SOURCE_LEDGER_FILENAME, index=False)

            plan_cfg = MMGenConfig(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                output_dir=str(out),
                append_new_data=True,
                plan_only=True,
                num_shards=3,
                shard_by="row_identity",
                emit_master_matrix=True,
                emit_full_outputs=False,
            )
            run_mmgen_append_new_data(plan_cfg)

            batches = sorted((out / "_mmgen_append_batches").iterdir())
            self.assertEqual(len(batches), 1)
            batch_dir = batches[0]
            batch_id = batch_dir.name
            plan = json.loads((batch_dir / "APPEND_PLAN.json").read_text())
            self.assertEqual(int(plan["processing"]["requested_num_shards"]), 3)
            self.assertEqual(int(plan["processing"]["active_num_shards"]), 2)

            captured_shards = {}

            def _fake_shard_process(inf_csv_folder_path, poscalc_folder_path, output_folder_path, min_bout, edge_skip, inf_tag, **kwargs):
                selected = list(kwargs.get("selected_files") or [])
                shard_files = select_files_for_shard(
                    selected,
                    num_shards=int(kwargs.get("num_shards", 1)),
                    shard_index=int(kwargs.get("shard_index", 0)),
                    strategy=str(kwargs.get("shard_by", "hash")),
                )
                shard_index = int(kwargs.get("shard_index", 0))
                captured_shards[shard_index] = shard_files

                shard_dir = Path(output_folder_path) / "_mmgen_shards" / f"shard_{shard_index:03d}_of_{int(kwargs.get('num_shards', 1)):03d}"
                work_dir = shard_dir / "_mmgen_work"
                work_dir.mkdir(parents=True, exist_ok=True)
                (shard_dir / "temp_store.h5").write_bytes(b"stub")
                (work_dir / "SHARD_DONE.json").write_text(
                    json.dumps(
                        {
                            "num_shards": int(kwargs.get("num_shards", 1)),
                            "shard_index": shard_index,
                            "shard_by": str(kwargs.get("shard_by", "row_identity")),
                        }
                    )
                )

            with patch("autobehaver.postinf.mmgen.append_mode.pipeline.run_post_inference_analysis", side_effect=_fake_shard_process):
                for shard_index in range(2):
                    shard_cfg = MMGenConfig(
                        inference_csv_dir=str(inf),
                        poscalc_dir=str(pos),
                        output_dir=str(out),
                        append_new_data=True,
                        process_only=True,
                        resume=False,
                        append_batch_id=batch_id,
                        shard_index=shard_index,
                        num_shards=3,
                        shard_by="row_identity",
                        emit_master_matrix=True,
                        emit_full_outputs=False,
                    )
                    run_mmgen_append_new_data(shard_cfg)

            self.assertEqual(set(captured_shards[0] + captured_shards[1]), {a0, a1, c0})
            self.assertTrue(({a0, a1}.issubset(set(captured_shards[0])) and {c0}.issubset(set(captured_shards[1]))) or ({a0, a1}.issubset(set(captured_shards[1])) and {c0}.issubset(set(captured_shards[0]))))

            def _fake_merge(*, output_folder, num_shards, **kwargs):
                self.assertEqual(int(num_shards), 2)
                Path(output_folder, "temp_store.h5").write_bytes(b"merged")
                return str(Path(output_folder, "temp_store.h5"))

            def _fake_finalize(inf_csv_folder_path, poscalc_folder_path, output_folder_path, min_bout, edge_skip, inf_tag, **kwargs):
                out_dir = Path(output_folder_path)
                pd.DataFrame([
                    {"individual": a_row, "group": "GROUPA", "sex": "FEMALE", "Feature_A": 10.0},
                    {"individual": c_row, "group": "GROUPC", "sex": "FEMALE", "Feature_A": 3.0},
                ]).to_csv(out_dir / "MASTER_FEATURE_MATRIX.csv", index=False)
                pd.DataFrame([
                    self._ledger_row(inf, pos, a0),
                    self._ledger_row(inf, pos, a1),
                    self._ledger_row(inf, pos, c0),
                ]).to_csv(out_dir / SOURCE_LEDGER_FILENAME, index=False)

            with patch("autobehaver.postinf.mmgen.append_mode.merge_shard_temp_stores", side_effect=_fake_merge), \
                 patch("autobehaver.postinf.mmgen.append_mode.pipeline.run_post_inference_analysis", side_effect=_fake_finalize):
                merge_cfg = MMGenConfig(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    append_new_data=True,
                    merge_shards=True,
                    append_batch_id=batch_id,
                    keep_temp_files=False,
                    emit_master_matrix=True,
                    emit_full_outputs=False,
                )
                run_mmgen_append_new_data(merge_cfg)

            merged = pd.read_csv(out / "MASTER_FEATURE_MATRIX.csv")
            self.assertEqual(set(merged["individual"].tolist()), {a_row, b_row, c_row})
            self.assertEqual(float(merged.loc[merged["individual"] == a_row, "Feature_A"].iloc[0]), 10.0)
            self.assertEqual(float(merged.loc[merged["individual"] == b_row, "Feature_A"].iloc[0]), 2.0)
            self.assertEqual(float(merged.loc[merged["individual"] == c_row, "Feature_A"].iloc[0]), 3.0)

            latest_report = json.loads((out / "MMGEN_APPEND_LATEST.json").read_text())
            self.assertEqual(latest_report["counts"]["active_num_shards"], 2)
            self.assertEqual(latest_report["notes"]["batch_mode"], "sharded")

            # Large append-batch HDF5 temp stores should be cleaned up when keep_temp_files=False.
            self.assertFalse((batch_dir / "temp_store.h5").exists())
            self.assertFalse((batch_dir / "_mmgen_shards" / "shard_000_of_002" / "temp_store.h5").exists())
            self.assertFalse((batch_dir / "_mmgen_shards" / "shard_001_of_002" / "temp_store.h5").exists())


if __name__ == "__main__":
    unittest.main()
