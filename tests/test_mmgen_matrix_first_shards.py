import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autobehaver.postinf.mmgen import cli, core
from autobehaver.postinf.mmgen.core import MMGenConfig
from autobehaver.postinf.mmgen.matrix_parts import (
    SHARD_MANIFEST_FILENAME,
    SHARD_MATRIX_DONE_FILENAME,
    finalize_shard_temp_store,
    merge_matrix_part_files,
)
from autobehaver.postinf.mmgen.runtime import MMGenRuntimeSettings


_VALID_NAME = "CONTROL-PLPdeltaR-plus_FEMALE_plp_deltaR_experiment_trial1_rpi-1_movie0_20250501124915PM_chamber5_positional_calcs_normalized_inference.csv"


class TestMMGenMatrixFirstCli(unittest.TestCase):
    def test_matrix_first_shards_forces_matrix_only_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            captured = []

            def _fake_run(cfg):
                captured.append(cfg)

            with patch("autobehaver.postinf.mmgen.core.run_mmgen", side_effect=_fake_run):
                cli.main([
                    "--inference_csv_dir", str(inf),
                    "--poscalc_dir", str(pos),
                    "--output_dir", str(out),
                    "--num_shards", "2",
                    "--shard_index", "0",
                    "--shard_by", "row_identity",
                    "--matrix_first_shards",
                ])

            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0].emit_master_matrix)
            self.assertFalse(captured[0].emit_full_outputs)
            self.assertTrue(captured[0].matrix_first_shards)


class TestMMGenMatrixFirstCore(unittest.TestCase):
    def test_run_mmgen_matrix_first_shards_runs_process_then_finalize(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()
            (inf / _VALID_NAME).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\n")

            cfg = MMGenConfig(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                output_dir=str(out),
                num_shards=2,
                shard_index=0,
                shard_by="row_identity",
                matrix_first_shards=True,
            )

            process_calls = []
            finalize_calls = []

            with patch("autobehaver.postinf.mmgen.core.pipeline.run_post_inference_analysis") as p_run, \
                 patch("autobehaver.postinf.mmgen.core.finalize_shard_temp_store") as p_finalize:
                p_run.side_effect = lambda *a, **k: process_calls.append((a, k))
                p_finalize.side_effect = lambda *a, **k: finalize_calls.append((a, k))
                core.run_mmgen(cfg)

            self.assertEqual(len(process_calls), 1)
            self.assertEqual(process_calls[0][1]["process_only"], True)
            self.assertEqual(process_calls[0][1]["num_shards"], 2)
            self.assertTrue(process_calls[0][1]["caller_holds_shard_lock"])
            self.assertEqual(len(finalize_calls), 1)
            self.assertEqual(finalize_calls[0][1]["num_shards"], 2)
            self.assertEqual(finalize_calls[0][1]["shard_by"], "row_identity")
            self.assertTrue(finalize_calls[0][1]["caller_holds_shard_lock"])

    def test_run_mmgen_merge_matrix_parts_dispatches_helper(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()
            cfg = MMGenConfig(
                inference_csv_dir=str(inf),
                poscalc_dir=str(pos),
                output_dir=str(out),
                merge_matrix_parts=True,
            )
            with patch("autobehaver.postinf.mmgen.core.merge_matrix_part_files") as p_merge:
                core.run_mmgen(cfg)
            self.assertEqual(p_merge.call_count, 1)


class TestMMGenMatrixParts(unittest.TestCase):
    @staticmethod
    def _write_shard_done(work_dir: Path, *, n_selected: int = 1, n_committed: int | None = None) -> None:
        if n_committed is None:
            n_committed = n_selected
        (work_dir / "SHARD_DONE.json").write_text(json.dumps({
            "n_inference_files_selected": int(n_selected),
            "n_committed_files": int(n_committed),
        }))

    def test_finalize_shard_temp_store_forces_matrix_only_and_preserves_empty_columns(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()
            (inf / _VALID_NAME).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\n")

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            shard_dir.mkdir(parents=True)
            (shard_dir / "temp_store.h5").write_text("")
            work_dir.mkdir(parents=True, exist_ok=True)
            self._write_shard_done(work_dir)

            captured = {}

            def _fake_finalize(*args, **kwargs):
                captured.update(kwargs)
                pd.DataFrame([
                    {"individual": "a1", "group": "g1", "sex": "FEMALE", "FeatA": 1.0, "FeatEmpty": None}
                ]).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            with patch("autobehaver.postinf.mmgen.pipeline.run_post_inference_analysis", side_effect=_fake_finalize):
                path = finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

            self.assertTrue(path.endswith("MASTER_FEATURE_MATRIX.csv"))
            self.assertFalse(captured["runtime_settings"].emit_full_outputs)
            self.assertTrue(captured["runtime_settings"].emit_master_matrix)
            self.assertEqual(captured["drop_all_na_feature_columns"], False)
            done_path = shard_dir / "_mmgen_work" / SHARD_MATRIX_DONE_FILENAME
            self.assertTrue(done_path.exists())

    def test_finalize_shard_temp_store_uses_frozen_shard_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            work_dir.mkdir(parents=True)
            (shard_dir / "temp_store.h5").write_text("")
            self._write_shard_done(work_dir, n_selected=2)
            selected_files = ["/frozen/path/animal_A_movie0.csv", "/frozen/path/animal_A_movie1.csv"]
            (work_dir / SHARD_MANIFEST_FILENAME).write_text(json.dumps({
                "num_shards": 2,
                "shard_index": 0,
                "shard_by": "row_identity",
                "n_selected_files": 2,
                "selected_files": selected_files,
            }))

            captured = {}

            def _fake_finalize(*args, **kwargs):
                captured.update(kwargs)
                pd.DataFrame([
                    {"individual": "a1", "group": "g1", "sex": "FEMALE", "FeatA": 1.0}
                ]).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            with patch("autobehaver.postinf.mmgen.pipeline.run_post_inference_analysis", side_effect=_fake_finalize):
                finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

            self.assertEqual(captured["selected_files"], selected_files)

    def test_finalize_shard_temp_store_honors_finalize_lock(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            work_dir.mkdir(parents=True)
            (shard_dir / "temp_store.h5").write_text("")
            self._write_shard_done(work_dir)
            (work_dir / SHARD_MANIFEST_FILENAME).write_text(json.dumps({
                "num_shards": 2,
                "shard_index": 0,
                "shard_by": "row_identity",
                "n_selected_files": 1,
                "selected_files": ["/frozen/path/animal_A_movie0.csv"],
            }))
            (work_dir / "_mmgen_shard_matrix.lock").mkdir()

            with self.assertRaises(RuntimeError):
                finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

    def test_finalize_shard_temp_store_requires_completed_shard_done(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            work_dir.mkdir(parents=True)
            (shard_dir / "temp_store.h5").write_text("")
            (work_dir / SHARD_MANIFEST_FILENAME).write_text(json.dumps({
                "num_shards": 2,
                "shard_index": 0,
                "shard_by": "row_identity",
                "n_selected_files": 1,
                "selected_files": ["/frozen/path/animal_A_movie0.csv"],
            }))

            with self.assertRaises(RuntimeError):
                finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

    def test_finalize_shard_temp_store_requires_fully_committed_shard_done(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            work_dir.mkdir(parents=True)
            (shard_dir / "temp_store.h5").write_text("")
            self._write_shard_done(work_dir, n_selected=2, n_committed=1)
            (work_dir / SHARD_MANIFEST_FILENAME).write_text(json.dumps({
                "num_shards": 2,
                "shard_index": 0,
                "shard_by": "row_identity",
                "n_selected_files": 2,
                "selected_files": ["/frozen/path/animal_A_movie0.csv", "/frozen/path/animal_A_movie1.csv"],
            }))

            with self.assertRaises(RuntimeError):
                finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

    def test_finalize_shard_temp_store_skips_provenance_and_honors_discard_temp_files(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            work_dir.mkdir(parents=True)
            temp_store = shard_dir / "temp_store.h5"
            temp_store.write_text("")
            self._write_shard_done(work_dir)
            (work_dir / SHARD_MANIFEST_FILENAME).write_text(json.dumps({
                "num_shards": 2,
                "shard_index": 0,
                "shard_by": "row_identity",
                "n_selected_files": 1,
                "selected_files": ["/frozen/path/animal_A_movie0.csv"],
            }))

            captured = {}

            def _fake_finalize(*args, **kwargs):
                captured.update(kwargs)
                pd.DataFrame([
                    {"individual": "a1", "group": "g1", "sex": "FEMALE", "FeatA": 1.0}
                ]).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            with patch("autobehaver.postinf.mmgen.pipeline.run_post_inference_analysis", side_effect=_fake_finalize):
                finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(keep_temp_files=False),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

            self.assertEqual(captured["write_provenance_artifacts"], False)
            self.assertEqual(captured["spatial_in_memory_for_matrix"], True)
            self.assertFalse(temp_store.exists())

    def test_merge_matrix_part_files_merges_and_drops_globally_empty_columns(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for i, rows in [
                (0, [{"individual": "a1", "group": "g1", "sex": "FEMALE", "FeatA": 1.0, "AlwaysEmpty": None}]),
                (1, [{"individual": "a2", "group": "g2", "sex": "MALE", "FeatA": 2.0, "AlwaysEmpty": None}]),
            ]:
                shard_dir = out / "_mmgen_shards" / f"shard_{i:03d}_of_002"
                work = shard_dir / "_mmgen_work"
                work.mkdir(parents=True)
                (work / SHARD_MATRIX_DONE_FILENAME).write_text(json.dumps({"n_selected_files": 1}))
                pd.DataFrame(rows).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            path = merge_matrix_part_files(output_dir=str(out), num_shards=2)
            merged = pd.read_csv(path)
            self.assertEqual(merged["individual"].astype(str).tolist(), ["a1", "a2"])
            self.assertIn("FeatA", merged.columns)
            self.assertNotIn("AlwaysEmpty", merged.columns)
            self.assertTrue((out / "MASTER_FEATURE_MATRIX_UNITS.csv").exists())

    def test_merge_matrix_part_files_detects_duplicate_animals(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for i in [0, 1]:
                shard_dir = out / "_mmgen_shards" / f"shard_{i:03d}_of_002"
                work = shard_dir / "_mmgen_work"
                work.mkdir(parents=True)
                (work / SHARD_MATRIX_DONE_FILENAME).write_text(json.dumps({"n_selected_files": 1}))
                pd.DataFrame([
                    {"individual": "same_animal", "group": f"g{i}", "sex": "FEMALE", "FeatA": float(i)}
                ]).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            with self.assertRaises(ValueError):
                merge_matrix_part_files(output_dir=str(out), num_shards=2)

    def test_merge_matrix_part_files_raises_for_missing_expected_shard_dir(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work = shard_dir / "_mmgen_work"
            work.mkdir(parents=True)
            (work / SHARD_MATRIX_DONE_FILENAME).write_text(json.dumps({"n_selected_files": 1}))
            pd.DataFrame([
                {"individual": "a1", "group": "g1", "sex": "FEMALE", "FeatA": 1.0}
            ]).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            with self.assertRaises(FileNotFoundError):
                merge_matrix_part_files(output_dir=str(out), num_shards=2)

    def test_merge_matrix_part_files_unions_sparse_feature_schemas(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            shard0 = out / "_mmgen_shards" / "shard_000_of_002"
            shard1 = out / "_mmgen_shards" / "shard_001_of_002"
            for shard_dir, rows in [
                (shard0, [{"individual": "a1", "group": "g1", "sex": "FEMALE", "FeatA": 1.0}]),
                (shard1, [{"individual": "a2", "group": "g2", "sex": "MALE", "FeatB": 2.0}]),
            ]:
                work = shard_dir / "_mmgen_work"
                work.mkdir(parents=True)
                (work / SHARD_MATRIX_DONE_FILENAME).write_text(json.dumps({"n_selected_files": 1}))
                pd.DataFrame(rows).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            path = merge_matrix_part_files(output_dir=str(out), num_shards=2)
            merged = pd.read_csv(path)
            self.assertEqual(merged["individual"].astype(str).tolist(), ["a1", "a2"])
            self.assertIn("FeatA", merged.columns)
            self.assertIn("FeatB", merged.columns)
            self.assertTrue(pd.isna(merged.loc[merged["individual"] == "a1", "FeatB"]).all())
            self.assertTrue(pd.isna(merged.loc[merged["individual"] == "a2", "FeatA"]).all())

    def test_merge_matrix_parts_reorders_columns_canonically(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            shard0 = out / "_mmgen_shards" / "shard_000_of_002"
            shard1 = out / "_mmgen_shards" / "shard_001_of_002"
            for shard_dir, rows in [
                (shard0, [{"individual": "a1", "group": "g1", "sex": "FEMALE", "TimeBudget_x": 0.1, "Spatial_z": 1.0}]),
                (shard1, [{"individual": "a2", "group": "g2", "sex": "MALE", "TimeBudget_x": 0.2, "KinematicBB_Pose+Orient_Groom-SideLeg+Ceiling_speed_mean_mean": 5.0, "Spatial_z": 2.0}]),
            ]:
                work = shard_dir / "_mmgen_work"
                work.mkdir(parents=True)
                (work / SHARD_MATRIX_DONE_FILENAME).write_text(json.dumps({"n_selected_files": 1}))
                pd.DataFrame(rows).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            path = merge_matrix_part_files(output_dir=str(out), num_shards=2)
            merged = pd.read_csv(path)
            self.assertEqual(
                merged.columns.tolist(),
                [
                    "individual",
                    "group",
                    "sex",
                    "TimeBudget_x",
                    "KinematicBB_Pose+Orient_Groom-SideLeg+Ceiling_speed_mean_mean",
                    "Spatial_z",
                ],
            )


    def test_finalize_shard_temp_store_honors_outer_shard_lock(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir(); pos.mkdir(); out.mkdir()

            shard_dir = out / "_mmgen_shards" / "shard_000_of_002"
            work_dir = shard_dir / "_mmgen_work"
            work_dir.mkdir(parents=True)
            (shard_dir / "temp_store.h5").write_text("")
            self._write_shard_done(work_dir)
            (work_dir / SHARD_MANIFEST_FILENAME).write_text(json.dumps({
                "num_shards": 2,
                "shard_index": 0,
                "shard_by": "row_identity",
                "n_selected_files": 1,
                "selected_files": ["/frozen/path/animal_A_movie0.csv"],
            }))
            (work_dir / "_mmgen_shard.lock").mkdir()

            with self.assertRaises(RuntimeError):
                finalize_shard_temp_store(
                    inference_csv_dir=str(inf),
                    poscalc_dir=str(pos),
                    output_dir=str(out),
                    min_bout=5,
                    edge_skip=5,
                    inf_tag="_normalized_inference",
                    runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,
                    num_shards=2,
                    shard_index=0,
                    shard_by="row_identity",
                )

    def test_merge_matrix_part_files_rejects_inconsistent_runtime_settings(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            settings_a = {
                "make_measurements": True,
                "calculate_occupancy": True,
                "calculate_postural_features": True,
                "make_detailed_transition_outputs": True,
                "filter_features_by_bout_count": True,
                "min_bout_count_for_features": 6,
                "min_bout_count_for_transitions": 6,
                "chamber_diameter_px": 530,
                "mm_per_px": 0.0169896744624207,
                "video_fps": 40.0,
                "kin_frames_signal_cols": [],
            }
            settings_b = dict(settings_a)
            settings_b["video_fps"] = 60.0
            for i, settings in [(0, settings_a), (1, settings_b)]:
                shard_dir = out / "_mmgen_shards" / f"shard_{i:03d}_of_002"
                work = shard_dir / "_mmgen_work"
                work.mkdir(parents=True)
                (work / SHARD_MATRIX_DONE_FILENAME).write_text(json.dumps({
                    "n_selected_files": 1,
                    "package_version": "0.test",
                    "matrix_schema_version": 1,
                    "min_bout": 5,
                    "edge_skip": 5,
                    "inf_tag": "_normalized_inference",
                    "file_filter": None,
                    "runtime_settings": settings,
                }))
                (work / "SHARD_DONE.json").write_text(json.dumps({
                    "num_shards": 2,
                    "shard_by": "row_identity",
                    "n_inference_files_selected": 1,
                    "n_committed_files": 1,
                }))
                pd.DataFrame([
                    {"individual": f"a{i}", "group": f"g{i}", "sex": "FEMALE", "FeatA": float(i)}
                ]).to_csv(shard_dir / "MASTER_FEATURE_MATRIX.csv", index=False)

            with self.assertRaises(RuntimeError):
                merge_matrix_part_files(output_dir=str(out), num_shards=2)


if __name__ == "__main__":
    unittest.main()
