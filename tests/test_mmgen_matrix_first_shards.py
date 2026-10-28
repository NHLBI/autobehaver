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


_VALID_NAME = "CONTROL-Example-reference_FEMALE_example_experiment_trial1_rpi-1_movie0_20250501124915PM_chamber5_positional_calcs_normalized_inference.csv"


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
    def test_legacy_placeholder_cannot_certify_completed_processing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            work=root/'_mmgen_shards/shard_000_of_002/_mmgen_work'
            work.mkdir(parents=True)
            (work/'SHARD_DONE.json').write_text(json.dumps({'n_inference_files_selected':1,'n_committed_files':1}))
            (work.parent/'temp_store.h5').write_bytes(b'')
            with self.assertRaises((FileNotFoundError,RuntimeError)):
                finalize_shard_temp_store(inference_csv_dir=str(root),poscalc_dir=str(root),output_dir=str(root),
                    min_bout=5,edge_skip=0,inf_tag='_inference',runtime_settings=MMGenRuntimeSettings(),
                    file_filter=None,num_shards=2,shard_index=0,shard_by='row_identity')

    def test_legacy_placeholder_cannot_certify_matrix_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            shard=root/'_mmgen_shards/shard_000_of_002'
            work=shard/'_mmgen_work';work.mkdir(parents=True)
            pd.DataFrame({'group':['G'],'individual':['G_FEMALE_rpi-1_trial1_chamber1'],'sex':['FEMALE'],'f':[1.]}).to_csv(shard/'MASTER_FEATURE_MATRIX.csv',index=False)
            (work/'SHARD_MATRIX_DONE.json').write_text(json.dumps({'n_selected_files':1}))
            with self.assertRaises((FileNotFoundError,RuntimeError)):
                merge_matrix_part_files(output_dir=str(root),num_shards=2)

if __name__ == "__main__":
    unittest.main()
