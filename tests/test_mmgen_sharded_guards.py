import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autobehaver.postinf.mmgen.pipeline import run_post_inference_analysis
from autobehaver.postinf.mmgen.shard_merge import merge_shard_temp_stores


class _SerialPool:
    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)


class TestMMGenShardedGuards(unittest.TestCase):
    def test_empty_shard_writes_done_marker_and_empty_store(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir()
            pos.mkdir()
            out.mkdir()

            # trial1 -> shard 1 when num_shards=3 and shard_by='trial', so shard 0 is empty.
            name = "CONTROL-PLPdeltaR-plus_FEMALE_plp_deltaR_experiment_trial1_rpi-1_movie0_20250501124915PM_chamber5_positional_calcs_normalized_inference.csv"
            (inf / name).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\n")

            run_post_inference_analysis(
                str(inf),
                str(pos),
                str(out),
                5,
                5,
                "_normalized_inference",
                process_only=True,
                num_shards=3,
                shard_index=0,
                shard_by="trial",
            )

            shard_dir = out / "_mmgen_shards" / "shard_000_of_003"
            self.assertTrue((shard_dir / "temp_store.h5").exists())
            payload = json.loads((shard_dir / "_mmgen_work" / "SHARD_DONE.json").read_text())
            self.assertEqual(int(payload["n_inference_files_selected"]), 0)
            self.assertEqual(int(payload["n_committed_files"]), 0)

    def test_process_only_raises_when_selected_files_are_not_committed(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir()
            pos.mkdir()
            out.mkdir()

            name = "CONTROL-PLPdeltaR-plus_FEMALE_plp_deltaR_experiment_trial1_rpi-1_movie0_20250501124915PM_chamber5_positional_calcs_normalized_inference.csv"
            (inf / name).write_text("Smoothed Pose Label,Smoothed Orient Label\nWalking,Floor\n")

            with patch("autobehaver.postinf.mmgen.pipeline.Pool", _SerialPool), \
                 patch(
                     "autobehaver.postinf.mmgen.pipeline.process_video_data",
                     return_value=(
                         None,
                         None,
                         0,
                         None,
                         None,
                         None,
                         None,
                         None,
                         {
                             "source_file": name,
                             "error_stage": "postural",
                             "error_type": "NameError",
                             "error_message": "name 'mm_per_px' is not defined",
                             "traceback": "Traceback...",
                         },
                     ),
                 ):
                with self.assertRaises(RuntimeError) as ctx:
                    run_post_inference_analysis(
                        str(inf),
                        str(pos),
                        str(out),
                        5,
                        5,
                        "_normalized_inference",
                        process_only=True,
                    )

            self.assertIn("did not fully commit all selected inference files", str(ctx.exception))
            failed_csv = out / "_mmgen_work" / "FAILED_SOURCE_FILES.csv"
            self.assertTrue(failed_csv.exists())
            failed_df = pd.read_csv(failed_csv)
            self.assertEqual(failed_df["source_file"].astype(str).tolist(), [name])
            self.assertEqual(failed_df["error_stage"].astype(str).tolist(), ["postural"])
            self.assertEqual(failed_df["error_type"].astype(str).tolist(), ["NameError"])
            self.assertIn("mm_per_px", failed_df["error_message"].astype(str).iloc[0])

    def test_merge_requires_clean_done_markers(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            shards_root = out / "_mmgen_shards"
            for i, committed in [(0, 1), (1, 0)]:
                shard_dir = shards_root / f"shard_{i:03d}_of_002"
                work_dir = shard_dir / "_mmgen_work"
                work_dir.mkdir(parents=True, exist_ok=True)
                with pd.HDFStore(str(shard_dir / "temp_store.h5"), mode="w"):
                    pass
                payload = {
                    "num_shards": 2,
                    "shard_index": i,
                    "n_inference_files_selected": 1,
                    "n_committed_files": committed,
                }
                (work_dir / "SHARD_DONE.json").write_text(json.dumps(payload))

            with self.assertRaises(RuntimeError) as ctx:
                merge_shard_temp_stores(output_folder=str(out), num_shards=2, strict_missing=True)

            self.assertIn("n_committed_files=0", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
