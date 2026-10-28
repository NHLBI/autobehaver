import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from autobehaver.postinf.mmgen import cli
from autobehaver.postinf.mmgen.time_budgets import calculate_and_save_time_budgets
from autobehaver.postinf.mmgen.transitions import calculate_and_save_transition_analysis


class TestMMGenCliOutputModes(unittest.TestCase):
    def test_no_full_outputs_sets_emit_flags(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir()
            pos.mkdir()
            out.mkdir()

            captured = []

            def _fake_run(cfg):
                captured.append(cfg)

            with patch("autobehaver.postinf.mmgen.core.run_mmgen", side_effect=_fake_run):
                cli.main([
                    "--inference_csv_dir", str(inf),
                    "--poscalc_dir", str(pos),
                    "--output_dir", str(out),
                    "--no_full_outputs",
                ])

            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0].emit_master_matrix)
            self.assertFalse(captured[0].emit_full_outputs)

    def test_hyphenated_no_full_outputs_sets_emit_flags(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir()
            pos.mkdir()
            out.mkdir()
            captured = []

            def _fake_run(cfg):
                captured.append(cfg)

            with patch("autobehaver.postinf.mmgen.core.run_mmgen", side_effect=_fake_run):
                cli.main(["--inference-csv-dir", str(inf), "--poscalc-dir", str(pos),
                          "--output-dir", str(out), "--no-full-outputs"])

            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0].emit_master_matrix)
            self.assertFalse(captured[0].emit_full_outputs)

    def test_refuses_when_no_outputs_requested(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir()
            pos.mkdir()
            out.mkdir()

            with self.assertRaises(SystemExit):
                cli.main([
                    "--inference_csv_dir", str(inf),
                    "--poscalc_dir", str(pos),
                    "--output_dir", str(out),
                    "--no_master_matrix",
                    "--no_full_outputs",
                ])


    def test_append_new_data_flag_reaches_config(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inf = base / "inf"
            pos = base / "pos"
            out = base / "out"
            inf.mkdir()
            pos.mkdir()
            out.mkdir()

            captured = []

            def _fake_run(cfg):
                captured.append(cfg)

            with patch("autobehaver.postinf.mmgen.core.run_mmgen", side_effect=_fake_run):
                cli.main([
                    "--inference_csv_dir", str(inf),
                    "--poscalc_dir", str(pos),
                    "--output_dir", str(out),
                    "--append_new_data",
                ])

            self.assertEqual(len(captured), 1)
            self.assertTrue(bool(captured[0].append_new_data))


class TestMMGenNoFullOutputsHelpers(unittest.TestCase):
    def test_time_budgets_can_skip_workbook_write(self):
        master_df = pd.DataFrame([
            {
                "animal_id": "a1",
                "individual": "a1_movie1",
                "group": "g1",
                "sex": "FEMALE",
                "category": "Pose",
                "bout_type": "Walking",
                "total_length_frames": 40,
            },
            {
                "animal_id": "a1",
                "individual": "a1_movie1",
                "group": "g1",
                "sex": "FEMALE",
                "category": "Orient",
                "bout_type": "Floor",
                "total_length_frames": 40,
            },
            {
                "animal_id": "a1",
                "individual": "a1_movie1",
                "group": "g1",
                "sex": "FEMALE",
                "category": "Pose+Orient",
                "bout_type": "Walking + Floor",
                "total_length_frames": 40,
            },
        ])

        with tempfile.TemporaryDirectory() as td:
            sheets = calculate_and_save_time_budgets(
                master_df,
                animal_total_frames={"a1": 100},
                output_dir=td,
                save_outputs=False,
            )
            self.assertIn("Pose (Overall)", sheets)
            self.assertFalse((Path(td) / "2_Time_Budgets.xlsx").exists())

    def test_transition_analysis_can_skip_output_files(self):
        aggregated = pd.DataFrame([
            {"individual": "a1_movie1", "group": "g1", "category": "Pose", "from_behavior": "Walking", "to_behavior": "Standing", "count": 3},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose", "from_behavior": "Standing", "to_behavior": "Walking", "count": 2},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose+Orient", "from_behavior": "Walking + Floor", "to_behavior": "Standing + Floor", "count": 3},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose+Orient", "from_behavior": "Standing + Floor", "to_behavior": "Walking + Floor", "count": 2},
        ])

        with tempfile.TemporaryDirectory() as td:
            result = calculate_and_save_transition_analysis(aggregated, td, save_outputs=False)
            self.assertFalse((Path(td) / "6_Pose-Only_Transitions.xlsx").exists())
            self.assertFalse((Path(td) / "7_Pose+Orient_Transitions.xlsx").exists())
            self.assertFalse((Path(td) / "6_Pose_All_Transitions_Tidy.csv").exists())
            self.assertGreater(len(result["pose_tidy_df"]), 0)
            self.assertGreater(len(result["pose_orient_tidy_df"]), 0)


    def test_transition_analysis_save_outputs_threads_run_transition_statistics(self):
        aggregated = pd.DataFrame([
            {"individual": "a1_movie1", "group": "g1", "category": "Pose", "from_behavior": "Walking", "to_behavior": "Standing", "count": 3},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose", "from_behavior": "Standing", "to_behavior": "Walking", "count": 2},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose", "from_behavior": "Walking", "to_behavior": "Standing", "count": 4},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose", "from_behavior": "Standing", "to_behavior": "Walking", "count": 1},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose+Orient", "from_behavior": "Walking + Floor", "to_behavior": "Standing + Floor", "count": 3},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose+Orient", "from_behavior": "Standing + Floor", "to_behavior": "Walking + Floor", "count": 2},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose+Orient", "from_behavior": "Walking + Floor", "to_behavior": "Standing + Floor", "count": 5},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose+Orient", "from_behavior": "Standing + Floor", "to_behavior": "Walking + Floor", "count": 2},
        ])

        with tempfile.TemporaryDirectory() as td:
            result = calculate_and_save_transition_analysis(
                aggregated,
                td,
                save_outputs=True,
                run_transition_statistics=False,
            )
            self.assertTrue((Path(td) / "6_Pose-Only_Transitions.xlsx").exists())
            self.assertTrue((Path(td) / "7_Pose+Orient_Transitions.xlsx").exists())
            self.assertFalse((Path(td) / "8_Transition_Comparisons").exists())
            self.assertGreater(len(result["pose_tidy_df"]), 0)
            self.assertGreater(len(result["pose_orient_tidy_df"]), 0)

    def test_transition_analysis_can_run_comparative_outputs_when_enabled(self):
        aggregated = pd.DataFrame([
            {"individual": "a1_movie1", "group": "g1", "category": "Pose", "from_behavior": "Walking", "to_behavior": "Standing", "count": 3},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose", "from_behavior": "Standing", "to_behavior": "Walking", "count": 2},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose", "from_behavior": "Walking", "to_behavior": "Standing", "count": 4},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose", "from_behavior": "Standing", "to_behavior": "Walking", "count": 1},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose+Orient", "from_behavior": "Walking + Floor", "to_behavior": "Standing + Floor", "count": 3},
            {"individual": "a1_movie1", "group": "g1", "category": "Pose+Orient", "from_behavior": "Standing + Floor", "to_behavior": "Walking + Floor", "count": 2},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose+Orient", "from_behavior": "Walking + Floor", "to_behavior": "Standing + Floor", "count": 5},
            {"individual": "a2_movie1", "group": "g2", "category": "Pose+Orient", "from_behavior": "Standing + Floor", "to_behavior": "Walking + Floor", "count": 2},
        ])

        with tempfile.TemporaryDirectory() as td:
            calculate_and_save_transition_analysis(
                aggregated,
                td,
                save_outputs=True,
                run_transition_statistics=True,
            )
            self.assertTrue((Path(td) / "8_Transition_Comparisons" / "Pose").exists())
            self.assertTrue((Path(td) / "8_Transition_Comparisons" / "Pose+Orient").exists())


if __name__ == "__main__":
    unittest.main()
