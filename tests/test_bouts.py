"""Unit tests for bout extraction and bout-level summaries."""


import unittest
import numpy as np
from autobehaver.postinf.mmgen.bouts import (
    extract_bouts_from_label_sequence,
    _build_valid_label_array_from_bouts,
    calculate_bout_summary_features,
    calculate_sleep_features,
)

class TestBouts(unittest.TestCase):
    def test_extract_bouts_min_bout(self):
        seq = ["A","A","B","B","B","A", np.nan, "A","A","A"]
        bouts = extract_bouts_from_label_sequence(seq, min_bout=3)
        labels = [(b["label"], b["start"], b["end"]) for b in bouts]
        self.assertEqual(labels, [("B", 2, 4), ("A", 7, 9)])

    def test_extract_bouts_exempt_label(self):
        seq = ["Jump", np.nan, "Jump", "Jump"]
        bouts = extract_bouts_from_label_sequence(seq, min_bout=3, exempt_labels={"Jump"})
        labels = [(b["label"], b["start"], b["end"]) for b in bouts]
        self.assertEqual(labels, [("Jump", 0, 0), ("Jump", 2, 3)])

    def test_build_valid_array(self):
        bouts = [{"start":2, "end":4, "label":"B"}]
        arr = _build_valid_label_array_from_bouts(bouts, length=6)
        self.assertTrue(np.isnan(arr[0]))
        self.assertEqual(arr[2], "B")
        self.assertEqual(arr[4], "B")
        self.assertTrue(np.isnan(arr[5]))

    def test_bout_summary_features_basic(self):
        bouts = [
            {"start":2, "end":4, "label":"B"},
            {"start":7, "end":9, "label":"A"},
        ]
        df = calculate_bout_summary_features(bouts, total_frames=10, fps=1)
        self.assertEqual(set(df["bout_type"].tolist()), {"A","B"})
        row_b = df[df["bout_type"]=="B"].iloc[0]
        self.assertEqual(int(row_b["n_bouts"]), 1)
        self.assertEqual(int(row_b["total_length_frames"]), 3)
        # total_seconds=10, so frequency should be 0.1
        self.assertAlmostEqual(float(row_b["bout_frequency"]), 0.1, places=6)
        self.assertAlmostEqual(float(row_b["bout_duration_mean_s"]), 3.0, places=6)

    def test_sleep_features(self):
        bouts = [{"start":0, "end":4, "label":"Standing"}]  # length 5
        df = calculate_sleep_features(bouts, total_frames=10, fps=1, sleep_threshold_frames=4)
        self.assertFalse(df.empty)
        row = df.iloc[0]
        self.assertEqual(row["bout_type"], "Standing")
        self.assertEqual(int(row["n_sleep_bouts"]), 1)
        self.assertAlmostEqual(float(row["total_sleep_time_s"]), 5.0, places=6)
        self.assertEqual(int(row["total_length_frames"]), 5)

if __name__ == "__main__":
    unittest.main()
