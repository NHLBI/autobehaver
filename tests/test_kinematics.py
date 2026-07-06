"""Unit tests for kinematics helpers (mmgen)."""


import unittest
import numpy as np
import pandas as pd
from autobehaver.postinf.mmgen.kinematics import (
    compute_head_movement_frame_signals,
    extract_bout_features_simple,
    _reconstruct_kinematic_table_from_aggregates,
    HEAD_KEYPOINTS,
    _safe_tail_means_from_array,
)

class TestKinematics(unittest.TestCase):
    def test_head_movement_frame_signals_drop_worst_likelihood(self):
        # Create 3 frames, 5 head keypoints
        n = 3
        df = pd.DataFrame(index=range(n))

        # Likelihoods: choose different worst keypoint per frame; third frame all NaN
        lk = np.array([
            [0.1, 0.9, 0.9, 0.9, 0.9],  # drop kp0
            [0.9, 0.1, 0.9, 0.9, 0.9],  # drop kp1
            [np.nan, np.nan, np.nan, np.nan, np.nan],  # drop none
        ], dtype=float)

        vel = np.array([1,2,3,4,5], dtype=float)
        acc = np.array([10,20,30,40,50], dtype=float)
        jerk = np.array([100,200,300,400,500], dtype=float)

        for j, kp in enumerate(HEAD_KEYPOINTS):
            df[f"{kp}_likelihood"] = lk[:, j]
            df[f"partvelocity_{kp}_inst"] = vel[j]
            df[f"partaccel_{kp}_inst"] = acc[j]
            df[f"partjerk_{kp}_inst"] = jerk[j]

        out_vel, out_acc, out_jerk = compute_head_movement_frame_signals(df, kinematic_suffix="_inst")

        # Expected means after dropping worst-likelihood
        exp0_vel = np.mean([2,3,4,5])        # 3.5
        exp1_vel = np.mean([1,3,4,5])        # 3.25
        exp2_vel = np.mean([1,2,3,4,5])      # 3.0 (no drop because likelihood all nan)

        self.assertAlmostEqual(float(out_vel[0]), exp0_vel, places=6)
        self.assertAlmostEqual(float(out_vel[1]), exp1_vel, places=6)
        self.assertAlmostEqual(float(out_vel[2]), exp2_vel, places=6)

        exp0_acc = np.mean([20,30,40,50])    # 35
        exp1_acc = np.mean([10,30,40,50])    # 32.5
        exp2_acc = np.mean([10,20,30,40,50]) # 30
        self.assertAlmostEqual(float(out_acc[0]), exp0_acc, places=6)
        self.assertAlmostEqual(float(out_acc[1]), exp1_acc, places=6)
        self.assertAlmostEqual(float(out_acc[2]), exp2_acc, places=6)

        exp0_jerk = np.mean([200,300,400,500])    # 350
        exp1_jerk = np.mean([100,300,400,500])    # 325
        exp2_jerk = np.mean([100,200,300,400,500])# 300
        self.assertAlmostEqual(float(out_jerk[0]), exp0_jerk, places=6)
        self.assertAlmostEqual(float(out_jerk[1]), exp1_jerk, places=6)
        self.assertAlmostEqual(float(out_jerk[2]), exp2_jerk, places=6)


    def test_tail_means_min_3_frame_guardrail(self):
        # For n=20, 5% would be 1 frame, but we enforce minimum 3 frames per tail.
        arr = np.arange(20, dtype=float)
        low, high = _safe_tail_means_from_array(arr, pmin=5, pmax=95, min_tail_frames=3)
        # Bottom 3: 0,1,2 -> mean=1; Top 3: 17,18,19 -> mean=18
        self.assertAlmostEqual(float(low), 1.0, places=6)
        self.assertAlmostEqual(float(high), 18.0, places=6)

        # For n=100, 5% is 5 frames; should use 5-frame tails
        arr2 = np.arange(100, dtype=float)
        low2, high2 = _safe_tail_means_from_array(arr2, pmin=5, pmax=95, min_tail_frames=3)
        # Bottom 5: 0..4 -> mean=2; Top 5: 95..99 -> mean=97
        self.assertAlmostEqual(float(low2), 2.0, places=6)
        self.assertAlmostEqual(float(high2), 97.0, places=6)

        # Too few samples -> NaN
        arr3 = np.array([1.0, 2.0])
        low3, high3 = _safe_tail_means_from_array(arr3, pmin=5, pmax=95, min_tail_frames=3)
        self.assertTrue(np.isnan(low3))
        self.assertTrue(np.isnan(high3))


    def test_bb_minmax_uses_percentile_thresholds_and_20_frame_guard(self):
        df = pd.DataFrame({"speed_inst": np.arange(20, dtype=float)})
        bouts = [
            {"start": 0, "end": 19, "label": "Walking", "bout_type": "Walking + Floor"}
        ]
        out = extract_bout_features_simple(
            df,
            bouts,
            individual_id="fly1",
            group_label="grp",
            sex_label="F",
            source_file="movie1",
            category="Pose+Orient",
            kinematic_suffix="_inst",
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.loc[0, "speed_min"]), float(np.nanpercentile(np.arange(20, dtype=float), 5)), places=6)
        self.assertAlmostEqual(float(out.loc[0, "speed_max"]), float(np.nanpercentile(np.arange(20, dtype=float), 95)), places=6)

        df_short = pd.DataFrame({"speed_inst": np.arange(19, dtype=float)})
        bouts_short = [
            {"start": 0, "end": 18, "label": "Walking", "bout_type": "Walking + Floor"}
        ]
        out_short = extract_bout_features_simple(
            df_short,
            bouts_short,
            individual_id="fly1",
            group_label="grp",
            sex_label="F",
            source_file="movie1",
            category="Pose+Orient",
            kinematic_suffix="_inst",
        )
        self.assertEqual(len(out_short), 1)
        self.assertTrue(np.isnan(out_short.loc[0, "speed_min"]))
        self.assertTrue(np.isnan(out_short.loc[0, "speed_max"]))

    def test_reconstruct_kinematics_includes_asym_and_drops_extrema_std(self):
        agg_df = pd.DataFrame(
            {
                "individual": ["fly1"],
                "group": ["grp"],
                "bout_type": ["Walking"],
                "frontleg_velocity_core_asym__sum": [3.0],
                "frontleg_velocity_core_asym__sum_sq": [2.2],
                "frontleg_velocity_core_asym__count": [6],
                "speed_min__sum": [12.0],
                "speed_min__sum_sq": [30.0],
                "speed_min__count": [6],
                "speed_max__sum": [60.0],
                "speed_max__sum_sq": [610.0],
                "speed_max__count": [6],
            }
        )
        out = _reconstruct_kinematic_table_from_aggregates({"_inst": agg_df}, min_bout_count_for_features=6)["_inst"]
        self.assertEqual(len(out), 1)
        self.assertIn("frontleg_velocity_core_asym_mean", out.columns)
        self.assertIn("frontleg_velocity_core_asym_std", out.columns)
        self.assertIn("speed_min_mean", out.columns)
        self.assertIn("speed_max_mean", out.columns)
        self.assertNotIn("speed_min_std", out.columns)
        self.assertNotIn("speed_max_std", out.columns)
        self.assertAlmostEqual(float(out.loc[0, "frontleg_velocity_core_asym_mean"]), 0.5, places=6)

if __name__ == "__main__":
    unittest.main()
