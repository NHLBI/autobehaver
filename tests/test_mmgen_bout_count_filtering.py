import unittest

import pandas as pd

from autobehaver.postinf.mmgen.feature_matrix import _filter_features_by_bout_count


class TestMMGenBoutCountFiltering(unittest.TestCase):
    def test_tw_kinematics_are_not_filtered_by_bout_count(self):
        master = pd.DataFrame(
            {
                "individual": ["fly1"],
                "group": ["grp"],
                "BoutSummary_Pose_Walking_bout_duration_mean_s": [2.0],
                "KinematicBB_Pose_Walking_speed_mean_mean": [3.0],
                "KinematicTW_Pose_Walking_speed_mean": [4.0],
            }
        )
        bout_counts = pd.DataFrame(
            {
                "individual": ["fly1"],
                "BoutSummary_Pose_Walking_bout_count": [1],
            }
        )

        out = _filter_features_by_bout_count(
            master.copy(),
            min_bout_count=6,
            min_transition_count=6,
            bout_count_df=bout_counts,
        )

        self.assertTrue(pd.isna(out.loc[0, "BoutSummary_Pose_Walking_bout_duration_mean_s"]))
        self.assertTrue(pd.isna(out.loc[0, "KinematicBB_Pose_Walking_speed_mean_mean"]))
        self.assertEqual(float(out.loc[0, "KinematicTW_Pose_Walking_speed_mean"]), 4.0)


if __name__ == "__main__":
    unittest.main()
