"""Check MMGen unit conversions."""


import unittest
import pandas as pd
from autobehaver.postinf.mmgen.feature_matrix import _infer_unit_for_feature_name
from autobehaver.postinf.mmgen.units import _convert_kinematic_columns_to_real_units

class TestUnits(unittest.TestCase):
    def test_convert_units_scaling(self):
        df = pd.DataFrame({
            "speed_inst": [1.0],
            "acceleration_inst": [1.0],
            "jerk_inst": [1.0],
            "angular_velocity_inst": [1.0],
            "angular_acceleration_inst": [1.0],
            "angular_jerk_inst": [1.0],
            "curvature_inst": [1.0],
        })
        out = _convert_kinematic_columns_to_real_units(df, mm_per_px=0.1, fps=40.0)
        self.assertAlmostEqual(out.loc[0, "speed_inst"], 4.0, places=6)
        self.assertAlmostEqual(out.loc[0, "acceleration_inst"], 160.0, places=6)
        self.assertAlmostEqual(out.loc[0, "jerk_inst"], 6400.0, places=6)
        self.assertAlmostEqual(out.loc[0, "angular_velocity_inst"], 40.0, places=6)
        self.assertAlmostEqual(out.loc[0, "angular_acceleration_inst"], 1600.0, places=6)
        self.assertAlmostEqual(out.loc[0, "angular_jerk_inst"], 64000.0, places=6)
        self.assertAlmostEqual(out.loc[0, "curvature_inst"], 10.0, places=6)



    def test_unit_manifest_labels_curvature_and_postural_asymmetry(self):
        self.assertEqual(_infer_unit_for_feature_name("KinematicBB_Pose_Walking_curvature_min_mean"), ("rad/mm", "curvature"))
        self.assertEqual(_infer_unit_for_feature_name("Postural_Floor_overall_asymmetry_LR_mean"), ("mm", "linear distance"))

if __name__ == "__main__":
    unittest.main()
