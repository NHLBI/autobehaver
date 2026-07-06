"""Smoke tests for standardized measurement extraction (mmgen)."""


import unittest
import numpy as np
import pandas as pd
from autobehaver.postinf.mmgen.measurements import extract_standardized_measurements_raw

class TestMeasurementsSmoke(unittest.TestCase):
    def test_extract_standardized_measurements_raw_basic(self):
        # 3 frames: one for each required orientation
        masked_poses = np.array(["Standing", "Standing", "Standing"], dtype=object)
        masked_orients = np.array(["Floor", "Wall-Side", "Ceiling"], dtype=object)

        # Minimal poscalc df containing all keypoint cols referenced by measurement_definitions
        cols = [
            "Antennae_x","Antennae_y","AbdomenTip_x","AbdomenTip_y",
            "EyeRight_x","EyeRight_y","EyeLeft_x","EyeLeft_y",
            "Ocelli_x","Ocelli_y","Mouthparts_x","Mouthparts_y",
            "ThoraxLeft_x","ThoraxLeft_y","ThoraxRight_x","ThoraxRight_y",
            "ThoraxTop_x","ThoraxTop_y","ThoraxBottom_x","ThoraxBottom_y",
            "AbdomenLeft_x","AbdomenLeft_y","AbdomenRight_x","AbdomenRight_y",
            "AbdomenTop_x","AbdomenTop_y","AbdomenBottom_x","AbdomenBottom_y",
            "WingLeft_x","WingLeft_y","WingRight_x","WingRight_y",
            "body_area_top","body_area_side",
        ]
        poscalc_df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in cols})

        raw = extract_standardized_measurements_raw(
            masked_poses, masked_orients, poscalc_df,
            individual_id="id1", group_label="grp", sex="FEMALE"
        )

        # Metadata keys always present
        self.assertEqual(raw["individual_id"], "id1")
        self.assertEqual(raw["group_label"], "grp")
        self.assertEqual(raw["sex"], "FEMALE")

        # Expect at least these measurement keys to be collected
        for k in ["length_top", "length_side", "length_bottom", "body_area_top", "body_area_side"]:
            self.assertIn(k, raw)
            self.assertFalse(raw[k].empty)

if __name__ == "__main__":
    unittest.main()
