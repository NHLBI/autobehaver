"""Unit tests for mmgen naming helpers."""


import unittest
from autobehaver.postinf.mmgen.naming import parse_group_and_individual, collapse_individual_id

class TestNaming(unittest.TestCase):
    def test_parse_and_collapse(self):
        fname = (
            "CONTROL-PLPdeltaR-plus_FEMALE_plp_deltaR_experiment_trial1_rpi-1_movie0_"
            "20250501124915PM_chamber5_positional_calcs_normalized_inference.csv"
        )
        group, individual, sex = parse_group_and_individual(fname)
        self.assertEqual(group, "CONTROL-PLPdeltaR-plus")
        self.assertEqual(sex, "FEMALE")
        self.assertEqual(individual, "CONTROL-PLPdeltaR-plus_FEMALE_rpi-1_trial1_movie0_chamber5")

        collapsed = collapse_individual_id(individual)
        self.assertEqual(collapsed, "CONTROL-PLPdeltaR-plus_FEMALE_rpi-1_trial1_chamber5")

if __name__ == "__main__":
    unittest.main()
