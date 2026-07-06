import tempfile
import textwrap
import unittest
from pathlib import Path

from autobehaver.postinf.mmgen.cli import _load_config_defaults


class TestMMGenCliConfig(unittest.TestCase):
    def test_mmgen_config_loads_feature_toggles_and_bout_count_settings(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    mmgen:
                      inference_csv_dir: inf
                      poscalc_dir: pos
                      output_dir: out
                      make_measurements: false
                      calculate_occupancy: false
                      calculate_postural_features: false
                      generate_quiver_plots: false
                      run_transition_statistics: false
                      make_detailed_transition_outputs: false
                      filter_features_by_bout_count: false
                      min_bout_count_for_features: 11
                      min_bout_count_for_transitions: 17
                    """
                ).strip()
            )

            defaults = _load_config_defaults(str(cfg_path))

            self.assertFalse(bool(defaults["make_measurements"]))
            self.assertFalse(bool(defaults["calculate_occupancy"]))
            self.assertFalse(bool(defaults["calculate_postural_features"]))
            self.assertFalse(bool(defaults["generate_quiver_plots"]))
            self.assertFalse(bool(defaults["run_transition_statistics"]))
            self.assertFalse(bool(defaults["make_detailed_transition_outputs"]))
            self.assertFalse(bool(defaults["filter_features_by_bout_count"]))
            self.assertEqual(int(defaults["min_bout_count_for_features"]), 11)
            self.assertEqual(int(defaults["min_bout_count_for_transitions"]), 17)


if __name__ == "__main__":
    unittest.main()
