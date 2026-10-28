"""Check feature-matrix constants and derived relationships."""


import unittest
import types
from autobehaver.postinf.mmgen import feature_matrix as fm

class TestFeatureMatrixConstants(unittest.TestCase):
    def test_constants_present(self):
        for name in [
            "SPATIAL_R_NORM_CENTER_THRESHOLD",
            "SPATIAL_R_NORM_PERIPHERY_THRESHOLD",
            "MIN_CONTEXT_DURATION_S",
            "MIN_CONTEXT_FRAMES",
            "GROOM_EPISODE_MAX_NON_GROOM_GAP_S",
            "GROOM_EPISODE_MAX_GAP_FRAMES",
            "DEFAULT_VIDEO_FPS",
            "DEFAULT_MIN_CONTEXT_FRAMES",
            "DEFAULT_GROOM_EPISODE_MAX_GAP_FRAMES",
        ]:
            self.assertTrue(hasattr(fm, name), f"Missing constant: {name}")

        # Derived constants must agree with their source definitions.
        self.assertEqual(int(fm.MIN_CONTEXT_FRAMES), round(float(fm.MIN_CONTEXT_DURATION_S) * float(fm.DEFAULT_VIDEO_FPS)))
        self.assertEqual(int(fm.GROOM_EPISODE_MAX_GAP_FRAMES), round(float(fm.GROOM_EPISODE_MAX_NON_GROOM_GAP_S) * float(fm.DEFAULT_VIDEO_FPS)))

    def test_re_imported(self):
        # Sheet-name parsing requires the regular-expression module.
        self.assertTrue(hasattr(fm, "re"))
        self.assertIsInstance(fm.re, types.ModuleType)

if __name__ == "__main__":
    unittest.main()
