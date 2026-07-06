"""Import-level smoke tests.

These tests ensure that key modules can be imported without pulling in optional dependencies
(e.g., OpenCV) unless explicitly requested."""


import sys
import unittest

class TestImports(unittest.TestCase):
    def test_mmgen_imports(self):
        # Core package import
        import autobehaver  # noqa: F401

        # mmgen submodules should import cleanly
        from autobehaver.postinf.mmgen import (
            pipeline, worker, feature_matrix, bouts, kinematics, transitions,
            occupancy, measurements, postural, time_budgets, hdf_store
        )
        # Basic sanity: key modules have expected attributes
        self.assertTrue(hasattr(bouts, "extract_bouts_from_label_sequence"))
        self.assertTrue(hasattr(feature_matrix, "create_feature_matrix"))

    def test_infmovie_imports(self):
        # infmovie should be importable without OpenCV at import time
        from autobehaver.postinf.infmovie import movies

        # Movie module should not import cv2 at module import time (lazy import).
        # (If some other module already imported cv2, this might not hold, so we check
        # that movies itself didn't bind cv2.)
        self.assertFalse("cv2" in movies.__dict__)

if __name__ == "__main__":
    unittest.main()
