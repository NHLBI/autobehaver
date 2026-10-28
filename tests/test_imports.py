"""Check core imports and deferred loading of rendering dependencies."""


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
        # Public summary functions must remain available.
        self.assertTrue(hasattr(bouts, "extract_bouts_from_label_sequence"))
        self.assertTrue(hasattr(feature_matrix, "create_feature_matrix"))

    def test_infmovie_imports(self):
        # infmovie should be importable without OpenCV at import time
        from autobehaver.postinf.infmovie import movies

        # The rendering module defers OpenCV loading until rendering is requested.
        # Inspect its bindings because other tests may already have loaded OpenCV.
        self.assertFalse("cv2" in movies.__dict__)

if __name__ == "__main__":
    unittest.main()
