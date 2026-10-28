"""Public source retains resource instructions without private asset payloads."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autobehaver.source_inventory import source_files


class SourceDistributionTests(unittest.TestCase):
    def test_resource_index_survives_source_export(self):
        root = Path(__file__).resolve().parents[1]
        selected = {path.as_posix() for path in source_files(root)}
        self.assertIn("ASSET_SHA256SUMS", selected)
        self.assertIn("docs/MODEL_DISTRIBUTION.md", selected)
        self.assertFalse(any(path.startswith("autobehaver/resources/") for path in selected))
        checksums = (root / "ASSET_SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(checksums), 4)
        for line in checksums:
            digest, relative = line.split(None, 1)
            self.assertEqual(len(digest), 64)
            self.assertTrue(relative.startswith("autobehaver/resources/"))

    def test_local_assets_and_caches_do_not_enter_source(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["README.md", "ASSET_SHA256SUMS", "autobehaver/__init__.py",
                         "autobehaver/resources/model.pth", "autobehaver/resources/norm_params.csv",
                         "autobehaver/__pycache__/module.pyc", "tests/pytest-cache-files-example/state.json",
                         ".pytest_cache/state.json", "tests/test_example.py"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            selected = {path.as_posix() for path in source_files(root)}
            self.assertEqual(selected, {"README.md", "ASSET_SHA256SUMS",
                                       "autobehaver/__init__.py", "tests/test_example.py"})


if __name__ == "__main__":
    unittest.main()
