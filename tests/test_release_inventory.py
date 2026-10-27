"""Release coverage detects source changes and omits separately distributed assets."""
import importlib.util
from pathlib import Path

import pytest


def test_release_inventory_detects_changed_and_new_source(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/release_inventory.py'
    spec = importlib.util.spec_from_file_location('release_inventory_fixture', script)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    source = tmp_path / 'autobehaver'
    source.mkdir()
    (source / '_version.py').write_text('__version__ = "1.2.3"\n')
    (source / 'resources').mkdir()
    (source / 'resources' / 'private.py').write_text('External assets are not source\n')
    result = release.update_or_verify(tmp_path, write=True)
    assert result['version'] == '1.2.3'
    assert result['file_count'] == 1
    release.update_or_verify(tmp_path)
    (source / '_version.py').write_text('__version__ = "1.2.4"\n')
    with pytest.raises(ValueError, match='current source/version'):
        release.update_or_verify(tmp_path)
    release.update_or_verify(tmp_path, write=True)
    (source / 'new_module.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='current source/version'):
        release.update_or_verify(tmp_path)
    release.update_or_verify(tmp_path, write=True)
    (tmp_path / 'SHA256SUMS').write_text('outdated\n')
    with pytest.raises(ValueError, match='SHA256SUMS'):
        release.update_or_verify(tmp_path)
