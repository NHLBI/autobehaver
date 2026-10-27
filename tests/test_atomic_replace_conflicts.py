"""Bookkeeping survives transient Windows sharing conflicts and retains real failures."""
import json

import pytest

from autobehaver import atomic_io


def sharing_error():
    error = PermissionError('fixture: Windows sharing conflict')
    error.winerror = 32
    return error


def test_manifest_uses_shared_retry_and_publishes_complete_record(tmp_path, monkeypatch):
    target = tmp_path / 'run.json'
    target.write_text('{"previous": true}')
    replace = atomic_io.os.replace
    calls = []

    def transient(source, destination):
        calls.append(destination)
        if len(calls) == 1:
            raise sharing_error()
        replace(source, destination)

    monkeypatch.setattr(atomic_io.os, 'replace', transient)
    monkeypatch.setattr(atomic_io.time, 'sleep', lambda delay: None)
    atomic_io.atomic_write_json({'status':'succeeded'}, target)
    assert json.loads(target.read_text()) == {'status':'succeeded'}
    assert len(calls) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ['run.json']


@pytest.mark.parametrize('windows_conflict,attempts', [(True, 6), (False, 1)])
def test_persistent_replace_failure_retains_previous_record(tmp_path, monkeypatch, windows_conflict, attempts):
    target = tmp_path / 'run.json'
    previous = b'{"previous": true}\n'
    target.write_bytes(previous)
    calls = []

    def denied(source, destination):
        calls.append(destination)
        raise sharing_error() if windows_conflict else PermissionError('not a Windows sharing conflict')

    monkeypatch.setattr(atomic_io.os, 'replace', denied)
    monkeypatch.setattr(atomic_io.time, 'sleep', lambda delay: None)
    with pytest.raises(PermissionError):
        atomic_io.atomic_write_json({'status':'succeeded'}, target)
    assert len(calls) == attempts
    assert target.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ['run.json']
