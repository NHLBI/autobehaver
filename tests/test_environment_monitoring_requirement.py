"""The benchmark environment check must distinguish required and optional imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('required,expected', [(False, 0), (True, 1)])
def test_missing_cpu_monitoring_is_required_only_when_requested(monkeypatch, capsys, required, expected):
    path = Path(__file__).resolve().parents[1] / 'environment' / 'check_environment.py'
    spec = importlib.util.spec_from_file_location('environment_check_fixture', path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    def imports(name):
        if name in {'psutil', 'pynvml'}:
            raise ImportError('fixture: unavailable telemetry')
        return SimpleNamespace(__version__='fixture')

    monkeypatch.setattr(checker.importlib, 'import_module', imports)
    monkeypatch.setattr(checker, '_check_dlc_imports', lambda **kwargs: (True, True, {}))
    arguments = ['--dlc-only', '--skip-cli-help']
    if required:
        arguments.append('--require-monitoring')
    assert checker.main(arguments) == expected
    output = capsys.readouterr().out
    assert 'fixture: unavailable telemetry' in output
    assert 'GPU telemetry' in output
