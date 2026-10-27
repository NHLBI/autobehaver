import pytest

from autobehaver.cli_summary import reported_stage, completion_details, validation_details
from autobehaver.validation import StageValidationReport


def test_success_summary_follows_cleanup_and_preserves_return(capsys):
    @reported_stage('Example')
    def run():
        completion_details(output='/results', unit='Files', counts={'processed': 3, 'reused': 2})
        try:
            return 42
        finally:
            print('cleanup finished', file=__import__('sys').stderr)
    assert run() == 42
    text = capsys.readouterr().err
    assert text.index('cleanup finished') < text.index('Example - complete')
    assert '3 processed | 2 reused' in text and 'Output location: /results' in text


def test_cleanup_error_never_prints_success(capsys):
    error = RuntimeError('final record could not be written')
    @reported_stage('Example')
    def run():
        try:
            completion_details(counts={'validated': 3})
        finally:
            raise error
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is error
    text = capsys.readouterr().err
    assert 'Example - failed' in text and 'Example - complete' not in text


def test_validation_failure_and_warning_are_reported(capsys):
    @reported_stage('Example')
    def run(fail):
        report = StageValidationReport.for_items('example', ['a'])
        report.add_outcome('a', 'reused')
        report.add_issue('registry', 'error' if fail else 'warning', 'Registry issue')
        validation_details(report, unit='Files', output='/results')
        report.raise_for_errors()
    run(False)
    assert 'complete with warnings' in capsys.readouterr().err
    with pytest.raises(Exception):
        run(True)
    text = capsys.readouterr().err
    assert 'Example - failed' in text and '1 reused' in text


def test_exit_codes_empty_selection_and_help_preserved(capsys):
    @reported_stage('DLC', exit_code=True)
    def empty():
        completion_details(status='nothing to process')
        return 2
    assert empty() == 2
    assert 'nothing to process' in capsys.readouterr().err
    @reported_stage('DLC', exit_code=True)
    def failure():
        return 1
    assert failure() == 1
    assert 'DLC - failed' in capsys.readouterr().err
    @reported_stage('CLI', active=False)
    def help_only():
        raise SystemExit(0)
    with pytest.raises(SystemExit) as caught:
        help_only()
    assert caught.value.code == 0 and capsys.readouterr().err == ''


def test_closed_stream_does_not_change_result(monkeypatch):
    import io
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr('sys.stderr', stream)
    assert reported_stage('Example')(lambda: 7)() == 7


@pytest.mark.parametrize('valid', [False, True])
def test_autocropper_summary_uses_final_validation(tmp_path, monkeypatch, capsys, valid):
    from argparse import Namespace
    from dataclasses import asdict
    from autobehaver.preinf.autocropper import pipeline
    from autobehaver.preinf.autocropper.config import AutocropperConfig
    from autobehaver.preinf.autocropper.core import run_autocropper
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'example.mp4').write_bytes(b'fixture')
    cfg = AutocropperConfig(input_dir=str(source), output_dir=str(tmp_path / 'output'),
                           require_exclusions=False, skip_labeling=True, max_retries=0)
    monkeypatch.setattr(pipeline, 'is_video_file_valid', lambda *a: valid)
    monkeypatch.setattr(pipeline, 'process_movies_in_parallel', lambda *a: None)
    monkeypatch.setattr(pipeline, 'verify_all_cropped_movies', lambda *a: [])
    monkeypatch.setattr(pipeline, 'ensure_user_permissions', lambda *a: None)
    if valid:
        run_autocropper(cfg=cfg, args=Namespace(**asdict(cfg)))
        text = capsys.readouterr().err
        assert 'Autocropper - complete' in text and '1 validated' in text
    else:
        from autobehaver.validation import StageValidationError
        with pytest.raises(StageValidationError):
            run_autocropper(cfg=cfg, args=Namespace(**asdict(cfg)))
        text = capsys.readouterr().err
        assert 'Autocropper - failed' in text and '1 failed' in text
        assert 'Autocropper - complete' not in text
