import io
import subprocess
import sys
import time

import pytest
from autobehaver.progress import Progress, progress
from autobehaver.ffmpeg_progress import run_ffmpeg


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_auto_terminal_and_redirected_output():
    terminal = Terminal()
    assert list(progress(range(10), file=terminal, mode="auto")) == list(range(10))
    assert '\r' in terminal.getvalue()
    log = io.StringIO()
    assert list(progress(range(10000), file=log)) == list(range(10000))
    assert '\r' not in log.getvalue()
    assert len(log.getvalue().splitlines()) == 2
    assert '10000/10000' in log.getvalue()


def test_sparse_log_no_unchanged_heartbeats(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('autobehaver.progress.time.monotonic', lambda: clock[0])
    log = io.StringIO()
    with Progress(total=100, file=log) as p:
        for tick in range(1, 1001):
            clock[0] = tick
            p.refresh()
        assert len(log.getvalue().splitlines()) == 1
        p.update(10)
        assert len(log.getvalue().splitlines()) == 2
        p.update(10)
        assert len(log.getvalue().splitlines()) == 2


def test_failure_and_disable(monkeypatch):
    log = io.StringIO()
    with pytest.raises(ValueError):
        with Progress(total=2, file=log):
            raise ValueError('actual failure')
    assert 'failed' in log.getvalue() and 'finished' not in log.getvalue()
    monkeypatch.setenv('AUTOBEHAVER_PROGRESS','off')
    log = io.StringIO()
    assert list(progress(range(3), file=log)) == [0,1,2]
    assert log.getvalue() == ''


def test_closed_stream_does_not_break_work():
    log=io.StringIO();log.close()
    assert list(progress(range(3), file=log)) == [0,1,2]


def test_prints_and_warnings_clear_live_line_and_restore_streams(monkeypatch):
    import logging
    output, errors = Terminal(), Terminal()
    monkeypatch.setattr(sys, 'stdout', output)
    monkeypatch.setattr(sys, 'stderr', errors)
    handler = logging.StreamHandler(errors)
    logger = logging.getLogger('progress-message-test')
    logger.addHandler(handler)
    try:
        with Progress(total=2, mode='bar') as bar:
            print('ordinary output')
            logger.warning('important warning')
            bar.update(2)
        assert 'ordinary output\n' in output.getvalue()
        assert 'important warning\n' in errors.getvalue()
        assert sys.stdout is output and sys.stderr is errors
        assert handler.stream is errors
    finally:
        logger.removeHandler(handler)


def test_weighted_items_do_not_change_work_or_claim_batches_as_windows():
    log = io.StringIO()
    batches = [[1, 2, 3], [4]]
    assert list(progress(batches, total=4, advance=len, file=log)) == batches
    assert '4/4 (100%)' in log.getvalue()


def _spawn_progress_worker(events, release):
    from autobehaver.progress import set_worker_queue
    set_worker_queue(events)
    with Progress(total=2, desc='Child frames', mode='log') as display:
        display.update(2)
    release.wait(10)


def test_spawned_worker_reports_to_parent_without_hanging():
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    events, release = ctx.Queue(maxsize=4), ctx.Event()
    child = ctx.Process(target=_spawn_progress_worker, args=(events, release))
    child.start()
    try:
        messages = [events.get(timeout=10), events.get(timeout=10)]
        assert 'Child frames' in messages[0]
        assert '2/2 (100%)' in messages[1]
    finally:
        release.set()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join()
        events.close()
        events.join_thread()
    assert child.exitcode == 0


def test_ffmpeg_actual_frames_and_equivalent_output(tmp_path):
    imageio = pytest.importorskip('imageio_ffmpeg')
    cv2 = pytest.importorskip('cv2')
    exe = imageio.get_ffmpeg_exe()
    original=tmp_path/'original.mp4';updated=tmp_path/'updated.mp4'
    command=[exe,'-y','-hide_banner','-loglevel','error','-f','lavfi','-i',
             'testsrc=size=64x64:rate=10','-t','1','-c:v','libx264','-threads','1',
             '-pix_fmt','yuv420p',str(original)]
    subprocess.run(command,check=True,capture_output=True)
    command[-1]=str(updated)
    run_ffmpeg(command,total_frames=10,desc='Test crop',timeout=30)
    # Metadata and encoded bytes are identical: only FFmpeg reporting changed.
    assert updated.read_bytes() == original.read_bytes()


def test_ffmpeg_failure_and_timeout(tmp_path):
    exe=pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_ffmpeg([exe,'-i',str(tmp_path/'missing.mp4')],total_frames=10,desc='Bad input',timeout=10)
    assert error.value.stderr
    with pytest.raises(subprocess.TimeoutExpired):
        run_ffmpeg([exe,'-re','-f','lavfi','-i','testsrc=size=64x64:rate=10',
                    '-t','30','-f','null','-'],total_frames=300,desc='Timeout',timeout=.2)
