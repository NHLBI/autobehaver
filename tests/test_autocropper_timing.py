"""Real encodes: preserve frame identity and timing for camera timestamp drift."""
import subprocess

import cv2
import numpy as np
import pytest

from autobehaver.preinf.autocropper.cropping import crop_single_chamber, _get_ffmpeg_executable
from autobehaver.preinf.autocropper.thread_limits import open_video_capture


def read_times(path):
    cap = open_video_capture(path)
    assert cap.isOpened()
    times = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    return np.array(times)


@pytest.mark.parametrize('drift', [False, True])
def test_crop_preserves_every_frame_and_timestamp(tmp_path, drift):
    source = tmp_path / 'source.mp4'
    ffmpeg = _get_ffmpeg_executable()
    # The drifting source retains 1200 frames but its timestamps run slightly
    # faster than the nominal 40 frames per second.
    filter_args = ['-vf', r'settb=1/1000000,setpts=N*24900+mod(N\,2)*5'] if drift else []
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                    'testsrc2=size=96x96:rate=40:duration=30', *filter_args,
                    '-c:v', 'libx264', '-threads', '1', '-bf', '0', '-vsync', '0',
                    '-enc_time_base', '1:1000000', str(source)], check=True, timeout=30)
    output = tmp_path / 'chamber1.mp4'
    assert crop_single_chamber(str(source), output, 48, 48, 64, timeout=30, crf=16)
    before, after = read_times(source), read_times(output)
    assert len(before) == len(after) == 1200
    np.testing.assert_allclose(after, before, rtol=0, atol=0.002)

    if drift:
        # Constant-frame-rate conversion changes this source's timing.
        old = tmp_path / 'old_default.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-threads', '1', '-i', str(source),
                        '-vf', 'crop=64:64:16:16', '-filter_threads', '1', '-c:v', 'libx264',
                        '-threads', '1', str(old)], check=True, timeout=30)
        assert len(read_times(old)) < len(before)


def test_video_decoder_thread_cap_is_applied_at_open(tmp_path, monkeypatch):
    path = tmp_path / 'video.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 40, (64, 64))
    assert writer.isOpened()
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setenv('OPENCV_FOR_THREADS_NUM', '1')
    cap = open_video_capture(path)
    try:
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_N_THREADS) == 1
        assert cap.read()[0]
    finally:
        cap.release()
