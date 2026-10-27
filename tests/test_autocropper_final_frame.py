"""Tiny real encodes preserve the final frame and drifting source timestamps."""

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from autobehaver.preinf.autocropper.cropping import (
    _get_ffmpeg_executable, crop_single_chamber, crop_chambers_in_parallel,
)


def decoded_times(path):
    capture = cv2.VideoCapture(str(path))
    times = []
    try:
        while capture.read()[0]:
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        capture.release()
    return np.asarray(times)


class CropFinalFrameTest(unittest.TestCase):
    def test_single_and_shared_encoders_preserve_all_frames_and_timestamps(self):
        with tempfile.TemporaryDirectory(prefix="ab_crop_") as directory:
            root = Path(directory)
            for drift in (False, True):
                with self.subTest(drift=drift):
                    source = root / f"source_{drift}.mp4"
                    filter_args = ["-vf", r"settb=1/1000000,setpts=N*24900+mod(N\,2)*5"] if drift else []
                    subprocess.run([_get_ffmpeg_executable(), "-v", "error", "-nostdin", "-f", "lavfi",
                        "-i", "testsrc2=size=96x96:rate=40:duration=0.8", *filter_args,
                        "-c:v", "libx264", "-threads", "1", "-bf", "0", "-vsync", "0",
                        "-enc_time_base", "1:1000000", "-use_editlist", "0", str(source)],
                        check=True, timeout=30, capture_output=True)
                    before = decoded_times(source)
                    self.assertEqual(len(before), 32)
                    single = root / f"single_{drift}.mp4"
                    self.assertTrue(crop_single_chamber(str(source), single, 48, 48, 64,
                                                       timeout=30, crf=16))
                    shared = root / f"shared_{drift}.mp4"
                    args = SimpleNamespace(timeout=30, crop_crf=16, ffmpeg_threads=1, max_workers=1)
                    result = crop_chambers_in_parallel(source, {"chamber1": (48, 48)},
                        {"chamber1": shared}, 64, args)
                    self.assertEqual(result["chamber1"], shared.name)
                    for output in (single, shared):
                        after = decoded_times(output)
                        self.assertEqual(len(after), 32)
                        np.testing.assert_allclose(after, before, rtol=0, atol=0.002)
                        capture = cv2.VideoCapture(str(output))
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 31)
                        self.assertTrue(capture.read()[0])
                        capture.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
