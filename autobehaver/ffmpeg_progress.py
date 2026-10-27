"""FFmpeg progress without changing encoding options or accumulating stdout."""
import queue
import subprocess
import tempfile
import threading
import time

from .progress import Progress


def run_ffmpeg(command, *, total_frames, desc, timeout):
    command = [command[0], '-progress', 'pipe:1', '-nostats', *command[1:]]
    events = queue.Queue(maxsize=4)
    with tempfile.TemporaryFile(mode='w+b') as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        def read():
            for line in process.stdout:
                if line.startswith('frame='):
                    try:
                        events.put_nowait(int(line.split('=', 1)[1]))
                    except (ValueError, queue.Full):
                        pass
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        start = time.monotonic()
        try:
            with Progress(total=total_frames, desc=desc) as bar:
                while process.poll() is None or not events.empty():
                    if time.monotonic() - start > timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        value = events.get(timeout=.2)
                        bar.update(max(0, value-bar.n))
                    except queue.Empty:
                        bar.refresh()
                code = process.wait()
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell()-65536))
                stderr = errors.read().decode('utf-8', errors='replace')
                if code:
                    raise subprocess.CalledProcessError(code, command, stderr=stderr)
                if total_frames is not None:
                    bar.update(max(0, total_frames-bar.n))
                return subprocess.CompletedProcess(command, code, stdout='', stderr=stderr)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            reader.join(timeout=5)
            process.stdout.close()
