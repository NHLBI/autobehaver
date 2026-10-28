"""Dependency-free, bounded progress for terminals and batch logs.

AUTOBEHAVER_PROGRESS=auto|bar|log|off controls display only. Workers are
silent unless explicitly given a queue; only the parent draws the terminal.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import shutil
import sys
import time
import threading
import logging
from contextlib import contextmanager

_worker_queue = None
_output_lock = threading.RLock()
_active = []
_restores = []


def _clear_line():
    for display in _active:
        if getattr(display, '_line_width', 0):
            try:
                display.stream.write('\r' + ' ' * display._line_width + '\r')
            except (OSError, ValueError):
                pass
            display._line_width = 0


class _MessageStream:
    """Keep ordinary prints and logging readable while a live line is visible."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        with _output_lock:
            _clear_line()
            return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _activate(display):
    with _output_lock:
        if not _active:
            originals = (sys.stdout, sys.stderr)
            for name in ('stdout', 'stderr'):
                original = getattr(sys, name)
                proxy = _MessageStream(original)
                _restores.append((sys, name, original, proxy))
                setattr(sys, name, proxy)
            loggers = [logging.getLogger()] + [v for v in logging.Logger.manager.loggerDict.values() if isinstance(v, logging.Logger)]
            for logger in loggers:
                for handler in logger.handlers:
                    original = getattr(handler, 'stream', None)
                    if original is not None and any(original is value for value in originals):
                        proxy = _MessageStream(original)
                        _restores.append((handler, 'stream', original, proxy))
                        handler.stream = proxy
        _active.append(display)


def _deactivate(display):
    with _output_lock:
        if display in _active:
            _active.remove(display)
        if not _active:
            for owner, name, original, proxy in reversed(_restores):
                if getattr(owner, name) is proxy:
                    setattr(owner, name, original)
            _restores.clear()


def set_worker_queue(value):
    global _worker_queue
    _worker_queue = value
    if value is not None:
        value.cancel_join_thread()  # Progress must never delay worker shutdown.


class Progress:
    def __init__(self, iterable=None, *, total=None, desc='Working', file=None,
                 disable=False, mode=None, mininterval=0.5, leave=True, advance=None, **kwargs):
        self.iterable = iterable
        self.advance = advance
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                pass
        self.total, self.desc, self.n = total, desc, 0
        self.stream = file if file is not None else sys.stderr
        while isinstance(self.stream, _MessageStream):
            self.stream = self.stream.stream
        self.mode = mode or os.environ.get('AUTOBEHAVER_PROGRESS', 'auto')
        if self.mode not in ('auto', 'bar', 'log', 'off'):
            self.mode = 'auto'
        if disable:
            self.mode = 'off'
        self.worker = multiprocessing.parent_process() is not None
        if self.worker and _worker_queue is None:
            self.mode = 'off'
        if self.mode == 'auto':
            try:
                self.mode = 'bar' if getattr(self.stream, 'isatty', lambda: False)() else 'log'
            except (OSError, ValueError):
                self.mode = 'off'
        self.started = time.monotonic()
        self.last_time = self.started
        self.last_n = 0
        self.last_detail = ''
        self.interval = max(0.5, mininterval)
        self.closed = False
        self.detail = ''
        self._line_width = 0
        if self.mode == 'bar' and not self.worker:
            _activate(self)
        self._emit('started')

    def _text(self, state):
        elapsed = max(0, time.monotonic() - self.started)
        count = f'{self.n:g}' if self.total is None else f'{self.n:g}/{self.total:g}'
        fraction = min(1, self.n / self.total) if self.total else None
        bar = ''
        if fraction is not None:
            count += f' ({fraction:.0%})'
            if self.mode == 'bar':
                bar = '[' + '#' * int(16*fraction) + '-' * (16-int(16*fraction)) + '] '
        eta = ''
        if fraction and self.n >= 3 and elapsed >= 5 and fraction < 1:
            eta = f' | ~{elapsed*(1-fraction)/fraction:.0f}s remaining'
        if self.detail:
            return f'{self.desc} {count} | {self.detail}' + (f' | {state}' if state else '')
        return f'{self.desc} | {bar}{count} | {elapsed:.0f}s elapsed{eta}' + (f' | {self.detail}' if self.detail else '') + (f' | {state}' if state else '')

    def _emit(self, state=''):
        if self.mode == 'off':
            return
        text = self._text(state)
        try:
            if self.worker:
                _worker_queue.put_nowait(text)
            elif self.mode == 'bar':
                width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
                with _output_lock:
                    _clear_line()
                    self.stream.write('\r' + text[:width].ljust(width) + ('\n' if self.closed else ''))
                    self._line_width = 0 if self.closed else width
                    self.stream.flush()
            else:
                self.stream.write(text + '\n')
                self.stream.flush()
        except (OSError, ValueError, queue.Full):
            # A closed output stream or full telemetry queue cannot fail science.
            pass

    def update(self, n=1):
        self.n += n
        self.refresh()

    def refresh(self):
        now = time.monotonic()
        changed = self.n != self.last_n or self.detail != self.last_detail
        milestone = self.total and self.n - self.last_n >= self.total * .1
        due = ((now-self.last_time >= self.interval) if self.mode == 'bar' or self.worker
               else (changed and now-self.last_time >= 30 and (milestone or now-self.last_time >= 120)))
        if due and not self.closed:
            self._emit()
            self.last_time, self.last_n = now, self.n
            self.last_detail = self.detail

    def close(self, state=None):
        if not self.closed:
            self.closed = True
            self._emit(state or ('finished' if self.total is not None and self.n >= self.total else 'stopped'))
            if self.mode == 'bar' and not self.worker:
                _deactivate(self)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        self.close('interrupted' if kind is KeyboardInterrupt else 'failed' if kind else 'finished')

    def __iter__(self):
        try:
            for item in self.iterable:
                yield item
                self.update(self.advance(item) if self.advance else 1)
        except GeneratorExit:
            self.close('stopped')
            raise
        except BaseException:
            self.close('interrupted')
            raise
        else:
            self.close('finished')


def progress(iterable=None, **kwargs):
    return Progress(iterable, **kwargs)


def drain_events(events, display):
    """Display the latest worker activity; do not replay a backlog into logs."""
    while True:
        try:
            display.detail = events.get_nowait()
        except queue.Empty:
            break
    display.refresh()


@contextmanager
def phase(name):
    """Name opaque operations without claiming a measurable percentage or ETA."""
    stop = threading.Event()
    with Progress(desc=name) as display:
        def tick():
            while not stop.wait(1):
                display.refresh()
        timer = threading.Thread(target=tick, daemon=True)
        timer.start()
        try:
            yield
        finally:
            stop.set()
            timer.join()
