"""Normalization must leave subsequent logging and interpreter exit working."""
import logging
import os
from pathlib import Path
import subprocess
import sys

from autobehaver.preinf.keyprenorm.logging_utils import setup_logging_queue


def test_listener_restores_handlers_and_delivers_later_warnings(caplog):
    root = logging.getLogger()
    previous_handlers, previous_level = list(root.handlers), root.level
    queue, listener = setup_logging_queue(log_level='INFO', log_file=None)
    logging.info('during normalization')
    listener.stop()
    listener.stop()  # Cleanup can safely be requested twice.
    assert root.handlers == previous_handlers
    assert root.level == previous_level
    assert queue._closed
    assert not queue._thread.is_alive()
    with caplog.at_level(logging.WARNING):
        logging.warning('after normalization')
    assert 'after normalization' in caplog.text


def test_process_exits_after_logging_beyond_normalization():
    code = '''
import logging
from autobehaver.preinf.keyprenorm.logging_utils import setup_logging_queue
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger().setLevel(logging.INFO)
queue, listener = setup_logging_queue(log_level='INFO', log_file=None)
listener.stop()
for i in range(2000):
    logging.info('after normalization %s', i)
print('clean exit')
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [
        str(Path(__file__).resolve().parents[1]), env.get('PYTHONPATH')]))
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'clean exit' in result.stdout


def test_synthetic_normalization_restores_logging_and_exits():
    code = '''
import logging
import unittest
from tests.test_keyprenorm_synthetic import TestKeypreNormSynthetic
root = logging.getLogger()
root.addHandler(logging.NullHandler())
root.setLevel(logging.INFO)
previous_handlers, previous_level = list(root.handlers), root.level
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestKeypreNormSynthetic)
result = unittest.TextTestRunner().run(suite)
assert result.wasSuccessful()
assert root.handlers == previous_handlers
assert root.level == previous_level
for i in range(2000):
    logging.error('after synthetic normalization %s', i)
print('clean exit')
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [
        str(Path(__file__).resolve().parents[1]), env.get('PYTHONPATH')]))
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'clean exit' in result.stdout
