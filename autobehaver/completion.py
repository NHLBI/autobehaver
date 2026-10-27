"""Verify reusable outputs using content identities and completion receipts.

Paths, timestamps and source documentation are excluded from scientific
identity. Stage-specific content checks remain required for reuse.
"""
from __future__ import annotations

import ast
import contextvars
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import uuid
import threading


_RUN = contextvars.ContextVar('autobehaver_verified_run', default=None)


def file_signature(path):
    """Detect replacement or modification of a file within a verified invocation."""
    absolute = os.path.normcase(os.path.abspath(os.fspath(path)))
    stat = os.stat(native_path(absolute))
    return (absolute, stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


@contextmanager
def verified_identity_scope():
    """Reuse verified identities only inside this invocation, never across runs.

    Nested stage helpers share the cache. Changed/replaced files invalidate their
    cached identity automatically; producers can explicitly invalidate after writes.
    """
    if _RUN.get() is not None:
        yield _RUN.get()
        return
    state = {'files': {}, 'code': {}, 'lock': threading.RLock()}
    token = _RUN.set(state)
    try:
        yield state
    finally:
        _RUN.reset(token)


def verified_run(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with verified_identity_scope():
            return function(*args, **kwargs)
    return wrapped


def invalidate_digest(path):
    """Forget an output before/after publication, including deliberate retiming."""
    state = _RUN.get()
    if state is not None:
        absolute = os.path.normcase(os.path.abspath(os.fspath(path)))
        with state['lock']:
            state['files'].pop(absolute, None)


forget_identity = invalidate_digest


def current_identity_scope():
    return _RUN.get()


def remember_verified_digest(path, digest, signature_before=None):
    """Register a digest already verified by a streaming copy/publication helper."""
    signature = file_signature(path)
    if signature_before is not None and signature != signature_before:
        raise RuntimeError(f'File changed during streaming verification: {path}')
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Expected a verified SHA-256 digest')
    state = _RUN.get()
    if state is not None:
        with state['lock']:
            state['files'][signature[0]] = (signature, digest)


def native_path(path):
    path = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        return '\\\\?\\UNC\\' + path[2:] if path.startswith('\\\\') else '\\\\?\\' + path
    return path


def file_digest(path, *, chunk_size=1024 * 1024):
    signature = file_signature(path)
    state = _RUN.get()
    if state is not None:
        with state['lock']:
            cached = state['files'].get(signature[0])
            if cached is not None and cached[0] == signature:
                return cached[1]
    digest = hashlib.sha256()
    with open(native_path(path), 'rb') as stream:
        for block in iter(lambda: stream.read(chunk_size), b''):
            digest.update(block)
    if file_signature(path) != signature:
        raise RuntimeError(f'File changed during content verification: {path}')
    value = digest.hexdigest()
    if state is not None:
        with state['lock']:
            state['files'][signature[0]] = (signature, value)
    return value


def code_digest(paths):
    """Hash Python syntax, omitting documentation-only changes."""
    paths = sorted(map(Path, paths), key=lambda p: p.name)
    signatures = tuple(file_signature(path) for path in paths)
    state = _RUN.get()
    if state is not None and signatures in state['code']:
        return state['code'][signatures]
    class StripDocs(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            body = getattr(node, 'body', None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body.pop(0)
            return node
    values = []
    for path in paths:
        tree = StripDocs().visit(ast.parse(path.read_text(encoding='utf-8-sig')))
        values.append((path.name, ast.dump(tree, include_attributes=False)))
    value = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
    if tuple(file_signature(path) for path in paths) != signatures:
        raise RuntimeError('Scientific source changed during identity calculation')
    if state is not None:
        state['code'][signatures] = value
    return value


def request_identity(stage, inputs, settings, code_paths=()):
    return {'stage': stage, 'inputs': [file_digest(p) for p in inputs],
            'settings': settings, 'code': code_digest(code_paths)}


def receipt_path(output):
    return Path(str(output) + '.done.json')


def read_receipt(output):
    try:
        return json.loads(receipt_path(output).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def receipt_matches(output, identity, outputs=None):
    receipt = read_receipt(output)
    if not receipt or receipt.get('request') != identity:
        return False
    paths = [output] if outputs is None else outputs
    try:
        return receipt['outputs'] == [file_digest(p) for p in paths]
    except (OSError, KeyError):
        return False


def write_receipt(output, identity, outputs=None, **metadata):
    paths = [output] if outputs is None else outputs
    payload = {'version': 1, 'request': identity,
               'outputs': [file_digest(p) for p in paths], **metadata}
    destination = receipt_path(output)
    temporary = destination.with_name('.receipt-' + uuid.uuid4().hex[:12] + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(temporary, destination)


def preserve_previous(path):
    """Move a stale result and its receipt into a short archive with a name map."""
    path = Path(path)
    invalidate_digest(path)
    if not path.exists():
        return
    archive = path.parent / '_old'
    archive.mkdir(exist_ok=True)
    token = uuid.uuid4().hex[:12]
    target = archive / (token + path.suffix)
    os.replace(path, target)
    note = {'original_name': path.name, 'saved_name': target.name}
    old_receipt = receipt_path(path)
    if old_receipt.exists():
        os.replace(old_receipt, receipt_path(target))
    (archive / (token + '.name.json')).write_text(json.dumps(note), encoding='utf-8')
