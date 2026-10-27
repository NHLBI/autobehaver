"""Short generated names and lossless name maps for Windows transfers."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

PATH_BUDGET = 240
TEMP_RESERVE = 24
_KNOWN_NAMES = {}
_DIRECTORY_INDEXES = {}


def _directory_key(directory):
    return os.path.normcase(os.path.abspath(os.fspath(directory)))


def _map_signature(directory):
    mapping_dir = Path(directory) / '_names'
    try:
        stat = mapping_dir.stat()
    except FileNotFoundError:
        return None
    # Windows can report the same directory timestamps after a new map arrives.
    # Native writers therefore publish a unique token atomically; compare its
    # content, rather than another timestamp. Older/imported maps need no token.
    try:
        generation = (mapping_dir / '.generation').read_text(encoding='ascii')
    except FileNotFoundError:
        generation = None
    return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, generation)


def _publish_map_generation(directory):
    """Make each completed native map write visible to other indexed readers."""
    mapping_dir = Path(directory) / '_names'
    fd, name = tempfile.mkstemp(prefix='.g_', suffix='.tmp', dir=mapping_dir)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='ascii', newline='\n') as stream:
            stream.write(uuid.uuid4().hex+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        _replace_map_file(temporary, mapping_dir / '.generation')
    finally:
        temporary.unlink(missing_ok=True)


def _indexes():
    from .completion import current_identity_scope
    scope = current_identity_scope()
    return scope.setdefault('name_maps', {}) if scope is not None else _DIRECTORY_INDEXES


def invalidate_name_maps(directory):
    key = _directory_key(directory)
    _DIRECTORY_INDEXES.pop(key, None)
    _indexes().pop(key, None)


def load_name_maps(directory, *, refresh=False):
    """Return checked name maps scoped to their directory.

    Reuse the directory index until its mapping signature changes. Directory
    scoping prevents unrelated projects from sharing a basename identity.
    """
    key = _directory_key(directory)
    signature = _map_signature(directory)
    indexes = _indexes()
    cached = indexes.get(key)
    if not refresh and cached is not None and cached[0] == signature:
        return cached[1]
    values = {}
    originals = {}
    # Each native map is immutable and published whole. A directory snapshot
    # remains usable while unrelated maps arrive; tag it with the starting
    # signature so the next discovery refreshes if publication overlapped.
    for path in (Path(directory) / '_names').glob('*.json'):
        item = _read_name_map(path)
        short_key = item['short_name'].casefold()
        if short_key in values and values[short_key] != item['original_name']:
            raise ValueError(f'Conflicting filename maps: {item["short_name"]}')
        values[short_key] = item['original_name']
        originals.setdefault(item['original_name'], set()).add(item['short_name'])
    indexes[key] = (signature, values, originals)
    _KNOWN_NAMES.update(values)
    return values


def _name_map_path(path):
    path = Path(path)
    key = hashlib.sha256(path.name.casefold().encode()).hexdigest()[:16]
    return path.parent / '_names' / (key + '.json')


def _read_name_map(path):
    item = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(item, dict)
            or not all(isinstance(item.get(key), str) and item[key]
                       for key in ('short_name', 'original_name'))):
        raise ValueError(f'Invalid filename map: {path}')
    if any(separator in item['short_name'] for separator in ('/', '\\')) or item['short_name'] in {'.', '..'}:
        raise ValueError(f'Filename map must identify a basename: {path}')
    return item


def _legacy_original_name(path):
    """Read only older maps; native records have a directly addressable key."""
    value = None
    for mapping in (path.parent / '_names').glob('*.json'):
        if len(mapping.stem) == 16 and all(char in '0123456789abcdef' for char in mapping.stem):
            continue
        item = _read_name_map(mapping)
        if item['short_name'].casefold() == path.name.casefold():
            if value is not None and value != item['original_name']:
                raise ValueError(f'Conflicting filename maps: {path.name}')
            value = item['original_name']
    return value


@contextmanager
def _name_map_lock(mapping):
    """Lock one map; the OS releases the lock if a writer exits unexpectedly.

    Keep the small lock file: deleting it could let waiting writers lock
    different file identities. Unrelated names have independent locks.
    """
    from .file_locks import FileLock
    with FileLock(mapping.with_suffix('.lock'), timeout_sec=30):
        yield


def _replace_map_file(temporary, destination):
    # Windows may briefly deny replacement while a reader has the generation
    # token open. Retry that sharing conflict, without restarting discovery.
    for attempt in range(21):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 20:
                raise
            time.sleep(0.05)


def _publish_name_map(path, original):
    path = Path(path)
    mapping = _name_map_path(path)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    payload = {'short_name': path.name, 'original_name': original}
    with _name_map_lock(mapping):
        if mapping.exists():
            existing = _read_name_map(mapping)
            if (existing['short_name'].casefold() != path.name.casefold()
                    or existing['original_name'] != original):
                raise ValueError(f'Conflicting filename map: {path.name}')
            return
        # Imported maps may use older, non-hashed map filenames.
        legacy = _legacy_original_name(path)
        if legacy is not None and legacy != original:
            raise ValueError(f'Conflicting filename map: {path.name}')
        descriptor, name = tempfile.mkstemp(prefix='.n_', suffix='.tmp', dir=mapping.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            _replace_map_file(temporary, mapping)
            _publish_map_generation(path.parent)
            invalidate_name_maps(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def compact_component(name, limit=100):
    if len(name) <= limit:
        return name
    stage_suffixes = ('_positional_calcs_normalized_inference.csv',
                     '_positional_calcs_normalized.h5', '_positional_calcs.csv',
                     '_normalized_inference.csv', '_normalized.h5',
                     '_inference.csv', '_labels.csv', '_labels.h5', '_logits.csv',
                     '_context_vectors.h5', '_cv.h5')
    suffix = next((s for s in stage_suffixes if name.endswith(s)), Path(name).suffix)
    token = hashlib.sha256(name.encode()).hexdigest()[:16]
    if limit < len(suffix) + 24:
        raise ValueError('Output root is too deep to preserve the stage filename suffix; choose a shorter root')
    return name[:max(0, limit - len(suffix) - 18)] + '__' + token + suffix


def generated_path(directory, name, *, reserve=TEMP_RESERVE, original=None):
    """Plan a usable full path and retain the complete original name on disk.

    Reuse an existing mapping when a project moves: its artifacts must not gain
    different names just because the new root has a different length. New names
    still use the available Windows path budget.
    """
    directory = Path(directory)
    logical_name = original if original is not None else name
    load_name_maps(directory)
    candidates = _indexes()[_directory_key(directory)][2].get(logical_name, set())
    existing = {candidate for candidate in candidates if (directory / candidate).exists()}
    candidates = existing or candidates
    if len(candidates) > 1:
        raise ValueError(f'Multiple mapped outputs identify {logical_name!r} in {directory}')
    if candidates:
        short = next(iter(candidates))
        if len(str((directory / short).absolute())) + reserve > PATH_BUDGET:
            raise ValueError(f'Existing output exceeds the portable path budget at {directory}; choose a shorter root.')
        return directory / short
    limit = min(140, PATH_BUDGET - reserve - len(str(directory.absolute())) - 1)
    if limit < 40:
        raise ValueError(f'Output folder is too deep: {directory}. Choose a shorter output root (for example C:\\ab).')
    short = compact_component(name, limit)
    if short != name or original is not None:
        directory.mkdir(parents=True, exist_ok=True)
        _publish_name_map(directory / short, original if original is not None else name)
    return directory / short


def record_original_name(path, name):
    _publish_name_map(Path(path), name)


def original_name(path):
    path = Path(path)
    mapping = _name_map_path(path)
    try:
        item = _read_name_map(mapping)
    except FileNotFoundError:
        # Retain compatibility with imported maps that use arbitrary filenames.
        return _legacy_original_name(path) or path.name
    if item['short_name'].casefold() != path.name.casefold():
        raise ValueError(f'Filename-map key collision: {path.name}')
    return item['original_name']


def normalized_path(source, directory):
    return generated_path(directory, Path(original_name(source)).stem + '_normalized.h5')


def prediction_path(source, directory, suffix='inference'):
    return generated_path(directory, Path(original_name(source)).stem + '_' + suffix + '.csv', reserve=40)


def recording_name(name):
    """Discard only DLC's generated scorer suffix, keeping biological identity."""
    import re
    # Native recordings end in a chamber token. DLC text within a genotype or
    # experiment is biological identity, even if it resembles a scorer name.
    chambers = list(re.finditer(r'(?:^|_)chamber\d+(?=_?DLC|_|$)', name, re.IGNORECASE))
    if chambers:
        boundary = chambers[-1].end()
        return name[:boundary] if re.match(r'_?DLC', name[boundary:]) else name
    # Retain the established generic/DLC-project naming convention for imported
    # recordings without chamber metadata; arbitrary embedded "DLCx" is not it.
    scorer = re.search(r'_?DLC(?:_|resnet|dlcrnet|mobilenet|hrnet|efficientnet)', name)
    return name[:scorer.start()] if scorer else name


def logical_recording_stem(name):
    """Resolve known stage suffixes while preserving the whole recording label."""
    stem = Path(name).stem
    suffixes = ('_positional_calcs_normalized_inference', '_positional_calcs_normalized',
                '_positional_calcs_inference',
                '_positional_calcs', '_normalized_inference', '_normalized',
                '_inference', '_labels', '_logits', '_context_vectors', '_cv')
    suffix = next((value for value in suffixes if stem.lower().endswith(value)), '')
    if suffix:
        stem = stem[:-len(suffix)]
    return recording_name(stem)
