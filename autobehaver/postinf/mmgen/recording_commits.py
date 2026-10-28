"""Store recording completion records as variable-length JSON in the HDF file.

Numerical tables retain their queryable layout. Completion records use byte
arrays with no reserved text width, and readers also accept older commit tables.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tables

from .hdf_store import HDF_KEYS

COMMIT_NODE = '/recording_commits'
COMMIT_FORMAT = 'autobehaver-json-v1'
_JSON_FIELDS = ('output_inventory', 'output_fingerprints')


def _native_node(store):
    node = store.get_node(COMMIT_NODE)
    if node is not None and (
        not isinstance(node, tables.VLArray)
        or not isinstance(node.atom, tables.UInt8Atom)
        or node.atom.shape != ()
        or getattr(node.attrs, 'record_format', None) != COMMIT_FORMAT
    ):
        raise RuntimeError('Unrecognized MMGen recording-completion storage')
    return node


def has_recording_commits(store) -> bool:
    """Check for modern or legacy completion storage, including empty storage."""
    return _native_node(store) is not None or HDF_KEYS['video_lengths_tmp'] in store


def _read_native(store) -> pd.DataFrame:
    node = _native_node(store)
    records = []
    if node is not None:
        for value in node:
            row = json.loads(value.tobytes().decode('utf-8'))
            if not isinstance(row, dict):
                raise RuntimeError('Invalid MMGen recording-completion record')
            # Keep the existing DataFrame contract used by validation and exports.
            for field in _JSON_FIELDS:
                if field in row:
                    row[field] = json.dumps(row[field], sort_keys=True, allow_nan=False)
            records.append(row)
    return pd.DataFrame(records)


def read_recording_commits(store) -> pd.DataFrame:
    """Read all completion records without converting or overwriting old stores."""
    frames = []
    legacy = HDF_KEYS['video_lengths_tmp']
    if legacy in store:
        frames.append(store.select(legacy))
    frames.append(_read_native(store))
    result = pd.concat(frames, ignore_index=True, sort=False)
    if 'source_file' in result and result['source_file'].astype(str).duplicated().any():
        raise RuntimeError('Duplicate MMGen recording completion records')
    return result


def append_recording_commits(store, frame: pd.DataFrame) -> None:
    """Append complete JSON records after their numerical tables are flushed."""
    if frame is None or frame.empty:
        return
    payloads = []
    for row in frame.to_dict('records'):
        for field in _JSON_FIELDS:
            if field in row and isinstance(row[field], str):
                row[field] = json.loads(row[field])
        payloads.append(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                   separators=(',', ':')).encode('utf-8'))
    node = _native_node(store)
    if node is None:
        node = store._handle.create_vlarray('/', COMMIT_NODE.lstrip('/'),
                                            atom=tables.UInt8Atom())
        node.attrs.record_format = COMMIT_FORMAT
    for payload in payloads:
        node.append(np.frombuffer(payload, dtype=np.uint8))
    node.flush()


def purge_native_recording_commit(store, source_file: str) -> None:
    """Filter modern records inside the caller's staged whole-file transaction."""
    if _native_node(store) is None:
        return
    frame = _read_native(store)
    if not frame.empty:
        if 'source_file' not in frame:
            raise RuntimeError('Invalid MMGen completion records: source_file missing')
        frame = frame.loc[~frame['source_file'].astype(str).eq(str(source_file))]
    store._handle.remove_node(COMMIT_NODE)
    append_recording_commits(store, frame)
