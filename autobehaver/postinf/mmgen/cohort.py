"""Freeze complete selected cohort membership before MMGen shard assignment.

This small shared inventory contains filenames and animal identities, not source
content. Each shard still validates/hashes only its own exact source files.
"""
from __future__ import annotations

import json
from pathlib import Path

from autobehaver.atomic_io import atomic_write_json
from autobehaver.file_locks import FileLock
from autobehaver.portable_paths import original_name
from .integrity import content_hash
from .naming import collapse_individual_id, parse_group_and_individual
from .sharding import assign_rows_to_shards, assign_shard


def cohort_path(shards_root, num_shards):
    return Path(shards_root) / f'SHARD_COHORT_{int(num_shards):03d}.json'


def build_shard_cohort(files, *, inference_csv_dir, num_shards, shard_by):
    records = []
    for name in sorted(set(map(str, files))):
        path = Path(inference_csv_dir) / name
        logical = original_name(path)
        row = collapse_individual_id(parse_group_and_individual(str(path))[1])
        records.append({'source_file': name, 'original_name': logical, 'row_identity': row})
    if str(shard_by).strip().lower() == 'row_identity':
        assignments = assign_rows_to_shards({r['source_file']: r['row_identity'] for r in records},
                                           num_shards=int(num_shards))
    else:
        assignments = {r['source_file']: assign_shard(r['original_name'], num_shards=int(num_shards),
                       strategy=str(shard_by)) for r in records}
    for record in records:
        record['shard_index'] = assignments[record['source_file']]
    payload = {'schema_version': 1, 'num_shards': int(num_shards),
               'shard_by': str(shard_by), 'files': records}
    return {**payload, 'cohort_sha256': content_hash(payload)}


def read_shard_cohort(shards_root, num_shards):
    path = cohort_path(shards_root, num_shards)
    if not path.is_file():
        raise RuntimeError(f'Shards lack a complete frozen cohort inventory; regenerate them in fresh outputs: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if (payload.get('schema_version') != 1 or payload.get('num_shards') != int(num_shards)
            or payload.get('cohort_sha256') != content_hash({k: v for k, v in payload.items() if k != 'cohort_sha256'})):
        raise RuntimeError(f'Shard cohort inventory is invalid or changed: {path}')
    records = payload.get('files')
    if (not isinstance(records, list)
            or any(not isinstance(row, dict) or not all(isinstance(row.get(key), str)
                for key in ('source_file', 'original_name', 'row_identity'))
                or not isinstance(row.get('shard_index'), int)
                or not 0 <= row['shard_index'] < int(num_shards) for row in records)):
        raise RuntimeError(f'Malformed shard cohort membership: {path}')
    names = [row['source_file'] for row in records]
    if names != sorted(set(names)):
        raise RuntimeError(f'Shard cohort filenames must be unique and sorted: {path}')
    return payload


def bind_shard_cohort(shards_root, files, *, inference_csv_dir, num_shards, shard_by):
    candidate = build_shard_cohort(files, inference_csv_dir=inference_csv_dir,
                                   num_shards=num_shards, shard_by=shard_by)
    path = cohort_path(shards_root, num_shards)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One brief native lock serializes simultaneous initializers. No whole-cohort
    # source-content hashing or frame-table loading occurs inside this lock.
    with FileLock(path.with_suffix('.lock'), timeout_sec=30):
        if path.exists():
            saved = read_shard_cohort(shards_root, num_shards)
            if saved != candidate:
                raise RuntimeError('Selected shard cohort changed (selection, discovery, identities or strategy); use the same frozen inputs or fresh outputs.')
        else:
            atomic_write_json(candidate, path)
    return candidate


def verify_shard_membership(cohort, shard_index, *, markers, sources):
    """Require every marker and exact source inventory to cover its frozen partition."""
    expected = {r['source_file']: r for r in cohort['files'] if r['shard_index'] == int(shard_index)}
    for marker in markers:
        if (not isinstance(marker, dict) or marker.get('cohort_sha256') != cohort['cohort_sha256']
                or marker.get('num_shards') != cohort['num_shards']
                or marker.get('shard_index') != int(shard_index)
                or marker.get('shard_by') != cohort['shard_by']):
            raise RuntimeError('Shard marker lacks or disagrees with complete cohort proof; regenerate the shard.')
        if 'selected_files' in marker and sorted(marker['selected_files']) != sorted(expected):
            raise RuntimeError('Shard selected-file manifest differs from its frozen cohort partition.')
    if set(sources) != set(expected):
        raise RuntimeError('Shard exact source inventory differs from its complete cohort partition.')
    for name, source in sources.items():
        record = expected[name]
        if (source.get('original_name') != record['original_name']
                or collapse_individual_id(parse_group_and_individual(record['original_name'])[1]) != record['row_identity']):
            raise RuntimeError(f'Shard source identity differs from frozen cohort membership: {name}')
    return sorted(expected)
