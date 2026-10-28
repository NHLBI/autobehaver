"""Declare the ordered MMGen scalar feature contract independently of measured animals.

Missing or disabled measurements remain NaN; observed zeros remain zero.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from autobehaver.atomic_io import atomic_write_json

ID_COLUMNS = ('individual', 'group', 'sex')
OPTIONAL_METADATA_COLUMNS = ('source_files', 'total_frames_analyzed')
SCHEMA_METADATA_FILENAME = 'MASTER_FEATURE_MATRIX_SCHEMA.json'
FAMILIES = ('BoutSummary_', 'TimeBudget_', 'StdMeasure_', 'Postural_',
            'KinematicBB_', 'Kinematic_', 'Transition_',
            'SleepFeat_', 'Spatial_', 'Global_')


def family_sort_key(name):
    return (next((i for i, prefix in enumerate(FAMILIES) if str(name).startswith(prefix)), len(FAMILIES)), str(name))


@lru_cache(maxsize=1)
def declared_catalog():
    """Load and validate the versioned ordered feature catalog."""
    data = json.loads(Path(__file__).with_name('feature_schema_catalog.json').read_text(encoding='utf-8'))
    names = data['features']
    if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
        raise RuntimeError('Malformed declared Mmgen feature catalog')
    data['features'] = tuple(sorted(names, key=family_sort_key))
    return data


def feature_columns():
    """Return the declared scalar feature columns in release order."""
    return declared_catalog()['features']


def complete_master_schema(frame):
    """Align valid historical/current columns to the same complete contract.

    Missing recognized features gain NaNs. Unknown columns are rejected so an
    accidental new/misspelled feature cannot silently become release schema.
    Populated values are never recalculated or filled with zero here.
    """
    if frame.columns.duplicated().any():
        raise ValueError('Master matrix contains duplicate column names')
    permitted = set(ID_COLUMNS) | set(OPTIONAL_METADATA_COLUMNS) | set(feature_columns())
    unexpected = sorted(set(frame.columns) - permitted)
    if unexpected:
        raise ValueError(f'Master matrix contains undeclared feature columns: {unexpected[:20]}')
    missing_ids = set(ID_COLUMNS) - set(frame.columns)
    if missing_ids:
        raise ValueError(f'Master matrix is missing identity columns: {sorted(missing_ids)}')
    metadata = [c for c in OPTIONAL_METADATA_COLUMNS if c in frame]
    return frame.reindex(columns=list(ID_COLUMNS) + list(feature_columns()) + metadata).copy()


def schema_metadata():
    """Summarize the feature contract, family counts, and ordered schema hash."""
    catalog = declared_catalog()
    names = feature_columns()
    return dict(schema_version=catalog['schema_version'],
                feature_count=len(names),
                feature_schema_sha256=hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest(),
                family_counts=dict(sorted(Counter(n.split('_')[0] for n in names).items())),
                feature_columns=list(names), provenance=catalog['provenance'],
                missing_value_policy='Unavailable/censored/disabled measurements remain NaN; observed zeros remain zero.')


def write_schema_metadata(output_dir):
    """Write the declared schema metadata beside MMGen outputs."""
    destination = Path(output_dir) / SCHEMA_METADATA_FILENAME
    atomic_write_json(schema_metadata(), destination)
    return destination
