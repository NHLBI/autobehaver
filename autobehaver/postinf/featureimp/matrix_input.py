"""Master-matrix numeric integrity and optional native MMGen receipt checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from autobehaver.atomic_io import sha256_file


def read_feature_matrix(path):
    """Preserve identifier spelling while retaining normal numeric NA parsing."""
    # Converters run before pandas NA/type inference for identifier columns.
    # Numeric features still use pandas' standard missing-measurement parsing.
    identifiers = ("individual", "group", "sex", "source_files")
    return pd.read_csv(path, converters={column: str for column in identifiers})


def verify_upstream_receipt(matrix_path):
    """Verify a supplied native completion record, or identify standalone input.

    Standalone CSV inputs may omit native generation records. A native master
    or explicitly named artifact must match its record and companions. Other
    CSVs may live beside that master,
    including a separately certified FeatureImp normalization output.
    """
    matrix = Path(matrix_path).resolve()
    receipt_path = matrix.parent / 'MMGEN_OUTPUT_RECEIPT.json'
    if not receipt_path.exists():
        return {'input_kind':'standalone_matrix', 'matrix_sha256':sha256_file(matrix)}
    try:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        artifacts = receipt.get('artifacts')
        if not isinstance(artifacts, dict):
            raise ValueError('Receipt artifacts are not an object')
        if matrix.name not in artifacts and matrix.name != 'MASTER_FEATURE_MATRIX.csv':
            return {'input_kind':'standalone_matrix', 'matrix_sha256':sha256_file(matrix)}
        if matrix.name not in artifacts:
            raise ValueError('Receipt does not certify the supplied matrix')
        for name, expected in artifacts.items():
            target = (matrix.parent / name).resolve()
            if matrix.parent not in target.parents or not target.is_file() or sha256_file(target) != expected:
                raise ValueError(f'Invalid or missing Mmgen output: {name}')
    except (ValueError, TypeError, OSError, AttributeError) as exc:
        raise ValueError(f'Mmgen completion record failed verification: {receipt_path}: {exc}') from exc
    return {'input_kind':'verified_mmgen_matrix','matrix_sha256':sha256_file(matrix),
            'mmgen_receipt_sha256':sha256_file(receipt_path),
            'calculation_sha256':receipt.get('calculation_sha256'),
            'sources_sha256':receipt.get('sources_sha256')}


def validate_numeric_columns(frame, columns):
    """Reject malformed cells with their feature and animal row; preserve NaNs."""
    for column in columns:
        values = frame[column]
        already_numeric = pd.api.types.is_numeric_dtype(values.dtype)
        converted = values if already_numeric else pd.to_numeric(values, errors='coerce')
        # Preserve explicitly nonmeasurement annotations in standalone/legacy
        # matrices. Any declared scientific name or partly numeric column must
        # validate numerically, rather than disappear due to one malformed cell.
        prefixes = ('BoutSummary_', 'TimeBudget_', 'StdMeasure_', 'Postural_',
                    'KinematicBB_', 'Kinematic_', 'Transition_',
                    'SleepFeat_', 'Spatial_', 'Global_')
        legacy_annotation = str(column) == 'Postural_group'
        recognized = str(column).startswith(prefixes) and not legacy_annotation
        if (legacy_annotation or not recognized) and not pd.api.types.is_numeric_dtype(values.dtype) and not converted.notna().any():
            continue
        malformed = values.notna() & converted.isna()
        if malformed.any():
            examples = [(str(frame.loc[i,'individual']), str(values.loc[i])) for i in frame.index[malformed][:5]]
            raise ValueError(f'Numeric feature {column!r} contains text/non-numeric measurements; animal/value examples: {examples}')
        if pd.api.types.is_complex_dtype(converted.dtype):
            raise ValueError(f'Numeric feature {column!r} contains complex measurements')
        infinite = np.isinf(converted.to_numpy(dtype=np.float64))
        if infinite.any():
            animals = frame.loc[frame.index[infinite], 'individual'].astype(str).tolist()[:5]
            raise ValueError(f'Numeric feature {column!r} contains infinite measurements; animals: {animals}')
        if not already_numeric:
            frame[column] = converted
