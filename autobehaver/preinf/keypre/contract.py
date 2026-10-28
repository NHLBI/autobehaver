"""Share KeyPre schema and completion checks between coordinators and workers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from autobehaver.completion import request_identity, receipt_matches, file_signature
from autobehaver.portable_paths import generated_path, original_name, recording_name
from . import settings
from .dlc_io import dlc_row_count
from .spec import parts


def output_path(input_csv, directory):
    """Return the portable positional-calculations output path for a DeepLabCut CSV."""
    stem = recording_name(Path(original_name(input_csv)).stem)
    return generated_path(directory, stem + '_positional_calcs.csv')


def keypre_request(input_csv, movie_dim=600, raw_only=False):
    """Identify input content, feature settings, and the KeyPre implementation."""
    folder = Path(__file__).parent
    modules = [p for p in folder.glob('*.py') if p.stem not in ('cli','core','logging_utils','manifest','config')]
    values = {name:getattr(settings,name) for name in ('include_position','rolling_windows','include_likelihoods','nan_edge_effects','edge_skip')}
    return request_identity('keypre-v1', [input_csv], {'movie_dim':int(movie_dim),'raw_only':bool(raw_only), **values}, modules)


def inspect_keypre_output(input_csv, output_csv, *, movie_dim=600, raw_only=False, require_receipt=False, expected_rows=None, cached_inspection=None):
    """Check row count, numeric schema, and the optional completion receipt.

    Return (ok, reason, rows, columns); preserve missing values allowed by KeyPre.
    """
    try:
        if not Path(output_csv).is_file():
            return False, 'output file is missing', 0, 0
        if (cached_inspection is not None
                and file_signature(input_csv) == cached_inspection[0]
                and file_signature(output_csv) == cached_inspection[1]):
            if require_receipt and not receipt_matches(output_csv, keypre_request(input_csv, movie_dim, raw_only)):
                return False, 'missing or stale completion receipt', 0, 0
            return cached_inspection[2]
        if expected_rows is None:
            expected_rows = dlc_row_count(input_csv)
        expected = ([f'{part}_{coord}_raw' for part in parts for coord in ('x','y','likelihood')]
                    if raw_only else json.loads((Path(__file__).parent / 'feature_schema.json').read_text()))
        rows, columns = 0, 0
        with pd.read_csv(output_csv, chunksize=4096) as reader:
            for frame in reader:
                columns = len(frame.columns)
                if set(frame.columns) != set(expected) or columns != len(expected):
                    return False, 'feature schema differs from the requested Keypre mode', rows, columns
                rows += len(frame)
                if np.isinf(frame.to_numpy(dtype=float)).any():
                    return False, 'infinite feature values', rows, columns
        if rows != expected_rows:
            return False, 'output row count differs from source frame count', rows, columns
        if require_receipt and not receipt_matches(output_csv, keypre_request(input_csv, movie_dim, raw_only)):
            return False, 'missing or stale completion receipt', rows, columns
        return True, 'ok', rows, columns
    except Exception as exc:
        return False, f'Keypre integrity check failed: {exc}', 0, 0
