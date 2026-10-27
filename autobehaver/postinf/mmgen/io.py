"""Resolve the positional-feature CSV paired with an MMGen inference file."""

from __future__ import annotations

import os
from pathlib import Path
from autobehaver.portable_paths import original_name, load_name_maps


CSV_TEXT_COLUMNS = (
    'individual', 'individual_id', 'animal_id', 'row_identity', 'group', 'group_label',
    'sex', 'source_file', 'source_path', 'poscalc_file', 'poscalc_path', 'original_name',
    'source_file_sha256', 'poscalc_file_sha256', 'input_sha256',
)


def read_metadata_csv(path, **kwargs):
    """Read MMGen CSVs without interpreting identifier text as missing/numeric data.

    Converters preserve literal metadata such as ``NA`` and ``001`` while pandas'
    normal missing-value parsing remains active for scientific value columns.
    """
    import pandas as pd
    converters = {column: str for column in CSV_TEXT_COLUMNS}
    converters.update(kwargs.pop('converters', {}))
    return pd.read_csv(path, converters=converters, **kwargs)


def find_matching_poscalc_file(inf_csv_filename: str, poscalc_dir: str, inf_tag: str):
    """Find the positional_calcs CSV corresponding to an inference CSV filename."""

    inf_csv_filename = original_name(inf_csv_filename)
    load_name_maps(poscalc_dir)
    # Remove the inf_tag from the filename
    if inf_tag in inf_csv_filename:
        poscalc_filename = inf_csv_filename.replace(inf_tag, "")
    else:
        poscalc_filename = inf_csv_filename  # fallback: no tag found

    # Full path to the poscalc file
    poscalc_file_path = os.path.join(poscalc_dir, poscalc_filename)

    # Check if the file exists
    if os.path.exists(poscalc_file_path):
        return poscalc_file_path

    matches=[str(p) for p in Path(poscalc_dir).glob('*.csv') if original_name(p)==poscalc_filename]
    if len(matches)>1:
        raise ValueError(f'Ambiguous positional-calcs partners for {inf_csv_filename}')
    if matches:
        return matches[0]
    print(f"Warning: No matching poscalc file for {inf_csv_filename}")
    return None

