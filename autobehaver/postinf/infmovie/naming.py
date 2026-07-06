"""Naming helpers for infmovie.

The inference-movie overlay workflow needs to match inference CSVs to the
corresponding source movie files. This module contains filename-key extraction
helpers used for that matching.
"""

from __future__ import annotations

import os
import re


def extract_inf_csv_filename_part(inf_csv_file_path: str) -> str:
    """Extract an identifying substring from an inference CSV filename.

    This helper extracts the source movie stem encoded in inference CSV names.
    """

    # Extract the base name of the file (no directory).
    base_name = os.path.basename(inf_csv_file_path)
    # Strip the .csv extension if present to simplify downstream operations.
    if base_name.lower().endswith(".csv"):
        base_name = base_name[:-4]

    # Remove everything before "MALE_" or "FEMALE_" and including it
    if "MALE_" in base_name:
        base_name = base_name.split("MALE_")[1]
    elif "FEMALE_" in base_name:
        base_name = base_name.split("FEMALE_")[1]

    # Keep "_chamber<digits>" and remove everything after it.
    # Chamber numbers can contain one or more digits.
    m = re.search(r"(.*?_chamber\d+)", base_name)
    if m:
        base_name = m.group(1)

    # Remove "_positional_calcs" and everything after.
    base_name = base_name.split("_positional_calcs")[0]

    # Replace double underscores with a single underscore
    base_name = base_name.replace("__", "_")

    return base_name
