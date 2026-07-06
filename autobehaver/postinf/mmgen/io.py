"""I/O and file-matching helpers for mmgen.

This module holds small helpers that locate and match related input files.
The behavior is intentionally simple and deterministic because it underpins file pairing across the pipeline.
"""

from __future__ import annotations

import os


def find_matching_poscalc_file(inf_csv_filename: str, poscalc_dir: str, inf_tag: str):
    """Find the positional_calcs CSV corresponding to an inference CSV filename."""

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

    print(f"Warning: No matching poscalc file for {inf_csv_filename}")
    return None

