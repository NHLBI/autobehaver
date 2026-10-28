"""Resolve source and output label paths for KeyPreNorm."""

from __future__ import annotations

import os
from autobehaver.portable_paths import generated_path
from typing import Optional, Tuple


def label_paths_for_csv(
    csv_base_name: str,
    labels_input_dir: str,
    labels_output_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the paired label CSV and HDF5 paths for a positional-calculations file.

    Keep the expected HDF5 path when the source CSV is absent so an existing
    paired annotation cannot silently escape integrity checks.
    """
    base_filename = csv_base_name.replace("_positional_calcs.csv", "")
    label_file_name = f"{base_filename}_labels.csv"
    label_csv_path = os.path.join(labels_input_dir, label_file_name)

    label_h5_path = str(generated_path(labels_output_dir, label_file_name.replace(".csv", ".h5")))
    return label_csv_path if os.path.exists(label_csv_path) else None, label_h5_path
