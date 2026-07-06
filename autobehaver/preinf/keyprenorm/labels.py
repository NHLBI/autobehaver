"""Label path helpers for keyprenorm."""

from __future__ import annotations

import os
from typing import Optional, Tuple


def label_paths_for_csv(
    csv_base_name: str,
    labels_input_dir: str,
    labels_output_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve label input/output paths for a given positional_calcs CSV base name.

    Returns:
        (label_csv_path, label_h5_path) if label CSV exists, else (None, None).
    """
    base_filename = csv_base_name.replace("_positional_calcs.csv", "")
    label_file_name = f"{base_filename}_labels.csv"
    label_csv_path = os.path.join(labels_input_dir, label_file_name)

    if not os.path.exists(label_csv_path):
        return None, None

    label_h5_path = os.path.join(labels_output_dir, label_file_name.replace(".csv", ".h5"))
    return label_csv_path, label_h5_path
