"""Resolve MMGen recording metadata from the supplied animal mapping or filename
parsing.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from .naming import parse_group_and_individual


def resolve_group_individual_sex(inf_csv_path: str, config: Optional[dict]) -> Tuple[str, str, str]:
    """Resolve (group, individual_id, sex) for an inference CSV path."""

    filename = os.path.basename(inf_csv_path)

    if config and "file_to_animal" in config and filename in config["file_to_animal"]:
        animal_id = config["file_to_animal"][filename]
        animal_info = config["animal_mapping"][animal_id]

        group = animal_info["group"]
        sex = animal_info["sex"]

        individual_id = None
        for rec in animal_info["recordings"]:
            if rec["filename"] == filename:
                individual_id = rec["individual_id"]
                break

        if not individual_id:
            if animal_info.get("individual_ids"):
                individual_id = animal_info["individual_ids"][0]
            else:
                _, individual_id, _ = parse_group_and_individual(inf_csv_path)

        return group, individual_id, sex

    group, individual_id, sex = parse_group_and_individual(inf_csv_path)
    return group, individual_id, sex
