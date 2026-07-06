"""Naming and ID parsing helpers for mmgen.

This module contains small, shared helpers that are used across the mmgen
pipeline (orchestration, worker processing, and optional movie generation).

These functions are shared across mmgen so that filename parsing and identity
collapse logic live in one place. They provide a
single source of truth.
"""

from __future__ import annotations

import re


def parse_group_and_individual(filename: str):
    """Parse a filename to extract group label, unique individual ID, and sex.

    Notes
    -----
    This logic defines the current filename parsing contract used across mmgen.
    implementation to avoid changing downstream outputs.
    """

    # Remove file extensions and common tags
    base = (
        filename.replace("_normalized_inference.csv", "")
        .replace("_positional_calcs.csv", "")
        .replace(".csv", "")
    )

    # --- Sex and Group Extraction ---
    sex = "Unknown"
    if "_FEMALE_" in base:
        sex = "FEMALE"
        group = base.split("_FEMALE_")[0]
    elif "_MALE_" in base:
        sex = "MALE"
        group = base.split("_MALE_")[0]
    else:
        # Fallback for simpler names
        group = base.split("_")[0]

    # --- Individual ID Extraction ---
    # Extract components separately: trial is part of identity, movie is a recording number
    trial_match = re.search(r"trial(\d+)", base)
    movie_match = re.search(r"movie(\d+)", base)
    rpi_match = re.search(r"(rpi-\d+)", base)
    chamber_match = re.search(r"(chamber\d+)", base)

    trial = f"trial{trial_match.group(1)}" if trial_match else ""
    movie = f"movie{movie_match.group(1)}" if movie_match else ""
    rpi = rpi_match.group(1) if rpi_match else ""
    chamber = chamber_match.group(1) if chamber_match else ""

    # A robust individual ID combines the unique parts.
    # Individual identity = rpi + chamber + trial (trial is part of identity)
    # Movie is included in the full individual_id for per-file tracking, but gets collapsed later
    individual = f"{group}_{sex}_{rpi}_{trial}_{movie}_{chamber}"

    # Clean up any extra underscores from missing parts
    individual = re.sub(r"_+", "_", individual).strip("_")

    return group, individual, sex


def collapse_individual_id(individual_id: str) -> str:
    """Collapse movie numbers from individual IDs.

    Movies (movie0, movie1, ...) are multiple recordings per individual and are
    removed for per-animal aggregation. Trial is kept as part of identity.
    """

    return re.sub(r"_movie\d+", "", str(individual_id))

