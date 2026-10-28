"""Parse MMGen group, sex, recording, and animal identities from filenames.

Trial remains part of animal identity; movie numbers distinguish recordings before
aggregation.
"""

from __future__ import annotations

import re
from autobehaver.portable_paths import original_name


def parse_group_and_individual(filename: str):
    """Parse group, recording identity, and sex from the original logical filename."""

    filename = original_name(filename)
    # Remove file extensions and common tags
    base = (
        filename.replace("_normalized_inference.csv", "")
        .replace("_positional_calcs.csv", "")
        .replace(".csv", "")
    )

    # Sex and Group Extraction
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

    # Individual ID Extraction
    # Extract components separately: trial is part of identity, movie is a recording number
    trial_match = re.search(r"trial(\d+)", base)
    movie_match = re.search(r"movie(\d+)", base)
    rpi_match = re.search(r"(rpi-\d+)", base)
    chamber_match = re.search(r"(chamber\d+)", base)

    trial = f"trial{trial_match.group(1)}" if trial_match else ""
    movie = f"movie{movie_match.group(1)}" if movie_match else ""
    rpi = rpi_match.group(1) if rpi_match else ""
    chamber = chamber_match.group(1) if chamber_match else ""

    # Combine the recording-identity tokens in a stable order.
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

