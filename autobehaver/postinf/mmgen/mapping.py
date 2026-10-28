"""Map filename-derived recording identities to MMGen animals and write the mapping for
inspection.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Tuple

import pandas as pd

from .naming import collapse_individual_id, parse_group_and_individual


def create_animal_mapping(
    inference_csv_dir: str,
    poscalc_dir: str,
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """Parse all input files and create a centralized animal mapping.

    Individual identity = rpi + chamber + trial (trial is part of identity).
    Movies (movie0, movie1, etc.) are multiple recordings per individual and get
    collapsed.

    The mapping is derived from the current inference filenames and written
    out for inspection alongside the matrix outputs.
    """

    animal_mapping = {}
    file_to_animal = {}  # Reverse lookup: file path -> animal_id

    def extract_recording_number(individual_id):
        """Extract the movie number used to distinguish recordings of one animal."""

        match = re.search(r"movie(\d+)", individual_id)
        if match:
            return {"type": "movie", "number": int(match.group(1))}

        # If no match, something is wrong (movie should always be present)
        print(f"Warning: Could not extract movie number from individual_id: {individual_id}")
        return None

    # Parse all inference files
    inf_files = [
        f
        for f in os.listdir(inference_csv_dir)
        if f.endswith(".csv") and not f.endswith("_bouts.csv")
    ]

    if not inf_files:
        print("Warning: No inference files found for animal mapping.")
        return {}, {}

    print(f"\nCreating animal mapping from {len(inf_files)} inference files...")

    for filename in inf_files:
        try:
            group, individual_id, sex = parse_group_and_individual(os.path.join(inference_csv_dir, filename))
            animal_id = collapse_individual_id(individual_id)

            # Extract trial as part of identity (not a recording)
            trial_match = re.search(r"_trial(\d+)", individual_id)
            trial_number = int(trial_match.group(1)) if trial_match else None

            # Extract movie as recording number
            recording_info = extract_recording_number(individual_id)

            if animal_id not in animal_mapping:
                animal_mapping[animal_id] = {
                    "animal_id": animal_id,
                    "group": group,
                    "sex": sex,
                    "files": [],
                    "individual_ids": [],
                    "recordings": [],  # Only movie recordings
                    "n_recordings": 0,
                    "rpi": None,
                    "chamber": None,
                    "trial": None,  # Store trial as identity component
                    "consistent_group": True,
                    "consistent_sex": True,
                    "warnings": [],
                }

            # Add file info
            full_path = os.path.join(inference_csv_dir, filename)
            animal_mapping[animal_id]["files"].append(full_path)
            animal_mapping[animal_id]["individual_ids"].append(individual_id)
            animal_mapping[animal_id]["n_recordings"] += 1

            # Add movie recording info (only movies, not trials)
            if recording_info:  
                animal_mapping[animal_id]["recordings"].append(
                    {
                        "individual_id": individual_id,
                        "filename": filename,
                        "recording_type": "movie",
                        "recording_number": recording_info["number"],
                    }
                )

            # Extract RPI, chamber, and trial from first file (should be consistent)
            if animal_mapping[animal_id]["rpi"] is None:
                rpi_match = re.search(r"(rpi-\d+)", individual_id)
                chamber_match = re.search(r"(chamber\d+)", individual_id)
                animal_mapping[animal_id]["rpi"] = rpi_match.group(1) if rpi_match else None
                animal_mapping[animal_id]["chamber"] = (
                    chamber_match.group(1) if chamber_match else None
                )
                animal_mapping[animal_id]["trial"] = trial_number

            # Reverse lookup (both filename and full path)
            file_to_animal[filename] = animal_id
            file_to_animal[full_path] = animal_id

        except Exception as e:
            print(f"  ERROR parsing {filename} for mapping: {e}")
            continue

    # Validate consistency
    for animal_id, data in animal_mapping.items():
        groups = set()
        sexes = set()
        for filename in data["files"]:
            try:
                group, _, sex = parse_group_and_individual(filename)
                groups.add(group)
                sexes.add(sex)
            except Exception:
                pass

        data["consistent_group"] = len(groups) <= 1
        data["consistent_sex"] = len(sexes) <= 1

        if not data["consistent_group"]:
            data["warnings"].append(f"Inconsistent groups: {groups}")
        if not data["consistent_sex"]:
            data["warnings"].append(f"Inconsistent sexes: {sexes}")

        # Sort recordings by movie number
        data["recordings"].sort(key=lambda x: x["recording_number"])

    # Save to JSON
    mapping_path = os.path.join(output_dir, "ANIMAL_MAPPING.json")
    try:
        with open(mapping_path, "w") as f:
            json.dump(animal_mapping, f, indent=2, sort_keys=True)
        print(f"Animal mapping saved to {mapping_path}")
    except Exception as e:
        print(f"Warning: Could not save animal mapping JSON: {e}")

    # Save a readable CSV summary
    try:
        summary_rows = []
        for animal_id, data in animal_mapping.items():
            movie_numbers = [str(r["recording_number"]) for r in data["recordings"]]
            filenames = [os.path.basename(f) for f in data["files"]]

            summary_rows.append(
                {
                    "animal_id": animal_id,
                    "group": data["group"],
                    "sex": data["sex"],
                    "rpi": data["rpi"] or "",
                    "chamber": data["chamber"] or "",
                    "trial": data["trial"] if data["trial"] is not None else "",
                    "n_movies": data["n_recordings"],
                    "movie_numbers": ", ".join(sorted(movie_numbers, key=int)),
                    "filenames": ", ".join(sorted(filenames)),
                    "consistent_group": data["consistent_group"],
                    "consistent_sex": data["consistent_sex"],
                    "n_warnings": len(data["warnings"]),
                    "warnings": "; ".join(data["warnings"]) if data["warnings"] else "",
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "ANIMAL_MAPPING_SUMMARY.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Animal mapping summary saved to {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save animal mapping summary CSV: {e}")

    print(f"Created mapping for {len(animal_mapping)} unique animals from {len(inf_files)} files.")

    return animal_mapping, file_to_animal
