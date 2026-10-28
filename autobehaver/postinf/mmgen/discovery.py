"""Discover MMGen inference files and report filename-derived sex and group counts."""

from __future__ import annotations

import os
from autobehaver.portable_paths import load_name_maps

import pandas as pd

from .naming import parse_group_and_individual


def list_inference_csv_files(inference_csv_dir: str) -> list[str]:
    """List inference CSV files in a folder.

    Notes
    -----
    - Includes *.csv and excludes *_bouts.csv.
    - Does NOT sort; ordering matches os.listdir().
    """

    load_name_maps(inference_csv_dir)
    return [
        f
        for f in os.listdir(inference_csv_dir)
        if f.endswith(".csv") and (not f.endswith("_bouts.csv"))
    ]


def print_inference_file_diagnostic(inf_files: list[str], inference_csv_dir: str = "") -> None:
    """Report input counts and filename-derived sex/group combinations, including parse
    failures.
    """

    # DIAGNOSTIC: Print all input files and their parsed labels
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Input Files During Feature Generation")
    print("=" * 80)
    print(f"Found {len(inf_files)} inference CSV files")
    print("\nParsing filenames to extract group, individual, and sex:")
    print("-" * 80)

    file_info: list[dict[str, str]] = []
    for filename in sorted(inf_files):
        try:
            group, individual_id, sex = parse_group_and_individual(os.path.join(inference_csv_dir, filename))
            file_info.append(
                {
                    "filename": filename,
                    "group": group,
                    "individual": individual_id,
                    "sex": sex,
                }
            )
        except Exception as e:
            print(f"  ERROR parsing {filename}: {e}")
            file_info.append(
                {
                    "filename": filename,
                    "group": "PARSE_ERROR",
                    "individual": "PARSE_ERROR",
                    "sex": "PARSE_ERROR",
                }
            )

    # Group by sex-genotype combination
    file_df = pd.DataFrame(file_info)
    file_df["sex_genotype"] = file_df["sex"].astype(str) + "_" + file_df["group"].astype(str)

    print("\nSample counts by Sex-Genotype combination (from filenames):")
    print("-" * 80)
    counts = file_df["sex_genotype"].value_counts().sort_index()
    for combo, count in counts.items():
        sex_val, group_val = combo.split("_", 1)
        print(f"  {sex_val:10s} | {group_val:40s} | {count:3d} files")

    print("\nDetailed file listing by Sex-Genotype combination:")
    print("-" * 80)
    for combo in sorted(file_df["sex_genotype"].unique()):
        sex_val, group_val = combo.split("_", 1)
        subset = file_df[file_df["sex_genotype"] == combo]
        print(f"\n  {sex_val} | {group_val} ({len(subset)} files):")
        for _, row in subset.iterrows():
            print(
                f"    - {row['filename']:60s} -> Individual: {row['individual']}"
            )

    # Check for parse errors
    parse_errors = file_df[file_df["group"] == "PARSE_ERROR"]
    if len(parse_errors) > 0:
        print("\n*** WARNING: Files with parsing errors ***")
        for _, row in parse_errors.iterrows():
            print(f"  - {row['filename']}")

    # Check for unknown sex
    unknown_sex = file_df[file_df["sex"].astype(str).str.upper() == "UNKNOWN"]
    if len(unknown_sex) > 0:
        print("\n*** WARNING: Files with unknown sex ***")
        for _, row in unknown_sex.iterrows():
            print(f"  - {row['filename']} -> Sex: {row['sex']}")

    print("\n" + "=" * 80 + "\n")
