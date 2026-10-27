"""Build explicit HPC batches from filenames and existing review metadata."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import re

from autobehaver.portable_paths import original_name, logical_recording_stem
from autobehaver.recording_identity import derive_pair_key
from autobehaver.preinf.autocropper.recording_db import load_recording_database_index
from autobehaver.preinf.autocropper.selection import filter_included_paths
from autobehaver.preinf.moviechecker.exclusions import ExclusionsStore
from autobehaver.postinf.mmgen.naming import collapse_individual_id, parse_group_and_individual

FAST_BATCH_SIZES = {"crop": 2, "dlc": 20, "features": 10, "posepred": 5, "mmgen": 80}
STAGES = ("crop", "dlc", "features", "posepred", "mmgen")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def inventory_dlc_outputs(root):
    """Check transferred coordinate headers without loading full recordings.

    Match KeyPre's flat/legacy DLC subfolder discovery and saved exclusions.
    Reject non-coordinate CSVs instead of silently shrinking the requested cohort.
    """
    directory = Path(root) / '3_dlc'
    if (directory / 'DLC').is_dir():
        directory = directory / 'DLC'
    from .preinf.autocropper.selection import read_selection
    paths = sorted(directory.glob('*.csv'))
    # No selection means every path is retained; avoid redundant map lookups.
    if read_selection(directory):
        paths = filter_included_paths(paths)
    if not paths:
        raise ValueError(f'No included DLC coordinate CSVs found in {directory}')
    from .preparation_io import io_results
    def inspect_header(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = [next(reader, []) for _ in range(3)]
        if [row[0].strip().lower() if row else '' for row in header] != ['scorer', 'bodyparts', 'coords']:
            raise ValueError(f'{path.name} is not a raw DLC coordinate CSV. KeyPre needs the '
                             'three-row scorer/bodyparts/coords header with x, y and likelihood; '
                             'PosePred inference CSVs cannot be used. Transfer the original DLC outputs into 3_dlc.')
        width = len(header[0])
        columns = list(zip(header[1][1:], header[2][1:]))
        parts = {}
        for part, coordinate in columns:
            parts.setdefault(part, set()).add(coordinate)
        if (width < 4 or any(len(row) != width for row in header)
                or len(columns) != len(set(columns)) or not parts
                or any(not part.strip() or coords != {'x', 'y', 'likelihood'} for part, coords in parts.items())):
            raise ValueError(f'Incomplete or duplicate DLC coordinate columns: {path.name}')
        return logical_recording_stem(original_name(path))
    chambers = list(io_results(inspect_header, paths, label='Checking DLC files', workers=8))
    if len({name.casefold() for name in chambers}) != len(chambers):
        raise ValueError('Multiple DLC files identify the same recording; retain one selected DLC result per recording.')
    return chambers


def inventory_recordings(root, start_at, *, moviechecker_mode="reviewed", only_original_movies=None):
    """Read directory entries and CSV metadata only; never decode or copy movies."""
    root = Path(root)
    if moviechecker_mode not in ("reviewed", "recording-only"):
        raise ValueError(f"Unknown MovieChecker mode: {moviechecker_mode}")
    if start_at not in ('crop', 'dlc', 'features'):
        raise ValueError('Start stage must be crop, dlc or features')
    if start_at == 'features':
        if only_original_movies:
            raise ValueError('Original-movie selection requires --start-at crop')
        return [], inventory_dlc_outputs(root), {}, 0
    if start_at == "dlc":
        if only_original_movies:
            raise ValueError("Original-movie selection requires --start-at crop")
        movies = sorted((root / "2_cropped_movies").glob("*.mp4"))
        movies = filter_included_paths(movies)
        chambers = [Path(original_name(movie)).stem for movie in movies]
        if not chambers:
            raise ValueError(f"No included cropped movies found in {root / '2_cropped_movies'}")
        return [], chambers, {}, 0

    movie_dir = root / "1_full_movies"
    movies = sorted(path for path in movie_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if only_original_movies:
        selected = list(only_original_movies)
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate original movie in selection")
        available = {movie.name for movie in movies}
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(f"Selected original movie does not exist: {missing}")
        movies = [movie for movie in movies if movie.name in set(selected)]
    exclusions_path = movie_dir / "exclusions.csv"
    if moviechecker_mode == "reviewed" and not exclusions_path.is_file():
        raise ValueError(f"Complete MovieChecker review first: {exclusions_path}")
    exclusions = None
    if moviechecker_mode == "reviewed":
        exclusions = ExclusionsStore(str(exclusions_path))
        exclusions.load()
    metadata_dir = root / "rec_data" / "trials"
    if not metadata_dir.is_dir():
        metadata_dir = movie_dir
    metadata = load_recording_database_index(metadata_dir)
    usable, chambers, by_movie = [], [], {}
    for movie in movies:
        logical = Path(original_name(movie))
        review = exclusions.get(derive_pair_key(str(movie))) if exclusions is not None else None
        if exclusions is not None and review is None:
            raise ValueError(f"MovieChecker exclusions do not cover {logical.name}")
        # Fully excluded cameras need no recording metadata or compute allocation.
        if review is not None and all(review.exclude_ch):
            continue
        occupancy = metadata.get_movie_exclusion_flags(logical)
        exp, subexp, trial, rpi = metadata.resolve_movie_identity(logical)
        selected = []
        for chamber in range(1, 7):
            if (review is not None and review.exclude_ch[chamber - 1]) or occupancy[chamber - 1]:
                continue
            prefix = metadata.get_label_prefix(experiment=exp, subexperiment=subexp,
                trial_number=trial, rpi_number=rpi, chamber_number=chamber)
            prefix = re.sub(r"_+", "_", prefix.strip()).rstrip("_")
            stem = re.sub(r"_+", "_", logical.stem).strip("_")
            selected.append(f"{prefix}_{stem}_chamber{chamber}")
        if selected:
            usable.append(movie.name)
            by_movie[movie.name] = selected
            chambers.extend(selected)
    if not chambers:
        raise ValueError("No usable chamber recordings remain after chamber exclusions")
    return usable, chambers, by_movie, len(movies) - len(usable)


def build_plan(*, speed, start_at, local_project=None, moviechecker_mode="reviewed", only_original_movies=None, batch_sizes=None, batching=None, new_only=False):
    """Return ordered jobs and their file lists; all outputs retain one dataset root."""
    if start_at not in ('crop', 'dlc', 'features'):
        raise ValueError('Start stage must be crop, dlc or features')
    if speed not in ('fast', 'slow'):
        raise ValueError('Speed must be fast or slow')
    if moviechecker_mode not in ("reviewed", "recording-only"):
        raise ValueError(f"Unknown MovieChecker mode: {moviechecker_mode}")
    if only_original_movies and speed != "fast":
        raise ValueError("Original-movie selection requires Fast preparation")
    if moviechecker_mode == "recording-only" and speed != "fast" and start_at != 'features':
        raise ValueError("Recording-only mode requires Fast preparation to verify metadata coverage")
    if batch_sizes and speed != "fast":
        raise ValueError("Batch-size overrides require Fast preparation")
    sizes = dict(FAST_BATCH_SIZES)
    limits = {}
    if speed == 'fast' and batching is None and not batch_sizes:
        batching = dict(full_movies=2, chambers=20, cpu_jobs=50, gpu_jobs=50)
    if batching is not None:
        if speed != 'fast':
            raise ValueError('Shared batching settings require Fast preparation')
        if batch_sizes:
            raise ValueError('Choose shared batching or legacy stage batch sizes, not both')
        if set(batching) != {'full_movies', 'chambers', 'cpu_jobs', 'gpu_jobs'}:
            raise ValueError('Shared batching needs full_movies, chambers, cpu_jobs and gpu_jobs')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in batching.values()):
            raise ValueError('Batch sizes and job caps must be positive integers')
        sizes = {stage: batching['full_movies' if stage == 'crop' else 'chambers'] for stage in STAGES}
        sizes['mmgen'] = max(80, sizes['mmgen'])
        limits = {stage: batching['gpu_jobs' if stage in ('dlc', 'posepred') else 'cpu_jobs'] for stage in STAGES}
    if new_only and speed != 'fast':
        raise ValueError('New-recordings mode requires Fast preparation to select exact inputs')
    if batch_sizes:
        unknown = set(batch_sizes) - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown Fast batch-size stages: {sorted(unknown)}")
        for stage, size in batch_sizes.items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"Fast {stage} batch size must be a positive integer")
            sizes[stage] = size
    if speed == "slow":
        jobs, previous = [], []
        lists = {}
        if start_at == 'features':
            if not local_project:
                raise ValueError('Starting at KeyPre needs the local/mounted dataset to check DLC inputs')
            lists['all_chambers.txt'] = inventory_dlc_outputs(local_project)
        for stage in STAGES[STAGES.index(start_at):]:
            jobs.append(dict(key=stage, stage=stage, depends_on=previous,
                             file_list='all_chambers.txt' if lists else '-',
                             shard_index=0, num_shards=1))
            previous = [stage]
        jobs.append(dict(key="collect", stage="collect", depends_on=[j["key"] for j in jobs],
                         file_list="-", shard_index=0, num_shards=1))
        plan = {"speed": speed, "target_minutes_per_job": None, "jobs": jobs}
        if lists:
            plan.update(usable_original_movies=0, chamber_movies=len(lists['all_chambers.txt']))
        return plan, lists
    if not local_project or not Path(local_project).is_dir():
        raise ValueError("Fast preparation needs --local-project pointing to the mounted/local dataset root")
    movies, chambers, by_movie, excluded = inventory_recordings(
        local_project, start_at, moviechecker_mode=moviechecker_mode,
        only_original_movies=only_original_movies)
    incremental = None
    if new_only:
        from .job_incremental import new_recordings
        movies, chambers, by_movie, incremental = new_recordings(local_project, movies, chambers, by_movie)
    if len(set(chambers)) != len(chambers):
        raise ValueError("Recording metadata produces duplicate chamber identities")
    jobs, lists, providers = [], {}, {}

    def add_batches(stage, items, size, previous):
        owners = {}
        if limits:
            count = min(limits[stage], max(1, len(items) // size))
            width, remainder = divmod(len(items), count)
            subsets, offset = [], 0
            for index in range(count):
                length = width + (index < remainder)
                subsets.append(items[offset:offset + length]); offset += length
        else:
            subsets = [items[offset:offset + size] for offset in range(0, len(items), size)]
        for index, subset in enumerate(subsets):
            key = f"{stage}_{index:03d}"
            list_name = f"{key}.txt"
            lists[list_name] = subset
            dependencies = sorted({owner for item in subset for owner in previous.get(item, [])})
            jobs.append(dict(key=key, stage=stage, depends_on=dependencies,
                             file_list=list_name, shard_index=0, num_shards=1))
            for item in subset:
                owners[item] = [key]
        return owners

    if start_at == "crop":
        movie_owners = add_batches("crop", movies, sizes["crop"], {})
        providers = {chamber: movie_owners[movie] for movie, entries in by_movie.items() for chamber in entries}
    for stage in ("dlc", "features", "posepred"):
        if STAGES.index(stage) < STAGES.index(start_at):
            continue
        providers = add_batches(stage, chambers, sizes[stage], providers)
    # Native row-aware sharding keeps movie0/movie1 of each animal together.
    identities = {collapse_individual_id(parse_group_and_individual(name +
                   "_positional_calcs_normalized_inference.csv")[1]) for name in chambers}
    count = min(len(identities), math.ceil(len(chambers) / sizes["mmgen"]))
    if limits:
        count = min(len(identities), limits['mmgen'], max(1, len(chambers) // sizes['mmgen']))
    if new_only:
        count = 1  # Native append owns one canonical update; no separate tiny planning job.
    lists["all_chambers.txt"] = chambers
    dependencies = sorted({owner for owners in providers.values() for owner in owners})
    for index in range(count):
        jobs.append(dict(key=f"mmgen_{index:03d}", stage="mmgen", depends_on=dependencies,
            file_list="all_chambers.txt", shard_index=index, num_shards=count))
    if count > 1:
        jobs.append(dict(key="mmgen_finish", stage="mmgen_finish",
            depends_on=[j["key"] for j in jobs if j["stage"] == "mmgen"],
            file_list="all_chambers.txt", shard_index=0, num_shards=count))
    jobs.append(dict(key="collect", stage="collect", depends_on=[j["key"] for j in jobs],
                     file_list="-", shard_index=0, num_shards=1))
    return {"speed": speed, "target_minutes_per_job": None if batching else [10, 20],
            "batching": batching, "job_limits": limits, "incremental": incremental,
            "moviechecker_mode": moviechecker_mode,
            "selected_original_movies": list(only_original_movies or []),
            "batch_sizes": sizes, "usable_original_movies": len(movies),
            "excluded_original_movies": excluded, "chamber_movies": len(chambers),
            "animal_rows": len(identities), "jobs": jobs}, lists
