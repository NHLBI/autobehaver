"""Tiny inventories, explicit input routing and fake-scheduler/publication checks."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from autobehaver.batch_inputs import select_batch_inputs
from autobehaver.job_batches import build_plan, inventory_recordings
from autobehaver.job_publication import publish_jobs
from autobehaver.preinf.autocropper.selection import read_selection, write_selection
from autobehaver.portable_paths import generated_path
_spec = importlib.util.spec_from_file_location("submission_test_helpers", Path(__file__).with_name("test_hpc_submission.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
prepare, bash_executable, bash_path = _helpers.prepare, _helpers.bash_executable, _helpers.bash_path


def recording_fixture(root):
    movies = root / "1_full_movies"
    metadata = root / "rec_data/trials"
    movies.mkdir(parents=True); metadata.mkdir(parents=True)
    rows = []
    for rpi in range(1, 5):
        for movie in range(2):
            (movies / f"exp_trial1_rpi-{rpi}_movie{movie}_20261018120000PM.mp4").touch()
        rows.append(dict(pair_key=f"exp_trial1_rpi-{rpi}_movie", reviewed_movie="",
                         **{f"exclude_ch{ch}": int(rpi == 4) for ch in range(1, 7)}))
    with (movies / "exclusions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (metadata / "crop_occupancy_test.csv").open("w", newline="") as stream:
        fields = ["experiment", "subexperiment", "trial_number", "rpi_name", "chamber_number", "label", "sex", "occupied"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for rpi in range(1, 4):
            for ch in range(1, 7):
                writer.writerow(dict(experiment="exp", subexperiment="", trial_number=1,
                    rpi_name=f"rpi-{rpi}", chamber_number=ch, label="G", sex="FEMALE", occupied=1))
    return root


def test_fast_covers_each_recording_once_and_keeps_animals_together(tmp_path):
    root = recording_fixture(tmp_path / "dataset")
    movies, chambers, _, excluded = inventory_recordings(root, "crop")
    plan, lists = build_plan(speed="fast", start_at="crop", local_project=root)
    assert len(movies) == 6 and len(chambers) == 36 and excluded == 2
    counts = {stage: sum(j["stage"] == stage for j in plan["jobs"]) for stage in
              ("crop", "dlc", "features", "posepred", "mmgen", "mmgen_finish", "collect")}
    assert counts == dict(crop=3, dlc=1, features=1, posepred=1, mmgen=1, mmgen_finish=0, collect=1)
    for stage in ("crop", "dlc", "features", "posepred"):
        entries = [item for j in plan["jobs"] if j["stage"] == stage for item in lists[j["file_list"]]]
        assert sorted(entries) == sorted(movies if stage == "crop" else chambers)
        assert len(entries) == len(set(entries))
    submitted = {}
    for job in plan["jobs"]:
        assert set(job["depends_on"]) <= set(submitted)
        if job["stage"] == "dlc":
            requested = lists[job["file_list"]]
            owners = {prior["key"] for prior in submitted.values() if prior["stage"] == "crop"
                      and any(Path(movie).stem in chamber for movie in lists[prior["file_list"]] for chamber in requested)}
            assert set(job["depends_on"]) == owners
        if job["stage"] in ("features", "posepred"):
            previous_stage = "dlc" if job["stage"] == "features" else "features"
            requested = set(lists[job["file_list"]])
            owners = {prior["key"] for prior in submitted.values() if prior["stage"] == previous_stage
                      and requested.intersection(lists[prior["file_list"]])}
            assert set(job["depends_on"]) == owners
        submitted[job["key"]] = job
    assert plan["animal_rows"] == 18
    assert all(j["num_shards"] == 1 for j in plan["jobs"] if j["stage"].startswith("mmgen"))


def test_fast_batch_size_overrides_preserve_coverage_and_dependencies(tmp_path):
    root = recording_fixture(tmp_path / "dataset")
    sizes = dict(crop=4, dlc=4, features=12, posepred=12, mmgen=12)
    plan, lists = build_plan(speed="fast", start_at="crop", local_project=root,
                             batch_sizes=sizes)
    assert plan["batch_sizes"] == sizes
    assert {stage: sum(j["stage"] == stage for j in plan["jobs"]) for stage in sizes} == \
        dict(crop=2, dlc=9, features=3, posepred=3, mmgen=3)
    for stage in ("crop", "dlc", "features", "posepred"):
        entries = [item for job in plan["jobs"] if job["stage"] == stage
                   for item in lists[job["file_list"]]]
        expected = plan["usable_original_movies"] if stage == "crop" else plan["chamber_movies"]
        assert len(entries) == len(set(entries)) == expected
    submitted = set()
    for job in plan["jobs"]:
        assert set(job["depends_on"]) <= submitted
        submitted.add(job["key"])
    assert all(job["num_shards"] == 3 for job in plan["jobs"]
               if job["stage"].startswith("mmgen"))
    with pytest.raises(ValueError, match="positive integer"):
        build_plan(speed="fast", start_at="crop", local_project=root,
                   batch_sizes={"dlc": 0})
    with pytest.raises(ValueError, match="Fast"):
        build_plan(speed="slow", start_at="crop", batch_sizes=sizes)


def test_fast_subset_uses_reviewed_movies_in_the_normal_dataset(tmp_path):
    root = recording_fixture(tmp_path / "dataset")
    selected = [f"exp_trial1_rpi-{rpi}_movie0_20261018120000PM.mp4" for rpi in (1, 2)]
    plan, lists = build_plan(speed="fast", start_at="crop", local_project=root,
                             only_original_movies=selected)
    assert plan["usable_original_movies"] == 2
    assert plan["chamber_movies"] == 12
    assert plan["selected_original_movies"] == selected
    assert sorted(lists["crop_000.txt"]) == sorted(selected)
    assert len([job for job in plan["jobs"] if job["stage"] == "mmgen"]) == 1
    with pytest.raises(ValueError, match="does not exist"):
        build_plan(speed="fast", start_at="crop", local_project=root,
                   only_original_movies=["absent.mp4"])
    with pytest.raises(ValueError, match="Fast"):
        build_plan(speed="slow", start_at="crop", only_original_movies=selected)


def test_recording_only_mode_ignores_moviechecker_and_uses_occupancy(tmp_path):
    root = recording_fixture(tmp_path / "dataset")
    selected = ["exp_trial1_rpi-1_movie0_20261018120000PM.mp4"]
    exclusion = root / "1_full_movies/exclusions.csv"
    rows = list(csv.DictReader(exclusion.open(newline="")))
    rows[0]["exclude_ch1"] = "1"
    with exclusion.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    reviewed, _ = build_plan(speed="fast", start_at="crop", local_project=root,
                             only_original_movies=selected)
    recording, lists = build_plan(speed="fast", start_at="crop", local_project=root,
                                  only_original_movies=selected, moviechecker_mode="recording-only")
    assert reviewed["chamber_movies"] == 5
    assert recording["chamber_movies"] == 6
    assert recording["moviechecker_mode"] == "recording-only"
    assert lists["crop_000.txt"] == selected
    exclusion.unlink()
    without_review, _ = build_plan(speed="fast", start_at="crop", local_project=root,
                                    only_original_movies=selected, moviechecker_mode="recording-only")
    assert without_review["chamber_movies"] == 6
    with pytest.raises(ValueError, match="MovieChecker"):
        build_plan(speed="fast", start_at="crop", local_project=root,
                   only_original_movies=selected)


def test_fast_is_default_and_slow_needs_no_inventory(tmp_path):
    with pytest.raises(ValueError, match="local-project"):
        prepare.prepare_jobs(output_dir=tmp_path / "missing", remote_job_dir="/data/jobs",
            project="/data/project", package="/data/code", python="/data/python", resources="/data/resources")
    assert not (tmp_path / "missing").exists()
    root = recording_fixture(tmp_path / "dataset")
    target = prepare.prepare_jobs(output_dir=tmp_path / "fast", remote_job_dir="/data/jobs",
        project="/data/project", package="/data/code", python="/data/python", resources="/data/resources",
        local_project=root)
    assert json.loads((target / "plan.json").read_text())["speed"] == "fast"
    slow, lists = build_plan(speed="slow", start_at="crop")
    assert len(slow["jobs"]) == 6 and not lists
    assert [j["stage"] for j in slow["jobs"]] == ["crop", "dlc", "features", "posepred", "mmgen", "collect"]
    assert not any(path.name.startswith(("trial_", "combined", "1_", "2_", "3_", "8_")) for path in target.rglob("*"))


def test_batch_selection_handles_suffixes_and_shortened_filenames(tmp_path):
    names = ["G"*180+"_FEMALE_exp_trial1_rpi-1_movie0_20261018120000PM_chamber1",
             "G"*180+"_FEMALE_exp_trial1_rpi-1_movie0_20261018120000PM_chamber10"]
    paths = []
    for name in names:
        output = generated_path(tmp_path, name + "_positional_calcs.csv")
        output.touch(); paths.append(str(output))
    listing = tmp_path / "batch.txt"; listing.write_text(names[0]+"\n")
    assert select_batch_inputs(paths, listing) == paths[:1]
    listing.write_text("absent_recording\n")
    with pytest.raises(ValueError, match="matched 0"):
        select_batch_inputs(paths, listing)
    listing.write_text(names[0]+"\n"+names[0]+"\n")
    with pytest.raises(ValueError, match="duplicate"):
        select_batch_inputs(paths, listing)


def test_concurrent_selection_publication_preserves_every_batch(tmp_path):
    def publish(i):
        write_selection(tmp_path, {f"recording{i}": {"included": True}})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publish, range(12)))
    assert len(read_selection(tmp_path)) == 12


def test_selection_retries_transient_atomic_replace_failures(tmp_path, monkeypatch):
    write_selection(tmp_path, {"existing": {"included": True}})
    replace = os.replace
    attempts = []
    def transient_replace(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError("Temporary Windows sharing lock")
        return replace(source, destination)
    monkeypatch.setattr(os, "replace", transient_replace)
    write_selection(tmp_path, {"new": {"included": True}})
    assert len(attempts) == 3
    assert set(read_selection(tmp_path)) == {"existing", "new"}
    assert not list(tmp_path.glob(".crop-selection-*.tmp"))
    from autobehaver.file_locks import file_lock_active
    assert not file_lock_active(tmp_path / ".crop-selection.lock")


@pytest.mark.parametrize("stage,runtime", [("dlc", "run_dlc"), ("keypre", "run_keypre"),
    ("keyprenorm", "run_keyprenorm"), ("posepred_inference", "run_posepred_inference"),
    ("mmgen", "run_mmgen")])
def test_batch_list_reaches_native_runtime(tmp_path, monkeypatch, stage, runtime):
    import importlib
    prefix = "postinf" if stage == "mmgen" else "preinf"
    cli = importlib.import_module(f"autobehaver.{prefix}.{stage}.cli")
    core = importlib.import_module(f"autobehaver.{prefix}.{stage}.core")
    listing = tmp_path / "batch.txt"; listing.write_text("recording_chamber1\n")
    captured = []
    def run(*args, **kwargs):
        captured.append(kwargs.get("cfg") or args[0])
        return 0
    monkeypatch.setattr(core, runtime, run)
    if hasattr(cli, runtime):
        monkeypatch.setattr(cli, runtime, run)
    try:
        arguments = ["--main_dir", str(tmp_path), "--file_list", str(listing)]
        if stage == "dlc":
            config = tmp_path / "config.yaml"
            config.write_text("{}\n")
            arguments += ["--dlc_config", str(config)]
        cli.main(arguments)
    except SystemExit as exc:
        assert exc.code == 0
    assert len(captured) == 1 and captured[0].file_list == str(listing)


def test_preparation_and_automatic_publication_use_the_frozen_source(tmp_path):
    target = prepare.prepare_jobs(output_dir=tmp_path / "local", remote_job_dir="/data/jobs/new",
        project="/data/dataset", python="/data/env/python", resources="/data/resources",
        speed="slow", publish_to=tmp_path / "mounted")
    receipt = json.loads((target / "Publication receipt.json").read_text())
    copied = tmp_path / "mounted"
    assert receipt["verified"] and receipt["models_copied"] == 0
    assert "/data/jobs/new/package" in (copied / "submit.sh").read_text()
    assert (copied / "package/autobehaver/job_batches.py").is_file()
    assert (copied / "schedule.tsv").read_bytes() == (target / "schedule.tsv").read_bytes()
    assert not (copied / "package/autobehaver/resources").exists()


def test_publication_verifies_copy_skips_models_and_refuses_overwrite(tmp_path):
    prepared = tmp_path / "prepared"; prepared.mkdir()
    (prepared / "submit.sh").write_text("#!/bin/bash\n")
    source = tmp_path / "source"; source.mkdir()
    (source / "autobehaver/resources/dlc").mkdir(parents=True)
    (source / "autobehaver/resources/dlc/model.bin").write_bytes(b"not copied")
    (source / "autobehaver/__init__.py").write_text("__version__='test'\n")
    (source / "pytest-cache-files-old").mkdir()
    (source / "pytest-cache-files-old/cache.txt").write_text("temporary test cache")
    destination = tmp_path / "published"
    receipt = publish_jobs(prepared, destination, source_root=source)
    assert receipt["verified"] and receipt["models_copied"] == 0
    assert (destination / "submit.sh").read_bytes() == (prepared / "submit.sh").read_bytes()
    assert not (destination / "package/autobehaver/resources").exists()
    assert not (destination / "package/pytest-cache-files-old").exists()
    assert not (destination / "PUBLICATION_PENDING.txt").exists()
    before = (destination / "submit.sh").read_bytes()
    with pytest.raises(FileExistsError):
        publish_jobs(prepared, destination, source_root=source)
    assert (destination / "submit.sh").read_bytes() == before


def test_fast_submission_dependencies_and_failure_receipt(tmp_path):
    dataset = recording_fixture(tmp_path / "dataset")
    for fail_at in (0, 4):
        case = tmp_path / str(fail_at); case.mkdir()
        target = prepare.prepare_jobs(output_dir=case / "job files", remote_job_dir=bash_path(case / "job files"),
            project="/data/one dataset", package="/data/code", python="/data/python",
            resources="/data/resources", local_project=dataset)
        fakebin = case / "bin"; fakebin.mkdir()
        fake = fakebin / "sbatch"
        fake.write_text('''#!/bin/bash
set -euo pipefail
count=0; [[ ! -f "$STATE" ]] || read -r count < "$STATE"
count=$((count+1)); printf '%s\\n' "$count" > "$STATE"
printf '%s\\t' "$@" >> "$CALLS"; printf '\\n' >> "$CALLS"
[[ "$count" != "$FAIL_AT" ]] || exit 1
printf '%s;hpc\\n' "$((100000+count))"
''', newline="\n")
        fake.chmod(0o755)
        calls = case / "calls"
        env = dict(os.environ, STATE=bash_path(case / "state"), CALLS=bash_path(calls), FAIL_AT=str(fail_at))
        env["PATH"] = str(fakebin) + os.pathsep + env["PATH"]
        result = subprocess.run([bash_executable(require_flock=True), "--noprofile", "--norc", str(target / "submit.sh")],
                                env=env, text=True, capture_output=True)
        assert result.returncode == (1 if fail_at else 0), result.stderr
        plan = json.loads((target / "plan.json").read_text())
        expected = 3 if fail_at else len(plan["jobs"])
        assert len((target / "jobs.tsv").read_text().splitlines()) == expected
        records = [line.rstrip("\t").split("\t") for line in calls.read_text().splitlines()]
        key_ids = {job["key"]: 100001+i for i,job in enumerate(plan["jobs"])}
        for job, arguments in zip(plan["jobs"], records):
            if job["depends_on"]:
                kind = "afterany" if job["stage"] == "collect" else "afterok"
                dependency = "--dependency="+kind+":"+":".join(str(key_ids[parent]) for parent in job["depends_on"])
                assert dependency in arguments
            script_at = next(i for i,value in enumerate(arguments) if value.endswith(".sbatch"))
            if job["stage"] != "collect":
                assert arguments[script_at+5] == "/data/one dataset"
                assert arguments[script_at+6].endswith("/batches/"+job["file_list"])
        assert {path.name for path in dataset.iterdir()} == {"1_full_movies", "rec_data"}
