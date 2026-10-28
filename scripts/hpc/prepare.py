"""Prepare a single-dataset HPC submission on the desktop; never submit jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys

# Direct execution from scripts/hpc must find the same source package as -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autobehaver.job_batches import build_plan


def prepare_jobs(*, output_dir, remote_job_dir, project, python, resources, package=None, start_at="crop", inference_model=None, speed="fast", local_project=None, publish_to=None, moviechecker_mode="reviewed", only_original_movies=None, batch_sizes=None, prepared_plan=None, batching=None, new_only=False):
    """Copy maintained templates and write one lightweight submission command."""
    models = tuple(inference_model or ())
    if publish_to:
        package = remote_job_dir.rstrip("/") + "/package"
    if not package:
        raise ValueError("Provide --package, or --publish-to to publish the current source package automatically")
    if models and len(models) != 3:
        raise ValueError("Provide exactly three PosePred model paths")
    paths = (remote_job_dir, project, package, python, resources) + models
    if any(not value.startswith("/") or "\x00" in value or "\n" in value or "\r" in value for value in paths):
        raise ValueError("HPC paths must be absolute Linux paths without control characters")
    if start_at not in ("crop", "dlc", "features"):
        raise ValueError("Start stage must be crop, dlc or features")
    if speed not in ("fast", "slow"):
        raise ValueError("Speed must be fast or slow")
    if PurePosixPath(remote_job_dir).is_relative_to(PurePosixPath(project)):
        raise ValueError("Keep the job directory outside the dataset's processing folders")
    if prepared_plan is None:
        plan, file_lists = build_plan(speed=speed, start_at=start_at,
            local_project=local_project, moviechecker_mode=moviechecker_mode,
            only_original_movies=only_original_movies, batch_sizes=batch_sizes,
            batching=batching, new_only=new_only)
    else:
        plan, file_lists = prepared_plan
    plan["project"] = project
    plan["start_at"] = start_at
    if plan.get('incremental'):
        plan['incremental'].setdefault('append_batch_id', PurePosixPath(remote_job_dir).name)
    target = Path(output_dir)
    if publish_to and target.resolve() == Path(publish_to).resolve():
        raise ValueError("Use a separate local --output-dir and mounted --publish-to destination")
    if local_project and target.resolve().is_relative_to(Path(local_project).resolve()):
        raise ValueError("Keep job storage outside the local dataset too")
    if publish_to and local_project and Path(publish_to).resolve().is_relative_to(Path(local_project).resolve()):
        raise ValueError("Keep mounted publication outside the dataset's processing folders")
    target.mkdir(parents=True, exist_ok=False)
    templates = target / "templates"
    templates.mkdir()
    source = Path(__file__).resolve().parent
    for name in ("submit.sh", "runtime.sh", "crop.sbatch", "dlc.sbatch", "features.sbatch", "posepred.sbatch", "mmgen.sbatch", "mmgen_finish.sbatch", "collect.sbatch"):
        shutil.copyfile(source / name, templates / name)
    for name in ("logs", "resources"):
        (target / name).mkdir()
    if file_lists:
        (target / "batches").mkdir()
    for name, entries in file_lists.items():
        (target / "batches" / name).write_text("\n".join(entries)+"\n", encoding="utf-8", newline="\n")
    (target / "plan.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8", newline="\n")
    (target / 'continuation_supported.txt').write_text('1\n', encoding='ascii', newline='\n')
    if plan.get('incremental'):
        info = plan['incremental']
        (target / 'append_batch_id.txt').write_text(info['append_batch_id']+'\n', encoding='utf-8', newline='\n')
        (target / 'incremental_receipt.sha256').write_text(
            info['receipt_sha256'] + '  ' + project.rstrip('/') + '/8_master_matrix/MMGEN_OUTPUT_RECEIPT.json\n', encoding='utf-8', newline='\n')
    lines = []
    for job in plan["jobs"]:
        if job.get("retained"):
            continue
        values = [job["key"], job["stage"], ",".join(job["depends_on"]) or "-",
                  job["file_list"], str(job["shard_index"]), str(job["num_shards"])]
        lines.append("\t".join(values))
    (target / "schedule.tsv").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    (target / "speed.txt").write_text(speed+"\n", encoding="ascii", newline="\n")
    (target / "review_mode.txt").write_text(moviechecker_mode+"\n", encoding="ascii", newline="\n")
    command = ["bash", remote_job_dir.rstrip("/") + "/templates/submit.sh", package, python, resources, remote_job_dir, project, start_at, *models]
    (target / "submit.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(command) + ' "$@"\n', encoding="utf-8", newline="\n")
    (target / "paths.txt").write_text("\n".join((project, package, python, resources, remote_job_dir, start_at)) + "\n", encoding="utf-8", newline="\n")
    if publish_to:
        from autobehaver.job_publication import publish_jobs
        receipt = publish_jobs(target, publish_to, source_root=Path(__file__).resolve().parents[2])
        (target / "Publication receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New local directory for prepared job files")
    parser.add_argument("--remote-job-dir", required=True, help="The same job directory's absolute HPC path")
    parser.add_argument("--project", required=True, help="One dataset root on HPC; all numbered stage folders stay here")
    parser.add_argument("--local-project", help="Mounted/local view of that dataset, used only for Fast desktop preparation")
    parser.add_argument("--speed", choices=("fast", "slow"), default="fast", help="Fast: balanced parallel batches with job caps; Slow: one full-dataset job per stage")
    parser.add_argument("--package", help="Existing Autobehaver source directory, required when --publish-to is omitted")
    parser.add_argument("--publish-to", help="New mounted HPC job-bundle directory; copy and verify jobs plus the current lightweight source package")
    parser.add_argument("--python", required=True, help="Existing main-environment Python executable on HPC")
    parser.add_argument("--resources", required=True, help="Existing model/normalization resource directory on HPC")
    parser.add_argument("--inference-model", nargs=3, help="Three existing PosePred checkpoint paths; default: RESOURCES/posepred/nh16_e301.pth, nh32_e301.pth, nh64_e301.pth")
    parser.add_argument("--start-at", choices=("crop", "dlc", "features"), default="crop", help="crop: full movies; dlc: existing crops; features: existing 3_dlc coordinates through KeyPre, normalization, PosePred and MMGen")
    parser.add_argument("--moviechecker-mode", choices=("reviewed", "recording-only"), default="reviewed",
                        help="Default uses completed MovieChecker review; recording-only uses recording occupancy metadata without MovieChecker")
    parser.add_argument("--only-original-movies", nargs="+", metavar="MOVIE.mp4",
                        help="Prepare a Fast run for these exact original movie basenames in the same dataset")
    parser.add_argument("--batch-sizes", nargs=5, type=int,
                        metavar=("CROP", "DLC", "FEATURES", "POSEPRED", "MMGEN"),
                        help="Legacy explicit per-stage sizes, without shared balancing/caps; prefer --batching")
    parser.add_argument('--batching', nargs=4, type=int, metavar=('FULL_MOVIES', 'CHAMBERS', 'CPU_JOBS', 'GPU_JOBS'),
                        help='Shared minimum inputs per job and maximum jobs per stage; defaults 2 20 50 50, independent of worker counts')
    parser.add_argument('--new-only', action='store_true', help='Process recordings absent from an existing MMGen source ledger; update the combined matrix')
    args = parser.parse_args(argv)
    if args.batch_sizes is not None:
        args.batch_sizes = dict(zip(("crop", "dlc", "features", "posepred", "mmgen"), args.batch_sizes))
    if args.batching is not None:
        args.batching = dict(zip(('full_movies', 'chambers', 'cpu_jobs', 'gpu_jobs'), args.batching))
    try:
        target = prepare_jobs(**vars(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Prepared job files: {target}")
    plan = json.loads((target / "plan.json").read_text())
    print(f"Speed: {plan['speed']}")
    for stage in ("crop", "dlc", "features", "posepred", "mmgen", "mmgen_finish", "collect"):
        count = sum(j["stage"] == stage for j in plan["jobs"])
        if count:
            print(f"  {stage}: {count} job(s)")
    print("No jobs submitted. All processing stages use the same dataset root.")
    if args.publish_to:
        print(f"Published and verified: {args.publish_to}")
    print(f"Submission command: bash {shlex.quote(args.remote_job_dir.rstrip('/')+'/submit.sh')}")


if __name__ == "__main__":
    main()
