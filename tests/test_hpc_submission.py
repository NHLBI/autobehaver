"""Small path-routing and fake-scheduler tests; no processing or cluster access."""

from pathlib import Path
import importlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
TEMPLATES = PACKAGE / "scripts/hpc"
spec = importlib.util.spec_from_file_location("hpc_prepare", TEMPLATES / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def bash_executable(*, require_flock=False):
    if require_flock and os.name == "nt":
        raise unittest.SkipTest("Native HPC submission locking requires Linux flock; Windows prepares jobs only")
    if require_flock and not shutil.which("flock"):
        raise unittest.SkipTest("Linux flock is unavailable")
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            return str(candidate)
    candidate = shutil.which("bash")
    if not candidate:
        raise unittest.SkipTest("Bash is not available")
    return candidate


def bash_path(path):
    value = str(path).replace("\\", "/")
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


def prepared_fixture(tmp_path, start_at="crop"):
    target = tmp_path / "job files"
    prepare.prepare_jobs(output_dir=target, remote_job_dir=bash_path(target),
                         project="/data/one dataset", package="/data/package",
                         python="/data/environment/bin/python", resources="/data/resources",
                         start_at=start_at, speed="slow")
    return target


def test_desktop_preparation_creates_only_job_storage(tmp_path):
    target = prepared_fixture(tmp_path)
    assert {p.name for p in target.iterdir() if p.is_dir()} == {"templates", "logs", "resources"}
    command = shlex.split((target / "submit.sh").read_text().splitlines()[-1])
    assert command[-3:] == ["/data/one dataset", "crop", "$@"]
    assert not any(p.name.startswith(("trial_", "combined", "1_", "2_", "3_", "4_", "5_", "6_", "8_")) for p in target.rglob("*"))
    with unittest.TestCase().assertRaises(FileExistsError):
        prepared_fixture(tmp_path)


def test_job_storage_cannot_be_put_inside_dataset(tmp_path):
    with unittest.TestCase().assertRaisesRegex(ValueError, "outside the dataset"):
        prepare.prepare_jobs(output_dir=tmp_path / "unused", remote_job_dir="/data/dataset/jobs",
                             project="/data/dataset", package="/data/package", python="/data/env/python",
                             resources="/data/resources", speed="slow")
    assert not (tmp_path / "unused").exists()


def check_one_dataset_chain(tmp_path, start_at, fail_at, accepted, models=None):
    target = prepared_fixture(tmp_path, start_at)
    if models:
        command = (target / "submit.sh").read_text().replace(' "$@"', ' '+shlex.join(models)+' "$@"')
        (target / "submit.sh").write_text(command, newline="\n")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    scheduler = fakebin / "sbatch"
    scheduler.write_text('''#!/bin/bash
set -euo pipefail
count=0; [[ ! -f "$STATE" ]] || read -r count < "$STATE"
count=$((count+1)); printf '%s\\n' "$count" > "$STATE"
printf '%s\\t' "$@" >> "$CALLS"; printf '\\n' >> "$CALLS"
[[ "$count" != "$FAIL_AT" ]] || { echo "simulated submission rejection" >&2; exit 1; }
printf '%s;hpc\\n' "$((100000+count))"
''', newline="\n")
    scheduler.chmod(0o755)
    state = tmp_path / "state"
    calls = tmp_path / "calls"
    environment = dict(os.environ, STATE=bash_path(state), CALLS=bash_path(calls), FAIL_AT=str(fail_at))
    environment["PATH"] = str(fakebin) + os.pathsep + environment["PATH"]
    arguments = [bash_executable(require_flock=True), "--noprofile", "--norc", str(target / "submit.sh")]
    finished = subprocess.run(arguments, env=environment, text=True, capture_output=True)
    assert finished.returncode == (0 if fail_at == 0 else 1), finished.stderr
    receipt = target / "jobs.tsv"
    records = receipt.read_text().splitlines() if receipt.exists() else []
    assert len(records) == accepted
    for record in calls.read_text().splitlines():
        values = record.rstrip("\t").split("\t")
        script_index = next(i for i, value in enumerate(values) if value.endswith(".sbatch"))
        stage = Path(values[script_index]).stem
        if stage != "collect":
            assert values[script_index + 5] == "/data/one dataset"
        assert not any("trial_" in value or "/combined" in value for value in values)
    if fail_at == 0:
        expected = ["crop", "dlc", "features", "posepred", "mmgen", "collect"]
        if start_at == "dlc":
            expected.remove("crop")
        assert [record.split("\t")[0] for record in records] == expected
        submitted = [row.rstrip("\t").split("\t") for row in calls.read_text().splitlines()]
        assert not any(value.startswith("--dependency=") for value in submitted[0])
        for i, values in enumerate(submitted[1:-1], 1):
            assert f"--dependency=afterok:{100000+i}" in values
        assert "--dependency=afterany:" + ":".join(str(100000+i) for i in range(1, accepted)) in submitted[-1]
        again = subprocess.run(arguments, env=environment, text=True, capture_output=True)
        assert again.returncode != 0 and len(calls.read_text().splitlines()) == accepted
    assert {p.name for p in target.iterdir() if p.is_dir()} == {"templates", "logs", "resources"}
    if models and not fail_at:
        pose_call = next(values for values in submitted if any(value.endswith("/posepred.sbatch") for value in values))
        assert pose_call[-3:] == models


def test_external_checkpoints_reach_posepred_job(tmp_path):
    models = ["/data/model16.pth", "/data/model32.pth", "/data/model64.pth"]
    prepared = prepare.prepare_jobs(output_dir=tmp_path / "direct", remote_job_dir="/data/jobs",
        project="/data/dataset", package="/data/package", python="/data/python", resources="/data/resources",
        inference_model=models, speed="slow")
    assert shlex.split((prepared / "submit.sh").read_text().splitlines()[-1])[-4:-1] == models
    case = tmp_path / "scheduler"
    case.mkdir()
    check_one_dataset_chain(case, "crop", 0, 6, models=models)


def test_pipeline_chain_cases(tmp_path):
    for i, (start_at, fail_at, accepted) in enumerate((("crop", 0, 6), ("dlc", 0, 5), ("crop", 1, 0), ("crop", 2, 1))):
        case = tmp_path / str(i)
        case.mkdir()
        check_one_dataset_chain(case, start_at, fail_at, accepted)


def test_native_stage_commands_resolve_standard_folders():
    from autobehaver.pipeline_layout import apply_main_dir_overrides, stage_paths_for_main_dir

    for path in TEMPLATES.glob("*.sbatch"):
        logical = path.read_text().replace("\\\n", " ")
        for line in logical.splitlines():
            if '"$python" -m autobehaver.' not in line:
                continue
            if 'AUTOBEHAVER_CROP_REUSE_CODES=' in line:
                continue  # Recovery provenance helper takes a JSON file, not stage paths.
            tokens = shlex.split(line.replace('"${selection_args[@]}"', '').replace('"${shard_args[@]}"', '').replace('"${resume_args[@]}"', '').replace('"${review_args[@]}"', '').replace('"${append_args[@]}"', ''))
            at = tokens.index("-m")
            module, arguments = tokens[at+1], tokens[at+2:]
            if module == "autobehaver.resource_monitor":
                assert arguments == ["$bundle/resources", "--output", "$resource_report"]
                continue
            assert "--main_dir" in arguments
            assert not {"--output_dir", "--input_dir", "--poscalc_dir", "--inference_csv_dir"}.intersection(arguments)
            arguments = ["/data/dataset" if value == "$project" else value for value in arguments]
            cli = importlib.import_module(module + ".cli")
            parser = (getattr(cli, "build_arg_parser", None) or cli.build_parser)()
            parsed = parser.parse_args(arguments)
            stage = module.rsplit(".", 1)[-1]
            apply_main_dir_overrides(parsed, stage=stage)
            for field, expected in stage_paths_for_main_dir(stage, "/data/dataset").items():
                assert getattr(parsed, field) == expected
            if stage == "mmgen":
                assert parsed.random_seed == 7 and parsed.source_frame_offset == 0
                assert parsed.video_fps == 40 and parsed.keep_temp_files is False
            if stage == "posepred_inference":
                assert parsed.inf_average == 10 and parsed.use_tta and len(parsed.inference_model) == 3
                assert parsed.tta_seed == 42


def test_crop_job_respects_review_mode(tmp_path):
    bundle = tmp_path / "bundle"
    for name in ("templates", "logs", "resources"):
        (bundle / name).mkdir(parents=True)
    (bundle / "templates/runtime.sh").write_text('run_stage() { shift; "$@"; }\n', newline="\n")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "lscpu").write_text('#!/bin/bash\necho test-cpu\n', newline="\n")
    (fakebin / "lscpu").chmod(0o755)
    python = fakebin / "stage-python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n', newline="\n")
    python.chmod(0o755)
    project = tmp_path / "dataset"
    (project / "rec_data/trials").mkdir(parents=True)
    env = dict(os.environ, CAPTURE=bash_path(tmp_path / "args.txt"), SLURM_JOB_ID="123")
    env["PATH"] = str(fakebin) + os.pathsep + env["PATH"]
    command = [bash_executable(), "--noprofile", "--norc", str(TEMPLATES / "crop.sbatch"),
               "/data/code", bash_path(python), "/data/resources", bash_path(bundle),
               bash_path(project), "-"]
    for mode in ("reviewed", "recording-only"):
        (bundle / "review_mode.txt").write_text(mode + "\n", newline="\n")
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        arguments = (tmp_path / "args.txt").read_text().splitlines()
        if mode == "reviewed":
            assert "--require_exclusions" in arguments
            assert "--no_require_exclusions" not in arguments
        else:
            assert "--no_require_exclusions" in arguments
            at = arguments.index("--exclusions_csv")
            assert arguments[at + 1].endswith("/NO_MOVIECHECKER_EXCLUSIONS.csv")


def test_bash_syntax():
    for path in [*TEMPLATES.glob("*.sh"), *TEMPLATES.glob("*.sbatch")]:
        subprocess.run([bash_executable(), "--noprofile", "--norc", "-n", str(path)], check=True, capture_output=True)


def test_native_features_retry_before_normalization(tmp_path):
    target = prepared_fixture(tmp_path)
    fakebin = tmp_path/'bin'
    fakebin.mkdir()
    time_tool = fakebin/'timing'
    time_tool.write_text('''#!/bin/bash
shift; report="$2"; shift 2
"$@"; code=$?
printf 'Exit status: %s\\n' "$code" > "$report"
exit "$code"
''', newline='\n')
    time_tool.chmod(0o755)
    # Substitute only GNU time; retain the actual native stage and retry logic.
    runtime = target/'templates/runtime.sh'
    runtime.write_text(runtime.read_text().replace('/usr/bin/time', shlex.quote(bash_path(time_tool))), newline='\n')
    hardware = fakebin/'lscpu'
    hardware.write_text('#!/bin/bash\necho local-test\n', newline='\n')
    hardware.chmod(0o755)
    python = fakebin/'stage-python'
    python.write_text('''#!/bin/bash
if [[ "$2" == autobehaver.preinf.keypre ]]; then
    count=0; [[ ! -f "$STATE" ]] || read -r count < "$STATE"
    count=$((count+1)); printf '%s\\n' "$count" > "$STATE"
    printf 'keypre %s\\n' "$count" >> "$HISTORY"
    [[ "$FAIL_MODE" != signal ]] || exit 130
    [[ "$FAIL_MODE" != persistent && "$count" != 1 ]] || exit 1
else
    echo normalized >> "$HISTORY"
fi
''', newline='\n')
    python.chmod(0o755)
    env = dict(os.environ, STATE=bash_path(tmp_path/'state'), HISTORY=bash_path(tmp_path/'history'),
               FAIL_MODE='once', SLURM_JOB_ID='123', AUTOBEHAVER_RETRY_DELAY='0')
    env['PATH'] = str(fakebin)+os.pathsep+env['PATH']
    command = [bash_executable(), '--noprofile', '--norc', str(target/'templates/features.sbatch'),
               '/data/package', bash_path(python), '/data/resources', bash_path(target), '/data/dataset', '-']
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path/'history').read_text().splitlines() == ['keypre 1', 'keypre 2', 'normalized']
    assert (target/'logs/keypre_123_time.txt').read_text().strip() == 'Exit status: 1'
    assert (target/'logs/keypre_123_time_retry.txt').read_text().strip() == 'Exit status: 0'
    (tmp_path/'state').unlink()
    (tmp_path/'history').unlink()
    env['FAIL_MODE'] = 'persistent'
    failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert failed.returncode == 1
    assert (tmp_path/'history').read_text().splitlines() == ['keypre 1', 'keypre 2']
    (tmp_path/'state').unlink()
    (tmp_path/'history').unlink()
    env['FAIL_MODE'] = 'signal'
    interrupted = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert interrupted.returncode == 130
    assert (tmp_path/'history').read_text().splitlines() == ['keypre 1']


def test_resume_rebuilds_failed_dependencies_without_duplicate_jobs(tmp_path):
    target = prepared_fixture(tmp_path)
    (target/'jobs.tsv').write_text('crop\t101\ndlc\t102\nfeatures\t103\nposepred\t104\nmmgen\t105\ncollect\t106\n')
    fakebin = tmp_path/'bin'
    fakebin.mkdir()
    scheduler = fakebin/'sbatch'
    scheduler.write_text('''#!/bin/bash
count=0; [[ ! -f "$STATE" ]] || read -r count < "$STATE"
count=$((count+1)); printf '%s\\n' "$count" > "$STATE"
printf '%s\\t' "$@" >> "$CALLS"; printf '\\n' >> "$CALLS"
echo "$((200+count))"
''', newline='\n')
    scheduler.chmod(0o755)
    accounting = fakebin/'sacct'
    accounting.write_text('#!/bin/bash\ncat "$ACCOUNTING"\n', newline='\n')
    accounting.chmod(0o755)
    report = tmp_path/'accounting'
    report.write_text('101|COMPLETED|\n102|COMPLETED|\n103|FAILED|\n104|CANCELLED|\n105|CANCELLED|\n106|COMPLETED|\n')
    env = dict(os.environ, STATE=bash_path(tmp_path/'state'), CALLS=bash_path(tmp_path/'calls'),
               ACCOUNTING=bash_path(report))
    env['PATH'] = str(fakebin)+os.pathsep+env['PATH']
    command = [bash_executable(require_flock=True), '--noprofile', '--norc', str(target/'submit.sh'), '--resume']
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    calls = [(line.rstrip('\t').split('\t')) for line in (tmp_path/'calls').read_text().splitlines()]
    assert len(calls) == 4
    assert not any(value.startswith('--dependency=') for value in calls[0])
    assert '--dependency=afterok:201' in calls[1]
    assert '--dependency=afterok:202' in calls[2]
    assert '--dependency=afterany:201:202:203' in calls[3]
    assert (target/'jobs.tsv').read_text().splitlines()[-4:] == [
        'features\t201', 'posepred\t202', 'mmgen\t203', 'collect\t204']
    report.write_text('101|COMPLETED|\n102|COMPLETED|\n201|COMPLETED|\n202|RUNNING|\n203|PENDING|\n204|PENDING|\n')
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert again.returncode == 0, again.stderr
    assert len((tmp_path/'calls').read_text().splitlines()) == 4
    report.write_text('101|COMPLETED|\n102|COMPLETED|\n201|FAILED|\n202|PENDING|\n203|PENDING|\n204|PENDING|\n')
    blocked = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert blocked.returncode != 0 and 'No jobs submitted' in blocked.stderr
    assert len((tmp_path/'calls').read_text().splitlines()) == 4
    report.write_text('101|COMPLETED|\n102|COMPLETED|\n')
    unknown = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert unknown.returncode != 0 and 'Scheduler state unavailable' in unknown.stderr
    assert len((tmp_path/'calls').read_text().splitlines()) == 4


def test_resume_keeps_independent_successful_branches(tmp_path):
    target = prepared_fixture(tmp_path)
    schedule = [
        ('features_0', 'features', '-'), ('features_1', 'features', '-'),
        ('posepred_0', 'posepred', 'features_0'), ('posepred_1', 'posepred', 'features_1'),
        ('mmgen_0', 'mmgen', 'posepred_0,posepred_1'), ('mmgen_finish', 'mmgen_finish', 'mmgen_0'),
        ('collect', 'collect', 'features_0,features_1,posepred_0,posepred_1,mmgen_0,mmgen_finish')]
    (target/'schedule.tsv').write_text(''.join(f'{key}\t{stage}\t{parents}\t-\t0\t1\n'
        for key, stage, parents in schedule), newline='\n')
    (target/'jobs.tsv').write_text(''.join(f'{key}\t{index+101}\n' for index, (key, _, _) in enumerate(schedule)), newline='\n')
    fakebin = tmp_path/'bin'
    fakebin.mkdir()
    scheduler = fakebin/'sbatch'
    scheduler.write_text('''#!/bin/bash
count=0; [[ ! -f "$STATE" ]] || read -r count < "$STATE"
count=$((count+1)); printf '%s\\n' "$count" > "$STATE"
printf '%s\\t' "$@" >> "$CALLS"; printf '\\n' >> "$CALLS"
echo "$((200+count))"
''', newline='\n')
    accounting = fakebin/'sacct'
    accounting.write_text('''#!/bin/bash
printf '101|FAILED|\\n102|COMPLETED|\\n103|CANCELLED|\\n104|COMPLETED|\\n105|CANCELLED|\\n106|CANCELLED|\\n107|COMPLETED|\\n'
''', newline='\n')
    scheduler.chmod(0o755)
    accounting.chmod(0o755)
    env = dict(os.environ, STATE=bash_path(tmp_path/'state'), CALLS=bash_path(tmp_path/'calls'))
    env['PATH'] = str(fakebin)+os.pathsep+env['PATH']
    completed = subprocess.run([bash_executable(require_flock=True), '--noprofile', '--norc', str(target/'submit.sh'), '--resume'],
                               env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    calls = [line.rstrip('\t').split('\t') for line in (tmp_path/'calls').read_text().splitlines()]
    assert len(calls) == 5
    assert '--dependency=afterok:201' in calls[1]
    assert '--dependency=afterok:202' in calls[2]
    assert '--dependency=afterok:203' in calls[3]
    assert '--dependency=afterany:201:202:203:204' in calls[4]


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for function in (test_desktop_preparation_creates_only_job_storage,
                     test_job_storage_cannot_be_put_inside_dataset,
                     test_external_checkpoints_reach_posepred_job,
                     test_pipeline_chain_cases,
                     test_native_features_retry_before_normalization,
                     test_resume_rebuilds_failed_dependencies_without_duplicate_jobs,
                     test_resume_keeps_independent_successful_branches):
        def invoke(function=function):
            with tempfile.TemporaryDirectory(prefix="autobehaver_submission_") as directory:
                function(Path(directory))
        suite.addTest(unittest.FunctionTestCase(invoke, description=function.__name__))
    for function in (test_native_stage_commands_resolve_standard_folders, test_bash_syntax):
        suite.addTest(unittest.FunctionTestCase(function))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
