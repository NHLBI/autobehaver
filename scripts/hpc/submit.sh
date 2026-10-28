#!/bin/bash
# Submit the desktop-prepared schedule; compute jobs perform all processing.
set -euo pipefail
resume=false
if [[ "${!#:-}" == --resume ]]; then
    resume=true
    set -- "${@:1:$#-1}"
fi
if (( $# != 6 && $# != 9 )); then
    echo "Usage: bash submit.sh PACKAGE MAIN_PYTHON RESOURCES JOB_DIR PROJECT crop|dlc|features [MODEL16 MODEL32 MODEL64]" >&2
    exit 2
fi
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="$5"
models=(); if (( $# == 9 )); then models=("$7" "$8" "$9"); fi
templates=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -d "$bundle/logs" && -d "$bundle/resources" && -f "$bundle/schedule.tsv" && -f "$bundle/speed.txt" ]] || {
    echo "Prepare the job directory on your desktop before submission." >&2; exit 1;
}
# Submission runs on HPC/Linux. Keep this descriptor open through every
# scheduler call and jobs.tsv update; process exit releases ownership.
command -v flock >/dev/null || { echo "Submission requires Linux flock; run this command on HPC." >&2; exit 1; }
exec 9>>"$bundle/.submission.lock"
flock -n 9 || { echo "Another submission command is active for this bundle; no jobs submitted." >&2; exit 1; }
[[ "$resume" == true || ! -f "$bundle/jobs.tsv" ]] || {
    echo "Accepted jobs are already recorded in $bundle/jobs.tsv. Use the same command with --resume to restart failed or cancelled work." >&2; exit 1;
}
[[ ! -f "$bundle/PUBLICATION_PENDING.txt" ]] || { echo "Job publication is incomplete; review the copy error before submission." >&2; exit 1; }
read -r speed < "$bundle/speed.txt"
[[ "$speed" == fast || "$speed" == slow ]] || { echo "Invalid prepared speed: $speed" >&2; exit 1; }
declare -A accepted
declare -A previous states replace
declare -A own_jobs
if [[ "$resume" == true && -f "$bundle/jobs.tsv" ]]; then
    while IFS=$'\t' read -r key job; do
        [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Invalid accepted job record; inspect before resuming.' >&2; exit 1; }
        own_jobs[$key]="$job"
    done < "$bundle/jobs.tsv"
fi
recovery=false
if [[ -f "$bundle/recovery_expected.tsv" ]]; then
    recovery=true
    read -r recovery_parent < "$bundle/recovery_parent.txt"
    [[ "$recovery_parent" == /* && -d "$recovery_parent" ]] || { echo 'Recovery parent unavailable' >&2; exit 1; }
    # Serialize recovery attempts of this run. Never cancel or alter its jobs.
    exec 8>>"$recovery_parent/.recovery.lock"
    flock -n 8 || { echo 'Another recovery submission is active' >&2; exit 1; }
    recovery_ids=()
    while IFS=$'\t' read -r key job role; do
        [[ "$job" =~ ^[0-9]+$ && "$key" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'Invalid recovery record' >&2; exit 1; }
        recovery_ids+=("$job")
        previous[$key]="$job"
        if [[ "$role" == replace ]]; then
            replace[$key]=true
            if [[ -f "$recovery_parent/recovery_claims.tsv" ]] && awk -F '\t' -v k="$key" -v j="$job" '$1==k && $2==j {found=1} END {exit !found}' "$recovery_parent/recovery_claims.tsv"; then
                # Continue this exact partially submitted recovery under both locks.
                # Claims belonging to another bundle (or without our durable job ID)
                # remain a hard stop; do not reconstruct missing receipts by guessing.
                if [[ "$resume" != true || -z "${own_jobs[$key]:-}" ]] || ! awk -F '\t' -v k="$key" -v j="$job" -v b="$bundle" '$1==k && $2==j && $4!=b {bad=1} END {exit bad}' "$recovery_parent/recovery_claims.tsv"; then
                    echo "A replacement for $key ($job) already exists. Open its recovery run." >&2; exit 1
                fi
            fi
        elif [[ "$role" == retain ]]; then
            replace[$key]=false; accepted[$key]="$job"
        else echo 'Invalid recovery role' >&2; exit 1; fi
    done < "$bundle/recovery_expected.tsv"
    recovery_list=$(IFS=,; echo "${recovery_ids[*]}")
    report=$(sacct -X -j "$recovery_list" --noheader --parsable2 --format=JobIDRaw,State 8>&- 9>&-) || exit 1
    while IFS='|' read -r job state extra; do
        [[ "$job" =~ ^[0-9]+$ ]] || continue
        state="${state%% *}"; states[$job]="${state%+}"
    done <<< "$report"
    while IFS=$'\t' read -r key job role; do
        state="${states[$job]:-}"
        if [[ "$role" == replace ]]; then
            case "$state" in FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) ;;
                *) echo "$key is now $state; rebuild recovery. No jobs submitted." >&2; exit 1;; esac
        else
            case "$state" in COMPLETED|PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING|STAGE_OUT) ;;
                *) echo "Retained job $key is now $state; refresh and rebuild recovery. No jobs submitted." >&2; exit 1;; esac
        fi
    done < "$bundle/recovery_expected.tsv"
fi
if [[ "$resume" == true && -f "$bundle/jobs.tsv" ]]; then
    while IFS=$'\t' read -r key job; do
        job="${job%$'\r'}"
        [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid recorded job: $key $job" >&2; exit 1; }
        previous[$key]="$job"
    done < "$bundle/jobs.tsv"
    accounting_ids=$(IFS=,; echo "${previous[*]}")
    report=$(sacct -j "$accounting_ids" --noheader --parsable2 --format=JobIDRaw,State 9>&-) || {
        echo "Cannot read scheduler state; no restart jobs submitted." >&2; exit 1;
    }
    while IFS='|' read -r job state extra; do
        [[ "$job" =~ ^[0-9]+$ ]] || continue
        state="${state%$'\r'}"; state="${state%% *}"; state="${state%+}"
        states[$job]="$state"
    done <<< "$report"
    # Decide the restart before submitting anything. Do not duplicate active
    # allocations or change the dependencies of already running jobs.
    while IFS=$'\t' read -r key stage dependencies list index count; do
        job="${previous[$key]:-}"
        state=""; [[ -z "$job" ]] || state="${states[$job]:-}"
        case "$state" in
            COMPLETED|PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|REQUEUE_FED|RESIZING|SIGNALING|STAGE_OUT) replace[$key]=false ;;
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED) replace[$key]=true ;;
            '')
                [[ -z "$job" ]] || { echo "Scheduler state unavailable for $key ($job); wait and retry --resume. No jobs submitted." >&2; exit 1; }
                replace[$key]=true ;;
            *) echo "Unrecognized scheduler state for $key ($job): $state. No jobs submitted." >&2; exit 1 ;;
        esac
        parent_replaced=false
        if [[ "$dependencies" != - ]]; then
            IFS=, read -r -a parents <<< "$dependencies"
            for parent in "${parents[@]}"; do
                [[ "${replace[$parent]:-false}" != true ]] || parent_replaced=true
            done
        fi
        if [[ "$parent_replaced" == true && "${replace[$key]}" == false ]]; then
            if [[ "$stage" == collect && "$state" == COMPLETED ]]; then
                replace[$key]=true
            else
                echo "Dependent job $key ($job) is still $state while its parent needs restart. Wait for scheduler cancellation before --resume. No jobs submitted." >&2
                exit 1
            fi
        fi
    done < "$bundle/schedule.tsv"
fi
if [[ -f "$bundle/incremental_receipt.sha256" ]]; then
    needs_update=true
    if [[ "$resume" == true ]]; then
        needs_update=false
        while IFS=$'\t' read -r key stage dependencies list index count; do
            if [[ "$stage" != collect && "${replace[$key]:-true}" == true ]]; then needs_update=true; fi
        done < "$bundle/schedule.tsv"
    fi
    # An already accepted append may have committed while submission was offline.
    # Finishing only accounting must not require its previous matrix generation.
    if [[ "$needs_update" == true ]]; then
        sha256sum -c "$bundle/incremental_receipt.sha256" || {
            echo 'The master matrix generation changed after Build. Inspect the completed update before preparing another run.' >&2; exit 1;
        }
    fi
fi
all_jobs=()
if [[ "$recovery" == true ]]; then
    while IFS=$'\t' read -r key job role; do
        [[ "$role" != retain ]] || all_jobs+=("$job")
    done < "$bundle/recovery_expected.tsv"
fi
submitted_count=0
while IFS=$'\t' read -r key stage dependencies list index count; do
    [[ -n "$key" ]] || continue
    if [[ "$resume" == true && "${replace[$key]:-true}" == false ]]; then
        accepted[$key]="${previous[$key]}"
        [[ "$stage" == collect ]] || all_jobs+=("${previous[$key]}")
        continue
    fi
    ids=()
    if [[ "$dependencies" != - ]]; then
        IFS=, read -r -a parents <<< "$dependencies"
        for parent in "${parents[@]}"; do
            [[ -n "${accepted[$parent]:-}" ]] || { echo "Unsubmitted dependency: $parent" >&2; exit 1; }
            parent_id="${accepted[$parent]}"
            # Completed parents have already satisfied the dependency. Slurm
            # may have removed their IDs from its live dependency registry.
            if [[ ( "$resume" == true || "$recovery" == true ) && "${states[$parent_id]:-}" == COMPLETED ]]; then
                continue
            fi
            ids+=("$parent_id")
        done
    fi
    options=(--parsable --kill-on-invalid-dep=yes
        --job-name="autobehaver_$key" --output="$bundle/logs/${key}_%j.out" --error="$bundle/logs/${key}_%j.err")
    if (( ${#ids[@]} )); then
        dependency=$(IFS=:; echo "${ids[*]}")
        kind=afterok; [[ "$stage" != collect ]] || kind=afterany
        options+=(--dependency="$kind:$dependency")
    fi
    selection=-
    if [[ "$list" != - ]]; then
        selection="$bundle/batches/$list"
        [[ -f "$selection" ]] || { echo "Missing prepared batch list: $selection" >&2; exit 1; }
    fi
    script="$stage"
    case "$stage" in
        crop|dlc|features) arguments=("$project" "$selection") ;;
        posepred) arguments=("$project" "$selection" "${models[@]}") ;;
        mmgen) [[ "$count" != 1 ]] || script=mmgen_finish; arguments=("$project" "$selection" "$index" "$count" process) ;;
        mmgen_finish)
            script=mmgen_finish
            action=merge
            [[ ! -f "$bundle/finalize_compatibility.json" ]] || action=finalize
            arguments=("$project" "$selection" 0 "$count" "$action") ;;
        collect) accounting_ids=$(IFS=,; echo "${all_jobs[*]}"); arguments=("$accounting_ids") ;;
        *) echo "Unknown prepared stage: $stage" >&2; exit 1 ;;
    esac
    if [[ "$stage" == dlc && "$speed" == slow ]]; then
        options+=(--time=7-00:00:00 --gres=gpu:a100:1,lscratch:200)
    fi
    # HPC asks for at most one separate sbatch submission per second.
    # Keep the pause between new jobs, including during --resume.
    if (( submitted_count > 0 )); then sleep 1; fi
    if ! result=$(sbatch "${options[@]}" "$templates/$script.sbatch" "$package" "$python" "$resources" "$bundle" "${arguments[@]}" 8>&- 9>&-); then
        echo "Submission stopped at $key. Accepted jobs, if any, are in $bundle/jobs.tsv." >&2; exit 1
    fi
    result="${result%%;*}"
    [[ "$result" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch result: $result" >&2; exit 1; }
    printf '%s\t%s\n' "$key" "$result" >> "$bundle/jobs.tsv"
    if [[ "$recovery" == true && -n "${previous[$key]:-}" ]]; then
        printf '%s\t%s\t%s\t%s\n' "$key" "${previous[$key]}" "$result" "$bundle" >> "$recovery_parent/recovery_claims.tsv"
    fi
    (( submitted_count += 1 ))
    accepted[$key]="$result"
    [[ "$stage" == collect ]] || all_jobs+=("$result")
done < "$bundle/schedule.tsv"
cat "$bundle/jobs.tsv"
