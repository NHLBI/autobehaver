#!/bin/bash
#SBATCH --job-name=autobehaver_benchmarks
#SBATCH --partition=norm
#SBATCH --constraint=e7543
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --time=00:15:00
set -euo pipefail
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="${5:-}"
export PYTHONPATH="$package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
# Each invocation retains its own reports, including collector retries.
collection_id="${SLURM_JOB_ID:-local}_$(date -u +%Y%m%dT%H%M%S)_$$"
scheduler_report="$bundle/scheduler_report_${collection_id}.psv"
resource_report="$bundle/resource_summaries_${collection_id}.csv"
sacct -j "$project" --units=M --parsable2 \
    --format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,TotalCPU,CPUTimeRAW,ReqTRES,AllocTRES \
    > "$scheduler_report"
"$python" -m autobehaver.resource_monitor "$bundle/resources" --output "$resource_report"
printf 'Scheduler report: %s\nResource summary: %s\n' "$scheduler_report" "$resource_report"
