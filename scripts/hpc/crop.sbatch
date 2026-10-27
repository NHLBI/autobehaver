#!/bin/bash
#SBATCH --job-name=autobehaver_crop
#SBATCH --partition=norm
#SBATCH --constraint=e7543
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=18:00:00
set -euo pipefail
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="$5"
source "$bundle/templates/runtime.sh"
unset AUTOBEHAVER_CROP_REUSE_CODES
if [[ -f "$bundle/recovery_crop_sources.json" ]]; then
    AUTOBEHAVER_CROP_REUSE_CODES=$(PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$package" "$python" -m autobehaver.preinf.autocropper.recovery_reuse "$bundle/recovery_crop_sources.json")
    export AUTOBEHAVER_CROP_REUSE_CODES
fi
selection_args=()
[[ "${6:--}" == - ]] || selection_args=(--movie_list "$6")
export PYTHONPATH="$package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Session recording metadata may be nested under rec_data; flat projects use the movie folder.
metadata="$project/rec_data/trials"
[[ -d "$metadata" ]] || metadata="$project/1_full_movies"
review_mode=reviewed
[[ ! -f "$bundle/review_mode.txt" ]] || read -r review_mode < "$bundle/review_mode.txt"
review_mode="${review_mode%$'\r'}"
case "$review_mode" in
    reviewed) review_args=(--require_exclusions) ;;
    recording-only)
        [[ ! -e "$bundle/NO_MOVIECHECKER_EXCLUSIONS.csv" ]] || { echo 'Unexpected MovieChecker override file' >&2; exit 1; }
        review_args=(--no_require_exclusions --exclusions_csv "$bundle/NO_MOVIECHECKER_EXCLUSIONS.csv") ;;
    *) echo "Unknown MovieChecker mode: $review_mode" >&2; exit 1 ;;
esac
lscpu > "$bundle/logs/crop_${SLURM_JOB_ID}_hardware.txt"
run_stage "$bundle/logs/crop_${SLURM_JOB_ID}_time.txt" "$python" -m autobehaver.preinf.autocropper \
    --main_dir "$project" "${selection_args[@]}" --recording_database_dir "$metadata" "${review_args[@]}" \
    --num_workers 2 --max_workers 3 --native_threads 1 --ffmpeg_threads 1 --timeout 7200 \
    --resource-monitor --no-resource-monitor-gpu \
    --resource-monitor-output "$bundle/resources/crop_$SLURM_JOB_ID" --resource-monitor-interval 2
