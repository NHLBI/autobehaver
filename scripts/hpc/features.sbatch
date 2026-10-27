#!/bin/bash
#SBATCH --job-name=autobehaver_features
#SBATCH --partition=norm
#SBATCH --constraint=e7543
#SBATCH --cpus-per-task=10
#SBATCH --mem=16g
#SBATCH --time=18:00:00
set -euo pipefail
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="${5:-}"
source "$bundle/templates/runtime.sh"
selection_args=()
[[ "${6:--}" == - ]] || selection_args=(--file_list "$6")
export PYTHONPATH="$package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lscpu > "$bundle/logs/features_${SLURM_JOB_ID}_hardware.txt"
run_stage "$bundle/logs/keypre_${SLURM_JOB_ID}_time.txt" "$python" -m autobehaver.preinf.keypre \
    --main_dir "$project" "${selection_args[@]}" --num_workers 8 --memory_gb 16 --resource-monitor --no-resource-monitor-gpu \
    --resource-monitor-output "$bundle/resources/keypre_$SLURM_JOB_ID" --resource-monitor-interval 2
run_stage "$bundle/logs/keyprenorm_${SLURM_JOB_ID}_time.txt" "$python" -m autobehaver.preinf.keyprenorm \
    --main_dir "$project" "${selection_args[@]}" --num_workers 8 --memory_gb 16 --edge_skip 5 \
    --norm_params "$resources/norm_params/norm_params.csv" \
    --resource-monitor --no-resource-monitor-gpu --resource-monitor-output "$bundle/resources/keyprenorm_$SLURM_JOB_ID" --resource-monitor-interval 2
