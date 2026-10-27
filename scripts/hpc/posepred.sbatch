#!/bin/bash
#SBATCH --job-name=autobehaver_posepred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24g
#SBATCH --time=24:00:00
set -euo pipefail
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="${5:-}"
source "$bundle/templates/runtime.sh"
selection_args=()
[[ "${6:--}" == - ]] || selection_args=(--file_list "$6")
model16="${7:-$resources/posepred/nh16_e301.pth}"
model32="${8:-$resources/posepred/nh32_e301.pth}"
model64="${9:-$resources/posepred/nh64_e301.pth}"
export PYTHONPATH="$package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_stage "$bundle/logs/posepred_${SLURM_JOB_ID}_time.txt" "$python" -m autobehaver.preinf.posepred_inference \
    --main_dir "$project" "${selection_args[@]}" --inference_model "$model16" "$model32" "$model64" --num_workers -1 --batch 0 --memory_gb 24 --sequence_length 41 \
    --inf_average 10 --use_tta --tta_noise_level 0.01 --tta_seed 42 --min_pose_bout 5 --min_orient_bout 5 \
    --resource-monitor --resource-monitor-output "$bundle/resources/posepred_$SLURM_JOB_ID" --resource-monitor-interval 2
