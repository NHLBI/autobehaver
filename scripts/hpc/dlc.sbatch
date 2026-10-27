#!/bin/bash
#SBATCH --job-name=autobehaver_dlc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch:100
#SBATCH --cpus-per-task=6
#SBATCH --mem=16g
#SBATCH --time=24:00:00
set -euo pipefail
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="${5:-}"
source "$bundle/templates/runtime.sh"
selection_args=()
[[ "${6:--}" == - ]] || selection_args=(--file_list "$6")
export PYTHONPATH="$package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export AUTOBEHAVER_SCRATCH_DIR="/lscratch/$SLURM_JOB_ID" TMPDIR="/lscratch/$SLURM_JOB_ID"
if ! type module >/dev/null 2>&1; then
    set +u; source /etc/profile; set -u
fi
module load DeepLabCut/2.3.9
export PYTHONPATH="$package"
export SINGULARITYENV_PYTHONPATH="$package" APPTAINERENV_PYTHONPATH="$package"
export APPTAINER_BINDPATH=/data,/vf,/lscratch SINGULARITY_BINDPATH=/data,/vf,/lscratch
export SINGULARITYENV_TMPDIR="$TMPDIR" APPTAINERENV_TMPDIR="$TMPDIR"
run_stage "$bundle/logs/dlc_${SLURM_JOB_ID}_time.txt" "$python" -m autobehaver.preinf.dlc \
    --main_dir "$project" "${selection_args[@]}" --dlc_python python-DLC --dlc_config "$resources/dlc/m1/config.yaml" \
    --scratch_dir "$AUTOBEHAVER_SCRATCH_DIR" --resource-monitor \
    --resource-monitor-output "$bundle/resources/dlc_$SLURM_JOB_ID" --resource-monitor-interval 2
