#!/bin/bash
#SBATCH --job-name=autobehaver_mmgen_finish
#SBATCH --partition=norm
#SBATCH --constraint=e7543
#SBATCH --cpus-per-task=12
#SBATCH --mem=80g
#SBATCH --time=48:00:00
set -euo pipefail
package="$1"; python="$2"; resources="$3"; bundle="$4"; project="$5"
source "$bundle/templates/runtime.sh"
selection_args=()
[[ "${6:--}" == - ]] || selection_args=(--file_list "$6")
shard_args=()
resume_args=()
index="${7:-0}"; count="${8:-1}"; action="${9:-process}"
append_args=()
if [[ -f "$bundle/append_batch_id.txt" ]]; then
    read -r append_id < "$bundle/append_batch_id.txt"
    append_args=(--append_new_data --append_batch_id "$append_id")
    [[ "$count" == 1 && "$action" == process ]] || { echo 'Desktop append uses one MMGen update job.' >&2; exit 1; }
fi
if [[ "$action" == finalize ]]; then
    [[ "$count" == 1 ]] || { echo 'Existing-store finalization requires count=1' >&2; exit 1; }
    shard_args=(--finalize_only)
    if [[ -f "$bundle/finalize_compatibility.json" ]]; then
        export AUTOBEHAVER_MMGEN_FINALIZE_COMPATIBILITY="$bundle/finalize_compatibility.json"
    fi
elif [[ "$action" == merge ]]; then
    shard_args=(--num_shards "$count" --shard_by row_identity --merge_shards)
elif (( count > 1 )); then
    resume_args=(--resume)
    shard_args=(--num_shards "$count" --shard_index "$index" --shard_by row_identity --process_only)
fi
export PYTHONPATH="$package" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lscpu > "$bundle/logs/mmgen_${SLURM_JOB_ID}_hardware.txt"
run_stage "$bundle/logs/mmgen_${SLURM_JOB_ID}_time.txt" "$python" -m autobehaver.postinf.mmgen \
    --main_dir "$project" "${selection_args[@]}" "${shard_args[@]}" "${resume_args[@]}" "${append_args[@]}" --num_workers 8 --source_frame_offset 0 --random_seed 7 \
    --edge_skip 5 --min_bout 5 --min_bout_count_for_features 6 --min_bout_count_for_transitions 6 \
    --video_fps 40 --chamber_diameter_px 530 --mm_per_px 0.0169896744624207 \
    --master_matrix --full_outputs --no_keep_temp_files --resource-monitor --no-resource-monitor-gpu \
    --resource-monitor-output "$bundle/resources/mmgen_$SLURM_JOB_ID" --resource-monitor-interval 2
