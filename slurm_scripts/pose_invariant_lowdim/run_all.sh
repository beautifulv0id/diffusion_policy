#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-7%1
#SBATCH --output=/home/urain/diffusion_policy/slurm_scripts/logs/%A_pose_invariant_lowdim/train_%a.out
#SBATCH -J pose_invariant_lowdim

FILE_DIR=$pwd
task_names=("open_drawer" "put_item_in_drawer" "stack_blocks" "turn_tap" "sweep_to_dustpan_of_size")
training_script=train_diffuser_actor_pose_invariant_lowdim.py
job_name=pose_invariant_lowdim
pids=()
for task_name in "${task_names[@]}"
do
    task_config=${task_name}_lowdim

    args="task=$task_config\
        training.resume=True\
        task.env_runner.n_procs_max=5\
        training.rollout_best_ckpt=True"

    kwargs=${@:1}
        
    args="$args $kwargs"

    HYDRA_RUN_DIR_FILE=/home/urain/diffusion_policy/slurm_scripts/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir_${task_name}.txt

    cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
    . run.sh $training_script \
                $task_name \
                $task_config \
                $SLURM_ARRAY_TASK_ID \
                $job_name \
                $HYDRA_RUN_DIR_FILE \
                $args &
    pids+=($!)
    cd ${FILE_DIR}
done


for pid in "${pids[@]}"
do
    wait $pid
done
