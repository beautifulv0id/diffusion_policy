#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-3%1
#SBATCH --output=/home/urain/diffusion_policy/slurm_scripts/logs/%A_sweep_to_dustpan_of_size_lowdim/train_%a.out
#SBATCH -J sweep_to_dustpan_of_size_lowdim

training_script=train_diffuser_actor_pose_invariant_lowdim.py
task_name=sweep_to_dustpan_of_size
task_config=sweep_to_dustpan_of_size_lowdim

args="task=$task_config\
    training.resume=True\
    task.env_runner.n_procs_max=5"

if [ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_COUNT ]; then
    args="$args mode=rollout"
fi

kwargs=${@:1}
    
args="$args $kwargs"

HYDRA_RUN_DIR_FILE=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir_${task_name}.txt
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run.sh $training_script \
            $task_name \
            $task_config \
            $SLURM_ARRAY_TASK_ID \
            $task_config \
            $HYDRA_RUN_DIR_FILE \
            $args \
