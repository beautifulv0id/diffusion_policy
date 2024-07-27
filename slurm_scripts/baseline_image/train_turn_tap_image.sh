#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-3%1
#SBATCH --output=/home/urain/diffusion_policy/slurm_scripts/logs/%A_turn_tap_image/train_%a.out
#SBATCH -J turn_tap_image


training_script=train_diffuser_actor.py
task_name=turn_tap
task_config=turn_tap_image


args="task=$task_config\
    training.resume=True\
    task.env_runner.n_procs_max=5\
    training.rollout_best_ckpt=True"



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
