#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-7%1
#SBATCH --output=../logs/%A_put_item_in_drawer_lowdim/train_%a.out
#SBATCH -J put_item_in_drawer_lowdim

training_script=train_diffuser_actor_lowdim.py
task_name=put_item_in_drawer
task_config=put_item_in_drawer_lowdim
job_name=$task_config

args="task=$task_config\
    training.resume=True\
    task.env_runner.n_procs_max=5"

if [ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]; then
    args="$args mode=rollout"
fi

kwargs=${@:1}
    
args="$args $kwargs"

HYDRA_RUN_DIR_FILE=${DIFFUSION_POLICY_ROOT}/slurm_scripts/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir_${task_name}.txt
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run.sh $training_script \
            $task_name \
            $task_config \
            $SLURM_ARRAY_TASK_ID \
            $task_config \
            $HYDRA_RUN_DIR_FILE \
            $args \