#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%1
#SBATCH --output=../data/logs/%A_diffuser_actor/train_%a.out
#SBATCH -J diffuser_actor

training_script=train_diffuser_actor.py
tasks="open_drawer"
job_name=diffuser_actor
data_dir=${DIFFUSION_POLICY_ROOT}/data/diffuser_actor.zarr

args="tasks=${tasks}\
    training.resume=True\
    env_runner.n_procs_max=5\
    dataset.use_precomputed_features=False\
    dataset.root=$data_dir\
    dataset.image_rescale=[0.75,1.25]"

if [ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]; then
    args="$args mode=rollout"
fi

kwargs=${@:1}
    
args="$args $kwargs"

HYDRA_RUN_DIR_FILE=${DIFFUSION_POLICY_ROOT}/data/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir.txt
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run.sh $training_script \
            $SLURM_ARRAY_TASK_ID \
            $HYDRA_RUN_DIR_FILE \
            $args \
