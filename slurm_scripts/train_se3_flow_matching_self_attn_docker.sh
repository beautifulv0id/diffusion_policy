#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%1
#SBATCH --output=../data/logs/%A_se3_flow_matching_self_attn/train_%a.out
#SBATCH -J se3_flow_matching_self_attn

config_name=train_se3_flow_matching_self_attn
tasks="[open_drawer]"
variations=$(seq -s, 0 199)
variations="[${variations}]"
job_name=se3_flow_matching_self_attn
data_dir=/docker_data_dir/diffuser_actor.zarr
instruction_dir=/docker_data_dir/rlbench_instructions/instructions.pkl

args="tasks=${tasks}\
    variations=${variations}\
    optimizer.lr=4e-4\
    dataset.cache_size=2000\
    dataset.root=$data_dir\
    instructions=$instruction_dir\
    dataloader.batch_size=8\
    val_dataloader.batch_size=8\
    training.model_evaluation_every=10"

kwargs=${@:1}
    
args="$args $kwargs"

HYDRA_FULL_ERROR=1
HYDRA_RUN_DIR_FILE=${DIFFUSION_POLICY_ROOT}/data/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir.txt
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run_docker.sh $config_name \
            $SLURM_ARRAY_TASK_ID \
            $HYDRA_RUN_DIR_FILE \
            $args \
