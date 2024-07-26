#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-3%1
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/%A_stack_blocks_mask/train_%a.out
#SBATCH -J stack_blocks_mask


training_script=train_diffuser_actor.py
task_name=stack_blocks
task_config=stack_blocks_mask

args="task=$task_config\
    num_episodes=-1\
    training.resume=True\
    dataloader.batch_size=8\
    val_dataloader.batch_size=8\
    task.env_runner.n_procs_max=5\
    training.visualize_every=1000\
    training.num_epochs=5000\
    training.rollout_every=5000"

kwargs=${@:1}
    
args="$args $kwargs"

cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run.sh $training_script \
            $task_name \
            $task_config \
            $SLURM_ARRAY_TASK_ID \
            $args \
