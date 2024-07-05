#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%1
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/%A_sweep_to_dustpan_of_size_image/train_%a.out
#SBATCH -J sweep_to_dustpan_of_size_image


training_script=train_diffuser_actor.py
task_name=sweep_to_dustpan_of_size
task_config=sweep_to_dustpan_of_size_image

args="task=$task_config\
    num_episodes=-1\
    training.resume=True\
    dataloader.batch_size=16\
    val_dataloader.batch_size=16"

kwargs=${@:1}
    
args="$args $kwargs"

. run.sh $training_script \
            $task_name \
            $task_config \
            $SLURM_ARRAY_TASK_ID \
            $args \
