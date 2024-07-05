#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-3%1
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/%A_open_drawer_image/train_%a.out
#SBATCH -J open_drawer_image

training_script=train_diffuser_actor.py
task_name=open_drawer
task_config=open_drawer_image

args="task=$task_config\
    num_episodes=-1\
    training.resume=True\
    dataloader.batch_size=48\
    val_dataloader.batch_size=48"

kwargs=${@:1}
    
args="$args $kwargs"

. run.sh $training_script \
            $task_name \
            $task_config \
            $SLURM_ARRAY_TASK_ID \
            $args \
