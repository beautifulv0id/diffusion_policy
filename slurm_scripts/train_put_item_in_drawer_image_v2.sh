#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-3%1
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/%A_put_item_in_drawer_mask/train_%a.out
#SBATCH -J put_item_in_drawer_mask

training_script=train_diffuser_actor_pose_invariant_v2.py
task_name=put_item_in_drawer
task_config=put_item_in_drawer_mask

args="task=$task_config\
    num_episodes=-1\
    training.resume=True\
    dataloader.batch_size=8\
    val_dataloader.batch_size=8\
    task.env_runner.n_procs_max=1\
    training.gradient_accumulate_every=6\
    training.visualize_every=1000\
    training.num_epochs=10000\
    training.rollout_every=10000"

kwargs=${@:1}
    
args="$args $kwargs"

. run.sh $training_script \
            $task_name \
            $task_config \
            $SLURM_ARRAY_TASK_ID \
            $args \
